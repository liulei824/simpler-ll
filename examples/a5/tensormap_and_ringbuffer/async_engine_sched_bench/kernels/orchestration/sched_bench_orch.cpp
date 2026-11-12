/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "orchestration_api.h"

namespace {

constexpr int kMaxXfer = 31;
constexpr int kMaxCompute = 56;
constexpr int kXferKernel = 0;
constexpr int kFusedKernel = 1;
constexpr int kConsumerKernel = 2;
constexpr int kComputeKernel = 3;

enum Mode : int {
    kL0 = 0,
    kFused = 1,
    kSplitWait = 2,
    kSplitDefer = 3,
    kOverlapWait = 4,
    kOverlapDefer = 5,
    kCompute = 6,
};

TaskOutputTensors alloc_f32(uint32_t count, uint32_t elems) {
    // Create-info pointers are consumed inside alloc_tensors, so the
    // descriptors have to outlive that call. A temporary would not.
    TensorCreateInfo infos[32];
    uint32_t shape[1] = {elems == 0 ? 1u : elems};
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; ++i) {
        infos[i] = TensorCreateInfo(shape, 1, DataType::FLOAT32);
        args.add_output(infos[i]);
    }
    return alloc_tensors(args);
}

void submit_xfer(
    const simpler::tmr::Tensor &window, const simpler::tmr::Tensor &dst, const simpler::tmr::Tensor &stamp,
    CommContext *comm, uint64_t nbytes, uint64_t byte_offset, uint32_t channel, int reps, int defer, int engine,
    int write_stamp
) {
    CoreTaskArgs args;
    args.add_input(window);
    args.add_output(dst);
    args.add_input(stamp);
    args.add_scalar(nbytes);
    args.add_scalar(byte_offset);
    args.add_scalar(static_cast<uint64_t>(channel));
    args.add_scalar(static_cast<uint64_t>(reps));
    args.add_scalar(static_cast<uint64_t>(defer));
    args.add_scalar(static_cast<uint64_t>(engine));
    args.add_scalar(static_cast<uint64_t>(write_stamp));
    args.add_scalar(reinterpret_cast<uint64_t>(comm));
    rt_submit_aiv_task(kXferKernel, args);
}

void submit_fused(
    const simpler::tmr::Tensor &window, const simpler::tmr::Tensor &dst, const simpler::tmr::Tensor &stamp,
    const simpler::tmr::Tensor &checksum, CommContext *comm, uint64_t nbytes, int reps, int engine, int write_stamp
) {
    CoreTaskArgs args;
    args.add_input(window);
    args.add_output(dst);
    args.add_input(stamp);
    args.add_inout(checksum);
    args.add_scalar(nbytes);
    args.add_scalar(static_cast<uint64_t>(0));
    args.add_scalar(static_cast<uint64_t>(0));
    args.add_scalar(static_cast<uint64_t>(reps));
    args.add_scalar(static_cast<uint64_t>(0));
    args.add_scalar(static_cast<uint64_t>(engine));
    args.add_scalar(static_cast<uint64_t>(write_stamp));
    args.add_scalar(reinterpret_cast<uint64_t>(comm));
    rt_submit_aiv_task(kFusedKernel, args);
}

void submit_compute(const simpler::tmr::Tensor &out, uint64_t cycles) {
    CoreTaskArgs args;
    args.add_output(out);
    args.add_scalar(cycles);
    rt_submit_aiv_task(kComputeKernel, args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
sched_bench_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 11};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return sched_bench_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void sched_bench_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 11) {
        LOG_ERROR("sched_bench: expected 3 tensors and 8 scalars");
        return;
    }

    const simpler::tmr::Tensor &window = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &checksum = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &stamp = orch_args.tensor(2).ref();
    auto *comm = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    const int mode = static_cast<int>(orch_args.scalar<uint64_t>(1));
    const uint64_t nbytes = orch_args.scalar<uint64_t>(2);
    const int reps = static_cast<int>(orch_args.scalar<uint64_t>(3));
    int n_xfer = static_cast<int>(orch_args.scalar<uint64_t>(4));
    int n_compute = static_cast<int>(orch_args.scalar<uint64_t>(5));
    const uint64_t cycles = orch_args.scalar<uint64_t>(6);
    const int engine = static_cast<int>(orch_args.scalar<uint64_t>(7));

    if (n_xfer < 0 || n_xfer > kMaxXfer || n_compute < 0 || n_compute > kMaxCompute) {
        LOG_ERROR("sched_bench: n_xfer or n_compute out of range");
        return;
    }
    if (engine != 0 && n_xfer > 1) {
        LOG_ERROR("sched_bench: URMA arm is single-transfer only");
        return;
    }


    const bool fused = mode == kL0 || mode == kFused;
    const bool split = mode == kSplitWait || mode == kSplitDefer || mode == kOverlapWait || mode == kOverlapDefer;
    const bool defer = mode == kSplitDefer || mode == kOverlapDefer;
    const bool compute = mode == kCompute || mode == kOverlapWait || mode == kOverlapDefer;
    if (mode == kL0 || mode == kFused || mode == kSplitWait || mode == kSplitDefer) {
        n_xfer = 1;
    }
    if (!compute) {
        n_compute = 0;
    }
    if (mode == kCompute) {
        n_xfer = 0;
    }


    const uint32_t elems = nbytes == 0 ? 1u : static_cast<uint32_t>(nbytes / sizeof(float));
    TaskOutputTensors dests;
    if (n_xfer > 0) {
        dests = alloc_f32(static_cast<uint32_t>(n_xfer), elems);
    }

    TaskOutputTensors compute_lo;
    TaskOutputTensors compute_hi;
    if (n_compute > 0) {
        const int lo = n_compute > 32 ? 32 : n_compute;
        compute_lo = alloc_f32(static_cast<uint32_t>(lo), 1);
        if (n_compute > 32) {
            compute_hi = alloc_f32(static_cast<uint32_t>(n_compute - 32), 1);
        }
    }


    if (fused) {
        submit_fused(window, dests.get_ref(0), stamp, checksum, comm, nbytes, reps, engine, mode == kL0 ? 1 : 0);
    } else if (split) {
        for (int i = 0; i < n_xfer; ++i) {
            submit_xfer(
                window, dests.get_ref(static_cast<uint32_t>(i)), stamp, comm, nbytes,
                static_cast<uint64_t>(i) * nbytes, static_cast<uint32_t>(i), 1, defer ? 1 : 0, engine,
                /*write_stamp=*/0
            );
        }
    }

    if (compute) {
        for (int i = 0; i < n_compute; ++i) {
            const simpler::tmr::Tensor &out =
                i < 32 ? compute_lo.get_ref(static_cast<uint32_t>(i)) : compute_hi.get_ref(static_cast<uint32_t>(i - 32));
            submit_compute(out, cycles);
        }
    }

    if (split) {
        CoreTaskArgs consumer;
        consumer.add_inout(checksum);
        for (int i = 0; i < n_xfer; ++i) {
            consumer.add_input(dests.get_ref(static_cast<uint32_t>(i)));
        }
        rt_submit_aiv_task(kConsumerKernel, consumer);
    }
}

}  // extern "C"
