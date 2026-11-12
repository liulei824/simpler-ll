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
constexpr int kL1ConsumerKernel = 1;
constexpr int kComputeKernel = 2;
constexpr int kL2ConsumerKernel = 3;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
sdma_wait_minimal_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 9};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return sdma_wait_minimal_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void sdma_wait_minimal_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 9) {
        LOG_ERROR("sdma_wait_minimal: expected 9 args");
        return;
    }

    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &result = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &stamp = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    const uint64_t defer = orch_args.scalar<uint64_t>(1);
    const uint64_t nbytes = orch_args.scalar<uint64_t>(2);
    const uint64_t reps = orch_args.scalar<uint64_t>(3);
    const uint64_t cycles = orch_args.scalar<uint64_t>(4);

    const uint32_t chunk_elems = nbytes == 0 ? 1u : static_cast<uint32_t>(nbytes / sizeof(float));
    const uint32_t window_elems = input.shapes[0];
    int n_xfer = static_cast<int>(window_elems / chunk_elems);
    if (n_xfer < 1) {
        n_xfer = 1;
    }
    if (n_xfer > kMaxXfer) {
        LOG_ERROR("sdma_wait_minimal: n_xfer %d exceeds %d", n_xfer, kMaxXfer);
        return;
    }

    if (n_xfer == 1 && cycles == 0) {
        CoreTaskArgs producer_args;
        producer_args.add_input(input);
        producer_args.add_output(out);
        producer_args.add_output(stamp);
        producer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
        producer_args.add_scalar(defer);
        producer_args.add_scalar(nbytes);
        producer_args.add_scalar(reps);
        producer_args.add_scalar(static_cast<uint64_t>(0));
        producer_args.add_scalar(static_cast<uint64_t>(0));
        producer_args.add_scalar(static_cast<uint64_t>(1));
        rt_submit_aiv_task(kXferKernel, producer_args);

        CoreTaskArgs consumer_args;
        consumer_args.add_input(out);
        consumer_args.add_output(result);
        consumer_args.add_inout(stamp);
        rt_submit_aiv_task(kL1ConsumerKernel, consumer_args);
        return;
    }

    uint32_t dest_shape[1] = {chunk_elems};
    TensorCreateInfo dest_infos[kMaxXfer];
    CoreTaskArgs dest_alloc;
    for (int i = 0; i < n_xfer; ++i) {
        dest_infos[i] = TensorCreateInfo(dest_shape, 1, DataType::FLOAT32);
        dest_alloc.add_output(dest_infos[i]);
    }
    TaskOutputTensors dests = alloc_tensors(dest_alloc);
    // Copy by value: later alloc_tensors can reuse the dummy-task payload
    // that dests.get_ref() borrows.
    simpler::tmr::Tensor dest_copy[kMaxXfer];
    for (int i = 0; i < n_xfer; ++i) {
        dest_copy[i] = dests.get_ref(static_cast<uint32_t>(i));
    }

    for (int i = 0; i < n_xfer; ++i) {
        CoreTaskArgs xfer;
        xfer.add_input(input);
        xfer.add_output(dest_copy[i]);
        if (i == 0) {
            xfer.add_output(stamp);
        } else {
            xfer.add_input(stamp);
        }
        xfer.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
        xfer.add_scalar(defer);
        xfer.add_scalar(nbytes);
        xfer.add_scalar(static_cast<uint64_t>(1));
        xfer.add_scalar(static_cast<uint64_t>(i));
        xfer.add_scalar(static_cast<uint64_t>(i) * nbytes);
        xfer.add_scalar(i == 0 ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0));
        rt_submit_aiv_task(kXferKernel, xfer);
    }

    // No scalar on the compute task: a scalar argument aborts AICPU execution.
    if (cycles > 0) {
        uint32_t one[1] = {1u};
        for (int i = 0; i < kMaxCompute; ++i) {
            TensorCreateInfo info(one, 1, DataType::FLOAT32);
            TaskOutputTensors spun = alloc_tensors(info);
            if (spun.size() == 0) {
                LOG_ERROR("sdma_wait_minimal: spin alloc failed at %d", i);
                return;
            }
            const simpler::tmr::Tensor &local = spun.get_ref(0);
            CoreTaskArgs spin;
            spin.add_inout(local);
            rt_submit_aiv_task(kComputeKernel, spin);
        }
    }

    {
        uint32_t idx[1] = {1};
        set_tensor_data(result, 1, idx, static_cast<float>(n_xfer));
    }

    CoreTaskArgs consumer;
    consumer.add_inout(result);
    for (int i = 0; i < n_xfer; ++i) {
        consumer.add_input(dest_copy[i]);
    }
    rt_submit_aiv_task(kL2ConsumerKernel, consumer);
}

}  // extern "C"
