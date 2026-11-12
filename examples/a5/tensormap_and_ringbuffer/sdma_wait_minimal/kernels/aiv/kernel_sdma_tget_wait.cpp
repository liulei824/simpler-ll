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

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "async_kernel_api.h"
#include "tensor.h"

using namespace pto;

template <typename T>
static inline __aicore__ __gm__ T *comm_remote_ptr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

static inline __aicore__ void publish_u64(__gm__ uint64_t *addr, uint64_t value) {
    *addr = value;
    dcci(
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<uintptr_t>(addr) & ~uintptr_t{63}), SINGLE_CACHE_LINE,
        CACHELINE_OUT
    );
}

template <int kElems>
static inline __aicore__ bool tget_sized(
    AsyncCtx &ctx, __gm__ float *dst, __gm__ float *src, __gm__ uint8_t *workspace, uint64_t defer, int reps,
    uint32_t channel
) {
    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT>;

    GlobalData dst_global(dst);
    GlobalData src_global(src);
    ScratchTile scratch_tile;
    TASSIGN(scratch_tile, 0x0);

    // The channel group is explicit: TMR leaves the CCE block index
    // unprogrammed, so the auto index would collide across concurrent tasks.
    pto::comm::AsyncSession session;
    pto::comm::sdma::SdmaBaseConfig config{pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    if (!pto::comm::BuildAsyncSession(scratch_tile, workspace, session, /*syncId=*/0, config, channel)) {
        return false;
    }
    if (defer != 0) {
        // The core returns here; the scheduler polls the condition and
        // releases the consumer. This is the arm the A/B measures against.
        // One post only: repeating without waiting would reuse the record.
        pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(dst_global, src_global, session);
        pto2::detail::register_pto_async_event(ctx, event, session);
        pto2::detail::defer_flush(ctx);
        return true;
    }
    // L0 drives the engine back-to-back from one core to read its ceiling.
    for (int i = 0; i < reps; ++i) {
        pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(dst_global, src_global, session);
        if (!event.Wait(session)) {
            return false;
        }
    }
    return true;
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *stamp_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    const uint64_t defer = static_cast<uint64_t>(args[4]);
    const uint64_t nbytes = static_cast<uint64_t>(args[5]);
    const int reps = static_cast<int>(args[6]);
    const uint32_t channel = static_cast<uint32_t>(args[7]);
    const uint64_t byte_offset = static_cast<uint64_t>(args[8]);
    const int write_stamp = static_cast<int>(args[9]);

    __gm__ float *local_in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    local_in = reinterpret_cast<__gm__ float *>(reinterpret_cast<uintptr_t>(local_in) + byte_offset);
    __gm__ float *local_out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);
    // A domain without SDMA selected publishes a null SDMA workspace.
    if (nranks < 2 || comm_workspace<CommWorkspaceKind::SDMA>(*comm_ctx).addr == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int peer_rank = (rank + 1) % nranks;

    __gm__ float *remote_in = comm_remote_ptr(comm_ctx, local_in, peer_rank);

    AsyncCtx async_ctx = get_async_ctx(args);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm_ctx).addr);

    __gm__ uint64_t *stamp = nullptr;
    if (write_stamp && stamp_tensor != nullptr) {
        stamp = reinterpret_cast<__gm__ uint64_t *>(stamp_tensor->buffer.addr) + stamp_tensor->start_offset;
        publish_u64(&stamp[0], get_sys_cnt());
    }

    bool ok;
    if (nbytes == 64ull * 1024ull) {
        ok = tget_sized<64 * 1024 / 4>(async_ctx, local_out, remote_in, workspace, defer, reps, channel);
    } else if (nbytes == 1024ull * 1024ull) {
        ok = tget_sized<1024 * 1024 / 4>(async_ctx, local_out, remote_in, workspace, defer, reps, channel);
    } else if (nbytes == 16ull * 1024ull * 1024ull) {
        ok = tget_sized<16 * 1024 * 1024 / 4>(async_ctx, local_out, remote_in, workspace, defer, reps, channel);
    } else {
        ok = tget_sized<64 * 1024 * 1024 / 4>(async_ctx, local_out, remote_in, workspace, defer, reps, channel);
    }
    if (stamp != nullptr) {
        publish_u64(&stamp[1], get_sys_cnt());
    }
    if (!ok) {
        pto2::detail::defer_error(async_ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
}
