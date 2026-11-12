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

#pragma once

#include <cstdint>

#include "backend/sdma/sdma_completion_kernel.h"
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

// channel is explicit. TMR does not program CCE get_block_idx(), and the
// default session builder would put every task on one SDMA queue.
template <typename GlobalData, typename ScratchTile>
inline __aicore__ bool submit_sdma_get(
    AsyncCtx &ctx, GlobalData &dst, GlobalData &src, ScratchTile &scratch, __gm__ uint8_t *workspace, uint32_t channel,
    int defer
) {
    pto::comm::AsyncSession session;
    pto::comm::sdma::SdmaBaseConfig config{pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    if (!pto::comm::BuildAsyncSession(scratch, workspace, session, /*syncId=*/0, config, channel)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(dst, src, session);
    if (defer) {
        pto2::detail::register_pto_async_event(ctx, event, session);
        pto2::detail::defer_flush(ctx);
        return true;
    }
    return event.Wait(session);
}

template <typename GlobalData>
inline __aicore__ bool
submit_urma_get(AsyncCtx &ctx, GlobalData &dst, GlobalData &src, __gm__ uint8_t *workspace, uint32_t peer, int defer) {
#if defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer, session)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(dst, src, session);
    if (defer) {
        if (!pto2::detail::register_urma_async_event(ctx, event, session, workspace)) {
            return false;
        }
        pto2::detail::defer_flush(ctx);
        return true;
    }
    return event.Wait(session);
#else
    (void)dst;
    (void)src;
    (void)workspace;
    (void)peer;
    (void)defer;
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return false;
#endif
}

template <int kElems>
inline __aicore__ bool issue_transfer(
    AsyncCtx &ctx, __gm__ float *dst, __gm__ float *src, __gm__ uint8_t *workspace, uint32_t channel, uint32_t peer,
    int defer, int engine
) {
    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    GlobalData dst_global(dst);
    GlobalData src_global(src);
    if (engine == 0) {
        using ScratchTile = Tile<TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT>;
        ScratchTile scratch;
        TASSIGN(scratch, 0x0);
        return submit_sdma_get(ctx, dst_global, src_global, scratch, workspace, channel, defer);
    }
    return submit_urma_get(ctx, dst_global, src_global, workspace, peer, defer);
}

inline __aicore__ bool issue_sized(
    AsyncCtx &ctx, __gm__ float *dst, __gm__ float *src, __gm__ uint8_t *workspace, uint64_t nbytes, uint32_t channel,
    uint32_t peer, int defer, int engine
) {
    if (nbytes == 64ull * 1024ull) {
        return issue_transfer<64 * 1024 / 4>(ctx, dst, src, workspace, channel, peer, defer, engine);
    }
    if (nbytes == 1024ull * 1024ull) {
        return issue_transfer<1024 * 1024 / 4>(ctx, dst, src, workspace, channel, peer, defer, engine);
    }
    if (nbytes == 16ull * 1024ull * 1024ull) {
        return issue_transfer<16 * 1024 * 1024 / 4>(ctx, dst, src, workspace, channel, peer, defer, engine);
    }
    if (nbytes == 64ull * 1024ull * 1024ull) {
        return issue_transfer<64 * 1024 * 1024 / 4>(ctx, dst, src, workspace, channel, peer, defer, engine);
    }
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return false;
}

inline __aicore__ void publish_u64(__gm__ uint64_t *addr, uint64_t value) {
    *addr = value;
#if defined(__CCE_AICORE__) || defined(__DAV_C220__)
    dcci(
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<uintptr_t>(addr) & ~uintptr_t{63}), SINGLE_CACHE_LINE,
        CACHELINE_OUT
    );
#endif
}

// Sequential Wait reps share one buffer. Deferred mode is one post: the core
// must return so the scheduler can observe the completion.
inline __aicore__ bool run_transfer(
    AsyncCtx &ctx, __gm__ Tensor *window, __gm__ Tensor *dst, __gm__ Tensor *stamp, __gm__ Tensor *checksum,
    __gm__ CommContext *comm, uint64_t nbytes, uint64_t byte_offset, uint32_t channel, int reps, int defer, int engine,
    int write_stamp, int write_checksum
) {
    if (comm == nullptr || comm->rankNum < 2 || window == nullptr || dst == nullptr) {
        return false;
    }
    const int rank = static_cast<int>(comm->rankId);
    const int nranks = static_cast<int>(comm->rankNum);
    const int peer = (rank + 1) % nranks;
    __gm__ uint8_t *workspace = engine == 0
                                    ? reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr)
                                    : reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    if (workspace == nullptr) {
        return false;
    }

    __gm__ float *local_base = reinterpret_cast<__gm__ float *>(window->buffer.addr) + window->start_offset;
    __gm__ float *local_ptr = reinterpret_cast<__gm__ float *>(reinterpret_cast<uintptr_t>(local_base) + byte_offset);
    const uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - comm->windowsIn[rank];
    __gm__ float *remote_ptr;
    if (engine == 0) {
        remote_ptr = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + offset);
    } else {
        remote_ptr = pto2::urma_backend::peer_mr_ptr<float>(workspace, static_cast<uint32_t>(peer), offset);
    }
    __gm__ float *dst_ptr = reinterpret_cast<__gm__ float *>(dst->buffer.addr) + dst->start_offset;

    __gm__ uint64_t *stamp_ptr = nullptr;
    if (write_stamp && stamp != nullptr) {
        stamp_ptr = reinterpret_cast<__gm__ uint64_t *>(stamp->buffer.addr) + stamp->start_offset;
        publish_u64(&stamp_ptr[0], get_sys_cnt());
    }

    const int trips = reps > 0 ? reps : 1;
    // A deferred kernel posts once and returns. Repeating it without waiting
    // would reuse the same post-done record.
    const int issued = defer ? 1 : trips;
    for (int i = 0; i < issued; ++i) {
        if (!issue_sized(ctx, dst_ptr, remote_ptr, workspace, nbytes, channel, static_cast<uint32_t>(peer), defer, engine)) {
            return false;
        }
    }

    if (write_stamp && stamp_ptr != nullptr) {
        publish_u64(&stamp_ptr[1], get_sys_cnt());
    }
    if (write_checksum && checksum != nullptr) {
        __gm__ float *out = reinterpret_cast<__gm__ float *>(checksum->buffer.addr) + checksum->start_offset;
        out[0] = dst_ptr[0];
    }
    return true;
}
