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

#include "async_kernel_api.h"
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

static inline __aicore__ void publish_u64(__gm__ uint64_t *addr, uint64_t value) {
    *addr = value;
    dcci(
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<uintptr_t>(addr) & ~uintptr_t{63}), SINGLE_CACHE_LINE,
        CACHELINE_OUT
    );
}

// One 64 MiB TGET. defer==0 waits in this AIV; defer!=0 registers the URMA
// completion and returns.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    constexpr int kElems = 64 * 1024 * 1024 / 4;
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *stamp_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    const uint64_t defer = static_cast<uint64_t>(args[4]);

    if (comm_ctx == nullptr || comm_ctx->rankNum < 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_workspace<CommWorkspaceKind::URMA>(*comm_ctx).addr == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float *local_in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *local_out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const uint32_t peer = (comm_ctx->rankId + 1u) % comm_ctx->rankNum;
    const uint64_t input_offset = reinterpret_cast<uint64_t>(local_in) - comm_ctx->windowsIn[comm_ctx->rankId];
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm_ctx).addr);
    __gm__ float *remote_in = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, input_offset);

    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    GlobalData dst_global(local_out);
    GlobalData src_global(remote_in);

    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer, session)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint64_t *stamp =
        reinterpret_cast<__gm__ uint64_t *>(stamp_tensor->buffer.addr) + stamp_tensor->start_offset;
    publish_u64(&stamp[0], get_sys_cnt());

    AsyncCtx async_ctx = get_async_ctx(args);
    pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(dst_global, src_global, session);
    bool ok = true;
    if (defer != 0) {
        ok = pto2::detail::register_urma_async_event(async_ctx, event, session, workspace);
        pto2::detail::defer_flush(async_ctx);
    } else {
        ok = event.Wait(session);
    }
    publish_u64(&stamp[1], get_sys_cnt());
    if (!ok) {
        pto2::detail::defer_error(async_ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
}
