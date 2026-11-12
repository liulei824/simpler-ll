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

#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif

#include <pto/pto-inst.hpp>

#include "xfer_common.h"

// Same scalars as kernel_xfer, plus a checksum tensor. Wait stays on this core,
// then the checksum is written here. That is the fused latency arm.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *window = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *dst = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *stamp = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *checksum = reinterpret_cast<__gm__ Tensor *>(args[3]);
    const uint64_t nbytes = static_cast<uint64_t>(args[4]);
    const uint64_t byte_offset = static_cast<uint64_t>(args[5]);
    const uint32_t channel = static_cast<uint32_t>(args[6]);
    const int reps = static_cast<int>(args[7]);
    const int defer = static_cast<int>(args[8]);
    const int engine = static_cast<int>(args[9]);
    const int write_stamp = static_cast<int>(args[10]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[11]);

    AsyncCtx async_ctx = get_async_ctx(args);
    (void)run_transfer(
        async_ctx, window, dst, stamp, checksum, comm, nbytes, byte_offset, channel, reps, defer, engine, write_stamp,
        /*write_checksum=*/1
    );
    pipe_barrier(PIPE_ALL);
}
