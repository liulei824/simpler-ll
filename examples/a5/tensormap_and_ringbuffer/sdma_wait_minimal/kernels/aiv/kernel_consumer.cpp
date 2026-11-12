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

#include "tensor.h"

using namespace pto;

// Depends on the transfer's destination, so the scheduler cannot release this
// task until the completion path — in-core Wait or deferred poll — has fired.
// Size-agnostic on purpose: the A/B sweeps transfer sizes.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *stamp_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    __gm__ float *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;
    __gm__ uint64_t *stamp =
        reinterpret_cast<__gm__ uint64_t *>(stamp_tensor->buffer.addr) + stamp_tensor->start_offset;

    result[0] = src[0] + 1.0f;
    stamp[2] = get_sys_cnt();
    dcci(
        reinterpret_cast<__gm__ int32_t *>(reinterpret_cast<uintptr_t>(&stamp[2]) & ~uintptr_t{63}), SINGLE_CACHE_LINE,
        CACHELINE_OUT
    );
    pipe_barrier(PIPE_ALL);
}
