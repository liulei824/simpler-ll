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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    // 2170000 ticks is one 64 MiB L0 transfer on the 1 GHz counter.
    // ndims is 1; the compare keeps the trip count a runtime value.
    uint32_t n = 2170000;
    if (out_tensor->ndims == 0) {
        n = 1;
    }
    const uint64_t start = get_sys_cnt();
    for (uint32_t i = 0; i < n; ++i) {
        if (get_sys_cnt() - start >= 2170000) {
            break;
        }
    }
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    out[0] = 1.f;
    pipe_barrier(PIPE_ALL);
}
