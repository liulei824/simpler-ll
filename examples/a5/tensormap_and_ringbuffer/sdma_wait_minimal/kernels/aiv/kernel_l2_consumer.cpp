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

// checksum[1] is the dest count. Dest tensors follow. 1 + 31 dests is the
// 32-tensor core limit.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *checksum = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(checksum->buffer.addr) + checksum->start_offset;
    const int n = static_cast<int>(out[1]);
    const int capped = n > 31 ? 31 : (n < 0 ? 0 : n);
    float sum = 0.f;
    for (int i = 0; i < capped; ++i) {
        __gm__ Tensor *dst = reinterpret_cast<__gm__ Tensor *>(args[1 + i]);
        __gm__ float *ptr = reinterpret_cast<__gm__ float *>(dst->buffer.addr) + dst->start_offset;
        sum += ptr[0];
    }
    out[0] = sum;
    pipe_barrier(PIPE_ALL);
}
