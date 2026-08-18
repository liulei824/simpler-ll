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
/**
 * CommContext — device-side distributed communication context.
 *
 * This struct is the ABI contract between host (comm_hccl.cpp / comm_sim.cpp)
 * and device kernels. PTO communication instructions (TREDUCE, TGET, TPUT)
 * access remote data through the GVA addresses in windowsIn[]/windowsOut[]
 * via MTE2 DMA.
 *
 * Host fills the struct from scratch:
 *   - comm_hccl.cpp (Path D): allocates a per-rank symmetric pool via the
 *     public ACL IPC primitives (aclrtMalloc + aclrtIpcMemGetExportKey +
 *     SetImportPid + ImportByKey), then writes rankId / rankNum / winSize /
 *     windowsIn[]. No HCCL-private struct is reinterpret_cast'd here; the
 *     layout is owned end-to-end by simpler.
 *   - comm_sim.cpp: same shape, filled with malloc'd host pointers.
 *
 * The layout is shared with pto-isa's parallel HcclDeviceContext
 * declaration and must stay byte-equivalent with it.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "common/dma_workspace.h"

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;

struct CommContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];
};

// The struct itself lives in this repo, so on the surface these asserts look
// like they only check that we do not contradict ourselves. Their real value
// is that this layout is consumed by *two* out-of-band parties that never see
// this header at the same time:
//
//   1. The pto-isa repo carries a parallel declaration (HcclDeviceContext)
//      that must be byte-equivalent to this struct -- pto-isa kernels read
//      windowsIn[]/winSize/rankId via that mirror. Any insert/reorder here
//      that is not matched in pto-isa silently shifts the device-side field
//      offsets and corrupts MTE2 reads. The locks below pin our side; the
//      pto-isa side should add its own mirror asserts.
//
//   2. Device kernels (AICore / AICPU) compiled with CCEC may apply slightly
//      different alignment rules than host gcc. A host-side sizeof/offset
//      lock is a necessary-but-not-sufficient guard.
//
// Treat the numbers below as a tripwire: changing them is a deliberate act
// that forces the editor to coordinate the matching change on the pto-isa
// side, not a routine "oh I just added a field" edit.
static_assert(sizeof(CommContext) == 1056, "CommContext size shifted");
static_assert(offsetof(CommContext, workSpace) == 0, "CommContext layout drift");
static_assert(offsetof(CommContext, workSpaceSize) == 8, "CommContext layout drift");
static_assert(offsetof(CommContext, rankId) == 16, "CommContext layout drift");
static_assert(offsetof(CommContext, rankNum) == 20, "CommContext layout drift");
static_assert(offsetof(CommContext, winSize) == 24, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsIn) == 32, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsOut) == 544, "CommContext layout drift");

// Per-domain async-DMA workspace slots, one per DmaWorkspaceKind.
//
// CommContext carries a single workSpace/workSpaceSize pair, which cannot
// express "this domain uses SDMA and URMA at the same time". Widening it is not
// an option: the layout above is byte-locked against pto-isa. So the slots live
// in a simpler-private trailer appended after the mirrored prefix instead.
//
// `magic` exists because the device side reaches the table by casting a
// CommContext* it was handed, with no way to know whether the allocation behind
// it actually carries the trailer. A context produced by an older host, or one
// whose trailer bytes were never written, reads as garbage; the accessor checks
// the magic and returns nullptr rather than handing a kernel a bogus address.
static constexpr uint32_t COMM_ASYNC_WORKSPACE_MAGIC = 0x53414D57u;  // 'SAMW'
static constexpr uint32_t COMM_ASYNC_WORKSPACE_VERSION = 1u;

// One slot per engine. `rank_count` / `domain_rank` are the domain's properties,
// not the engine's -- an SDMA workspace does not know which ranks share it, so
// the domain layer fills them and a kernel holding several domains' contexts
// reads each engine's numbering from that domain's own slot.
struct CommEngineSlot {
    uint64_t addr;  // 0 = this domain did not provision this engine
    uint64_t size;
    uint32_t backend;  // engine-private backend id; 0 when the engine has one backend
    uint32_t rank_count;
    uint32_t domain_rank;
    uint32_t reserved;
};

struct CommAsyncWorkspaceTable {
    uint32_t magic;
    uint32_t version;
    struct CommEngineSlot slots[DMA_WORKSPACE_KIND_COUNT];
};

// What the host actually allocates for every device context. The CommContext
// prefix is what pto-isa sees; the trailer is ours. Because `ctx` sits at offset
// 0, the pointer handed out as `device_ctx_out` is unchanged -- pto-isa keeps
// reading the same bytes at the same addresses, and simpler recovers the table
// by casting back to the block.
struct CommContextBlock {
    CommContext ctx;
    CommAsyncWorkspaceTable async;
};

static_assert(offsetof(CommContextBlock, ctx) == 0, "CommContextBlock prefix must alias CommContext");
static_assert(offsetof(CommContextBlock, async) == 1056, "CommContextBlock trailer offset drift");
static_assert(sizeof(CommContextBlock) == 1160, "CommContextBlock size drift");

// The device side indexes slots[] by kind, so a stride change silently reads the
// wrong engine's bytes rather than failing. Pin the stride, not just the total.
static_assert(sizeof(CommEngineSlot) == 32, "CommEngineSlot stride drift");
static_assert(sizeof(CommAsyncWorkspaceTable) == 8 + 32 * DMA_WORKSPACE_KIND_COUNT, "trailer size drift");

// Device-side accessors for the per-domain async trailer. Available only in
// translation units that define __gm__ (AICore kernels). Host code must not
// call these — the trailer is filled on the host via CommAsyncWorkspaceTable.
//
// Both recover the block by casting through the offset-0 CommContext prefix,
// then check magic/version so a context published without a trailer (or with a
// stale one) yields nullptr instead of a garbage workspace address.
#ifdef __gm__
#ifndef __aicore__
#define __aicore__
#endif
// A non-null return means the engine is usable: the table is stamped by a
// matching host and this domain provisioned that engine. Callers then read
// domain_rank / rank_count off the slot instead of translating rank numbers.
static __aicore__ inline __gm__ const CommEngineSlot *get_comm_engine_slot(__gm__ CommContext *ctx, int kind) {
    if (ctx == nullptr || kind < 0 || kind >= DMA_WORKSPACE_KIND_COUNT) return nullptr;
    __gm__ CommContextBlock *block = reinterpret_cast<__gm__ CommContextBlock *>(ctx);
    if (block->async.magic != COMM_ASYNC_WORKSPACE_MAGIC || block->async.version != COMM_ASYNC_WORKSPACE_VERSION) {
        return nullptr;
    }
    __gm__ const CommEngineSlot *slot = &block->async.slots[kind];
    return slot->addr == 0 ? nullptr : slot;
}

// Address-only view of the same slot, kept for kernels that need nothing else.
static __aicore__ inline __gm__ uint8_t *get_comm_dma_workspace(__gm__ CommContext *ctx, int kind) {
    __gm__ const CommEngineSlot *slot = get_comm_engine_slot(ctx, kind);
    return slot == nullptr ? nullptr : reinterpret_cast<__gm__ uint8_t *>(slot->addr);
}
#endif
