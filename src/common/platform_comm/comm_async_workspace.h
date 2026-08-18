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
 * Host-side construction of the CommContextBlock async-workspace trailer.
 *
 * Every comm backend (HCCL on a2a3 / a5, POSIX-shm under simulation) publishes
 * device contexts by allocating a CommContextBlock and filling it. Two rules
 * apply to all of them, and getting either wrong is silent corruption rather
 * than a build error:
 *
 *   - The whole block must be written, not just the CommContext prefix. A block
 *     sized allocation whose trailer is never written leaves the magic reading
 *     uninitialised memory, which can match by accident and hand a kernel a
 *     garbage workspace address.
 *   - The trailer must be stamped even when no engine is provisioned. A stamped
 *     table with zeroed slots is what makes a kernel self-skip; an unstamped one
 *     is indistinguishable from a context that predates the trailer.
 *
 * make_comm_context_block() is the single place that gets both right, so a
 * backend's publish path is one memcpy of one value.
 */

#pragma once

#include "comm_context.h"

/** An empty but valid table: stamped, with every engine slot unprovisioned. */
inline CommAsyncWorkspaceTable make_empty_comm_async_table() {
    CommAsyncWorkspaceTable table{};
    table.magic = COMM_ASYNC_WORKSPACE_MAGIC;
    table.version = COMM_ASYNC_WORKSPACE_VERSION;
    return table;
}

/**
 * Record one engine's workspace in the table.
 *
 * `domain_rank` / `rank_count` describe the domain, not the engine — an SDMA
 * workspace cannot report them and a per-domain URMA one would only be
 * repeating what its caller already knows. Funnelling every slot write through
 * here keeps that split with one writer instead of one per backend.
 */
inline void fill_comm_engine_slot(
    CommAsyncWorkspaceTable &table, int kind, uint64_t addr, uint64_t size, uint32_t backend, uint32_t domain_rank,
    uint32_t rank_count
) {
    if (kind < 0 || kind >= DMA_WORKSPACE_KIND_COUNT) return;
    CommEngineSlot &slot = table.slots[kind];
    slot.addr = addr;
    slot.size = size;
    slot.backend = backend;
    slot.rank_count = rank_count;
    slot.domain_rank = domain_rank;
}

/** Pair `ctx` with an empty trailer, ready to be copied to the device as a unit. */
inline CommContextBlock make_comm_context_block(const CommContext &ctx) {
    CommContextBlock block{};
    block.ctx = ctx;
    block.async = make_empty_comm_async_table();
    return block;
}
