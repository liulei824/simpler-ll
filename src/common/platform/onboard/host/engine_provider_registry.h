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
 * Per-engine workspace providers behind allocate_domain(engines=...).
 *
 * Every async engine answers the same two questions -- "give this domain your
 * workspace" and "let it go" -- but differs in what it needs to answer them and
 * in how long the answer lives. Before this registry each backend spelled the
 * differences out as an #ifdef chain per engine inside its own
 * fill_domain_async_table, so adding an engine meant editing every backend.
 * Here the engine-specific part is one provider entry and the backends only
 * supply an AcquireCtx.
 *
 * The providers a build carries are decided by the same CMake options as
 * before, so a2a3 sees SDMA only and a5 sees SDMA and/or URMA. The #ifdefs are
 * confined to the table's definition.
 */

#pragma once

#include <cstdint>

#include "common/dma_workspace.h"
#include "platform_comm/comm_context.h"

// Whether Acquire hands out a shared resource or makes a new one, which is what
// decides who releases it. Device-scope engines (SDMA) keep one workspace per
// card no matter how many domains ask -- creating a second one mid-run stalls
// STARS teardown -- so the ref belongs to the comm handle and outlives any
// domain. Domain-scope engines (URMA) are built per domain from that domain's
// window and die with it.
enum class EngineScope { Device, Domain };

// What a provider hands back. `opaque` is the provider's own bookkeeping: for a
// domain-scope engine it is the object whose lifetime the domain now owns, and
// the caller adopts it into its per-domain state.
struct EngineInstance {
    uint64_t addr = 0;
    uint64_t size = 0;
    uint32_t backend = 0;
    void *opaque = nullptr;
};

// Everything a provider may need, flattened out of the backends' private state.
// CommHandle deliberately does not appear: its struct is defined separately by
// each backend's comm_hccl.cpp, so a registry that named it could not be shared.
struct AcquireCtx {
    void *hccl_comm = nullptr;          // HcclComm for URMA's link setup; SDMA ignores it
    void **sdma_bind_handle = nullptr;  // &h->sdma_provider_handle -- the comm handle's single ref
    const uint32_t *rank_ids = nullptr;
    uint32_t rank_count = 0;
    uint32_t domain_rank = 0;
    void *local_window = nullptr;  // this domain's window; URMA's symmetric memory
    uint64_t window_size = 0;
    int rank = 0;  // base rank, for logging only
};

// A function-pointer entry rather than a virtual class, matching the C shape of
// the sdma_workspace_provider_* entry points it wraps. `release` is null when
// the engine has nothing to undo per domain.
struct EngineProvider {
    int kind;
    EngineScope scope;
    int (*acquire)(const AcquireCtx &ctx, EngineInstance *out);
    void (*release)(EngineInstance *inst);
};

/**
 * Acquire every engine named by engine_mask and publish it into the domain's
 * trailer.
 *
 * `table_out` is stamped here, so a mask of 0 still yields a valid empty table
 * and the caller never has to remember to stamp one. `instances_out` must have
 * DMA_WORKSPACE_KIND_COUNT entries and reports what was acquired, indexed by
 * kind: after success the caller adopts any non-null `opaque` into its
 * per-domain state.
 *
 * All-or-nothing: a mask bit with no provider in this build, or a provider that
 * fails, releases whatever was already acquired and returns non-zero with the
 * table stamped empty again.
 */
int engine_registry_acquire(
    uint32_t engine_mask, const AcquireCtx &ctx, CommAsyncWorkspaceTable *table_out, EngineInstance *instances_out
);
