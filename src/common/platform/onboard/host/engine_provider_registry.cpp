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

#include "platform/onboard/host/engine_provider_registry.h"

#include "common/unified_log.h"
#include "platform/onboard/host/sdma_workspace_provider.h"
#include "platform_comm/comm_async_workspace.h"

#include <memory>

#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE
#include "hccl/hccl_types.h"
#include "pto/comm/async/urma/urma_workspace_manager.hpp"
#endif

namespace {

#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
// Device scope: bind resolves the card's single workspace, creating it only on
// the first domain that asks. There is deliberately no release counterpart --
// the ref sits on the comm handle and comm_destroy drops it, because tearing a
// live SDMA workspace down mid-run to serve one domain costs a 300s STARS stall.
int sdma_acquire(const AcquireCtx &ctx, EngineInstance *out) {
    if (ctx.sdma_bind_handle == nullptr) return -1;
    return sdma_workspace_provider_bind(ctx.sdma_bind_handle, &out->addr, &out->size);
}
#endif

#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE
using pto::comm::urma::UrmaWorkspaceManager;

uint64_t urma_workspace_bytes(uint32_t rank_count) {
    using namespace pto::comm::urma;
    constexpr uint32_t qp_num = 1;
    return sizeof(UrmaInfo) +
           static_cast<uint64_t>(rank_count) *
               (2ULL * sizeof(UrmaWQCtx) * qp_num + 2ULL * sizeof(UrmaCqCtx) * qp_num + sizeof(UrmaMemInfo) * qp_num);
}

// URMA's HCCL communicator is the base one, so the rank ids it is given index
// the base rank space. Until the manager takes a domain-to-base map, the only
// safe domains are those whose members happen to be numbered identically in
// both spaces; anything else would silently connect to the wrong peers.
bool rank_ids_are_dense_prefix(const uint32_t *rank_ids, uint32_t rank_count) {
    if (rank_ids == nullptr) return false;
    for (uint32_t i = 0; i < rank_count; ++i) {
        if (rank_ids[i] != i) return false;
    }
    return true;
}

// Domain scope: one manager per domain, built from that domain's window. The
// manager is handed back through `opaque` so the backend can adopt it into its
// own per-domain state -- that struct is backend-private and cannot be named here.
int urma_acquire(const AcquireCtx &ctx, EngineInstance *out) {
    if (ctx.hccl_comm == nullptr || ctx.local_window == nullptr || ctx.window_size == 0 ||
        ctx.domain_rank >= ctx.rank_count) {
        return -1;
    }
    if (!rank_ids_are_dense_prefix(ctx.rank_ids, ctx.rank_count)) {
        LOG_ERROR("[comm rank %d] alloc_domain: URMA requires dense-prefix rank_ids (rank_ids[i]==i)", ctx.rank);
        return -1;
    }
    auto manager = std::make_unique<UrmaWorkspaceManager>();
    if (!manager->Init(
            static_cast<HcclComm>(ctx.hccl_comm), ctx.domain_rank, ctx.rank_count, ctx.local_window, ctx.window_size
        )) {
        return -1;
    }
    void *workspace = manager->GetWorkspaceAddr();
    if (workspace == nullptr) return -1;
    out->addr = reinterpret_cast<uint64_t>(workspace);
    out->size = urma_workspace_bytes(ctx.rank_count);
    out->opaque = manager.release();
    return 0;
}

void urma_release(EngineInstance *inst) {
    delete static_cast<UrmaWorkspaceManager *>(inst->opaque);
    inst->opaque = nullptr;
}
#endif

// The only place engine availability is spelled out. A kind absent from this
// table has no provider in this build, which is how a2a3 rejects URMA and how
// every build rejects RoCE.
constexpr int kNoKind = -1;
constexpr EngineProvider kProviders[] = {
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    {DMA_WORKSPACE_SDMA, EngineScope::Device, sdma_acquire, nullptr},
#endif
#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE
    {DMA_WORKSPACE_URMA, EngineScope::Domain, urma_acquire, urma_release},
#endif
    // Matches no kind: a build with every overlay off would otherwise declare a
    // zero-length array.
    {kNoKind, EngineScope::Device, nullptr, nullptr},
};

const EngineProvider *find_provider(int kind) {
    for (const EngineProvider &provider : kProviders) {
        if (provider.kind == kind) return &provider;
    }
    return nullptr;
}

// Undo a partial acquire, newest first, and hand back a table that claims
// nothing: the caller is about to fail the whole domain.
void rollback(CommAsyncWorkspaceTable *table_out, EngineInstance *instances) {
    for (int kind = DMA_WORKSPACE_KIND_COUNT - 1; kind >= 0; --kind) {
        if (instances[kind].addr == 0) continue;
        const EngineProvider *provider = find_provider(kind);
        if (provider != nullptr && provider->release != nullptr) provider->release(&instances[kind]);
        instances[kind] = EngineInstance{};
    }
    *table_out = make_empty_comm_async_table();
}

}  // namespace

int engine_registry_acquire(
    uint32_t engine_mask, const AcquireCtx &ctx, CommAsyncWorkspaceTable *table_out, EngineInstance *instances_out
) {
    if (table_out == nullptr || instances_out == nullptr) return -1;
    *table_out = make_empty_comm_async_table();
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
        instances_out[kind] = EngineInstance{};

    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        if ((engine_mask & (uint32_t{1} << kind)) == 0) continue;
        const EngineProvider *provider = find_provider(kind);
        if (provider == nullptr) {
            LOG_ERROR("[comm rank %d] alloc_domain: engine kind=%d has no provider in this build", ctx.rank, kind);
            rollback(table_out, instances_out);
            return -1;
        }
        EngineInstance instance{};
        if (provider->acquire(ctx, &instance) != 0 || instance.addr == 0) {
            LOG_ERROR("[comm rank %d] alloc_domain: engine kind=%d acquire failed", ctx.rank, kind);
            if (provider->release != nullptr) provider->release(&instance);
            rollback(table_out, instances_out);
            return -1;
        }
        instances_out[kind] = instance;
        fill_comm_engine_slot(
            *table_out, kind, instance.addr, instance.size, instance.backend, ctx.domain_rank, ctx.rank_count
        );
    }
    return 0;
}
