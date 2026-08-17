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
 * Per-device SDMA workspace provider shared by a2a3 / a5 onboard host runtimes.
 *
 * One SdmaWorkspaceManager (48 STARS streams + scratch) is created per ACL
 * device on first acquire. Two entry points share it:
 *   - Worker(enable_sdma=True) → dma_workspace_provision (a2a3)
 *   - allocate_domain(engines including sdma) → domain trailer fill (a5+)
 *
 * Destroy only via sdma_workspace_provider_release (Worker finalize / comm
 * teardown). Never create or destroy mid-run for a second domain on the same
 * card — that path hits the 300s STARS teardown stall.
 */

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ensure the current ACL device has an initialized SDMA workspace, bump the
 * per-device refcount, and return an opaque handle the caller must later pass
 * to sdma_workspace_provider_release.
 *
 * On success *addr_out / *size_out are non-zero and *handle_out owns one ref.
 * On failure all outs are left cleared and the return is non-zero (hard fail).
 */
int sdma_workspace_provider_acquire(uint64_t *addr_out, uint64_t *size_out, void **handle_out);

/**
 * Drop one ref from a successful acquire. When the last ref for that device
 * is dropped the underlying SdmaWorkspaceManager is destroyed. Null is a no-op.
 */
void sdma_workspace_provider_release(void *handle);

/**
 * Peek the current device's workspace without taking a ref. Returns 0 and
 * writes addr/size when a live provider exists; otherwise returns non-zero.
 */
int sdma_workspace_provider_peek(uint64_t *addr_out, uint64_t *size_out);

/**
 * Resolve the workspace for a caller that holds at most one ref for its whole
 * lifetime (a comm handle serving many domains): acquires on the first call and
 * reuses the same address afterwards, so N domains on one card cost one ref
 * rather than N.
 *
 * *handle_inout must be null on the first call and is left owning the ref;
 * release it once with sdma_workspace_provider_release. Returns non-zero
 * without touching the outs when no workspace can be resolved.
 */
int sdma_workspace_provider_bind(void **handle_inout, uint64_t *addr_out, uint64_t *size_out);

#ifdef __cplusplus
}
#endif
