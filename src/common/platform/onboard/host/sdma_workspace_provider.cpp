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

#include "sdma_workspace_provider.h"

#include "common/unified_log.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "acl/acl.h"

#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
#endif

namespace {

#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE

using SdmaManager = pto::comm::sdma::SdmaWorkspaceManager;

struct DeviceSlot {
    std::unique_ptr<SdmaManager> manager;
    int refs = 0;
};

struct ProviderHandle {
    int device_id = -1;
};

std::mutex g_mu;
std::unordered_map<int, DeviceSlot> g_slots;

int current_device_id(int *device_id_out) {
    int32_t device_id = -1;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS || device_id < 0) {
        LOG_ERROR("sdma_workspace_provider: aclrtGetDevice failed");
        return -1;
    }
    *device_id_out = static_cast<int>(device_id);
    return 0;
}

int ensure_locked(int device_id, DeviceSlot **slot_out) {
    auto &slot = g_slots[device_id];
    if (!slot.manager) {
        auto manager = std::make_unique<SdmaManager>();
        bool init_ok = false;
        try {
            // Init creates the 48 STARS streams + workspace. It may fail after
            // creating only a subset; destructing that partial manager on an
            // error-state card can itself stall, so leak it on failure.
            init_ok = manager->Init();
        } catch (...) {
            LOG_ERROR("sdma_workspace_provider: SdmaWorkspaceManager::Init threw; abandoning partial resources");
        }
        if (!init_ok) {
            (void)manager.release();
            g_slots.erase(device_id);
            return -1;
        }
        if (manager->GetWorkspaceAddr() == nullptr) {
            (void)manager.release();
            g_slots.erase(device_id);
            LOG_ERROR("sdma_workspace_provider: manager returned a null workspace address");
            return -1;
        }
        slot.manager = std::move(manager);
        slot.refs = 0;
    }
    *slot_out = &slot;
    return 0;
}

#endif  // SIMPLER_ENABLE_PTO_SDMA_WORKSPACE

}  // namespace

extern "C" int sdma_workspace_provider_acquire(uint64_t *addr_out, uint64_t *size_out, void **handle_out) {
    if (!addr_out || !size_out || !handle_out) return -1;
    *addr_out = 0;
    *size_out = 0;
    *handle_out = nullptr;
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    int device_id = -1;
    if (current_device_id(&device_id) != 0) return -1;
    try {
        std::lock_guard<std::mutex> lock(g_mu);
        DeviceSlot *slot = nullptr;
        if (ensure_locked(device_id, &slot) != 0) return -1;
        auto *handle = new (std::nothrow) ProviderHandle{device_id};
        if (handle == nullptr) return -1;
        slot->refs += 1;
        *addr_out = reinterpret_cast<uint64_t>(slot->manager->GetWorkspaceAddr());
        *size_out = static_cast<uint64_t>(pto::comm::sdma::detail::kSdmaWorkspaceBytes);
        *handle_out = handle;
        return 0;
    } catch (...) {
        LOG_ERROR("sdma_workspace_provider_acquire: exception");
        return -1;
    }
#else
    return -1;
#endif
}

extern "C" void sdma_workspace_provider_release(void *handle) {
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    if (!handle) return;
    try {
        std::unique_ptr<ProviderHandle> owned(static_cast<ProviderHandle *>(handle));
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_slots.find(owned->device_id);
        if (it == g_slots.end()) return;
        if (it->second.refs > 0) it->second.refs -= 1;
        if (it->second.refs == 0) {
            try {
                it->second.manager.reset();
            } catch (...) {
                LOG_ERROR("sdma_workspace_provider_release: exception while destroying manager");
            }
            g_slots.erase(it);
        }
    } catch (...) {
        LOG_ERROR("sdma_workspace_provider_release: exception");
    }
#else
    (void)handle;
#endif
}

extern "C" int sdma_workspace_provider_bind(void **handle_inout, uint64_t *addr_out, uint64_t *size_out) {
    if (!handle_inout) return -1;
    if (*handle_inout == nullptr) return sdma_workspace_provider_acquire(addr_out, size_out, handle_inout);
    return sdma_workspace_provider_peek(addr_out, size_out);
}

extern "C" int sdma_workspace_provider_peek(uint64_t *addr_out, uint64_t *size_out) {
    if (!addr_out || !size_out) return -1;
    *addr_out = 0;
    *size_out = 0;
#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    int device_id = -1;
    if (current_device_id(&device_id) != 0) return -1;
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_slots.find(device_id);
    if (it == g_slots.end() || !it->second.manager) return -1;
    *addr_out = reinterpret_cast<uint64_t>(it->second.manager->GetWorkspaceAddr());
    *size_out = static_cast<uint64_t>(pto::comm::sdma::detail::kSdmaWorkspaceBytes);
    return *addr_out == 0 ? -1 : 0;
#else
    return -1;
#endif
}
