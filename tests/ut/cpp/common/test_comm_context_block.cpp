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
 * CommContextBlock layout and trailer construction.
 *
 * comm_context.h already static_asserts its own layout, but until this test the
 * only translation units that compiled it were the comm backends -- which need
 * ACL / HCCL and therefore do not build on a no-hardware runner. Including the
 * header here puts those asserts on the default cpput path, so a layout change
 * that would silently shift the device-side field offsets fails at build time
 * on every CI job rather than only where a device is present.
 */

#include "platform_comm/comm_async_workspace.h"
#include "platform_comm/comm_context.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

// The prefix must stay byte-identical to what pto-isa's mirrored declaration
// expects; appending the trailer must not disturb any of it.
TEST(CommContextBlockLayout, PrefixKeepsTheMirroredCommContextLayout) {
    EXPECT_EQ(sizeof(CommContext), 1056u);
    EXPECT_EQ(offsetof(CommContextBlock, ctx), 0u);
    EXPECT_EQ(offsetof(CommContextBlock, async), sizeof(CommContext));
    EXPECT_EQ(sizeof(CommContextBlock), sizeof(CommContext) + sizeof(CommAsyncWorkspaceTable));
}

// Backends hand out `&block.ctx` as the device context. That only works because
// the prefix sits at offset 0 -- pto-isa reads through the same address the
// block was allocated at, and aclrtFree gets that same address back.
TEST(CommContextBlockLayout, ContextPointerAliasesTheBlockBase) {
    CommContextBlock block{};
    EXPECT_EQ(static_cast<void *>(&block.ctx), static_cast<void *>(&block));
}

TEST(CommAsyncWorkspaceTable, EmptyTableIsStampedWithEveryEngineSlotUnprovisioned) {
    const CommAsyncWorkspaceTable table = make_empty_comm_async_table();

    EXPECT_EQ(table.magic, COMM_ASYNC_WORKSPACE_MAGIC);
    EXPECT_EQ(table.version, COMM_ASYNC_WORKSPACE_VERSION);
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        EXPECT_EQ(table.addr[kind], 0u) << "kind " << kind;
        EXPECT_EQ(table.size[kind], 0u) << "kind " << kind;
    }
}

// The whole point of building the block as one value is that a single copy
// covers the trailer. Start from dirty storage so a missing trailer write would
// leave the magic reading the garbage rather than a zero that happens to pass.
TEST(CommAsyncWorkspaceTable, BlockConstructionCopiesTheContextAndStampsTheTrailer) {
    CommContext ctx{};
    ctx.rankId = 3;
    ctx.rankNum = 8;
    ctx.winSize = 4096;
    ctx.windowsIn[3] = 0xdeadbeefULL;

    CommContextBlock block;
    std::memset(&block, 0xa5, sizeof(block));
    block = make_comm_context_block(ctx);

    EXPECT_EQ(block.ctx.rankId, 3u);
    EXPECT_EQ(block.ctx.rankNum, 8u);
    EXPECT_EQ(block.ctx.winSize, 4096u);
    EXPECT_EQ(block.ctx.windowsIn[3], 0xdeadbeefULL);
    EXPECT_EQ(block.async.magic, COMM_ASYNC_WORKSPACE_MAGIC);
    EXPECT_EQ(block.async.addr[DMA_WORKSPACE_SDMA], 0u);
    EXPECT_EQ(block.async.addr[DMA_WORKSPACE_URMA], 0u);
}
