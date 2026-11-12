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

#include <stdint.h>

#include "orchestration_api.h"
#include "platform_comm/comm_context.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
urma_l1_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return urma_l1_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_l1_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() + orch_args.scalar_count() != 6) {
        LOG_ERROR("urma_l1: expected 6 args");
        return;
    }

    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &result = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &stamp = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    const uint64_t defer = orch_args.scalar<uint64_t>(1);

    CoreTaskArgs producer_args;
    producer_args.add_input(input);
    producer_args.add_output(out);
    producer_args.add_output(stamp);
    producer_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    producer_args.add_scalar(defer);
    rt_submit_aiv_task(0, producer_args);

    CoreTaskArgs consumer_args;
    consumer_args.add_input(out);
    consumer_args.add_output(result);
    consumer_args.add_inout(stamp);
    rt_submit_aiv_task(1, consumer_args);
}

}  // extern "C"
