#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""2-rank URMA L1: one 64 MiB TGET, in-core Wait versus deferred completion."""

import ctypes
import os
import pathlib

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

NRANKS = 2
NBYTES = 64 * 1024 * 1024
ELEMS = NBYTES // 4
# URMA rejects a payload that starts at the window base.
URMA_DATA_OFFSET_NBYTES = 64 * 4
STAMP_CSV = os.environ.get("ASYNC_SCHED_URMA_CSV", "outputs/async-engine-sched/urma_l1_stamps.csv")


def _record(defer, rank, start, after_xfer, consumer):
    path = pathlib.Path(STAMP_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a") as fh:
        if new:
            fh.write("defer,rank,xfer_ticks,issue_to_consumer_ticks\n")
        fh.write(f"{defer},{rank},{after_xfer - start},{consumer - start}\n")


def urma_l1_orch_fn(orch, callables, task_args, config):
    window = URMA_DATA_OFFSET_NBYTES + NBYTES
    with orch.allocate_domain(
        workspace_engines=("urma",),
        name="default",
        workers=list(range(NRANKS)),
        window_size=window,
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // 4,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=ELEMS, nbytes=NBYTES),
        ],
    ) as handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((ELEMS,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"stamp_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(int(task_args.defer.value))
            orch.submit_next_level(callables.urma_l1, args, config, worker=rank)


@pytest.mark.urma(worker_workspace=False)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaL1(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_l1_orch_fn,
        "callables": [
            {
                "name": "urma_l1",
                "orchestration": {
                    "source": "kernels/orchestration/urma_l1_orch.cpp",
                    "function_name": "urma_l1_orchestration",
                    "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_wait.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.INOUT],
                    },
                ],
            }
        ],
    }
    CASES = [
        {
            "name": arm,
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"defer": defer},
        }
        for arm, defer in (("wait", 0), ("defer", 1))
    ]
    RTOL = 0.0
    ATOL = 0.0

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.arange(ELEMS, dtype=torch.float32) % 251.0 + float(rank * 1000)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(ELEMS, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(1, dtype=torch.float32)),
                    TensorArg(f"stamp_{rank}", torch.zeros(4, dtype=torch.int64)),
                ]
            )
        specs.append(Scalar("defer", ctypes.c_uint64(params["defer"])))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            peer = (rank + 1) % NRANKS
            expected = getattr(args, f"in_{peer}")
            getattr(args, f"out_{rank}").copy_(expected)
            getattr(args, f"result_{rank}")[0] = expected[0] + 1.0
            getattr(args, f"stamp_{rank}").zero_()

    def compare_outputs(self, test_args, golden_args, output_names, params):
        for rank in range(NRANKS):
            for name in (f"out_{rank}", f"result_{rank}"):
                actual = getattr(test_args, name)
                golden = getattr(golden_args, name)
                if not torch.equal(actual, golden):
                    raise AssertionError(f"{name} mismatch")
            stamp = getattr(test_args, f"stamp_{rank}")
            start, after_xfer, consumer = int(stamp[0]), int(stamp[1]), int(stamp[2])
            if start <= 0 or after_xfer <= start or consumer <= start:
                raise AssertionError(f"rank {rank} stamp start={start} after={after_xfer} consumer={consumer}")
            _record(params["defer"], rank, start, after_xfer, consumer)
        del output_names


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
