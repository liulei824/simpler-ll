#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 TMR benchmark for deferred-completion scheduling versus in-core Wait.

Modes
  0 L0: one core, repeated TGET + Wait. Stamp the device counter around the loop.
  1 fused: Wait and the checksum stay in that core.
  2 split Wait: the core Waits, then a dependent consumer sums the first words.
  3 split defer: the core posts and returns; the scheduler releases the consumer.
  4 overlap Wait: 31 explicit-channel gets plus 56 AIV spin tasks.
  5 overlap defer: the same graph, completions polled by the scheduler.
  6 compute: the spin tasks alone, used to pick a cycle budget.

SDMA channels are passed in. URMA stays at one in-flight get because its
session builder still reads the unprogrammed CCE block index. A consumer
plus 31 dests is the 32-tensor core limit.
"""

import ctypes
import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

N_XFER = int(os.environ.get("ASYNC_SCHED_N_XFER", "31"))
N_COMPUTE = 56
# 2 ms at the 1 GHz AICore counter. Replaced after the L0 64 MiB measurement.
COMPUTE_CYCLES = int(os.environ.get("ASYNC_SCHED_COMPUTE_CYCLES", "2000000"))

SIZES = (
    ("64kib", 64 * 1024),
    ("1mib", 1024 * 1024),
    ("16mib", 16 * 1024 * 1024),
    ("64mib", 64 * 1024 * 1024),
)

MODE_L0 = 0
MODE_FUSED = 1
MODE_WAIT = 2
MODE_DEFER = 3
MODE_OVERLAP_WAIT = 4
MODE_OVERLAP_DEFER = 5
MODE_COMPUTE = 6


def _case(name, *, mode, nbytes, nranks, engine, reps, n_xfer, n_compute, cycles):
    return {
        "name": name,
        "platforms": ["a5"],
        "config": {"device_count": nranks, "num_sub_workers": 0},
        "params": {
            "mode": mode,
            "nbytes": nbytes,
            "nranks": nranks,
            "engine": engine,
            "reps": reps,
            "n_xfer": n_xfer,
            "n_compute": n_compute,
            "cycles": cycles,
        },
    }


def _build_cases():
    cases = []
    for size_name, nbytes in SIZES:
        cases.append(
            _case(
                f"l0_{size_name}",
                mode=MODE_L0,
                nbytes=nbytes,
                nranks=2,
                engine=0,
                reps=8,
                n_xfer=1,
                n_compute=0,
                cycles=0,
            )
        )
        for mode_name, mode in (("fused", MODE_FUSED), ("wait", MODE_WAIT), ("defer", MODE_DEFER)):
            cases.append(
                _case(
                    f"l1_{mode_name}_{size_name}",
                    mode=mode,
                    nbytes=nbytes,
                    nranks=2,
                    engine=0,
                    reps=1,
                    n_xfer=1,
                    n_compute=0,
                    cycles=0,
                )
            )
        for mode_name, mode in (("wait", MODE_OVERLAP_WAIT), ("defer", MODE_OVERLAP_DEFER)):
            cases.append(
                _case(
                    f"l2_{mode_name}_{size_name}",
                    mode=mode,
                    nbytes=nbytes,
                    nranks=2,
                    engine=0,
                    reps=1,
                    n_xfer=N_XFER,
                    n_compute=N_COMPUTE,
                    cycles=COMPUTE_CYCLES,
                )
            )
    for cycles in (500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000):
        cases.append(
            _case(
                f"compute_{cycles}",
                mode=MODE_COMPUTE,
                nbytes=0,
                nranks=2,
                engine=0,
                reps=1,
                n_xfer=0,
                n_compute=N_COMPUTE,
                cycles=cycles,
            )
        )
    for size_name, nbytes in SIZES:
        for mode_name, mode in (("l0", MODE_L0), ("fused", MODE_FUSED), ("wait", MODE_WAIT), ("defer", MODE_DEFER)):
            reps = 8 if mode == MODE_L0 else 1
            cases.append(
                _case(
                    f"urma_{mode_name}_{size_name}",
                    mode=mode,
                    nbytes=nbytes,
                    nranks=2,
                    engine=1,
                    reps=reps,
                    n_xfer=1,
                    n_compute=0,
                    cycles=0,
                )
            )
    for size_name, nbytes in (("1mib", 1024 * 1024), ("64mib", 64 * 1024 * 1024)):
        for mode_name, mode in (("wait", MODE_OVERLAP_WAIT), ("defer", MODE_OVERLAP_DEFER)):
            cases.append(
                _case(
                    f"ring_{mode_name}_{size_name}",
                    mode=mode,
                    nbytes=nbytes,
                    nranks=4,
                    engine=0,
                    reps=1,
                    n_xfer=N_XFER,
                    n_compute=N_COMPUTE,
                    cycles=COMPUTE_CYCLES,
                )
            )
    return cases


def _effective_n_xfer(params):
    mode = params["mode"]
    if mode == MODE_COMPUTE:
        return 0
    if mode in (MODE_L0, MODE_FUSED, MODE_WAIT, MODE_DEFER):
        return 1
    return params["n_xfer"]


def sched_bench_orch(orch, callables, task_args, config):
    params_mode = int(task_args.mode.value)
    nbytes = int(task_args.nbytes.value)
    nranks = int(task_args.nranks.value)
    engine = int(task_args.engine.value)
    n_xfer = _effective_n_xfer({"mode": params_mode, "n_xfer": int(task_args.n_xfer.value)})
    elems = max(nbytes // 4, 1)
    window_elems = max(n_xfer, 1) * elems
    engines = ("urma",) if engine == 1 else ("sdma",)
    needs_comm = params_mode != MODE_COMPUTE

    def submit(rank, handle):
        args = TaskArgs()
        if handle is None:
            window = _rehosted_ref_for(task_args, getattr(task_args, f"inp_{rank}"))
            comm = 0
        else:
            window = handle[rank].buffers["input"].tensor((window_elems,), DataType.FLOAT32)
            comm = handle.device_context(rank, workspace_engines=engines)
        args.add_tensor(window, TensorArgType.INPUT)
        args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"checksum_{rank}")), TensorArgType.INOUT)
        args.add_tensor(
            _rehosted_ref_for(task_args, getattr(task_args, f"stamp_{rank}")), TensorArgType.OUTPUT_EXISTING
        )
        args.add_scalar(comm)
        args.add_scalar(params_mode)
        args.add_scalar(nbytes)
        args.add_scalar(int(task_args.reps.value))
        args.add_scalar(int(task_args.n_xfer.value))
        args.add_scalar(int(task_args.n_compute.value))
        args.add_scalar(int(task_args.cycles.value))
        args.add_scalar(engine)
        orch.submit_next_level(callables.sched_bench, args, config, worker=rank)

    if not needs_comm:
        for rank in range(nranks):
            submit(rank, None)
        return

    input_nbytes = window_elems * 4
    with orch.allocate_domain(
        workspace_engines=engines,
        name="sched_bench",
        workers=list(range(nranks)),
        window_size=max(input_nbytes, 4096),
        buffers=[CommBufferSpec(name="input", dtype="float32", count=window_elems, nbytes=input_nbytes)],
    ) as handle:
        for rank in range(nranks):
            source = getattr(task_args, f"inp_{rank}")
            orch.copy_to(handle[rank].buffers["input"], _rehosted_buffer_for(task_args, source))
        for rank in range(nranks):
            submit(rank, handle)


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAsyncEngineSchedBench(SceneTestCase):
    CALLABLE = {
        "orchestration": sched_bench_orch,
        "callables": [
            {
                "name": "sched_bench",
                "orchestration": {
                    "source": "kernels/orchestration/sched_bench_orch.cpp",
                    "function_name": "sched_bench_orchestration",
                    "signature": [
                        D.IN,
                        D.INOUT,
                        D.OUT,
                        D.IN,
                        D.IN,
                        D.IN,
                        D.IN,
                        D.IN,
                        D.IN,
                        D.IN,
                        D.IN,
                    ],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_xfer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_fused.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.INOUT],
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.INOUT],
                    },
                    {
                        "func_id": 3,
                        "source": "kernels/aiv/kernel_compute.cpp",
                        "core_type": "aiv",
                        "signature": [D.OUT],
                    },
                ],
            }
        ],
    }
    CASES = [case for case in _build_cases() if case["params"]["nranks"] == 2]
    RTOL = 0.0
    ATOL = 0.0

    def generate_args(self, params):
        nranks = params["nranks"]
        n_xfer = _effective_n_xfer(params)
        elems = max(params["nbytes"] // 4, 1)
        specs = []
        for rank in range(nranks):
            if n_xfer == 0:
                inp = torch.zeros(1, dtype=torch.float32)
            else:
                parts = [
                    torch.full((elems,), float(rank * 1000 + i + 1), dtype=torch.float32) for i in range(n_xfer)
                ]
                inp = torch.cat(parts)
            checksum = torch.zeros(2, dtype=torch.float32)
            checksum[1] = float(n_xfer)
            specs.extend(
                [
                    TensorArg(f"inp_{rank}", inp),
                    TensorArg(f"checksum_{rank}", checksum),
                    TensorArg(f"stamp_{rank}", torch.zeros(4, dtype=torch.int64)),
                ]
            )
        specs.extend(
            [
                Scalar("mode", ctypes.c_uint64(params["mode"])),
                Scalar("nbytes", ctypes.c_uint64(params["nbytes"])),
                Scalar("nranks", ctypes.c_uint64(nranks)),
                Scalar("engine", ctypes.c_uint64(params["engine"])),
                Scalar("reps", ctypes.c_uint64(params["reps"])),
                Scalar("n_xfer", ctypes.c_uint64(params["n_xfer"])),
                Scalar("n_compute", ctypes.c_uint64(params["n_compute"])),
                Scalar("cycles", ctypes.c_uint64(params["cycles"])),
            ]
        )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        nranks = params["nranks"]
        n_xfer = _effective_n_xfer(params)
        for rank in range(nranks):
            checksum = getattr(args, f"checksum_{rank}")
            if n_xfer == 0:
                checksum.zero_()
            else:
                peer = (rank + 1) % nranks
                checksum[0] = float(sum(peer * 1000 + i + 1 for i in range(n_xfer)))
                checksum[1] = float(n_xfer)
            getattr(args, f"stamp_{rank}").zero_()

    def compare_outputs(self, test_args, golden_args, output_names, params):
        del output_names
        for rank in range(params["nranks"]):
            actual = getattr(test_args, f"checksum_{rank}")
            golden = getattr(golden_args, f"checksum_{rank}")
            if not torch.allclose(actual, golden, rtol=0.0, atol=0.0):
                raise AssertionError(f"rank {rank} checksum {actual.tolist()} != {golden.tolist()}")
            if params["mode"] == MODE_L0:
                stamp = getattr(test_args, f"stamp_{rank}")
                start = int(stamp[0])
                end = int(stamp[1])
                if end <= start or start <= 0:
                    raise AssertionError(f"rank {rank} L0 stamp start={start} end={end}")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
