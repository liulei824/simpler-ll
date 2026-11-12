#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""4-rank SDMA ring, peer = (rank + 1) % 4.

Lives in its own module: a class's device_count is the max over every case,
so these cases cannot share a class with the 2-rank sweep.
"""

import ctypes
import os
import pathlib
import tempfile

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

try:
    from simpler.task_interface import _bind_host_log_session_directory

    _bind_host_log_session_directory()
except Exception:
    pass

NRANKS = 4
DTYPE_NBYTES = 4
N_XFER = int(os.environ.get("ASYNC_SCHED_N_XFER", "31"))
COMPUTE_CYCLES = int(os.environ.get("ASYNC_SCHED_COMPUTE_CYCLES", "2170000"))
# Same 256 MiB runtime-heap budget as the 2-rank sweep.
_HEAP_BUDGET = 192 * 1024 * 1024
SIZES = {
    "64kib": 64 * 1024,
    "1mib": 1024 * 1024,
    "16mib": 16 * 1024 * 1024,
    "64mib": 64 * 1024 * 1024,
}
WIN = os.environ.get("ASYNC_SCHED_RING4_WIN", "1mib")
LOSS = os.environ.get("ASYNC_SCHED_RING4_LOSS", "16mib")
_REPO = pathlib.Path(__file__).resolve().parents[4]
WALL_LOG = os.environ.get("ASYNC_SCHED_RING4_WALL", str(_REPO / "outputs/async-engine-sched/l2_ring4_device_wall.log"))
_WALL_OFFSETS = {}


def _n_for(nbytes):
    fit = max(1, _HEAP_BUDGET // int(nbytes))
    return min(N_XFER, fit)


def ring4_orch_fn(orch, callables, task_args, config):
    nbytes = int(task_args.nbytes.value)
    n_xfer = int(task_args.n_xfer.value)
    elems = nbytes // DTYPE_NBYTES
    window_elems = max(n_xfer, 1) * elems
    input_nbytes = window_elems * DTYPE_NBYTES
    with orch.allocate_domain(
        workspace_engines=("sdma",),
        name="default",
        workers=list(range(NRANKS)),
        window_size=max(input_nbytes, 4 * 1024),
        buffers=[CommBufferSpec(name="input_window", dtype="float32", count=window_elems, nbytes=input_nbytes)],
    ) as handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(
                domain.buffers["input_window"].tensor((window_elems,), DataType.FLOAT32), TensorArgType.INPUT
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.INOUT
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"stamp_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(int(task_args.defer.value))
            args.add_scalar(nbytes)
            args.add_scalar(int(task_args.reps.value))
            args.add_scalar(int(task_args.cycles.value))
            orch.submit_next_level(callables.sdma_wait_minimal, args, config, worker=rank)


def _cases():
    chosen = []
    for label, size_name in (("win", WIN), ("loss", LOSS)):
        if size_name not in SIZES or size_name in {name for _, name in chosen}:
            continue
        chosen.append((label, size_name))
    out = []
    for label, size_name in chosen:
        for arm, defer in (("wait", 0), ("defer", 1)):
            out.append(
                {
                    "name": f"l2_{arm}_{size_name}",
                    "platforms": ["a5"],
                    "config": {"device_count": NRANKS, "num_sub_workers": 0},
                    "params": {
                        "defer": defer,
                        "nbytes": SIZES[size_name],
                        "reps": 1,
                        "n_xfer": _n_for(SIZES[size_name]),
                        "cycles": COMPUTE_CYCLES,
                        "point": label,
                    },
                }
            )
    return out


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestSdmaL2Ring4(SceneTestCase):
    CALLABLE = {
        "orchestration": ring4_orch_fn,
        "callables": [
            {
                "name": "sdma_wait_minimal",
                "orchestration": {
                    "source": "kernels/orchestration/sdma_wait_minimal_orch.cpp",
                    "function_name": "sdma_wait_minimal_orchestration",
                    "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": func_id,
                        "source": source,
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN],
                    }
                    for func_id, source in enumerate(
                        [
                            "kernels/aiv/kernel_sdma_tget_wait.cpp",
                            "kernels/aiv/kernel_consumer.cpp",
                            "kernels/aiv/kernel_compute.cpp",
                            "kernels/aiv/kernel_l2_consumer.cpp",
                        ]
                    )
                ],
            }
        ],
    }
    CASES = _cases()
    RTOL = 0.0
    ATOL = 0.0

    def generate_args(self, params):
        n_xfer = int(params["n_xfer"])
        elems = params["nbytes"] // DTYPE_NBYTES
        specs = []
        for rank in range(NRANKS):
            parts = [torch.full((elems,), float(rank * 1000 + i + 1), dtype=torch.float32) for i in range(n_xfer)]
            result = torch.zeros(2, dtype=torch.float32)
            result[1] = float(n_xfer)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", torch.cat(parts)),
                    TensorArg(f"out_{rank}", torch.zeros(elems, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", result),
                    TensorArg(f"stamp_{rank}", torch.zeros(4, dtype=torch.int64)),
                ]
            )
        specs.extend(
            [
                Scalar("defer", ctypes.c_uint64(params["defer"])),
                Scalar("nbytes", ctypes.c_uint64(params["nbytes"])),
                Scalar("reps", ctypes.c_uint64(1)),
                Scalar("n_xfer", ctypes.c_uint64(n_xfer)),
                Scalar("cycles", ctypes.c_uint64(params["cycles"])),
            ]
        )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        n_xfer = int(params["n_xfer"])
        for rank in range(NRANKS):
            peer = (rank + 1) % NRANKS
            result = getattr(args, f"result_{rank}")
            result[0] = float(sum(peer * 1000 + i + 1 for i in range(n_xfer)))
            result[1] = float(n_xfer)
            getattr(args, f"stamp_{rank}").zero_()

    def compare_outputs(self, test_args, golden_args, output_names, params):
        for rank in range(NRANKS):
            actual = getattr(test_args, f"result_{rank}")
            golden = getattr(golden_args, f"result_{rank}")
            if not torch.equal(actual, golden):
                raise AssertionError(f"result_{rank} mismatch {actual.tolist()} != {golden.tolist()}")
            stamp = getattr(test_args, f"stamp_{rank}")
            start, after_xfer = int(stamp[0]), int(stamp[1])
            if start <= 0 or after_xfer <= start:
                raise AssertionError(f"rank {rank} stamp start={start} after={after_xfer}")
        _harvest_device_wall(params)
        del output_names


def _descendant_pids(root):
    by_ppid = {}
    for status in pathlib.Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status.parent.name)
            ppid = None
            for line in status.read_text().splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid is not None:
                by_ppid.setdefault(ppid, []).append(pid)
        except (OSError, ValueError):
            continue
    found = []
    stack = list(by_ppid.get(root, []))
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(by_ppid.get(pid, []))
    return found


def _harvest_device_wall(params):
    path = pathlib.Path(WALL_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    dirs = []
    try:
        from _task_interface import _host_log_directory

        bound = _host_log_directory() or ""
        if bound:
            dirs.append(pathlib.Path(bound))
    except Exception:
        pass
    tmp = pathlib.Path(tempfile.gettempdir())
    for pid in [os.getpid(), *_descendant_pids(os.getpid())]:
        dirs.extend(p for p in tmp.glob(f"simpler-host-logs-{pid}-*") if p.is_dir())
    with path.open("a") as out:
        out.write(f"# nbytes={params['nbytes']} defer={params['defer']} n={params.get('n_xfer', 1)}\n")
        seen = set()
        for directory in dirs:
            if str(directory) in seen or not directory.is_dir():
                continue
            seen.add(str(directory))
            for src in directory.glob("host.*.log"):
                key = str(src)
                offset = _WALL_OFFSETS.get(key, 0)
                with src.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                    _WALL_OFFSETS[key] = offset + len(chunk)
                for line in chunk.decode("utf-8", "replace").splitlines():
                    if "name=chip.run.runner_run.device_wall " in line:
                        out.write(line + "\n")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
