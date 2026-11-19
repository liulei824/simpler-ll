# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-step decode inputs, output checksums, and raw bf16 output dumps.

``hidden_states`` comes from an independent CPU generator (std 1, then bf16),
so drawing a step does not advance the fixture RNG. ``seq_lens`` and
``slot_mapping`` reuse the paged layout in ``qwen3_14b_decode``. The digest is
blake2b-64 over the buffer ``out_raw_bytes`` returns; a dump is that buffer
appended once per step, ``OUT_BYTES_PER_STEP`` bytes at offset
``step * OUT_BYTES_PER_STEP``.
"""

from __future__ import annotations

import hashlib
import os
from typing import NamedTuple

import torch

from simpler_setup.goldens import qwen3_14b_decode as qwen3_decode

_DIGEST_SIZE = 8
OUT_BYTES_PER_STEP = qwen3_decode.BATCH * qwen3_decode.HIDDEN * 2


class StepInputs(NamedTuple):
    """Varying inputs for one decode step."""

    step: int
    seq_len: int
    hidden_states: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor


class OutChecksum(NamedTuple):
    """Checksum of one bf16 ``out``. ``digest`` is blake2b-64 over its raw bytes."""

    sum_f64: float
    max_abs: float
    digest: str
    finite: bool


def _require_seq_len(seq_len: int) -> int:
    if not (1 <= seq_len <= qwen3_decode.MAX_SEQ):
        raise ValueError(f"seq_len must be in [1, {qwen3_decode.MAX_SEQ}], got {seq_len}")
    return seq_len


def step_hidden_states(seed: int, step: int) -> torch.Tensor:
    """Draw ``(BATCH, HIDDEN)`` bf16 hidden states for one step.

    The generator is seeded with ``seed + step`` and is not the process RNG.
    Values are float32 ``normal_(0, 1)`` cast to bf16, matching the fixture
    ``hidden_states`` standard deviation.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed + step)
    hidden = torch.empty(qwen3_decode.BATCH, qwen3_decode.HIDDEN, dtype=torch.float32, device="cpu")
    hidden.normal_(0.0, 1.0, generator=generator)
    return hidden.to(torch.bfloat16)


def step_position_inputs(start_seq_len: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(seq_lens, slot_mapping)`` for sequence length ``start_seq_len + step``."""
    seq_len = _require_seq_len(start_seq_len + step)
    seq_lens = torch.full([qwen3_decode.BATCH], seq_len, dtype=torch.int32)
    slot_mapping = qwen3_decode._paged_block_table_slot_mapping(seq_lens)[1]
    return seq_lens, slot_mapping


def step_inputs(seed: int, start_seq_len: int, step: int) -> StepInputs:
    """Build the varying inputs for one decode step."""
    seq_len = _require_seq_len(start_seq_len + step)
    seq_lens, slot_mapping = step_position_inputs(start_seq_len, step)
    return StepInputs(
        step=step,
        seq_len=seq_len,
        hidden_states=step_hidden_states(seed, step),
        seq_lens=seq_lens,
        slot_mapping=slot_mapping,
    )


def max_steps(start_seq_len: int) -> int:
    """How many steps keep ``seq_len`` inside ``[1, MAX_SEQ]`` from ``start_seq_len``."""
    return qwen3_decode.MAX_SEQ - start_seq_len + 1


def out_raw_bytes(out: torch.Tensor) -> bytes:
    """Row-major bf16 bytes of ``out``.

    ``out_checksum`` hashes this buffer, and the dump stores it unchanged.
    """
    bits = out.detach().cpu().contiguous().view(torch.int16)
    return bits.numpy().tobytes()


def out_checksum(out: torch.Tensor) -> OutChecksum:
    """Checksum ``out``. ``digest`` is blake2b-64 of ``out_raw_bytes(out)``."""
    raw = out_raw_bytes(out)
    values = out.detach().to(dtype=torch.float64)
    return OutChecksum(
        sum_f64=float(values.sum().item()),
        max_abs=float(values.abs().max().item()),
        digest=hashlib.blake2b(raw, digest_size=_DIGEST_SIZE).hexdigest(),
        finite=bool(torch.isfinite(out).all().item()),
    )


def load_saved_out(path: str, step: int) -> torch.Tensor:
    """Load one dumped step as a contiguous CPU bf16 tensor of shape ``(BATCH, HIDDEN)``.

    The record at ``step`` is the ``OUT_BYTES_PER_STEP`` bytes written by
    ``out_raw_bytes``, starting at byte offset ``step * OUT_BYTES_PER_STEP``.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    offset = step * OUT_BYTES_PER_STEP
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(OUT_BYTES_PER_STEP)
    if len(raw) != OUT_BYTES_PER_STEP:
        raise ValueError(f"step {step} needs {OUT_BYTES_PER_STEP} bytes at offset {offset}, got {len(raw)}")
    bits = torch.frombuffer(bytearray(raw), dtype=torch.int16).clone()
    return bits.view(torch.bfloat16).reshape(qwen3_decode.BATCH, qwen3_decode.HIDDEN)


def saved_out_steps(path: str) -> int:
    """Number of whole steps in a dump. A trailing partial record is ignored."""
    return os.path.getsize(path) // OUT_BYTES_PER_STEP
