# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for per-step Qwen3-14B decode inputs, checksums, and output dumps."""

import hashlib

import pytest
import torch

from simpler_setup.goldens.qwen3_14b_decode import BLOCK_SIZE, MAX_BLOCKS_PER_SEQ, MAX_SEQ
from simpler_setup.goldens.qwen3_14b_decode_steps import (
    OUT_BYTES_PER_STEP,
    load_saved_out,
    max_steps,
    out_checksum,
    out_raw_bytes,
    saved_out_steps,
    step_hidden_states,
    step_inputs,
    step_position_inputs,
)

_SEED = 1234


def test_same_seed_and_step_match_and_steps_differ():
    torch.manual_seed(0)
    first = step_hidden_states(_SEED, 0)
    torch.manual_seed(999)
    second = step_hidden_states(_SEED, 0)
    other = step_hidden_states(_SEED, 1)
    assert out_raw_bytes(first) == out_raw_bytes(second)
    assert out_raw_bytes(first) != out_raw_bytes(other)

    torch.manual_seed(1)
    step0 = step_inputs(_SEED, 3500, 0)
    torch.manual_seed(2)
    step0_again = step_inputs(_SEED, 3500, 0)
    step1 = step_inputs(_SEED, 3500, 1)
    assert out_raw_bytes(step0.hidden_states) == out_raw_bytes(step0_again.hidden_states)
    assert torch.equal(step0.seq_lens, step0_again.seq_lens)
    assert torch.equal(step0.slot_mapping, step0_again.slot_mapping)
    assert out_raw_bytes(step0.hidden_states) != out_raw_bytes(step1.hidden_states)
    assert step0.seq_len != step1.seq_len


def test_step_inputs_shape_dtype_and_device():
    inputs = step_inputs(_SEED, 3500, 4)
    hidden = inputs.hidden_states
    assert inputs.step == 4
    assert inputs.seq_len == 3504
    assert hidden.shape == (16, 5120)
    assert hidden.dtype == torch.bfloat16
    assert hidden.is_contiguous()
    assert hidden.device.type == "cpu"
    assert inputs.seq_lens.shape == (16,)
    assert inputs.seq_lens.dtype == torch.int32
    assert inputs.seq_lens.is_contiguous()
    assert inputs.seq_lens.device.type == "cpu"
    assert torch.equal(inputs.seq_lens, torch.full((16,), 3504, dtype=torch.int32))
    assert inputs.slot_mapping.shape == (16,)
    assert inputs.slot_mapping.dtype == torch.int32
    assert inputs.slot_mapping.is_contiguous()
    assert inputs.slot_mapping.device.type == "cpu"


@pytest.mark.parametrize("seq_len", [1, 128, 129, 3500, 5500])
def test_slot_mapping_matches_paged_formula(seq_len):
    assert MAX_BLOCKS_PER_SEQ == 43
    assert BLOCK_SIZE == 128
    seq_lens, slot_mapping = step_position_inputs(seq_len, 0)
    pos = seq_len - 1
    assert torch.equal(seq_lens, torch.full((16,), seq_len, dtype=torch.int32))
    for lane in range(16):
        expected = (lane * 43 + pos // 128) * 128 + pos % 128
        assert int(slot_mapping[lane]) == expected

    shifted_lens, shifted_slots = step_position_inputs(seq_len - 1, 1) if seq_len > 1 else (seq_lens, slot_mapping)
    assert torch.equal(shifted_slots, slot_mapping)
    assert torch.equal(shifted_lens, seq_lens)


@pytest.mark.parametrize(
    ("start_seq_len", "step"),
    [(0, 0), (1, -1), (-4, 0), (MAX_SEQ + 1, 0), (MAX_SEQ, 1), (5400, 101)],
)
def test_seq_len_out_of_range_raises(start_seq_len, step):
    with pytest.raises(ValueError, match=r"seq_len must be in \[1, 5500\]"):
        step_position_inputs(start_seq_len, step)
    with pytest.raises(ValueError, match=r"seq_len must be in \[1, 5500\]"):
        step_inputs(_SEED, start_seq_len, step)


def test_max_steps_and_inclusive_seq_len_bounds():
    assert max_steps(3500) == 2001
    assert max_steps(1) == MAX_SEQ
    assert max_steps(MAX_SEQ) == 1
    for seq_len in (1, MAX_SEQ):
        seq_lens, _slot_mapping = step_position_inputs(seq_len, 0)
        assert int(seq_lens[0]) == seq_len
        assert step_inputs(_SEED, seq_len, 0).seq_len == seq_len


def test_out_checksum_is_stable_sensitive_and_reports_nonfinite():
    out = step_hidden_states(_SEED, 0)
    first = out_checksum(out)
    second = out_checksum(out)
    assert first == second
    assert len(first.digest) == 16
    assert first.digest == hashlib.blake2b(out_raw_bytes(out), digest_size=8).hexdigest()
    assert first.sum_f64 == float(out.to(torch.float64).sum())
    assert first.max_abs == float(out.to(torch.float64).abs().max())
    assert first.finite is True
    assert isinstance(first.sum_f64, float)
    assert isinstance(first.max_abs, float)

    mutated = out.clone()
    bits = mutated.view(torch.int16)
    bits[0, 0] = bits[0, 0] ^ 1
    assert out_raw_bytes(mutated) != out_raw_bytes(out)
    assert out_checksum(mutated).digest != first.digest

    nonfinite = out.clone()
    nonfinite[0, 0] = float("nan")
    assert out_checksum(nonfinite).finite is False
    nonfinite = out.clone()
    nonfinite[1, 2] = float("inf")
    assert out_checksum(nonfinite).finite is False


def test_saved_out_roundtrip(tmp_path):
    path = tmp_path / "steps.out.bin"
    tensors = [step_hidden_states(_SEED, step) for step in range(3)]
    assert len({out_checksum(tensor).digest for tensor in tensors}) == 3
    with path.open("wb") as handle:
        for tensor in tensors:
            handle.write(out_raw_bytes(tensor))

    assert OUT_BYTES_PER_STEP == 163840
    assert path.stat().st_size == 3 * OUT_BYTES_PER_STEP
    assert saved_out_steps(path) == 3
    for step, tensor in enumerate(tensors):
        loaded = load_saved_out(path, step)
        assert loaded.shape == (16, 5120)
        assert loaded.dtype == torch.bfloat16
        assert loaded.is_contiguous()
        assert loaded.device.type == "cpu"
        assert out_raw_bytes(loaded) == out_raw_bytes(tensor)
        assert out_checksum(loaded).digest == out_checksum(tensor).digest
