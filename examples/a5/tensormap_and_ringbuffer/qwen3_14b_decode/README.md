# `qwen3_14b_decode/` — Qwen3-14B 40-layer decode (CANN fused attention)

> **a5 port.** This is the a5 port of
> [`examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/`](../../a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/).
> The kernels are the a2a3 harvest with the arch deltas listed in
> "Relationship to the a2a3 example" below — the math is untouched, but the
> a2a3 harvest is not byte-portable: a5 spells the L0A fractal, the legal
> `pipe_barrier` pipes and the core-count accessor differently, and the a2a3
> `__DAV_C220_*` guards name a macro ccec does not predefine on a5.

Self-contained standalone-driver port of pypto-lib
`models/qwen3/14b/decode_fwd.py` entry `decode_fwd_layers` with
`_CHUNK_NLAYERS == 40`: **the whole Qwen3-14B decode stack as one fused
dispatch** (hidden → hidden, no LM head), with the FP32 inter-layer residual
carry. A simpler developer builds and runs it directly — no descent through
pypto-lib / the JIT, no auto-built intermediate artifacts.

The layer loop is a real loop in the generated orchestration
(`for (int64_t i = 0; i < 40; i += 1)`), not an unrolled one, so a 40-layer
chunk costs one extra literal over a 2-layer chunk rather than 20× the kernels.

## Parameter regime — matches `stress_profile.py`

The fixture mirrors the vLLM serving stress run (`stress_profile.py`):

| Param | Value | Source |
| ----- | ----- | ------ |
| `BATCH` | 16 | `CONCURRENCY` (aligned with decode kernel BATCH=16) |
| `MAX_SEQ` | 5500 | `max_model_len` (KV-pool / RoPE sizing) |
| decode `seq_len` | 3500 | the ~3500-token prompt |
| layers | 40 | full model (`decode_fwd_layers` N=40) |

Per the lib's const-layer-0 stacked-fwd reference, every layer reuses layer-0
weights (weights + paged KV pool are stacked ×40 along dim 0, one slice per
layer); each layer still reads and writes its own KV pool.

Footprint at this regime, bf16:

| component | size |
| --------- | ---: |
| weights (×40) | 24.61 GiB |
| paged KV pool (×40) | 13.44 GiB |
| **total fixture** | **38.05 GiB** |

Die HBM is 64 GiB, so it fits with ~26 GiB of headroom for ring heap and
workspace. The run needs no `runtime_env` tuning — the default ring heap, task
window and dep pool carry the 40-layer graph, because each layer's
intermediates live inside that iteration's scope and are freed at its end, so
the live set does not grow with layer count.

## Dataflow per layer (`_decode_layer`)

input RMSNorm → split-K SPMD Q/K/V (seed + atomic-add) → **`paged_attention_rope_cce`**
→ split-K out_proj + residual → post-RMSNorm → SwiGLU FFN → `dcr_xgamma`.

`copy_hidden` embeds the bf16 input; the FP32 residual is carried between
layers; `copy_out` does the single FP32→bf16 round at the chunk tail.

The attention stage is one **CANN `FusedInferAttentionScore` extern** that
subsumes what used to be seven generated kernels — it folds per-head Q/K
RMS-norm, RoPE, the paged KV write, the flash-attention inner loop and the
online softmax into a single mixed (AIC + 2×AIV) task, gated by an
`AscendC::SyncAll<false>()` FFTS barrier. `paged_attention_tiling_cce` builds
its runtime tiling metadata first.

Paged KV uses vLLM's **BSND** layout: a page holds `[BLOCK_SIZE, KV_HIDDEN]`
ordered `[page, token, kv_head, dim]`, so `slot_mapping[b]` is directly the
row index. (The previous harvest used NSND; the golden was updated to match.)

## Provenance — how the C++ was produced

| component | source |
| --------- | ------ |
| pypto-lib | `45be52c` |
| pypto | `d64380cb` |
| ptoas | `v0.48` |
| pto-isa | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` (`pto_isa.pin`) |

`kernels/orchestration/` + `kernels/aic/` (18) + `kernels/aiv/` (16) are
harvested pypto codegen for `decode_fwd_layers` (`_CHUNK_NLAYERS=40`,
`PTO2_MANUAL_MAX_SEQ=5500`) — license header prepended, otherwise verbatim.
The `CALLABLE` is transcribed from that run's `kernel_config.py`, and
`simpler_setup/goldens/qwen3_14b_decode.py` ports the per-layer
`golden_decode_layer` math (RoPE θ=1e4, controlled scales, FP32 residual, bf16
cast points) composed over 40 layers with FP32 carry + per-layer KV pools.

**There are no hand-edits.** The previous harvest patched `fa_fused_aiv` to work
around a `[[block_local]] static`; that kernel no longer exists (attention is
the extern now), and the current codegen emits no such construct.

### `kernels/vendor/paged_attention_cce/` — the attention extern

Copied verbatim from pypto-lib
`models/qwen3/14b/kernels/paged_attention_cce/`. The tree must stay intact:
`kernel/fai_body.hpp` reaches its dependencies through relative includes, so
splitting it would mean patching the source and re-patching on every refresh.

- `attention_rope/`, `tiling/`, `kernel/`, `generated/` — PyPTO-authored glue.
- `vendor/fused_infer_attention_score/` — **CANN `FusedInferAttentionScore`,
  Copyright (c) 2025 Huawei Technologies, CANN Open Software License 2.0**
  (~16 k LOC), upstream's own vendored copy, left where upstream put it.
- `attention/` is the non-RoPE variant of the extern. `decode_fwd_layers` does
  not use it; it is kept so a refresh is a plain directory copy.

**Why it sits under `kernels/vendor/`.** Nothing below a `vendor/` directory is
ours to reformat: the repo's header, formatting and language lint all skip that
path (`.pre-commit-config.yaml`, `tests/lint/check_headers.py`). Without that,
`clang-format` rewrites the glue files and `end-of-file-fixer` touches the CANN
headers, and the next refresh diffs against *our* reformatting instead of
against upstream — which is how the drift this example is meant to expose starts.
The carve-out keys on the directory, not on this operator's name, so harvesting
another extern is a matter of dropping it in `kernels/vendor/` with no lint
change at all.

Building these needs **CANN devkit headers** (`$ASCEND_HOME_PATH/aarch64-linux/asc/…`,
`tikcpp/…`), declared per-incore via `extra_include_dirs` in the `CALLABLE`.
`$ASCEND_HOME_PATH` keeps them machine-independent; paths a given CANN layout
does not ship are dropped rather than failing the build.

They also depend on simpler linking incore objects before extracting `.text`
([#1497](https://github.com/hw-native-sys/simpler/pull/1497)): AscendC declares
`g_vecTPipePtr` / `g_kfcClient` as block-local globals, whose relocations no
amount of inlining removes.

### To regenerate

From a simpler worktree with pypto + pypto-lib cloned under `build/` (see the
[`multi-repo-setup`](../../../../.claude/skills/multi-repo-setup/SKILL.md)
skill) and `eval "$(pypto-setup --export)"`:

```python
# PTO2_MANUAL_MAX_SEQ must be set before importing decode_fwd (read at import).
os.environ["PTO2_MANUAL_MAX_SEQ"] = "5500"
D = <import build/pypto-lib/models/qwen3/14b/decode_fwd.py by path>
D._CHUNK_NLAYERS = 40          # read at trace time; rebind before the first call
D.decode_fwd_layers(*inputs, out, config=RunConfig(
    platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=OUT))
```

`codegen_only` needs no device. Then copy `OUT/orchestration/`, `OUT/kernels/`
and `models/qwen3/14b/kernels/paged_attention_cce/` into `kernels/vendor/` here, and
re-transcribe `CALLABLE` from `OUT/kernel_config.py` (which already records
`func_id`, `core_type`, per-kernel `signature`, and `extra_include_dirs`).

The harvest is a2a3 codegen, so re-apply the deltas from "Relationship to the
a2a3 example" afterwards. The two mechanical ones are one pass each:

```bash
cd examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode
grep -rlZ '__DAV_C220_' kernels | xargs -0 sed -i \
    's/__DAV_C220_VEC__/__DAV_VEC__/g; s/__DAV_C220_CUBE__/__DAV_CUBE__/g'
grep -rlZ 'pipe_barrier(PIPE_\(V\|MTE1\|FIX\));' kernels | xargs -0 sed -i \
    '/^[[:space:]]*pipe_barrier(PIPE_\(V\|MTE1\|FIX\));[[:space:]]*$/d'
```

The Left-tile fractal and the `fai_body.hpp` edits (include order, soft Mix
`SYNCALL`) are not mechanical — take them from this tree's history.

One deliberate deviation from `kernel_config.py`: `decode_fwd_layers` declares
`k_cache` / `v_cache` as plain inputs, but the extern writes the current token's
KV into them. The callable marks them `INOUT`; correctness runs explicitly copy
the pools back so the golden can verify all 40 layers' KV writes, not just the
hidden output.

## Running

```bash
# pytest (hardware; wrap in task-submit on shared boxes)
pytest examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode \
    --platform a5 --device ${DEVICE} --manual include

# standalone correctness
python examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py \
    -p a5 -d ${DEVICE}

# benchmark: one fixture upload, then device-only parameter reuse across rounds
python examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py \
    -p a5 -d ${DEVICE} --rounds 100 --skip-golden
```

DFX is opt-in on the standalone driver — no kernel changes needed:

```bash
python .../qwen3_14b_decode/main.py -p a5 -d ${DEVICE} \
    --enable-dep-gen

python .../qwen3_14b_decode/main.py -p a5 -d ${DEVICE} \
    --enable-chip-swimlane 1
```

Note that dependency generation or chip swimlane on the full 40-layer graph can
overflow the per-run SHM record buffer ("records dropped"); pypto-lib warns
about the same thing for `decode_fwd.py --fwd-layers`. Capture on a smaller
harvest if you need a clean trace.

[Varying-input mode](#varying-input-mode) is the step loop that rewrites three
inputs and reads `out` back on every step. The `--rounds` commands above leave
that flag off.

## Status — a5 port

Passes on a5 silicon against the same torch reference as the a2a3 source, at
`RTOL=5e-2 / ATOL=1e-1`: the hidden output **and all 40 layers' K and V caches**
match, with every element inside tolerance (worst element `out` 0.0625,
`k_cache` / `v_cache` 0.03125 — one bf16 quantum at those magnitudes).

## Memory and round behavior

All 20 entry tensors are allocated once with `Worker.malloc`. In
`--skip-golden` benchmark mode, the 19 inputs are generated and uploaded one at
a time, so the host never retains the complete 38 GiB fixture. The flag still
uploads valid model weights, KV cache, and paging metadata; it skips only the
torch reference and explicit D2H comparison. Every round reuses the same device
addresses and does no automatic parameter staging or copy-back.

Correctness mode materializes the host fixture once, uploads that same fixture,
computes the torch golden in place, then reads back `out`, `k_cache`, and
`v_cache` one at a time after the last round. The freshly allocated `out` buffer
is not uploaded because the final `copy_out` task overwrites every element.

## Varying-input mode

`--varying-input` keeps the one-time `Worker.malloc` and the one-time fixture
upload. Each step then rewrites three inputs, calls `worker.run`, and copies
`out` back. The flag defaults to off. Without it, `--steps`,
`--start-seq-len`, `--warmup-steps`, `--step-record`, and `--save-out` are
ignored, and the driver stays on the path in
[Memory and round behavior](#memory-and-round-behavior). The exit code and
`[qwen] PASSED` mean the same thing on both paths.

### Flags

| Flag | Type / default | Meaning |
| ---- | -------------- | ------- |
| `--varying-input` | `store_true` / `False` | Run the step loop |
| `--steps` | int / `200` | Steps to run and record, including warmup. Takes effect only with `--varying-input` |
| `--start-seq-len` | int / `None` | `seq_len` of step 0. `None` falls back to `--seq-len` (default `3500`) |
| `--warmup-steps` | int / `3` | Leading steps still run and are recorded. The steady-state summary leaves them out |
| `--step-record` | path / `None` | JSONL path for the per-step record. `None` prints the summary and writes no JSONL file |
| `--save-out` | path / `None` | Append every step's full `out` as raw bf16 bytes. When `--step-record` is set and this flag is omitted, the dump is that path's directory and stem with a `.out.bin` suffix. An explicit `""`, or no `--step-record`, writes no dump |

`run()` checks the combination before compilation and before the fixture
upload. A violation raises `ValueError`:

- With `--varying-input`, `rounds` must be 1. The step loop and the `--rounds`
  loop are mutually exclusive.
- `steps >= 1` and `0 <= warmup_steps < steps`.
- `start_seq_len >= 1` and `start_seq_len + steps - 1 <= 5500` (`MAX_SEQ`).
  The error names `max_steps(start_seq_len)`, which is
  `5500 - start_seq_len + 1`. A longer sequence indexes `rope_cos` /
  `rope_sin` past the last row.
- With `--varying-input` and without `--skip-golden`, `steps` must be 1.
  The golden is one forward of one input set.
- The parent directory of `--save-out` must already exist and be writable.
  When the target file already exists, `run()` raises `ValueError` and
  leaves that file untouched.
- Without `--varying-input`, the five flags above are ignored.

```bash
python examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py \
    -p a5 -d ${DEVICE} --varying-input --skip-golden \
    --steps 200 --start-seq-len 3500 --warmup-steps 3 \
    --step-record /tmp/qwen3_steps.jsonl
```

`--step-record` is set and `--save-out` is omitted, so this also writes
`/tmp/qwen3_steps.out.bin`. Pass `--save-out ""` to keep the JSONL and skip
the binary dump.

- `--varying-input --skip-golden` uploads the fixture, skips the torch
  golden, and on every step generates the inputs, takes the latency window
  below, then checksums `out` and appends the records. It prints the summary
  and closes the files.
- `--varying-input` without `--skip-golden` (so `--steps 1`) builds the
  fixture, replaces `hidden_states`, `seq_lens`, and `slot_mapping` in place
  with step 0, uploads, runs `_decode_golden`, runs that one step through the
  same timing and record path, then `_copy_and_compare`.
- Without `--varying-input`, the driver runs the `--rounds` loop and, when a
  golden was built, one `_copy_and_compare` after the last round.

### Inputs each step

After the upload, only three tensors are written again. Each write is
`worker.copy_to` into the buffer already allocated for that name. The step
keeps the existing task arguments, the existing device allocations, and the
same orchestration.

| Tensor | dtype / shape | Each step |
| ------ | ------------- | --------- |
| `hidden_states` | bf16 `(16, 5120)`, 160 KiB | a new random hidden state |
| `seq_lens` | int32 `(16,)` | all 16 lanes set to `start_seq_len + step` |
| `slot_mapping` | int32 `(16,)` | recomputed from `seq_lens` |

Step `k` (from 0, warmup included) has `seq_len = start_seq_len + k`. Position
advance is entirely that change to `seq_lens`.

The rest of the fixture stays resident for the whole run:

- `rope_cos` and `rope_sin` are the full `(5500, 128)` fp32 tables, covering
  positions 0 through 5499. The kernel indexes them at `seq_lens[b] - 1`.
- `block_table`, shape `(688,)`, is the identity table over all `16×43`
  physical pages. This fixture keeps that full-pool table on every step.
- The eleven weight tensors are uploaded once and reused.
- `k_cache` and `v_cache` stay resident. The attention extern writes the new
  row selected by `slot_mapping`. Each step uses a new slot, so the row
  written on the previous step stays as it was.

### Latency

`latency_us` is the primary per-step latency: a single host wall clock. The
clock starts before the first `copy_to` and stops after `copy_from(out)`
returns. With `t0` and `t1` taken from `time.monotonic_ns()` at those two
points, `latency_us = (t1 - t0) / 1000` (microseconds). The window contains,
in order, the three `copy_to` calls, the blocking `worker.run`, and
`copy_from(out)`.

`gen_us` (building this step's inputs), `checksum_us` (the `out` checksum and
digest), and `io_us` (appending the JSONL line and the `out` bytes, then
flushing both) sit outside that window. The host buffer that receives `out`
is allocated once before the loop and reused, so that allocation is outside
the window as well.

Inside the window the driver also records three segments: `copy_in_us` (the
three `copy_to` calls), `run_us` (`worker.run`), and `copy_out_us`
(`copy_from(out)`).

```text
segments_us = copy_in_us + run_us + copy_out_us
glue_us = latency_us - segments_us
```

`segments_us` is the sum of those three segments. The step latency is
`latency_us`. `glue_us` is the Python glue and the timing calls between the
segments, so `segments_us <= latency_us` and `glue_us >= 0`.

The second layer is the `chip.run` trace. `[STRACE]` stays on, each step is
one `chip.run`, and the log holds one round line per step:

```bash
python -m simpler_setup.tools.strace_timing <log> --rounds-table
```

`<log>` is the driver stdout. `--rounds-table` prints one row per step,
including Host / Device / Orch / Sched, on the same definitions as a
fixed-input `--rounds` run.
That table explains the composition of `latency_us`. `copy_to` and
`copy_from` are bare copies and sit outside the `chip.run` tree, so
`latency_us` minus the `chip.run` host time is the three input copies, the
`out` copy, and the Python driver overhead around them.

### Records

With `--step-record`, the driver writes JSONL. The first line is one `meta`
object, and each step appends one `step` object. Every time value is a float
number of microseconds. The file is opened before the loop, and the `meta`
line is flushed before step 0. Each later append is flushed outside the
`latency_us` window. An early exit leaves a valid JSONL prefix.

`meta` fields:

| Field | Value |
| ----- | ----- |
| `kind` | `"meta"` |
| `schema` | `"phase2-v1"` |
| `case` | case name |
| `platform` | platform passed on the command line |
| `device` | device id |
| `seed` | fixture seed |
| `start_seq_len` | `seq_len` of step 0 |
| `steps` | step count, warmup included |
| `warmup_steps` | warmup count |
| `batch` | `16` |
| `hidden` | `5120` |
| `n_layers` | `40` |
| `latency_window` | `"before first copy_to .. after copy_from(out) returns"` |
| `out_dump` | file name of the bf16 dump, when a dump is written |
| `out_dtype` | `"bfloat16"` |
| `out_shape` | `[16, 5120]` |
| `out_bytes_per_step` | `163840` |

`step` fields:

| Field | Value |
| ----- | ----- |
| `kind` | `"step"` |
| `step` | index from 0, warmup included |
| `seq_len` | `start_seq_len + step`, in `[1, 5500]` |
| `warmup` | true when `step < warmup_steps` |
| `latency_us` | primary host wall clock |
| `copy_in_us` | the three `copy_to` calls, inside the window |
| `run_us` | `worker.run`, inside the window |
| `copy_out_us` | `copy_from(out)`, inside the window |
| `segments_us` | `copy_in_us + run_us + copy_out_us` |
| `glue_us` | `latency_us - segments_us` |
| `gen_us` | input generation, outside the window |
| `checksum_us` | checksum and digest, outside the window |
| `io_us` | JSONL and `out.bin` append plus flush, outside the window |
| `sum_f64` | float64 sum of `out` |
| `max_abs` | float64 maximum absolute value of `out` |
| `digest` | blake2b over the raw bf16 bytes, `digest_size=8`, 16 hex digits |
| `finite` | true when `torch.isfinite(out).all()` holds |
| `out_offset` | byte offset of this step in the dump, equal to `step * 163840` |

`digest` is the byte-for-byte check. `sum_f64` and `max_abs` describe
magnitude. The bytes hashed into `digest` are the bytes appended to the dump.

`--save-out` appends that raw bf16 image to one binary file. Each step is
`163840` bytes (`16 * 5120 * 2`). Step `k` begins at offset `k * 163840`.
The number of complete steps in the file is its length divided by `163840`,
and that count matches the number of flushed `step` lines. The dump is
appended and flushed on the same schedule as the JSONL, so a partial file is
a valid prefix of complete steps. `load_saved_out(path, step)` in
`simpler_setup.goldens.qwen3_14b_decode_steps` reads step `k` back as a bf16
tensor.

After the last step the driver prints a steady-state summary on stdout.
Warmup steps are left out of the percentiles, the buckets, and the slope.
Percentiles are nearest-rank (`sorted[ceil(q * n) - 1]`). The steady steps
are split into 10 buckets of width `max(1, steady // 10)`, where `steady` is
`steps - warmup_steps`. The slope is a least-squares fit of `latency_us` on
those steady steps, in microseconds per step.

```text
[qwen][steps] n=500 warmup=3 steady=497 seq=3500..3999 record=<path> outdump=<path>
[qwen][pctl] metric=latency_us p50=... p90=... p99=... min=... max=...
[qwen][pctl] metric=run_us p50=...
[qwen][pctl] metric=copy_in_us|copy_out_us|segments_us|glue_us|gen_us|checksum_us|io_us ...
[qwen][bucket] steps=3-52 seq=3503-3552 latency_us_p50=... run_us_p50=...
[qwen][slope] metric=latency_us us_per_step=0.0123
[qwen][digest] first=<hex> last=<hex> nonfinite_steps=0 out_bytes=81920000
```

## Historical cost

Before the device-resident migration, the measured breakdown below came from
the a2a3 source on one device:

| Phase | Wall |
| ----- | ---- |
| kernel compilation — 36 incores + 1 orchestration | **59 s** |
| `generate_inputs` (the 38 GiB fixture) | **13 s** |
| `compute_golden` (40 layers, torch, thread-capped) | **~43 s** |
| host→device upload, device run, comparison | the remainder; the device itself is busy for tens of ms |
| | **~115 s** |

Two details behind those numbers. The 38 GiB is mostly replication —
`torch.cat([w] * 40)` of one layer's weights — so only ~2 GB is actually
generated; and the vendor FAI kernel is the compile outlier at 8.4 s for its AIV
variant against ~1.2 s for an ordinary incore.

The golden is thread-capped by the scene-test framework, and that matters more
than its size: its 3584 small slice operations per layer took **6.35 s per layer
at 320 torch threads against 1.05 s at 4**, so on a many-core host the untamed
thread pool cost 5-8x the work it was doing. Before the cap this case held a
device for **406 s** (ci run 30507320146, job 90761014912) and had to be kept out
of the sweep with a `--ignore` and a step of its own.

## Relationship to the a2a3 example

The kernels are the a2a3 harvest with the arch deltas below. The math — tiling,
loop structure, GM offsets, the KV write's BSND row arithmetic — is identical,
so a refresh is still "re-harvest a2a3, then re-apply this table".

| Delta | Where | Why a5 needs it |
| ----- | ----- | --------------- |
| Left(A) tile `BLayout::RowMajor` → `ColMajor` (×288) | AIC kernels | a5 cube takes L0A in the ColMajor fractal: pto-isa's own `TileLeft` alias is `BLayout::ColMajor` off a2a3, and `CheckMadValid` asserts `!TileLeft::isRowMajor`. SFractal stays `RowMajor`, and the L1 `Mat` tiles the `TEXTRACT` reads from are unchanged |
| drop `pipe_barrier(PIPE_V)` (×135), `pipe_barrier(PIPE_MTE1)` (×32), `pipe_barrier(PIPE_FIX)` (×3) | AIV + AIC kernels | a5 `pipe_barrier` accepts only `MTE3` / `M` / `ALL`; anything else is a compile error. Nothing is lost: a5's own `pto::Event` emits no barrier for a same-pipe V event either, and every cross-pipe order here is already carried by a `set_flag` / `wait_flag` pair |
| `__DAV_C220_{VEC,CUBE}__` → `__DAV_{VEC,CUBE}__` (3 files) | attention extern | `__DAV_C220_*` is the a2a3 (`dav-c220`) ccec predefine. a5 builds as `dav-c310`, so on a5 those guards silently compiled out the whole RoPE/QK-norm/KV-write phase **and** the vendor FAI's cube and vector bodies — the task ran to completion writing nothing. `__DAV_VEC__` / `__DAV_CUBE__` are predefined on both arches and are what every other kernel here already uses |
| `#include <pto/pto-inst.hpp>` moved after the vendor header | `kernel/fai_body.hpp` | `pto` and `KernelCommon` both declare `NUM_32` / `NUM_128` / `NUM_256` at namespace scope and `tensor.h` opens `namespace pto`, so including pto first makes those names ambiguous inside the vendor kernel |
| hard `AscendC::SyncAll<false>` → `pto::SYNCALL<Soft, Mix>` | `kernel/fai_body.hpp`, `kernel/metadata_layout.h`, `tiling/entry.cpp` | the hard barrier is the FFTS flag region, whose base a2a3 gets from `rtGetC2cCtrlAddr`; that call is unsupported on a5 and simpler's a5 runtime sets no FFTS base, so the barrier never releases. The soft barrier's counter (`kSyncAllCounterWord`, one int32 past the kernel's own 48 slots) is zeroed by `clear_barrier` alongside them — it is a monotonic ticket counter, so a non-zero start splits the first round's arrivals across a participant boundary and hangs the half that lands above it |
| `platforms: ["a5"]`, `ring_dep_pool=65536` | test `CASES` | the 40-layer graph's in-flight dependency footprint (~32 k) overflows a5's default 16 k pool |
| Refresh inverse-RMS input lines before the generated RoPE body | `kernel/fai_body.hpp` | The generated body reads the 16-float inverse-RMS input through Scalar. Single-line DCCI followed by DSB makes each invocation observe the preceding RMS task's values without depending on an executor-wide cache flush. The dependency chain excludes concurrent writers during these reads; HBG Graph replay isolates and reuses this scratch range across layers. |

The inverse-RMS refresh covers every cache line touched by the 16-float input,
including a range that crosses a cache-line boundary. Its dependency chain
excludes concurrent writes during the Scalar reads. The standalone test does
not exercise HBG Graph resident replay; the historical resident-replay A5
validation is recorded in PR #2821, not automated coverage of the current
mainline runtime.

The vendor `FusedInferAttentionScore` tree is otherwise untouched, including its
`Arch::AtlasA2` budget (UB 192 KiB, L1 512 KiB). a5 has L1 512 KiB and UB 248 KiB,
so that budget is valid there — conservative on UB, not wrong.

One arch-specific knob carries over unchanged: `attention_core_num` in
`kernels/orchestration/decode_fwd_layers.cpp` (the AIC `block_num` the fused
attention task spans). The a2a3 value is **24** = the a2a3 per-device AIC count.
On a5 a device exposes **36 AIC** (2 dies × 18), so 24 is still a valid
`block_num` — it runs, just leaving 12 cores idle. Raising it toward 36 for full
utilization is a tuning follow-up, not a correctness requirement. It must stay
**≥ 16**: the RoPE phase runs on `block_idx * 2 + sub_block_idx` lanes and needs
all `kQwenRopeCores == 32` of them.
