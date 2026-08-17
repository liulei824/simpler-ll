# urma_deferred_completion_demo — the same protocol, over URMA

The a5-only twin of
[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/). Both run the
identical two-rank protocol; they differ in which transport moves the bytes and
which completion path reports it done.

```text
producer:  TGET_ASYNC the peer's input from the window into local `out`,
           register the AsyncEvent through the deferred-completion path
consumer:  depends on that producer output, writes result = out + 1
```

Checking both `out` and `result` is what makes it a test of two things at once:
`out` proves completion polling saw the transfer land, and `result` proves the
deferred-release dependency held the consumer back until it had.

The transfer kernel reads the URMA workspace from the domain trailer via
`get_comm_dma_workspace(comm_ctx, DMA_WORKSPACE_URMA)`. The domain must be
allocated with `engines=("urma",)` (and ranks must form a dense HCCL prefix).

## What actually differs from the SDMA demo

`kernels/aiv/kernel_consumer.cpp` is **byte-identical** between the two
directories. Only the transfer kernel, orchestration, and engine declaration
change:

| What | URMA | SDMA |
| ---- | ---- | ---- |
| Transfer kernel | `kernel_urma_tget_async.cpp` | `kernel_sdma_tget_async.cpp` |
| Completion header | `backend/urma/urma_completion_kernel.h` | `backend/sdma/sdma_completion_kernel.h` |
| Domain engines | `engines=("urma",)` | `engines=("sdma",)` |

Read them side by side and the transport is the only variable — which is
exactly what you want when deciding which one a workload should use.

## Both overlays can coexist in one build

A5 host CMake treats `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` and
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE` as independent options (both default ON).
Each engine's workspace lives in the per-domain `CommContextBlock` trailer;
`CommContext.workSpace` is left unused (0). Declaring `engines=` at
`allocate_domain` time selects which slots are filled.

A stock a5 build can therefore run both demos without rebuild. Kernels that
see a null trailer slot for their kind mean the domain was allocated without
that engine.

## Gates

| Gate | Effect |
| ---- | ------ |
| `@pytest.mark.platforms(["a5"])` | deselected on any other `--platform` |
| `@pytest.mark.device_count(2)` | needs two dies |
| `run()` raises | if `platform != "a5"` or the device count is not 2 |

## Run

```bash
pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo \
  --platform a5 --device 0-1
```

Wrap the hardware run in `task-submit` on a shared box.

## See also

[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/) — the
SDMA variant of the same protocol.
