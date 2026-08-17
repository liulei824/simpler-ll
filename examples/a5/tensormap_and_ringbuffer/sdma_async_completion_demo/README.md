# sdma_async_completion_demo — deferred completion over SDMA

Two ranks, one transfer, one dependency:

```text
producer:  TGET_ASYNC the peer rank's input from the HCCL window into local
           `out`, then register the PTO AsyncEvent via defer_pto_async_event
consumer:  depends on the producer's output, writes result = out + 1
```

Checking `out` and `result` tests two separate things: `out` proves SDMA
completion polling saw the transfer land, `result` proves the deferred-release
dependency held the consumer until it had. A consumer that ran early would
still produce a plausible `result` from a partially written `out`, which is why
both are checked.

The remote address is plain symmetric-window arithmetic — take the local
pointer's offset from `windowsIn[rankId]` and add it to `windowsIn[peer_rank]`.
Every rank's window is laid out identically, so an offset is rank-independent.

The transfer kernel reads the SDMA workspace from the domain trailer via
`get_comm_dma_workspace(comm_ctx, DMA_WORKSPACE_SDMA)`. The domain must be
allocated with `engines=("sdma",)`.

## Requirements

The a5 host runtime builds with SDMA and URMA overlays both available by
default. This demo only needs the SDMA slot filled for its domain.

| Gate | Effect |
| ---- | ------ |
| `@pytest.mark.platforms(["a5"])` | deselected on any other `--platform` |
| `@pytest.mark.device_count(2)` | needs two dies |

```bash
pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo \
  --platform a5 --device 0-1
```

Wrap the hardware run in `task-submit` on a shared box.

## Compare with

- [`../urma_deferred_completion_demo/`](../urma_deferred_completion_demo/) — the same protocol over URMA. `kernel_consumer.cpp` is byte-identical; only the transfer kernel, completion header, and `engines=("urma",)` declaration differ. Both demos can pass against the same host runtime build.
- [`examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/`](../../../a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/) — the a2a3 port of this demo.
