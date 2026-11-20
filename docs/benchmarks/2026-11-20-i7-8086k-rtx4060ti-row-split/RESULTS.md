# Row-split ternary GEMM - 9B proxy (2026-11-20)

## What was tested

`tools/row_split_gemm.py` is an opt-in probe for one ternary `ffn_down` matrix. The CPU owns the
lower rows and the GPU the upper rows. The CPU uses the exported `kc_gemv`; the GPU engine loads only
its row range with `load_partial_tensor` and runs its normal `gemv`. The probe has three levels:

- `gpu-layer`: a GPU-owned layer with its `ffn_down` split; the CPU slice and the GPU slice run
  sequentially, then merge in the GPU residual buffer.
- `cpu-matrix`: the same split on a standalone CPU-owned GEMV, compared with the full CPU GEMV.
- `cpu-layer`: a CPU-owned layer; attention/GDN and gate/up stay in the CPU engine, then the down
  GEMV is split and merged into the CPU residual stream.

Host: i7-8086K, RTX 4060 Ti 8 GB, Windows native CPU build (clang 18.1.8, `avx2-fma-f16c`,
6 threads). Pack: `bonsai-9b`. Tensor: `blk.0.ffn_down.weight` (4,096 rows x 12,288 columns;
layer 3 is the control). The CPU/GPU matrix and layer rows are 30 paired iterations after four
warmups: full and split alternate, so a thermal drift applies to both. The GPU-layer rows are the
earlier sequential probe (12-15 iterations after warmups), which is sufficient for the direction but
less balanced.

## Verdict

**No-go for the roadmap's one-matrix gate.** A CPU-owned matrix does get faster: the best split
(25% CPU / 75% GPU) is **1.52x** the full CPU GEMV, and 50/50 is **1.18x**. That gain does not survive
the layer: the best CPU-owned layer is **1.01x** (25/75) and the 50/50 layers are **0.97x**, because
the split adds a host/device round trip and a second GEMV launch to every layer, while `ffn_down` is
only about 0.77 ms of a 3.3-3.7 ms CPU layer.

The GPU-owned layer is worse in every split tested (**0.43x-0.64x**): the CPU slice is much slower
than the GPU rows it replaces and the sequential merge adds latency. This is not a concurrency
result; no second CUDA stream was used.

The roadmap row also asks for the 27B at 24K and a full decode. Neither was measured: there is no
27B pack on this host, and no `TandemEngine` integration was built because the layer-level result
already misses the 1.10x gate. No `kindi verify` 4/4 claim is made.

## Results

`full_ms` is the unsplit owner; `split_ms` is the row split; `ratio` is `full / split`, so values
below 1 mean the split is slower.

| mode | layer | CPU rows | GPU rows | full ms p50 | split ms p50 | ratio | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpu-layer | 0 | 1,024 (25%) | 3,072 | 0.757 | 1.184 | 0.64 | 2.38e-07 |
| gpu-layer | 0 | 2,048 (50%) | 2,048 | 0.730 | 1.697 | 0.43 | 2.38e-07 |
| gpu-layer | 0 | 3,072 (75%) | 1,024 | 0.739 | 1.669 | 0.44 | 2.98e-07 |
| gpu-layer | 3 | 2,048 (50%) | 2,048 | 0.813 | 1.562 | 0.52 | 5.96e-07 |
| cpu-matrix | 0 | 1,024 (25%) | 3,072 | 0.774 | 0.508 | **1.52** | 9.54e-07 |
| cpu-matrix | 0 | 2,048 (50%) | 2,048 | 0.797 | 0.673 | **1.18** | 9.54e-07 |
| cpu-matrix | 0 | 3,072 (75%) | 1,024 | 0.920 | 1.025 | 0.90 | 9.54e-07 |
| cpu-layer | 0 | 1,024 (25%) | 3,072 | 3.730 | 3.698 | 1.01 | 1.91e-06 |
| cpu-layer | 0 | 2,048 (50%) | 2,048 | 3.339 | 3.458 | 0.97 | 1.91e-06 |
| cpu-layer | 3 | 2,048 (50%) | 2,048 | 3.528 | 3.626 | 0.97 | 2.68e-07 |

The matrix-level 1.52x is real for the isolated GEMV, but it is not a layer or token result. The
CPU-owned layer at 25/75 spends the matrix gain on the transfer and orchestration; at 50/50 it is
slightly slower than the full CPU layer.

## Memory

`blk.0.ffn_down.weight` is about 13.4 MB in the ternary pack (4,096 x 12,288 plus scales), so one
half is about 6.7 MB. A CPU-owned split keeps the full matrix in the CPU engine and adds the GPU
half, which raises the GPU peak rather than lowering total memory. A split that loaded only each
side's rows could move the same 6.7 MB between host and device, but that is a redistribution, not a
saving, and it was not measured as a peak here. The GPU-owned layer also retains the CPU copy for
its slice, so there is no total-memory win in either direction.

## Why it misses the gate

The roadmap gate is about one large FFN matrix at 50/50 with p50 decode at least 1.10x. On this host
one `ffn_down` is about 21% of a CPU layer (0.77 ms of 3.3-3.7 ms). Even a free split of that matrix
could not take a 3.3 ms layer below 2.6 ms, and the measured split does not approach that because
the CPU/GPU merge costs more than the saved GPU work at layer granularity. The GPU-owned direction
is strictly worse. A future attempt would need a larger share of the layer (for example gate/up too)
and an asynchronous result, not just one matrix.

## Not verified

- 27B at 24K (`gdn:12`): no pack on this host.
- Full tandem decode: no `TandemEngine` row-split integration; no p50 tok/s claim.
- Token exactness: no `kindi verify` 4/4 run; only the layer-output max abs difference above.
- Transfer/overhead as a percentage of decode: not measured separately.
- Peak VRAM of the split: not sampled; the memory section is byte accounting, not a measurement.
- A concurrent two-stream implementation: this is the sequential merge described above.

## Evidence

- `gpu-layer-25-L0.{log,json}`, `gpu-layer-50-L0.{log,json}`, `gpu-layer-75-L0.{log,json}`,
  `gpu-layer-50-L3.{log,json}`
- `cpu-matrix-25.{log,json}`, `cpu-matrix-50.{log,json}`, `cpu-matrix-75.{log,json}`
- `cpu-layer-25.{log,json}`, `cpu-layer-50-L0.{log,json}`, `cpu-layer-50-L3.{log,json}`

Raw files are under `bench-logs/row-split-20261120/` (gitignored local evidence, so it is not
linked from this page).
