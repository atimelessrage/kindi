# Arc A770: four rows per work-item in the packed ternary GEMV

Host: i7-8086K, Windows 10, Arc A770 16 GB (512 compute units, 16 GB, driver 32.0.101.9175,
OpenCL C 1.2). Date: 2026-11-25. Pack layout: the on-disk 4-trit/byte `[slot][row]` matrix with one
fp16 scale per 128 weights. The probe checks every output element against a host reference bit for
bit; an arm with one mismatch aborts. No model, no backend, no support claim.

## Why

The 2026-11-24 round split the GEMV over the 128-weight phase axis (5.75x-14.14x the one-work-item
shape). What was left inside a work-item was 1,024 single-byte loads with a 2,048-byte stride, four
trits decoded per byte with `% 3` / `/ 3`, and a second kernel to add the parts.

## What changed

`kindi/intel_ternary.py` gains a second GEMV kernel, `ternary_gemv_part4`. The layout stores a slot's
bytes as consecutive **rows**, so `vload4` on one slot reads four rows at once: four single-byte
loads become one 4-byte load, with the same per-phase int32 accumulation and the same scaled-bytes
order, so the partials are bit-identical to the scalar kernel. The decoded trits use
`(b * 171) >> 9` and its successors instead of integer division, and a row count that is not a
multiple of four takes a guarded path that never reads past the buffer. The probe selects it through
`--kernel auto|scalar|rows4`; `auto` (the default) uses `rows4` whenever four rows fit, because both
kernels pass the same exactness check and the choice is performance only.

## Results

`bench.py`, three runs of 20 launches per arm, medians, on the installed A770. `per call` is one
GEMV (the part kernel plus the reduce kernel); effective bandwidth counts weights, scales, `x` and
`y`.

| shape (rows x cols) | kernel | per call | effective GB/s | GFLOP/s |
|---|---|---:|---:|---:|
| 1024 x 2048 | scalar | 0.0461 ms | 12.36 | 91.05 |
| 1024 x 2048 | **rows4** | **0.0433 ms** | **13.13** | **96.74** |
| 2048 x 4096 | scalar | 0.0571 ms | 39.49 | 294.18 |
| 2048 x 4096 | **rows4** | **0.0467 ms** | **48.29** | **359.68** |
| 4096 x 8192 | scalar | 0.0936 ms | 95.75 | 716.78 |
| 4096 x 8192 | **rows4** | **0.0690 ms** | **129.88** | **972.80** |
| 4096 x 128 (launch floor) | scalar | 0.0432 ms | 3.62 | 24.29 |
| 4096 x 128 (launch floor) | **rows4** | **0.0410 ms** | **3.80** | **25.54** |

Speedups: **1.065x** at 1024x2048, **1.223x** at 2048x4096, **1.357x** at 4096x8192 and 1.054x at the
launch-floor shape. Every arm reports 0 mismatches against the host reference.

The 4096x128 shape moves 128 KB of weights and still costs 0.041 ms per call: that is the price of
two enqueued kernels (the part kernel and the reduce) on this driver, ~40 microseconds with almost no
data. Below roughly 2 MiB of packed weights a GEMV is therefore overhead-bound, not bandwidth-bound -
which is what the small-shape rows show, and why the largest shape gains the most.

## What this shows

- On the installed A770, four rows per work-item with 4-byte vector loads is faster than one row per
  work-item at every shape measured: 1.36x at a layer-sized 4096x8192 matrix, 1.22x at 2048x4096.
- The kernel stays bit-exact: the same host check that guards the scalar kernel passes on every arm.
- The per-call floor of two OpenCL launches is about 40 microseconds on this driver, so a future
  backend's GEMV latency at small shapes is dispatch, not arithmetic.

## What this does not show

- No Kindi backend, engine path, model or `kindi verify` run: this is still the probe's synthetic
  matrix with unquantized activations. M2 stays partial and there is no Intel support claim.
- Windows only. The first support target is Linux with SYCL/oneAPI, and the oneAPI install on this
  host is currently incomplete (only its installer directory remains), so this round used the
  display driver's OpenCL C runtime.
- 129.88 GB/s at 4096x8192 is about a quarter of the card's memory bandwidth: the launch floor above
  dominates the rest of the gap at that size, and a single-kernel design would have to give up the
  coalesced row-major reads or the exact float sum order to remove it. Not attempted here.
- Three runs of 20 launches, greedy nothing, one context per probe: this is a kernel A/B, not a
  steady-state throughput claim.

## Reproduce

```sh
python docs/benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-rows4/bench.py --out bench.json --runs 3 --repeats 20
python -m kindi intel-ternary --rows 4096 --cols 8192 --kernel rows4 --repeats 20
python -m kindi intel-ternary --rows 4096 --cols 8192 --kernel scalar --repeats 20
```

`bench.json` holds every run's per-call time, effective bandwidth, GFLOP/s and mismatch count, plus
the medians and speedups above.
