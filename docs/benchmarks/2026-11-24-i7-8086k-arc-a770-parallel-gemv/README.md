# Intel Arc A770: phase-parallel packed ternary GEMV (2026-11-24)

The 2026-11-23 packed ternary GEMV probe ran **one work-item per row**, which leaves most of the card
idle: 512 rows is 512 work-items, while the A770 has 512 EUs. The packed layout is `[slot][row]`, so
the only coalesced access has consecutive work-items on consecutive rows and the spare parallelism
has to come from the 128-weight phase axis. This page records the same probe split over the phases
(`--parts`): the kernel takes the phase `ph` whose `ph % parts == part`, writes one exact partial per
(row, part), and a second kernel adds the parts.

No change to the data: the same `bcodes`/`dpm` bytes are uploaded, the same host reference checks every
row, and a mismatch is still a failure.

## What ran

`python -m kindi intel-ternary --rows R --cols C --repeats 20 --json` on the Windows A770 host
(i7-8086K), Intel OpenCL 3.0 NEO driver 32.0.101.9175. Three consecutive runs per shape; the table
quotes the **median** run and its JSON is the file in this directory. `--parts` was left at its
default, which is one part per 128-weight phase capped at 64.

## Result

| rows x cols | parts | mismatches (3 runs) | 20 launches (median) | GFLOP/s | `--parts 1` (median) | speedup |
|---|---|---|---|---|---|---|
| 64 x 1024 | 8 | 0, 0, 0 | 1.054 ms | 2.49 | 2.825 ms | 2.68x |
| 128 x 4096 | 32 | 0, 0, 0 | 1.081 ms | 19.40 | 8.870 ms | 8.21x |
| 512 x 4096 | 32 | 0, 0, 0 | 1.111 ms | 75.54 | 8.880 ms | 7.99x |
| 1024 x 8192 | 64 | 0, 0, 0 | 1.224 ms | 274.12 | 17.306 ms | 14.14x |
| 2048 x 4096 | 32 | 0, 0, 0 | 1.152 ms | 291.15 | 9.001 ms | 7.81x |
| 4096 x 4096 | 32 | 0, 0, 0 | 1.577 ms | 425.60 | 9.064 ms | 5.75x |

Reports: [64x1024](ternary-gemv-64x1024.json), [128x4096](ternary-gemv-128x4096.json),
[512x4096](ternary-gemv-512x4096.json), [1024x8192](ternary-gemv-1024x8192.json),
[2048x4096](ternary-gemv-2048x4096.json), [4096x4096](ternary-gemv-4096x4096.json); all six runs per
shape, both splits, in [summary.json](summary.json).

The rate is 2 * rows * cols * launches over the wall time of the launches plus `clFinish`. It is an
observation from a probe kernel, not a throughput claim: nothing here is a model, an engine or a
support row.

## Why the split cannot change the answer

Each 128-weight group is accumulated in int32 and scaled once, exactly as the unsplit kernel did; a
part therefore holds a sum of at most 64 products of exactly representable values, and the reduction
adds at most 64 such floats. Every intermediate value is an integer below 2^24, so both steps are
exact and the order of summation is irrelevant. The probe does not rely on that reasoning alone:
`kindi.intel_ternary.partials` computes the same decomposition on the host, and the run **refuses to
launch** if summing its parts differs from the reference for the requested shape. The device is then
still compared bit for bit (0 mismatches at every shape above).

## Scope and limits

- Still the on-disk pack layout, not the engine's load-time 5-trit lane repack (`kindi/layout.py`), and
  the activation vector is still not quantized: this is layout decoding and exactness, not the engine's
  Q8 GEMV.
- A launch costs about 1 ms of fixed overhead at these sizes, which is why 64 x 1024 gains only 2.7x:
  at that shape the kernel runs for tens of microseconds.
- No SYCL/oneAPI, no `libksycl`, no Kindi backend, engine or model load, no support-matrix change. M2 is
  still partial and Arc is still unsupported.
