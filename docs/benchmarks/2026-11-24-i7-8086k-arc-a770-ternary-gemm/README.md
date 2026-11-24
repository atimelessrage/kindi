# Intel Arc A770: ternary GEMM in the engine's q8 shape (2026-11-24)

`kindi intel-gemm` is the follow-up to the phase-parallel GEMV probe. It runs the arithmetic the
engine's prompt path actually does, at the tile the CUDA kernels use, and holds it to the same
bit-exact discipline:

* `q8x` ports `k_q8x_T2` (`kindi/chunk.cu`): one 256-wide group per 128-column group and 64-token
  band, one threshold pair per token, `rint` round-half-even, four int8 per packed word. Its XQ/XS
  output feeds the GEMM directly, so a mismatch anywhere in the chain fails the run.
* `gemm_tile` mirrors `k_mm_g5q`'s arithmetic: the block covers 128 tokens x 128 rows with 256
  work-items (8 x 8 outputs each); per 128-column group the dot products accumulate in int32 and
  are scaled once, `f * (xs * dj)`, before an fp32 add, in group order, with the group's scale read
  per row. Two deliberate differences, both load-time choices rather than arithmetic: the weight
  bytes are the pack's on-disk 4-trit layout (the engine repacks to 5-trit lanes at load; the
  decode here is the 4-trit one), and there is no local-memory staging - every load comes from
  global memory, so this is the un-tiled shape to measure before tiled variants, not the fast one.

## What ran

`python -m kindi intel-gemm --rows R --tokens T --cols K --repeats 10 --json` on the Windows A770
host (i7-8086K), Intel OpenCL C 1.2 driver 32.0.101.9175. One run per shape; the JSON is in this
directory.

| rows x tokens x cols | groups | quantizer | mismatches | 10 launches | GFLOP/s |
|---|---|---|---|---|---|
| 128 x 128 x 512 | 4 | XQ 0, XS 0 | 0 | 40.204 ms | 4.173 |
| 256 x 128 x 1024 | 8 | XQ 0, XS 0 | 0 | 85.832 ms | 7.819 |
| 256 x 256 x 2048 | 16 | XQ 0, XS 0 | 0 | 283.782 ms | 9.459 |
| 512 x 512 x 4096 | 32 | XQ 0, XS 0 | 0 | 581.610 ms | 36.923 |

Reports: [128x128x512](gemm-128x128x512.json), [256x128x1024](gemm-256x128x1024.json),
[256x256x2048](gemm-256x256x2048.json), [512x512x4096](gemm-512x512x4096.json).

## Why the result is exact

The activations are small integers with a per-group maximum of exactly 127 (forced at each group's
first two columns), so the quantizer computes `xs = 1.0`, `id = 1.0` and passes the values through;
the scales are fp16-exact (1.0/0.5/2.0/1.5). Every integer stays below 2^24 and every product is
exactly representable, so the device and the host reference must agree bit for bit - and the host
reference itself is checked against an independent float64 dot before the device is launched (the
run refuses to launch if the fp32 stepping disagrees with it). OpenCL C 1.2 has no dot-product
instruction, so the int32 dot is computed byte by byte on the device; that is exactly the integer
`dp4a` computes, not a different arithmetic.

## What the numbers mean

The rate rises with the tile count because a launch is about 4 ms of fixed overhead at these sizes
(entirely in the quantizer's local-memory barriers at small shapes: 3.766 ms for the 128-token
tile). At 512 x 512 x 4096 the GEMM reaches 36.9 GFLOP/s with global loads only and a byte-wise
dot. Two follow-ups are named so they are not mistaken for measurements: local-memory staging
(the CUDA kernels' A/B tiles, worth measuring on the same shapes) and the engine's 5-trit lane
unpack instead of this 4-trit decode.

## Scope and limits

- Still a probe: no model load, no engine path, no SYCL, no support claim. Windows and OpenCL are
  precursor work; the plan's target is SYCL/Level Zero on Linux. M2 is still partial and Arc is
  still unsupported.
- The exactness construction (amax = 127) exercises the quantizer as a pass-through; a general
  activation matrix (arbitrary threshold, `id != 1`) is proven by the engine's own verify on GPU
  hosts, not by this probe.
- The weight decode is the pack layout's 4 trits/byte; the engine's load-time repack to 5-trit
  lanes is a different unpack stage and a separate probe.
