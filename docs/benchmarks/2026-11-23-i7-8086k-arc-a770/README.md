# Intel Arc A770: packed ternary GEMV probe through OpenCL (2026-11-23)

A probe, not a backend: it runs one small GEMV kernel over weights in the **pack file's ternary
layout** and checks every output row bit for bit against a host reference that decodes the same
packed bytes. No activation quantization, SYCL, model, engine or support claim.

## What ran

`python -m kindi intel-ternary --rows R --cols C --json` on the Windows A770 host (i7-8086K), with
the Intel OpenCL 3.0 NEO runtime and driver 32.0.101.9175 from the existing `kindi intel-cl` path.

The matrix is built the way `kindi convert` writes it (`convert.phase_bytes`):

| tensor | layout | bytes |
|---|---|---|
| weights | 4 trits per byte, byte = t0 + 3*t1 + 9*t2 + 27*t3, stored `[column/4][row]` | R*C/4 |
| scales | one fp16 per 128 weights, stored `[column/128][row]`; widened to float32 on the host | R*C/128*2 |

The kernel takes those two buffers unchanged (plus a float32 activation vector and a float32
output), decodes the four trits of each byte, accumulates each 128-weight group in int32, multiplies
by the group's scale and adds the groups in phase order. Inputs are small integers and the scales
are 1.0/0.5/2.0/1.5, so every product and sum is exactly representable; the comparison requires
**0 mismatches**, not a tolerance. One work-item computes one row (local size 64), one warm-up
launch precedes five timed launches, and the host reference is computed from the packed bytes with
`kindi/intel_ternary.reference`.

## Result

| rows x cols | phases | mismatches | max abs error | warm-up | 5 launches | GFLOP/s (wall) |
|---|---|---|---|---|---|---|
| 64 x 1024 | 8 | **0** | 0.0 | 4.989 ms | 0.944 ms | 0.694 |
| 128 x 4096 | 32 | **0** | 0.0 | 4.637 ms | 2.470 ms | 2.123 |
| 512 x 4096 | 32 | **0** | 0.0 | 4.807 ms | 2.437 ms | 8.604 |

Reports: [64x1024](ternary-gemv-64x1024.json), [128x4096](ternary-gemv-128x4096.json),
[512x4096](ternary-gemv-512x4096.json). The rate is 2 * rows * cols * launches over the wall time
of the launches plus `clFinish`, recorded as an observation; it is not a performance claim and the
first-launch build/compile is excluded (the warm-up covers it).

## Scope and limits

- This is the on-disk pack layout, not the engine's load-time 5-trit lane repack (`kindi/layout.py`);
  the lane layout is a GPU-kernel optimization and stays out of this probe.
- The activation vector is not quantized and the kernel is one work-item per row: it tests layout
  decoding and exactness on the card, not throughput.
- No SYCL/oneAPI, no `libksycl`, no Kindi backend or engine, no model load, no support-matrix change.
  M2 is still partial and Arc is still unsupported.
