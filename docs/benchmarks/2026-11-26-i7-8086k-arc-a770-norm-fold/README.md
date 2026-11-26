# Arc A770: the norm+fold and the embedding row, through `kcpu.c`'s arithmetic

Host: i7-8086K, Windows 10, Arc A770 16 GB (driver 32.0.101.9175, OpenCL C 1.2, no
`cl_khr_fp64`). Date: 2026-11-26. Pack: the 2B (`bonsai-2b`, 320 tensors, Hadamard block 512) — its
sign tables and norm weights are real, its `token_embd.weight` is a real 248,320x2,048 ternary
matrix. Probe: `python -m kindi intel-norm --pack ~/.kindi/packs/bonsai-2b --device-name Arc`.

This is K1 and K2 of [INTEL-ARC-KERNELS.md](../../INTEL-ARC-KERNELS.md), which ranked them first: K1
`norm_fold` is called 193 times a token and every matvec depends on it; K2 `embed_row` was budgeted
half a day. Both now exist as OpenCL C, checked against `kcpu.c` rather than against themselves.

## Method

Two kernels, both transcribed from the CPU engine rather than designed from scratch:

* **`rms_sum`** — the residual add into `h`, the sum of squares, `1/sqrtf((float)(acc/D) + 1e-6f)`.
  `kcpu.c` sums in *double* and this driver has no fp64, so the sum is carried as a two-float pair:
  Fast2Sum remainders plus each square's `fma` product error, with the division arranged to round
  once as the CPU's `(float)(acc/D)` does.
* **`fold_sign`** — `xn = (h*s)*w` on the unfolded vector, then `xp = FWHT_blocks(sg*xn)` in
  `fwht_block`'s stage order with `1/sqrtf(HB)` last. One work-group per Hadamard block, one lane per
  pair of the stage. With `add=0`, `w=1` and `s=1` it is also the embedding's fold.

The reference is `kcpu.c`'s `rmsfold()`/`fold()` transcribed to numpy with the same operations in the
same order. Timing is a bound-argument loop of `repeats` calls with one `clFinish` at the end, plus a
no-op kernel for the launch floor.

## Results

| call | elements | us/call (2 enqueues) | calls/token | ms/token |
|---|---:|---:|---:|---:|
| rmsfold + fold | 2,048 | 43.09 | — | — |
| rmsfold + fold | 5,120 | 45.99 | 65 | 2.99 |
| rmsfold + fold | 6,144 | 45.50 | 64 | 2.91 |
| rmsfold + fold | 17,408 | 54.33 | 64 | 3.48 |
| **the 27B's 193 passes** | | | **193** | **9.38 in 386 enqueues** |

The 5,120 and 17,408 rows use synthetic weights: the 27B pack is not on this host. The 2,048 row is
the 2B's own `attn_norm.weight` and sign table.

**The launch floor is most of that.** A no-op kernel costs **10.99 us per enqueue** and one launch
from idle costs **317.30 us**, so 386 enqueues are ~4.2 ms of the 9.38 ms before anything is
computed. K1 as two kernels per call is unaffordable at 193 calls; it has to be fused into the
layer's launches, which is what Phase D assumed and what this number now prices.

## Exactness

| width | sigma bit-equal to the CPU | worst relative | xp differing | worst abs |
|---:|---|---:|---:|---:|
| 2,048 | 8 of 12 random vectors | 8.51e-08 | 1,555 | 2.86e-06 |
| 6,144 | 10 of 12 | 8.43e-08 | 4,763 | 7.15e-07 |

The folded vector moves by at most one ulp when the scale's last bit differs, which is inside the
plan's ~1e-5 bar for this chain — but **"bit-exact" is not available on this driver**, and the plan
should say *verify passes*, not *exact*, for the Arc path.

## The butterfly's stage walk

The first version walked each stage as blocks (`i += 2*step`), which leaves the last stage's 256 pairs
on a single lane; those dependent local-memory reads dominated the call. Walking the stage flat
(`m` enumerates every pair, one per lane) keeps the CPU's pairs, the CPU's order within a pair, and
cuts the call by more than half. Barriers were never the cost: 1, 5 and 9 stages differ by about the
price of one enqueue.

## The embedding row

`token_embd.weight` is one row of the pack: **544 B** for 2,048 columns (16 phases x 32 bytes of
base-3 trits, plus one fp16 scale per phase). The decode is host work: 237.10 us through numpy and a
`memmap` here, which is a Python artifact, not a floor — the engine does the same read in C. The fold
is the same kernel with unit weight and unit scale: **27.99 us** per call and **0 differing** elements
against `kcpu.c`'s embed. **K2 needs no kernel of its own.**

## What this changes in the plan

1. **K1 is built, but its launch shape has to change.** Two enqueues per call cannot be the shipped
   form; the fold belongs in the layer's own launch, next to the matvec that consumes `xp`.
2. **K2 is closed** — the decode stays on the host, the fold is K1's second kernel.
3. **The Arc path's exactness bar is `verify`, not bit-equality** (`docs/INTEL-ARC.md` now says so).
4. Next in Phase A is the wiring: a layer loop over the resident pack with the host doing conv, the
   GDN update, attention and the sampling.

Not a model run: no layer loop, no engine, no Intel backend, no support claim. The harness is
`kindi/intel_layer.py` (the probe, tracked with its CLI entry) and the artifact beside this page is
`norm.json`, written by the command above.
