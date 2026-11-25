# Arc A770: where the packed ternary GEMV's time goes on the 2.5 27B shapes

Host: i7-8086K, Windows 10, Arc A770 16 GB (512 compute units, driver 32.0.101.9175, OpenCL C 1.2).
Date: 2026-11-25. Layout: the on-disk 4-trit/byte `[slot][row]` matrix with one fp16 scale per 128
weights. Harness: a throwaway probe (see Files) that checks every kernel variant against the host
reference of `kindi/intel_ternary.py` before timing it. No model, no backend, no support claim.

## Why

`kindi intel-pack` on the 2.5 27B packs reads 7.14 GB in 49.9 ms, which is 143 GB/s against a card
spec of 560 GB/s. Two explanations compete: the kernel is not moving enough bytes per unit of time
(bandwidth/occupancy), or it spends its time launching 401 small matrices one after another
(per-launch overhead). This page separates the two on the real shapes and prices a single-launch
kernel against more work per thread.

## Method

The matrix list is the model's own: every packed ternary tensor of
`Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` except the embedding lookup — **401 matrices, 6.800 GB of
codes+scales per token** (`rows*cols/4 + rows*cols/128*2` each). Activations are small integers and
scales are fp16-exact, so every arm is compared bit for bit with a host model of the packed bytes.

Axes:

* **rows per work-item (RPT)**: 1, 2, 4, 8, 16. The layout stores a slot's bytes as consecutive
  rows, so RPT rows of one slot are one `vloadN` (`ucharN`), with the same per-phase int32
  accumulation.
* **phase parts**: phase `ph` goes to part `ph % parts`, one exact partial per part per row, summed
  by a second kernel unless `parts == 1`.
* **enqueue pattern**: launch+reduce per matrix (the shipped shape), the same with a `clFinish`
  after every matrix, one launch per matrix at `parts=1`, and one persistent launch over a work
  list of all 401 matrices (binary search on a prefix array) plus one batched reduce.

Anchors measured in the same process: a pure `uint4` streaming read over the same 318 MB buffer
(the card's practical read bandwidth) and a 1-work-item noop kernel (per-launch cost).

## Results

Per-shape, existing per-matrix kernel (`rpt`, `parts`), best arm per shape:

| shape | packed | best | GB/s | parts=1 |
|---|---|---|---|---|
| 1024 x 5120 | 1.4 MB | rpt8 p40 | 12.7 | 1.9 |
| 4096 x 5120 | 5.6 MB | rpt2 p40 | 48.4 | 8.1 |
| 12288 x 5120 | 16.7 MB | rpt4 p40 | 103.2 | 22.5 |
| 17408 x 5120 | 23.7 MB | rpt2 p20 | 124.9 | 23.7 |
| 248320 x 5120 | 337.7 MB | rpt8 p4 | 263.2 | 131.9 |

The per-token figure is a *sequence* of small matrices: 400 of the 401 are at most 24 MB (the
largest of them 23.7 MB), and only the 338 MB output projection reaches streaming rates on its own.

The whole token, per-matrix enqueue (802 launches), best arms:

| arm | ms/token | GB/s | launches |
|---|---|---|---|
| rpt8 p40 | 47.38 | 143.5 | 802 |
| rpt2 p40 | 47.78 | 142.3 | 802 |
| rpt8 permat | 48.98 | 138.8 | 802 |
| rpt1 p40 | 59.47 | 114.3 | 802 |
| rpt1 p1 | 282.42 | 24.1 | 401 |
| `clFinish` per matrix | 99.08 | 68.6 | 802 |

The whole token, one persistent launch:

| arm | ms/token | GB/s | launches |
|---|---|---|---|
| parts=1, grid 1M | **26.87** | **253.1** | 1 |
| parts=1, grid 64K | 27.40 | 248.2 | 1 |
| rpt8 parts=8, grid 1M | 28.23 | 240.8 | 2 |
| rpt8 parts=20, grid 1M | 28.43 | 239.1 | 2 |
| rpt8 parts=40, grid 1M | 29.93 | 227.2 | 2 |

Work per thread with the card already full (persistent, parts=40, grid 1M): rpt1 46.81 ms, rpt2
36.67, rpt4 31.66, **rpt8 29.93, rpt16 29.84** — the gain runs out at eight rows per work-item.

Anchors: streaming read **384.8 GB/s** (69% of the 560 GB/s spec; 288 GB/s at 8K threads, 375-385
at 64K-1M); noop launch **8.78 us** when 802 are queued (9.47 us alone).

## Where the 47.38 ms goes

| component | ms | share |
|---|---|---|
| CPU-side enqueue: 802 x 8.78 us | 7.04 | 15% |
| reading 6.800 GB at the measured 384.8 GB/s | 17.67 | 37% |
| GPU-side ramp/drain and idle around 401 small launches | 22.67 | 48% |

So on this host the gap is **not mainly "the kernel is slow at reading"**. Launch cost that is
purely CPU-side is 15%; the other half of the loss is the card going idle because each launch
carries one small matrix and the queue drains between them. A `clFinish` after every matrix doubles
the token to 99 ms (65 us per matrix), which is the same effect at its worst.

One persistent launch removes most of it: **47.38 -> 26.87 ms (1.76x)** with `parts=1`, where there
is no reduce kernel and no partial scratch at all. That is 253 GB/s, i.e. 66% of the streaming read
and 45% of the spec. The remaining 9.2 ms over the 17.67 ms ideal is the kernel's own efficiency:
40 strided single-byte streams per work-item per phase and the `% 3`-style decode.

## Verdict

**Single-launch first.** It is where 48% of the current time is, it is a scheduling change rather
than a kernel rewrite, and it lands the token at 66% of the practical read bandwidth. Chasing
occupancy further would not have paid first: work per thread is already flat between 8 and 16 rows
per work-item, and `parts=1` with enough rows beats every `parts>1` arm once the work list is
persistent (fewer partials, no reduce). After that, the lever is the access pattern — how many
independent byte streams a work-item walks — not more threads.

Caveats: the host is not quiet (repeats of one config spread 26.2-30.4 ms across runs, see the JSON
ranges), so differences under ~10% are not conclusions; zero-filled weights read up to 2.4x faster
than random bytes at the same configuration (memory compression), so every number here uses random
bytes; and the persistent arms with `parts>1` need a 664 MB partial scratch for the whole token,
which is why `parts=1` is both the simplest and the fastest.

## Files

`sweep.json` is the full report of the run: every arm of every shape, the aggregate table with
min/max, the anchors, the exactness checks (12 of 12 small-shape arms bit-exact, 25 of 25 sampled
matrices equal between the persistent list and the per-matrix path) and the launch-versus-bytes fit
(which is not clean: its intercept absorbs the strided-read cost, so this page quotes the direct
measurements instead). The harness itself is `bench-logs/arc_gemv_sweep.py`, not tracked.
