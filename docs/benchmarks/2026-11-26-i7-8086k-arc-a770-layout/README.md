# Arc A770: the packed GEMV's read pattern was never the limiter

Host: i7-8086K, Windows 10, Arc A770 16 GB (driver 32.0.101.9175, OpenCL C 1.2, max allocation
4095 MiB). Date: 2026-11-26. Model shapes: the Bonsai 2.5 27B's 401 packed ternary matrices,
6,799,564,800 bytes of codes+scales per token (the embedding lookup excluded, as the committed
figure does).

This page answers the open question the [overhead page](../2026-11-25-i7-8086k-arc-a770-gemv-overhead/README.md)
left: at 253 GB/s against a 384.8 GB/s streaming read, was the packed kernel slow at *reading*, or
already working on something else? It was something else.

## Method

Two kernels over the same bytes, same work list, same launch shape (one persistent launch per
upload chunk, `parts=1`, grid 1 MiB, eight rows per work-item):

* **L1** — the shipped `pack_gemv_persist` layout exactly as the pack stores it,
  `[phase][slot][row]`: per phase a work-item walks 32 slot loads, each `rows` bytes apart
  (5,120-17,408 B on these shapes).
* **L2** — the same arithmetic over a repacked `[tile][phase][slot][row]` layout, 256 rows per
  tile. A wave of 32 work-items covers one tile, so its 32 slot loads per phase are one
  contiguous 8 KB run read front to back. Every 27B shape is a multiple of 256 rows and 128
  columns, so the tiling is exact and no padding is needed.

Arm **L1-ro** and **L2-ro** are the same two address patterns with the decode deleted: the same
loads, 8 integer adds per load instead of the 8 x `TRI_DOT` (about 128 integer ops per load). They
separate "can the card read this pattern" from "can it decode it fast enough".

The repack is a device kernel plus one buffer copy per matrix, timed separately: it is a load-time
cost, not a per-token one.

Distinct bytes, not a shared buffer: the committed 26.87 ms arm pointed every matrix at one
318 MB region. These runs upload the model's real 6.80 GB into two 3.76 GiB chunks (the card's
4 GiB per-allocation limit) and every matrix keeps its own bytes. The same L1 kernel in this
configuration measures 27.44 ms, so the two buffer schemes differ by about 2% and the A/B below is
only compared inside this rig.

## Results

Fourteen timed samples per arm (7 per cycle, two cycles), alternating L1, L2, L1-ro, L2-ro:

| arm | ms/token | tok/s | GB/s | range |
|---|---:|---:|---:|---|
| L1, pack layout, full GEMV | 27.44 | 36.4 | 247.8 | 26.64-29.34 |
| L2, blocked layout, full GEMV | **26.31** | **38.0** | **258.5** | 25.25-27.39 |
| L2, blocked layout, decode removed | 18.05 | 55.4 | 376.7 | 17.78-18.55 |
| L1, pack layout, decode removed | **16.79** | **59.6** | **405.0** | 16.45-17.29 |

Paired per cycle: L2 is faster than L1 in **13 of 14** samples, by 4.1% of the median.

In one line: the shipped layout reads this model in 27.44 ms/token (247.8 GB/s), the blocked
layout in 26.31 ms/token (258.5 GB/s). With the decode removed the shipped layout reads at
16.79 ms/token (405.0 GB/s) and the blocked one at 18.05 ms/token (376.7 GB/s), and the blocked
arm still won 13 of 14 paired samples.

## What it means

**The strided byte pattern is not what is left.** With the decode removed, the *shipped* layout
already reads at 405 GB/s, above the 384.8 GB/s streaming anchor this card gets from a plain
`uint4` read. There was nothing to fix in the read pattern: 32 concurrent 8-byte streams at a
5 KB stride still saturate the memory system when nothing else is happening.

**The blocked layout wins 4%, and not from bandwidth.** Its decode-free arm is *slower* (376.7
GB/s against 405.0), so the bytes are not arriving faster. What changes is the address arithmetic:
in L1 the slot stride is `rows`, a runtime multiply per load in a 64-bit address chain; in L2 it
is a constant 256, a shift. In a kernel already at the ALU limit, that is worth more than the
memory locality it costs.

**The remaining ~8 ms/token is the decode.** Decode-free reads take 16.79-18.05 ms; the GEMV takes
26.31-27.44 ms. The difference is `TRI_DOT`: four multiply-shifts, four subtracts and four
multiply-accumulates per weight, eight rows deep. At 26.31 ms over 26.9 G weights that is
1.0 T weights/s, and the four to five integer operations each weight needs put the decode at
4-5 T integer ops/s, which is where this card's integer rate sits.

Two things that did not help, for the record: sixteen rows per work-item over the blocked layout
is **worse** (33.6 ms, register pressure with 16 accumulators), and more rows per work-item had
already been flat at eight in the overhead sweep.

## Verdict

Do not repack the weights yet. A 75 ms load-time pass buys 4% of the token and costs a 318 MB scratch
buffer; the same 75 ms spent on the decode is where the next 40% is. The
measured target is now exact: if the decode were free, the token would be 16.8-18.1 ms
(55-60 tok/s) with no layout change at all. The next arm should cut `TRI_DOT` — a 256-entry
byte-to-four-trits table in local memory, or a pair-code decode that produces two weights per
multiply — and it should be measured against the 26.31 ms blocked arm and the 27.44 ms shipped
arm from this page, not against the earlier 26.87 ms (different buffer scheme).

Caveats: the host is not quiet, so this page reads medians with ranges and a paired per-cycle
count rather than a single number; the bytes are splitmix64 fill, not the pack's own (every number
here is a bandwidth and instruction-count measurement, and the decode is bit-exact against the
host model on a real 1024x5120 matrix with exactly representable scales); the 27B pack is not on
this host, so the matrix list is rebuilt from the model's shape mix and checked against the
committed 6,799,564,800-byte total before anything runs; and there is still no Intel backend, no
model run and no support claim.

## Files

`ab.json` is the probe's own output: both arms' samples, the decode-free arms, the exactness
checks, the chunking, the paired count and the committed baseline it compares to. The harness is
`bench-logs/arc_layout_probe.py`, not tracked.
