# Bonsai 2.5 27B on the Arc A770: every ternary matmul, from the pack's own bytes

Host: i7-8086K, Windows 10, 12 cores, Python 3.13. Card: Intel(R) Arc(TM) A770 Graphics, 15932 MiB (OpenCL C 1.2 ).
Date: 2026-11-25. Command:

```sh
python -m kindi convert Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --out pack
python -m kindi intel-pack --pack pack --device-name Arc --repeats 5 --json run-base.json
python -m kindi convert Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf --out pack
python -m kindi intel-pack --pack pack --device-name Arc --repeats 5 --json run-mtp.json
```

Per-shape timing is `mean_ms` in a new run's JSON: it is the arithmetic mean over the five timed
launches. The two artifacts on this page were written before that key was named honestly and store the
same mean as `median_ms` (round-5 audit F3); the claims that quote them point into those files and were
not re-measured.

`intel-pack` reads `bcodes.bin` and `dpm.bin` exactly as the converter wrote them, uploads each
shape's slice unchanged, and runs the four-row phase-split GEMV kernel (one warm-up, five timed
launches per shape). One real tensor of the pack, `blk.3.attn_k.weight` (1,024 x 5,120), is decoded
on the host at one phase part and compared with the device output bit for bit: **0 mismatches**.

## What this is, and is not

It is the memory floor of one decode token: a single pass over every ternary tensor in the pack at
its own shape. It is **not** a model run: attention, norms, the GDN state update and the f32 output
head's own work are excluded, each launch is measured from the host, and the KV cache is not read.
A backend will therefore be slower than this number. No engine path exists on Arc and there is no
Intel support claim; M2 stays partial.

The host was otherwise idle during these runs (no Kindi process, no conversion in flight).

## Plain Bonsai 2.5 27B pack

Pack: Ternary-Bonsai-2.5-27B-PTQ1_0.gguf, 402 ternary of 851 tensors, 8 distinct shapes, 6717440000 bcodes bytes.

| rows x cols | tensors | ms per call | GB/s |
|---|---:|---:|---:|
| 248320 x 5120 | 2 | 1.5319 ms | 220.461 GB/s |
| 5120 x 17408 | 64 | 0.1483 ms | 159.685 GB/s |
| 17408 x 5120 | 128 | 0.1489 ms | 158.977 GB/s |
| 12288 x 5120 | 16 | 0.121 ms | 138.159 GB/s |
| 10240 x 5120 | 48 | 0.1098 ms | 126.88 GB/s |
| 6144 x 5120 | 48 | 0.0815 ms | 102.5 GB/s |
| 5120 x 6144 | 64 | 0.0808 ms | 103.465 GB/s |
| 1024 x 5120 | 32 | 0.062 ms | 22.469 GB/s |

Projection: **7.14 GB** of packed weights read once, **49.88 ms** per pass, **143.081 GB/s** effective,
**20.047 tokens/s** ceiling.

## `-mtp` pack

Pack: Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf, 402 ternary of 851 tensors, 8 distinct shapes, 6717440000 bcodes bytes. The 65th block (the draft
head) is not a model layer, so the pack is the same 64-layer model and the probe times the same
shapes. Validation: **0 mismatches** on `blk.3.attn_k.weight`.

| rows x cols | tensors | ms per call | GB/s |
|---|---:|---:|---:|
| 248320 x 5120 | 2 | 1.5328 ms | 220.32 GB/s |
| 5120 x 17408 | 64 | 0.1462 ms | 161.913 GB/s |
| 17408 x 5120 | 128 | 0.1485 ms | 159.47 GB/s |
| 12288 x 5120 | 16 | 0.1198 ms | 139.52 GB/s |
| 10240 x 5120 | 48 | 0.1088 ms | 127.953 GB/s |
| 6144 x 5120 | 48 | 0.083 ms | 100.721 GB/s |
| 5120 x 6144 | 64 | 0.0826 ms | 101.16 GB/s |
| 1024 x 5120 | 32 | 0.0604 ms | 23.042 GB/s |

Projection: **7.14 GB** of packed weights read once, **49.77 ms** per pass, **143.405 GB/s** effective,
**20.092 tokens/s** ceiling.

## With and without the MTP head

The two runs describe the same work: the `-mtp` file's head is excluded at conversion, so both
packs hold the same 64-layer model and the projections agree within run-to-run noise (20.092 vs 20.047 tokens/s).
On this card there is nothing to benchmark *between* the files yet - a difference would only appear
once a 27B drafter can consume the head (roadmap row I8), which no platform has.

## Persistent work list: eight rows per item, one launch per chunk

The per-shape numbers above pay two enqueues per matrix - the GEMV and, when `parts > 1`, its reduce -
and one work-item per four rows. The persistent kernel gives each work-item **eight rows** of one
matrix and runs a **work list over every ternary tensor of the pack**, with `parts=1` so each row has a
single partial and the reduce pass disappears: one enqueue covers a whole upload chunk.

A 27B pack's 6.7 GB of codes does not fit the card's 4 GiB per-allocation limit
(`CL_DEVICE_MAX_MEM_ALLOC_SIZE`, 4095 MiB here), so the pass uploads in chunks sized from that limit -
two chunks of 3.76 GiB or less - and the chunk times are summed. The chunk count is not cosmetic: the
same pack measured 154 GB/s at seven 1 GiB chunks against 235 GB/s at two, and 168 GB/s in one
five-launch run while the card warmed up (untracked diagnostics, `bench-logs/persist_chunk_probe.py`).
The runs below therefore use twenty timed launches per chunk.

Three spaced runs per pack, host idle between runs:

| pack | run | persistent | per pass | per-shape | speedup |
|---|---|---:|---:|---:|---:|
| 2.5 27B `-mtp` | 1 | 221.3 GB/s | 32.26 ms | 148.7 GB/s | 1.489x |
| 2.5 27B `-mtp` | 2 | 219.4 GB/s | 32.53 ms | 146.9 GB/s | 1.493x |
| 2.5 27B `-mtp` | 3 | 223.6 GB/s | 31.92 ms | 149.2 GB/s | 1.499x |
| Bonsai 2 27B | 1 | 227.2 GB/s | 31.42 ms | 146.8 GB/s | 1.548x |
| Bonsai 2 27B | 2 | 225.5 GB/s | 31.65 ms | 145.7 GB/s | 1.548x |
| Bonsai 2 27B | 3 | 228.3 GB/s | 31.26 ms | 137.7 GB/s | 1.658x |

Artifacts: `run-25mtp-persist-{1,2,3}.json` (Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf) and
`run-bonsai2-persist-{1,2,3}.json` (Ternary-Bonsai-2-27B-PTQ1_0.gguf); both hold 402 ternary tensors of
851, 8 distinct shapes and 7,137,280,000 packed weight bytes, so they are the same work on the same
byte count. The 2.5 pack is the `-mtp` file: the plain 2.5 file is no longer on this host, and the
Bonsai 2 27B pack stands in for it (the probe times shapes, not layer identities).

Persistent validation: **0 mismatches** - the smallest ternary tensor's rows from the work-list kernel
are compared with the host model bit for bit at `parts=1`, so the speed is measured on bytes that were
decoded correctly, not on bytes that merely moved.

### Why the pack does not reach the synthetic 253 GB/s

The synthetic sweep (`bench-logs/arc_gemv_sweep.py`, untracked) reported 253 GB/s for one launch over
401 dummy matrices of these shapes. Two differences survive on real weights: the pack needs two
allocations, and the second chunk is smaller and slower, so the best chunk here (238.1 GB/s on
Bonsai 2 run 1) is the fair comparison against the sweep, not the pack total. The sweep also put every
matrix at buffer offset zero, so same-shape matrices re-read the same bytes; the pack cannot, and this
page quotes the pack total for that reason.

## What this does not show

- **Not a token rate.** No attention, GDN, norm or head kernel exists on Arc; this is the
  weight-streaming ceiling, not `kindi bench`.
- **Not a comparison with the CPU engine.** The CPU engine runs the whole model; these numbers
  cover only the ternary matmuls, measured through OpenCL.
- **Not M2 passed.** Windows is not the plan's target, and `libksycl.so` and engine device
  selection still do not exist.
- **One host, one driver.** The card, driver and OpenCL runtime are recorded in the JSON; a
  different driver can change the enqueue cost that dominates the small shapes.

## Where this leads

The next steps are the gate-only rows in `docs/ROADMAP.md` ("Intel Arc A770 decode hypotheses
(2026-11-25)"): `s8 x s2` DPAS, a 2-bit swizzled layout, a single-launch kernel, `dp4a`-shaped
prefill, split-K queues, a popcount comparison and the non-ternary kernels. Each needs a
measurement before any code is adopted, and the vendor premises behind the DPAS rows are unverified
on this card.
