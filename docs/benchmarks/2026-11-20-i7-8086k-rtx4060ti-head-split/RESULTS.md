# Exact tandem head row split - 9B proxy (2026-11-20)

## What was tested

`KINDI_TANDEM_HEAD_SPLIT=<percent>` splits the exact output head by vocabulary rows: the CPU owns
the lower rows and the GPU the upper rows. Both run the same output norm and a disjoint slice of
`output.weight`; the greedy token is the full host argmax after both slices are copied back. The
split is opt-in and is refused for a pack whose head is tied to `token_embd.weight` (the 2B/4B) and
for the plan owner (its head lives in the card layout; this run uses the plain owner).

Host: i7-8086K, RTX 4060 Ti 8 GB, Windows native CPU build (clang 18.1.8, `avx2-fma-f16c`,
6 threads). Pack: `bonsai-9b` (32 layers, width 4096, separate `output.weight` 248,320 x 4,096
ternary, 258 MiB). Placement: `cpu-first`, 16 CPU layers (0..15), 16 GPU layers (16..31), plain
GPU, requested context 24,576. Bench request: 1,024 prompt + 32 generated; the generation probes
at positions 100/2,000/8,000/16,000/24,000 are single runs.

## Verdict

**No-go for speed.** The 50/50 split is token-exact but decodes at **7.7 tok/s** against **10.8
tok/s** for the same placement with the full GPU head (0.71x; about 40% more decode time). The
25/75 split is closer to parity at **10.5 tok/s**, still not faster. The split only redistributes
the head: the GPU peak falls by the CPU's half, while the CPU gains that same half in host RAM. It
is GPU headroom, not a total-memory saving, and it does not reach the roadmap's 1.10x speed gate.

The 27B gate row was not measured: there is no 27B pack on this host. The 2B/4B cannot be tested
without duplicating a tied embedding/head.

## Exactness

All three configurations pass `kindi verify` 4/4 token-exact against the llama.cpp reference:

| run | verify | tok/s over the verify run |
|---|---|---|
| no split | ALL PASS 4/4 | 13.1 |
| 25% CPU / 75% GPU | ALL PASS 4/4 | 12.7 |
| 50% CPU / 50% GPU | ALL PASS 4/4 | 12.5 |

## Bench: speed, head slice, peak

| run | decode 1,024+32 | prefill | peak VRAM | head probe avg | all-token probe avg |
|---|---:|---:|---:|---:|---:|
| no split | **10.8 tok/s** | 14.1 tok/s | **3,186 MiB** | 1.584 ms | 74.405 ms |
| 25/75 | 10.5 tok/s | 13.9 tok/s | 3,087 MiB | 2.600 ms | 78.100 ms |
| 50/50 | **7.7 tok/s** | 13.2 tok/s | **3,060 MiB** | 3.185 ms | 81.534 ms |

`head probe avg` is the existing `KINDI_TANDEM_DEBUG_TIMING` field averaged over all 1,141 tokens
of the bench run; most are prefill tokens and never evaluate the head, so it is not a decode-only
head latency. The decode rates are the direct measurement.

Generation probes (tokens/s at position), one run each:

| run | 100 | 2,000 | 8,000 | 16,000 | 24,000 |
|---|---:|---:|---:|---:|---:|
| no split | 14.5 | 12.4 | 8.3 | 5.5 | 3.9 |
| 25/75 | 9.2 | 9.9 | 6.5 | 4.1 | 3.0 |
| 50/50 | 11.2 | 10.0 | 7.1 | 4.3 | 3.2 |

Run-to-run thermal variance is visible (the 25/75 point at position 100 is below the 50/50 point),
but every direct decode comparison is slower than the full head.

## Memory accounting

The 9B head is 257.7 MiB in the ternary pack (248,320 rows x 4,096 columns plus scales). A 50%
split puts about 128.8 MiB on each side; the observed GPU peak reduction is 126 MiB. The 25% split
puts about 64.4 MiB on the CPU and the observed GPU peak reduction is 99 MiB (sampled peak, so
allocator and sampling noise are included). The CPU host allocation grows by the same half. Total
model bytes are unchanged; only the device/host boundary moves.

## Implementation note

The split is sequential: the CPU slice and the GPU slice are evaluated one after the other, then
merged. This test therefore does not establish what a concurrent two-stream head could do; the
roadmap's gate is about the exact split as built, and its 1.10x target is not met here. A concurrent
version would need an asynchronous GPU head result and a separate decode-only gate before it could
be considered.

## Evidence

- `verify-base.log`, `verify-25.log`, `verify-50.log`
- `bench-base.log`, `bench-25.log`, `bench-50.log`
- `bench-base.json`, `bench-25.json`, `bench-50.json`
