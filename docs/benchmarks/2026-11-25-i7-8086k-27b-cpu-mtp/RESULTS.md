# 27B CPU MTP: the integrated head vs the 2B drafter (2026-11-25)

The first measurement of `kindi/mtp_cpu.py` on the real 27B: the `-mtp` GGUF's own
`blk.64` draft head and an external 2B pack drafter, scored against the model's own greedy
tokens on the CPU engine. The drafts are scored, never substituted, so every run below is
token-identical to plain greedy decode (`exact: true` on all four rows in `results.json`).

## Method

Same 27B pack (`bonsai25-27b`), same two prompts, `k=4`, 16 tokens, ctx 2048, 6 threads,
i7-8086K (`avx2-fma-f16c`):

```
python tools/mtp_cpu_bench.py --pack PACK --gguf <the -mtp GGUF> --tokens 16 --k 4 --out <run.json>
python tools/mtp_cpu_bench.py --pack PACK --drafter-pack <2B pack> --tokens 16 --k 4 --out <run.json>
```

The CPU engine verifies one row per forward pass, so a drafted run is strictly slower than
plain decode: this run measures acceptance and cost, not speed, and makes no speedup claim.
`results.json` in this directory is the sanitized record of both runs (machine-local paths
removed).

## Result

| drafter | accepted | rate | depth [1..4] | plain tok/s | drafted tok/s |
|---|---|---|---|---|---|
| integrated `blk.64` head | 7/108 | 6.5% | [3, 1, 2] | 1.69 / 1.51 | 1.11 / 1.02 |
| 2B pack drafter | 33/108 | 30.6% | [15, 8, 7, 3] | 2.03 / 2.12 | 1.01 / 1.16 |

Per prompt, the 2B drafter accepted 23/54 (43%) on the first prompt and 10/54 (19%) on the
second; the head accepted 1/54 and 6/54.

## What it decided

The external 2B pack is the better draft source by about five times the acceptance, and it
is still slower than plain decode. Neither route is a speedup until CP1's batched T-row
verify exists, and the measurement settles which drafter CP1 builds against: the 2B pack,
not the head. The head-side follow-ups (output-normed hidden, verified-row seeding, the
variant ladder) are in `docs/MTP.md`; they are probe runs, not part of this page.
