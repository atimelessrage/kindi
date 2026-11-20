# Prefill vs decode as a share of wall clock: 2B at 4K and 8K

Measured 2026-11-17 on the i7-8086K + RTX 4060 Ti, pack `bonsai-2b` (5-trit 2B), with `kindi bench`
at its defaults (exact head, no speculation, automatic K/V, chunked prefill). Two runs per context;
the figures below are medians over the runs that measured each prompt length (512 and 4096 ran in
both contexts, so four times each; 8192 ran twice).

## Method

`kindi bench` grew `--split-at` (default `4096,8192`) and `--split-generated` (default 128). Each
split row is **one timed request**: a real `feed_batch` prefill of P tokens - the chunked path the
server uses - followed by 128 greedy `step`s, with a small warm-up request first and a GPU sync
before and after each block. The reported share is each block's wall clock divided by the request's
total. The prompt is a fixed id ramp, so two runs do the same work.

```sh
KINDI_PACK=<packs>/bonsai-2b KINDI_CTX=4224 python -m kindi bench \
    --split-at 512,4096 --split-generated 128 --json .../split-4k-r1.json
KINDI_PACK=<packs>/bonsai-2b KINDI_CTX=8448 python -m kindi bench \
    --split-at 512,4096,8192 --split-generated 128 --json .../split-8k-r1.json
# each repeated as -r2
```

The context is the prompt plus the 128 generated tokens rounded up (4224 for 4K, 8448 for 8K), so no
request overflows the context.

## The measurement

| prompt | generated | prefill | prefill share | decode | decode share | request total |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 0.08 s (6,189 tok/s) | **13.6%** | 0.52 s (244.5 tok/s) | **86.4%** | 0.61 s |
| 4,096 | 128 | 0.94 s (4,349 tok/s) | **64.1%** | 0.53 s (242.7 tok/s) | **35.9%** | 1.47 s |
| 8,192 | 128 | 2.55 s (3,215 tok/s) | **82.2%** | 0.54 s (233.2 tok/s) | **17.8%** | 3.10 s |

Run-to-run spread is small: the 512 rows are 13.5-14.9%, 4096 is 63.3-64.1%, 8192 is 82.1-82.4%.
The decode rate is consistent with the generation sweep in the same JSON rows (225-235 tok/s at
position 100, 2,000 and 8,000), so the split and the old numbers agree where they overlap.

## What it decides

- **The crossover is between 512 and 4,096 prompt tokens.** A short chat prompt plus a 128-token
  reply is decode-bound (86% decode); at 4K prefill is already the larger half (64%); at 8K it is
  82%. A single "prefill dominates" statement is wrong without the prompt length, and so is a single
  "decode dominates" one.
- **The old pair of rates cannot produce this share.** `kindi bench`'s `prompt reading` line times 20
  single-token `feed` calls: 300.6-304.5 tokens/s after warm-up, and 63.0 on the first run in a cold
  process. The split's prefill block measures the same work chunked: **4,349 tok/s at 4K and 3,215 at
  8K, 11-14x the read line**. Estimating a 4K request from 302 tok/s gives 13.6 s of prefill; the
  request actually spends 0.94 s there. The two reported numbers were not a decomposition of a
  request, and the read number was pessimistic by an order of magnitude (and unstable cold).
- **What to tune, for this shape:** at 4K and 8K prefill is the majority of the clock, so prefill
  work (chunk size, batching) matters more than decode kernels; at 512 the opposite holds. Decode
  costs about 0.53 s per 128 tokens at every depth, which is the figure to shrink for short prompts.

## Artifacts

`split-4k-r1.json`, `split-4k-r2.json`, `split-8k-r1.json`, `split-8k-r2.json` - the full `kindi
bench` result of each run, including the generation sweep and free VRAM (6,547 MiB free at 4K, 6,517
at 8K with the 2B loaded). The `pack` field is reduced to the pack's name in these committed copies;
the run used the pack in `~/.kindi/packs`.

## Not verified

One card and one pack; the 2B only, so the 27B half of the ledger gate is not measured here. Greedy
decode, 128 generated tokens, one request at a time, no speculation, and `KINDI_CHUNK` at its
default - a chunk-size change or a second concurrent request would move the prefill rate and the
share. The share is wall clock on this host; nothing here measures the server's queueing or
tokenizer time, which the bench does not include.
