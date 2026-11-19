# KINDI_CHUNK prefill sweep: 2B and 9B at 4K and 8K

Measured 2026-11-19 on the i7-8086K + RTX 4060 Ti 8 GB, native Windows 10 22H2, Python 3.13.6,
CuPy 14.2.0 and driver 617.42, with the `bonsai-2b` and `bonsai-9b` packs (both 5-trit). Three
fresh-process passes per chunk size; each table reports p50/p95 over those passes. The packs were
run separately; the 9B output is under `9b/`.

## Method

`run_chunk_sweep.py` runs one `kindi bench` process per (chunk, pass) with `KINDI_CHUNK` set and
`KINDI_CTX=8320`, and passes `--split-at 4096,8192 --split-generated 128 --sample-vram --json`.
Each split row is one timed request: a real `feed_batch` prefill of the prompt followed by 128
greedy steps, synced before and after each block. `kindi bench` reports the prefill and decode
milliseconds of that request, and `--sample-vram` samples device-wide `nvidia-smi memory.used`
every 0.5 s and reports the peak.

```sh
python run_chunk_sweep.py --pack ~/.kindi/packs/bonsai-2b --out .
python run_chunk_sweep.py --pack ~/.kindi/packs/bonsai-9b --out 9b
```

Both prompt lengths run in one process at the 8K context (8,192 + 128 fits 8,320), so every chunk
sees the same allocated cache and load. p50/p95 are linear-interpolation percentiles over the
three passes (numpy's default definition); the peak is the largest sampled `memory.used` of the
three passes, with its delta over the idle baseline. The tracked bench JSONs under `logs/` and
`9b/logs/` replace the pack directory with `<pack>` and the local home with `<local-home>`, and
each JSON's `pack`
field keeps only the pack's name. The per-pass stdout `.log` files are covered by the repository's
`*.log` ignore rule and are not tracked; `results.json` carries every pass's numbers.

## The measurement

| chunk | 4K prefill p50 / p95 | 4K total p50 / p95 | 8K prefill p50 / p95 | 8K total p50 / p95 | peak used (delta) |
|---:|---:|---:|---:|---:|---:|
| 128 | **1,179.81 / 1,180.75 ms** (3,472 tok/s) | 1,742.29 / 1,757.41 ms | **3,015.29 / 3,020.26 ms** (2,717 tok/s) | 3,595.70 / 3,620.55 ms | 1,576 MiB (+642) |
| 256 | **962.15 / 962.91 ms** (4,257 tok/s) | 1,511.37 / 1,526.24 ms | **2,602.72 / 2,604.77 ms** (3,148 tok/s) | 3,169.99 / 3,194.55 ms | 1,602 MiB (+673) |
| 512 | 974.82 / 979.46 ms (4,202 tok/s) | 1,524.35 / 1,527.19 ms | 2,653.89 / 2,654.76 ms (3,087 tok/s) | 3,211.30 / 3,224.37 ms | 1,656 MiB (+727) |
| 1024 | 978.66 / 982.63 ms (4,185 tok/s) | 1,531.79 / 1,546.11 ms | 2,643.53 / 2,651.75 ms (3,099 tok/s) | 3,210.27 / 3,221.94 ms | 1,761 MiB (+832) |

Prefill's share of the request is 67-68% at 4K and 82-84% at 8K in every cell, matching the
[bench-split page](../2026-11-17-i7-8086k-rtx4060ti-bench-split/RESULTS.md).

## 9B measurement

The 9B pack fits the same 8,320-token context: the loader reported 5,079 MiB free after the
5-trit lane and K/V cache were allocated. Three fresh-process passes were run for each chunk, as
for the 2B, and the sampled peak is device-wide. The full artifact is
[`9b/results.json`](9b/results.json); the per-pass JSONs are under `9b/logs/`.

| chunk | 4K prefill p50 / p95 | 4K total p50 / p95 | 8K prefill p50 / p95 | 8K total p50 / p95 | peak used (delta) |
|---:|---:|---:|---:|---:|---:|
| 128 | 3,515.56 / 3,522.53 ms (1,165 tok/s) | 5,007.60 / 5,033.66 ms | 8,966.72 / 9,115.61 ms (914 tok/s) | 10,529.32 / 10,685.38 ms | 3,318 MiB (+2,122) |
| 256 | 3,414.99 / 3,452.57 ms (1,199 tok/s) | 4,904.95 / 4,940.29 ms | 8,978.64 / 9,040.01 ms (912 tok/s) | 10,444.35 / 10,585.15 ms | 3,443 MiB (+2,258) |
| 512 | 3,151.39 / 3,154.33 ms (1,300 tok/s) | 4,535.56 / 4,578.21 ms | 8,185.80 / 8,417.01 ms (1,001 tok/s) | 9,646.45 / 10,178.99 ms | 3,447 MiB (+2,194) |
| 1024 | **3,118.80 / 3,122.36 ms** (1,313 tok/s) | **4,502.79 / 4,532.50 ms** | **8,071.40 / 8,084.76 ms** (1,015 tok/s) | **9,520.88 / 9,534.15 ms** | 3,631 MiB (+2,378) |

Prefill is 69-70% of the 4K request and 84-86% of the 8K request in every 9B cell. The 1024
p50-p95 spread is 3.6 ms at 4K and 13.4 ms at 8K, narrower than the corresponding 128/256 tails.

**1024 is the measured 9B optimum at both prompt lengths.** It is 1.0% faster than 512 at 4K and
1.4% faster at 8K, and 9.5%/11.2% faster than 256; it also has the highest sampled peak
(3,631 MiB, 184 MiB above 512). `KINDI_CHUNK=1024` still verifies 4/4 on the 9B: `python -m kindi
--pack ~/.kindi/packs/bonsai-9b verify` printed `ALL PASS` and 40/40 tokens on each of the four
prompts. The transcript is in [`9b/verify-1024.txt`](9b/verify-1024.txt).

## What it decides

- **256 is the measured optimum at both prompt lengths, and it is already the default.** It is
  the fastest cell at 4K and 8K; 512 and 1024 are 1.3-2.0% slower in prefill and 1.3-1.4% slower
  end to end, while raising the sampled peak by 54 and 159 MiB over 256.
- **128 is a real loss, not noise.** It is 22.6% slower at 4K and 15.9% at 8K prefill, and 15.3%
  / 13.4% slower per request; the p50-p95 spread is under 5 ms in every cell, so the ordering is
  not pass-to-pass variance.
- **No global default change.** The fixed heuristic (`256`, or `512` on fp16-prompt cards) already
  matches the 2B; 128 is a real loss there, and 512/1024 cost 1.3-2.0% in 2B prefill while raising
  peak. The 9B's 1024 optimum is model-specific, not evidence for changing the 2B default.
- **The 9B arm changes the model-specific answer.** 1024 is the 9B optimum, 1.0-1.4% ahead of 512
  and 9.5-11.2% ahead of 256. A future per-model/autotune candidate should use 1024 on the 9B; the
  current global default stays 256 until the 4B and the other models are measured.
- **`KINDI_CHUNK=256` still verifies 4/4 on the 2B, and `KINDI_CHUNK=1024` verifies 4/4 on the 9B.**
  `python -m kindi --pack ~/.kindi/packs/bonsai-2b verify` with `KINDI_CHUNK=256` printed ALL PASS,
  40/40 tokens on each of the four prompts; the 9B run is noted above.

## Not verified

One card and two packs: the 2B and 9B only. The 9B fits at 8,320 context; the 4B and 27B arms were
not run, nor were other chunk sizes. One request at a time, greedy decode, no speculation, no server
queueing or tokenizer time, and one 8,320-token context for both prompt lengths. The peak is
device-wide, so it includes the desktop's ~934-1,253 MiB; the delta column is the figure to compare
across chunks.
