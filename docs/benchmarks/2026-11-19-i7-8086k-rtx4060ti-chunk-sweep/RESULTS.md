# KINDI_CHUNK prefill sweep: 2B, 4B and 9B at 4K and 8K

Measured 2026-11-19 on the i7-8086K + RTX 4060 Ti 8 GB, native Windows 10 22H2, Python 3.13.6,
CuPy 14.2.0 and driver 617.42, with the `bonsai-2b` and `bonsai-9b` packs; the `bonsai-4b` arm
was added 2026-11-20 on the same host and toolchain. All three packs are 5-trit. Three fresh-process
passes per chunk size; each table reports p50/p95 over those passes. The packs were run separately;
the 9B output is under `9b/`, the 4B under `4b/`.

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
python run_chunk_sweep.py --pack ~/.kindi/packs/bonsai-4b --out 4b
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

## 4B measurement

The 4B fits the same 8,320-token context: the loader reported 5,675 MiB free after the 5-trit lane
and K/V cache were allocated. Three fresh-process passes were run for each chunk; the sampled peak
is device-wide. The full artifact is [`4b/results.json`](4b/results.json); the per-pass JSONs are
under `4b/logs/`.

| chunk | 4K prefill p50 / p95 | 4K total p50 / p95 | 8K prefill p50 / p95 | 8K total p50 / p95 | peak used (delta) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2,587.26 / 2,588.61 ms (1,583 tok/s) | 3,523.69 / 3,527.49 ms | 7,056.09 / 7,057.38 ms (1,161 tok/s) | 8,046.39 / 8,049.14 ms | 2,281 MiB (+1,261) |
| 256 | 2,456.43 / 2,456.79 ms (1,667 tok/s) | 3,390.93 / 3,393.68 ms | 6,805.34 / 6,806.10 ms (1,204 tok/s) | 7,794.66 / 7,797.59 ms | 2,323 MiB (+1,303) |
| 512 | 2,326.57 / 2,328.72 ms (1,760 tok/s) | 3,260.21 / 3,271.87 ms | 6,558.49 / 6,559.40 ms (1,249 tok/s) | 7,547.47 / 7,550.50 ms | 2,389 MiB (+1,369) |
| 1024 | **2,254.26 / 2,254.89 ms** (1,817 tok/s) | **3,188.25 / 3,196.80 ms** | **6,331.19 / 6,331.60 ms** (1,294 tok/s) | **7,322.49 / 7,322.72 ms** | 2,527 MiB (+1,507) |

Prefill is 70.7-73.5% of the 4K request and 86.5-87.7% of the 8K request. The 1024 p50-p95 spread
is 0.6 ms at 4K and 0.4 ms at 8K.

**1024 is the measured 4B optimum at both prompt lengths.** It is 3.1% faster than 512 at 4K and
3.5% faster at 8K, and 8.2%/7.0% faster than 256; it costs 138 MiB over 512 and 204 MiB over 256
in sampled peak. `KINDI_CHUNK=1024` verifies 4/4 on the 4B: the transcript is in
[`4b/verify-1024.txt`](4b/verify-1024.txt) and printed `ALL PASS` with 40/40 tokens on each
prompt.

## What it decides

- **256 is the measured 2B optimum at both prompt lengths, and it is already the default.** It is
  the fastest 2B cell at 4K and 8K; 512 and 1024 are 1.3-2.0% slower in prefill and 1.3-1.4% slower
  end to end, while raising the 2B sampled peak by 54 and 159 MiB over 256.
- **128 is a real loss for the 2B, not noise.** It is 22.6% slower at 4K and 15.9% at 8K prefill,
  and 15.3% / 13.4% slower per request; the p50-p95 spread is under 5 ms in every cell, so the
  ordering is not pass-to-pass variance.
- **4B and 9B are model-specific 1024 cases.** The 4B's 1024 cell is 3.1%/3.5% faster than 512
  and 8.2%/7.0% faster than 256 at 4K/8K, at +138/+204 MiB sampled peak; the 9B's is 1.0%/1.4%
  ahead of 512 and 9.5%/11.2% ahead of 256, at +184 MiB over 512. The 4B result confirms the shape;
  it does not make 1024 a global default.
- **No global default change.** The fixed heuristic (`256`, or `512` on fp16-prompt cards) stays:
  a global 1024 would make the 2B 1.6-1.7% slower and raise its sampled peak by 159 MiB, and 512
  would still regress the 2B while giving up the 4B/9B best cell. Use 1024 as a per-model/autotune
  candidate for the 4B and 9B; the 27B remains unmeasured.
- **All three measured verify cells pass 4/4.** `KINDI_CHUNK=256` on the 2B,
  `KINDI_CHUNK=1024` on the 4B and `KINDI_CHUNK=1024` on the 9B all printed `ALL PASS`; the 4B and
  9B transcripts are under `4b/` and `9b/`.

## Not verified

One card and three packs: the 2B, 4B and 9B only. The 4B and 9B fit at 8,320 context; the 27B arm
was not run, nor were other chunk sizes. One request at a time, greedy decode, no speculation, no
server queueing or tokenizer time, and one 8,320-token context for both prompt lengths. The peak is
device-wide, so it includes the desktop's ~934-1,253 MiB; the delta column is the figure to compare
across chunks.
