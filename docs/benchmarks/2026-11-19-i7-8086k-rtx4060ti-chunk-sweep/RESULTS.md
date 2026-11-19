# KINDI_CHUNK prefill sweep: 2B at 4K and 8K

Measured 2026-11-19 on the i7-8086K + RTX 4060 Ti 8 GB, native Windows 10 22H2, Python 3.13.6,
CuPy 14.2.0 and driver 617.42, with the `bonsai-2b` pack (5-trit 2B). Three fresh-process passes
per chunk size; the table reports p50/p95 over those passes.

## Method

`run_chunk_sweep.py` runs one `kindi bench` process per (chunk, pass) with `KINDI_CHUNK` set and
`KINDI_CTX=8320`, and passes `--split-at 4096,8192 --split-generated 128 --sample-vram --json`.
Each split row is one timed request: a real `feed_batch` prefill of the prompt followed by 128
greedy steps, synced before and after each block. `kindi bench` reports the prefill and decode
milliseconds of that request, and `--sample-vram` samples device-wide `nvidia-smi memory.used`
every 0.5 s and reports the peak.

```sh
python run_chunk_sweep.py --pack ~/.kindi/packs/bonsai-2b --out .
```

Both prompt lengths run in one process at the 8K context (8,192 + 128 fits 8,320), so every chunk
sees the same allocated cache and load. p50/p95 are linear-interpolation percentiles over the
three passes (numpy's default definition); the peak is the largest sampled `memory.used` of the
three passes, with its delta over the idle baseline. The tracked bench JSONs under `logs/` replace
the pack directory with `<pack>` and the local home with `<local-home>`, and each JSON's `pack`
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

## What it decides

- **256 is the measured optimum at both prompt lengths, and it is already the default.** It is
  the fastest cell at 4K and 8K; 512 and 1024 are 1.3-2.0% slower in prefill and 1.3-1.4% slower
  end to end, while raising the sampled peak by 54 and 159 MiB over 256.
- **128 is a real loss, not noise.** It is 22.6% slower at 4K and 15.9% at 8K prefill, and 15.3%
  / 13.4% slower per request; the p50-p95 spread is under 5 ms in every cell, so the ordering is
  not pass-to-pass variance.
- **No default change and no `autotune` candidate.** The fixed heuristic (`256`, or `512` on
  fp16-prompt cards) already matches what this host measures; the 512 arm is not justified here,
  and the gate's "change a default only after the sweep" clause resolves to "keep it".
- **`KINDI_CHUNK=256` still verifies 4/4.** `python -m kindi --pack ~/.kindi/packs/bonsai-2b
  verify` with `KINDI_CHUNK=256` printed ALL PASS, 40/40 tokens on each of the four prompts.

## Not verified

One card and one pack: the 2B only, so the roadmap's "9B if it fits" arm was not run. One request
at a time, greedy decode, no speculation, no server queueing or tokenizer time, and one 8,320-token
context for both prompt lengths. The peak is device-wide, so it includes the desktop's ~934 MiB;
the delta column is the figure to compare across chunks.
