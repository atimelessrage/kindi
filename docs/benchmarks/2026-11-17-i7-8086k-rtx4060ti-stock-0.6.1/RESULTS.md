# Stock Kindi v0.6.1 on the RTX 4060 Ti (2026-11-17)

**Stock upstream release, not the fork.** This page uses only Kindi `v0.6.1`
(tag commit `7fececbff13adc88bf21b8f07dea4e1c91996a9d`), extracted with
`git archive v0.6.1`. The extracted package reports `kindi 0.6.1`; no fork
module, kernel or benchmark number is used. The stock `kindi bench` path runs
the plain engine with the exact head (its log says `head plain`), not the
fork's tensor-core or speculation paths.

Host: i7-8086K + NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42,
native Windows, Python 3.13.6, CuPy 14.2.0, CUDA runtime 12.9 (12090).
Packs: `bonsai-2b` (source `Ternary-Bonsai-2-2B-PTQ1_0.gguf`, sha256
`36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`) and
`bonsai-9b` (source `Ternary-Bonsai-2-9B-PTQ1_0.gguf`, sha256
`3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`),
both under `~/.kindi/packs/`. No `KINDI_*` variables were set for these runs.

## Windows loader qualification

Unmodified stock v0.6.1 does not start on native Windows: `kindi/chunk1.py`
hard-codes `ctypes.CDLL("libcuda.so.1")`, which raises
`FileNotFoundError: Could not find module 'libcuda.so.1'` before any engine
work. That failure is kept verbatim (local paths sanitized) in
[`2b-512-unmodified-stock-failure.log`](2b-512-unmodified-stock-failure.log).

The measured runs use [`run_stock.py`](run_stock.py), a small launcher that
maps that one `ctypes.CDLL` name to `nvcuda.dll` on Windows before importing
stock Kindi. It does not patch a Kindi module, kernel, plan or engine decision.
The numbers below are therefore **stock v0.6.1 + a native-Windows loader
shim**, not an unmodified stock binary and not fork numbers.

## Commands

```sh
git archive v0.6.1 | tar -x -C bench-logs/stock-v0.6.1
cp docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-stock-0.6.1/run_stock.py \
   bench-logs/stock-v0.6.1/run_stock.py
cd bench-logs/stock-v0.6.1

python run_stock.py --pack ~/.kindi/packs/bonsai-2b --ctx 512 bench
python run_stock.py --pack ~/.kindi/packs/bonsai-2b --ctx 8192 bench
python run_stock.py --pack ~/.kindi/packs/bonsai-9b --ctx 512 bench
python run_stock.py --pack ~/.kindi/packs/bonsai-9b --ctx 8192 bench
```

`~/.kindi/packs/...` is expanded by the shell; the stock CLI receives the
normal pack path. The command and its output are repeated at the top of each
log.

## Results

One pass per cell. Stock `kindi bench` warms up once and then averages 10
steps at each position; these are indicative single-run numbers, not medians.

| model | context | prompt reading (tok/s) | generation @100 | @2,000 | @8,000 | free MiB |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 512 | 225.9 | 188.4 | — | — | 6,505 |
| 2B | 8,192 | 223.3 | 191.4 | 191.1 | 191.7 | 6,411 |
| 9B | 512 | 93.9 | 85.6 | — | — | 4,613 |
| 9B | 8,192 | 55.3 | 86.3 | 84.3 | 79.7 | 4,371 |

“Context” is the `--ctx` KV budget. Stock's own position sweep is 100, 2,000
and 8,000; it prints only positions that fit, so the 512-context run has only
the position-100 generation row, while the 8K-context run has all three. The
two 8K cells are therefore the only ones with a near-8K generation point.

The 2B stays flat across both contexts. The 9B changes shape: prompt reading
falls from 93.9 to 55.3 tok/s as the context grows, while decode at position
100 is effectively unchanged (85.6 to 86.3 tok/s); at 8K its decode is
79.7 tok/s. Free memory after the engine fit is 4,371 MiB on the 9B at 8K.

## Artifacts

- [`2b-512.log`](2b-512.log), [`2b-8192.log`](2b-8192.log),
  [`9b-512.log`](9b-512.log), [`9b-8192.log`](9b-8192.log): the four measured
  runs, including commands and stock output.
- [`2b-512-unmodified-stock-failure.log`](2b-512-unmodified-stock-failure.log):
  the unmodified-stock loader failure, not a benchmark result.
- [`run_stock.py`](run_stock.py): the loader shim used for the runs.

Logs are verbatim except that local Python and pack prefixes were replaced
with `<python>` and `~/.kindi`; timings and engine lines are unchanged.

## Limits

This is one pass per model/context, on one card and one pack set, with the
stock plain engine and exact head. It does not measure tensor cores,
speculation, server/cache behavior, token exactness, longer contexts, or
repeated-run variance. Do not compare these rows to the fork's `tc` and
`tc-spec` tables as if they were the same configuration.
