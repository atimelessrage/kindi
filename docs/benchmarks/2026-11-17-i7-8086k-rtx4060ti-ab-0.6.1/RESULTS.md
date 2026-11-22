# Same-settings A/B: official Kindi v0.6.1 vs this fork (2026-11-17)

**What this is.** The same `kindi bench` cell run on the same pack and card with
two code trees: stock upstream `v0.6.1` (tag commit
`7fececbff13adc88bf21b8f07dea4e1c91996a9d`, extracted with
`git archive v0.6.1`) and this fork's worktree at `a79137f`. It closes the "run
`v0.6.1` and this tree on the same pack, flags and card" row that §4.3 of
`CURRENT-VS-KINDI-0.7.md` carried as unverified.

Host: i7-8086K + NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42, native
Windows, Python 3.13.6, CuPy 14.2.0, CUDA runtime 12.9 (12090). Packs:
`~/.kindi/packs/bonsai-2b` (source `Ternary-Bonsai-2-2B-PTQ1_0.gguf`, sha256
`36061ad4…`) and `~/.kindi/packs/bonsai-9b` (source
`Ternary-Bonsai-2-9B-PTQ1_0.gguf`, sha256 `3561df7a…`), the same packs the
earlier stock page used.

## Settings on both sides

- `--ctx 8192`, one pass = the bench's own warmup and 10 averaged steps per
  position.
- Plain engine, exact head, no speculation, no tensor-core, tandem or cache
  setting.
- Stock has no plan engine and its `kindi bench` is the plain path; the fork was
  run with `KINDI_TP=0`, its plain path, so both sides compare the same engine.
  The fork's `--split-at` default would add whole-request timings that stock does
  not have, so the fork ran with `--split-at off`.
- Stock still needs the native-Windows loader shim: `kindi/chunk1.py` hard-codes
  `ctypes.CDLL("libcuda.so.1")`, and `run_stock.py` maps that one name to
  `nvcuda.dll` before importing stock Kindi. It changes no engine, plan or kernel
  decision. The unmodified failure is recorded on the earlier stock page.
- No `KINDI_*` variable was set on the stock side.

Five interleaved passes per cell: stock 2B, fork 2B, stock 9B, fork 9B, repeated
five times. The logs are the evidence; `summarize.py` parses them into `ab.json`
(run values, median, min and max per cell), and rerunning it re-derives every
figure below.

## Commands

```sh
# stock side, from bench-logs/stock-v0.6.1 (git archive v0.6.1 + run_stock.py)
python run_stock.py --pack ~/.kindi/packs/bonsai-2b --ctx 8192 bench
python run_stock.py --pack ~/.kindi/packs/bonsai-9b --ctx 8192 bench

# fork side, from the worktree
KINDI_TP=0 python -m kindi --pack ~/.kindi/packs/bonsai-2b --ctx 8192 bench --split-at off
KINDI_TP=0 python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 8192 bench --split-at off
```

## Results

Medians of five passes, tok/s; the range is the five-pass min-max.

| model | engine | prompt reading | gen @100 | @2,000 | @8,000 |
|---|---|---:|---:|---:|---:|
| 2B | stock v0.6.1 | 215.5 (203.6-222.2) | 180.5 (159.4-183.4) | 182.9 (162.7-183.4) | 176.4 (148.2-182.1) |
| 2B | this fork | 211.1 (188.6-225.0) | 178.8 (159.6-180.8) | 179.9 (177.3-181.1) | 178.6 (174.0-180.9) |
| 9B | stock v0.6.1 | 83.9 (52.0-92.7) | 83.8 (62.6-83.9) | 81.3 (77.0-82.6) | 77.4 (73.2-77.7) |
| 9B | this fork | 72.4 (56.2-89.7) | 81.7 (80.2-83.0) | 81.5 (78.2-82.2) | 77.2 (76.6-77.8) |

Free memory after the fit was 6,411 MiB (stock) vs 6,409 MiB (fork) on the 2B
and 4,371 vs 4,369 MiB on the 9B; the 2 MiB difference is allocator noise.

## What the numbers say

- **Decode is the same path on both trees at this granularity.** Every decode
  median is within 2.6% of stock's and the five-pass ranges overlap. The widest
  decode median gap is the 2B at position 100 (180.5 vs 178.8); the sign flips at
  position 8,000 (176.4 vs 178.6), which is what run-to-run drift looks like.
- **Prompt reading is not resolved.** The 2B medians are 215.5 and 211.1 with
  overlapping ranges. The 9B medians are 83.9 and 72.4, but the same five passes
  range from 52.0 to 92.7 on stock and 56.2 to 89.7 on the fork, so the medians
  sit inside overlapping distributions; this run establishes neither a
  regression nor parity for the 9B reading cell.
- There is one visible outlier per side in the raw runs (stock 2B gen@100 159.4,
  fork 2B gen@100 159.6, stock 9B gen@100 62.6); medians and ranges are reported
  instead of claiming a best run.

## Limits

This is one card, one pack per size, five passes, plain engine and exact head.
It does **not** measure the fork's plan engine, tensor cores, speculation,
tandem, prompt cache or server, nor token exactness against llama.cpp. Stock
numbers come from the tag's source through the native-Windows loader shim, not
from the published release tarball or a container image. GPU clocks and Windows
WDDM scheduling are uncontrolled, which is why the ranges are reported and why
no single-pass number is quoted as the result.

## Artifacts

- `runs/stock-{2b,9b}-8192-r{1..5}.log`: the ten stock passes, each with its
  command, the GPU/engine lines and the bench cells.
- `runs/fork-{2b,9b}-8192-r{1..5}.log` and `.json`: the ten fork passes; the JSON
  is the fork's own `--json` record of the same cells.
- [`summarize.py`](summarize.py): parses the logs and writes
  [`ab.json`](ab.json) (per-run values, median, min, max).
- `ab.json`: the machine-readable summary the report's claim reads.

Logs are verbatim except that the local Python prefix became `<python>` and the
pack prefix became `~/.kindi`.
