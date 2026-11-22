# 9B prompt-reading recheck: 20 interleaved passes with GPU telemetry (2026-11-18)

**Decision: closed as noise.** The 2026-11-17 five-pass 9B reading medians
(83.9 stock vs 72.4 fork) did not reproduce. Twenty fresh-process passes per
side give 70.7 stock vs 75.7 fork; the within-pair median difference is
**-1.4 tok/s** (stock minus fork), the exact sign test is **p = 0.648**, and the
paired bootstrap interval includes zero. No 9B reading difference is claimed.

## Why this run exists

The same-settings A/B in `CURRENT-VS-KINDI-0.7.md` §4.6 recorded the 9B
prompt-reading cell as unresolved: stock 83.9 (range 52.0-92.7) and fork 72.4
(56.2-89.7), five passes each. Those distributions overlap, but the medians were
11.5 tok/s apart. This run adds passes and measures the GPU clock and power
during the cell so the gap can be resolved or explicitly retired.

## Host and settings

Host: i7-8086K + NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42, native
Windows, Python 3.13.6, CuPy 14.2.0, CUDA runtime 12.9 (12090). Pack:
`~/.kindi/packs/bonsai-9b`, the same pack as the A/B.

Both sides use the same plain engine, exact head, `--ctx 8192`, and no
tensor-core, speculation, tandem or cache setting. Stock is v0.6.1 (`7fececb`)
with the native-Windows `libcuda.so.1` -> `nvcuda.dll` loader shim; the fork is
`549e0cf` on its plain path (`KINDI_TP=0`) with `--split-at off`.

```sh
# stock, from bench-logs/stock-v0.6.1
python run_stock.py --pack ~/.kindi/packs/bonsai-9b --ctx 8192 bench

# fork, from the worktree
KINDI_TP=0 python -m kindi --pack ~/.kindi/packs/bonsai-9b --ctx 8192 bench --split-at off
```

Each pass is a fresh process and a fresh engine load. The order alternates:
pass 1 stock then fork, pass 2 fork then stock, and so on. This balances the
observed warm-second-run effect across the two trees.

## Results

Medians of 20 passes, tok/s; the range is min-max.

| side | prompt reading | range | mean | SD | read-window SM clock (median) | read-window power (median) |
|---|---:|---:|---:|---:|---:|---:|
| stock v0.6.1 | **70.7** | 50.1-94.0 | 71.3 | 16.2 | 643 MHz | 12.5 W |
| this fork | **75.7** | 45.9-93.9 | 72.7 | 18.4 | 500 MHz | 12.1 W |

Paired comparison:

| statistic | value |
|---|---:|
| median within-pair difference, stock - fork | **-1.4 tok/s** |
| paired bootstrap 95% interval for that median | -16.65 to +10.80 tok/s |
| passes where stock was faster | 8 |
| passes where the fork was faster | 11 |
| ties | 1 |
| exact two-sided sign-test p | **0.648** |

The marginal medians (70.7 vs 75.7) and the paired median (-1.4) point in
opposite directions because the read cell swings with the clock/warm state; the
paired test is the one reported for the side comparison. Neither estimate is
distinguishable from zero at this spread.

## Clock and power evidence

A persistent `nvidia-smi` stream sampled every 50 ms around each pass. The
read window is the interval from the post-load `GPU memory` log line to the
`prompt reading` line; it spans roughly 0.4-0.6 s because the cell times 20
single-token feeds.

- Read-window clocks were mostly P5/P8: stock median 643 MHz (range 387-1,391),
  fork median 500 MHz (343-878).
- Read-window power was 10.3-20.4 W, median about 12 W on both sides; the card
  did not show a sustained boost clock during the read.
- Generation immediately afterward ran at its ~2.94 GHz boost clock and a median
  peak of 34 W, so the read cell is a short launch/ramp burst, not a steady-state
  prefill rate.
- The second run in each pair had a higher median (76.7) than the first (61.0),
  a 15.7 tok/s position effect larger than either tree's marginal difference.
  Alternating the order is what keeps that effect from being assigned to one
  tree; it also shows why a five-pass, fixed-order pass can manufacture a gap.

The clock/rate relation is not monotonic at this sampling interval, so the
telemetry bounds the card state and explains the instability; it does not
support a per-pass "clock caused this exact rate" model.

## Conclusion

The five-pass 83.9 vs 72.4 gap is retired as sampling noise. The 20-pass
medians are 70.7 and 75.7, the paired difference is -1.4 tok/s with an interval
spanning zero, and the sign test is 0.648. The read cell is too short and too
clock-dependent to support a stock-versus-fork reading claim; §4.6 keeps the
five-pass values as history and marks the 9B reading cell resolved to noise.
Decode results in §4.6 are unaffected.

## Limits

One card, one 9B pack, 20 pairs, no clock lock and no thermal chamber. The
read window is sub-second and sampled at 50 ms; clocks are observed, not
controlled. The stock side still runs through the loader shim. This page does
not measure tensor-core reading, prompt-cache/server prefill, token exactness or
any model other than the 9B.

## Artifacts

- `runs/stock-9b-read-r01..20.log` and `runs/fork-9b-read-r01..20.log`: the 40
  raw pass logs, including the prompt-reading line.
- `clocks/stock-9b-read-r01..20.clocks.csv` and
  `clocks/fork-9b-read-r01..20.clocks.csv`: 50 ms `nvidia-smi` samples with
  timestamp, SM/memory clock, power, temperature, P-state and utilization.
- [`summary.json`](summary.json): medians, ranges, per-run values, paired
  statistics and read-window telemetry.
- [`runs.json`](runs.json): the per-run records the summary was built from.
