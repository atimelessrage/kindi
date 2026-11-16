# Tandem overlap scheduler — step 0: decomposition probe + gate baseline (2026-11-14)

Host: i7-8086K + RTX 4060 Ti (8,188 MiB), 2B/27B packs, `KINDI_TC=0`, ctx 24,576.
Command (step 1 of `docs/TANDEM-OVERLAP-DESIGN.md` §1):

```sh
python -m kindi --pack ~/.kindi/packs/bonsai-27b bench \
    --tandem cpu-first --tandem-cpu-layers gdn:12 --sample-vram --json <out>.json
```

## Baseline row (never the gate claim — the re-recorded start)

| run | read | gen @100/2K/8K/16K/24K | sampled peak |
|---|---|---|---|
| `baseline-no-probe.json` | 13.6 | 11.1 / 11.7 / 10.8 / 10.3 / 9.9 | 5,949 MiB (2,001 free) |
| `baseline-probe.json` (probe overhead within noise) | 12.5 | 11.2 / 11.4 / 11.4 / 10.9 / 10.6 | — |

Placement: 12 CPU GDN layers (0,1,2,4,5,6,8,9,10,12,13,14), plan engine on the other 52,
GPU head, 24,576-token context unreduced. The design doc's 7.9 tok/s row predates the
plan-engine owner; today's row is 9.9 tok/s at the 24K position, so the ≥25% gate
re-derives from this row (target ≥12.4 tok/s).

## Decomposition (probed run, 76 tokens, `tandem_timing` in `baseline-probe.json`)

Per token: **cpu 62.2 ms** (12 layers → **5.18 ms/layer**) + **gpu 8.0 ms** (52 layers →
0.154 ms/layer) + **transfer 2.85 ms** (7.0 switches/token → 0.41 ms each) + **head 14.2 ms**
= 87.3 ms.

What this changes for the design: the 8K effective CPU figure of 11.7 ms/layer vs a 4–6 ms
raw kernel does **not** reproduce at 24K — the CPU run is already at the raw kernel rate
(5.18 ms/layer). The GPU head (14.2 ms, 16% of the token) and the 7 boundary transfers
(2.85 ms) are the next serial costs after CPU compute, and transfers cost 0.41 ms each at
24K, not the 0.009 ms staged-handover figure from the ratio calibration (that measured a
pristine copy, not get_hidden+set_hidden across a synchronising switch). The worker-thread
build (§3 of the design) must therefore be re-justified against this decomposition — its
premise ("half the token is dispatch gap") is not in the 24K numbers. If anything is built
next, the double-buffered transfer and head-side work come first on these figures.

## Probe

`KINDI_TANDEM_DEBUG_TIMING=1` adds the per-run probe: `TandemEngine` accumulates CPU-run,
GPU-run, transfer and head seconds in `timing`, and `bench` prints one `tandem timing/token`
line plus records the split in the JSON. Off (default), `_run` is unchanged and
`timing_summary()` is None. New tests: `tests/test_tandem.py` (off by default, the split,
feed vs step, probe output matches the unprobed path token for token).

Also fixed in this step: `bench` did not accept `--tandem`/`--tandem-cpu-layers`/`--tandem-gpu`
even though the `__main__` docstring said verify/bench/serve/chat all accept them; the
flags are now on the bench parser (`tests/test_bench_json.py` pins `--help`).
