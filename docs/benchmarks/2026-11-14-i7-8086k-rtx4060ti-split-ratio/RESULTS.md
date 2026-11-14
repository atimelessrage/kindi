---
tags: [benchmark, tandem, split-ratio, cpu-pcie]
---

# The CPU-vs-PCIe split ratio and the `rate` planner

Roadmap prior-art item 1 (from tandem.cpp): replace the hand-set CPU layer count with a choice
between the CPU hot path and the PCIe bus, measured on this machine, so the tandem engine picks
its split by rate. `tools/measure_split_ratio.py` measures it; `kindi/tandem.py` consumes it
through `--tandem-cpu-layers rate[:frac]`.

The calibration JSON in this directory (`split-ratio.json`, also keyed by pack back into
`~/.cache/kindi/tandem/`) was measured live 2026-11-14 on the i7-8086K + RTX 4060 Ti 8 GB
(CUDA 13.2, CuPy 13.6.0, driver 566.36) with the 27B pack (`53107f530aa5...`). PCIe copes are
pageable-staged (the same path `cp.asarray` uses inside the engine), CPU times are the engine's
own kernels at AVX2 (6 threads), and the split probe runs each N in a fresh subprocess (CuPy
keeps allocations, so two engines in one process died with a VRAM error in early tests; the probe
uses an 8-bit... `.q8_0` plan owner).

## Measured (this machine, 27B)

| Quantity | Value | Notes |
|---|---|---|
| Host->device staged (256 MiB) | 4.49 GiB/s (4.06 on the re-run) | PCIe 3.0 x16 slot; pageable-staged, not pinned-native |
| Device->host staged (256 MiB) | 2.25 GiB/s (1.99) | same path. D2H is the slower direction as on most CPU-side launches |
| Hidden handover (5120 f32, two-way) | 0.0085-0.0094 ms | negligible beside the per-layer cost. The boundary is never the problem |
| Raw CPU per GDN layer | **4.0-6.2 ms** (median 4.355, range 3.96-6.03; second session: 6.15 [5.80, 7.54]) | `kindi/cpu.py` `_gdn_scan`, 6 threads AVX2 |
| Raw CPU per full-attention layer @ pos 0 | 3.7-5.8 ms | plus 0.176 ms/K-position scan term |
| GPU per layer (24K) | 0.5174 ms | 33.1 ms token / 64 layers |
| **Effective CPU GDN layer (tandem pipeline)** | **11.7 ms** | slope over the N-sweep in a live `TandemEngine` step: 91.5 (N=11), 135.0 (13), 125.1 (16), 201.5 (20) ms token |

The ratio, then: one CPU GDN layer costs **~22x** one GPU layer (11.7/0.517). The device
boundary is unmeasurable in comparison (0.009 ms), and the raw CPU kernel (4-6 ms) covers only
about half of what the split actually costs per layer: per-layer pickle/dispatch overhead makes up
the rest. So the split is worth it only when the layer set cannot fit; moving more layers off the
GPU is a rate decision the machine answers as **the smallest fitting count**, and the ratio table
is what lets the caller see that tradeoff with a number.

### Prediction table (as written by the tool)

The tool predicts tok/s = 1/token-time for every surveyed N at 8K and 24K with the *effective*
cost. At 24K: N=0 → 26.8/..., N=11 → 6.2, N=16 → 4.6. Predicted best-fit = 11 CPU layers
(~6,000+ MiB of GPU side). Validation against the recorded 8K CPU-first sweep errs by **−10% ...
−25%** (the model is systematically a bit pessimistic; recorded runs span the same band).

## The `rate` pick at 27B@24K and the gate verdict

`python -m kindi --pack T:/kindi-packs/bonsai-27b --ctx 24576 bench --tandem cpu-first --tandem-cpu-layers rate --sample-vram` (2026-11-14, run at git `cdfc1b2`, log `bench-logs/split-ratio/27b-24k-rate.log` - gitignored evidence... the tracked bench JSON is `bench-logs/split-ratio/27b-24k-rate.json`):

12 CPU layers (`0,1,2,4,5,6,8,9,10,12,13,14`) by the fastest fitting split with the 512 MiB
reserve (1,226 MiB free after load, 24,576 unreduced, plan owner). Read 8.4 tok/s; generation
7.2/7.1/7.7/8.0/**8.0** tok/s at 100/2K/8K/16K/**24K**; sampled VRAM peak **6,151 MiB**.

| Gate clause | Measured | Verdict |
|---|---|---|
| Predicts the best `gdn:N` within one step | `rate` picked **12** CPU layers (smallest fitting count with the reserve at the free VRAM the session found); the recorded `auto` best-fit is **11** | **met**. Within one step of the recorded best-fit boundary, which on this machine is what the model can hold. The earlier hand-set `gdn:16` was 5 steps further and slightly slower. |
| Beats the recorded 7.9 tok/s at the same sampled peak by >=10% | 24K generation **8.0 tok/s at 6,151 MiB peak** vs recorded plain `gdn:16` 7.9 @ 7,744 and plan `auto` 7.9 @ 6,449 | **not met**: 8.0 is +1.3% over 7.9, not +10% (needs 8.7). Hmm, and the peak is 1,593 MiB (plain) / 298 MiB (plan `auto`) **lower** than the recorded sampled peaks - the split the ratio found is one CPU layer more, at a lower peak for the same rate. |

So the ratio delivers the pick (fewer steps than the hand search to the same boundary, one more
CPU layer for 298 MiB less peak) and the beat clause is the honest negative: on this machine the
smallest fitting count is the fastest possible, and 7.9 tok/s is what it is there. A repeat run
that lands 8.7+ would meet the letter of the clause (the recorded plan-auto row spans 4.5–9.3
tok/s in TANDEM-RUNTIME, so the check is noisy by construction), but the single honest measurement
is 8.0.

One note on prediction accuracy: in this run the planner printed a predicted 12.4 tok/s (from the
stale `nocard` cache file, written by the first buggy tool run); the live calibration file predicts
5.8 tok/s at N=12, next to the measured 8.0 (the model is systematically a bit pessimistic, like the
8K validation). The stale `nocard` file has since been fixed so the live measurement is preferred
(a live measurement whose `gpu` is non-null outranks a stale no-card file; same `pack_sha` match).

What the ratio changed versus the hand-set count: nothing to the count itself on this machine

What the ratio changed versus the hand-set count: nothing to the count itself on this machine
(`rate` and `auto` land on the same fitting boundary), but the pick now comes with a printed
predicted rate, a rate floor if the user asks for one, and a calibration file that stays in sync
with the pack hash. When the memory moves, the count follows a measurement instead of a fresh
search by hand.
