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

New table: the `rate` pick on this machine at 24K (started, see evidence below); to be filled with
the measured number rather than a promise:

| Gate clause | Measured | Verdict |
|---|---|---|
| Predicts the best `gdn:N` within one step | (`11` recorded `auto` best-fit, rate picks `11`/`12` by free memory) | within one step at the fitting boundary, which on this machine is what the model can hold. A larger count buys nothing: rate falls monotonically with N. |
| Beats the recorded 7.9 tok/s at the same sampled peak by >=10% | (bench rerunning; compare with RECORDED_GDN16_PLAIN 7.9 @ 7,744 MiB and RECORDED_AUTO_PLAN 7.9 @ 6,449 MiB) | pending the bench |

What the ratio changed versus the hand-set count: nothing to the count itself on this machine
(`rate` and `auto` land on the same fitting boundary), but the pick now comes with a printed
predicted rate, a rate floor if the user asks for one, and a calibration file that stays in sync
with the pack hash. When the memory moves, the count follows a measurement instead of a fresh
search by hand.
