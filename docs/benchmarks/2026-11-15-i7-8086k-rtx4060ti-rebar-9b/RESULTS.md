# RTX 4060 Ti, ReBAR on — 9B (2026-11-15)

Same host as the 2026-11-10 support-matrix run, after the user enabled
Resizable BAR in the BIOS. Purpose: measure what ReBAR gains on the 9B.

## Hardware state during the run

- GPU: NVIDIA GeForce RTX 4060 Ti, 8188 MiB, compute 8.9
- Driver 617.42 (note: nvidia-smi on this Windows box reports it as
  deprecated / will be removed in CUDA 14.0), CUDA 13.4, CuPy 14.2.0
- **ReBAR on:** BAR1 total 8192 MiB (full VRAM window; the classic
  window is 256 MiB)
- PCIe: slot is Gen3 (Host Max 3, Device Max 4 — the 4060 Ti is an
  x8 card), idle at Generation 1. Bandwidth ceiling is Gen3 x8.

## Run

`python tools/verify_matrix.py --models 9b --modes plain,tc --out
bench-logs/verify-matrix-rebar-9b` (evidence in bench-logs, gitignored).

| model | mode | status | tok/s | note |
|---|---|---|---|---|
| 9b | plain | pass 4/4 | 73.1 | bench row: 61.4 tok/s gen at ctx 24576 |
| 9b | tc | pass 4/4 | 114.7 | Hadamard-4096 high-precision activations (`KINDI_TC_HP`) |

## Comparison against the pre-ReBAR record (2026-11-10)

| | pre-ReBAR (2026-11-10) | ReBAR on (2026-11-15) | delta |
|---|---|---|---|
| 9B plain | 79.4 tok/s | 73.1 tok/s | −7.9% (cross-day) |
| 9B tc | refused | 114.7 tok/s | +114.7 (software, not ReBAR) |

Attribution, honestly:

- **The tc pass is NOT a ReBAR gain.** The 2026-11-10 refusal was
  `the tensor-core plan engine supports Hadamard blocks of 512 and
  1024 (this pack has 4096)`. The Hadamard-4096 high-precision
  activation path (`KINDI_TC_HP=1`, "measured 4/4 at 122 tok/s") has
  since landed, so the 9B TC cell passes on its own merit.
- **Plain went down 7.9%.** That is a cross-day, cross-driver
  comparison (the 2026-11-10 driver/BIOS were different; run-to-run
  variance and verify-cell noise are ±few %). For a fully resident
  2 GB model on an 8 GB card, ReBAR should be ~neutral: it enlarges
  the host→VRAM window (256 MiB → full VRAM), which matters for
  page-faulted / host-spill traffic — the plain engine is resident.
- **A single-day A/B was not done** (toggling ReBAR needs a BIOS
  round-trip + reboot). The exact command to close it when the user
  toggles: the same `tools/verify_matrix.py --models 9b` invocation,
  then compare `bench-logs/verify-matrix-rebar-9b/results.json`
  cell-for-cell.

ReBAR's real leverage on this rig is the *spill* paths: the 27B with
host spill (shared host layers / queue), MoE expert traffic, and
future large-context runs that fault host pages into VRAM over PCIe.
Those are worth measuring next: 27B plain + tc at 24K ctx against
the pre-ReBAR step-0 baselines (gen 8.2–8.4 tok/s at 24K,
head 17.6 ms/token).
