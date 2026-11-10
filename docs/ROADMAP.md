# Roadmap

Status: **living document, updated 2026-11-10.** The generated capability tables are in
[SUPPORT-MATRIX.md](SUPPORT-MATRIX.md); this page is the single index of what was tried and
rejected, why it was rejected, and what evidence would reopen it. The detailed documents stay as
appendices and are linked from their rows.

A no-go is a result, not a deleted branch: each experiment below keeps its measurement, so the
next person does not have to rediscover the wall.

## Archived no-go experiments

| experiment | verdict | gate that failed | evidence | reopen when |
|---|---|---|---|---|
| Continuous batching (multi-sequence decode) | **NO-GO** 2026-11-08 | Phase-0 spike: B=4 reached 131.8 tok/s = 0.69x of single decode; the gate was 1.5x | [CONTINUOUS-BATCHING.md](CONTINUOUS-BATCHING.md), phase-0 spike logs in `bench-logs/phase0-spike/` | a new batch-GEMM spike clears 1.5x aggregate at B=4 on the 2B |
| Per-slot plain-engine refactor (Phase 1 of the batching plan) | **NO-GO/archived** 2026-11-08 | inherits the batching gate; the plan was never authorized | [PER-SLOT-REFACTOR.md](PER-SLOT-REFACTOR.md) | the batching gate passes |
| Tandem CPU+GPU layer split | **NO RUNTIME** 2026-11-09 | the WIP `tandem.py` does not parse; `main` has only the host-RAM preflight and the design | [TANDEM-STATUS.md](TANDEM-STATUS.md), [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) | a clean implementation of the restored design beats the GPU-only path on one tier |
| `k8v4` 4-bit V K/V cache | **QUARANTINED** 2026-11-10 | 2B `kindi verify` is 2/4 against the fp16 reference (4B is 4/4); a 4-bit V cannot be token-exact | [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md), `bench-logs/k8v4-cache/RESULTS.md` | the project accepts lossy K/V behind a flag, or a V format that reproduces the reference exists |

## Why each one stopped

**Continuous batching.** The measured wall is the chunk-GEMM path at small T: rows amortize
(3.90x relative to B=1) but the absolute B=4 rate is below one plain decode, so the scheduler work
would have made the server slower per token. The full 26-38 person-day plan is preserved; the
architecture is sound, the kernel economics on this card are not.

**Per-slot refactor.** It was the Phase-1 code plan for the batching MVP and is archived with it:
fixed slots for K/V, GDN state and conv state, with slot 0 left as the current single-sequence
state. Nothing was implemented, so there is no code to remove.

**Tandem CPU+GPU.** The interrupted worktree could not be merged: its `tandem.py` is unparsable,
the only historical blob is the corrupt one, and a branch merge would revert later audit, server
and test work. The intact design was restored, and the WIP branches are preserved (see the status
page for the cleanup gate).

**`k8v4`.** K stays q8_0 and only V drops to 4 bits, which is enough to change two of four 2B
prompts. It is refused unless `KINDI_KV_EXPERIMENTAL=1`, and the message names the measured loss.
q8_0 remains the default and its kernels and arithmetic are untouched.

## Live work

- **Formats:** `PQ2_0`/`Q2_0` import (2B done, byte-identical packs); 4B/9B/27B conversion and
  verification are open because those files were never downloaded.
- **CPU:** native Windows engine and a CPU server chat request verified on the i7-8086K; a repeat
  timing run and the other CPUs (i5-2500K AVX-only fallback, Xeon W-2133, i9-13900K) are open.
- **Plan engine:** 2B T=4 token-exact; 9B is refused before graph build (Hadamard block 4096).
- **Serving:** bounded queue with 429, `/metrics`, Ollama-compatible endpoints; multi-GPU
  replicas need `kindi/tp.py` work.
- **Cards:** single/dual card support targets are in [CARD-SUPPORT.md](CARD-SUPPORT.md).

## Standing rules

- Every experiment gets a numeric gate before implementation, and the gate is written down before
  the measurement.
- A failed gate produces a no-go entry here and in the experiment's own page; it does not produce
  production code behind a hidden flag.
- Features that cannot be token-exact are opt-in, named as lossy, and pinned by a test.
