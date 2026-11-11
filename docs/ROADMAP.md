# Roadmap

Status: **living document, updated 2026-11-11.** The generated capability tables are in
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
| Tandem CPU+GPU layer split | **PARTIAL** 2026-11-11: dense 2B runtime in both placements, token-exact; no larger-than-VRAM or MoE run | the original WIP `tandem.py` was corrupt; a clean dense runtime now replaces it. The remaining gate is a 27B-class split with peak VRAM and exactness against the full-GPU reference | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [TANDEM-STATUS.md](TANDEM-STATUS.md) | a larger-than-VRAM model runs token-exact with a recorded VRAM reduction; the MoE approach needs a real MoE backend first |
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
page for the cleanup gate). On 2026-11-11 the CPU engine gained a per-layer entry point, the GPU
engine gained a selective layer range, and `TandemEngine` connected them in `cpu-first` and
`gpu-first` placements. The real 2B is 4/4 prompts token-exact in both placements; a 27B-class run
and the MoE expert-cache policy remain open ([TANDEM-RUNTIME.md](TANDEM-RUNTIME.md)).

**`k8v4`.** K stays q8_0 and only V drops to 4 bits, which is enough to change two of four 2B
prompts. It is refused unless `KINDI_KV_EXPERIMENTAL=1`, and the message names the measured loss.
q8_0 remains the default and its kernels and arithmetic are untouched.

## Live work

- **Formats:** `PQ2_0`/`Q2_0` import (2B done, byte-identical packs); 4B/9B/27B conversion and
  verification are open because those files were never downloaded.
- **CPU:** native Windows engine and CPU server chat requests for the 2B/4B/9B verified on the
  i7-8086K; a repeat timing run and the other CPUs (i5-2500K AVX-only fallback, Xeon W-2133, i9-13900K) are open.
- **Plan engine:** 2B T=4 token-exact; 9B is refused before graph build (Hadamard block 4096).
- **Serving:** bounded queue with 429, `/metrics`, Ollama-compatible endpoints; multi-GPU
  replicas need `kindi/tp.py` work.
- **Cards:** single/dual card support targets are in [CARD-SUPPORT.md](CARD-SUPPORT.md). The P106-100
  single-card (2B/4B/9B) and two-card (27B) offers are installer-tested but not measured on hardware.
  The dense CPU+GPU tandem runtime now exists and is 2B token-exact in both placements; the 27B split
  and the MoE policy are still open ([TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [P106.md](P106.md)).
- **Intel Arc A770:** a separate SYCL/oneAPI backend is planned for one card, two cards and CPU+Arc
  tandem; the goal, gates and milestones are in [INTEL-ARC.md](INTEL-ARC.md). M0 (architecture audit)
  is done; M1 is blocked on Arc hardware. This is not a support claim.

## Review follow-ups (2026-11-11)

These are planned, not implemented. The review that produced them is an audit, not a support
claim; each item needs its own numeric or acceptance gate before it becomes a feature.

| item | why | first gate |
|---|---|---|
| Multi-ISA CPU dispatch | the current build is `-march=native` with an AVX floor; the i5-2500K still cannot use the AVX2/FMA/F16C kernels | an AVX-only and an AVX2 build selected at runtime, with token-exact verify on both |
| Tandem CPU+GPU runtime | 27B/33B models do not fit an 8 GB card; only the per-layer CPU primitive exists | an end-to-end `TandemEngine` beats GPU-only on one tier, with no token drift |
| Asymmetric two-card tensor parallelism | two-card support assumes equal cards and the current host has one GPU | a mixed-VRAM layout passes the 27B verify and server/cache checks on two cards |
| Self-hosted GPU CI runner | the GPU smoke workflow exists but has not been dispatched | one green `gpu-smoke` run on a registered runner |
| Static analysis and secret scanning | secret scanning is now wired into pre-commit and CI; CUDA static analysis remains a manual tool | a baseline with no new secret, plus a CI static-analysis job for the CUDA TUs |
| Coverage and performance regression gates | the 230+ CPU suite is broad, but coverage and tok/s have no enforced floor | a documented threshold and a PR job that reports both |
| Windows install documentation | native CPU and GPU paths exist, but the installer is Linux-only | a tested Windows CPU/GPU quickstart with the exact commands and limits |
| Structured request logging | the server logs free-form lines; request IDs and latency fields are not stable | an opt-in JSON log line with request id, status, queue wait and tokens/s |
| `AGENTS.md` trim and ownership index | the handoff file is large and mixes history with current instructions | a shorter current-state section plus one owner per shared file |
| Dependency and GitHub Actions pinning | CI uses floating action tags and unpinned test dependencies | Dependabot or Renovate plus reviewed lock updates |

## Standing rules

- Every experiment gets a numeric gate before implementation, and the gate is written down before
  the measurement.
- A failed gate produces a no-go entry here and in the experiment's own page; it does not produce
  production code behind a hidden flag.
- Features that cannot be token-exact are opt-in, named as lossy, and pinned by a test.
