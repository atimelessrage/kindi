# Roadmap

Status: **living document, updated 2026-11-12.** The generated capability tables are in
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
| Tandem CPU+GPU layer split | **PARTIAL** 2026-11-11: dense runtime in both placements, token-exact; joint prompt-cache checkpoints pass on 2B/9B/27B; 27B larger-than-VRAM exactness + peak measured; 2B/4B/9B `cpu-first` throughput recorded | the original WIP `tandem.py` was corrupt; a clean dense runtime now replaces it. The larger-than-VRAM gate landed 2026-11-12: both 27B placements are token-exact 4/4 at 8,192 context (peaks 7,005 / 7,017 MiB vs 7,238 MiB full-GPU). 2B/4B/9B `cpu-first` CROSS-MODE throughput is measured; tandem does not beat GPU-only. Still open: a larger-than-VRAM (27B) throughput row with sampled peak, and the MoE approach (needs a real MoE backend first) | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [TANDEM-STATUS.md](TANDEM-STATUS.md), [MOE-TANDEM.md](MOE-TANDEM.md) | a larger-than-VRAM model runs token-exact with a recorded VRAM reduction; the MoE approach needs a real MoE backend first |
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
`gpu-first` placements. The real 2B is 4/4 prompts token-exact in both placements; a prepare-only run also measured
320/325 and 256/261 token reuse and a 19.27 MiB joint recurrent checkpoint (8.7-8.9 ms median
save, 3.3-3.5 ms median restore). Both 27B placements passed on 2026-11-12 (4/4 prompts;
peaks 7,005 / 7,017 MiB vs 7,238 MiB full-GPU) after the plain engine grew generic grouped-query
support for the 6:1 head ratio; 2B/4B/9B `cpu-first` throughput is recorded and tandem does not beat GPU-only. A larger-than-VRAM throughput row with sampled peak remains open. The MoE expert-cache path has a model-free slot/LRU/metrics prototype in
[MOE-TANDEM.md](MOE-TANDEM.md) and a [Gemma 4 26B adaptation](GEMMA4-26B-MOE.md), but no backend or GPU copy path.

**`k8v4`.** K stays q8_0 and only V drops to 4 bits, which is enough to change two of four 2B
prompts. It is refused unless `KINDI_KV_EXPERIMENTAL=1`, and the message names the measured loss.
q8_0 remains the default and its kernels and arithmetic are untouched.

## Live work

- **Formats:** `PQ2_0`/`Q2_0` import (2B done, byte-identical packs); 4B/9B/27B conversion and
  verification are open because those files were never downloaded. The native FP8 (E4M3/E5M2)
  prompt GEMM is implemented and synthetic-tested on Ada; a manual Blackwell sm_100/sm_120
  verification lane exists; the local gate returned `unsupported` on the Ada-only host and the
  workflow has not been dispatched, so a real FP8 model, a Blackwell result and a Blackwell
  throughput number remain open. The local stores checked on 2026-11-11 hold only
  `gemma4:26b` Q4_K_M and Qwen3.6-35B Q4_K_XL, so no real-model BF16/FP8 token-exactness run has
  happened.
- **CPU:** native Windows engine and CPU server chat requests for the 2B/4B/9B verified on the
  i7-8086K; the i5-2500K AVX-only fallback is implemented but native hardware is unverified, and
  the Xeon W-2133/i9-13900K plus a repeat timing run are open.
- **Plan engine:** 2B T=4 token-exact; 9B is refused before graph build (Hadamard block 4096).
- **Serving:** bounded queue with 429, optional per-source `KINDI_RATE_LIMIT_PER_MIN`, `/metrics`,
  Ollama-compatible endpoints; multi-GPU replicas need `kindi/tp.py` work.
- **Cards:** single/dual card support targets are in [CARD-SUPPORT.md](CARD-SUPPORT.md). The P106-100
  single-card (2B/4B/9B) and two-card (27B) offers are installer-tested but not measured on hardware.
  The dense CPU+GPU tandem runtime now exists and is 2B token-exact in both placements; the 27B split
  exactness/memory gate passed on 2026-11-12 (4/4, 40/40 tokens each; peaks 7,005/7,017 MiB vs
  7,238 MiB full-GPU), while a larger-than-VRAM throughput row with sampled peak remains open
  (2B/4B/9B `cpu-first` throughput is recorded). The MoE path has a model-free cache prototype
  but no backend
  ([MOE-TANDEM.md](MOE-TANDEM.md), [P106.md](P106.md)).
- **Intel Arc A770:** a separate SYCL/oneAPI backend is planned for one card, two cards and CPU+Arc
  tandem; the goal, gates and milestones are in [INTEL-ARC.md](INTEL-ARC.md). M0 (architecture audit)
  is done; M1 is blocked on Arc hardware. This is not a support claim.

## Long-term model goals

These are architecture goals, not support claims. A goal moves into live work only after its first
phase has a checksummed checkpoint, a license decision and a measured gate; nothing here is in the
catalog, the installer or the generated support matrix.

| goal | target | first gate | status |
|---|---|---|---|
| Raon-1 mini (MoE) | 28.8 B total / 4.3 B active parameters, 6 of 128 routed experts plus one shared expert, 18 RDA linear-attention + 7 gated MLA layers, 262,144-token context; the whole quantized model resident in host RAM with a GPU-resident expert cache on an 8 GB card | R0: tensor manifest, license decision and a reproducible reference command; then a host reference that is 4/4 greedy-exact at 1K-2K | **R0 artifacts complete; model-free cache prototype; reference goldens pending; no backend** — 19,434 base / 731 GGUF tensors, Apache-2.0 and the pinned reference are in [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md); architecture, R0-R7 phases and acceptance matrix in [RAON-1-MINI.md](RAON-1-MINI.md) |
| Gemma 4 26B-A4B (MoE) | 25.2 B total / 3.8 B active, 30 layers, 128 routed experts with top-8 plus a dense shared MLP, 704-wide experts, 262,144-token context; Q4 dense core and a per-layer expert cache on an 8 GB card | G0: 26B header plus Q5_1 support and a pinned BF16 or same-Q4 llama.cpp reference; then G1 host-resident exactness at 1K-2K | **design adaptation only; no backend, reference or support claim** — local Q4_K_M header measured; slots 3,717,120-4,336,640 B, 14.48 GB experts, 2.58 GB dense; 256K KV is 5.19 GiB and does not fit 8 GB with the dense core and reserve; [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md) |
| Dory1.0-4B (looped nGPT) | 4 B-parameter looped (latent-recurrent) model: 72 physical layers in 16 input + 40 shared recurrent + 16 output blocks, 27 sliding-window + 9 full-attention layers, 262,144-token context, and 1-4 recurrent loops chosen at inference; official BF16, MXFP8, NVFP4 and Q5_K_M GGUF exports | D0: tensor manifest plus a pinned llama.cpp/vLLM golden command; then D1: GGUF import 4/4 greedy-exact against pinned llama.cpp at k=1 | **D0 reference evidence complete; D1 pending** — manifest, hashes and k=1/2/4 goldens in [bench-logs/dory-d0/README.md](../bench-logs/dory-d0/README.md); architecture, memory/KV envelopes, D1-D7 phases and acceptance matrix in [DORY.md](DORY.md) |

## Review follow-ups (2026-11-11)

These are planned, not implemented. The review that produced them is an audit, not a support
claim; each item needs its own numeric or acceptance gate before it becomes a feature.

| item | why | first gate |
|---|---|---|
| Native FP8 tensor-core GEMM | FP8 Gemma 4 prompt batches dequantized every weight matrix to f32 and used cuBLAS; the FP8 tensor cores were unused | **done 2026-11-11:** E4M3/E5M2 `mma.sync` m16n8k32 kernels on sm_89+ with per-token activation scaling and a `KINDI_G4_FP8=0` opt-out; 20 focused tests pass, and synthetic Ada timings are 0.11-0.76 ms (15.0-35.5 TFLOPS, `bench-logs/fp8-mma-ada.log`); a manual Blackwell sm_100/sm_120 lane exists; a local gate returned unsupported (exit 2) on Ada and no Blackwell result exists; no real FP8 model, so not a model/support claim |
| Multi-ISA CPU dispatch | the current build is `-march=native` with an AVX floor; the i5-2500K still cannot use the AVX2/FMA/F16C kernels | **done 2026-11-11, 2B completed 2026-11-12:** five explicit tiers (`avx2-fma-f16c`, `avx2-f16c-no-fma`, `avx2-fma`, `avx2`, `avx-only`) are selected at startup or by `KINDI_CPU_ISA`; 2B verify is 4/4 token-exact on all five at 43.6/43.9/31.9/29.1/8.9 tok/s (forced tiers on an AVX2 host; evidence `bench-logs/cpu-isa-2b/RESULTS.md` and `bench-logs/cpu-dispatch-*.log`) |
| Tandem CPU+GPU runtime | 27B/33B models do not fit an 8 GB card; only the per-layer CPU primitive exists | an end-to-end `TandemEngine` beats GPU-only on one tier, with no token drift | the dense runtime is token-exact on 2B/9B, 27B exactness and peaks are measured, and 2B/4B/9B `cpu-first` 24K throughput benchmarks are recorded (2B read 80.9, generation 65.7/71.3/55.5/43.4/35.8; 4B read 62.5, generation 52.1/51.4/33.5/24.8/20.0; 9B read 39.9, generation 28.5/35.8/28.2/20.1/16.0 tok/s); tandem does **not** beat GPU-only, so the beat claim remains open, as does a larger-than-VRAM throughput row with sampled peak |
| Mixed-card pairing and asymmetric TP | installer offers mixed pairs; opt-in `KINDI_TP_SPLIT=auto` now gives each card a complete aligned layout (FFN and output head proportional on the 27B; GDN/attention stay equal where HB alignment leaves no choice), with `equal` remaining the compatible default; still no mixed-card hardware run and no `mixedx2` installer profile | equal-split mixed pair passes 27B verify/server/cache, then an asymmetric layout passes the same plus token-exact and peak-VRAM gates; see [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) |
| Self-hosted GPU CI runner | the GPU smoke workflow exists but has not been dispatched | one green `gpu-smoke` run on a registered runner |
| Static analysis and secret scanning | **done 2026-11-12:** secret scanning is wired into pre-commit and CI (`secret-scan` job + baseline), and the CUDA static-analysis harness (`tests/static/run_static_analysis.py`) runs in CI as the `static-analysis` job (cppcheck + clang-tidy-18, GPU-free CuPy stub) | a baseline with no new secret, plus a CI static-analysis job for the CUDA TUs |
| Coverage and performance regression gates | the CPU suite is broad, but coverage and tok/s have no enforced floor | **done 2026-11-12:** the CPU job enforces a 27% `kindi` line-and-branch coverage floor (27.15% measured locally; hardware/model paths are excluded from the claim, not the denominator), and `tools/check_perf_baseline.py` gates the recorded CROSS-MODE numbers (15% tolerance, 9 cells) in the same job. A same-runner GPU throughput regeneration remains the manual lane's job. |
| Windows install documentation | native CPU and GPU paths exist, but the installer is Linux-only | **done 2026-11-12:** [WINDOWS.md](WINDOWS.md) records the verified native CPU quickstart, clang/libomp setup, server commands, thread/context limits and the explicit non-claim for native Windows GPU/AMD. The installer remains Linux-only; no native Windows GPU run exists. |
| Structured request logging | the server logs free-form lines; request IDs and latency fields are not stable | **done 2026-11-11:** `log_json = 1` writes one JSON line per inference request with request id, status, duration, queue wait and token counts, and sets `X-Request-ID` |
| `AGENTS.md` trim and ownership index | the handoff file is large and mixes history with current instructions | **done 2026-11-12:** `AGENTS.md` opens with a live "Current state" section (HEAD, shipped work, open gates, shared-file ownership via the lease table); the dated Snapshot/Handoff history below it is labeled historical and the worktree HEAD is guarded by `tests/test_agents_handoff.py`. The full history trim remains future work |
| Dependency and GitHub Actions pinning | CI uses floating action tags and unpinned test dependencies | **done 2026-11-11:** workflow actions are pinned to commit SHAs (`actions/checkout@11d5960`, `actions/setup-python@a26af69`, `actions/upload-artifact@ea165f8`) and `.github/dependabot.yml` opens weekly reviewed PRs for the test requirements and those pins |
| Nanbeige 33B host-layer optimization | K6000 33B is 3.56 tok/s at 8K; `nanbeige.py` already overlaps 55% of host-layer FFN rows on the CPU, but copies 13 host layers into two device slots each pass; the proposal's resident 80/20 slices, CPU drafter, CPU K/V and output-head split are not implemented | a `KINDI_NB_CPUF`/slot sweep identifies the bottleneck, then a resident FFN slice beats 3.56 tok/s at 8K with token-exact output | [TANDEM-PROPOSAL-REVIEW.md](TANDEM-PROPOSAL-REVIEW.md) |

## Standing rules

- Every experiment gets a numeric gate before implementation, and the gate is written down before
  the measurement.
- A failed gate produces a no-go entry here and in the experiment's own page; it does not produce
  production code behind a hidden flag.
- Features that cannot be token-exact are opt-in, named as lossy, and pinned by a test.
