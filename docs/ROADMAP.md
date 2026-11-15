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
| Tandem CPU+GPU layer split | **PARTIAL** 2026-11-11: dense runtime in both placements, token-exact; joint prompt-cache checkpoints pass on 2B/9B/27B; 27B exactness, rate and sampled peak measured; 9B/27B saved-opening reloads exact at 4K/8K; 2B/4B/9B `cpu-first` throughput (medians of three) and the 2B/27B layer boundaries recorded | the original WIP `tandem.py` was corrupt; a clean dense runtime now replaces it. Both 27B placements are token-exact 4/4 at 8,192 context (peaks 7,005 / 7,017 MiB vs 7,238 MiB full-GPU). On 2026-11-13 the 27B rate plus sampled peak was measured at 8K and 24K against a full-GPU baseline (7.2-8.4 vs 34.8 tok/s at 100; 446 MiB less peak at 8K), the joint checkpoint cost (54.5-61.5 ms save, 27.0-28.5 ms restore, 149.6 MiB), the 2B layer knee (4-6 layers) and the 27B boundary (24K needs >=16 CPU layers; 24-32 layers lower the peak to 5.1-6.1 GiB at a 35-50% rate cost; the 8K sweep 8-28 layers in steps of 4 shows a flat exchange of 428-432 MiB peak saved per step for 15-37% decode cost per step, no knee, read 16.8 -> 6.5). Capacity is settled on this card: requesting 65,536/131,072 tokens auto-reduces to 37,120 (full GPU) and 30,464 (tandem) — the tandem is always the more reduced engine, so it buys headroom, not context. Tandem does not beat GPU-only. **2026-11-14, placement:** `--tandem-cpu-layers` now takes a count, an explicit layer set (`0-2,4-6`), `gdn:N` (the first N gated-delta layers, attention stays on the GPU) or `auto[:MiB]`, which computes the smallest GDN-first set that makes the context fit from the pack's own tensor sizes and the free VRAM, and refuses instead of splitting when the model already fits (27B at 24K: 19 layers, context unreduced, 522 MiB free, peak 7,257 MiB; 9B at 24K: refused with '~3282 MiB needed, 7075 MiB free'). The CPU owns layers that are cheap per token, which flattens long-context decode: 9B at 24K 23.6 vs 11.0 tok/s and 27B at 24K 7.9 vs 2.9 (medians of three, same layer count). Two limits are now measured and named: the 9B split saves no peak VRAM at 8K on this card (2.3 GiB pack, ~5 GiB free) so the right answer there is no split, and a split frees far less VRAM than its layer count implies because its GPU owner is the plain engine (the 27B plain engine cannot load full at 24K at all, while the plan engine can). **2026-11-14, plan-engine GPU owner:** the GPU half can now be the plan engine (`--tandem-gpu auto|plan|plain`, default auto): `TPEngine(layers=...)` loads only the owned tensors, allocates q8_0 K/V only for the owned attention layers, reads the f32 file per tensor and steps one prebuilt plan per block with the residual add resolved inside it (`k_padd`). Same 27B placement at 24,576 (`gdn:16`): peak **6,004 MiB** (5,862-6,083, n=5) and 1,985 MiB free after load against **7,721 MiB**/197 MiB for the plain owner, with decode inside this machine's run-to-run noise; the `auto` planner charges the owner's real allocation and predicts 6,548 MiB against 6,444-6,756 MiB measured. The capacity premise is now met on context: a 49,152-token request keeps all 49,152 tokens with 19 GDN layers (peak 6,876 MiB, 1,240 MiB free, 5.3 read, 4.9/4.9/2.8/2.7/2.0 tok/s) where the unsplit plan engine reduces it to 30,976 (122 MiB free). Exactness re-checked with the plan owner: 2B both head placements and 9B `gdn:6`, 4/4 each. Later the same day the plain engine gained the same q8_0 K/V cache (`KINDI_KV=q8_0`; the tandem's plain owner always uses it): token-exact against the fp16 references on the 2B, 9B and 27B, and the 27B 24K `gdn:16` peak falls from 7,721 MiB (fp16) to ~7,110 MiB (995 MiB free; byte accounting expects ~225 MiB of that), with the plan owner re-run the same day at 6,091 MiB (1,985 MiB free). Still open: tandem beats GPU-only (the CPU-owned layers dominate the token), a split across two GPU cards, and the MoE approach (needs a real MoE backend first) | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [TANDEM-STATUS.md](TANDEM-STATUS.md), [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md), [plan-owner results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md), [plain-q8 results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plain-q8/RESULTS.md), [MOE-TANDEM.md](MOE-TANDEM.md) | a larger-than-VRAM model runs token-exact with a recorded VRAM reduction; the MoE approach needs a real MoE backend first |
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
support for the 6:1 head ratio; 2B/4B/9B/27B `cpu-first` throughput and sampled peaks are recorded
(2026-11-13) and tandem does not beat GPU-only. On the 4060 Ti the full-GPU 27B still loads at
24,576 context, so a genuinely larger-than-VRAM row needs a longer context or a smaller card. The
9B and 27B joint caches are exact at 4K/8K in both placements, including saved-opening reloads. The
MoE expert-cache path has a model-free slot/LRU/metrics prototype in
[MOE-TANDEM.md](MOE-TANDEM.md) and a [Gemma 4 26B adaptation](GEMMA4-26B-MOE.md), but no backend or GPU copy path: **design-only until a backend exists**.

**`k8v4`.** K stays q8_0 and only V drops to 4 bits, which is enough to change two of four 2B
prompts. It is refused unless `KINDI_KV_EXPERIMENTAL=1`, and the message names the measured loss.
q8_0 remains the default and its kernels and arithmetic are untouched.

## Live work

- **Formats:** `PQ2_0`/`Q2_0` import (2B done, byte-identical packs); 4B PTQ1_0 conversion
  provenance-verified 2026-11-15 (reconvert from the downloaded GGUF → byte-identical weights,
  converted pack verifies 3+tie exactly like the existing pack; `bench-logs/convert-4b-verify/RESULTS.md`);
  9B/27B pack verify re-run the same day (9B 4/4, 27B 4/4 token-exact). Remaining: the 4B/9B/27B
  `PQ2_0`/`Q2_0` files were never downloaded and Hugging Face is unreachable from this host, so the
  142/42 import claim stays 2B-only. The native FP8 (E4M3/E5M2)
  prompt GEMM is implemented and synthetic-tested on Ada; a manual Blackwell sm_100/sm_120
  verification lane exists; the local gate returned `unsupported` on the Ada-only host and the
  workflow has not been dispatched, so a real FP8 model, a Blackwell result and a Blackwell
  throughput number remain open. The local stores checked on 2026-11-11 hold only
  `gemma4:26b` Q4_K_M and Qwen3.6-35B Q4_K_XL, so no real-model BF16/FP8 token-exactness run has
  happened.
- **CPU:** native Windows engine and CPU server chat requests for the 2B/4B/9B verified on the
  i7-8086K; the i5-2500K AVX-only fallback is implemented but native hardware is unverified, and
  the Xeon W-2133/i9-13900K plus a repeat timing run are open.
- **Plan engine:** 2B T=4 token-exact; 9B is 4/4 token-exact with `KINDI_TC_HP=1` (122.7 tok/s vs 76.3–80.0 plain) — the kernels already dispatched the 4096 folds, so `make_engine` now selects HP for HB-4096 packs (`KINDI_TC_HP=0` still refuses; with the default int8 activations one prompt flips at token 36, 3/4: isolated to activation precision by toggling fold-split/FF/FD/ATT_TC/TMAX, all unchanged). The plan engine now has `tc` rows in the cross-mode matrix (medians of three at 24K: 2B read 471.7, 4B 214.1, 9B 133.9; generation at 24K 299.2/127.8/97.0 tok/s, 1.5–1.9x the GPU-only engine) and its cells are guarded by the perf gate. The verify matrix gained a `tc-hp` mode (2026-11-13; `tools/verify_matrix.py`) so the 9B exact cell is reproducible (PASS 4/4 at 123.9 tok/s). `make_engine` now uses the plan engine by default for every model it supports (2026-11-13; verified: 2B default verify is 4/4 at 253.4 tok/s via the plan engine); `KINDI_TP=0` keeps the plain engine. `bench --json` also records `ctx_requested` and `ctx_reduced` so a reduced run is never recorded as the requested context.
- **Serving:** bounded queue with 429, optional per-source `KINDI_RATE_LIMIT_PER_MIN`, `/metrics`,
  JSON access log, Ollama-compatible endpoints; multi-GPU replicas need `kindi/tp.py` work.
- **Backend plugin interface ([REQ-0019](development/requirements/REQ-0019.md), experimental):**
  T4.1 landed 2026-11-15 — the contract (`kindi/backends/base.py`), the registry
  (resolve by model id and by GGUF `general.architecture`, external backends via
  `KINDI_BACKEND_MODULES`) and a deterministic `  FakeBackend` for CPU-only tests, with
  the repo map in [docs/dev/REPO-MAP.md](dev/REPO-MAP.md). T4.3 landed 2026-11-15:
  `kindi backends [--json]` lists registered backends (including external ones from
  `KINDI_BACKEND_MODULES`) and [docs/BACKENDS.md](BACKENDS.md) is the authoring guide.
  The existing engines are
  not adapters yet (T4.2), so there is still no arbitrary-GGUF claim; the generic

  fallback stays decision-gated (llama.cpp proxy recommended, native generic
  backend design-only). The OpenAI error envelope and the conformance suite
  (`kindi/api/errors.py`, `tests/api/`) also landed as the T5.1 foundation for
  REQ-0018; the per-family tool-call parsers, scheduler, metrics T5.5 and
  structured output (T5.4) remain planned.
- **Agent API hardening (REQ-0018, proposed):** the next layer over the 0.6.0 base (token
  counting, sessions, priorities, progress comments): (0) unsupported
  `n > 1` and structured-output `response_format` are now rejected with an
  enveloped 400 (naming the field in `param`) instead of silently ignored
  (**landed 2026-11-15**, decision D2) — constrained decoding itself remains open; (1) grammar / JSON-schema constrained
  decoding — nothing exists beyond the Gemma 4 tool-call `CallGrammar` precedent, and the
  Gemma grammar is the starting point for a general constrained sampler; (2) a per-model
  tool-call conformance suite — **landed 2026-11-15**
  (`tests/test_tool_call_conformance.py`: Gemma 4 native, Nanbeige/Lightning XML+JSON,
  DeepSeek DSML truncated blocks; all parsers hold the contract, no leak bugs found);
  (3) real concurrent request handling — the server serves one active
  request under a bounded queue (REQ-0005), the continuous-batching design is NO-GO at its
  Phase-0 spike gate (0.69x, see [CONTINUOUS-BATCHING.md](CONTINUOUS-BATCHING.md)), and no
  concurrency claim ships until a new spike passes; (4) metrics are already shipped
  (`/metrics`, access log) and must grow with each new path. The gate is in REQ-0018: no
  drop-in-for-agent-frameworks claim until the conformance suite and a schema-constrained
  reply both pass. The goal is a backend an agent framework can rely on, not just a fast
  chat endpoint.
- **Cards:** single/dual card support targets are in [CARD-SUPPORT.md](CARD-SUPPORT.md). The P106-100
  single-card (2B/4B/9B) and two-card (27B) offers are installer-tested but not measured on hardware.
  Whether the CPU+GPU tandem helps pre-tensor-core cards — Pascal (GTX 1080, P106-100) and the
  FirePro/Quadro M4000 — is unmeasured and worth a dedicated pass one day: on a card without tensor
  cores the GPU side is slower per token, so the exchange rate of the split (VRAM freed per point of
  decode lost) may differ from the RTX 4060 Ti's, and the optimal placement (GDN-first vs
  contiguous, `cpu-first` vs `gpu-first`) may differ too. Measure per card and model (2B/9B; 27B if
  it loads): `kindi verify` 4/4 token-exact, sampled peak, read + decode at 100/2K/8K/24K, tandem
  against the same card's full-GPU run — see [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) for the commands
  and the cells to record.
  The dense CPU+GPU tandem runtime now exists and is token-exact in both placements on 2B, 9B and
  27B; the 27B split exactness/memory gate passed on 2026-11-12 (4/4, 40/40 tokens each; peaks
  7,005/7,017 MiB vs 7,238 MiB full-GPU), and its rate plus sampled peak is recorded at 8K and 24K
  (2026-11-13) against a full-GPU baseline. The full-GPU 27B still loads at 24K on this card, so a
  genuinely larger-than-VRAM case is open (2B/4B/9B/27B `cpu-first` throughput is recorded). The MoE
  path has a model-free cache prototype but no backend
  ([MOE-TANDEM.md](MOE-TANDEM.md), [P106.md](P106.md)).
- **Intel Arc A770 ([REQ-0020](development/requirements/REQ-0020.md), proposed):** a separate
  SYCL/oneAPI backend is planned for one card, two cards and CPU+Arc tandem; the goal, gates and
  milestones are in [INTEL-ARC.md](INTEL-ARC.md). M0 (architecture audit) is done. **One A770 16 GB
  was purchased 2026-11-15 and is in transit (not installed)**, so M1 (hardware qualification) is
  unblocked on arrival — the arrival checklist is in INTEL-ARC.md. The   card is registered in the support matrix as `not measured` and is not offered by the installer. This is not a support claim.
- **CPU + Intel integrated graphics ([REQ-0021](development/requirements/REQ-0021.md), proposed):**
  investigation in [INTEL-IGPU-TANDEM.md](INTEL-IGPU-TANDEM.md). The iGPU route is a retarget
  of the REQ-0020 SYCL backend (Xe-LP/Xe2-LPG, shared system memory, no PCIe boundary — the
  tandem transfer term becomes ~0 by construction), with gates G0-G3 and a ≥1.15x CPU-only
  decode gate for the tandem (a decode loss is a documented no-go). First target is built-in
  Arc; UHD/Iris Xe is the second tier; pre-11th-gen (incl. this host's UHD 630) is excluded.
  No backend code; matrix rows are `not measured`.

## Long-term model goals

These are architecture goals, not support claims. A goal moves into live work only after its first
phase has a checksummed checkpoint, a license decision and a measured gate; nothing here is in the
catalog, the installer or the generated support matrix.

| goal | target | first gate | status |
|---|---|---|---|
| Raon-1 mini (MoE) | 28.8 B total / 4.3 B active parameters, 6 of 128 routed experts plus one shared expert, 18 RDA linear-attention + 7 gated MLA layers, 262,144-token context; the whole quantized model resident in host RAM with a GPU-resident expert cache on an 8 GB card | R0: tensor manifest, license decision and a reproducible reference command; then a host reference that is 4/4 greedy-exact at 1K-2K | **R0 artifacts complete; model-free cache prototype; reference goldens pending; no backend** — 19,434 base / 731 GGUF tensors, Apache-2.0 and the pinned reference are in [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md); architecture, R0-R7 phases and acceptance matrix in [RAON-1-MINI.md](RAON-1-MINI.md) |
| Gemma 4 26B-A4B (MoE) | 25.2 B total / 3.8 B active, 30 layers, 128 routed experts with top-8 plus a dense shared MLP, 704-wide experts, 262,144-token context; Q4 dense core and a per-layer expert cache on an 8 GB card | G0: 26B header plus Q5_1 support and a pinned BF16 or same-Q4 llama.cpp reference; then G1 host-resident exactness at 1K-2K | **design adaptation only; no backend, reference or support claim** — local Q4_K_M header measured; slots 3,717,120-4,336,640 B, 14.48 GB experts, 2.58 GB dense; 256K KV is 5.19 GiB and does not fit 8 GB with the dense core and reserve; [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md) |
| Kindi tandem scheduler (tandem.cpp-grade overlap) | CPU+GPU overlap instead of capacity only: a CPU worker thread plus double-buffered pinned staging, a numerically measured CPU/PCIe split rule instead of a fixed byte budget, and an LRU hot-weight cache in RAM; the current layer split becomes "capacity mode" | beat the 27B at 24K tandem row (re-recorded 9.9 tok/s at the 24K position, same CPU layer budget) by >=25% with token-exact 4/4 (`kindi verify` TC=0 vs tandem vs full GPU for every tuning change) | **designed 2026-11-14; step 0 landed** — the decomposition probe (`KINDI_TANDEM_DEBUG_TIMING`, per-token CPU/GPU/transfer/head split) and the re-recorded baseline are in [tandem-overlap RESULTS](benchmarks/2026-11-14-i7-8086k-rtx4060ti-tandem-overlap/RESULTS.md): the CPU run at 24K is already at the raw kernel rate (5.18 ms/layer), so the 8K "~6 ms dispatch gap" premise does not hold at the gate row and the worker thread must be re-justified before it is built (head 14.2 ms and 7 boundary transfers 2.85 ms come first on these numbers) — **re-measured 2026-11-15** ([fresh RESULTS](benchmarks/2026-11-15-i7-8086k-rtx4060ti-tandem-overlap/RESULTS.md): cpu 74.7 ms/token = 6.2 ms/layer, head 17.6 ms, transfer 3.09 ms, 8.2-8.4 tok/s at 24K on the Windows native CPU build), so the worker thread is **not justified and should not be built** — the plan-engine owner (peak 6,004 MiB vs 7,721 MiB at the same 27B 24K placement) is the memory half; [TANDEM-OVERLAP-DESIGN.md](TANDEM-OVERLAP-DESIGN.md) drafts the speed half, its status line now carrying the step-0 verdict. The split rule now has an external reference: tandem.cpp measures CPU against PCIe and uses a 2.25 factor over Strata's 1.85 for +2.7% on one host, and its gains appear only on capacity-bound models (a fitting 35B-A3B gains 1.04x, a budget-bound one 1.4-1.9x), so the >=25% gate must be run on a capacity-bound row. Prior-art item 1 measured our own ratio first |
| Dory1.0-4B (looped nGPT) | 4 B-parameter looped (latent-recurrent) model: 72 physical layers in 16 input + 40 shared recurrent + 16 output blocks, 27 sliding-window + 9 full-attention layers, 262,144-token context, and 1-4 recurrent loops chosen at inference; official BF16, MXFP8, NVFP4 and Q5_K_M GGUF exports | D0: tensor manifest plus a pinned llama.cpp/vLLM golden command; then D1: GGUF import 4/4 greedy-exact against pinned llama.cpp at k=1 | **D0 reference evidence complete; D1 pending** — manifest, hashes and k=1/2/4 goldens in [benchmarks/dory-d0/README.md](benchmarks/dory-d0/README.md); architecture, memory/KV envelopes, D1-D7 phases and acceptance matrix in [DORY.md](DORY.md) |

## Prior-art follow-ups (2026-11-14)

From llama.cpp v1.0.0, Strata 0.2.13 and tandem.cpp (2026-11-11); evidence and the full learning list
in [EXTERNAL-ENGINES.md](EXTERNAL-ENGINES.md). Ordered by what they unblock, not by effort. Each is
planned only: no code exists for these yet, and each needs its gate measured before it is called done.

| # | item | source | why it is worth doing | first gate |
|---|---|---|---|---|
| 1 | Measured CPU/PCIe split rule replacing the hand-picked `gdn:N` | tandem.cpp 2026-11-11 | our `auto` planner only asks "does it fit"; the rate question (which layers the CPU should own for *speed*) is answered today by a hand-set 16. Their rule bench-measures CPU per-layer cost against PCIe upload cost and splits by that ratio | on the 4060 Ti + 8086K at 24K, a measured ratio predicts the best `gdn:N` within one step — **met 2026-11-14**: `--tandem-cpu-layers rate` predicts the smallest fitting count (12 CPU layers, plan owner), the same boundary as the recorded `auto` best-fit (11), instead of the hand-set 16 (gdn:16 is 5 steps further and slightly slower). Effective CPU GDN layer 11.7 ms vs the raw kernel's 4–6 ms, predicted rates validate −10%...−25% vs the recorded 8K sweep; PCIe staged handover 0.009 ms, never the constraint. Then beats the recorded 7.9 tok/s at the same sampled peak by >=10% — **not met 2026-11-14**: the live rate-pick bench measured 8.0 tok/s at 24K and 6,151 MiB sampled peak (298 MiB less than the plan `auto` row, 1,593 MiB less than plain `gdn:16`) - +1.3% over 7.9 at a lower peak, not +10%. Verdict in `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-split-ratio/RESULTS.md` |
| 2 | Calibration cache: persist the per-machine tuning once | tandem.cpp (`~/.cache/tandem/<key>.json`, `--calibrate`) | we re-timed 5-trit kernel shapes at every start and discarded the result (on the paths that time them: non-TC plans, HIP), and `KINDI_SPEC_K`/placement were set by hand. A keyed file (GPU, pack hash, engine build) turns that into: 5-trit configs, best draft count, measured CPU/PCIe ratio, chosen placement | **done 2026-11-14** (`kindi/calib.py` + `kindi calibrate`): `tp.g5_tuned` reads the keyed file before timing and stores after; the file also records the ratio snapshot, planner picks and draft_k as data (env/config still win). Parity **met** on the 4060 Ti: 27B `KINDI_TC=0` bench fresh-tuned vs cached 33.1 vs 33.2, 31.6 vs 31.6, 30.2 vs 30.2 tok/s at 8K/16K/24K (<=0.3%; second run: no tuning, 8 shapes from file). The >=20 s startup clause is **not met and not measurable here**: g5 tuning on NVIDIA is 0.2 s (6.7 vs 6.5 s build) and the TC production path has no g5 timing at all; the discard costs tens of seconds only on HIP (60+ configs x 24 tensors at start), so that half needs a HIP rig |
| 3 | `kindi kld`: a quality gate for every lossy format | tandem.cpp quality table | `k8v4`'s quarantine rested on "2 of 4 prompts differ", which cannot rank two lossy caches; q8_0 K/V and any FP8 candidate had no graded number at all. Teacher-forced KLD + same-top-token share + PPL ratio against the fp16 whole-model reference | **done 2026-11-14** (`kindi/kld.py`, `kindi kld`): on the 2B (4060 Ti 8 GB), q8_0 measures 2.07-2.74e-07 nats KLD, 100% same-top, PPL ratio 1.000002-1.00001 at 2K/8K (plain and plan engines identical, so the gate is not picking up engine noise), while `k8v4` measures 1.29e-04/2.27e-04 nats (~470x/~1,090x q8_0, +0.09%/+0.11% PPL) and grows 1.75x with context. The quarantine is re-derived from that: it stands because the drift is 2-3 orders above q8_0's, not because of the prompt count; q8_0 is distribution-lossless in practice. Evidence: `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-kld-2b/`. Gate 8 (FP8) must beat the q8_0 row here |
| 4 | Determinism gate (same prompt, same tokens, every run) | Strata 0.2.13 Arc fix | we claimed exactness against llama.cpp but never asserted that ten runs agree. Strata found a vendor matmul whose result changed per launch and replaced it with a fixed-order kernel; our 9B int8-activation flip is the same class of problem | **done 2026-11-14**: 10/10 identical greedy completions through one server process each on the 2B (`180e7392dce5`) and 9B (`d5b027f65ade`), exact head, speculation off, 4060 Ti; the flip does not reproduce under defaults here. The legitimate-breaker list (lossy caches, dynamic offload, speculation, sampling; a default-settings failure stays a bug) is in `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-determinism/RESULTS.md` |
| 5 | `kindi plan`: a dry-run placement report with reasons | tandem.cpp launcher | `--tandem-cpu-layers auto` decides silently and can only be checked by loading a 27B; with `--assume-vram-gib/--assume-ram-gib/--assume-cores` the placement rules for the P106/M4000 (roadmap: Cards) become testable before the hardware exists | **done 2026-11-14** (`kindi/plan.py`, `kindi plan`): manifest + `index.json`/`bcodes_index.json` only (no weight tensor read, asserted by the fake-pack tests), prints model/context/hardware/owner/placement with one reason per choice plus the exact follow-up bench and verify commands, exit 0 even when nothing fits (the planner's refusal is reported, not raised). `--assume-vram-gib/--assume-ram-gib/--assume-cores` substitutes for the hardware probes. Agreement verified the same day on the 4060 Ti: live `plan` (12 layers, ~6,559 MiB, the list 0,1,2,4,5,6,8,9,10,12,13,14) == the `bench --tandem auto` run that followed it (same 12, same estimate); assume-mode at 6.0 GiB picks 21 layers. Also fixed the same parser class of bug as `bench`'s: `--tandem*`/`--vram-sample-s` were attached to the `kld` parser (`sub` never reassigned after `calibrate`); they are on `calibrate` now. The report's strategy is per model shape (2026-11-14): a `kindi_lightning` manifest gets the MoE expert-tier strategy (dense core on the card; experts tiered card / pinned host / mmap with last-used device slots), computed from the GGUF header with the engine's fit arithmetic and `kindi/moe.py` placement, `--usage` for a measured-use tier and `--moe-cache-mib`/`--dcache` for the planner-only budget/slot overrides. |
| 6 | K/V prefix sharing for parked conversations and saved openings | llama.cpp v1.0.0 refcounted checkpoints | our checkpoints are already state-only (K/V stays live), but parking a conversation each copied the whole K/V prefix: three agent sessions with the same 4K opening held three copies on the host | **mechanism done 2026-11-14** (`kindi/cache.py`): the shared prefix is parked as content-keyed blocks (the same key `Turns` uses for its disk blocks: hash of all tokens up to the block end, so equal keys mean identical rows), stored once and refcounted; only the sub-block tail and the end state are per-conversation. Test: three 4596-token siblings sharing a 4K opening hold four 1024-blocks (one copy, refcount 3) and restores are byte-identical. The WP5a <=1.2x host-RAM gate was **not met as written and is unsatisfiable jointly** with the 90% sibling-dedupe rule (surviving siblings need ≥456 post-opening tokens, and the per-conversation end state is not shared): written down as a shortfall, not retconned into a pass. |
| 7 | Thread count from the process CPU set, not `os.cpu_count()` | tandem.cpp | the CPU half of a split sizes itself from `os.cpu_count()`, which over-subscribes a 6c/12t host and an E-core one, and ignores container limits | **done 2026-11-14**: `kindi/cpu.py` `_configure_threads` defaults to one thread per physical core (`feature_report`), bound with `OMP_PROC_BIND=close`/`OMP_PLACES=cores`; `OMP_NUM_THREADS`/explicit values still win. Gate measured the same day (2B tandem, 8086K): at 6 threads read 104.6 tok/s vs at 12 threads 86.1 — the physical count is not just no-slower, it is faster; the tandem startup line logs the engine's thread count and `kindi plan` prints physical vs logical. |
| 8 | FP8 E4M3 K/V cache, decided by gate 3 | llama.cpp v1.0.0 `-ctk/-ctv f8_e4m3` | 8 bits per value against q8_0's 8.5 (6% smaller), wider dynamic range for outliers, hardware conversion on Ada+; we already ship FP8 kernels and the `KINDI_KV` plumbing; groundwork 2026-11-14: the host f32→FP8 reference codec (`f32_to_fp8`, round-to-nearest ties-to-even) is now in `kindi/dtypes.py` with all-pattern round-trip tests, tracked as [REQ-0014](development/requirements/REQ-0014.md) (experimental). **Implemented 2026-11-15 (decode-only):** `k_store_fp8` / `k_attn_sf8t` / `k_attn_vf8w` behind `KINDI_KV=fp8` (opt-in, experimental), prompt/chunk/tensor-core/batch paths disabled because their kernels are q8-only; the cache uses the q8 layout (8.5 bits/value), not a static-scale 8-bit layout. | behind `KINDI_KV=fp8`: KLD no worse than q8_0 at the same context, 4/4 exact against the fp16 reference at verify length on Ada, and the peak measured against q8_0 — **measured 2026-11-15 on the 2B (RTX 4060 Ti): NOT met.** KLD 2.48e-06 (2K) / 2.80e-06 (8K), ~9-14x the q8_0 rows (2.78e-07 / 2.06e-07), 100% same-top, PPL ratio 1.00006/1.00008 (k8v4 1.30e-04/2.26e-04); verify 3/4 (prompt 1 flips at token 30; q8_0 4/4); same cache size as q8_0 (8.5 bits/value), sampled peak 2,856 vs 2,895 MiB at 8K (saving is the disabled chunk buffers). Verdict: an experimental **lossy** opt-in, nicer than k8v4 and not lossless; do not use as a default. Evidence: gitignored `bench-logs/req0014-fp8-kld/kld.json`, [REQ-0014](development/requirements/REQ-0014.md) |
| 9 | Windows unbuffered pack load | llama.cpp v1.0.0 DirectIO parity | our `file_to_card` drops the page cache per chunk on Linux and does nothing on Windows, where a 5-7 GB pack load competes with the file cache | **measured 2026-11-14, gate NOT met**: `FILE_FLAG_NO_BUFFERING` + 64 KB-aligned chunk reads prototype (correct bytes verified against the buffered path) on the 27B bcodes (T: drive, 6.3 GiB): unbuffered 15.6 s vs buffered 14.5 s cold and 4.7 s cache-warm. NO_BUFFERING is not a speedup here — the disk read is the bottleneck either way (430 MB/s), and skipping the cache only forfeits the warm-reload benefit, at the cost of 64 KB-alignment plumbing in the hot load path. Keep the buffered path; a cache-cleanup knob would only help the multi-server footprint case, which is unmeasured. |

## Small-MoE compatibility (2026-11-14)

The six requested small MoEs below are recognized by `kindi/moe_compat.py` (and, for Cascade 2,
`kindi/cascade.py`) so `kindi convert` can refuse each with its actual missing milestone.
Recognition is not a backend: none of them enters the catalog, installer, or support matrix until
`kindi verify` passes token-exact against a pinned llama.cpp reference.

| model | arch | Kindi status | next milestone |
|---|---|---|---|
| Nemotron Cascade 2 30B-A3B | `nemotron_h_moe` | converter/validator only (`kindi_cascade`): header + exact F32/Q5_0/Q8_0/Q4_K dequant + pinned-reference gate (`kindi/cascade.py`, reference generator `tools/cascade_reference.py`) | GPU K-quant Nemotron-H matmul kernels so `kindi verify` can do token comparison against the pinned reference |
| Gemma 4 26B-A4B | `gemma4` MoE | recognized, refused: backend is E4B-only (42 layers, width 2560) | separate 26B backend + pinned reference (contract in [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md)) |
| Qwen3-Coder-30B-A3B | `qwen3moe` | recognized, refused | Qwen3-MoE loader + pinned reference |
| Qwen3.6-35B-A3B | `qwen35moe` | recognized, refused | DeltaNet/hybrid attention loader, MoE router and expert path + pinned reference |
| North Mini Code 1.0 (30B/~3B) | `cohere2moe` | recognized, refused | Cohere2-MoE loader, chat template and tokenizer checks + pinned reference |
| Recursive Flint 40B/2.5B | `flint` | recognized, refused | loader, router and expert path + pinned reference |

Cascade 2 goes first among these (it is closest to the existing `nemotron_h_moe` Lightning
backend): its G1 acceptance gate is a pinned reference on the exact GGUF plus a token-exact
`kindi verify` against it, with no catalog entry before that.

MoE design parity (not a build item): when a MoE backend lands, the expert cache should offer
llama.cpp's semantics — `moe_cache_size`, a CPU share for misses, LRFU/ClassLRU replacement, and
missed experts computed on the CPU — and be compared against `-cmoe --moe-cache-mib auto` on the same
model, RAM and card. See [MOE-TANDEM.md](MOE-TANDEM.md) and [EXTERNAL-ENGINES.md](EXTERNAL-ENGINES.md).

## Chat session history (2026-11-14)

`kindi chat` now records each run as a session under `state_dir/sessions/`
(a `--no-record` flag opts out). The store (`kindi/sessions.py`) supports
record/append, triage statuses (`open` → `needs-review` → `resolved`), tags
(slugified, deduped, filterable), and raw-markdown export/​import: the exact
message text verbatim, round-tripped through `kindi sessions
list|show|export|triage|tag|status|import` (host-only; no pack or GPU).
Tracked as [REQ-0016](development/requirements/REQ-0016.md) (experimental).
**2026-11-15: the server-side `/sessions` endpoint landed** (list with
status/tag filters, full record, raw-markdown export, triage status and tags
over HTTP, auth-gated, host-only), pinned by `tests/test_server_sessions.py`.
Concurrent mutations are serialized (one lock around the read-modify-write;
what used to race and lose updates). **Resume in the chat loop landed
2026-11-15:** `kindi chat --resume SESSION_ID` seeds the REPL with the
stored turns and appends new turns back to the same record on exit
(`SessionStore.append_messages`, pinned by `tests/test_sessions.py`).

## External suggestions review (2026-11-14)

Four outside suggestions were reviewed against the current tree. Two duplicate plans that
already exist; one is a real new direction; one is a long-term goal.

| suggestion | verdict | roadmap status |
|---|---|---|
| 1. Broader native architecture support (standard llama/qwen attention, sub-2-bit/ternary variants of any family, no custom conversion) | **Long-term goal, partly already in flight.** Kindi converts only `qwen35` (ternary Bonsai), `gemma4`, `llama` K/I-quant (Nanbeige profile), and `nemotron_h_moe`; a dense standard-attention llama/qwen GGUF is refused. The MoE half of this is the small-MoE table above (Cascade 2 first); dense standard attention needs a non-GDN attention engine, which is a new backend, not a loader flag. First gate: recognition-only refusal for `llama`/`qwen3` dense files naming the missing engine, then a host reference that is 4/4 greedy-exact against a pinned llama.cpp for one dense size. | Long-term goals table; no REQ yet (the MoE REQs cover the sparse half) |
| 2. Multi-vendor heterogeneous scaling (NVIDIA + AMD in one TP ring, arbitrary PCIe tiers, dynamic load balance) | **Same-vendor half already planned; vendor mixing deferred.** [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) P1–P6 covers mixed-generation same-vendor pairs (feasibility, equal-split validation, `KINDI_TP_SPLIT=auto` layout done but GPU-unverified). Mixing NVIDIA and AMD in one tensor-parallel ring needs a cross-vendor exchange path (CUDA graphs do not cross), so it is a later phase behind P2, not a reopening of P1. | [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) phases P1–P6 |
| 3. Native Intel Arc (oneAPI/SYCL) backend | **Already fully planned; hardware purchased, in transit.** [INTEL-ARC.md](INTEL-ARC.md) M0 (audit) is done; one A770 16 GB was purchased 2026-11-15 and is on its way, so M1 (hardware qualification) can start on arrival. No code until M1. | [INTEL-ARC.md](INTEL-ARC.md); roadmap Live work; [REQ-0020](development/requirements/REQ-0020.md) |
| 4. Speculative decoding with arbitrary draft families, acceptance monitors, adaptive draft length | **New and feasible; 2B gate passed.** `kindi/draft_policy.py` (pure-python controller, `limit_k` clamp tested) adopted in `kindi/spec.py` behind `KINDI_DRAFT_POLICY=1`: it may only shorten the hand-tuned base k (never lengthen it), so the engine's per-card tuning stays primary. Measured on the RTX 4060 Ti, 2B `draft2b`, ctx 1024: policy-on 4/4 PASS at   368.4 tok/s vs fixed-k 318.2 tok/s (+15.8%, no regression). The 27B row passed 2026-11-15 once the pack was on the host (fixed-k median 40.1 vs policy-on 40.1 tok/s, exactness identical: 3 exact + same tie; evidence in gitignored `bench-logs/req0015-gate-27b/`). The generalized arbitrary-draft-GGUF harness stays a separate gate (a second draft engine, same exactness + speedup gate). | [REQ-0015](development/requirements/REQ-0015.md) (experimental) |

## Review follow-ups (2026-11-11)

These are planned, not implemented. The review that produced them is an audit, not a support
claim; each item needs its own numeric or acceptance gate before it becomes a feature.

| item | why | first gate |
|---|---|---|
| Native FP8 tensor-core GEMM | FP8 Gemma 4 prompt batches dequantized every weight matrix to f32 and used cuBLAS; the FP8 tensor cores were unused | **done 2026-11-11:** E4M3/E5M2 `mma.sync` m16n8k32 kernels on sm_89+ with per-token activation scaling and a `KINDI_G4_FP8=0` opt-out; 20 focused tests pass, and synthetic Ada timings are 0.11-0.76 ms (15.0-35.5 TFLOPS, `bench-logs/fp8-mma-ada.log`); a manual Blackwell sm_100/sm_120 lane exists; a local gate returned unsupported (exit 2) on Ada and no Blackwell result exists; no real FP8 model, so not a model/support claim |
| Multi-ISA CPU dispatch | the current build is `-march=native` with an AVX floor; the i5-2500K still cannot use the AVX2/FMA/F16C kernels | **done 2026-11-11, 2B completed 2026-11-12:** five explicit tiers (`avx2-fma-f16c`, `avx2-f16c-no-fma`, `avx2-fma`, `avx2`, `avx-only`) are selected at startup or by `KINDI_CPU_ISA`; 2B verify is 4/4 token-exact on all five at 43.6/43.9/31.9/29.1/8.9 tok/s (forced tiers on an AVX2 host; evidence `bench-logs/cpu-isa-2b/RESULTS.md` and `bench-logs/cpu-dispatch-*.log`) |
| Tandem CPU+GPU runtime | 27B/33B models do not fit an 8 GB card; only the per-layer CPU primitive exists | an end-to-end `TandemEngine` beats GPU-only on one tier, with no token drift | the dense runtime is token-exact on 2B/9B/27B; 2B/4B/9B `cpu-first` 24K throughput is recorded as medians of three (2B read 84.3, generation 79.8/72.3/58.8/43.2/34.9; 4B read 49.3, generation 47.0/43.8/31.7/23.7/16.7; 9B read 38.1, generation 37.5/34.7/25.4/20.5/15.2 tok/s), the 27B has a rate-plus-peak row at 8K/24K (7.2-8.4 vs 34.8 tok/s at 100; 446 MiB less peak at 8K), the joint checkpoint costs 54.5-61.5 ms to save, and the 2B CPU-layer knee is 4-6 layers; tandem does **not** beat GPU-only, and the full-GPU 27B still loads at 24K, so the beat claim and a true larger-than-VRAM row remain open |
| Mixed-card pairing and asymmetric TP | installer offers mixed pairs; the installer now sets `KINDI_TP_SPLIT=auto` for a pair of unequal-capacity cards (for example 6 GB + 8 GB), giving each card a complete aligned layout (FFN and output head proportional on the 27B; GDN/attention stay equal where HB alignment leaves no choice), while equal-capacity pairs keep `equal`; still no mixed-card hardware run and no `mixedx2` installer profile | equal-split mixed pair passes 27B verify/server/cache, then an asymmetric layout passes the same plus token-exact and peak-VRAM gates; see [REQ-0017](development/requirements/REQ-0017.md) and [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) |
| Self-hosted GPU CI runner | the GPU smoke workflow exists but has not been dispatched | one green `gpu-smoke` run on a registered runner |
| Static analysis and secret scanning | **done 2026-11-12:** secret scanning is wired into pre-commit and CI (`secret-scan` job + baseline), and the CUDA static-analysis harness (`tests/static/run_static_analysis.py`) runs in CI as the `static-analysis` job (cppcheck + clang-tidy-18, GPU-free CuPy stub) | a baseline with no new secret, plus a CI static-analysis job for the CUDA TUs |
| Sampled-peak VRAM in `kindi bench` | a larger-than-VRAM capacity claim needs the transient device peak, and the only measurement harness lived in `bench-logs/` | **done 2026-11-13:** `kindi bench --sample-vram` samples device-wide `nvidia-smi` memory on a background thread (`kindi/vram.py`) and logs the peak with its delta over the idle baseline and the phase it occurred in; `--vram-sample-s` sets the interval, and a host without `nvidia-smi` logs that and still finishes. `tests/test_vram_monitor.py` pins peak/delta, no-GPU and CLI-flag behavior. The 27B 8K tandem throughput row uses it. |
| Coverage and performance regression gates | the CPU suite is broad, but coverage and tok/s have no enforced floor | **done 2026-11-12:** the CPU job enforces a 27% `kindi` line-and-branch coverage floor (27.15% measured locally; hardware/model paths are excluded from the claim, not the denominator), and `tools/check_perf_baseline.py` gates the recorded CROSS-MODE numbers (15% tolerance, 9 cells) in the same job. A same-runner GPU throughput regeneration remains the manual lane's job. |
| Windows install documentation | native CPU and GPU paths exist, but the installer is Linux-only | **done 2026-11-12:** [WINDOWS.md](WINDOWS.md) records the verified native CPU quickstart, clang/libomp setup, server commands, thread/context limits and the explicit non-claim for native Windows GPU/AMD. The installer remains Linux-only; no native Windows GPU run exists. |
| Structured request logging | the server logs free-form lines; request IDs and latency fields are not stable | **done 2026-11-11:** `log_json = 1` writes one JSON line per inference request with request id, status, duration, queue wait and token counts, and sets `X-Request-ID` |
| `AGENTS.md` trim and ownership index | the handoff file is large and mixes history with current instructions | **done 2026-11-12, completed 2026-11-14:** `AGENTS.md` opens with a live "Current state" section (HEAD, shipped work, open gates, shared-file ownership) and the worktree HEAD is guarded by `tests/test_agents_handoff.py`; the dated Snapshot, Handoff, full lease table, old plan and resume protocol were moved verbatim to `docs/development/AGENTS-HISTORY.md`, leaving a ~215-line live handoff (was 1,057). The live Test/Definition-of-done sections now name the real gate (`pytest -q`, `project_memory --check`, `config_reference --check`, the perf baseline, `git diff --check`) |
| Dependency and GitHub Actions pinning | CI uses floating action tags and unpinned test dependencies | **done 2026-11-11:** workflow actions are pinned to commit SHAs (`actions/checkout@11d5960`, `actions/setup-python@a26af69`, `actions/upload-artifact@ea165f8`) and `.github/dependabot.yml` opens weekly reviewed PRs for the test requirements and those pins |
| Nanbeige 33B host-layer optimization | K6000 33B is 3.56 tok/s at 8K; `nanbeige.py` already overlaps 55% of host-layer FFN rows on the CPU, but copies 13 host layers into two device slots each pass; the proposal's resident 80/20 slices, CPU drafter, CPU K/V and output-head split are not implemented | a `KINDI_NB_CPUF`/slot sweep identifies the bottleneck, then a resident FFN slice beats 3.56 tok/s at 8K with token-exact output | [TANDEM-PROPOSAL-REVIEW.md](TANDEM-PROPOSAL-REVIEW.md) |

## Standing rules

- Every experiment gets a numeric gate before implementation, and the gate is written down before
  the measurement.
- A failed gate produces a no-go entry here and in the experiment's own page; it does not produce
  production code behind a hidden flag.
- Features that cannot be token-exact are opt-in, named as lossy, and pinned by a test.
