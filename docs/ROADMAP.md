# Roadmap

Status: **living document, updated 2026-11-17.** The generated capability tables are in
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
| Tandem CPU+GPU layer split | **PARTIAL** 2026-11-11: dense runtime in both placements, token-exact; joint prompt-cache checkpoints pass on 2B/9B/27B; 27B exactness, rate and sampled peak measured; 9B/27B saved-opening reloads exact at 4K/8K; 2B/4B/9B `cpu-first` throughput (medians of three) and the 2B/27B layer boundaries recorded | the original WIP `tandem.py` was corrupt; a clean dense runtime now replaces it. Both 27B placements are token-exact 4/4 at 8,192 context (peaks 7,005 / 7,017 MiB vs 7,238 MiB full-GPU). On 2026-11-13 the 27B rate plus sampled peak was measured at 8K and 24K against a full-GPU baseline (7.2-8.4 vs 34.8 tok/s at 100; 446 MiB less peak at 8K), the joint checkpoint cost (54.5-61.5 ms save, 27.0-28.5 ms restore, 149.6 MiB), the 2B layer knee (4-6 layers) and the 27B boundary (24K needs >=16 CPU layers; 24-32 layers lower the peak to 5.1-6.1 GiB at a 35-50% rate cost; the 8K sweep 8-28 layers in steps of 4 shows a flat exchange of 428-432 MiB peak saved per step for 15-37% decode cost per step, no knee, read 16.8 -> 6.5). Capacity is settled on this card: requesting 65,536/131,072 tokens auto-reduces to 37,120 (full GPU) and 30,464 (tandem) — the tandem is always the more reduced engine, so it buys headroom, not context. Tandem does not beat GPU-only. The 9B is measured as the control (2026-11-17, 8,192-131,072 on both placements): the full-GPU engine keeps every request (peak 5,149 MiB, 3,003 MiB free after loading 131,072) and the split saves only 486-488 MiB (`gdn:8`) or 872-874 MiB (`gdn:16`) at 2.8-3.0x / 4.5-4.6x decode cost, so where the pack fits the remaining gap is throughput, not context. **2026-11-14, placement:** `--tandem-cpu-layers` now takes a count, an explicit layer set (`0-2,4-6`), `gdn:N` (the first N gated-delta layers, attention stays on the GPU) or `auto[:MiB]`, which computes the smallest GDN-first set that makes the context fit from the pack's own tensor sizes and the free VRAM, and refuses instead of splitting when the model already fits (27B at 24K: 19 layers, context unreduced, 522 MiB free, peak 7,257 MiB; 9B at 24K: refused with '~3282 MiB needed, 7075 MiB free'). The CPU owns layers that are cheap per token, which flattens long-context decode: 9B at 24K 23.6 vs 11.0 tok/s and 27B at 24K 7.9 vs 2.9 (medians of three, same layer count). Two limits are now measured and named: the 9B split saves no peak VRAM at 8K on this card (2.3 GiB pack, ~5 GiB free) so the right answer there is no split, and a split frees far less VRAM than its layer count implies because its GPU owner is the plain engine (the 27B plain engine cannot load full at 24K at all, while the plan engine can). **2026-11-14, plan-engine GPU owner:** the GPU half can now be the plan engine (`--tandem-gpu auto|plan|plain`, default auto): `TPEngine(layers=...)` loads only the owned tensors, allocates q8_0 K/V only for the owned attention layers, reads the f32 file per tensor and steps one prebuilt plan per block with the residual add resolved inside it (`k_padd`). Same 27B placement at 24,576 (`gdn:16`): peak **6,004 MiB** (5,862-6,083, n=5) and 1,985 MiB free after load against **7,721 MiB**/197 MiB for the plain owner, with decode inside this machine's run-to-run noise; the `auto` planner charges the owner's real allocation and predicts 6,548 MiB against 6,444-6,756 MiB measured. The capacity premise is now met on context: a 49,152-token request keeps all 49,152 tokens with 19 GDN layers (peak 6,876 MiB, 1,240 MiB free, 5.3 read, 4.9/4.9/2.8/2.7/2.0 tok/s) where the unsplit plan engine reduces it to 30,976 (122 MiB free). Exactness re-checked with the plan owner: 2B both head placements and 9B `gdn:6`, 4/4 each. Later the same day the plain engine gained the same q8_0 K/V cache (`KINDI_KV=q8_0`; the tandem's plain owner always uses it): token-exact against the fp16 references on the 2B, 9B and 27B, and the 27B 24K `gdn:16` peak falls from 7,721 MiB (fp16) to ~7,110 MiB (995 MiB free; byte accounting expects ~225 MiB of that), with the plan owner re-run the same day at 6,091 MiB (1,985 MiB free). Still open: tandem beats GPU-only (the CPU-owned layers dominate the token), a split across two GPU cards, and the MoE approach itself (it needs a real MoE backend; since 2026-11-17 the dense split can no longer be built over a MoE manifest - `TandemEngine` refuses a non-dense family by name and `make_engine` explains and drops a `--tandem` flag on `kindi_lightning`/`kindi_dsv41` packs - so what remains there is the expert-tier runtime, not a layer split) | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [TANDEM-STATUS.md](TANDEM-STATUS.md), [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md), [plan-owner results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md), [plain-q8 results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plain-q8/RESULTS.md), [9B tandem results](benchmarks/2026-11-17-i7-8086k-rtx4060ti-tandem-9b/RESULTS.md), [MOE-TANDEM.md](MOE-TANDEM.md) | a larger-than-VRAM model runs token-exact with a recorded VRAM reduction; the MoE approach needs a real MoE backend first |
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
  The Bonsai engine is a registered adapter since the T4.2-bonsai slice landed
  2026-11-15 (dense Bonsai packs resolve through the registry in `make_engine`,
  valve-gated by `KINDI_LEGACY_ENGINE_PATH=1`; `kindi verify` 2B is 4/4 on both paths
  on the RTX 4060 Ti, and the gate was extended to 4B (3+tie both paths) and 9B
  (4/4 both paths) with adapter bench decode rows agreeing with the valve within ~1% (the read
  row differs by 1.4%)
  — evidence `docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-t42-bonsai-4b-9b/RESULTS.md`;
  the 27B gate closed 2026-11-16 (pack at `~/.kindi/packs/bonsai-27b`; its
  manifest `model` is the GGUF's placeholder "Hf", so resolution needed a
  `source_file` size fallback — verify 4/4 both paths, bench within ~1%,
  evidence `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-t42-bonsai-27b/RESULTS.md`),
  so the Bonsai family gate is complete for all four packs on this host.
  The Nanbeige adapter is ported (2026-11-16) but its GPU gate is blocked: the
  catalog's ahtx GGUF is corrupt as served by Hugging Face (owner action in
  `docs/MANUAL-HELP-NEEDED.md`); the Lightning adapter is ported 2026-11-16
  and its GPU gate is now closed on the RTX 4060 Ti with a clean catalog
  download (fresh pack; verify 5 exact + 7 ties, 0 fail; bench ctx 8192
  read 788.3–821.3 tok/s, write 165.96–167.41 tok/s; evidence
  `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`);
  the Gemma 4
  and DeepSeek V4.1 adapters are ported 2026-11-16 and CPU-tested with both
  GPU gates blocked (neither model is on this host; a DeepSeek V4.1 pack is
  hand-written, there is no converter), so all five families now have an
  adapter; there is still no arbitrary-GGUF claim; the generic fallback
  stays decision-gated (llama.cpp proxy recommended, native generic
  backend design-only). The OpenAI error envelope and the conformance suite
  (`kindi/api/errors.py`, `tests/api/`) also landed as the T5.1 foundation for
  REQ-0018; the per-family tool-call parsers, scheduler, metrics T5.5 and
  structured output (T5.4) landed for the dense and Gemma 4 paths (the
  T5.4 item in the Agent API bullet below) and now for Lightning/Nanbeige too
  (CPU-tested 2026-11-16; no GPU structured-output gate claimed for either).
- **Agent API hardening (REQ-0018, proposed):** the next layer over the 0.6.0 base (token
  counting, sessions, priorities, progress comments): (0) unsupported
  `n > 1` and structured-output `response_format` are now rejected with an
  enveloped 400 (naming the field in `param`) instead of silently ignored
  (**landed 2026-11-15**, decision D2); (1) grammar / JSON-schema constrained
  decoding — **landed 2026-11-15** for the dense path and later the same day for
  the **Gemma 4 path** (`Kindi._generate_gemma` + `_g4_json_token`; structured
  output takes over the reply, stop tokens only once the automaton is done): a
  schema-ordered JSON automaton (`kindi/jsongrammar.py`) narrows the sampler's
  candidate set to schema-legal tokens (`kindi/sampling.py` `allowed`), so
  `json_object`/`json_schema` replies parse against the schema by construction.
  The Lightning and Nanbeige paths accepted the same constraint 2026-11-16
  (their `{x}_gen` loops now advance the same `JsonGrammar`, skip a scripted
  illegal top logit, and stop only once the automaton is done; CPU-only fake
  engines — no GPU gate claimed), and arbitrary GBNF grammars **landed 2026-11-16**
  (`response_format` type `grammar`, `kindi/gbnf.py`; a recursive rule is refused at
  compile time by default, and `KINDI_GBNF_MAX_DEPTH` opts into a bounded unroll whose
  language is exact at the depths it allows — `json_schema` stays the unbounded
  recursive-JSON path); (2) a per-model
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
- **An Anthropic Messages endpoint ([REQ-0028](development/requirements/REQ-0028.md), proposed):**
  `kindi serve` speaks OpenAI shapes only, while Claude Code and several harnesses in the ranking
  expect `POST /v1/messages` (content blocks, `tool_use`/`tool_result`, the SSE event order). A
  translation adapter over the existing chat path would let those harnesses point at Kindi without a
  proxy; it changes no decoding, so the gate is an app-level test with a Claude-Code-shaped request
  plus one real harness run, not a benchmark. Proposed by the external knowledge-base scan of
  2026-11-17 (`docs/development/AI-INVENTORY-SCAN.md`), which is inspiration only.
- **Hardware identification and autotuning ([REQ-0022](development/requirements/REQ-0022.md),
  experimental):** `kindi caps` reports every card the driver sees (compute capability, total/free
  VRAM, PCI bus, UUID, driver) with pynvml and CuPy fallbacks, keeps the hardware fact table
  separate from the Kindi paths that capability enables, and reports the CPU half from the engine's
  own probe plus model name, core counts, L2/L3, AVX-512/AMX and the CPU-engine toolchain.
  `kindi autotune` turns that into a measurement: candidates derived from the hardware and the pack
  manifest (full card, two-card TP split, tandem auto/gdn:N, Lightning decode cache, CPU engine),
  one real `kindi bench --sample-vram` child run each, ranked by decode tok/s with a reduced-context
  run ranked below a full one and a failed run recorded with its own error. Measured on this host
  2026-11-17 (2B, ctx 8192): `gpu0` 97.4-97.8 tok/s decode / 111.7-113.0 read / 3,050 MiB peak,
  `tandem-auto` refused as unnecessary, `tandem-gdn:8` blocked by a missing compiler — all three
  reported honestly in [AUTOTUNE.md](AUTOTUNE.md). Open: two-card and MoE candidates are unit-tested
  only (one card here, no Lightning pack); exactness stays with `kindi verify`.
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

The small MoEs below are recognized by `kindi/moe_compat.py` (and, for Cascade 2,
`kindi/cascade.py`) so `kindi convert` can refuse each with its actual missing milestone.
Recognition is not a backend: none of them enters the catalog, installer, or support matrix until
`kindi verify` passes token-exact against a pinned llama.cpp reference. The four Ollama-fetched
candidates below were inspected in the local `~/.ollama/models` store (GGUF blobs linkable by
`tools/ollama_links.ps1`): Gemma 4 26B-A4B and Qwen3-Coder 30B-A3B are already recognized;
GLM-4.7-Flash is added as `glm4moelite`; Granite 4.1 30B is **not** an MoE (architecture
`granite`, 64 dense layers, no expert tensors) and is listed as a generic dense long-context
candidate, not as an MoE backend.

| model | arch | Kindi status | next milestone |
|---|---|---|---|
| Nemotron Cascade 2 30B-A3B | `nemotron_h_moe` | converter/validator only (`kindi_cascade`): header + exact F32/Q5_0/Q8_0/Q4_K dequant + pinned-reference gate (`kindi/cascade.py`, reference generator `tools/cascade_reference.py`) | GPU K-quant Nemotron-H matmul kernels so `kindi verify` can do token comparison against the pinned reference |
| Gemma 4 26B-A4B (MoE) | `gemma4` MoE | recognized, refused: backend is E4B-only (42 layers, width 2560) | separate 26B backend + pinned reference (contract in [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md)) |
| Qwen3-Coder-30B-A3B | `qwen3moe` | recognized, refused | Qwen3-MoE loader + pinned reference |
| Qwen 3.6-35B-A3B (True MoE) | `qwen35moe` | recognized, refused; 40 layers (`10 × (3 Gated DeltaNet → 1 Gated Full Attention)`), hidden 2048, 248,320 padded vocab, 262,144 native ctx; 256 experts/layer (top-8 + 1 shared, intermediate 512) | **Q0 pinned GGUF/reference gate: byte-pinned Bartowski Q4_K_M (22.29 GB, Apache-2.0) or Unsloth UD-Q4_K_M, tensor inventory over every tensor, llama.cpp Q0 reference run.** Then Q1 Gated DeltaNet state/recurrence (30 layers; 16 QK + 32 V heads, head_dim 128, SSM inner 4096, conv kernel 4, time-step rank 32), Q2 Gated Full Attention (10 layers at 3/7/…/39; 16 Q / 2 KV heads, head_dim 256, partial RoPE 64, attn/output gate, mRoPE `[11,11,10]`), Q3 MoE router + experts (`ffn_gate_inp` `(256,2048)` no bias/correction; fused `ffn_{gate,up,down}_exps` + shared `ffn_*_shexp`; ~32.2 B expert params), Q4 host-resident expert cache, Q5 tokenizer/chat/no-vision text-only `kindi verify` |
| GLM-4.7-Flash | `glm4moelite` | recognized, refused | GLM4-MoE-Lite loader, MLA/KV attention path, routed-plus-shared expert path + pinned reference |
| North Mini Code 1.0 (30B/~3B) | `cohere2moe` | recognized, refused | Cohere2-MoE loader, chat template and tokenizer checks + pinned reference |
| Recursive Flint 40B/2.5B | `flint` | recognized, refused | loader, router and expert path + pinned reference |

**Qwen 3.6 Q0 first gate (do not start the backend before this):** pin one text-only GGUF at a
repository revision, record its SHA-256 and Apache-2.0 license chain, and dump a tensor inventory
that proves the layer pattern (30 `blk.N.ssm_*` GDN blocks; no `ssm_*` on blocks 3, 7, 11, …, 39),
the full-attention tensors, and every `ffn_gate_inp` / `ffn_{gate,up,down}_exps` /
`ffn_*_shexp` tensor. The same artifact runs once under the pinned upstream llama.cpp
`qwen35moe` reference for a fixed 1K-2K greedy prompt; its logits/tokens are the gate every later
DeltaNet, attention and MoE milestone is compared against. Vision and MTP are explicitly out of
scope for this text-only backend. Do **not** use the Ollama-converted layout as the first pin:
it has been observed to differ (`ssm_dt` without `.bias`, `rope.dimension_sections` length 3 vs 4);
use a Bartowski/Unsloth llama.cpp-compatible GGUF, or normalize those two deltas in a Kindi
converter with byte-level tests first.

**Standard llama.cpp GGUF route for a 16 GB NVIDIA card (checked 2026-11-18):**
[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) publishes ordinary
llama.cpp files whose `general.architecture` is `qwen35moe`; upstream llama.cpp has a loader and
model implementation for it
([src/models/qwen35moe.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp)),
so this is the path a 16 GB card can run today, unlike the Apple-only ternary builds below. The
repository's published sizes are decimal GB and cover weights only; a 16 GiB card has at most 17.2
decimal GB of VRAM before the driver reserve, so context/KV and runtime overhead come out of the
difference.

| Unsloth file | published GB | decimal GB left on a 16 GiB card |
|---|---:|---:|
| `UD-IQ2_XXS` | 10.8 | 6.4 |
| `UD-IQ2_M` | 11.5 | 5.7 |
| `UD-Q2_K_XL` | 12.3 | 4.9 |
| `UD-IQ3_XXS` | 13.2 | 4.0 |
| `UD-IQ3_S` | 13.7 | 3.5 |
| `UD-Q3_K_S` | 15.4 | 1.8 |
| `UD-Q3_K_M` | 16.6 | 0.6 |
| `UD-Q3_K_XL` | 16.8 | 0.4 |

The IQ2 files leave 4.9-6.4 GB and the IQ3 files 3.5-4.0 GB for the GDN state, the ten
full-attention layers' K/V and the runtime reserve. `UD-Q3_K_S` leaves only 1.8 GB, and
`UD-Q3_K_M`/`UD-Q3_K_XL` are already 16.6/16.8 GB, so those two need partial CPU offload
(`-ngl` below every layer) and are not whole-GPU candidates. A starting point for the IQ2 route is
stock `llama-server -m Qwen3.6-35B-A3B-UD-IQ2_M.gguf -ngl 99 -c 8192` (or `UD-IQ3_S` for a
longer context budget check); the llama.cpp build must include `qwen35moe` support.

This is a llama.cpp route, not a Kindi one. Kindi still recognizes `qwen35moe` and refuses it
because it has no such backend, so none of these GGUFs can be served or verified by Kindi yet. The
still-missing gate is the backend itself: the Q0 pinned-GGUF/reference artifact and the GDN,
full-attention and MoE milestones in the table above, not another quantization. For the 16 GB
target, choose `UD-IQ2_M` (11.5 GB) as the byte-pinned artifact; the Q4_K_M file is an optional
quality reference, not a second required pin. Until that backend lands, the concrete 16 GB/NVIDIA recommendation
is stock llama.cpp with an IQ2 or IQ3 file, and the ternary community builds below remain
Q0-ineligible.

**Qwen3.6-35B-A3B ternary community builds (external, checked 2026-11-17):** the official model
uses the `qwen35moe` architecture, and these are real third-party quantizations, but they are not
Q0 artifacts. Both are listed in the [ternary-MoE census](#ternary-moe-census-2026-11-18), which
records their formats and why neither is a llama.cpp-compatible candidate yet.
[russellwmy/qwen3.6-35b-a3b-ternary](#ternary-moe-census-2026-11-18) stores 9.1 GB of 2-bit
ternary expert codes plus 2.3 GB of the rest and runs only through ironbark on Apple Silicon; its
card says it is not loadable by `transformers` or MLX-LM.
[manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64](#ternary-moe-census-2026-11-18) is 12.6 GB
of 3-bit attention, ternary gate/up experts and 4-bit down experts in the TurboQuant-MLX format,
also Apple Silicon only. Its card reports a separate 9.4 GB pure-ternary sibling and a quality
split on one agentic task (0/4 pure ternary, 3/3 asymmetric down4), which is a hypothesis about
asymmetric expert precision, not a Kindi measurement. Neither file replaces the pinned-GGUF Q0 gate or gives
this repository a 16 GB fit today. A later, separate experiment may test ternary gate/up plus a
higher-bit down projection, but only after a byte-pinned GGUF export (or a converter with
bit-level dequant tests) and a token-exact `kindi verify` plus the existing KLD/PPL quality gate.

### Ternary-MoE census (2026-11-18)

The two Qwen3.6-35B-A3B ternary MoE builds reviewed so far are quantizations of that model; neither
card's base-architecture label is asserted here for both. "llama.cpp-compatible candidate" below
means a file with a standard GGUF `general.architecture = qwen35moe` tensor layout that upstream
llama.cpp can load and that can be pinned for the Q0 reference gate; neither qualifies.

| build | contents | format and runtime | why it cannot count as a llama.cpp candidate yet |
|---|---|---|---|
| [russellwmy/qwen3.6-35b-a3b-ternary](https://huggingface.co/russellwmy/qwen3.6-35b-a3b-ternary) | 9.1 GB of 2-bit ternary expert codes plus 2.3 GB of the rest; a bundle, not a runnable checkpoint | ironbark pack for Apple Silicon; its card says it is not loadable by `transformers`, `mlx-lm` or `mlx-vlm` | custom safetensors bundle and ironbark trit-pack format, not GGUF; no `qwen35moe` tensor inventory; no byte-level dequant/export test here; no upstream llama.cpp loader or NVIDIA kernel path; no pinned llama.cpp reference or token-exact `kindi verify` |
| [manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64) | 12.6 GB: 3-bit attention, ternary gate/up experts and 4-bit down experts; its separate pure-ternary sibling is 9.4 GB | TurboQuant-MLX safetensors for Apple Silicon | not GGUF; the mixed 3-bit/ternary/4-bit expert layout needs a converter with bit-level dequant tests (none in this repository); no upstream llama.cpp loader or NVIDIA path; the 0/4 vs 3/3 quality split is one agentic task, not a token-exact reference |

Both Qwen3.6 builds stay out of the Q0 candidate set until one of two things exists: a byte-pinned
standard GGUF with the `qwen35moe` tensor inventory, or a converter whose dequant is tested
bit-for-bit. Either route then needs the same token-exact `kindi verify` against the pinned
llama.cpp reference and the existing KLD/PPL gate. The 16 GB route above uses the ordinary Unsloth
`UD-IQ2`/`UD-IQ3` GGUFs; neither Apple-only bundle changes that.

**Other ternary MoE builds with published evidence (checked 2026-11-18).** These are outside the
Qwen3.6 Q0 question, and none is a Kindi measurement. The Nemotron row differs in kind from the
rest: Kindi already ships that build, so it is not a lead.

| build | base and ternary scope | format and runtime | status here |
|---|---|---|---|
| [Nota AI Nemotron 3.5 Lightning 30B-A3B, GGUF by boro64](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `nemotron_h_moe`: 52 blocks, 128 routed experts (top-6) + one shared; Nota AI's ternary MLX checkpoint repacked bit-for-bit; 162 linear tensors at Q2_0, embeddings and lm_head at Q8_0, router/norms/SSM scalars F32 | 8.82 GB standard GGUF Q2_0; the card reports stock llama.cpp (b11555+/b11796) and the PrismML fork; 262K native | **Already served:** `kindi use lightning-30b` pins this file (`kindi/catalog.py`) and `kindi_lightning` reads Q2_0/Q8_0/F32 from it. It is Nota's intermediate checkpoint at ~89% of the BF16 teacher, not a Qwen3.6 Q0 artifact. |
| [DeepGrove Maple-Preview](https://deepgrove.ai/maple-preview) | 20B-A1B, natively trained ternary; 24 layers, 256 experts (top-8), 3:1 SWA-512/global attention; 5.31 GB 2-bit MLX deployment checkpoint and official TQ1_0/TQ2_0 GGUFs | MLX; GGUF TQ1_0/TQ2_0; merge-ready llama.cpp PR [#27000](https://github.com/ggml-org/llama.cpp/pull/27000) adds `maple` for CPU and reports the official GGUFs load there | no `maple` backend in Kindi; unlike the other rows, the weights were trained ternary from the start rather than post-quantized. |
| [Schackay3/Qwen3-Next-80B-A3B-Instruct-ternary-GGUF](https://huggingface.co/Schackay3/Qwen3-Next-80B-A3B-Instruct-ternary-GGUF) | `qwen3next`: 512 experts x 48 layers; 24,576 expert matrices ternary (asymmetric group-128), shared expert ternary; DeltaNet/attention, router, embeddings, norms and lm_head bf16 | 29.3 GB GGUF **Q2_K** (2.94 bpw), not a native 1.58-bit container; llama.cpp b10331 | 80B is far outside the 16 GB target; `qwen3next` is not `qwen35moe`, and the card itself says the file is Q2_K rather than true ternary. |
| [kabcpc/Ternary-Flint-40B-GGUF](https://huggingface.co/kabcpc/Ternary-Flint-40B-GGUF) | `flint`: 39.8B total / 2.5B active, 49 layers, 256 routed experts + one shared; 124 exps at PTQ1_0 and 17 sensitive down-projection exps at Q4_1 in the hybrid | 9.31-12.49 GiB GGUF PTQ1_0 (type 143) / PQ2_0 (type 142) / hybrids; production path is a `llama-flint` build with a Metal LUT; 262K | Kindi recognizes `flint` and refuses it (`kindi/moe_compat.py`); it needs its own loader and pinned reference. |
| [ToxeH/Qwen3-Coder-30B-A3B-Instruct-Ternary2x](https://huggingface.co/ToxeH/Qwen3-Coder-30B-A3B-Instruct-Ternary2x) | `qwen3_moe`: 128 experts (top-8); 2-plane ternary experts (3-bit MLX), 3-plane ternary attention/embeddings/lm_head (6-bit), router/norms bf16 | 12.99 GiB MLX safetensors; MLX engines on Apple Silicon | Kindi recognizes `qwen3moe` and refuses it; this is a multi-plane ternary decomposition, not a pure ternary checkpoint. |
| [micro-terse](https://github.com/michelangeloromerochisco/micro-terse) | 423M total / ~320M active, 12 layers, 4 experts (top-2) on odd layers; Q/K/V/O and gate/up/down ternary, embeddings/head/norms/router/biases full precision | 182 MB TQ2_0 GGUF (tied embedding Q6_K); custom `terse` llama.cpp fork; 4K context, CPU | research proof-of-concept; Kindi has no `terse` architecture. |

**Unread leads (no published artifact to read).** These are recorded so the census does not lose
them, but they are not counted as builds:

- **MoTE** ([arXiv:2506.14435](https://arxiv.org/abs/2506.14435), v2 2026-01-07): a method for
  training ternary routed experts up-cycled from a dense checkpoint; the experiments are multimodal
  (SigLIP-L vision encoder + Qwen2.5). No public checkpoint or model card was found in the
  2026-11-18 search, so there is nothing to inspect beyond the paper.
- **ButterflyMoE** ([arXiv:2601.13563](https://arxiv.org/abs/2601.13563)): a method paper on
  structured butterfly orbits for ternary experts. No released checkpoint was found; paper only.
- **Nota AI further-recovered Nemotron checkpoint:** the published Nota card says the further
  recovered model is available from the authors on request. There is no public file to read, so the
  89%-recovery intermediate above is the only Nota artifact counted.

**Checked and not counted:** [UlukaDev/bitnet-ternary-moe-gguf](https://huggingface.co/UlukaDev/bitnet-ternary-moe-gguf)
is a dense `microsoft/bitnet-b1.58-2B-4T` base plus two LoRA adapters and an external router, not a
single ternary MoE checkpoint. [stiger1000/TC-MoE](https://huggingface.co/stiger1000/TC-MoE) applies
`{-1, 0, 1}` to expert *choice*, not to expert weights. The Prism Ternary-Bonsai models (1.7B, 8B
and 27B) are dense, not MoE.

**Not an MoE (local Ollama candidate):** Granite 4.1 30B (`granite`, 64 blocks,
131,072 context) is a dense 30B GGUF, so it is not part of the small-MoE recognition table.
It is already on disk under `~/.ollama/models` and can be linked with `tools/ollama_links.ps1`
for a future native/generic-engine evaluation, but it needs a Granite loader and pinned reference
like any other dense architecture; Qwen3 30B (non-coder) shares the `qwen3moe` architecture and
therefore follows the Qwen3-Coder row once a loader exists.

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
list|show|export|triage|tag|status|import|prune|stats` (host-only; no pack or GPU).
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

## Evaluation ideas review (2026-11-17)

Fifteen measurement and documentation ideas were triaged against the current tree. Seven are
accepted below with a gate written first; six already exist as gates or evidence and are not
duplicated; two are not this repository's shape. The triage found two real gaps: a pack can
declare `vision` in its manifest and the server would accept image parts, but nothing writes or
implements that flag, and the bounded request queue has code but no numbers. The vision one is fixed
2026-11-17 (see the accepted table); the queue numbers are still open.

| idea | verdict | where it lands |
|---|---|---|
| 1. Independent exact-artifact benchmarks (SWE-bench / Terminal-Bench) per file | **Narrowed and accepted.** A published score is the model publisher's claim, not the engine's. The engine question is whether the *same file* answers a task the same way through Kindi and through a pinned llama.cpp once the turn is long enough for KV reuse, prefix caching and tool loops to matter. | Accepted as the task-level equivalence suite (below); exactness today is 4-token greedy probes from `kindi verify` |
| 2. Machine-specific throughput and fit on our cards | **Already tracked.** The Cards, CPU and Intel Arc bullets name every unmeasured card and the command set for it (P106-100 pair, M4000/FirePro, A770 in transit, i5-2500K, Xeon W-2133). Only the RTX 4060 Ti has [M] rows. | No new row |
| 3. Cross-engine exact match (Kindi vs llama.cpp) on the same file, card and context | **Already shipped.** `kindi verify` compares against pinned llama.cpp goldens and `tools/verify_matrix.py` reproduces them per mode. One residual: the run-to-run determinism test that [EXTERNAL-ENGINES.md](EXTERNAL-ENGINES.md) row 4 adopted was never written, so it rides along with item 1. | Folded into the equivalence suite |
| 4. Agent-loop failure-mode catalog from our own runs | **Accepted.** The repo has operational troubleshooting and per-format tool-call conformance, but no record of *which task types fail and why* (think-block exhaustion, stop-token miss, grammar refusal, a reduced context, a truncated tool call). | `docs/FAILURE-MODES.md` (below) |
| 5. KV-cache and context memory budget table per (file, card, context) | **Accepted.** The pieces exist (`kindi bench --sample-vram`, `ctx_requested`/`ctx_reduced`, `kindi plan`) and the numbers are scattered across RESULTS pages, so the table has to be generated, never hand-typed. This is the same trap the Windows run exposed: a fit check that reads the wrong free-space figure. | Generator + table (below) |
| 6. Complete Bonsai family cards (4B/9B) and a routing table across park machines | **Already covered, partly out of scope.** [BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md) already carries the per-size table (2B/4B/9B/27B, expert counts, context, tier) and the drafter split. A routing table across a fleet of "park" machines is not this repository's shape: one host, one support matrix. | No new row |
| 7. Vision support verification for Bolt E4B | **Accepted, inverted — this was a real gap, fixed 2026-11-17.** `kindi/server.py` accepted image parts when the pack manifest set `vision` or `mmproj`, but no converter writes the field and no projector runs, so the gate trusted a flag nobody could honestly set and would have answered about an image the engine never saw. The cheap correct half was honesty, not a projector: it is now always a 400 that names the missing projector for a vision-declaring pack. | Fixed in `kindi/server.py` + `tests/test_server.py`; a real projector is still its own REQ |
| 8. Quantization trade-offs (IQ3_XXS vs Q4_K_M, PTQ1_0 vs the installed GGUF) | **Accepted, download-bound.** The conversion provenance work already proves a reconversion is byte-identical, which is equivalence, not a trade-off. Comparing quants needs the alternative files, and Hugging Face is unreachable from this host. | Quant comparison table (below) |
| 9. Drafter (DSpine) speedup and quality on our cards | **Already measured.** [DSPINE.md](DSPINE.md) records 3.10 tokens per verify pass and 115.5 tok/s at context 32768 on an RTX 3080 at 190 W, and the [REQ-0015](development/requirements/REQ-0015.md) policy gate ran on the 4060 Ti. The residual is one row on this host, not a project. | No new row; add the 4060 Ti cell when the DSpine pack is next benchmarked |
| 10. License / redistribution audit for the custom quants | **Accepted.** Licence notes are scattered ([BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md) carries the Apache-2.0 vs AGPL-3.0 combined-work question, [DSPINE.md](DSPINE.md) names the drafter's licence), and the installer downloads packs automatically, so every downloadable artifact should have a row. | Pack licence table (below) |
| 11. CPU-engine feasibility on Machine 3/4 (the AVX2 gate) | **Already tracked.** Five ISA tiers are measured on this host and the `avx-only` fallback is implemented; what is missing is *hardware*, named in the CPU bullet (i5-2500K, Xeon W-2133, i9-13900K). | No new row |
| 12. Concurrency and multi-request behaviour per engine | **Accepted, small.** The bounded queue, the 429 with `Retry-After` and the rate limit exist; the *numbers* do not (where waiting starts to hurt). A queueing curve is what tells an agent framework whether to run tasks serially. Continuous batching stays NO-GO at its own measured 0.69x. | Queueing curve (below) |
| 13. Cold-start and operational cost (first-run weight writes, ~12 min for the 27B) | **Accepted, small.** The ~12 minutes is recorded as a fact with no breakdown, and nobody has recorded what the second run costs when the pack is already written. | Cold-start table (below) |
| 14. Multi-card scaling efficiency (2x RTX 3080 and friends) | **Already tracked.** [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) P1-P6 covers pairing, and equal-split scaling is P2's own gate; this host has no second card, which is why the rows are unmeasured rather than forgotten. | No new row |
| 15. Release-diff automation for changelogs | **Mostly covered.** `tools/project_memory.py --check`, the docs-truth and docs-links tests (a benchmarks RESULTS page must be linked from an index) and the perf baseline gate already catch the silent rot. A "stale number" audit would mean re-running each page's commands, which is the benchmark lane itself. | No new row |

Accepted items, gate first (the standing rule is that the gate is written down before the
measurement):

| item | first gate |
|---|---|
| Task-level equivalence suite | one recorded task set (a tool loop, a long-context turn past the KV reuse point, a repeat run for determinism) ends with the **same final answer** through Kindi and a pinned llama.cpp, same file, same card, same context; then the same suite run twice on Kindi must be byte-identical |
| `docs/FAILURE-MODES.md` | five entries taken from existing runs, each naming the run, the symptom, the cause and the check that catches it; any entry a test can catch becomes a test |
| KV / peak budget generator | one command emits the table (pack x card x context 100/2K/8K/24K: sampled peak, KV bytes, `ctx_reduced`), and `kindi plan` must refuse rather than trim when the requested context does not fit; the perf gate pins the table |
| Vision-gate honesty | **done 2026-11-17.** An image part (and Ollama `images`) is an enveloped 400 even when a manifest sets `vision`/`mmproj`, with the message naming the missing projector in that case; pinned by an app-level test that builds a vision-declaring pack and still gets the 400. No shipped pack declares the flag until a projector exists. |
| Quant comparison table | for one model, two quants on the same card: KLD against the reference logits, `kindi verify` exactness, and tok/s read + write; blocked on the files (Hugging Face is unreachable from this host) |
| Pack licence table | every pack the installer can download has a row: source repository, licence, redistribution allowed (yes/no/ask), conversion provenance; the AGPL-3.0 vs Apache-2.0 combined-work question stays open in the same table |
| Queueing curve | 2/4/8/16 concurrent clients against one server: wait p50/p95, tokens/s, and the client count where 429 starts; states the serial-only line honestly |
| Cold-start table | pack directory wiped, then 2B/9B/27B first load (pack write + load) and second load from the written pack, sampled peak alongside; the 27B row explains the recorded ~12 minutes |

## Memory-tier follow-ups (2026-11-17)

Drawn from the user-supplied comparison with Strata, KTransformers and BigMoeOnEdge, checked against
the tree and triaged in [EXTERNAL-ENGINES.md](EXTERNAL-ENGINES.md). None of their numbers is ours;
each row names the gate that would make it a Kindi result. `kindi io` (this round) is the measurement
half: RAM copy, unbuffered disk read and pinned PCIe copies on this host, so a slot or streaming
budget no longer has to be assumed. The generic-GGUF MoE item is not repeated here because T4.4 and
the small-MoE rows already own it, and the Anthropic request surface is REQ-0028.

| # | item | direction | gate | status |
|---|---|---|---|---|
| 1 | NVMe cold-expert tier | keep the dense core and hot experts as today; stream a routed expert that is neither in a card slot nor in the pinned host pool from the pack file, unbuffered and through a bounded lookahead queue; report flash MiB/token and the compute/I/O split | a Lightning pack larger than RAM: `kindi verify` 4/4, decode tok/s better than the same model's recorded mmap-fault path, and no decode fault on a warm slot; run at n=1, because expert streaming and speculation/batching conflict (BigMoeOnEdge's own limitation) | **started 2026-11-17, gate open.** `KINDI_LT_COLD=1` keeps the hottest host experts pinned under `KINDI_LT_HOST_MB` (default: available host RAM less 10% or 1 GiB, whichever is larger) and makes only the overflow file-backed: the split uses measured pick counts when available, prompts stage the host block through `kindi/cold_experts.py`, decode fetches only routed experts not already in a warm slot, and the decode CUDA graph is disabled because the read happens after the router. The reader is unbuffered where the OS allows it, coalesces neighbouring ranges into one aligned read, keeps a bounded lookahead queue for the next cold layer's most-used experts, and reports requested vs transferred bytes plus exact pinned/disk tier bytes (`Engine.cold_stats()`); `kindi plan` applies the same split to its planned host set and reports pinned versus file-backed counts and bytes under the same budget. Eighteen CPU tests cover the flag, the host/disk split and per-layer byte accounting, the staged host block's layout and read spans, exact bytes, coalescing, the prefetch queue and its next-layer selection, order, warm-slot skips and short reads; the plan has a separate cold-tier report test. **No larger-than-RAM pack, `kindi verify` run, GPU run or tok/s is claimed; those gate clauses are not measured.** |
| 2 | CPU-compute lane for misses | a miss is computed on the CPU from host-resident weights while the GPU computes the hits; only the expert's output vector crosses PCIe | bit-identical expert output between the CPU and GPU kernel for the quant format (M2 in [MOE-TANDEM.md](MOE-TANDEM.md)), then a capacity-bound MoE row beats the copy lane's tok/s with `kindi verify` 4/4 | **parked on M2.** `kindi/moe_tandem.py` already schedules `card`/`copy`/`cpu` lanes with a deterministic `cpu_share`; there are no CPU expert kernels and no backend wiring |
| 3 | I/O-aware slot and cache sizing | use the measured RAM/PCIe/disk rates, the usage file and the room to choose the per-layer slot count, and the RAM cache size for a streamed model; print the reason per choice | predicted slots/cache within 10% of the measured best for a Lightning pack on this host, and a plan whose numbers a live `bench` reproduces | **arithmetic implemented 2026-11-17, gate open.** `ltfit.rate_slots` predicts `n_used * (1 + ceil((RAM + disk) / PCIe))` slots, capped by `n_exp` and the fit; `kindi plan` prints the three rates and uses that count, its follow-up `bench` command carries the planned context and count with the global `--ctx` and portable `--dcache` flags so the child engine does not fall back to defaults, and `kindi autotune` tests it with half/double candidates plus an `expert-cache-<MiB>` cold-tier budget (`KINDI_LT_COLD=1` + `KINDI_LT_HOST_MB`) sized by the same rate model and capped by the host experts. Both commands can reuse a saved `kindi io --json FILE` report via `--io-report`, falling back to one probe when it is missing or unusable. Unit-tested only: no Lightning pack on this host, so the within-10% and live-bench clauses are **not measured** |
| 4 | Mixed-precision tiers | keep fast-unpacking or higher-precision experts in VRAM and compact 2-3 bit/ternary ones in RAM or on disk | bytes/token for the same served share, plus exactness where the format is exact and a KLD tolerance where it is not | **parked.** `MoESpec` already takes per-layer expert bytes and the Q2_0/PQ2_0/FP8 codecs exist, but both tiers use the same format today |

## Engine-liveness follow-ups (2026-11-17)

Drawn from the 0.6.1 port (§41 of `CURRENT-VS-KINDI-0.6.md`): the MTP head's CUDA graph had not
built since 0.5.0, the engine logged one line about it, `/health` still reported drafting as
enabled, `kindi verify` cannot run under `KINDI_SPEC` at all, and the only way to get a speculation
number was to start a server by hand and `curl` it. The general lesson is not "MTP"; it is that a
documented feature can be dead while every gate stays green. Four items are accepted with their gate
written first; the rest are parked with the measurement that would reopen them.

| # | idea | verdict | where it lands |
|---|---|---|---|
| 1 | **The kernel-arity check covers one kernel; make it a launch-shape contract for every kernel and every graph call site** | **A1 + A3 done 2026-11-17.** `tests/test_kernel_arity.py` now measures **105 launches** across `spec.py` (26), `dspine.py` (37), `mtp.py` (3) and `tp.py` (39) against the declared parameter sets, pins the **37** unresolvable sites (variable / list element / `getattr` kernel, or a run-time argument tuple) with a reason each instead of skipping them, pins the coverage count so a silent drop fails, and proves every measured site by mutating a copy and requiring the checker to catch it. The DSpine graph is reached: 31 `kf["..."]` lookups are resolved from the literal that builds `self.kf`, and `self.k` / `self.k8 if n <= 8 else self.k16` from their `get_function(...)` assignments (both suffix variants); the `_Planner` wrapper and `self.hmv` are the two dspine pins left. It encodes two sub-contracts found on the way: `k_tc_q8` is 7 parameters on HIP/Pascal and 8 on CUDA (the call adds `hp` conditionally), and the generated `k_*_T` wrappers (`kindi/chunk.py::_wrappers`, `params + T_ + one stride per pointer`) are declared too. **A2 (grid/block literals) stays open** | [REQ-0023](development/requirements/REQ-0023.md) |
| 2 | **Report the *effective* speculation state, not the requested one — and check the head's integrity when it loads** | **A1 + A2 + A3 + A4 done 2026-11-17.** `spec.resolve_mode()` is now the one mode resolver (`spec_begin`, `/health`, the startup log), and `drafting()` reports `requested`, the effective `method`, `enabled`, a `reason` (folded in from the refused head), the loaded `head` and its `head_sha256` prefix instead of echoing `KINDI_SPEC`. The startup log says `speculation: requested <mode>, effective <mode> (<reason>)`, and a refusal is a `warning: speculation off: ...` line. `kindi/mtp_heads.py` checks `sha256` and `bytes` against `kindi/mtp_heads.json` before `MTPDraft` calls `np.load`; a mismatch or an unregistered file is refused with the expected and found digest, unless `KINDI_MTP_HEAD_ALLOW_UNREGISTERED=1` (documented in the regenerated reference). `kindi info` prints the request plus the head's verdict, not a mode it cannot know before loading. Tests pin the loaded, refused and unregistered cases (mutations caught); `/metrics` exports `kindi_spec_mode_effective{mode}` from `drafting()` so the gauge reports the effective mode, not the request (A2) | [REQ-0024](development/requirements/REQ-0024.md) |
| 3 | **An exactness gate that can run *under* speculation** | **A1 + A2 + A4 done 2026-11-17, A3 four of five.** `kindi verify --spec <mode>` selects the mode for the run and prints `speculation: requested …, effective …, head …` first, so a PASS says which engine produced it; a drafter that did not attach exits 2 with the reason instead of verifying the plain engine, and `auto` alone may settle on another mode. Measured on the 2B / RTX 4060 Ti (4 prompts x 40 greedy tokens): plain 4/4 (279.0 tok/s), `ngram` 4/4 (435.3, 51/152 accepted), `mtp` 4/4 (203.5, 98/248) - mode A is token-exact, so `docs/MTP.md`'s "same output" sentence is now gated and carries the matrix - `draft2b` 4/4 (311.1, 128/128) and `both` 4/4 (276.2, 102/242) with the same 2B as target and drafter, which proves the path, not the 27B pairing. `dspine` has no cell: the weights are not on this host (`docs/MANUAL-HELP-NEEDED.md` row 7) | [REQ-0025](development/requirements/REQ-0025.md) |
| 4 | **`kindi bench --spec`: a reproducible speculation measurement** | **Adopted.** Speculation numbers today need a hand-started server and `curl`, so none of them can be a tracked `docs/benchmarks/` row or a `tools/check_claims.py` cite — including this round's 0.66x mode-A result, which therefore lives in prose only | [REQ-0026](development/requirements/REQ-0026.md) |
| 5 | Mode-A break-even: make `auto` refuse a drafter whose measured acceptance is too low | **Parked.** The 22.5% acceptance measured here is far under break-even, but one prompt family on one card does not define the threshold; `auto`'s k policy (k=2 up to 8K) would still pick it | gate: an acceptance-vs-position sweep (512/8K/24K, two prompt families) that predicts the measured tok/s within 10% on the 2B, then `auto` uses it |
| 6 | A cheaper guard probe (per-token rate over ~4-token windows instead of 48-token spec windows) | **Parked.** The guard cannot rescue a short reply because its first windows are spec windows — the loss is paid before it may pause (measured: 230.9-247.2 vs 236.9-239.6 tok/s with the guard forced on) | gate: lands within 5% of plain on a 400-token reply, and no worse than today's guard above 32K; rides with REQ-0026's harness |
| 7 | **A liveness check for every documented knob** (`KINDI_*` in the generated configuration reference): setting it must change an effective setting in a test, or be listed inert with a reason | **Parked, cheapest half first.** This is the general form of the bug just fixed: a documented knob can be dead while every gate is green | gate: start with the 12 speculation/tensor-core knobs, each with a test or an "inert by design" row; then the whole reference |
| 8 | The long-context mode-A sweep (8K/24K/32K positions) | **Parked behind #4.** `GUARD_FROM` assumes speculation pays above 32K on every card; this round only measured short replies | gate: the sweep through `kindi bench --spec`, mode A vs off per position |
| 9 | Register this round's mode-A run as a tracked benchmark directory with claim entries | **Rides with #4**, not an item of its own: without a reproducible command a registered number would be a hand-typed one, which `docs/claims/README.md` forbids | `docs/benchmarks/<date>-<host>-mtp-mode-a/RESULTS.md` + a claims JSON once REQ-0026 exists |
| 10 | Per-mode acceptance as `/metrics` gauges | **Done through #2 (REQ-0024 A2).** `kindi_spec_mode_effective{mode}` reports the effective mode; the token counters remain global and their help says to pair the ratio with that gauge | REQ-0024 A2 |
| 11 | 0.6.1 container packaging (`compose.yaml`, `docs/CONTAINERS.md`) | **Not adopted.** Deployment surface, not an engine gap; recorded as an owner decision in §41 of the comparison report | no row |

## Constrained-decoding cost (2026-11-17)

The grammar-cost run priced the mask that every constrained reply pays for. It is not a grammar-size
problem: the filter sweeps all 248,320 vocabulary entries in Python once per **new automaton state**,
so the baseline is set by the sweep, not by the 831-1,737 states of the constraint.

| item | direction | baseline the gate must beat | gate | status |
|---|---|---|---|---|
| Constrained-decoding mask cost ([REQ-0027](development/requirements/REQ-0027.md)) | compute the mask once per state instead of once per token pair: either a **token-class table** (group ids by the character/prefix class the state accepts, so a mask is assembled from a few precomputed sets) or an **on-card mask** (accept test as a gather/compare kernel over the token-text table, ids returned in device memory) | **231 ms/token** for `json_schema` and **2,720 ms/token** for the unrolled `json.gbnf` on the 2B (22-token document, greedy); 57 and 463 ms/token at 200 tokens | same host, pack, document and harness (`tools/grammar_cost.py`): `json_schema` at most **23 ms/token** and the unrolled grammar at most **272 ms/token** (a 10x improvement), with the 200-token rows 57 → 6 and 463 → 46; the mask must equal today's `vocab_allowed` set for every vocabulary entry (CPU test, no GPU), keep the §39 structured-output exactness cell, and leave unconstrained decode within 1%. A miss on either row is a no-go recorded here | **planned, not started**; direction named only |

Not in this item: the whitespace trap. Both constraints permit `[ \t\n\r]*`, and a greedy 2B spends a
whole 200-token budget on it (9 characters of reply) — measured in the same run. A faster mask does not
fix that, and it must not be counted toward the gate; bounding whitespace in the constraints or the
sampler is its own change.

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
