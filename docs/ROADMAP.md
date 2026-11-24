# Roadmap

Status: **living document, updated 2026-11-24.** The generated capability tables are in
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
| Tandem CPU+GPU layer split | **PARTIAL** 2026-11-11: dense runtime in both placements, token-exact; joint prompt-cache checkpoints pass on 2B/9B/27B; 27B exactness, rate and sampled peak measured; 9B/27B saved-opening reloads exact at 4K/8K; 2B/4B/9B `cpu-first` throughput (medians of three) and the 2B/27B layer boundaries recorded | the original WIP `tandem.py` was corrupt; a clean dense runtime now replaces it. Both 27B placements are token-exact 4/4 at 8,192 context (peaks 7,005 / 7,017 MiB vs 7,238 MiB full-GPU). On 2026-11-13 the 27B rate plus sampled peak was measured at 8K and 24K against a full-GPU baseline (7.2-8.4 vs 34.8 tok/s at 100; 446 MiB less peak at 8K), the joint checkpoint cost (54.5-61.5 ms save, 27.0-28.5 ms restore, 149.6 MiB), the 2B layer knee (4-6 layers) and the 27B boundary (24K needs >=16 CPU layers; 24-32 layers lower the peak to 5.1-6.1 GiB at a 35-50% rate cost; the 8K sweep 8-28 layers in steps of 4 shows a flat exchange of 428-432 MiB peak saved per step for 15-37% decode cost per step, no knee, read 16.8 -> 6.5). Capacity is settled on this card: requesting 65,536/131,072 tokens auto-reduces to 37,120 (full GPU) and 30,464 (tandem) — the tandem is always the more reduced engine, so it buys headroom, not context. Tandem does not beat GPU-only. The 9B is measured as the control (2026-11-17, 8,192-131,072 on both placements): the full-GPU engine keeps every request (peak 5,149 MiB, 3,003 MiB free after loading 131,072) and the split saves only 486-488 MiB (`gdn:8`) or 872-874 MiB (`gdn:16`) at 2.8-3.0x / 4.5-4.6x decode cost, so where the pack fits the remaining gap is throughput, not context. **2026-11-14, placement:** `--tandem-cpu-layers` now takes a count, an explicit layer set (`0-2,4-6`), `gdn:N` (the first N gated-delta layers, attention stays on the GPU) or `auto[:MiB]`, which computes the smallest GDN-first set that makes the context fit from the pack's own tensor sizes and the free VRAM, and refuses instead of splitting when the model already fits (27B at 24K: 19 layers, context unreduced, 522 MiB free, peak 7,257 MiB; 9B at 24K: refused with '~3282 MiB needed, 7075 MiB free'). The CPU owns layers that are cheap per token, which flattens long-context decode: 9B at 24K 23.6 vs 11.0 tok/s and 27B at 24K 7.9 vs 2.9 (medians of three, same layer count). Two limits are now measured and named: the 9B split saves no peak VRAM at 8K on this card (2.3 GiB pack, ~5 GiB free) so the right answer there is no split, and a split frees far less VRAM than its layer count implies because its GPU owner is the plain engine (the 27B plain engine cannot load full at 24K at all, while the plan engine can). **2026-11-14, plan-engine GPU owner:** the GPU half can now be the plan engine (`--tandem-gpu auto|plan|plain`, default auto): `TPEngine(layers=...)` loads only the owned tensors, allocates q8_0 K/V only for the owned attention layers, reads the f32 file per tensor and steps one prebuilt plan per block with the residual add resolved inside it (`k_padd`). Same 27B placement at 24,576 (`gdn:16`): peak **6,004 MiB** (5,862-6,083, n=5) and 1,985 MiB free after load against **7,721 MiB**/197 MiB for the plain owner, with decode inside this machine's run-to-run noise; the `auto` planner charges the owner's real allocation and predicts 6,548 MiB against 6,444-6,756 MiB measured. The capacity premise is now met on context: a 49,152-token request keeps all 49,152 tokens with 19 GDN layers (peak 6,876 MiB, 1,240 MiB free, 5.3 read, 4.9/4.9/2.8/2.7/2.0 tok/s) where the unsplit plan engine reduces it to 30,976 (122 MiB free). Exactness re-checked with the plan owner: 2B both head placements and 9B `gdn:6`, 4/4 each. Later the same day the plain engine gained the same q8_0 K/V cache (`KINDI_KV=q8_0`; the tandem's plain owner always uses it): token-exact against the fp16 references on the 2B, 9B and 27B, and the 27B 24K `gdn:16` peak falls from 7,721 MiB (fp16) to ~7,110 MiB (995 MiB free; byte accounting expects ~225 MiB of that), with the plan owner re-run the same day at 6,091 MiB (1,985 MiB free). Still open: tandem beats GPU-only (the CPU-owned layers dominate the token), a split across two GPU cards, and the MoE approach itself (it needs a real MoE backend; since 2026-11-17 the dense split can no longer be built over a MoE manifest - `TandemEngine` refuses a non-dense family by name and `make_engine` explains and drops a `--tandem` flag on `kindi_lightning`/`kindi_dsv41` packs - so what remains there is the expert-tier runtime, not a layer split). **A measured tandem gain exists, but is not MoE:** the dense 2B/4B/9B rows are 4.3-5.8x CPU-only at 24K, and the capacity-bound Nanbeige 33B host-layer/CPU-row split on the K6000 writes 3.85/3.55/3.56 tok/s at 480/2,016/8,160 tokens against llama.cpp's 2.19/1.96/1.52; the big-MoE CPU-compute lane still has no kernel or backend | [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [TANDEM-STATUS.md](TANDEM-STATUS.md), [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md), [plan-owner results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md), [plain-q8 results](benchmarks/2026-11-14-i7-8086k-rtx4060ti-plain-q8/RESULTS.md), [9B tandem results](benchmarks/2026-11-17-i7-8086k-rtx4060ti-tandem-9b/RESULTS.md), [MOE-TANDEM.md](MOE-TANDEM.md) | a larger-than-VRAM model runs token-exact with a recorded VRAM reduction; the MoE approach needs a real MoE backend first |
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
- **An Anthropic Messages endpoint ([REQ-0028](development/requirements/REQ-0028.md), landed
  2026-11-24):** `POST /v1/messages` now answers the same engine path in Anthropic's dialect — content
  blocks (`text`, `tool_use`, `tool_result`), `system`, `tools`/`tool_choice`, `stop_sequences`, and
  the SSE order `message_start` → `content_block_start` → deltas → `content_block_stop` →
  `message_delta` → `message_stop`. `kindi/api/anthropic.py` is the whole translation (no FastAPI, no
  engine import, so it unit-tests on a CPU-only host); the route shares `_relay` with the OpenAI
  streaming path. Auth takes `x-api-key` or a bearer token. Evidence: `tests/test_anthropic.py`
  (A1/A2 plus A3's app-level gate with a stub engine). **Open:** A3's real Claude Code run against
  `kindi serve`, and A4's no-decoding-change check — the endpoint is *available*, not *compatible*. Proposed by the external knowledge-base scan of
  2026-11-17 (`docs/development/AI-INVENTORY-SCAN.md`), which is inspiration only.
- **n8n workflow integration ([REQ-0029](development/requirements/REQ-0029.md), proposed):** a
  self-hosted n8n workflow should use `kindi serve` as an OpenAI-compatible provider without a
  Kindi-specific node: an HTTP Request node posting `/v1/chat/completions`, or the OpenAI Chat
  Model node with a custom `/v1` base URL and the exact model id from `/health`. The first gate
  is a recorded end-to-end run plus a Docker-host variant (non-loopback bind, API key,
  `host.docker.internal` or LAN address), serialized execution with bounded retries, and a
  stable system/tool prefix so the cache is exercised. Chat completions are the surface;
  `/v1/responses` is separate. No engine or API change is proposed until a recorded run names a
  concrete gap.
- **Selectable native hot paths ([REQ-0030](development/requirements/REQ-0030.md), proposed):** keep
  Python as the reference and default, but allow a measured pure-Python hot function to be swapped
  for a C or Rust implementation at runtime (`KINDI_NATIVE=python|c|rust`). C is built through the
  existing compiler/cache path and checked with cppcheck; Rust uses the same C ABI as a `cdylib`
  and is checked with cargo fmt/clippy/test. Start with a profile, not a rewrite: the 2026-11-22
  2B/9B profile in REQ-0030 leaves the constrained-decoding mask as the only qualifying candidate;
  the BPE tokenizer and stream framing were removed, and no native code was added. `kcpu.c` and
  the CUDA kernels are already native and excluded. A swap is adopted only if it is token/mask/byte-identical and
  improves the measured function by at least 1.25x with at least 5% end-to-end p50, with no >1%
  unconstrained-decode regression; otherwise it stays Python. The external sketch's 15-35%
  numbers are not measurements of this tree and are not adopted.
- **Hardware identification and autotuning ([REQ-0022](development/requirements/REQ-0022.md),
  experimental):** `kindi caps` reports every card the driver sees (compute capability, total/free
  VRAM, PCI bus, UUID, driver) with pynvml and CuPy fallbacks, keeps the hardware fact table
  separate from the Kindi paths that capability enables, and reports the CPU half from the engine's
  own probe plus model name, core counts, L2/L3, AVX-512/AMX and the CPU-engine toolchain.
  `kindi autotune` turns that into a measurement: candidates derived from the hardware and the pack
  manifest (full card, two-card TP split, tandem auto/gdn:N, Lightning decode cache, CPU engine),
  one real `kindi bench --sample-vram` child run each, ranked by decode tok/s with a reduced-context
  run ranked below a full one and a failed run recorded with its own error. Measured on this host
  2026-11-17 (2B, ctx 8192): `gpu0` 97.4-97.6 tok/s decode / 111.7-113.0 read / 3,050 MiB peak,
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
  is installed in the Windows test rig**; M1 hardware qualification is partly measured there, and
  the M2 memory and device-kernel halves now pass (`kindi intel-smoke`: device USM + shared USM
  round trip; `kindi intel-cl`: runtime-compiled OpenCL vector-add/GEMM with zero mismatches),
  while the SYCL `libksycl.so` path, Kindi device selection and a model run remain open. The card
  is registered in the support matrix as `not measured` and is not offered by the installer. This
  is not a support claim.
  - **27B pack + K8/V4 on the A770 (2026-11-24, blocked):** the pack already exists as a copy at
    `T:/kindi-models/packs/bonsai-27b` and its `kindi_pack.json` source sha256 matches the GGUF at
    `T:/kindi-models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (verified the same day; the GGUF was not
    moved or deleted), so no re-convert is needed and none is run - a fresh convert needs ~6.8 GiB
    of scratch the host does not have. Every single-card K8/V4 check (`kindi verify` / `kindi kld`
    `tp-k8v4` on the CUDA plan engine; [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) records 2B 2/4, 4B 4/4,
    9B/27B n/a) needs a CUDA device or the not-yet-existing Intel backend: this host has neither
    (`cudaErrorNoDevice`, no SYCL, CPU engine does not support the 27B), so `kindi verify` on the
    27B pack fails before any kernel runs. The two-card half still needs a second identical card
    plus REQ-0020 M3-M6 first (single-card 2B exact, then 27B exact, then the layer split), and
    27B K8/V4 itself has no kernel path (`k8v4` refuses other head sizes; the plain engine rejects
    it), so even a two-card run would have no K8/V4 row to measure. Reopen when M2's SYCL smoke
    passes on Linux.
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
| SenseNova-Loop-9B (looped dense, multimodal) | ~9B distinct / ~14.6B inference-equivalent; 32 stored decoder layers, layers 4-27 executed twice (56 layer executions); two-stream mHC with Sinkhorn 20; three linear-attention + one full-attention layer per four-layer group; 512K context; text+images -> text; official BF16, 4-bit AWQ and GGUF (Q4_K_M 5.82GB; 4-bit weights ~4.9GB per the model card) | Q0: pin one text-only Q4_K_M GGUF and a fixed greedy prompt under the official OpenSenseNova/llama.cpp fork; record SHA-256, Apache-2.0, tensor/loop/mHC/attention inventory and the vendor memory table. Then Q1: Kindi loader + 32/56 execution + mHC + hybrid attention, `kindi verify` 4/4 at 2K text-only against the pinned reference. No vision, MTP or ternary in Q1 | **proposed; no backend, pack, catalog entry or support claim** — official release 2026-11-17; GGUF needs the vendor's llama.cpp fork (stock llama.cpp does not load `sensenova_loop` yet). Details, ternary caveat and gate ladder in [SenseNova-Loop-9B](#sensenova-loop-9b-2026-11-20) |
| Inkling 2 Mini (looped MoE, multimodal) | 26.4B total / 3.4B active (5.4B weight activations per token with the loop); 24 stored decoder layers, layers 6-17 executed three times (48 applications); 128 routed experts, top-8; hybrid local/global attention with a 1K sliding window; 1M context; text+images -> text; mixed NVFP4/MXFP8 experts; public GGUF 7.8-22.9GB (Q4_K_M 16.6GB), bundled DSpark drafter | I0: pin one public GGUF and a greedy reference under a pinned mainline llama.cpp build (the mirror ships mainline-support copies); record SHA-256, Apache-2.0, tensor/loop/MoE inventory and the 24->48 map. Then I1: host-RAM expert lane + loop remap, `kindi verify` 4/4 at 2K text-only | **proposed; header inventoried 2026-11-22, no backend, pack, catalog entry or support claim** - 24 stored blocks serve 48 logical layers; the Q4_K_M file is 16.6GB, so an 8GB card cannot hold it and the first target is the host-RAM lane, not a fit. Gates and the measured map in [Inkling 2 Mini](#inkling-2-mini-2026-11-22) and [REQ-0031](development/requirements/REQ-0031.md) |

## SenseNova-Loop-9B (2026-11-20)

**Status: proposed, no backend, no pack, no support claim.** [SenseNova-Loop-9B](https://github.com/OpenSenseNova/SenseNova-Loop) is a genuinely different low-memory option from the MoE goals above: the [official model card](https://huggingface.co/sensenova/SenseNova-Loop-9B) reports approximately **9B distinct parameters** stored in **32 decoder layers**, with layers 4-27 executed twice on the same weights (**56 layer executions**, approximately **14.6B inference-equivalent**). It combines two-stream manifold-constrained hyper-connections (mHC) with **three linear-attention layers and one full-attention layer in every four-layer group**, accepts text and images, generates text, and supports a **512K-token** context. It is Apache-2.0 and the official artifacts include BF16, 4-bit AWQ (with a matching MTP draft) and [GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) at BF16, Q8_0, Q5_K_M and Q4_K_M.

**Memory caveat.** The model card says 4-bit weights occupy about **4.9GB**; the official GGUF Q4_K_M file is **5.82GB**. The vendor's minimum-GPU-memory table is approximately **9-10GiB at 64K** and **12-15GiB at 256K** depending on the KV cache. **An 8GB card is below the vendor's 64K minimum**; short-context use is an experiment, not a supported fit. 16GB is the realistic target for long context. The 16-22 tok/s P4000 figure is an external estimate, not a Kindi measurement or a gate baseline.

**What a Kindi backend has to preserve.** The looped block is `B(B(h))`: layers 0-3, then the shared layers 4-27 twice, then layers 28-31 and the output head. The mHC carries two residual streams, mixes them with a Sinkhorn-normalized map (20 iterations in the checkpoint) and keeps its routing parameters in FP32; the loop range, pass count and routing parameters are part of the execution configuration. The hybrid decoder interleaves three linear-attention layers with one full-attention layer (16 query / 4 KV heads, hidden 4096, FFN 12288, vocabulary 248,320). A loader that reorders or fuses any of these changes the model, not just its speed.

**Ecosystem caveat.** The official GGUF is not stock-llama.cpp-compatible: the card builds [OpenSenseNova/llama.cpp at `24ea7bb`](https://github.com/OpenSenseNova/llama.cpp/tree/24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761), and the upstream [support issue #31571](https://github.com/ggml-org/llama.cpp/issues/31571) is still open. vLLM and SGLang support is in pull requests rather than releases. This is enough for a pinned Q0 reference, but not a plug-and-play Kindi pack.

**Shared map check (2026-11-22).** `kindi/looped.py` now turns either measured map into a full
logical execution plan (logical layer, stored block, phase and loop pass) and checks the GGUF
tensor index for missing or unexpected stored blocks. `kindi loop-info FILE.gguf [--json]` prints
that report without loading tensor data, and a changed `block_count` is refused by name. This is
host-side map validation for the eventual loader, not a pack, engine or support claim.

**Do not start with ternary.** The low-memory option is the official 4-bit AWQ/GGUF weights, not a naive `{-1, 0, +1}` PTQ. The 24 shared layers are reused twice, so rounding error is applied twice; the mHC routing and attention projections are sensitive and stay higher precision in the reference. Any ternary experiment keeps embeddings, attention and mHC at higher precision, quantizes only selected MLP/linear weights, and compares KLD, PPL and task outputs against the 4-bit baseline. It is lossy and opt-in; QAT is a separate step only if mixed-precision PTQ fails. A generic 2-bit container without a ternary kernel is not a memory or speed win.

| phase | gate | status |
|---|---|---|
| Q0 reference | Pin one text-only **Q4_K_M GGUF** (SHA-256, byte size, Apache-2.0 chain) and one fixed greedy prompt under the vendor's `24ea7bb` llama.cpp fork; record the 32-layer/56-execution config, mHC fields, 3:1 attention map, tokenizer and output tokens/logits. No vision or MTP. | **header inventory done 2026-11-22; full SHA-256 and the reference run open.** |
| Q1 backend | A Kindi loader/converter for the pinned format plus CPU tensor/layout tests; `kindi verify` 4/4 token-exact at 2K text-only against Q0, with the looped and mHC execution order asserted. | **proposed; no code.** |
| Q2 memory | On the RTX 4060 Ti, 2B-class baseline not applicable: load Q4_K_M and record sampled peak at 4K/8K/16K/64K where it fits, requested versus actual context, and the KV format. An 8GB 64K run is expected to fail the vendor minimum; record it as a no-go, not a reduced-context pass. Repeat on a 16GB card if one is available. | **proposed; hardware-dependent.** |
| Q3 speed | Same host, same pack, same context: p50 prefill and decode for 4K/8K, with mHC stream-mixing fused and unfused; no number is adopted until measured. | **proposed; after Q1.** |
| T0 ternary | Mixed-precision PTQ on selected MLP/linear weights only; KLD, PPL and a fixed task set against the 4-bit baseline; a memory or speed win with a named lossy tolerance. QAT only if PTQ misses. | **parked behind Q1/Q2; not a support claim.** |

### SenseNova Q0 header inventory (2026-11-22)

A 32MiB range request read the public Q4_K_M header without downloading the weights. The file's
`block_count` is **57 logical**, but the tensor names cover only **33 stored blocks**: 0-27 and
52-56. Logical 28-51 are the second pass over 4-27 and have no tensors of their own, so
`blk.28`-`blk.51` do not exist. The post-loop layers keep their logical indices (52-55) and the
nextn layer is 56; Inkling below uses the opposite convention, which is why the map lives in
`kindi/looped.py` per family. Also measured: `loop.start=4`, `loop.end=28`, `loop.count=2`;
mHC on 4-28 with 2 streams and 20 Sinkhorn iterations; `full_attention_interval=4`; 16 query /
4 KV heads at key/value length 256; SSM groups 16, state 128, time step rank 32; context 524,288;
490 tensors in F32/Q4_K/Q6_K. Upstream [issue #31571](https://github.com/ggml-org/llama.cpp/issues/31571)
is still open (checked 2026-11-22), so Q0's reference still needs the vendor
`OpenSenseNova/llama.cpp` at `24ea7bb`. Full table:
[REQ-0031](development/requirements/REQ-0031.md); gates in
[REQ-0031](development/requirements/REQ-0031.md).

**`kindi loop-info` (2026-11-23).** On a fresh 32 MiB range (revision `b489b15c...`, range
sha256 `e9c95bf1...`) the command reports **33/33 stored blocks present**, no missing and no
unexpected block tensors, exit 0, with the plan `pre 0-3; loop 4-27 twice; post 52-56`. Evidence: the Q0
header inventory's benchmark page, linked from [REQ-0031](development/requirements/REQ-0031.md). This is a map
check only;
the status stays proposed with no loader, pack or support claim.

## Inkling 2 Mini (2026-11-22)

**Status: proposed, no backend, no pack, no catalog entry and no support claim.**
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines
Lab's local model (released 2026-11-20, Apache-2.0): a **looped MoE** with **26.4B total** and
**3.4B active** parameters (the model card and the HF launch post also give **5.4B weight
activations per token** because the loop touches more weights), 24 stored decoder layers whose
middle 12 (6-17) run **three times**, giving 48 layer applications, 128 routed experts at top-8
with no shared expert, hybrid local/global attention with a 1K sliding window, and a 1M-token
context. It accepts text and images and generates text.

**Why it is interesting and why it is not a small-card model.** The loop buys depth without
storing another copy of the middle layers, and the bundled 3-layer DSpark drafter is the same
drafter family the DeepSeek path already carries. But the smallest public GGUF is **7.8GB** and
the Q4_K_M file is **16.6GB**; the full model needs about 18GB of memory. An 8GB card cannot hold
the file at all, so any "runs on 8GB" claim is false. The plausible Kindi target is the
host-RAM/expert lane (Lightning and Nanbeige already keep experts in host memory), not the dense
fit that SenseNova's ~9-10GiB minimum still allows at short context.

**Measured header (2026-11-22).** `general.architecture=inkling2`, `block_count=48` logical over
**24 stored blocks** (0-23); logical 18-41 are the second and third passes over 6-17 and have no
tensors, and the post-loop blocks 42-47 are stored at the **physical** indices 18-23. `loop.core_start=6`,
`loop.core_end=18`, `num_loops=3`, `loop.gate_rank=64`; global `loop_res_query` and
`loop_kv_res_query` tensors; 128 experts, top-8, expert FFN 1152; embedding 2304; context
1,048,576; sliding window 1024; SITU activations; final logit softcap 30; vocab 201,024; 734
tensors in F32/Q8_0/Q5_K/IQ4_XS/MXFP4-class. Full table:
[REQ-0031](development/requirements/REQ-0031.md).

**`kindi loop-info` (2026-11-23).** On the mirror's 32 MiB range (revision `800c946d...`, range
sha256 `229bbecf...`) the command reports **24/24 stored blocks present**, no missing and no
unexpected tensors, exit 0, with the plan `pre 0-5; loop 6-17 three times; post 42-47 stored at
18-23`. The mirror's `main` moved to that revision on 2026-11-23; the Q4_K_M size is unchanged.
Evidence: the Q0 header inventory's benchmark page, linked from
[REQ-0031](development/requirements/REQ-0031.md). Map check only;
no loader, pack or support claim.

**Reference.** Unlike SenseNova, the GGUF mirror ships `Mainline_Support/` copies and the HF
launch post lists llama.cpp support, so I0 can pin a mainline build instead of a vendor fork;
the exact binary and revision are still to be recorded. Vision, the DSpark drafter and any
ternary experiment stay out of I4.

| phase | gate | status |
|---|---|---|
| I0 reference | Pin one public GGUF (SHA-256, byte size, Apache-2.0 chain) and a greedy text-only run under a pinned mainline llama.cpp build; record the tokenizer revision, output token ids and the 24->48 map. No vision or DSpark. | **open; header inventoried 2026-11-22.** |
| I1 loader | Host-RAM expert lane plus the loop remap and the MoE router; CPU tensor/layout tests; `kindi convert` must reject a missing stored block instead of reusing a neighbour. | **proposed; no code.** |
| I2 exactness | `kindi verify` 4/4 token-exact at 2K text-only against I0, asserting the three loop passes and the top-8 router. | **proposed; needs I0/I1.** |
| I3 memory | Record requested vs actual context and sampled peak; 8GB is expected to fail on file size, so record it as a no-go, not a reduced-context pass. Repeat on a 24GB card if available. | **hardware-dependent.** |
| I4 speed | Same host, pack and context: p50 prefill/decode with the loop fused and unfused; DSpark measured separately, not folded into the base claim. | **after I2.** |
| T0 precision | Any ternary/low-bit experiment compares KLD, PPL and task output against the official mixed-precision baseline; the loop applies the error on every pass. | **parked behind I2.** |

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

## Agent-side token economy (2026-11-21)

**Parked, complementary — not an engine change.** [TokenEconomizer](https://github.com/microsoft/TokenEconomizer)
is an external controller for coding agents (Claude Code, Codex, Copilot CLI), not an inference engine.
It learns per repository across sessions and applies three mechanisms: micro-skills (short operational
facts injected at session start), tool pruning (tool definitions the agent never uses are removed through
the agent's own settings), and a failure-loop guard (a judge model redirects a stuck turn, at most once).
The paper reports 18.4-30.9% lower mean billed cost on 277 SWE-Bench tasks across five agent-model
settings, 16-21% on Web-Bench and 6-20% on security-audit tasks; Pass@1 improved in four settings and
fell 1.0 point in the fifth, and the savings include the controller's own model calls. Its state lives
in `~/.tokeneconomizer`; it is Python 3.10+ and stdlib-only.

**Why Kindi does not implement it.** The mechanisms act on the agent's prompt, tool list and control
flow. A server cannot apply them without rewriting client requests, which would change agent semantics
and is out of scope. The compatible path is agent -> TokenEconomizer -> Kindi: Codex and other
OpenAI-shaped clients can point at `kindi serve` today; Claude Code needs
[REQ-0028](development/requirements/REQ-0028.md)'s `/v1/messages`. Kindi already returns
`prompt_tokens_details.cached_tokens` and has a prefix cache, so the ledger's cache accounting has a
real signal.

**Why fewer tokens can still be faster here.** Less prompt/tool/history text means shorter prefill and
a smaller K/V working set, which matters most on old cards. But the plugin's judge and fact-proposer
calls run through the same agent and consume local decode time; its dollar model assumes frontier API
prices and hard-fails on an unknown billed model id in experiment mode (live mode has a labeled
fallback). Tool pruning also changes the prompt prefix, which can miss Kindi's prefix cache and pay a
full re-prefill; the plugin waits for an idle hour for exactly that reason. It reduces the number of
tokens the agent sends, not the per-token cost of the engine.

**Gate before any claim.** On an old card, run the same repository/task sequence paired with and without
the plugin, and record per task: prompt/completion/cached tokens, prefill and decode seconds,
wall-clock, and pass/fail. The gate is >=15% fewer prompt tokens and >=10% lower median task wall-clock,
including the plugin's own calls, with no pass-rate drop. A dollar saving alone is not a Kindi result.
No Kindi run exists yet, and no code is proposed.

**Licensing.** The repository's README badge says MIT while `pyproject.toml` declares Apache-2.0;
resolve that before vendoring any of it. Running it as an external agent plugin does not put its code
in Kindi.

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
| Qwen 3.6-35B-A3B (True MoE) | `qwen35moe` | recognized, refused; 40 layers (`10 × (3 Gated DeltaNet → 1 Gated Full Attention)`), hidden 2048, 248,320 padded vocab, 262,144 native ctx; 256 experts/layer (top-8 + 1 shared, intermediate 512) | **Q0 pinned GGUF/reference gate: one byte-pinned `qwen35moe` GGUF (current 16 GB candidate: Unsloth `UD-IQ2_M`, 11.5 GB / 11,522,702,304 B at revision `a483e9e6...`, Apache-2.0), tensor inventory over every tensor, and one fixed greedy run under pinned upstream llama.cpp; this is a stock-llama.cpp reference, not yet a Kindi run.** Then Q1 Gated DeltaNet state/recurrence (30 layers; 16 QK + 32 V heads, head_dim 128, SSM inner 4096, conv kernel 4, time-step rank 32), Q2 Gated Full Attention (10 layers at 3/7/…/39; 16 Q / 2 KV heads, head_dim 256, partial RoPE 64, attn/output gate, mRoPE `[11,11,10]`), Q3 MoE router + experts (`ffn_gate_inp` `(256,2048)` no bias/correction; separate `ffn_gate_exps`/`ffn_up_exps` + `ffn_down_exps`, or fused `ffn_gate_up_exps`; shared `ffn_*_shexp`; ~32.2 B expert params), Q4 host-resident expert cache, Q5 tokenizer/chat/no-vision text-only `kindi verify` |
| GLM-4.7-Flash | `glm4moelite` | recognized, refused | GLM4-MoE-Lite loader, MLA/KV attention path, routed-plus-shared expert path + pinned reference |
| North Mini Code 1.0 (30B/~3B) | `cohere2moe` | recognized, refused | Cohere2-MoE loader, chat template and tokenizer checks + pinned reference |
| Recursive Flint 40B/2.5B | `flint` | recognized, refused | loader, router and expert path + pinned reference |

**Qwen 3.6 Q0 first gate (do not start the backend before this):** pin one text-only GGUF at a
repository revision, record its SHA-256 and Apache-2.0 license chain, and dump a tensor inventory
that proves the layer pattern (30 `blk.N.ssm_*` GDN blocks; no `ssm_*` on blocks 3, 7, 11, …, 39),
the full-attention tensors, and every router, routed-expert and shared-expert tensor. The same
artifact runs once under the pinned upstream llama.cpp `qwen35moe` reference for a fixed 1K-2K
greedy prompt; its logits/tokens are the gate every later DeltaNet, attention and MoE milestone is
compared against. Vision and MTP are explicitly out of scope for this text-only backend: pin the
LM GGUF only (the separate `mmproj-*.gguf` is not part of Q0) and record whether the main file
carries an MTP block. Do **not** use the Ollama-converted layout as the first pin: it has been
observed to differ (`ssm_dt` without `.bias`, `rope.dimension_sections` length 3 vs 4); use an
Unsloth llama.cpp-compatible GGUF, or normalize those deltas in a Kindi converter with byte-level
tests first.

**Standard llama.cpp GGUF route for a 16 GB NVIDIA card (checked 2026-11-19):**
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
full-attention layers' K/V and the runtime reserve. `UD-Q3_K_S` leaves only 1.8 GB.
`UD-Q3_K_M`/`UD-Q3_K_XL` do fit within the 16 GiB card (16.6/16.8 decimal GB of weights), but
leave only 0.6/0.4 decimal GB for context, K/V and runtime overhead, so useful-context runs need
partial CPU offload (`-ngl` below every layer) and are not whole-GPU candidates. A starting point
for the IQ2 route is stock `llama-server -m Qwen3.6-35B-A3B-UD-IQ2_M.gguf -ngl 99 -c 8192` (or
`UD-IQ3_S` for a
longer context budget check); the llama.cpp build must include `qwen35moe` support.

This is a llama.cpp route, not a Kindi one. Kindi still recognizes `qwen35moe` and refuses it
because it has no such backend, so none of these GGUFs can be served or verified by Kindi yet. The
still-missing gate is the backend itself: the Q0 pinned-GGUF/reference artifact and the GDN,
full-attention and MoE milestones in the table above, not another quantization. For the 16 GB
target, choose `UD-IQ2_M` (11.5 GB) as the byte-pinned artifact; the Q4_K_M file is an optional
quality reference, not a second required pin. Until that backend lands, the concrete 16 GB/NVIDIA recommendation
is stock llama.cpp with an IQ2 or IQ3 file, and the ternary community builds remain
Q0-ineligible (their census is linked below).

### Qwen3.6 Q0 audit (2026-11-19)

**Recognition-only today.** `kindi/moe_compat.py` maps `general.architecture = qwen35moe` to
Qwen3.6-35B-A3B and returns a specific refusal; `tests/test_moe_compat.py` pins the label. There is
no `qwen35moe` converter, loader, engine, catalog entry, installer asset or support-matrix row.
Nothing in this audit adds one, and the Apple-only community files are not recognized by this path
at all.

**Name and card audit.** The official Hugging Face config is `model_type: qwen3_5_moe` (class
`Qwen3_5MoeForConditionalGeneration`, current revision `345d467b...`); the llama.cpp/GGUF
architecture string is `qwen35moe`. `moe_compat.py` sees only the GGUF name, so a future loader
must map the two or consume GGUF only. The official card/config confirm 40 layers (30 Gated DeltaNet
+ 10 Gated Full Attention), hidden 2048, 248,320 padded vocab, 256 experts per layer with top-8 + 1
shared, expert intermediate 512, 262,144 native context, a vision encoder, and a separate one-block
MTP module. The Unsloth GGUF card (revision `a483e9e6...`) is the ordinary `qwen35moe` route; its
`UD-IQ2_M` file is 11,522,702,304 B and `UD-Q4_K_M` is 22,134,528,992 B, with a separate
`mmproj-*.gguf`. Those are publisher metadata, not a local download or hash. The benchmark scores
in the census remain external card claims, not gate evidence.

**Header check (not a payload hash).** A 16 MiB range read of the current `UD-IQ2_M` at that
revision parses as GGUF v3 with **733 tensors**, `data_start = 10,990,048`, `general.file_type = 29`
(mostly IQ2_M) and `general.license = apache-2.0`. The metadata says `block_count = 40`,
`rope.dimension_sections = [11, 11, 10, 0]`, 16 Q / 2 KV heads with head_dim 256, 256 experts top-8
plus one shared expert (both intermediate 512), and GDN conv 4 / inner 4096 / state 128 / rank 32 /
16 groups. The 30 GDN blocks each have 19 tensors (including `ssm_dt.bias`); the 10 full-attention
blocks each have 16 (separate `attn_q`, `attn_k`, `attn_v`, `attn_output`, Q/K norms and no
`ssm_*`). Every block uses separate `ffn_gate_exps` + `ffn_up_exps` (not fused `ffn_gate_up_exps`)
and the four `ffn_*_shexp` tensors. There is no `blk.40` and no MTP tensor; the only top-level
tensors are `token_embd`, `output_norm` and `output`. This is header evidence only: no payload was
hashed and no token/logit was produced.

**Q0 reference checklist (still open).** One artifact is enough, but it must be pinned and its
inventory must be explicit:

1. Download the chosen text-only GGUF at a fixed revision and record its local SHA-256, byte size
   and Apache-2.0 chain. Do not treat the HF LFS object id or card size as a verified local hash.
2. Check `general.architecture = qwen35moe`, `block_count = 40`, `context_length = 262144`,
   `expert_count = 256`, `expert_used_count = 8`, expert/shared intermediate 512, 16 Q / 2 KV
   heads with head_dim 256, and the GDN values (conv kernel 4, inner 4096, state 128, time-step
   rank 32, group count 16). Record `full_attention_interval = 4` or the equivalent recurrent-layer
   array. The official mRoPE section is `[11, 11, 10]`; llama.cpp requires a four-entry
   `rope.dimension_sections`, and the audited file uses `[11, 11, 10, 0]`, so keep the fourth value
   explicit and do not normalize it away without a byte-level test.
3. Inventory 40 trunk blocks. GDN blocks (0-2, 4-6, …, 36-38) need `attn_qkv`, `attn_gate`,
   `ssm_conv1d`, `ssm_dt.bias`, `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm` and `ssm_out`; full
   attention blocks (3, 7, …, 39) need the Q+gate form (`attn_q`, or fused `attn_qkv` if the file
   uses it), `attn_k`, `attn_v`, `attn_output`, `attn_q_norm` and `attn_k_norm`, and must carry no
   `ssm_*`. Every block needs `attn_norm`, `post_attention_norm`, `ffn_gate_inp`, either
   `ffn_gate_up_exps` or separate `ffn_gate_exps` + `ffn_up_exps`, `ffn_down_exps`, and the shared
   `ffn_gate_inp_shexp` / `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` set; record the
   top-level `token_embd`, `output_norm` and `output` (or tied fallback). The audited file has 19
   tensors per GDN block and 16 per full-attention block, with separate `ffn_gate_exps` +
   `ffn_up_exps`; record the same counts for the pinned hash.
4. Record whether the file contains an MTP block. The audited `UD-IQ2_M` has none (`block_count =
   40`, no `blk.40`); if another pin carries one, a text-only Q0 run must skip it and prove the
   trunk output is unchanged. Build the pinned upstream llama.cpp `qwen35moe` commit, run a fixed
   1K-2K greedy prompt, and store tokens/logits, prompt, context and sampling settings. No Kindi
   `verify` is possible before the loader exists.

**Converter and backend work (not started).** The existing converter cannot turn this file into a
Kindi pack: `kindi/convert.py` has no `qwen35moe` path and its Bonsai reader accepts only
PTQ1_0/PQ2_0/Q2_0 plus F32/BF16/FP8, while `UD-IQ2_M` is IQ2_M. A first backend therefore needs
the following, in order:

- a GGUF loader/pack for the pinned quant types, the 248,320-token tokenizer and the text-only chat
  template, with vision and MTP explicitly rejected or skipped;
- Gated DeltaNet recurrence: conv state (kernel 4), delta rule, A/beta/alpha/dt, gated norm and
  f32 state across the 30 GDN layers;
- Gated Full Attention: Q+gate, 16 Q / 2 KV heads, head_dim 256, partial RoPE 64, mRoPE
  `[11,11,10]`, Q/K RMS norm, sigmoid output gate across the 10 full-attention layers;
- MoE: softmax top-8 of 256 with no bias/correction, routed gate/up/down experts (fused or
  separate as pinned), the gated shared expert, and 40 layers of expert packing;
- token-exact `kindi verify` against the Q0 tokens/logits, then the existing KLD/PPL gate, and only
  then a catalog/installer/support-matrix entry. The host-resident expert cache (Q4) is a later
  performance follow-up, not a correctness prerequisite.

**Qwen3.6-35B-A3B ternary community builds (external, checked 2026-11-19):** the community builds,
their contents, formats and why none is a llama.cpp-compatible Q0 artifact are in the canonical
[ternary-MoE census](TERNARY_MOE.md). Neither replaces the pinned-GGUF Q0 gate or gives
this repository a 16 GB fit today. A later, separate experiment may test ternary gate/up plus a
higher-bit down projection, but only after a byte-pinned GGUF export (or a converter with
bit-level dequant tests) and a token-exact `kindi verify` plus the existing KLD/PPL quality gate.

**External card claims (not Kindi measurements).** The official
[Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reports **SWE-bench Verified
73.4** and **LiveCodeBench v6 80.4** for the base model. The
[russellwmy ternary card](https://huggingface.co/russellwmy/qwen3.6-35b-a3b-ternary) reports
**HumanEval pass@1 140/164 Python** and **108/156 Rust**, with **reply perplexity 1.227**, against
**153/164**, **142/156** and **1.102** for its 4-bit MLX source. These are the publishers' harness
results, not Kindi runs, and they say nothing about an NVIDIA or standard-GGUF path. The
Apple-only/ironbark limitation is unchanged: the ternary bundle still needs an Apple Silicon Mac
plus `ironbark` and is not loadable by `transformers`, `mlx-lm` or `mlx-vlm`; the other ternary
build remains TurboQuant-MLX on Apple Silicon.

### Ternary-MoE census (2026-11-18)

The canonical, machine-checked census of the Qwen3.6-35B-A3B ternary builds and the other
published ternary MoE builds is [TERNARY_MOE.md](TERNARY_MOE.md). It owns the rows, the row
and slot counts, the expert-slot arithmetic, the unread leads and the checked-and-not-counted
list; `scripts/check_ternary_moe.py` fails if a row or count drifts. This roadmap keeps only
the Qwen3.6 backend gates and links here, so there is one source.

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
| 2 | CPU-compute lane for misses | a miss is computed on the CPU from host-resident weights while the GPU computes the hits; only the expert's output vector crosses PCIe | bit-identical expert output between the CPU and GPU kernel for the quant format (M2 in [MOE-TANDEM.md](MOE-TANDEM.md)), then a capacity-bound MoE row beats the copy lane's tok/s with `kindi verify` 4/4 | **expert gate and combine contract passed 2026-11-19; CPU lane not adopted on cost and moved to `tools/` as an experiment; model row blocked.** `tools/moe_expert_lane.py` computes a Q2_0 Lightning expert on the host with the kernel's own arithmetic (int32 run sums, one `fmaf` per run into lane `h % 32`, the five-step XOR tree) from the card's `lt_quant` runs; `tools/q2_expert_lane.py` compares it with the real `lt_mv_q2`/`lt_mv_q2_down` at hidden 2688 / width 1856 on both device-resident and host-mapped weights and gets **0 differing elements** on both matrices in both paths. The mixed card/CPU combine contract is now implemented and gated: `Q2Matrix.partials`, `lt_mv_q2_down_parts`, `lt_q2_shared_parts` and `lt_q2_combine` reproduce `lt_mv_q2_down`'s per-lane gateway sum and XOR tree, and `tools/q2_combine_lane.py` mixes two host and two card experts plus the shared expert with **0 differing elements** against one `lt_mv_q2_down` launch (zero and non-zero residuals, and the shared-only branch); host partials match the parts kernel. Cost per 2,806,272-byte expert (three fresh processes, medians): CPU **2.395 ms**, H2D copy **0.643 ms** at **4.065 GiB/s**, GPU kernels **0.140 ms**, copy lane **0.783 ms**, and the engine's direct host-mapped GPU read **0.662 ms**; the host-mapped path is **1.18x** faster than the copy lane and **3.62x** faster than the CPU lane, while the CPU lane is **3.06x** slower than the copy lane. The host-mapped read is the existing engine path; the CPU lane is not wired into `lightning.Engine`. The capacity-bound row is **blocked** by the missing `kindi_lightning` pack (no verify 4/4, no tok/s). Evidence: [expert-lane RESULTS.md](benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-expert-lane/RESULTS.md), [combine RESULTS.md](benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-combine/RESULTS.md). |
| 3 | I/O-aware slot and cache sizing | use the measured RAM/PCIe/disk rates, the usage file and the room to choose the per-layer slot count, and the RAM cache size for a streamed model; print the reason per choice | predicted slots/cache within 10% of the measured best for a Lightning pack on this host, and a plan whose numbers a live `bench` reproduces | **arithmetic implemented 2026-11-17, gate open.** `ltfit.rate_slots` predicts `n_used * (1 + ceil((RAM + disk) / PCIe))` slots, capped by `n_exp` and the fit; `kindi plan` prints the three rates and uses that count, its follow-up `bench` command carries the planned context and count with the global `--ctx` and portable `--dcache` flags so the child engine does not fall back to defaults, and `kindi autotune` tests it with half/double candidates plus an `expert-cache-<MiB>` cold-tier budget (`KINDI_LT_COLD=1` + `KINDI_LT_HOST_MB`) sized by the same rate model and capped by the host experts. Both commands can reuse a saved `kindi io --json FILE` report via `--io-report`, falling back to one probe when it is missing or unusable. Unit-tested only: no Lightning pack on this host, so the within-10% and live-bench clauses are **not measured** |
| 4 | Mixed-precision tiers | keep fast-unpacking or higher-precision experts in VRAM and compact 2-3 bit/ternary ones in RAM or on disk | bytes/token for the same served share, plus exactness where the format is exact and a KLD tolerance where it is not | **parked.** `MoESpec` already takes per-layer expert bytes and the Q2_0/PQ2_0/FP8 codecs exist, but both tiers use the same format today |

## Engine-liveness follow-ups (2026-11-17)

Drawn from the 0.6.1 port (§41 of `CURRENT-VS-KINDI-0.7.md`): the MTP head's CUDA graph had not
built since 0.5.0, the engine logged one line about it, `/health` still reported drafting as
enabled, `kindi verify` cannot run under `KINDI_SPEC` at all, and the only way to get a speculation
number was to start a server by hand and `curl` it. The general lesson is not "MTP"; it is that a
documented feature can be dead while every gate stays green. Four items are accepted with their gate
written first; the rest are parked with the measurement that would reopen them.

| # | idea | verdict | where it lands |
|---|---|---|---|
| 1 | **The kernel-arity check covers one kernel; make it a launch-shape contract for every kernel and every graph call site** | **A1 + A3 done 2026-11-17.** `tests/test_kernel_arity.py` now measures **105 launches** across `spec.py` (26), `dspine.py` (37), `mtp.py` (3) and `tp.py` (39) against the declared parameter sets, pins the **37** unresolvable sites (variable / list element / `getattr` kernel, or a run-time argument tuple) with a reason each instead of skipping them, pins the coverage count so a silent drop fails, and proves every measured site by mutating a copy and requiring the checker to catch it. The DSpine graph is reached: 31 `kf["..."]` lookups are resolved from the literal that builds `self.kf`, and `self.k` / `self.k8 if n <= 8 else self.k16` from their `get_function(...)` assignments (both suffix variants); the `_Planner` wrapper and `self.hmv` are the two dspine pins left. It encodes two sub-contracts found on the way: `k_tc_q8` is 7 parameters on HIP/Pascal and 8 on CUDA (the call adds `hp` conditionally), and the generated `k_*_T` wrappers (`kindi/chunk.py::_wrappers`, `params + T_ + one stride per pointer`) are declared too. **A2's block half landed 2026-11-24:** `test_no_launch_asks_for_more_threads_than_its_kernel_declares` cross-checks the literal block tuple against every kernel's `__launch_bounds__` ceiling - 28 (site, kernel) pairs measured, 135 sites pinned (96 kernels declare no bound, the rest are dynamic), both counts fixed, and a 1024-vs-256 launch is proven to be reported. **A2's grid half stays open:** a zero grid launches nothing silently, but grid elements are runtime expressions, so that needs a planner-time assertion on a GPU host | [REQ-0023](development/requirements/REQ-0023.md) |
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
| Constrained-decoding mask cost ([REQ-0027](development/requirements/REQ-0027.md)) | compute the mask once per state instead of once per token pair: either a **token-class table** (group ids by the character/prefix class the state accepts, so a mask is assembled from a few precomputed sets) or an **on-card mask** (accept test as a gather/compare kernel over the token-text table, ids returned in device memory) | **231 ms/token** for `json_schema` and **2,720 ms/token** for the unrolled `json.gbnf` on the 2B (22-token document, greedy); 57 and 463 ms/token at 200 tokens | same host, pack, document and harness (`tools/grammar_cost.py`): `json_schema` at most **23 ms/token** and the unrolled grammar at most **272 ms/token** (a 10x improvement), with the 200-token rows 57 → 6 and 463 → 46; the mask must equal today's `vocab_allowed` set for every vocabulary entry (CPU test, no GPU), keep the §39 structured-output exactness cell, and leave unconstrained decode within 1%. A miss on either row is a no-go recorded here | **done 2026-11-22 (token-class table + startup prewarm: 15 ms/token `json_schema`, 124 ms/token unrolled GBNF at 22 tokens; 3 and 20 ms/token at 200), then re-passed 2026-11-24 after profiling removed an 8,228-character probe per NFA state and a whole-vocabulary sort per mask miss: 10.0 ms/token `json_schema`, 57.1 ms/token unrolled GBNF depth 1 and 9.5 ms/token at depth 2/3, 1.7 ms/token for the schema and 7.5 ms/token for the record GBNF at 200; all rows clear the gate, the language is unchanged (0 mismatches against the pre-token-class sweep on 14 real 2B states), and the vocabulary build still runs in a background thread at `kindi serve` (`KINDI_PREWARM_GRAMMAR=0` opts out).** [2026-11-22 run](benchmarks/2026-11-22-i7-8086k-grammar-prewarm/RESULTS.md), [2026-11-24 run](benchmarks/2026-11-24-i7-8086k-grammar-mask-edges/RESULTS.md) |

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
| Tandem CPU+GPU runtime | 27B/33B models do not fit an 8 GB card; only the per-layer CPU primitive exists | an end-to-end `TandemEngine` beats GPU-only on one tier, with no token drift | the dense runtime is token-exact on 2B/9B/27B; 2B/4B/9B `cpu-first` 24K throughput is recorded as medians of three (2B read 84.3, generation 79.8/72.3/58.8/43.2/34.9; 4B read 49.3, generation 47.0/43.8/31.7/23.7/16.7; 9B read 38.1, generation 37.5/34.7/25.4/20.5/15.2 tok/s), the 27B has a rate-plus-peak row at 8K/24K (7.2-8.4 vs 34.8 tok/s at 100; 446 MiB less peak at 8K), the joint checkpoint costs 54.5-61.5 ms to save, and the 2B CPU-layer knee is 4-6 layers; tandem does **not** beat GPU-only on this card, and the full-GPU 27B still loads at 24K, so the GPU-only beat claim and a true larger-than-VRAM MoE row remain open. A CPU-only gain is measured (4.3-5.8x at 24K), and the capacity-bound Nanbeige/K6000 split beats llama.cpp's CPU+GPU split, but neither is a MoE CPU-compute result. |
| Mixed-card pairing and asymmetric TP | installer offers mixed pairs; the installer now sets `KINDI_TP_SPLIT=auto` for a pair of unequal-capacity cards (for example 6 GB + 8 GB), giving each card a complete aligned layout (FFN and output head proportional on the 27B; GDN/attention stay equal where HB alignment leaves no choice), while equal-capacity pairs keep `equal`; still no mixed-card hardware run and no `mixedx2` installer profile | equal-split mixed pair passes 27B verify/server/cache, then an asymmetric layout passes the same plus token-exact and peak-VRAM gates; see [REQ-0017](development/requirements/REQ-0017.md) and [MIXED-CARD-ROADMAP.md](MIXED-CARD-ROADMAP.md) |
| Self-hosted GPU CI runner | the GPU smoke workflow exists but has not been dispatched | one green `gpu-smoke` run on a registered runner |
| Static analysis and secret scanning | **done 2026-11-12:** secret scanning is wired into pre-commit and CI (`secret-scan` job + baseline), and the CUDA static-analysis harness (`tests/static/run_static_analysis.py`) runs in CI as the `static-analysis` job (cppcheck + clang-tidy-18, GPU-free CuPy stub) | a baseline with no new secret, plus a CI static-analysis job for the CUDA TUs |
| Sampled-peak VRAM in `kindi bench` | a larger-than-VRAM capacity claim needs the transient device peak, and the only measurement harness lived in `bench-logs/` | **done 2026-11-13:** `kindi bench --sample-vram` samples device-wide `nvidia-smi` memory on a background thread (`kindi/vram.py`) and logs the peak with its delta over the idle baseline and the phase it occurred in; `--vram-sample-s` sets the interval, and a host without `nvidia-smi` logs that and still finishes. `tests/test_vram_monitor.py` pins peak/delta, no-GPU and CLI-flag behavior. The 27B 8K tandem throughput row uses it. |
| Coverage and performance regression gates | the CPU suite is broad, but coverage and tok/s have no enforced floor | **done 2026-11-12:** the CPU job enforces a 27% `kindi` line-and-branch coverage floor (27.15% measured locally; hardware/model paths are excluded from the claim, not the denominator), and `tools/check_perf_baseline.py` gates the recorded CROSS-MODE numbers (15% tolerance, 9 cells) in the same job. A same-runner GPU throughput regeneration remains the manual lane's job. |
| Windows install documentation | native CPU and a source-tree GPU path exist, but the installer is Linux-only | **done 2026-11-12; GPU non-claim narrowed 2026-11-20:** [WINDOWS.md](WINDOWS.md) records the verified native CPU quickstart, clang/libomp setup, server commands, thread/context limits and the explicit non-claim for native Windows AMD and unmeasured GPU cases. The installer remains Linux-only. The 2026-11-19 source-tree run measures the 2B on an RTX 4060 Ti 8 GB (plain and tensor-core verify 4/4, an 8K bench and one server request), so native GPU is no longer a blanket non-claim; it is not an installer/WSL2 or general Windows GPU support claim. Windows 11, the 4B/9B/27B packs, speculation/MTP, tandem, cache restart, other cards and AMD remain unmeasured. Follow-up #1 below carries the remaining gate. |
| Structured request logging | the server logs free-form lines; request IDs and latency fields are not stable | **done 2026-11-11:** `log_json = 1` writes one JSON line per inference request with request id, status, duration, queue wait and token counts, and sets `X-Request-ID` |
| `AGENTS.md` trim and ownership index | the handoff file is large and mixes history with current instructions | **done 2026-11-12, completed 2026-11-14:** `AGENTS.md` opens with a live "Current state" section (HEAD, shipped work, open gates, shared-file ownership) and the worktree HEAD is guarded by `tests/test_agents_handoff.py`; the dated Snapshot, Handoff, full lease table, old plan and resume protocol were moved verbatim to `docs/development/AGENTS-HISTORY.md`, leaving a ~215-line live handoff (was 1,057). The live Test/Definition-of-done sections now name the real gate (`pytest -q`, `project_memory --check`, `config_reference --check`, the perf baseline, `git diff --check`) |
| Dependency and GitHub Actions pinning | CI uses floating action tags and unpinned test dependencies | **done 2026-11-11:** workflow actions are pinned to commit SHAs (`actions/checkout@11d5960`, `actions/setup-python@a26af69`, `actions/upload-artifact@ea165f8`) and `.github/dependabot.yml` opens weekly reviewed PRs for the test requirements and those pins |
| Nanbeige 33B host-layer optimization | K6000 33B is 3.56 tok/s at 8K; `nanbeige.py` already overlaps 55% of host-layer FFN rows on the CPU, but copies 13 host layers into two device slots each pass; the proposal's resident 80/20 slices, CPU drafter, CPU K/V and output-head split are not implemented | a `KINDI_NB_CPUF`/slot sweep identifies the bottleneck, then a resident FFN slice beats 3.56 tok/s at 8K with token-exact output | [TANDEM-PROPOSAL-REVIEW.md](TANDEM-PROPOSAL-REVIEW.md) |

## Follow-ups from the 2026-11-19 review

Ranked by expected value, not effort. The external AI knowledge base could not be read this session
(`$KINDI_AI_INVENTORY` is unset), so this section contains only repository-derived items; it does
not attribute any idea to the base. The first three were accepted for a measurement or a
reconciliation, not a support claim; each row records how far it got. The fourth and fifth are proposed with their
gates, written before the work, per the standing rules.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 1 | Native Windows GPU: validate the path the benchmarks already use, then either claim it or correct the docs | `docs/WINDOWS.md` says native CUDA/CuPy has not been validated and makes no claim, while `docs/benchmarks/2026-11-09-i7-8086k-rtx4060ti/README.md` records GPU runs on Windows 10 with CuPy 14.2.0 and driver 617.42. That contradiction hides a working path behind WSL2/Docker and makes the support matrix understate this host. | a clean native Windows 10/11 run of the current tree on the RTX 4060 Ti: `kindi verify` 2B/4B/9B 4/4, one streamed `/v1/chat/completions` request, pack hashes and CUDA/CuPy/driver recorded; then `WINDOWS.md` says exactly what was validated and `CARD-SUPPORT.md` keeps other Windows cards unmeasured. If it fails, record the failure and keep the WSL2-only non-claim. | **partially validated 2026-11-19; measured scope is the 2B on the RTX 4060 Ti 8 GB.** The source-tree run passed plain and tensor-core 2B `verify` 4/4, an 8K `bench` at 212.7-226.1 generation tok/s, and one server request on Windows 10 22H2 with CuPy 14.2.0 and driver 617.42. The gate's 4B/9B cells and pack hashes were not run or recorded, so this is a 2B source-tree result, not an installer/WSL or Windows GPU support claim; other cards and Windows AMD remain unmeasured. Evidence: [WINDOWS.md](WINDOWS.md#native-windows-gpu-validation-2026-11-19), [CARD-SUPPORT.md](CARD-SUPPORT.md). |
| 2 | Prefill chunk-size sweep (`KINDI_CHUNK`) | `kindi bench` measured 64% of a 4K request and 82% of an 8K request in prefill, and the chunk size is a fixed heuristic (256, or 512 on fp16-prompt cards), never tuned on this host. It is the largest remaining latency knob for long prompts and changes no arithmetic. | 2B, 4B and 9B, 4K/8K prompts, chunks 128/256/512/1024, three runs each: p50/p95 prefill and whole-request time plus sampled peak; `kindi verify` 4/4 with the winning value; record in `docs/benchmarks/<date>-<host>-chunk-sweep/`. Change a default only after the sweep; otherwise add it as an `autotune` candidate. | **measured 2026-11-19 on the 2B and 9B, and 2026-11-20 on the 4B (RTX 4060 Ti): 256 wins on the 2B and is already the default; 1024 wins on the 4B and 9B.** On the 2B, three-pass p50 at 4K/8K is **962.15**/**2602.72** ms for 256 against **1179.81**/**3015.29** ms for 128; 512 and 1024 sit between. The 4B fits the same 8,320 context: 1024 gives **2254.26**/**6331.19** ms p50 against **2326.57**/**6558.49** for 512 and **2456.43**/**6805.34** for 256, 3.1-3.5% ahead of 512 and 7.0-8.2% ahead of 256; its sampled peak is **2527** MiB against **2323** at 256 (+204), and `KINDI_CHUNK=1024` 4B verify is 4/4 (`ALL PASS`). The 9B over three passes, 1024 gives **3118.80**/**8071.40** ms p50 against **3151.39**/**8185.80** for 512 and **3414.99**/**8978.64** for 256; 1024 is 1.0-1.4% ahead of 512 and 9.5-11.2% ahead of 256. **No global default change:** 256 remains the 2B optimum; a global 1024 would make the 2B 1.6-1.7% slower and raise its peak by 159 MiB, while 512 would still regress the 2B and give up the 4B/9B best cell. 1024 is a 4B/9B-specific/autotune candidate; the 27B is unmeasured. `KINDI_CHUNK=1024` 9B verify is 4/4 (`ALL PASS`). Evidence: [chunk-sweep page](benchmarks/2026-11-19-i7-8086k-rtx4060ti-chunk-sweep/RESULTS.md) (p50/p95, peak and per-pass rows; 4B under `4b/`, 9B under `9b/`). |
| 3 | Intel M1 probe/report (`kindi intel-info --json`) | The A770 arrival gates REQ-0020; the first hour's PCI/DRM/Level Zero/oneAPI/ReBAR state is easy to lose, and the current checklist is prose. A read-only report makes M1 reproducible and gives M2 a baseline without pretending a backend exists. | CPU-only fixture tests over synthetic DRM/DRI trees; then on the A770 capture `lspci -nnk`, `/dev/dri`, `sycl-ls`, `clinfo`, Level Zero/oneAPI/OpenVINO versions, ReBAR and PCIe link state into one JSON/Markdown file under `docs/benchmarks/<date>-<host>-arc-a770/`; no model load, no support claim. Reuse the tracked fork's `discover()`/device naming only as prior art. | **done 2026-11-22 for the probe; M1 open; M2 memory path partial.** Windows CIM/registry discovery and a read-only Level Zero enumeration (1 driver / 1 device) run on the installed A770, and the CPU 2B/4B/9B baselines are 4/4 token-exact. The `kindi intel-smoke` device/shared-USM round trip and the `kindi intel-cl` OpenCL kernels pass; the SYCL `libksycl.so` path, ReBAR/PCIe and the llama.cpp SYCL baseline remain open. Evidence: the 2026-11-22 M1/M2 entries in [REQ-0020](development/requirements/REQ-0020.md), which link the run page. |
| 4 | OpenAI Responses endpoint (`POST /v1/responses`) | `kindi serve` speaks OpenAI chat, Ollama and (proposed) Anthropic shapes, but not the Responses API that current OpenAI SDKs and agent harnesses use; without it they need a proxy. It is a translation over the existing chat path, so it does not touch decoding or exactness. | app-level test through `create_app` with an OpenAI-SDK-shaped Responses request (streaming and one tool call), plus one real harness run; same tokens as the chat path; the Anthropic endpoint (REQ-0028) stays a separate dialect. | **proposed; gate only** |
| 5 | No-thinking Pi adapter for ternary models (Qwen3.8-pi pattern) | The published [Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) is a rank-32 LoRA on **dense Qwen3.8-27B**, not a Bonsai or Nemotron ternary pack, so it cannot be loaded by Kindi as a ternary adapter. The reusable part is the harness-specific no-thinking objective and the [pi-llama](https://github.com/huggingface/pi-llama) serving path; Kindi already exposes `enable_thinking: false` and `thinking_budget_tokens` on its chat path. | **`pi-nothink-v1` (12 fixed tasks, defined below):** 2B and 9B, three runs per task per arm; thinking-on `enable_thinking=true` with `thinking_budget_tokens=2048` against no-thinking `enable_thinking=false`. Pass only if the baseline reaches 9/12, no-thinking matches or beats it with no baseline-passing task regressing, total completion tokens are <=75% of baseline (no task >125%), turns do not increase, tool calls are 100% valid, no-thinking emits no reasoning, and `kindi verify` is 4/4. Training is not part of this gate. | **proposed; gate only.** No adapter, fixture or benchmark exists. The external adapter is Qwen3.8-27B-specific and its published pass/token figures are its author's, not a Kindi measurement; no ternary adapter, pack or benchmark exists here. |

### pi-nothink-v1 gate (proposed)

`pi-nothink-v1` is a fixed 12-task local proxy for the Pi harness, not the Pi benchmark itself.
The suite is frozen before the first run in `tools/pi_nothink_tasks/tasks.jsonl`; each task names
one fixture directory, one literal prompt, the fixed tool schema and one deterministic
`check.py`. A run copies the fixture into a fresh temporary workspace, serves one `kindi serve`
process, drives the OpenAI chat/tool loop with the same system prompt, and scores the workspace
with `check.py`; there is no model judge and no network.

**Fixed conditions.** 2B and 9B packs on the RTX 4060 Ti 8 GB; `KINDI_TC=0`, `KINDI_TP=1`,
`KINDI_SPEC=0`, `KINDI_HEAD=exact`, `KINDI_CTX=8192`, `KINDI_KV=q8_0`; `temperature=0`,
`max_completion_tokens=4096`, `max_turns=8`, `parallel_tool_calls=false`; tools are
`read_file(path)`, `write_file(path, content)`, `apply_patch(unified_diff)` and
`run_shell(command)` (30 s timeout, 8 KiB output cap). Three runs per task per arm. The
thinking-on arm sets `enable_thinking=true` and `thinking_budget_tokens=2048`; the no-thinking
arm sets `enable_thinking=false`. 27B/Lightning are not in v1; they get separate rows when their
packs exist.

| id | group | fixture and fixed prompt | deterministic pass check |
|---|---|---|---|
| `tool-single` | tool call | `get_weather` fixture; prompt: "Use `get_weather` to get the weather for Paris. Then answer with the temperature and sky condition." | exactly one `get_weather` call with `city=Paris`; final answer contains `18` and `clear` |
| `tool-two` | tool call | two-city fixture; prompt: "Use `get_weather` for Paris and Rome, then summarize both." | calls for exactly Paris and Rome; both conditions in the final answer; at most 3 turns |
| `tool-retry` | tool call | `get_weather` rejects `Pari`, accepts `Paris`; prompt: "Call `get_weather` for `Pari`. If it reports an unknown city, correct it to `Paris` and retry once, then answer." | first call `Pari`, second `Paris`, no third call; final answer present |
| `tool-none` | tool call | arithmetic prompt: "What is 17 + 25? Answer directly and do not call a tool." | zero tool calls; final answer is `42` |
| `repair-python` | repair | `add.py` returns `a - b`; `test_add.py` expects `2 + 3 == 5`; prompt: "Run `pytest -q`, fix `add.py` so the test passes, and rerun `pytest -q`. Do not edit the test." | `pytest` exits 0; `add.py` changed; test file hash unchanged |
| `repair-shell` | repair | `run.sh` uses unquoted `$FILES` with a space-containing filename; prompt: "Run `./run.sh`. Fix its quoting so filenames with spaces work, then rerun it. Do not rename the input files." | script exits 0; both files processed; only `run.sh` changed |
| `repair-json` | repair | invalid `config.json`; `check.py` requires `name=kindi` and `enabled=true`; prompt: "Make `config.json` valid JSON while keeping `name` as `kindi` and `enabled` as `true`; run `python check.py`." | JSON parses; both values present; `check.py` exits 0 |
| `repair-patch` | repair | `mul` is `a + b`; `fix.patch` has stale context; `test_mul.py` expects `12` for `3 * 4`; prompt: "Apply `fix.patch` with `apply_patch`. If it does not apply, repair the change so `python test_mul.py` passes; do not edit the test." | test exits 0; `mul(3, 4) == 12`; test hash unchanged |
| `terminal-find` | terminal | `src/alpha.py` and `src/beta.py`, with `NEEDLE` only in beta; prompt: "Find the file under `src/` that defines `NEEDLE` and report its relative path. Do not edit anything." | answer is `src/beta.py`; no files changed |
| `terminal-run` | terminal | `tests/test_ok.py` with three passing tests; prompt: "Run `python -m pytest -q tests/test_ok.py` and report how many tests passed. Do not edit anything." | answer reports `3` passed; no files changed |
| `terminal-write` | terminal | empty `out/` directory; prompt: "Create `out/answer.txt` containing exactly `kindi-pi-ok` followed by one newline. Report its SHA-256." | file bytes exact; reported hash equals the file hash |
| `terminal-clean` | terminal | git fixture with `keep.txt`, `delete.me` and `nested/delete.me`; prompt: "Delete only `delete.me` in the repository root. Leave `keep.txt` and `nested/delete.me`; show `git status --short`." | root file gone; `keep.txt` and `nested/delete.me` present; status lists only that deletion |

**Metrics.** A task passes when `check.py` exits 0 in at least 2 of its 3 runs. `pass_rate` is
tasks passed / 12. `tokens` is the sum over tasks of the median `usage.completion_tokens` across
the three runs (reasoning and content included); `turns` is the same sum of median assistant
messages, where a tool-result message is not a turn. Tool calls, invalid calls and calls to a
tool not in the schema are counted separately. `reasoning` is the non-empty reasoning channel
(or thought text in `content`); the baseline may have it, no-thinking must not.

**Pass/fail threshold.** The gate passes only when all of these hold for the same pack and build:
(1) the thinking-on baseline is calibrated at **>=9/12**; below that, record "suite too hard" and
do not compare; (2) no-thinking `pass_rate` is **>=** thinking-on, with **zero** baseline-passing
tasks regressing; (3) total completion tokens are **<=75%** of thinking-on, and no task passed by
both arms uses **>125%** of its baseline tokens; (4) total turns are **<=** thinking-on, and no
task uses more turns; (5) invalid and unoffered tool calls are **0** (100% valid); (6) no-thinking
emits no reasoning, and `kindi verify` is 4/4 on the same pack. Record the result under
`docs/benchmarks/<date>-<host>-pi-nothink/` with exact prompts, `KINDI_*` environment, pack
hashes, per-task run rows and the three medians.

Training is not authorized by this gate. A failure is a no-go result, not a reason to start a
LoRA in this task; any later adapter proposal needs its own gate and must not merge the Qwen3.8
LoRA into a Bonsai pack.

## Pascal follow-ups (2026-11-20)

Three gate-only rows from the Pascal research. None is implemented, measured or a support claim:
no sm_61 run exists, no kernel changed, and the proposed items stay "gate only" until the runs
below exist. They sit beside the Cards notes ([CARDS.md](CARDS.md), [OLD-CARD-RIG.md](OLD-CARD-RIG.md)).

Provenance: the external P100 9B user report (426,752-token context, 41.3 tokens/s; scuderiapk,
GitHub issue #22, recorded in [CARDS.md](CARDS.md)) and every llama.cpp row in
[BENCHMARKS.md](BENCHMARKS.md) are **theirs**, not Kindi measurements. The 9B report is sm_60,
not sm_61, and neither source is a gate baseline or evidence for the rows below.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 1 | `KINDI_TC=1` + MTP on sm_61 (2B) | `pascal8` leaves `KINDI_TC` off, and `docs/MTP.md` says MTP is sm_80+ only, but the attach path has no sm_80 check: with `KINDI_TC=1` the sm_61 `tcp.cu` dp4a plans use the same tc-layout weights the head's output GEMV reads. Forcing the existing path tests that contradiction without a new kernel. | On a P106-100 or P4000, 2B pack and a registered MTP head: `KINDI_TC=1 KINDI_SPEC=mtp KINDI_MTP_HEAD=<registered head>`; require `/health` `drafting.mode=mtp` (not a fallback), `kindi verify --spec mtp` 4/4 token-exact, and `drafted`/`accepted` > 0; then three passes at 512/2K/8K against `KINDI_SPEC=0` on the same TC path and `KINDI_TC=0`. Exactness/attach is a pass, not a speed claim: adoption needs p50 generation >=1.05x at 512 and >=1.00x at 2K; otherwise record the measured ratio as a no-go. | **proposed; gate only.** No sm_61 run exists; the P100/sm_60 row and the sm_80+ MTP rows are not evidence. `docs/MTP.md`'s sm_80-only sentence is corrected only from a result. |
| 2 | CUDA graph replay on sm_61 (2B, T=1..8) | CUDA `Graph` is already the default for TC plans, but `KINDI_TC_GRAPHS` only selects `Replay` on HIP; no sm_61 graph run is recorded. A graph that builds but replays a stale token/position, or fails on the 470 driver, is a correctness/portability bug that a compile check cannot catch. | On a P106-100 or P4000, `KINDI_TC=1` and `KINDI_TC_TMAX=8`: build and launch the T=1..8 plans; `kindi verify` 4/4; ten identical greedy completions from one process; at 512/2K, three passes against a test-only eager node-by-node launch of the same plans: graph p50 decode must be <=1.02x eager and token-identical. Record graph build time, first-request latency, driver/CUDA version and any `cuGraph*` return code. A failed replay is a no-go; do not hide an eager production path behind a flag. | **proposed; gate only.** No sm_61 graph run or eager A/B exists; CUDA graphs remain the TC default until the gate says otherwise. |
| 3 | 2B/9B chunk and tile-shape sweep on sm_61 | The 2026-11-19 `KINDI_CHUNK` sweep is RTX 4060 Ti; Pascal's SM count, bandwidth and dp4a path differ, so its 256/1024 optima need not transfer. On the 2B/9B, `KINDI_CHUNK_TILE=64` switches the chunk attention to its small tile; `128` only selects the 2-bit `k_mm_g4` tile, which these 5-trit packs do not use. Chunk size, attention tile and split are the shapes to sweep. | On a P106-100 or P4000, 2B and 9B (if 8K fits; otherwise record the reduced context), 4K/8K, three passes: first `KINDI_CHUNK` 128/256/512/1024; then, at the best chunk, `KINDI_CHUNK_TILE` 64/128, `KINDI_CHUNK_TS` 32/64 and `KINDI_CHUNK_SPLIT` 1/4/8 one variable at a time, plus one combined cell. Record p50/p95 prefill and whole-request time, sampled peak and the effective settings; the winning cell must pass `kindi verify` 4/4. Change a default only for a repeatable >=5% prefill win at 4K or 8K, with no >1% decode regression and no peak that reduces the fitted context; otherwise record the table and leave the defaults. | **proposed; gate only.** The 4060 Ti 2B/9B sweep is not Pascal evidence; `KINDI_CHUNK_TILE` is included as the 64/128 attention-tile test; its 128 setting changes only the 2-bit chunk GEMM. |

## Follow-ups from the 2026-11-20 review

Ranked by expected value, not effort. `$KINDI_AI_INVENTORY` was unset in this session's shell and in
both Windows scopes, so the external base was not read and no row below is attributed to it. These
rows come from this tree's current state, the refreshed v0.6.1 comparison, the 9B default-vs-TC page,
the grammar-cost page and the handoff tests. Each is **gate only**: no implementation was started, no
support claim is made, and no number below is a new measurement.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 1 | Release provenance and a Python 3.6 gate | The rebuilt image now passes `/health` and one chat, but `/health` still reports `commit: null`, no image is published, and the earlier Python 3.6 failure was a 3.7-only import no CPU gate catches. The release checklist is prose, so a stale version string, archive hash or image digest can pass review. | A clean container build stamps the commit and digest into the image; `python3.6 -m compileall kindi` (or an import smoke) runs in the container/CI build; one `kindi release verify` checks the version pins, archive SHA-256, image digest, `/health` commit, one chat and the claims gate; two clean archive builds are byte-identical. Fail on `commit: null`, a mismatch or a missing artifact. | **proposed; gate only.** No image pushed. |
| 2 | One effective runtime-configuration contract | Two diagnostics bugs this week had the same shape: the default path named the wrong engine, and the HP line was emitted before the tensor-core layout was confirmed. `/health` also used to echo the requested speculation mode instead of the effective one. Fixing each site leaves the next site free to drift. | One resolver returns the engine path, TC/HP state, KV format, effective spec mode and head digest, chunk size, thread count and VRAM budget/reserve. The startup log, `/health`, `kindi info`, `kindi plan` and `bench --json` all read that object. CPU fake-engine tests assert the surfaces agree; live 2B default and 9B TC+HP smokes match the log and `/health`; no new per-request probe. | **landed 2026-11-22 for the loaded-process surfaces.** `kindi/runtime.py:describe` resolves the engine, TC/HP state, K/V format, context, effective speculation mode and head digest, chunk, threads and VRAM budget/reserve in one object; the startup log, `/health` (new `runtime` block), `kindi info` and `bench --json` read it. Fake-engine tests pin the engine-vs-request fallback (a requested fast head on a no-fast-head engine reports exact) and `/health` publishes the same `head`/`drafting` values. `kindi plan` deliberately keeps its pre-load planner output because no engine exists there. Live 9B CPU smoke (2026-11-22, `KINDI_DEVICE=cpu`): the startup log and `/health` agree on `engine=Engine (qwen35)`, `tc=no`, `ctx=24576/24576`, `spec=off`, `head=exact`, 6 threads and `vram=0/512 MiB`; the live 2B default and 9B TC+HP smokes were not run because this host has no CUDA device. |
| 3 | Handoff and comparison facts gate | `AGENTS.md` names `3b9224a` while HEAD is `ee2ba46`. `tests/test_agents_handoff.py` accepts HEAD's parent even when HEAD did not touch `AGENTS.md`, so the stale line passes. The comparison refresh also required manual re-basing of the commit, file and line counts in several places. | The handoff test accepts a parent hash only when `git show HEAD -- AGENTS.md` adds it; otherwise the live line must name HEAD. A small facts block (HEAD, test counts, claims gates/numbers, `v0.6.1..HEAD` counts, tracked files, release status) is generated from git and checked in `AGENTS.md` and the comparison header; a mismatch fails. A synthetic repository where HEAD moves without an `AGENTS.md` change must fail the test. | **proposed; gate only.** The current tree fails the first clause until the line is refreshed. |
| 4 | Whitespace-bounded constrained decoding | The mask now clears its 10x gate, but a whitespace-permitting schema still let the greedy 2B spend all 200 tokens on 9 characters. [API.md](API.md) tells users to bound or omit whitespace instead of the engine doing it, so a successful grammar-cost win still leaves a broken reply. | A fixed 200-token JSON document produces valid JSON with non-whitespace content inside the budget; every schema-legal output stays legal; unconstrained decode stays within 1%; the §39 structured-output exactness cell still passes; record before/after token counts and wall time. | **proposed; gate only.** The cost half is done in §39; this is the separate semantics/policy change named there. |
| 5 | VRAM ceiling and reserve hardware gate | `ee2ba46` added `--vram-budget-mib` and `--vram-reserve-mib` with unit tests, but no tracked page measures them and the comparison refresh explicitly did not re-benchmark them. A budget that silently reduces context, or an over-budget run that still loads, would be a memory-safety surprise. | On the RTX 4060 Ti, 2B and 9B at 8K (27B only if it fits): budgets 6144/7168 MiB, reserves 0/512/1024; `kindi verify` 4/4, `bench --sample-vram` peak at or below the budget, `ctx_requested`/`ctx_reduced` and the selected `gdn:N` recorded, and an over-budget request refuses rather than trimming silently. Three passes before quoting a rate; page and claims entry. | **proposed; gate only.** No budget/reserve benchmark exists. |
| 6 | `kindi selftest` for an installed engine | The container smoke is a hand-written `/health` plus chat, and `kindi verify` needs reference tokens. A user who installs a pack has no single command that proves the pack, tokenizer, engine, effective path and server route work together. | `kindi selftest --json` loads the selected pack, runs one short deterministic generation, calls `/health` and prints the effective configuration, pack hash and timings; exit nonzero on any failure. Fake-engine CPU tests cover missing pack, bad tokenizer and a failed generation; 2B CPU and GPU runs pass. The installer may call it after install unless `--no-selftest` is given. | **landed 2026-11-22 for CPU; GPU and the installer call open.** `kindi selftest [--json FILE]` loads the selected pack once, checks `/health`, runs one short greedy chat request through the real in-process route and prints the effective runtime state, pack hash and timings; missing pack, load failure, non-200 health and an empty generation all exit non-zero. Fake-engine tests cover those paths plus the JSON report and CLI wiring. Live `KINDI_DEVICE=cpu` runs: 2B **PASS in 24.5 s** (`/health` 200 in 5.3 ms; 2 tokens in 434 ms), 9B **PASS in 112.1 s** (`/health` 200 in 5.7 ms; 2 tokens in 1,913 ms). No GPU run (this host has no CUDA device); the installer call and `--url` landed 2026-11-23. The installer now runs `kindi selftest --url http://127.0.0.1:PORT` inside each serving container after `/health` (`--no-selftest` opts out), and `--url` is standard-library only and loads no pack, so no second copy competes for the card. The in-process mode calls the ASGI app directly, so it no longer needs fastapi's test client (the images install neither `httpx` nor `requests`). Covered by 10 selftest tests and 6 installer fixture tests; no GPU run. |
| 7 | Tensor-core/HP default-selection matrix | The 9B default is 4/4 at 95.7 tok/s versus 4/4 at 119.5 with TC+HP; the 2B default is already 253 tok/s, while 4B and 27B have no matching default-vs-TC+HP row. The auto-selection rule is heuristic, so a pack can leave a large win unused or flip tokens. | 2B, 4B, 9B and 27B: default 5-trit, TC and TC+HP (where offered) each `kindi verify` 4/4; three-pass p50 decode at 512/8K plus sampled peak; select HP only when it is both exact and faster; record in a benchmark page and claims. A pack that fails exactness stays on the default. | **proposed; gate only.** The 9B default-vs-TC cell is measured; the full matrix is not. |
| 8 | Adaptive fast-head shortlist | The opt-in fast head is fixed at 256 clusters: **99.4%** top-token agreement and about 15% faster than exact; 128 clusters give 98.1%. A margin-triggered expansion could push the opt-in path toward 99.9% without giving up the speedup, and there is no per-model quality metric today. | 2B/4B/9B, greedy 256-token prompts: exact versus fast at 128/256/adaptive, top-token agreement, p50 decode and sampled peak. Pass at **>=99.9%** agreement with **<=5%** speed loss against fixed 256; the default stays exact; record the result per model. | **proposed; gate only.** No code changed. |

**External proposal review (2026-11-20).** The 20 external proposals were checked against this tree. Model breadth is already in the long-term goals and small-MoE tables; parallel slots remain a measured NO-GO (0.69x at B=4); `/health` with live progress, `/metrics`, JSON logs and SSE keepalive already cover status and long-prompt progress; CUDA graphs (TC default; Pascal replay is follow-up #2), ternary and fused kernels, fused attention, expert prefetch, DSpine/DSpark and DeepSeek NUMA placement are already present or gated; AVX-512/AMX remain absent from the CPU engine and now have their own gate below; GPU CI remains an open gate; the container base is deliberately pinned for Kepler, and release provenance is item 1 above. ARM/Mac is out of scope. No duplicate rows were added.

## Agent speed experiments (2026-11-20)

Two gate-only rows from the 2026-11-20 review. Both use existing machinery (the prompt cache and
the attached drafters); no implementation was started, no code changed and no speed claim is made.
The adaptive fast-head row (#8 above) and the zero-tile row (#5 below) are unchanged and are not
duplicated here.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 1 | Stable-prefix agent prompt reuse | Agent clients resend a long system prompt and tool schema on every turn, then append the assistant message and tool result. `kindi/cache.py` already has checkpoints, parked conversations and saved openings, and the API reports `cached_tokens`, but no roadmap gate proves that a stable agent prefix is reused rather than silently re-read when the suffix changes or the server restarts. | 2B and 9B server (27B only if it fits), 8K context: a fixed system+tools prefix of at least 2K tokens followed by four turns that append an assistant message and tool result; fresh state, cache on versus `KINDI_PROMPT_CACHE=0`. On every warm turn require `cached_tokens >= S - 256` (S = stable-prefix tokens; 256 is the checkpoint quantum) and `prompt_tokens - cached_tokens <= new suffix + 256`; after one server restart, the first matching request meets the same counts. Cache-on and cache-off replies must be greedy-token-identical on all four turns, and warm p50 prompt-read must be **>=1.10x** the cache-off read. Record per-turn prompt/cached/re-read, the stable-prefix hash and the restart loader line. A miss is a no-go for a new stable-prefix lane. | **proposed; gate only.** No code change; the existing cache may already pass. |
| 2 | Task-class draft policy | `DraftPolicy` learns one acceptance history and `auto` picks a drafter from the target and context, while agent traffic mixes tool calls, code and prose whose acceptance differs. A policy that selects the already-attached n-gram, MTP or 2B drafter per task class can stop paying rejected drafts in one class without under-drafting another; no new drafter or training. | 2B with the MTP head and n-gram; 27B with the 2B drafter at 8K (record any context reduction), four fixed prompts per class (tool-call, code, prose), 128 greedy tokens, three passes. Compare `KINDI_SPEC=auto` + `KINDI_DRAFT_POLICY=1` with a per-class controller over the same drafter set. Require `kindi verify` 4/4 and token-identical output to no speculation on every run; tool calls parse with `finish_reason=tool_calls`. Overall p50 decode must be **>=1.05x** the single-policy baseline, at least two classes **>=1.05x**, and no class **<0.98x**; record kept/drafted, effective drafter and k per class. A class with no measured win keeps the current policy; if no class wins, no-go. The benchmark labels the classes; a production classifier is a separate gate. | **proposed; gate only.** No new drafter, classifier or code; existing drafters only. |

## Ternary speed proposals (2026-11-20)

The ten proposals were checked against the current 5-trit lane layout (`kindi.cu` `g5_core`/`g5_tables`, `chunk.cu` `g5n_unpack`/`k_mm_g5q`), the fast head, the drafters and the CPU engine. Four items are new and bounded enough to gate; the rest are already implemented, measured no-go, or would need model retraining. Every row is **gate only**: no code changed and no speed claim.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 2b | 5-trit base-3 decode on sm_30/sm_35 | The 5-trit lane path is selected from sm_60; K420/K1200 still use the 2-bit `k_g4` unpack path. A 5-trit table decode would read fewer weight bytes, but table lookups may be slower on Kepler. | On a K420 and a K1200, 2B (4B only if it fits): current 2-bit path versus 5-trit table decode; require `kindi verify` 4/4, p50 decode **>=1.05x** and peak VRAM no higher; otherwise no-go. | **proposed; gate only.** No sm_30/sm_35 run; the lane kernel's hardware support is not assumed. |
| 5 | Zero-tile density audit and structured skip | The claim that about two thirds of ternary weights are zero is unverified, and the kernels skip nothing today. A useful skip needs all-zero tiles, not isolated zeros, so measure before any model or kernel work. | Scan the 2B/4B/9B packs (27B if present) at 8x8, 16x16 and 1x128 tile granularity; report zero and all-zero tile fractions per tensor and overall. Only if an aligned tensor has **>=15%** all-zero tiles, prototype a metadata skip; require `kindi verify` 4/4, p50 decode **>=1.05x** and no peak-VRAM growth; otherwise record a no-go. No retraining or pruning. | **proposed; gate only.** The density number is not measured here. |
| 9 | Integer fast-head candidate path | The exact head is a large per-step cost, and the current fast head still scores fp32 centroids and rows. The existing `k_mm_g5q`/dp4a path can score int8 activations against int8 trits, so an opt-in integer shortlist is testable without a new dtype. | 2B/9B (4B if it fits), 256 greedy tokens: exact head versus the current fast head versus the int8 candidate path. Pass at **>=99.9%** top-token agreement and p50 decode **>=1.05x** the current fast head; the default stays exact, and the lossy path gets no `kindi verify` claim. A miss on either row is a no-go. | **proposed; gate only.** Builds on follow-up #8; shortlist size and head arithmetic are separate variables. |
| 10 | AVX-512/VNNI CPU ternary path | `kcpu.c` is AVX2/FMA only. VNNI `vpdpbusd` can turn ternary x int8 into integer dots, but int8 activations are lossy and no VNNI host exists here. The W-2133 is AVX-512F without VNNI, so compiler auto-vectorization is a separate first check. | Stage 1 on a W-2133: current AVX2 versus `-mprefer-vector-width=256` and compiler AVX-512 auto-vectorization on 2B/4B/9B; require 4/4 exact and **>=1.05x**, else stop. Stage 2 on a VNNI host: opt-in ternary x int8 GEMV with `vpdpbusd`; require **>=99.9%** top-token agreement against fp32 and p50 decode **>=1.2x**; the default stays exact, with no `kindi verify` claim for the lossy path. AMX only after VNNI. A miss is a no-go. | **proposed; gate only.** Stage 1 hardware-blocked; stage 2 needs a VNNI host. |

The rest are already present or measured: (1) decode reads packed 5-trit bytes through 243-entry tables without unpacking, and the prompt GEMM unpacks each weight tile once for up to 128 tokens; (2) the 5-trit base-3 lane layout is already in use on sm_60+ (1.75 bits per weight including scales), while K420/K1200 still use the 2-bit path, which is why 2b is gated; (3) those tables replace per-weight FP multiplies in decode; (4) the 5-trit decode and q8 prompt path scale group sums once; (6) continuous batching is a measured **NO-GO** (0.69x at B=4); (7) the prompt GEMM already reuses a weight tile across tokens, while batch-1 decode has no reuse; (8) the 2B drafter, DSpine/DSpark and the adaptive draft policy already exist. Full int8 decode is not added: it is lossy and decode is weight-bandwidth-bound; only the integer head is gated. No duplicate rows were added.

## Tandem proposals (2026-11-20)

The nine proposals were checked against the dense `TandemEngine`, the measured 27B/24K decomposition (CPU 74.7 ms + GPU 8.1 ms + transfers 3.09 ms + head 17.6 ms = 103.4 ms/token) and the MoE/cache paths. Two are new and bounded; the rest are already implemented, already gated, or invalid for a batch-1 dependency chain. Every row is **gate only** unless its status says a measurement was made.

| # | item | why it is worth doing | first gate | status |
|---|---|---|---|---|
| 2 | Exact head row split | The **`cpu-first` GPU head** is 17.6 ms of a 103.4 ms token (17%) and the largest non-compute slice; `gpu-first` puts the whole head on the CPU, so that 17.6 ms figure is not its baseline. A disjoint vocabulary-row split lets both devices compute exact rows and merge the argmax, unlike an approximate shortlist. | 27B, 24K, `gdn:12`, measured in **both placements separately**: `cpu-first` with its GPU head (the 17.6 ms baseline) and `gpu-first` with its CPU head. In each placement, split CPU/GPU head rows 50/50, 25/75 and 75/25 against that placement's unsplit head; `kindi verify` 4/4 against the full-GPU and current CPU/GPU head; p50 decode **>=1.10x** the same placement; record unsplit and split `head_ms`; peak no higher. The `cpu-first` result does not stand in for `gpu-first`; a miss in either placement is a no-go. | **NO-GO on the 9B proxy for `cpu-first` only (2026-11-20); `gpu-first` remains gate-only, not measured.** Opt-in exact row split implemented (`KINDI_TANDEM_HEAD_SPLIT`); 9B/24K `cpu-first`, 16 CPU layers and the plain GPU: 50/50 `verify` 4/4 but decode **0.71x** the same placement, and 25/75 is **0.97x**, so the 1.10x gate misses. Peak is 126/99 MiB lower at 50/25 because the CPU half replaces the GPU half in host RAM; total model memory is unchanged. The 27B pack is absent and the 2B/4B tie the head to the embedding, so neither is a testable cell. Evidence is the local head-split run log; no further work on `cpu-first`, while the separate `gpu-first` gate stays open. |
| 1 | Row-split ternary GEMM across CPU and GPU | `TandemEngine._run` is a strict layer ping-pong: one owner computes a whole layer while the other is idle. A row split of one large FFN matrix could run both devices concurrently and merge partial rows. | 27B/24K `gdn:12` (or 9B if it fits): one large FFN matrix, CPU/GPU row split 50/50, exact 4/4 against the current tandem, p50 decode **>=1.10x**, partial-sum transfer/overhead **<=10%** and peak no higher; otherwise record no-go. | **NO-GO on the 9B proxy (2026-11-20).** Opt-in probe `tools/row_split_gemm.py` splits one `ffn_down` matrix by rows. A CPU-owned standalone GEMV does speed up: 25% CPU / 75% GPU is **1.52x**, 50/50 **1.18x**; but the CPU-owned layer is only **1.01x** at 25/75 and **0.97x** at 50/50, and a GPU-owned layer is **0.43-0.52x** because its CPU slice and sequential merge cost more than the GPU work they replace. The 27B/24K row, a full tandem decode and `kindi verify` 4/4 were not measured (no 27B pack; no TandemEngine integration). Transfer/overhead and peak are not claimed. Evidence is the local row-split run log; no further work. |

The rest are already present, already gated, or not a viable batch-1 pipeline: (3) layer pipelining has a hard dependency, layer n+1 needs layer n, and cross-token overlap is the measured batching NO-GO; (4) sampling, stop strings, tokenization and detokenization are already host-side (`sampling.py` keeps temperature/top-k/top-p on the host); (5) the MoE CPU lane and router/copy plan exist in `moe_tandem.py`, but the CPU lane measured 3.06x slower than the copy lane and the Lightning backend/pack is the open blocker; (6) DeepSeek already runs GPU dense plus CPU experts with dual-Xeon NUMA (`dsv41 numa --rowsplit`); (7) a CPU neural drafter was rejected in the Nanbeige review because the 2B GPU drafter is faster and P0/P1 must first show idle CPU capacity; (8) tokenization is already host-side and does not remove the GPU prompt read, which is the long-prompt cost; (9) checkpoint host copies are pinned, saved-turn disk writes and idle checkpoint copies already use background threads, and a fully async snapshot would need double-buffering, not a quick win. Double-buffered staging is already the next step in [TANDEM-OVERLAP-DESIGN.md](TANDEM-OVERLAP-DESIGN.md) for the measured 3.09 ms/token; no duplicate row was added.

## Standing rules

- Every experiment gets a numeric gate before implementation, and the gate is written down before
  the measurement.
- A failed gate produces a no-go entry here and in the experiment's own page; it does not produce
  production code behind a hidden flag.
- Features that cannot be token-exact are opt-in, named as lossy, and pinned by a test.
