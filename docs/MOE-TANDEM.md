# MoE expert-cache tandem path

Status: **design plus a model-free cache prototype; Raon-1 mini R0 and a Gemma 4 26B-A4B
adaptation recorded, 2026-11-12. **The MoE placement strategy has its first runnable surface:
`kindi plan` on a `kindi_lightning` pack reports the expert-tier strategy (dense core + shared
expert on the card, routed experts tiered card / pinned host / mmap, last-used decode slots) from
the GGUF header and the engine's own fit arithmetic, with one reason per choice and the actual
`bench`/`verify` commands for tenants who don't own the card (`--assume-vram-gib`, `--usage`,
`--moe-cache-mib`, `--dcache`).** The first planned backend target is still Raon-1 mini;
`kindi/expert_cache.py` implements the copy-only slot, pin, LRU, transient-slot and metrics state
machine, and `tests/test_moe_cache.py` covers it with synthetic slots. The Gemma 4 profile is a
second backend contract in [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md), not a second implementation.
Its execution-level copy and combine contract is in
[GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md); the policy-preserving cache-to-copy
seam is in [EXPERT-CACHE-COPY-INTERFACE.md](EXPERT-CACHE-COPY-INTERFACE.md). There is still no MoE
backend, converter, pack, GPU copy path, catalog entry, installer offer or measured inference
result in Kindi. This document is not a support claim.**

> **Not a backend:** `kindi/expert_cache.py` is only the model-free slot/LRU/metrics prototype. It has
> no router, no pack, no engine wiring, no GPU copy path and no model support; do not treat this page as
> a Gemma, Raon or MoE support claim.

This is the design for a **CPU+GPU expert-cache tandem** path, not another dense layer split. The
dense split is already implemented in [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md). The MoE model contract
and long-term acceptance gates are in [RAON-1-MINI.md](RAON-1-MINI.md); the older,
broader sketches in [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) are historical where they
conflict with this document.

## 1. Why a separate design

A dense model needs every weight on every token, so `TandemEngine` can split complete layers between
the CPU and one GPU. A sparse MoE keeps a dense core (attention, router, norms, shared expert and
output path) but routes each token to a small number of experts. The routed experts are too large to
keep all of them in VRAM, and only a few are needed per token.

The cache is therefore **weight-oriented**, keyed by `(layer, expert_id)`, not token-oriented. It is
not a prompt cache and it is not a layer split. The GPU owns the core and a byte-budgeted set of hot
expert weights; host RAM owns the whole expert set. A miss must fetch or compute the **complete**
selected expert before the router-weighted sum is formed.

**Where the dense tandem ends.** `kindi/tandem.py` splits *complete dense layers*, so it cannot run a
routed layer: every token still needs every expert. As of 2026-11-17 that is enforced rather than
implied. `TandemEngine` refuses any manifest whose `backend` is not the dense Bonsai family, and names a
MoE pack's own split (expert tiers, not layers) instead of splitting the GDN recurrence of a model that
does not have one; `make_engine` turns a `--tandem` flag on a `kindi_lightning`/`kindi_dsv41` pack off
with a log line pointing at `kindi plan`, so a config that sets tandem globally cannot silently build
the dense engine for a MoE. The equivalent split for a MoE is the one its own engine already runs, and
`kindi plan` reports it, including the decode-slot trade (each slot displaces one expert per layer, and
`Usage.coverage()` compares the measured picks the card serves with the slots against the same bytes as
residency). None of that is new execution: the MoE backends below are still design and prototype.

The first backend must answer three questions before any performance work:

1. Is the router exactly the released router, including its normalization and tie rule?
2. Can a cold, warm, evicted and reloaded expert produce the same tokens?
3. Does a cache miss ever change arithmetic, or only latency?

## 2. First real backend

The pilot is **Raon-1 mini**. R0 found the released checkpoint and verified the contract; the full
manifest is in [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md):

| item | verified value |
|---|---|
| total / active parameters | 28,783,218,602 header elements / 4,317,319,082 active |
| routed experts | 128, top-6 per token, plus 1 shared expert |
| attention | 18 RDA linear-attention + 7 gated MLA layers; NoPE |
| context target | 262,144 tokens |
| target host | 8 GB NVIDIA GPU, 32 GB-class host RAM |
| base weight format | BF16 core + MXFP8 E4M3 routed experts, 8,616,960 B per expert |
| Q4 reference file | 19,686,434,240 B, lossy; not an exactness reference |

R0 also fixed the tensor names, layer schedule, hidden size, router arithmetic and license. The
router has no explicit tie rule; the HF reference expands the MLA cache to 175 KiB/token while the
published compressed target is 7.875 KiB/token. Both remain R1 validation items, not assumptions.

A second profile, **Gemma 4 26B-A4B**, is adapted in [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md). It is
the local 26B artifact's contract, not a second cache implementation. It selects **top-8 of 128**
experts in each of 30 layers, uses a fused gate/up expert tensor, runs an ordinary dense MLP in
parallel with the routed branch, and combines in expert-id order. Those differences are normative
for a 26B backend; the Raon values must not be reused by default. The eight-slot host-to-GPU
copy lane and the ordered-combine kernel contract are in
[GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md).

A backend qualifies as a **real MoE backend** only when it can execute the full layer schedule
(router, selected experts, shared expert, attention and output) against a pinned reference. Reading
MoE metadata from a GGUF is not a backend.

> **Gemma 4 is the second profile, not the Raon pilot:** the local `gemma4:26b` file is a Q4_K_M
> MoE, but the current Gemma 4 backend is E4B-only (42 layers, width 2560), rejects Q5_1 and has no
> router or expert path. Its measured contract and cache adaptation are in
> [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md); the local file's inventory is also recorded in
> [FP8-BF16.md](FP8-BF16.md). It still needs a separate 26B backend and a reference. The historical
> Nemotron MoE row in [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) has no engine either.

The six small MoEs Kindi recognizes (Cascade 2 converter/validator, Gemma 4 26B, Qwen3-Coder,
Qwen3.6, North Mini Code, Recursive Flint) and the milestone each is missing are tabulated in
the small-MoE compatibility section of [ROADMAP.md](ROADMAP.md).

## 3. Architecture

```
┌──────────────────────────── host RAM (resident, memory-mapped pack) ─────────────────────────┐
│ all routed experts, shared expert, dense core weights, tokenizer/metadata                    │
│                         bounded pinned staging buffers                                      │
└───────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        │ H2D expert weights, or CPU expert output
                                        ▼
┌──────────────────────────────────── GPU (8 GB target) ──────────────────────────────────────┐
│ dense core: attention + router + shared expert + norms + output                               │
│ KV / linear-attention state                                                                  │
│ expert cache: fixed-size slots keyed by (layer, expert_id)                                   │
│ workspaces and per-selected-expert output buffers                                            │
└───────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                        │ router plan
                                        ▼
                         CPU worker pool: optional miss compute
                         (enabled only after exactness parity)
```

The CPU is a **miss/staging engine**, not a layer owner in the first path. It may copy weight bytes,
or (later) compute an expert. All layers' attention, router, shared expert and combine remain on the
GPU. This keeps the prompt-cache state in one place and avoids a second joint-checkpoint design.

A later phase may split complete MoE layers between CPU and GPU, but that composes with the dense
checkpoint mechanism and is not part of this path.

## 4. Backend contract

The new backend should be separate from the dense `kindi.engine.Engine` and `kindi.tandem` code. A
minimal interface (names are proposals; R0 fixes the model-specific parts):

```python
class MoEManifest:
    layers: int
    hidden: int
    routed_experts: int
    top_k: int
    shared_experts: int
    expert_tensors: tuple[str, ...]       # gate/up/down and scales
    quant_format: str
    layer_schedule: tuple[str, ...]       # linear / full attention per layer
    router_tie_rule: str
    per_expert_bytes: int

class MoEBackend:
    manifest: MoEManifest

    def embed(self, token): ...
    def core_layer(self, layer, hidden, pos, state): ...
    def route(self, layer, hidden) -> RouterPlan: ...
    def shared_expert(self, layer, hidden): ...
    def expert_gpu(self, layer, expert_id, hidden, out): ...
    def expert_cpu(self, layer, expert_id, hidden, out): ...   # parity-gated
    def combine(self, plan, outputs) -> Tensor: ...
    def head(self, hidden): ...
```

`RouterPlan` contains the selected expert ids and router weights. It may preserve the router's
top-k order for scheduling; the backend's `reference_order(plan)` determines arithmetic order, and
for Gemma 4 that is expert-id order. A backend must not deduplicate or renormalize unless the
reference does. The tie rule is part of the backend contract, not a cache policy choice.

The cache-facing interface is deliberately smaller:

```python
class ExpertKey(NamedTuple):
    layer: int
    expert: int

class ExpertCache:
    def plan(self, keys: list[ExpertKey], plan_key: object) -> ExpertPlan: ...
    def acquire(self, key, plan_key: object) -> Slot: ...
    def release(self, plan_key: object): ...
    def hit_rate(self) -> float: ...
    def metrics(self) -> dict: ...
```

The prototype names that opaque key `token`; it is used as a dictionary key, so a Gemma 4 caller can
pass `(request_id, layer)`. `ExpertPlan` pins every selected expert for the plan and prevents eviction until the combine is
complete. The plan key is opaque: a Raon plan covers a token, while a Gemma 4 plan covers one layer
(`(request_id, layer)`). A slot is `FREE`, `LOADING`, `READY` or `IN_USE`; a slot in `LOADING` or `IN_USE` is never
a victim.

The future copy stream meets this interface through immutable `CopyJob`/`CopyCompletion` records;
it cannot choose or evict a slot. The exact ticket, generation and failure contract is in
[EXPERT-CACHE-COPY-INTERFACE.md](EXPERT-CACHE-COPY-INTERFACE.md).

## 5. Exact execution and combine order

A MoE layer's output must be assembled in the backend's reference order, never in an order chosen
by the cache. Raon uses router order. Gemma 4's official expert loop visits `expert_hit` in
ascending expert-id order, so its combine order is expert-id order even though `topk` returned
descending probabilities. The cache may schedule copies in router order; it must not change the
arithmetic order.

```
routed = 0
for expert_id, weight in backend.reference_order(plan):   # backend-defined
    routed += weight * expert_out[expert_id]              # fp32 accumulation
y = shared_combine(layer, x, routed)                      # backend-defined
```

`expert_out` is a complete `[hidden]` vector. The combine kernel reads the slots in reference
order; it does **not** use `atomicAdd` or completion order. This is required even when several
experts are computed in parallel. For Gemma 4, the shared/dense MLP runs in parallel with the routed
branch and is added after the separate post norms; it is not appended after the routed sum. The
generic `shared_combine` placeholder is deliberate because the two backends differ.

The first path keeps attention, router, shared branch and combine on the GPU. The router output is
produced once per layer and is never recomputed on the CPU to "fix" a difference.

### Two miss lanes

| lane | what happens | arithmetic | use |
|---|---|---|---|
| `copy` (first exact path) | the complete expert is copied H2D into a cache or transient slot, then computed by the same GPU expert kernel as a hit | all expert arithmetic is GPU; cache state cannot change output | M1 exactness gate, default |
| `cpu` (optional) | a CPU worker computes the expert from host-resident weights; its output vector is copied H2D | only enabled if the CPU and GPU expert kernels produce bit-identical expert outputs for the chosen quant format | M2, gated |

The CPU lane is not enabled merely because it is faster. Floating-point reduction order differs
between CPU and GPU, and a cold expert would then take a different token than a warm one. The gate is
**bitwise equality of the selected expert output** (or of the complete logits, whichever the
reference exposes), not "close enough". If parity fails, the lane stays disabled and misses use
`copy`; a CPU-compute lane cannot be advertised as exact.

For the first exact path, even a cache miss is GPU-computed, so cold, warm, forced-eviction and
reload runs must produce identical logits and tokens. That is a stronger and simpler claim than
tandem.cpp's CPU-miss behavior.

## 6. Per-plan execution

1. Run the layer's attention/core path on the GPU; the dense core and shared branch are pinned.
2. Run the router on the GPU. Produce `RouterPlan` with the selected ids and router weights; keep
   scheduling order separate from the reference combine order.
3. Ask `ExpertCache.plan()` for a slot for every selected expert in the plan. For Gemma 4 the plan
   is one layer, so this step runs 30 times per token, not once for all 240 experts.
4. For each selected expert, in the backend's scheduling order:
   - hit: enqueue the GPU expert kernel in its slot;
   - `copy` miss: enqueue a complete H2D copy into a slot/transient buffer, wait for its event,
     then enqueue the same GPU expert kernel;
   - `cpu` miss: enqueue the CPU expert task, then copy its output vector H2D; do not combine until
     the copy is complete.
5. Wait for all selected experts. A partially arrived expert is never used.
6. Combine in the backend's reference order on the GPU; add the shared branch in the backend-defined
   order. For Gemma 4 that is ascending expert-id order and a parallel dense MLP, not router order
   plus a post-hoc shared expert.
7. Release the plan's pins; apply the cache policy to slots that were transient. For Gemma 4, run
   steps 3-7 once per layer and release before the next layer; never pin all 240 experts at once.

### Concurrency, cancellation and cache identity

Kindi's server serves one active request at a time, and a plan is scoped to one backend execution
unit (a Raon token or a Gemma 4 layer), not to the whole model. Pins are released only after the
combine, and a slot in `LOADING` or `IN_USE` is never a victim. On
cancellation, the scheduler releases the token's pins, discards partial expert outputs and does not
reuse a slot until its copy or compute event has completed. The expert cache holds weights only; it
is never part of a prompt checkpoint or saved opening. A cache is tied to the loaded pack, backend
and quantization format; changing any of them invalidates its slots rather than mixing weights.

The copy stream, compute stream and combine are separate. H2D of a miss may overlap with GPU work on
hits; the combine is the only barrier. No CPU worker writes the final residual or a cache slot
directly.

The first path is batch size 1. Continuous batching is a separate no-go and must not be smuggled in
as an expert-cache optimization.

### Prefetching

M1 does no speculative prefetch. It copies only the experts the current router selected. This keeps
the exactness proof small and avoids charging extra PCIe traffic for a guess.

M3 may add lookahead using a previous token's route or a small predictor. Prefetch is an optimization
only: it may not evict a current-token expert, change the plan, or delay the combine. Accuracy and
bytes copied are measured separately from hit rate.

## 7. Cache policy and memory

Key: `(layer, expert_id)`. Value: complete expert tensors and scales, never a partial tensor.

- **Slot size** is `max(per_expert_bytes)` over the released pack, rounded for alignment. A slot is
  reused only after its previous expert is no longer pinned.
- **Capacity** is `KINDI_MOE_CACHE_MB`, with an automatic value from free VRAM after the dense core,
  KV/state, workspaces and a safety reserve. The reserve is explicit and must not be silently
  consumed by the cache.
- **Eviction** starts with LRU over unpinned `READY` slots; a clock policy may be measured later.
  Pinned shared/core weights are never candidates. In-flight and current-token slots are never
  candidates.
- **Transient slots** handle a miss when the cache is full. If no transient slot exists, the request
  fails before computing a partial expert; it does not approximate or skip.
- **Minimum cache** is backend-scoped. A Raon token-level plan must hold all selected experts plus
  the shared expert; a Gemma 4 layer-level plan needs eight cache slots plus a transient slot. If
  the minimum cannot fit, the model is not offered on that GPU.
- **Host side** keeps the whole expert set resident and memory-mapped. A bounded pinned staging
  area holds one or a few expert tensors; decode must not fault expert weights from disk. `mlock`
  is optional and must not be assumed.
- **Accounting** includes tensor bytes, scales and alignment. A cache hit rate without bytes and
  slot count is not a useful metric.

The model-free prototype in `kindi/expert_cache.py` implements this policy. Its fixed `slot_bytes` is
only an accounting value: it does not allocate VRAM, perform an H2D copy or run an expert.
`plan()` is atomic, `mark_ready()` completes a copy, and `release()` leaves an in-flight `LOADING`
slot unavailable until its event is accounted for.

Measured envelope for the pilot: the base is **32,699,649,368 bytes**, of which
26,471,301,120 are routed experts at 8,616,960 B per expert; the dense core is 6,228,348,248 B. The
Q4_K_M file is 19,686,434,240 B and its expert slots are 5,048,320-6,092,800 B, so a 4-bit host set
is plausible on 32 GB. On an 8 GB card the cache is still a small fraction of the expert set; the
dense core, KV/state and workspaces have priority. Q4 is lossy and must not inherit the base's
exactness claim.

Measured envelope for Gemma 4 26B-A4B: the Q4 dense core is **2,578,661,496 B**, the routed expert
weights are **14,479,917,056 B** (plus 15,360 B of scales), and the fixed slot is **4,336,640 B**.
Eight slots need 34,693,120 B plus a transient slot. Its layer plan must be released after each
layer; top-8 over 30 layers is 240 experts, not one token's pin set. The 256K KV cache alone is
about 5.19 GiB and does not fit an 8 GB card with the Q4 dense core and reserve, so that context
requires host-backed KV or a larger card. Full details and gates: [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md).

## 8. Prompt cache and served state

The expert cache is a **weight cache**, not conversation state. It does not belong in prompt
checkpoints, parked conversations or saved openings. A prompt cache checkpoint contains the model's
recurrent/attention state; the expert cache may be cold or warm independently.

Consequences:

- A cold start does not need to load every expert into VRAM before serving.
- A saved opening does not carry expert weights.
- Changing `KINDI_MOE_CACHE_MB` does not invalidate a prompt opening when the miss lane is `copy`,
  because the arithmetic is identical.
- If a future miss mode changes arithmetic, it must be in the prompt-cache fingerprint and the
  opening must not be shared across modes.
- The first path keeps all layers on the GPU, so the normal single-engine `cache.Runner` can
  checkpoint the model state. A later CPU/GPU layer split would need the joint tandem view; do not
  reuse the dense `TandemRunner` for the expert cache itself.

## 9. Integration plan

Implemented, model-free:

| file | purpose |
|---|---|
| `kindi/expert_cache.py` | copy-only slot manager, pin-aware LRU, transient fallback and JSON-safe metrics; no tensors, GPU or engine wiring |
| `tests/test_moe_cache.py` | synthetic-slot tests for transitions, eviction, pinning, transient slots, failed copies and metrics |
| `tests/test_moe_cache_properties.py` | 32 seeded randomized route/acquire traces; checks capacity, pin ownership, slot states, atomic rejects and metrics after every operation |

Still proposed, not implemented:

| file | purpose |
|---|---|
| `kindi/moe.py` | manifest, backend protocol, router plan and scheduler; it will call `expert_cache` |
| `kindi/moe_cpu.c` / `kindi/moe_cpu.py` | host expert kernels and worker pool (M2 only) |
| `kindi/moe.cu` | GPU router, expert FFN, copy/event helpers and ordered combine |
| `kindi/moe_convert.py` | Raon-aware expert manifest and pack layout; R0/R1 |
| `tests/test_moe_exactness.py` | cold/warm/forced-eviction token checks, gated by a pack |

Existing code:

- `kindi/catalog.py` gets an entry only after a checksummed pack and license decision.
- `kindi/convert.py` gets a Raon-specific path; it must not pretend the dense Bonsai pack is a MoE
  pack.
- `kindi/config.py` gets the cache/miss settings only when the backend exists.
- `kindi/server.py` gets a model variant and expert-cache health metrics; it must not advertise the
  model before the gates.
- `kindi/engine.py`, `kindi/tp.py` and `kindi/tandem.py` remain dense paths. A MoE backend is not a
  flag on `Engine`.

## 10. Metrics and failure behavior

Expose at least:

- cache hits, misses, evictions, admissions and transient slots;
- bytes copied H2D and expert bytes per token;
- CPU expert tasks, worker wait and CPU seconds;
- GPU expert seconds, copy wait and combine wait;
- hit rate, route count and top-k distribution (without prompt text);
- peak VRAM, peak host RSS and page-fault/disk-read counters;
- tokens/s and prompt-read time, separated from cache statistics.

`kindi/expert_cache.py` exposes the model-free subset now: plans, selected keys, hits, misses,
coalesced loads, evictions, cache/transient admissions, copy bytes, pins, rejects and slot-state
counts. It does no copy or compute, so it cannot provide H2D wait, CPU/GPU seconds, VRAM/RSS or
disk counters; the backend must add those before operational metrics are complete.

`/health` must report the fitted cache and the mode (`copy` or parity-passed `cpu`). `/metrics` must
carry counters. A cache error, failed copy, partial expert or parity failure is a hard error, not a
silent fallback.

## 11. Phases and gates

### M0 — backend contract and reference (artifacts complete)

The released config, license, tokenizer and every tensor are recorded in
[RAON-1-MINI-R0.md](RAON-1-MINI-R0.md): 19,434 base tensors, 731 Q4_K_M GGUF tensors and the exact
RDA/MLA schedule. The exact greedy harness is `bench-logs/raon-r0/reference_greedy.py`; Q4 is a
separate lossy runtime check.

**Gate:** manifest, license decision and reference command are recorded. The greedy goldens were
not run (no base load, no 4/4 result), so M1 starts with them. The model-free cache prototype does
not change this gate; M1 still needs the model, a copy path and an exactness run.

### M1 — GPU core + host experts, copy-only misses

Upload the core, router and shared expert; keep all routed experts in host RAM; add a byte-budgeted
slot cache with H2D promotion; run cold, warm, forced-eviction and reload prompts through the same
GPU expert kernel.

**Gate:** 4/4 greedy prompts token-exact at 1K-2K for every cache state; no OOM; no partial expert;
no disk read during decode.

### M2 — CPU miss lane, parity-gated

Implement CPU expert kernels using the same quantized dot product and accumulation order as the GPU
expert kernel. Enable the lane only if cold and warm expert outputs are bit-identical on the pilot
prompts.

**Gate:** cold/warm/forced-eviction outputs are identical with the CPU lane enabled; if parity fails,
the lane is disabled and this phase is a documented no-go, not a support claim.

### M3 — overlap and prefetch

Run H2D, GPU hits and CPU misses concurrently; add a fixed-order combine barrier; measure a
lookahead predictor separately from demand misses.

**Gate:** no output change versus M1; measured copy wait, CPU wait and hit-rate trade-off; no
starvation of the current token.

### M4 — server, prompt cache and restart

Wire the engine into `kindi serve`, non-stream, stream and cancellation; reuse the normal
single-engine prompt cache; expose health/metrics; test a cold/warm/restart cycle.

**Gate:** same tokens with cache cold/warm; prompt-cache resume; stream `[DONE]`; no 5xx; no stale
expert state.

### M5 — 32K, 128K and 262,144 tokens

Fit the KV/state budget at each level; test retrieval at the beginning, middle and end; record
whether the full context uses VRAM, host-backed state or a documented quantized mode.

**Gate:** no silent truncation; memory and latency recorded at every level.

### M6 — support claim

Installer host qualification, catalog entry, measured 8 GB and larger-card rows, and the full
acceptance matrix.

**Gate:** every cell has evidence; otherwise the model remains planned.

## 12. Acceptance matrix

All cells are pending. A checked cell needs a measured result and a link to its evidence.

| setup | 1K exact | 8K | 32K | 128K | 262,144 | peak host RAM | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| host reference (CPU/BF16 or chosen reference) | pending | pending | pending | pending | pending | pending | — |
| 8 GB GPU, cache `copy` | pending | pending | pending | pending | pending | pending | pending |
| 8 GB GPU, cache `cpu` parity | pending | pending | pending | pending | pending | pending | pending |
| 16 GB GPU, cache `copy` | pending | pending | pending | pending | pending | pending | pending |

The 8 GB `copy` row is the first support target. The `cpu` row is added only after bitwise parity.
The 16 GB row is a comparison, not a separate backend.

## 13. Risks and stop conditions

| risk | consequence | mitigation / stop |
|---|---|---|
| cache too small | every token streams experts; throughput collapses | measure bytes/token and hit rate before tuning; stop if the 8 GB minimum cannot hold the active set |
| PCIe bandwidth | H2D dominates decode | pinned staging, complete-expert copies, overlap; record bytes/token |
| CPU/GPU arithmetic mismatch | cold and warm tokens differ | default `copy` lane; CPU lane only after bitwise parity; otherwise no support claim |
| router mismatch | wrong experts, non-reproducible output | preserve released router arithmetic and tie rule; router tests |
| combine order | nondeterministic logits | per-expert output buffers, backend reference order (expert-id for Gemma 4), no atomics |
| host page cache eviction | decode reads experts from disk | whole pack resident, pre-fault, disk-read counter; fail gate on runtime disk reads |
| 262K KV | VRAM exhaustion or unusable latency | 32K/128K first; host-backed or quantized state only as an explicit, tested mode |
| no reference or license | cannot prove or ship | the gate stops; do not invent a comparison |

## 14. R0 answers and remaining gaps

1. **Names/geometry/expert size — resolved:** 19,434 base tensors; 128 experts × 24 MoE layers;
   8,616,960 B per MXFP8 expert; full names and shapes in [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md).
2. **Router — partly resolved:** sigmoid, selection-only bias, top-6, renormalize, scale 1.0, shared
   added after; there is no explicit tie rule, so tied scores need a reference test.
3. **Formats — resolved:** BF16 core + MXFP8 experts; Q4_K_M/Q5_K_M/Q6_K/Q8_0 GGUF. Q4 is lossy;
   the first exact reference is the base MXFP8 checkpoint.
4. **Attention — resolved:** 18 RDA + 7 gated MLA, NoPE, RDA conv-4/recurrent state, MLA 192/128;
   recurrence recorded from the released code.
5. **K/V — two answers:** published compressed 8,064 B/token; HF reference expanded 179,200
   B/token. The exact llama.cpp cache layout is still open; use neither number silently.
6. **Reference — pinned:** exact greedy base harness plus a Q4 runtime sanity command; no golden
   has been produced on this host.
7. **License/access — resolved:** Apache-2.0, access request and HF login required; no silent
   download claim.
8. **CPU/GPU parity — open:** only the `copy` miss lane is the first exact design. CPU compute must
   be bit-identical before it is enabled.

## 15. What this document does not claim

- It does not add Raon-1 mini, or any other MoE model, to Kindi.
- It does not add a model or backend; `kindi/expert_cache.py` is a standalone state machine with no
  tensor copy, GPU allocation or inference path.
- It does not add a converter, pack, catalog entry, installer offer or support-matrix row.
- It does not claim a cache hit rate, tokens/s, VRAM figure or host-RAM figure.
- It does not claim that CPU miss compute is exact; only the `copy` lane is the first exact design.
- It does not change the dense `TandemEngine` or its joint prompt cache: the tandem only gained the
  refusal described in §1, which rejects a non-dense manifest before any backend is built.

Related records: [RAON-1-MINI.md](RAON-1-MINI.md), [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md),
[GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md), [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md),
[EXPERT-CACHE-COPY-INTERFACE.md](EXPERT-CACHE-COPY-INTERFACE.md), [FP8-BF16.md](FP8-BF16.md),
[TANDEM-RUNTIME.md](TANDEM-RUNTIME.md), [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md),
[ROADMAP.md](ROADMAP.md), [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here).
