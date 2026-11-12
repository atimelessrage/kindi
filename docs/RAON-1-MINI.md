# Raon-1 mini: long-term MoE support goal

Status: **R0 artifacts complete, 2026-11-11; R1 not started.** The released config, license, full
tensor manifest and reference-command checklist are recorded in
[RAON-1-MINI-R0.md](RAON-1-MINI-R0.md). Raon-1 mini is not in `kindi/catalog.py`, has no converter,
engine, pack, installer offer or measured inference result. This is a goal and an acceptance
contract, not a support claim.

The target is a single OpenAI-compatible Kindi server for an open-weight Raon-1 mini checkpoint,
with the **whole quantized model resident in system RAM** and a **GPU-resident expert cache** for
the routed experts that are hot at the current route distribution. The first useful target is an
8 GB NVIDIA card; larger cards are easier cases, not separate architectures.

## Model contract (verified in R0)

The R0 audit found the public `KRAFTON/Raon-1-mini-preview` checkpoint and verified the supplied
contract against its config, safetensors headers and GGUF header. The complete names, shapes,
dtypes and layer schedule are in [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md).

| item | released value |
|---|---|
| parameters | **28,783,218,602** complete tensor elements; HF summary 28,783,218,402 excluding 200 scalars |
| active parameters | **4,317,319,082** (6/128 routed-expert elements plus the dense core, 15.0%) |
| routing | **128 routed experts, top-6 per token, plus 1 shared expert**; sigmoid router |
| attention | **18 RDA linear-attention + 7 gated MLA full-attention layers**; layer 0 is dense, layers 1-24 are MoE |
| maximum context | **262,144 tokens**; NoPE |
| weights | base BF16 core + **MXFP8 E4M3 routed experts**; Q4_K_M/Q5_K_M/Q6_K/Q8_0/IQ4_XS GGUF |
| license | **Apache-2.0**; access is granted on request and needs the user's HF login |

The “active 4.3 B” figure is a compute and bandwidth budget, not a statement that only 4.3 B of
weights exist. The 32,699,649,368 base tensor bytes still have to be stored; 26,471,301,120 of them
are routed-expert bytes.

## What “supported” means

The final claim uses the same contract as [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here):

1. The installer detects a viable host and offers the setup only when the measured gates below pass.
2. The model loads from a checksummed Kindi pack, reports the fitted context and answers chat.
3. Greedy `kindi verify` is token-exact against the selected reference at the short-context gate.
4. The expert cache is semantically transparent: cold, warm, evicted and reloaded experts produce
   the same logits and tokens.
5. Long-context retrieval passes at 32K, 128K and the full 262,144 tokens, with the allocated context
   and peak host/GPU memory recorded; no silent truncation.
6. A real server run covers non-stream, stream, cancellation and prompt-cache restart.
7. The measured result is recorded in the project docs.

Until all of that happens, the model is a roadmap item. It must not appear in the installer, the
download catalog or the generated support matrix as supported.

## Why this needs a new backend

The existing Kindi engines are dense Bonsai paths. They do not contain a MoE router, an expert
dispatch/combine path, a host-backed expert cache or the Raon-1 hybrid layer schedule.

| area | current reality | implication for Raon-1 mini |
|---|---|---|
| `kindi/catalog.py` | dense Bonsai, Gemma 4/Nemotron and Nanbeige entries | no model id, URL, checksum or size; do not add one before the pack exists |
| `kindi/convert.py` | Bonsai PTQ1_0 packs plus Gemma 4/Nanbeige GGUF paths | needs a Raon-specific tensor and expert inventory, not a second dense conversion path |
| `kindi/engine.py` | dense Bonsai kernels; one set of weights and dense FFN matrices | needs router, top-6 dispatch, shared-expert path, expert combine and sparse FFN kernels |
| `kindi/tp.py` | two-card dense tensor/plan engine | not an expert-aware split; expert ownership and cache are separate problems |
| `kindi/tandem.py` | dense layer split between CPU and GPU | not a substitute for an expert cache; expert weights are not layers |
| `kindi/cache.py` | prompt checkpoints, parked conversations and saved openings | needs a second, weight-oriented cache with its own eviction and accounting |
| `kindi/cpu.py` / `kindi/kcpu.c` | dense 2B/4B/9B CPU engine | no MoE reference path yet; a host reference is needed before GPU optimization |
| server / installer | one dense-engine contract and dense model profiles | needs expert-cache health metrics and a memory-based host qualification; no offer yet |

The design must therefore add a **new model backend**, not a flag on the Bonsai engine.

## Memory architecture

The defining constraint is 28.8 B total parameters against an 8 GB GPU. The whole model cannot be
GPU-resident. The target layout is:

```
host RAM (resident, memory-mapped pack)
  ├─ all routed experts, layer by layer
  ├─ shared experts and dense/attention weights
  └─ tokenizer / pack metadata

GPU VRAM
  ├─ resident core: attention, router, norms, shared expert, output path
  ├─ KV / linear-attention state
  └─ expert cache: hot (layer, expert) weights
        ↕ pinned staging + asynchronous H2D copies on a miss
```

### Weight-size envelope (measured)

The released artifact is BF16 core plus MXFP8 routed experts; this is not the same as BF16 or a
4-bit pack. The measured numbers below replace the earlier planning envelope.

| artifact | bytes | notes |
|---|---:|---|
| base tensors | **32,699,649,368** | all 19,434 tensors; 7 shards are 32,702,145,088 file bytes |
| base routed experts | **26,471,301,120** | 3,072 slots × 8,616,960 B; 80.95% of tensor bytes |
| base dense/core | **6,228,348,248** | attention, mHC, norms, router, shared, embedding/head, dense FFN |
| Q4_K_M GGUF | **19,686,434,240** | lossy reference-runtime file; 72 packed expert tensors |
| Q4 expert slot | 5,048,320-6,092,800 | depends on the layer's down-projection type |
| six cold MXFP8 experts | 51,701,760 | before staging/alignment; Q4 is 30,289,920-36,556,800 |

The 8 GB target still needs a host-resident expert set plus an explicit GPU cache. A 4-bit pack is
smaller and lossy; it must not inherit the base checkpoint's exactness claim. The first exact
reference is the 32.7 GB base on a larger host, not Q4 on the 8 GB card.

The whole model must be **resident in host RAM**, not streamed from the SSD during decode. The
implementation should memory-map the pack, pre-fault the layers needed by the active model, and
keep a bounded resident set. `mlock` is optional (it may exceed the user's locked-memory limit);
the acceptance run must at least show that decode does not fault experts in from disk.

### Expert cache

The cache is keyed by `(layer, expert_id)` and is separate from the prompt/KV cache.

Required behavior:

- pin the shared expert and the non-expert core; they are never eviction candidates;
- route exactly six of 128 routed experts, preserving the model's router arithmetic and tie rules;
- on a miss, copy one complete expert from resident host memory into a pinned staging buffer, then
  asynchronously into its cache slot; do not compute with a partially arrived expert;
- use an explicit eviction policy (start with LRU or clock; record the choice);
- prefetch the next likely experts only when the prefetch cannot starve the current token;
- expose hit rate, misses, bytes copied, evictions, stall time and cache capacity in `/health` and
  `/metrics`;
- make the cache size a configuration value, with a hard VRAM reserve for KV and workspaces;
- never approximate a missing expert, and never silently change top-k when an expert is cold.

Start with batch size 1. A later small-batch step may union the experts required by several rows, but
continuous batching is a separate no-go until its own gate passes.

### Hybrid attention and 262K context

The released model has 18 RDA and 7 gated MLA layers. RDA keeps a fixed fp32 recurrent state plus a
short-conv window; the seven MLA layers are NoPE and carry the growing cache. The model card's
published cache is **8,064 B/token BF16** (7.875 KiB; 1.96875 GiB at 262,144 tokens) for the
compressed 576-value latent. The Hugging Face reference expands every head and stores
**179,200 B/token** (175 KiB; 43.75 GiB at 262,144), so the two numbers are not interchangeable. The
RDA state adds 48,844,800 B (46.58 MiB) per sequence. The exact llama.cpp cache layout and any paged
or host-backed mode remain R0/R1 items; do not use the HF expanded number as the 8 GB budget.

The context rollout is deliberate:

1. establish exact output at 1K-2K with the cache and state paths;
2. 32K retrieval and memory accounting;
3. 128K retrieval and host/VRAM pressure;
4. 262,144 tokens, with the allocated context, KV bytes and peak memory recorded.

If the full K/V does not fit in VRAM at 262K, host-backed or quantized K/V is allowed only as an
explicit, tested mode with a named quality/latency cost. The server must fail loudly rather than
silently truncate a request that exceeds the fitted context.

## Phased plan

Each phase produces evidence for the next; no phase is skipped because a later one looks easier.

### R0 — architecture and reference audit (artifacts complete)

- obtained the released config, tokenizer, license and checksummed shard metadata;
- enumerated all **19,434** base tensors and all **731** Q4_K_M GGUF tensors, including every one of
  the 3,072 `(layer, expert)` slots; recorded per-layer shapes and the exact RDA/MLA schedule;
- identified the reference implementation, the Apache-2.0 license and the exact greedy protocol;
- selected the base MXFP8 checkpoint, not Q4_K_M, as the exactness reference;
- replaced the planning memory envelope with the measured 32,699,649,368-byte base and 8,616,960 B
  per-expert figures.

**Gate:** the artifact record has a complete tensor manifest, a license decision and a reproducible
reference command ([RAON-1-MINI-R0.md](RAON-1-MINI-R0.md)). The reference **goldens have not been
run**: no base model was loaded, no 4/4 token-exact result exists and no model code is merged. R1
starts with those goldens.

### R1 — host-resident pack and CPU reference

- add a Raon-aware converter to a checksummed pack with an expert index and a manifest;
- implement a simple host reference path for router, expert dispatch, shared expert, hybrid layers
  and greedy decode;
- keep the full pack memory-mapped in host RAM; measure peak RSS and page behavior;
- verify short prompts against the reference.

**Gate:** 4/4 greedy prompts token-exact at 1K-2K; peak RSS is within the documented host-RAM
envelope; decode does not read expert weights from disk.

### R2 — GPU resident core

- upload the non-expert core, the shared expert and the router;
- add the GPU router, top-6 dispatch, shared-expert combine and output path;
- keep routed experts on the host for this phase;
- make the 8 GB memory budget explicit and fail before an OOM.

**Gate:** short-context output is identical to R1; `/health` reports VRAM, host RSS and the fitted
context; no expert cache is claimed yet.

### R3 — GPU expert cache

- add the host-to-device expert cache, pinned staging and asynchronous copies;
- implement a deterministic eviction policy and complete-tensor arrival before compute;
- expose hit/miss/transfer/stall metrics;
- run cold, warm, forced-eviction and reload cases.

**Gate:** identical logits/tokens for cold and warm runs; forced eviction and reload reproduce the
same output; an expert miss is slower but never changes the answer; no OOM at the first context
target.

### R4 — hybrid attention and 32K

- implement the released linear-attention recurrence and the seven full-attention layers;
- add the first paged/position-indexed K/V path and the context fit calculation;
- verify the layer schedule against the reference;
- measure prompt read and decode at 1K/8K/32K.

**Gate:** exact short-context output; 32K retrieval passes; peak host and VRAM are recorded; no
silent context truncation.

### R5 — 128K and 262K

- extend the K/V path and the host/VRAM budget;
- test retrieval at the beginning, middle and end of the context;
- record whether full 262K uses VRAM, host-backed K/V or a documented quantized mode.

**Gate:** 128K and 262,144-token requests complete without truncation; retrieval is correct at the
tested depths; the memory and latency costs are documented.

### R6 — server, prompt cache and installer

- wire the engine into `kindi serve`, non-stream and stream, cancellation and prompt cache;
- add expert-cache metrics to the existing queue/health/monitoring contract;
- qualify hosts by RAM and VRAM rather than by model name alone;
- only after R1-R5 pass, add the installer profile.

**Gate:** a real server cold/warm/restart cycle, stream `[DONE]`, prompt-cache resume, cancellation,
no 5xx and no stale expert state.

### R7 — performance and support claim

- benchmark on the 8 GB target and at least one larger card;
- compare against a tuned reference runtime on the same host; record the exact command;
- publish the expert-cache hit rate and transfer volume, not only tokens/s;
- update the support matrix and the card/CPU pages with measured rows.

**Gate:** a numeric decode/read gate agreed in R0 passes, 8K memory is within budget and the full
acceptance matrix is green. Until then the model remains “planned”.

## Provisional performance gate

The first target is an 8 GB RTX 4060 Ti-class card with 32 GB of host RAM and a 4-bit-ish pack.
The exact gate is fixed after R0, when expert sizes and transfer volume are known. The provisional
floor is:

- **at least 8 decode tokens/s at 8K** with a warm expert cache;
- **at least 80% expert-cache hit rate** on a repeated four-prompt set, with hit rate reported next
  to throughput;
- **at least 0.75x** the tuned reference runtime on the same host, or a documented reason why the
  reference cannot run the model;
- no OOM, no silent fallback and no output change between cold and warm cache runs.

A lower number is allowed only by changing the recorded gate before the measurement, not after it.

## Acceptance matrix

All cells are pending. A checked cell requires a measured result and a link to its evidence; a
“builds” result is not support.

| setup | 1K exact | 8K | 32K | 128K | 262,144 | peak host RAM | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| host reference, BF16 | pending | pending | pending | pending | pending | pending | — |
| 8 GB GPU, Q4-ish | pending | pending | pending | pending | pending | pending | pending |
| 16 GB GPU, Q4-ish | pending | pending | pending | pending | pending | pending | pending |
| 8 GB GPU + larger host | pending | pending | pending | pending | pending | pending | pending |

The 8 GB row is the first support target; the 16 GB row is a comparison, not a separate backend.

## Risks and stop conditions

| risk | consequence | mitigation / stop condition |
|---|---|---|
| expert cache too small | every token streams many experts; throughput collapses | measure hit rate and bytes/token before optimizing kernels; stop if the 8 GB target cannot hold the working set at 8K |
| PCIe bandwidth | host-to-device copies dominate decode | pinned staging, asynchronous copies and prefetch; record transfer volume per token |
| router mismatch | wrong experts and non-reproducible output | preserve released router arithmetic and tie rules; exact routing tests |
| quantized experts change output | verify fails or quality drops | keep a BF16 reference; document the first lossy format; do not call it exact |
| host page cache eviction | decode faults from disk | pre-fault, keep the model resident, record RSS and disk reads; fail the gate on runtime disk reads |
| hybrid attention mismatch | wrong long-context state | implement from the released config, not from a guessed Qwen variant; compare at each layer |
| 262K K/V | VRAM exhaustion or unusable latency | phase 32K/128K first; host-backed or quantized K/V is an explicit mode |
| no reference runtime | no exactness target | require the released reference code or a documented golden; do not invent a comparison |
| license or checkpoint access | cannot ship the model | resolve before converter work; the goal is not a redistribution promise |

## R0 answers and remaining gaps

The eight questions are answered with file-level evidence in
[RAON-1-MINI-R0.md](RAON-1-MINI-R0.md). Summary:

| question | result |
|---|---|
| geometry and recurrence | 25 layers, hidden 2,560, 40 heads, 18 RDA + 7 MLA, RDA 128, MLA 192/128, NoPE, mHC-4; recurrence recorded |
| routed/shared/dense | 128 routed + 1 shared, top-6, 1,088 expert FFN, 12,032 dense layer 0; MXFP8 expert 8,616,960 B |
| dtype/tokenizer/template | BF16 core + MXFP8 experts; 259,840 BPE; ChatML thinking template; Apache-2.0 |
| router | sigmoid, bias for selection only, top-6, renormalize, scale 1.0, shared after; **tie rule unspecified** |
| K/V | published 8,064 B/token compressed; HF reference 179,200 B/token expanded; exact llama.cpp layout pending |
| paged/sliding/RoPE | NoPE, no sliding window, fixed RDA state; paged/host-backed cache is a runtime decision |
| license/access | Apache-2.0; gated request, user login required; no automatic download claim |
| reference | exact greedy base harness is pinned; Q4_K_M llama.cpp command is a lossy runtime check; neither golden was run here |

The remaining R1 blockers are the two reference runs, the router tie test, the exact llama.cpp
cache layout and the first pack-format decision. No answer is inferred from the Bonsai engine.

## Related design records

- [RAON-1-MINI-R0.md](RAON-1-MINI-R0.md) — the verified manifest, source pins and reference checklist.
- [HOW-IT-WORKS.md](HOW-IT-WORKS.md) — the current dense Bonsai execution path.
- [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here) — the support contract.
- [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) — the dense CPU/GPU layer split, which is not an expert
  cache.
- [MOE-TANDEM.md](MOE-TANDEM.md) — the expert-cache tandem design for this goal (design plus a model-free cache prototype; no backend).
- [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md) — the second backend profile: Gemma 4 26B-A4B top-8 routing, fused experts, parallel dense MLP and 8 GB context limits.
- [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md) — the host/device staging design that can
  inform the expert transfer path.
- [ROADMAP.md](ROADMAP.md) — the project index and archived no-go decisions.
- [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) — generated capability table; Raon-1 mini is absent until
  it has a catalog entry and a verified status.
