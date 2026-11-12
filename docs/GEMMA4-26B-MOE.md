# Gemma 4 26B-A4B expert-cache profile

Status: **design adaptation and execution contract, 2026-11-12. The local Q4_K_M header and the
official model contract were measured; there is no 26B backend, expert-cache integration, GPU copy
path, reference golden or support claim. The eight-slot copy lane, ordered combine and
failure/forced-eviction behavior are specified in
[GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md). Q4_K_M is lossy and cannot be used as an
exactness reference.**

This is the Gemma 4-specific companion to [MOE-TANDEM.md](MOE-TANDEM.md). It does not replace the
generic cache state machine in `kindi/expert_cache.py`; it fixes the model constants, slot layout,
routing order and context limits that a 26B backend would have to honour. The model-free state
machine already supports `(layer, expert)` keys, fixed-size slots, pins, LRU eviction and transient
slots. The concrete eight-slot host-to-GPU path, deterministic combine and hard-error behavior are
in [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md). No Gemma 4 code was added with this
document.

## 1. Verified model contract

Sources: the official [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), the
official [26B-A4B config](https://huggingface.co/google/gemma-4-26B-A4B/raw/main/config.json) and the
[HF Gemma 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py).
The local artifact is the Ollama `gemma4:26b` text decoder plus its separate projector.

| item | verified value |
|---|---|
| parameters | 25.2 B total, 3.8 B active (model card) |
| text layers | 30 |
| hidden size | 2,816 |
| dense/shared FFN width | 2,112 |
| routed experts | 128 per layer |
| experts selected per token | top-8 |
| routed expert width | 704 |
| shared branch | one ordinary dense MLP per layer |
| attention | 16 query heads; 8 KV heads on sliding layers, 2 on global layers |
| head dimensions | 256 sliding, 512 global |
| layer schedule | 25 sliding-window layers, global layers at 5, 11, 17, 23 and 29 |
| sliding window | 1,024 tokens |
| context | 262,144 tokens |
| vocabulary | 262,144; tied output embedding |
| activation | `gelu_pytorch_tanh` |
| per-layer token embeddings | none (`embedding_length_per_layer_input = 0`) |
| V/K sharing between layers | none (`num_kv_shared_layers = 0`) |
| global V projection | absent; the five global layers derive V from the K projection (`attention_k_eq_v`) |
| license | Apache-2.0 |
| local text artifact | 17,074,419,072 B, Ollama digest `a8bdd07a49b895083bb1420bb895ad9ab3e3b8f7d5cc42c7582ca0fa765eede2` |
| local projector | 1,194,828,128 B, separate mmproj file |

The local GGUF has 658 tensors: 392 F32, 207 Q8_0, 29 Q5_1, 28 Q4_K and 2 Q5_K. It has 30
`attn_k` tensors but only 25 `attn_v` tensors; the five global layers have no V projection. It is a
`Q4_K_M` file, so its expert sizes are useful for cache planning but not for an exactness claim.
The exact reference must be the official BF16 checkpoint, or a pinned llama.cpp run over the same
artifact when the gate is explicitly Q4-to-Q4.

The current `kindi/gemma4.py` does not satisfy this contract. `inspect_model()` requires the E4B
geometry (42 layers, width 2,560) and the E4B per-layer embedding tensors, and `_QUANTS` has no
Q5_1 entry. Its required-tensor list also demands an `attn_v` for every layer; the 26B global layers
do not have one. The 26B has 30 layers, width 2,816, no per-layer embeddings and Q5_1 down experts,
so it is rejected before a 26B engine could start.

## 2. Expert envelope

The routed experts are stored per layer as one fused gate/up tensor, one down tensor and one
per-expert scale:

| tensor | local shape | type | purpose |
|---|---|---|---|
| `blk.N.ffn_gate_up_exps.weight` | `[2816, 1408, 128]` | Q4_K, except layers 9/15 Q5_K | fused gate and up rows for 128 experts |
| `blk.N.ffn_down_exps.weight` | `[704, 2816, 128]` | Q5_1, except layer 29 Q8_0 | down projection for 128 experts |
| `blk.N.ffn_down_exps.scale` | `[128]` | F32 | official per-expert scale, local name; keep resident and apply at the reference point |

The dense branch is the always-on shared MLP: `ffn_gate.weight`, `ffn_up.weight` and
`ffn_down.weight`, each `2112 x 2816`, plus the router `ffn_gate_inp.weight`
(`[2816, 128]` F32) and `ffn_gate_inp.scale` (`[2816]` F32). The dense branch is not an expert-cache
candidate; it stays resident with attention, embeddings, norms and the router.

Measured Q4 sizes:

| slice | bytes | notes |
|---|---:|---|
| 27 normal layers, per expert | 3,717,120 | Q4_K gate/up (2,230,272) + Q5_1 down (1,486,848) |
| layers 9 and 15, per expert | 4,212,736 | Q5_K gate/up (2,725,888) + Q5_1 down (1,486,848) |
| layer 29, per expert | 4,336,640 | Q4_K gate/up (2,230,272) + Q8_0 down (2,106,368) |
| all 3,840 routed expert weights | 14,479,917,056 | excludes the 15,360 B of scales |
| all dense weights | 2,578,661,496 | attention, dense MLP, router, embeddings, norms |
| top-8 experts in all 30 layers | 904,994,816 | one token with no cache reuse |
| dense plus top-8 weights | 3,483,656,312 | 3.24 GiB of Q4 weight bytes |

The cache slot stores the two weight segments, not the scale. All three per-expert weight sizes are
naturally 16-byte aligned. The fixed slot size is the largest one, **4,336,640 B** (4.14 MiB); allocate
**34,693,120 B** for eight cache slots, plus one transient slot and metadata. A slot layout is:

```text
offset 0        gate_up[2,230,272 or 2,725,888]   # 704 gate rows, then 704 up rows
offset gate_up  down[1,486,848 or 2,106,368]
```

The per-expert scales are only 30 x 128 x 4 = 15,360 B. Keeping them in resident metadata removes a
third copy segment and makes every slot 16-byte aligned. The official router applies them to the
selected routing weights; the local tensor is named `ffn_down_exps.scale`, so G0 must pin the exact
application point rather than assume a different epilogue. If a backend chooses to
keep a scale in the slot instead, the fixed size becomes 4,336,656 B and the ready event must cover
all three copies.

## 3. What changes from the Raon design

| requirement | Raon-1 mini design | Gemma 4 26B requirement |
|---|---|---|
| routed selection | top-6 of 128 | **top-8 of 128** |
| active experts per token | 6 per layer | **8 per layer, 240 across 30 layers** |
| expert payload | mostly one 8.6 MB expert matrix | **two contiguous weight segments, 3.7-4.3 MB total** |
| gate/up | separate or backend-defined | **fused `ffn_gate_up_exps`** |
| shared branch | one shared expert added after the routed sum | **ordinary dense MLP computed in parallel with the MoE branch** |
| router weights | top-6, shared expert | **softmax over all 128, top-8, renormalize, per-expert scale** |
| combine order | router order | **official loop's expert-id order, not router order** |
| cache plan scope | token-level design | **must be layer-scoped; release after each layer** |
| minimum slots | enough for the selected set | **8 cache slots plus a transient slot** |

The top-8 count and 30 layers mean a single token-level pin set would hold 240 experts, about
0.84 GiB of weights, and would exhaust any small cache before the first layer's combine. A 26B
backend must call the cache once per layer:

```python
token = (request_id, layer)
plan = cache.plan(keys, token)
# copy/acquire all eight, compute, combine
cache.release(token)
```

Only eight keys are live at a time. LRU and eviction remain global across layers, but the current
layer's eight pins must not be evicted. The cache must have at least eight usable slots; if it has
fewer, reject the plan before changing arithmetic. A transient slot handles a ninth in-flight copy
while the eight current slots are pinned. The current prototype can express this without a policy
change; it needs `slot_bytes=4_336_640`, at least eight slots, and a caller that uses a per-layer
token.

The fused gate/up tensor is a single copy, not separate gate and up copies. A slot is not `READY`
until both weight segments (and the scale, if stored in the slot) have arrived. If either copy
fails, discard the whole slot; never expose a partially initialised expert to the router.

## 4. Routing and combine contract

The official router is not the Raon router and must not be replaced by a generic top-k helper:

```text
x = router_norm(residual)
x = x * router_scale * scalar_root_size
logits = x @ ffn_gate_inp.weight              # 128 values, fp32
p = softmax(logits, dim=-1)                  # over all 128 experts, fp32
top_p, top_i = topk(p, 8)                    # no explicit tie rule in the reference
top_p = top_p / top_p.sum()                  # renormalise the selected eight
weights = top_p * per_expert_scale[top_i]    # local ffn_down_exps.scale
```

There is no router bias, group-limited routing, sigmoid gate, `routed_scaling_factor` or auxiliary
loss in this path. The tie rule of `torch.topk` is not specified by the model; a backend must pin the
reference implementation rather than invent a lexicographic tie-break.

The shared branch is **parallel**, not a post-hoc addition to the routed sum. In the official
decoder the dense MLP and the MoE branch use separate pre/post norms and are added after their
respective post norms; the result is then added to the residual. The cache must keep the dense MLP
resident and must not route it, evict it or include it in expert hit-rate metrics.

The official `Gemma4TextExperts` loop collects `expert_hit = nonzero(selected_mask)` and calls
`index_add_` for each expert. `nonzero` yields expert ids in ascending order, so the reference
accumulation order is **expert-id order**, not the descending-probability order returned by `topk`.
For a single token, a backend must combine in ascending expert-id order (or reproduce the
reference's exact accumulation). A GPU combine must use a deterministic ordered reduction, not
`atomicAdd` or completion order. The cache may load experts in router order; that is a scheduling
choice and must not change the combine order.

Keeping the per-expert scale in resident metadata is also an arithmetic decision. The official
router applies it to the selected routing weights; the local tensor is named
`ffn_down_exps.scale`. Preserve the reference's exact application point rather than silently moving
it into the down epilogue. If a future implementation folds it differently, it must prove bitwise
equality before it can share a cache mode with the unfolded path.

## 5. Memory and context envelope

The cache is not the only consumer of the 8 GB card. The text decoder's Q4 dense core is about
2.401 GiB, the routed expert set is about 13.49 GiB in host RAM, and the per-layer KV cache is:

| context | global KV (5 layers) | sliding KV (25 layers, ring 1,024) | total KV |
|---:|---:|---:|---:|
| 1,024 | 20 MiB | 200 MiB | 220 MiB |
| 8,192 | 160 MiB | 200 MiB | 360 MiB |
| 32,768 | 640 MiB | 200 MiB | 840 MiB |
| 131,072 | 2.5 GiB | 200 MiB | 2.70 GiB |
| 262,144 | 5.0 GiB | 200 MiB | 5.19 GiB |

The global figure is 5 layers x 2 KV heads x 512 dimensions x K and V x 2 bytes = 20 KiB/token.
The sliding figure is 25 layers x 8 KV heads x 256 dimensions x K and V x 2 bytes for each of 1,024
ring positions. The model has no cross-layer KV sharing. The five global layers have no `attn_v`
weight and derive V from the K projection, but V is still a separate normalized tensor in the first
cache; `attention_k_eq_v` is not treated as a free halving. Deriving V from cached K would need a
new exact kernel and is not assumed here.

A rough 8 GB budget, using the Q4 dense core, a 1.2 GB reserve and 0.5 GiB of workspaces:

| context | dense + KV + reserve + workspace | accounting room for experts |
|---:|---:|---:|
| 8K | 4.37 GiB | about 3.6 GiB |
| 32K | 4.84 GiB | about 3.2 GiB |
| 128K | 6.71 GiB | about 1.3 GiB |
| 256K | 9.21 GiB | over budget |

These are accounting estimates, not measured allocations; the driver, CUDA context and other
desktop use reduce the real free-VRAM figure. The useful conclusion is stable: an 8 GB card can hold
the Q4 dense core and a useful expert cache at 8K-32K, but **not** the full 256K KV cache. A 256K
Q4 run needs host-backed KV and a host-resident expert set, or a larger card. It must not be
advertised as an 8 GB, 256K configuration.

The host side is also tight. The local Q4 text file is 15.89 GiB, the projector is 1.11 GiB and the
256K KV is 5.19 GiB, about 22.2 GiB before OS and page-cache pressure; a 32 GB host is the minimum
practical Q4 host. An official BF16 text decoder is about 47 GiB before the projector, so the exact
reference needs a 64 GB-class host, not the 8 GB target.

```text
8 GB target, Q4:
  resident dense core + router + shared MLP   ~2.40 GiB
  KV at 8K                                    ~0.35 GiB
  reserve + workspaces                        ~1.62 GiB
  available for expert slots                  ~3.6 GiB  (>= 8 slots; not a hit-rate claim)

8 GB target, 256K:
  dense + KV                                  ~7.60 GiB before reserve or workspaces
  result                                      does not fit; use host KV or a larger card
```

## 6. Backend work required

The existing dense `Engine` is not a MoE engine. A 26B backend needs a separate path, for example a
future `kindi/gemma4_moe.py`, even if it reuses attention and dense kernels:

1. Extend the GGUF parser with Q5_1 (type 7) and the 26B tensor names, and make `inspect_model()`
   branch on the 30-layer, width-2,816, no-PLE geometry instead of rejecting it.
2. Split loading into a resident dense core (attention, dense/shared MLP, router, embeddings,
   norms) and a host-mapped expert store. The current `TensorFile.load_resident()` loads every
   tensor; loading 15.9 GiB onto an 8 GB card is not a valid fallback.
3. Implement the 128-way router, top-8 selection, renormalisation and per-expert scale exactly as
   in section 4. Keep the router and all 3,840 scales resident.
4. Implement one fused gate/up expert kernel and one down expert kernel over the host-mapped expert
   slices. A miss copies 3.7-4.3 MB into a slot; a hit launches the same kernel. The copy lane is
   the only first exact lane. The eight-slot allocation, pinned staging, copy events, failure
   cleanup and forced-eviction trace are specified in
   [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md).
5. Run the dense/shared MLP in parallel with the MoE branch and add their post-norm outputs in the
   reference order. Do not append the shared MLP as a ninth cached expert.
6. Add deterministic expert-id-ordered reduction. The router's descending top-k order may be used
   for scheduling, never as the arithmetic order. The ordered-combine kernel and its
   slot/copy-order independence contract are in
   [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md).
7. Keep the 30-layer attention schedule, 16/8/2 head split, 256/512 head dimensions, 1,024 sliding
   rings, global layers 5/11/17/23/29 and the 256K maximum context. The five global layers have no
   `attn_v` weight; derive V from the K projection only if the kernel reproduces the reference
   exactly. The current engine's per-layer `attn_v` requirement, dense assumptions and PLE path do
   not transfer.
8. Keep the vision projector separate. Text-only requests do not need it; image requests add soft
   tokens but do not change expert-cache policy. The 1.19 GB projector is not an expert and must not
   be admitted to the cache.
9. Leave `gemma4_cache.py` as a prompt/state cache. The expert cache is a weight cache and must not
   be serialized into a prompt checkpoint or saved opening.
10. Do not claim two-card support. `gemma4_tp.py` is a dense two-card path and has no expert-aware
    placement or cache.

## 7. Gates

These are acceptance gates, not completed work. Q4 and BF16 results must not be mixed.

| gate | requirement | status |
|---|---|---|
| G0 | Parse the 26B header, accept Q5_1, verify 30/128/top-8/no-PLE, and pin an exact BF16 or same-Q4 llama.cpp reference | pending |
| G1 | Host-resident dense core plus top-8 experts per layer is greedy token-exact against the pinned reference at 1K-2K | pending; no BF16 26B local file |
| G2 | 8 GB Q4 copy lane at 1K and 8K: cold, warm, forced-eviction, copy-failure and reload runs are token-identical; eight slots are never oversubscribed; the seven-eviction-plus-transient trace in [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md#53-the-forced-eviction-trace) passes; per-layer release is visible in metrics | pending |
| G3 | Same exactness at 32K with the sliding ring and global KV | pending |
| G4 | 128K, then 256K with host-backed KV; 256K is not an 8 GB result | pending; 256K does not fit 8 GB |
| G5 | Server, prompt-cache restart and OpenAI-compatible paths; expert-cache health metrics; support-matrix entry only after G2-G4 | pending |

G2 must include a forced-eviction trace with all eight selected experts missing and a transient
copy in flight, plus a copy-failure run that never combines a partial expert. A warm run that
happens to fit entirely in cache is not an eviction test. G4 must
show where the KV tensors live; reporting a 256K context on 8 GB while silently keeping KV in host
memory would be a different configuration and must be labelled as such.

## 8. Acceptance and non-claims

| target | Q4 status | exactness status |
|---|---|---|
| 8 GB, 1K-8K | design target; cache room about 3.6 GiB at 8K | Q4-to-Q4 only; not BF16-exact |
| 8 GB, 32K | tight but within the accounting envelope | pending; not measured |
| 8 GB, 128K | about 1.3 GiB cache room | pending; tight but testable |
| 8 GB, 256K | does not fit | not supported |
| 16 GB, Q4, 256K | plausible VRAM-only target | pending |
| 32 GB host, Q4, 256K host KV | plausible host-backed target | pending |
| 64 GB host, BF16 | exact reference target | pending; no BF16 artifact local |

This document does **not**:

- add Gemma 4 26B to the catalog, installer, server model list or support matrix;
- implement a 26B parser, router, expert kernel, GPU copy path or cache integration; the copy and
  combine contract is design-only in [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md);
- claim that Q4_K_M is token-exact with the official BF16 model;
- claim 256K context on an 8 GB card;
- claim a cache hit rate, tokens/s or peak VRAM figure;
- change `kindi/expert_cache.py`, `kindi/gemma4.py` or any dense engine path.

The cache state machine remains the model-free prototype described in
[MOE-TANDEM.md](MOE-TANDEM.md). The next implementation step is G0, not a performance claim.

## 9. Evidence

Local header and sizes were read from the Ollama model blob and manifest
(`D:\ollamaModels\manifests\registry.ollama.ai\library\gemma4\26b`); the text layer digest and size
are in section 1. The official router and decoder ordering are from the linked HF
`modeling_gemma4.py`; the geometry, top-8 and context values are from the linked config and model
card. The local Q4 file is the only 26B artifact on this host, so G1-G4 have no result yet.

Related records: [GEMMA4-26B-COPY-COMBINE.md](GEMMA4-26B-COPY-COMBINE.md),
[MOE-TANDEM.md](MOE-TANDEM.md), [RAON-1-MINI.md](RAON-1-MINI.md), [FP8-BF16.md](FP8-BF16.md),
[ROADMAP.md](ROADMAP.md).
