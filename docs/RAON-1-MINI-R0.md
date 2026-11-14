# Raon-1 mini R0: tensor manifest and reference record

Status: **R0 artifacts complete, 2026-11-11; reference goldens and R1 not started.** The released
config, tokenizer metadata, license, tensor manifest and a pinned reference command are recorded
here. No Raon backend, converter, pack, catalog entry, installer offer or support claim exists.
The Q4_K_M GGUF is a lossy runtime artifact and is **not** an exactness reference for the MXFP8
base checkpoint.

This page resolves the R0 questions in [RAON-1-MINI.md](RAON-1-MINI.md) and the M0 questions in
[MOE-TANDEM.md](MOE-TANDEM.md). Full machine-readable evidence is in
[`bench-logs/raon-r0/`](benchmarks/raon-r0/README.md).

## 1. Sources and pins

The model is public; the card says access is granted on request (the API's `gated` flag was false
during this audit). No Hugging Face token was used for the config, index or header reads; the full
shards and GGUF payload were not downloaded.

| item | pinned value |
|---|---|
| base model | [`KRAFTON/Raon-1-mini-preview`](https://huggingface.co/KRAFTON/Raon-1-mini-preview), revision `8c9b802acacdbda9df16606689fcf437cf7262fb` |
| GGUF repo | [`KRAFTON/Raon-1-mini-preview-gguf`](https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf), revision `cd61e81afcdfdf4b13cb345b3587c0ba4b918d00` |
| base format | BF16 embeddings/attention/shared/dense, **MXFP8 E4M3 routed experts** with E8M0 `U8` block scales (`weight_block_size [1, 32]`) |
| base size | 7 shards, 32,702,145,088 file bytes; **32,699,649,368 tensor bytes** |
| GGUF reference pack | `raon-1-mini-preview-Q4_K_M.gguf`, 19,686,434,240 bytes, SHA-256 `ba56ad4bd6d15f8c4255876ff252890143074a5d3835fccfc656cee915999ee8` |
| license | **Apache-2.0**; `LICENSE` SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`; `general.license = apache-2.0` in the GGUF |
| reference implementation | [`krafton-ai/Raon-1-mini-preview`](https://github.com/krafton-ai/Raon-1-mini-preview), commit `cfa219538a11a0f10a6ae40ede76bdb8efc4dc4e` |
| llama.cpp reference | [`krafton-ai/llama.cpp`](https://github.com/krafton-ai/llama.cpp), branch `raon-1-mini-preview`, commit `fb895662d25868756a19253ca5e0688f99b18994` (2026-10-20); upstream support pending |
| config | SHA-256 `ccd1541ed7d4e26dd86e0b9c706f13e5dbdf968c68824c27c2eb53f23d735ee3` |
| safetensors index | SHA-256 `698935d27fd0a7be12be30beb0f93408e378b158cb0913390652c7c6a51713a2` |
| model code | `modeling_raon1_mini.py` SHA-256 `ca0742570e8c9b730115f2feedbcd9f424dc47d21b132729f27c4d8ac6998fc6` |
| chat template | SHA-256 `34a9f0219d315211631885b22d8e4973943c092ef319b57c11853127e392b0ac` |
| tokenizer config | SHA-256 `d5d612aeef88d4564d2010ab7f65b7e54821acfd7ccda80739dd19761840bcde` |
| full pins | [`source-pins.json`](benchmarks/raon-r0/source-pins.json) |

Base shard SHA-256 values (full list in `source-pins.json`):

| shard | bytes | SHA-256 |
|---|---:|---|
| 1 | 4,998,364,808 | `c18545414b88b1b1c694c23a275141885e26168bedca23f821b9b87d316f3867` |
| 2 | 5,000,376,992 | `bbf5fb70a5a6c4af680b84ed482891c7e819829b4805b7511ef6b7daf98a9af1` |
| 3 | 5,000,397,960 | `7e46f3844fd2d659c833bac9d09944407011c541bcf9bc7e71900bf7ac4148f4` |
| 4 | 4,998,818,792 | `6264c2de64c006fb41d2dd215c87e5132681dd2e0e3419d4a0206daa8ae5ea59` |
| 5 | 4,974,921,360 | `ad0dcf3f8ae44e80f1d5171156ef86cd8633b89a826a34a93e0367cfcf0d190e` |
| 6 | 4,989,223,544 | `6628b23347ac6f3a2f8b42dfb93edb7e6e1c08a3a6bae6f09acefb7340190e25` |
| 7 | 2,740,041,632 | `64f717749ef89be949fa95a82118aed668257057d98a807862761b9ed2af74ec` |

The GGUF SHA-256 is copied from the repository's
[`SHA256SUMS`](https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf/raw/main/SHA256SUMS). Only
the first 128 MiB of that file was fetched for the header; the published hash is the full-file
hash. The GGUF repo also publishes `IQ4_XS`, `Q5_K_M`, `Q6_K`, `Q8_0` and a DFlash2 draft; none
was downloaded whole.

## 2. Verified model contract

The model card's rounded “30B-A4B” label resolves to the measured values below. The header total
is 200 parameters larger than Hugging Face's `safetensors.parameters` summary because HF excludes
the 200 scalar tensors (50 F32 `eta`, 150 BF16 `alpha`); the header is the complete tensor count.

| field | released config / measured |
|---|---|
| architecture | `Raon1MiniForCausalLM`, model type `raon1_mini`; GGUF architecture `raon1` |
| total parameters | **28,783,218,602** header elements; HF summary **28,783,218,402** excluding 200 scalars |
| active parameters | **4,317,319,082** (6/128 of routed-expert elements plus the dense core; 15.0%) |
| layers | **25** |
| attention schedule | **18 RDA linear-attention + 7 gated MLA full-attention**; 0-based RDA `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]`, MLA `[3,7,11,15,19,23,24]` |
| MLP schedule | layer 0 **dense**, layers 1-24 **MoE** (`first_k_dense_replace=1`, `moe_layer_freq=1`) |
| hidden size | 2,560 |
| attention heads | 40 for both RDA and MLA |
| RDA head dimension | 128; projection width 5,120; conv kernel 4; decay rank 128; gate lower bound -5.0 |
| MLA geometry | 192 QK (128 NoPE + 64 extra channels) / 128 V; Q LoRA 576; KV LoRA 512; output gate |
| position encoding | **NoPE**; the 64 extra MLA channels are not rotated; `rope_parameters: null` |
| routed experts | **128**, **top-6** per token, sigmoid router, renormalize, scale 1.0 |
| shared expert | **1**, always on |
| expert intermediate size | 1,088 |
| dense FFN intermediate size | 12,032 (layer 0 only) |
| activation | **SiTU-GLU**, beta 4.0 and linear beta 25.0 |
| residual | Raon-mHC with **4 streams**; final output is the mean over streams, then RMSNorm |
| vocabulary | 259,840; untied **lm_head**; pad 259,200; EOS 259,202 (and 259,200 in the HF generation config) |
| context | 262,144 tokens |
| norms | RMSNorm epsilon 1e-5; MLA latent norm epsilon 1e-6 |
| base storage | 9,216 F8_E4M3 expert matrices + 9,216 U8 scale tensors; 892 BF16 + 110 F32 other tensors |
| GGUF storage | 731 tensors: 51 Q4_K, 143 Q6_K, 82 Q8_0, 12 Q5_0, 122 F16, 321 F32 |

Model-card access: Apache-2.0 permits commercial use and redistribution of the model components,
subject to the license. The card asks for an access request (the API's `gated` flag was false during
this audit); an installer should honor that request/login flow and must not silently fetch the
32.7 GB base. This page is not a redistribution promise.

## 3. Tensor manifest

The complete inventories are generated from the actual headers, not from a template:

| artifact | contents |
|---|---|
| [`base-tensor-inventory.json`](benchmarks/raon-r0/base-tensor-inventory.json) | **19,434** base tensors with `name`, `dtype`, `shape`, `bytes`, shard |
| [`gguf-q4km-tensor-inventory.json`](benchmarks/raon-r0/gguf-q4km-tensor-inventory.json) | **731** GGUF tensors with `name`, `shape`, `type_id`, `type`, `offset`, `bytes` |
| [`manifest-summary.json`](benchmarks/raon-r0/manifest-summary.json) | counts, per-layer bytes, cache formulas, comparisons |
| [`build_manifest.py`](benchmarks/raon-r0/build_manifest.py) | the reproducible validator/builder |

### 3.1 Base role counts

| role | name pattern | count | dtype / shape | bytes |
|---|---|---:|---|---:|
| embedding | `model.embed_tokens.weight` | 1 | BF16 `[259840, 2560]` | 1,330,380,800 |
| LM head | `lm_head.weight` | 1 | BF16 `[259840, 2560]` | 1,330,380,800 |
| final norm | `model.norm.weight` | 1 | BF16 `[2560]` | 5,120 |
| layer norms | `model.layers.N.{input_layernorm,post_attention_layernorm}.weight` | 50 | BF16 `[2560]` | 256,000 |
| mHC sites | `model.layers.N.{self_attn,mlp}_raon_mhc.*` | 500 | 10 tensors/site, 2 sites/layer; BF16/F32 | 17,410,200 |
| RDA | `model.layers.N.self_attn.*` | 270 | 15 tensors/layer × 18 layers | 2,404,646,208 |
| MLA | `model.layers.N.self_attn.*` | 56 | 8 tensors/layer × 7 layers | 543,636,352 |
| dense FFN | `model.layers.0.mlp.{gate,up,down}_proj.weight` | 3 | BF16 `[12032,2560]` or `[2560,12032]` | 184,811,520 |
| router | `model.layers.N.moe.gate.{weight,e_score_correction_bias}` | 48 | BF16 `[128,2560]` + F32 `[128]` | 15,740,928 |
| shared expert | `model.layers.N.moe.shared_experts.*` | 72 | BF16, 3 tensors/layer × 24 | 401,080,320 |
| routed experts | `model.layers.N.moe.experts.E.*` | **18,432** | 6 tensors/expert: F8_E4M3 matrix + U8 scale for gate/up/down | **26,471,301,120** |
| **total** | | **19,434** | | **32,699,649,368** |

The exact per-layer bytes are in `manifest-summary.json`; the compact schedule is:

| layers | attention | MLP | tensors/layer | bytes/layer |
|---|---|---|---:|---:|
| 0 | RDA | dense | 40 | 319,109,624 |
| 1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22 (17) | RDA | MoE | 810 | 1,254,636,536 |
| 3,7,11,15,19,23,24 (7) | MLA | MoE | 803 | 1,198,707,416 |
| top level | — | — | 3 | 2,660,766,720 |

### 3.2 Expert sizes

Every routed expert is the same size in the base checkpoint: **8,616,960 bytes** (8.21875 MiB):

```
gate_proj  F8_E4M3 [1088,2560]  2,785,280 B
up_proj    F8_E4M3 [1088,2560]  2,785,280 B
down_proj  F8_E4M3 [2560,1088]  2,785,280 B
three U8 scale tensors [1088,80]/[1088,80]/[2560,34]  261,120 B
                                              total  8,616,960 B
```

There are 24 MoE layers × 128 experts = **3,072 slots**, 18,432 routed-expert tensors, and
26,471,301,120 routed-expert bytes (80.95% of the base tensor bytes). Six cold experts are
**51,701,760 bytes** before staging/alignment; 6/128 of the whole routed set is
**1,240,842,240 bytes** if fully resident. The dense/core remainder is **6,228,348,248 bytes**.

The Q4_K_M pack is 19,686,434,240 bytes; its 72 packed expert tensors are
`[2560,1088,128]` gate/up and `[1088,2560,128]` down. A complete Q4 expert is
**5,048,320-6,092,800 bytes** depending on the layer's down-projection type; six cold experts are
**30,289,920-36,556,800 bytes**. Q4 lowers the host footprint but is lossy: it must not be compared
token-for-token with the MXFP8 base.

### 3.3 GGUF name mapping

| base role | GGUF name |
|---|---|
| embedding / head / final norm | `token_embd.weight` / `output.weight` / `output_norm.weight` |
| RDA q,k,v,o | `blk.N.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight` |
| RDA short convs | `ssm_conv1d_q.weight`, `ssm_conv1d_k.weight`, `ssm_conv1d_v.weight` |
| RDA decay/gates | `ssm_f_a.weight`, `ssm_f_b.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_beta.weight`, `ssm_rda.weight`, `ssm_g.weight`, `ssm_norm.weight` |
| MLA q latent | `attn_q_a.weight`, `attn_q_a_norm.weight`, `attn_q_b.weight` |
| MLA kv latent | `attn_kv_a_mqa.weight`, `attn_kv_a_norm.weight`, `attn_k_b.weight`, `attn_v_b.weight` |
| MLA output gate / projection | `attn_gate.weight`, `attn_output.weight` |
| mHC | `hc_attn_base.weight`, `hc_attn_fn.weight`, `hc_attn_scale.weight`, and `hc_ffn_*` |
| dense FFN | `blk.0.ffn_gate/up/down.weight` |
| router | `ffn_gate_inp.weight` + `exp_probs_b.bias` |
| shared expert | `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, `ffn_down_shexp.weight` |
| routed experts | `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight` (packed last dimension 128) |

The GGUF has **one packed tensor per routed projection per layer**, not 128 separate tensors;
that is why 731 GGUF tensors cover the same 3,072 expert slots. The base names are the converter's
authoritative source; the GGUF names are the reference runtime's layout.

## 4. Attention, state and cache

### 4.1 RDA

RDA is a gated delta-rule linear attention. Per token, with q/k L2-normalized (eps 1e-6), q scaled
by `1/sqrt(128)`, and fp32 state:

```
log_alpha_t = -5 * sigmoid(exp(A_log) * (f_b(f_a(x_t)) + dt_bias))
beta_t      = sigmoid(b_proj(x_t))
v_t         = v_t * exp(logsigmoid(beta_raw_t + rda_w(x_t))
                        - logsigmoid(beta_raw_t))
state_t     = state_{t-1} * exp(log_alpha_t)
pred_t      = state_t @ k_t
state_t     = state_t + beta_t * (v_t - pred_t) outer k_t
out_t       = state_t @ q_t
```

The three depthwise causal convolutions have kernel 4 and keep the raw q/k/v window. The output is
per-head RMSNorm, sigmoid output gate, then `o_proj`; there is no RoPE and no growing RDA K/V.

State per sequence: 18 layers × (conv state `3*5120*3` BF16 + recurrent state `40*128*128` FP32)
= **48,844,800 bytes (46.58 MiB)**, independent of context. The conv part is 1,658,880 bytes and
the recurrent part is 47,185,920 bytes.

### 4.2 MLA and the two cache sizes

The seven MLA layers are NoPE. The published cache claim is **7.875 KiB/token BF16**:
7 layers × (512 latent + 64 extra key channels) × 2 bytes = **8,064 bytes/token**. At 262,144
tokens that is **2,113,536,000 bytes = 1.96875 GiB**; with the RDA state the sequence state is
**2.014 GiB**.

The released Hugging Face reference is not compressed: `Raon1MiniDynamicCache` stores expanded
per-head K/V `[B,40,T,192]` and `[B,40,T,128]`, i.e. **179,200 bytes/token = 175 KiB/token**.
At 262,144 tokens that is **43.75 GiB**, or **43.795 GiB** with the RDA state. An 8 GB card cannot
hold that. The GGUF metadata exposes `key_length=576`, `value_length=512`,
`key_length_mla=192`, `value_length_mla=128`; the exact llama.cpp cache layout must be read from
the pinned branch before Kindi fixes its own budget. Do not silently use the HF expanded cache
number as the target, and do not assume the 8 KiB published number is implemented by every
runtime.

The practical R4/R5 budget therefore has two named modes: **compressed latent** (published 8 KiB
class; 2.014 GiB at 262K for one sequence) and **expanded per-head** (43.795 GiB at 262K; reference
implementation only). Any quantized K/V is a separate lossy mode.

## 5. Router, experts and residual order

`Raon1MiniRouter` computes `logits = F.linear(x.float(), W.float())`, `scores = sigmoid(logits)`,
then selects the top-6 of `scores + e_score_correction_bias` with `torch.topk(..., sorted=True)`.
The selected values are the **sigmoid scores, not the biased scores**, renormalized by their sum;
`routed_scaling_factor` is 1.0. The shared expert is added **after** the routed sum, as
`out = routed + shared_experts(x)`. The native grouped path sorts token-expert pairs by expert for
the GEMM, then scatters back to `(token, slot)` order and performs a fixed-order sum over the six
slots; it never uses atomics for the final combine.

There is **no explicit tie-break** in the released code beyond `torch.topk`. A tied router score is
therefore not a specified model contract. R1 must test tied scores against the pinned reference;
Kindi must not invent a lower-index tie-break and call it exact. The cache design's rule — preserve
the reference order, do not sort/deduplicate/renormalize unless the reference does — is unchanged.

Each routed expert is `down(situ_glu(gate(x), up(x)))` with SiTU-GLU betas 4.0 and 25.0. The dense
layer 0 FFN uses the same activation with intermediate size 12,032; the shared expert uses 1,088.

Raon-mHC keeps four residual streams. Each attention/MLP site has its own selector: 4 `phi_pre` +
4 `phi_post` + 9 `phi_z` rows over the 10,240-wide stream vector; the chart uses a Helmert basis,
the `res.eta` contraction and a sigmoid scale. The initial stream is the embedding repeated four
times; the final hidden state is the mean over streams, then `model.norm`. The 20 mHC tensors per
layer are part of the executable contract, not an optional residual tweak.

## 6. Reference commands

### 6.1 Exact reference: base MXFP8 with greedy Transformers

The exactness reference is the **base MXFP8 checkpoint**, not Q4_K_M. The base is 32.7 GB of
tensors; use a host with enough memory (the model card suggests a 32 GB GPU with CPU offload, but
an 80 GB-class device avoids offload differences). Install the card's pinned versions and run the
harness:

```bash
pip install "transformers==5.16.1" "accelerate==1.15.0"
hf auth login   # accept the automatic Raon-1 mini access request first
python bench-logs/raon-r0/reference_greedy.py \
    --output bench-logs/raon-r0/goldens/transformers-greedy.json
```

The harness uses `do_sample=False`, `presence_penalty=0.0`, thinking on, a fixed four-prompt set,
`max_new_tokens=64`, revision `8c9b802...`, and records prompt token IDs, output token IDs, decoded
text, package versions and per-prompt SHA-256. Run it twice into separate files and require the
output token IDs to match exactly before treating it as a golden. The four prompts and the exact
flags are in the script.

Why `presence_penalty=0`: the released `generation_config.json` sets 1.5 as the served default, and
`Raon1MiniForCausalLM` applies it even for greedy generation. A greedy gate with 1.5 would not be
the same arithmetic as a plain argmax reference. The official sampling recipe uses 1.5 for chat and
0 for coding/math; neither is the R0 exactness lane.

### 6.2 Q4_K_M runtime sanity (not exact)

The model card's llama.cpp branch is a runtime check for the 19.7 GB quantized pack. It is **not**
token-exact against the MXFP8 base. Build the pinned commit and verify the file:

```bash
git clone -b raon-1-mini-preview https://github.com/krafton-ai/llama.cpp
cd llama.cpp
git checkout fb895662d25868756a19253ca5e0688f99b18994
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Download the exact Q4_K_M file, then:
grep ' raon-1-mini-preview-Q4_K_M.gguf$' SHA256SUMS | sha256sum -c -   # expected ba56ad4b...
```

Published sampling command (the model card's chat recipe):

```bash
./build/bin/llama-cli \
  -m /path/raon-1-mini-preview-Q4_K_M.gguf \
  --ctx-size 262144 --n-predict 32768 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --repeat-penalty 1.0 --frequency-penalty 0.0 --presence-penalty 1.5 \
  --reasoning on \
  --chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
```

Deterministic short-context runtime probe (still lossy; compare only Q4-to-Q4):

```bash
./build/bin/llama-cli \
  -m /path/raon-1-mini-preview-Q4_K_M.gguf \
  --ctx-size 2048 --n-predict 32 \
  --temp 0 --top-k 1 --top-p 1.0 --min-p 0 \
  --repeat-penalty 1.0 --presence-penalty 0 --seed 0 \
  --reasoning on \
  --chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
```

The branch's flags were checked in `common/arg.cpp` at the pinned commit: `--reasoning`,
`--chat-template-kwargs`, `--presence-penalty`, `--spec-type`, `--spec-draft-n-max` and
`--spec-draft-sampling` exist. The build and runs were **not** executed here: this host has no Raon
llama.cpp binary, and the 19.7 GB Q4 file plus a 262K context does not fit the 8 GB RTX 4060 Ti.
The model card's official command uses `-hf ...:Q4_K_M`; a local file is used above so the exact
hash is checked first.

### 6.3 Checklist

- [ ] Accept the model access request; `hf auth login`; never commit the token.
- [ ] Verify the base revision and all seven shard hashes (`source-pins.json`); verify the Q4 file's
      SHA-256 before running it.
- [ ] Run `reference_greedy.py` twice on a host that holds the 32.7 GB base; require identical
      output token IDs; save both JSON files and their hashes.
- [ ] Record the GPU, driver, CUDA, PyTorch, Transformers and Accelerate versions in the golden.
- [ ] Run the Q4 command only as a runtime sanity check; label it lossy and do not use it as the
      exactness target.
- [ ] If either reference cannot run, record the blocker and stop R1. Do not substitute Q4 or a
      sampled output for the exact greedy reference.
- [ ] Before any Kindi pack, decide whether it preserves MXFP8 or uses a documented lossy format;
      the 4-bit pack cannot inherit the base's exactness claim.

## 7. R0 answers and remaining gaps

| R0 question | answer |
|---|---|
| Q1 exact names/geometry/recurrence | **resolved**: 25 layers, hidden 2,560, 40 heads, 18 RDA + 7 MLA, RDA head 128, MLA 192/128, q/kv LoRA 576/512, NoPE, SiTU-GLU, mHC-4; full names/shapes and recurrence above |
| Q2 routed/shared/dense sizes | **resolved**: 128 routed experts, top-6 + 1 shared, 1,088 expert FFN, dense layer 0 FFN 12,032; 8,616,960 B per MXFP8 expert; Q4 per-expert sizes above |
| Q3 dtype/tokenizer/template | **resolved**: BF16 core + MXFP8 routed experts in base; Q4_K_M/Q5_K_M/Q6_K/Q8_0/IQ4_XS GGUF; 259,840-token BPE; ChatML template with thinking; Apache-2.0 |
| Q4 router normalization/tie/shared order | **partly resolved**: sigmoid, bias for selection only, top-6, renormalize selected sigmoid scores, scale 1.0, shared added after; **tie rule unspecified** and must be pinned by reference |
| Q5 K/V bytes | **two answers**: published compressed latent **8,064 B/token (7.875 KiB)**; HF reference expanded **179,200 B/token (175 KiB)**; exact llama.cpp cache layout still to confirm from the branch |
| Q6 paged/sliding/RoPE | **NoPE**, no sliding window; 7 full-attention layers; RDA state is fixed and recurrent; no paged cache is in the HF reference; paged/host-backed cache is a Kindi/runtime decision, not a model requirement |
| Q7 license/access | **Apache-2.0**, commercial use and redistribution subject to the license; the card asks for an access request, so automatic download should honor the HF login/request flow; no token committed |
| Q8 reference runtime/command | **pinned**: exact greedy Transformers base command and harness; llama.cpp Q4 runtime sanity command; neither golden has been run on this host |

The remaining R1 blockers are the two reference runs, the router tie test, the exact llama.cpp
cache layout, and a decision on the first pack format. The Q4 pack does not close the exactness
gate.

## 8. What this does not claim

- It does not add Raon-1 mini to `kindi/catalog.py`, `install.sh`, the support matrix or the
  installer.
- It does not provide a converter, pack, engine, expert cache, server profile or measurement.
- It does not claim Q4_K_M is token-exact against MXFP8, or that the HF expanded cache is the
  published 8 KiB cache.
- It does not claim a tokens/s, VRAM, host-RAM, cache-hit or reference-output result.
- It does not claim a router tie rule that the released code does not specify.

Related: [RAON-1-MINI.md](RAON-1-MINI.md), [MOE-TANDEM.md](MOE-TANDEM.md),
[ROADMAP.md](ROADMAP.md), [`bench-logs/raon-r0/README.md`](benchmarks/raon-r0/README.md).
