# Ternary-MoE census

Status: **canonical, machine-checked census** (builds and external card claims checked 2026-11-19;
arithmetic audited 2026-11-19). This page is the one source for the Qwen3.6-35B-A3B ternary builds, the other published
ternary MoE builds, their expert-slot count and the leads that are not counted. The roadmap keeps
only the Qwen3.6 backend gates and links here.

`scripts/check_ternary_moe.py` parses the two build tables and the counts below. It fails if a row
disappears, a row count changes, a slot product does not multiply out, the slot total drifts, or the
unread/checked lists stop matching their counts.

This is an external-artifact census, not a Kindi measurement. A row here is evidence about somebody
else's file; it is not a model run, a fit claim or a benchmark result.

## Counts

| what | count |
|---|---:|
| Qwen3.6-35B-A3B build rows | 2 |
| other published-build rows | 6 |
| total counted build rows | 8 |
| expert slots across the six other-build rows | 51,864 |
| unread leads (not counted) | 3 |
| checked and not counted | 3 |

The slot total is the sum of the per-row products in the other-build table; it is not a count of
stored tensors, and the Qwen3.6 pair is not added to it because both rows are quantizations of the
same base model.

## Qwen3.6-35B-A3B builds

The two Qwen3.6-35B-A3B ternary MoE builds reviewed so far are quantizations of that model; neither
card's base-architecture label is asserted here for both. The official Hugging Face config is
`model_type: qwen3_5_moe`; llama.cpp and GGUF call the same text architecture `qwen35moe`, and
"llama.cpp-compatible candidate" below means a file with a standard GGUF
`general.architecture = qwen35moe` tensor layout that upstream llama.cpp can load and that can be
pinned for the Q0 reference gate; neither qualifies.

| build | contents | format and runtime | why it cannot count as a llama.cpp candidate yet |
|---|---|---|---|
| [russellwmy/qwen3.6-35b-a3b-ternary](https://huggingface.co/russellwmy/qwen3.6-35b-a3b-ternary) | 9.1 GB of 2-bit ternary expert codes plus 2.3 GB of the rest; a bundle, not a runnable checkpoint | ironbark pack for Apple Silicon; its card says it is not loadable by `transformers`, `mlx-lm` or `mlx-vlm` | custom safetensors bundle and ironbark trit-pack format, not GGUF; no `qwen35moe` tensor inventory; no byte-level dequant/export test here; no upstream llama.cpp loader or NVIDIA kernel path; no pinned llama.cpp reference or token-exact `kindi verify` |
| [manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64) | 12.6 GB: 3-bit attention, ternary gate/up experts and 4-bit down experts; its separate pure-ternary sibling is 9.4 GB | TurboQuant-MLX safetensors for Apple Silicon | not GGUF; the mixed 3-bit/ternary/4-bit expert layout needs a converter with bit-level dequant tests (none in this repository); no upstream llama.cpp loader or NVIDIA path; the 0/4 vs 3/3 quality split is one agentic task, not a token-exact reference |

### External card claims (not Kindi measurements)

The official [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) reports
**SWE-bench Verified 73.4** and **LiveCodeBench v6 80.4** for the base model. Those are the
publisher's benchmark claims, not a Kindi run and not a claim about either ternary build.

The [russellwmy ternary card](https://huggingface.co/russellwmy/qwen3.6-35b-a3b-ternary) compares its
ternary experts with the 4-bit MLX checkpoint it was quantized from (reasoning off, all experts
resident, Apple Silicon): HumanEval pass@1 is **140/164 Python** and **108/156 Rust** for ternary
versus **153/164** and **142/156** for the 4-bit source; reply perplexity is **1.227 versus 1.102**
(higher is worse). This is the card author's harness on Apple Silicon, not a Kindi verification,
and it does not transfer to an NVIDIA or standard-GGUF path.

The Apple-only limitation is unchanged at the 2026-11-19 recheck: the russellwmy build still
requires an Apple Silicon Mac and an `ironbark` build and states that the bundle is not loadable by
`transformers`, `mlx-lm` or `mlx-vlm`; the manjunathshiva build still targets TurboQuant-MLX on
Apple Silicon. Neither has an NVIDIA kernel path or a standard `qwen35moe` GGUF. Neither is
recognized by `kindi/moe_compat.py` either: that path recognizes only the standard `qwen35moe`
architecture, so these custom bundles cannot reach the backend refusal or become Q0 evidence.

Both Qwen3.6 builds stay out of the Q0 candidate set until one of two things exists: a byte-pinned
standard GGUF with the `qwen35moe` tensor inventory, or a converter whose dequant is tested
bit-for-bit. Either route then needs the same token-exact `kindi verify` against the pinned
llama.cpp reference and the existing KLD/PPL gate. The 16 GB route in the roadmap uses the ordinary
Unsloth `UD-IQ2`/`UD-IQ3` GGUFs; those are stock-llama.cpp artifacts, not Kindi-convertible today
(the existing converter has no `qwen35moe` path and does not read IQ2_M). Neither Apple-only bundle
changes that.

## Other published builds

These are outside the Qwen3.6 Q0 question, and none is a Kindi measurement. The Nemotron row differs
in kind from the rest: Kindi already ships that build, so it is not a lead.

| build | base and ternary scope | format and runtime | status here |
|---|---|---|---|
| [Nota AI Nemotron 3.5 Lightning 30B-A3B, GGUF by boro64](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF) | `nemotron_h_moe`: 52 blocks (23 MoE, 23 Mamba-2, 6 attention), 128 routed experts per MoE layer (top-6) + one shared, i.e. 2,944 routed-expert slots (23 × 128; 69 fused gate/up/down tensors); Nota AI's ternary MLX checkpoint repacked bit-for-bit; 162 linear tensors at Q2_0, embeddings and lm_head at Q8_0, router/norms/SSM scalars F32 | 8.82 GB standard GGUF Q2_0; the card reports stock llama.cpp (b11555+/b11796) and the PrismML fork; 262K native | **Already served:** `kindi use lightning-30b` pins this file (`kindi/catalog.py`) and `kindi_lightning` reads Q2_0/Q8_0/F32 from it. It is Nota's intermediate checkpoint at ~89% of the BF16 teacher, not a Qwen3.6 Q0 artifact. |
| [DeepGrove Maple-Preview](https://deepgrove.ai/maple-preview) | 20B-A1B, natively trained ternary; 24 layers, 256 experts per layer (top-8), i.e. 6,144 expert slots (24 × 256); 3:1 SWA-512/global attention; 5.31 GB 2-bit MLX deployment checkpoint and official TQ1_0/TQ2_0 GGUFs | MLX; GGUF TQ1_0/TQ2_0; merge-ready llama.cpp PR [#27000](https://github.com/ggml-org/llama.cpp/pull/27000) adds `maple` for CPU and reports the official GGUFs load there | no `maple` backend in Kindi; unlike the other rows, the weights were trained ternary from the start rather than post-quantized. |
| [Schackay3/Qwen3-Next-80B-A3B-Instruct-ternary-GGUF](https://huggingface.co/Schackay3/Qwen3-Next-80B-A3B-Instruct-ternary-GGUF) | `qwen3next`: 24,576 experts (512 × 48) with ternary gate/up/down (asymmetric group-128), shared expert ternary; DeltaNet/attention, router, embeddings, norms and lm_head bf16 | 29.3 GB GGUF **Q2_K** (2.94 bpw), not a native 1.58-bit container; llama.cpp b10331 | 80B is far outside the 16 GB target; `qwen3next` is not `qwen35moe`, and the card itself says the file is Q2_K rather than true ternary. |
| [kabcpc/Ternary-Flint-40B-GGUF](https://huggingface.co/kabcpc/Ternary-Flint-40B-GGUF) | `flint`: 39.8B total / 2.5B active, 49 layers (2 dense + 47 MoE), 256 routed experts per MoE layer (top-6) + one shared, i.e. 12,032 expert slots (47 × 256); 141 fused expert projection tensors (47 × gate/up/down), 124 at PTQ1_0 and 17 sensitive down-projection tensors at Q4_1 in the hybrid | 9.31-12.49 GiB GGUF PTQ1_0 (type 143) / PQ2_0 (type 142) / hybrids; production path is a `llama-flint` build with a Metal LUT; 262K | Kindi recognizes `flint` and refuses it (`kindi/moe_compat.py`); it needs its own loader and pinned reference. |
| [ToxeH/Qwen3-Coder-30B-A3B-Instruct-Ternary2x](https://huggingface.co/ToxeH/Qwen3-Coder-30B-A3B-Instruct-Ternary2x) | `qwen3_moe`: 48 layers, 128 experts per layer (top-8), i.e. 6,144 expert slots (48 × 128); 2-plane ternary experts (3-bit MLX), 3-plane ternary attention/embeddings/lm_head (6-bit), router/norms bf16 | 12.99 GiB MLX safetensors; MLX engines on Apple Silicon | Kindi recognizes `qwen3moe` and refuses it; this is a multi-plane ternary decomposition, not a pure ternary checkpoint. |
| [micro-terse](https://github.com/michelangeloromerochisco/micro-terse) | 423M total / ~320M active, 12 layers, 4 experts per odd layer (6 MoE layers), i.e. 24 expert slots (6 × 4), top-2; Q/K/V/O and gate/up/down ternary, embeddings/head/norms/router/biases full precision | 182 MB TQ2_0 GGUF (tied embedding Q6_K); custom `terse` llama.cpp fork; 4K context, CPU | research proof-of-concept; Kindi has no `terse` architecture. |

## Counting convention

"Experts" is a per-MoE-layer count unless the cell shows a product; the product is the total expert
slots, not the number of stored tensors. Each slot owns gate/up/down matrices, while GGUF/MLX
commonly fuses all experts of one layer into one tensor per projection. The Flint row's 124 + 17
counts are those fused projection tensors (47 MoE layers × 3), not 141 experts.

## Unread leads

These are recorded so the census does not lose them, but they are not counted as builds:

- **MoTE** ([arXiv:2506.14435](https://arxiv.org/abs/2506.14435), v2 2026-01-07): a method for
  training ternary routed experts up-cycled from a dense checkpoint; the experiments are multimodal
  (SigLIP-L vision encoder + Qwen2.5). No public checkpoint or model card was found in the
  2026-11-18 search, so there is nothing to inspect beyond the paper.
- **ButterflyMoE** ([arXiv:2601.13563](https://arxiv.org/abs/2601.13563)): a method paper on
  structured butterfly orbits for ternary experts. No released checkpoint was found; paper only.
- **Nota AI further-recovered Nemotron checkpoint:** the published Nota card says the further
  recovered model is available from the authors on request. There is no public file to read, so the
  89%-recovery intermediate above is the only Nota artifact counted.

## Checked and not counted

- [UlukaDev/bitnet-ternary-moe-gguf](https://huggingface.co/UlukaDev/bitnet-ternary-moe-gguf) is a
  dense `microsoft/bitnet-b1.58-2B-4T` base plus two LoRA adapters and an external router, not a
  single ternary MoE checkpoint.
- [stiger1000/TC-MoE](https://huggingface.co/stiger1000/TC-MoE) applies `{-1, 0, 1}` to expert
  *choice*, not to expert weights.
- The Prism Ternary-Bonsai models (1.7B, 8B and 27B) are dense, not MoE.

## Not an MoE

Granite 4.1 30B (`granite`, 64 blocks, 131,072 context) is a dense 30B GGUF, so it is not part of
the small-MoE recognition table and not a row here. It is already on disk under `~/.ollama/models` and can be linked with
`tools/ollama_links.ps1` for a future native/generic-engine evaluation, but it needs a Granite
loader and pinned reference like any other dense architecture; Qwen3 30B (non-coder) shares the
`qwen3moe` architecture and therefore follows the Qwen3-Coder row once a loader exists.
