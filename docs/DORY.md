# Dory1.0-4B: long-term looped-model support goal

Status: **planning only, 2026-11-11. Dory1.0-4B is not in `kindi/catalog.py`, has no converter,
engine, pack, installer offer or measured result. This is a goal and an acceptance contract, not a
support claim.**

The target is an OpenAI-compatible Kindi server for `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B`, with a
runtime-selectable number of recurrent loops. Dory is the first model in this project whose depth is
a *parameter of the request*, not a property of the checkpoint: the same weights run 72 physical-layer
applications at one loop and 192 at four. That, plus the normalized-GPT (nGPT) residual rule, is why
this needs its own backend rather than a catalog row.

Official sources:

- <https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16>
- <https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4>
- <https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF>
- [Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998) (arXiv 2611.03998)

## Model contract

Read from the released `config.json` unless marked otherwise. Anything not listed must be checked
against the reference implementation before implementation; do not infer it from a dense engine.

| item | value |
|---|---|
| parameters | **4 B** (BF16 checkpoint reported at 3,975,229,440 parameters) |
| architecture | **looped (latent-recurrent) normalized-GPT**, `model_type: dory`, `DoryForCausalLM` |
| physical layers | **72**, each a single sublayer: attention or MLP, never both |
| block partition | **16 input + 40 shared recurrent + 16 output** layers |
| loops | `n_recurrent_loops: 2` in the checkpoint; **1-4** supported by the release, base model evaluated to 8 |
| attention | **27 sliding-window + 9 full-attention** layers, a 3:1 pattern inside each eight-layer group |
| sliding window | **1,024 tokens** |
| RoPE | sliding θ = **1e4**, full θ = **1e7**, applied to all Q/K dimensions |
| heads | **16 query / 4 KV** (GQA), `head_dim: 256` |
| hidden / FFN | **2,560** / **7,168**, SwiGLU, no biases |
| vocab | **131,072**, untied embeddings and LM head |
| context | **262,144 tokens** (256K), validated by NVIDIA |
| normalization | nGPT **L2 (`justnorm`)** only; no LayerNorm or RMSNorm anywhere |
| residual | learned per-channel blend of normalized sublayer output with the normalized residual, then L2; initial effective rate 0.05 |
| license | **OpenMDW 1.1** (commercial and non-commercial use) |
| release | 2026-10-16 |
| formats | BF16 (W16A16), MXFP8 (W8A16), **NVFP4** (W4A16), **GGUF Q5_K_M** |
| GGUF reference | `dory-q5_k_m.gguf`, **3,673,288,864 bytes**, sha256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0` |
| sampling | temperature 1.0, top_p 0.95, top_k 0, min_p 0, repeat penalty 1 |
| chat | `<|im_start|>`/`<|im_end|>`, ` thinking` blocks (reasoning always on), XML `<tool_call>`/`<tool_response>` |
| reference runtimes | vLLM PR 60149 and llama.cpp PR 30306 (both unreleased as of this date) |

### Execution model

`k` is the loop count. `N` is a per-token L2 normalization; `x` is the input block's output, which is
re-injected before every loop and never updated:

```
x = input_block(embed(tokens))        # 16 physical layers, one pass
s = x
for t in 1..k:
    u = N(s + x)                      # loop-boundary add + normalize
    s = N(recurrent_block(u))         # 40 physical layers, weights shared across loops
out = output_block(s)                 # 16 physical layers, one pass
logits = lm_head(out)                 # untied
```

Effective forward depth is `32 + 40k`: **72, 112, 152 and 192** physical-layer applications for
k = 1..4. Two loops are the recommended default; the second loop carries most of the accuracy gain.

Two properties matter for the engine design:

1. **Loop count is deterministic and part of the model definition.** k=1 and k=2 are different
   functions with the same weights, not an approximation of each other. Exactness is always relative
   to the reference at the *same* k.
2. **The layers are heterogeneous.** Even indices are attention (three sliding, then one full, per
   eight-layer group); odd indices are MLP. A loop over "decoder layers" that assumes attention +
   MLP as a pair does not match this stack.

### KV cache modes

- **Per-loop** (default, exact): every recurrent attention layer has one cache per loop, so KV grows
  with k. This is the reference-equivalent path.
- **Last-loop** (opt-in, lossy): one cache per recurrent attention layer, overwritten each loop. KV
  footprint becomes independent of k. NVIDIA measures a −2.12 to +0.42 point change at k=4 versus
  per-loop, with nearly unchanged output lengths. It therefore cannot be token-exact and must be a
  named, tested mode behind a flag, in the same spirit as `k8v4`.

## What "supported" means

The claim uses the same contract as [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here):

1. The installer offers the setup only after the measured gates below pass.
2. The model loads from a checksummed Kindi pack, reports the fitted context and answers chat.
3. Greedy output is token-exact against a **pinned reference command** at the same loop count, for
   k=1 and k=2, on the same quantized file.
4. All four loop counts run; k=3 and k=4 are exact against the reference, and a repeat run is
   deterministic.
5. Per-loop KV is the exact default; last-loop KV is opt-in, documented as lossy, and its measured
   delta is recorded.
6. Long-context retrieval passes at 32K, 128K and 262,144 tokens for the configurations that fit,
   with the allocated context and peak memory recorded; no silent truncation or context shift.
7. A real server run covers non-stream, stream, cancellation and prompt-cache restart.
8. The measured result is recorded in the project docs.

Until then, Dory is a roadmap item. It must not appear in the installer, the catalog or the generated
support matrix as supported.

## Why the current engines cannot run it

| area | current reality | implication for Dory1.0-4B |
|---|---|---|
| `kindi/engine.py` | dense Bonsai: RMSNorm, `h += f(x)` residuals, attention + dense FFN pairs | needs L2-only normalization, per-channel residual blending, single-sublayer layers, gated attention output, gated GQA |
| layer schedule | fixed dense stack, one shape for all models | 72 heterogeneous physical layers with a 3:1 sliding/full pattern |
| `kindi/cache.py` | per-sequence K/V, dense attention | sliding-window eviction and the per-loop/last-loop cache mode |
| `kindi/spec.py`, `kindi/plan` | fixed decode graphs, speculative drafting | the graph shape depends on k; self-speculation (k=1 draft, k=4 verify) is a later option |
| `kindi/tp.py` | two-card dense split | no expert or loop dimension; a single rank is enough for a 4 B model |
| `kindi/convert.py` | Bonsai packs, Gemma 4 and Nanbeige GGUF | needs a Dory tensor map and the `dory.*` GGUF metadata |
| `kindi/catalog.py` | dense and MoE entries without Dory | no id, URL, checksum or size until the pack exists |
| server / installer | dense profiles, fixed context | loop count is a request/config value; no offer yet |

The design must add a **new model backend**. A flag on the Bonsai engine would only hide the nGPT
math and the loop schedule.

## Memory envelope

All numbers are planning envelopes computed from the released config and parameter count. They are
not pack measurements; block scales and metadata add overhead, and the final numbers must come from
the released files.

### Weights

| format | bytes per parameter | ~4 B weights | fit on 8 GB |
|---|---:|---:|---|
| BF16 | 2 | **7.95 GB** (7.4 GiB) | no, before KV |
| MXFP8 | ~1.03 | **~4.1 GB** | tight with KV |
| NVFP4 | ~0.56 | **~2.24 GB** | yes |
| GGUF Q5_K_M | measured | **3.67 GB** (3.42 GiB) | yes |

### K/V

One token in one full-attention layer costs `2 × 4 heads × 256 dims × 2 B = 4 KiB`. Sliding layers
hold at most 1,024 positions, so their cost is fixed at `27 × 1,024 × 4 KiB = 108 MiB` per cache set.

| mode | full-attention layers | KV per token | 8K | 32K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|
| k=1 (per-loop = last-loop) | 9 | 36 KiB | 288 MiB | 1.125 GiB | 4.5 GiB | **9.0 GiB** |
| k=2, per-loop | 14 | 56 KiB | 448 MiB | 1.75 GiB | 7.0 GiB | **14.0 GiB** |
| k=4, per-loop | 24 | 96 KiB | 768 MiB | 3.0 GiB | 12.0 GiB | **24.0 GiB** |
| last-loop, any k | 9 | 36 KiB | 288 MiB | 1.125 GiB | 4.5 GiB | **9.0 GiB** |

The one-loop number already exceeds an 8 GB card at 256K even with NVFP4 weights: the full 262,144
context needs a 16 GB card for last-loop, or a larger card for per-loop k=2. The 8 GB target tops out
around **32K comfortable / 64K maximum** on Q5_K_M.

| card | first realistic configuration | what it cannot do |
|---|---|---|
| 8 GB | Q5_K_M, k=2, last-loop, 32K (≈ 3.4 GiB weights + 1.1 GiB KV) | 256K; per-loop k=2 beyond ~48K |
| 16 GB | Q5_K_M or NVFP4, 256K **last-loop** (≤ 9.0 GiB KV) | 256K per-loop k=2 |
| 24 GB | Q5_K_M, 256K **per-loop k=2** (14.0 GiB KV) | 256K per-loop k=4 |
| 32 GB+ | BF16, 256K per-loop k=2; k=4 256K only with last-loop or more memory | — |

## Phased plan

Each phase produces evidence for the next; no phase is skipped because a later one looks easier.

### D0 — architecture and reference audit

- download the BF16 config, tokenizer and the Q5_K_M GGUF; record sizes and checksums;
- enumerate every tensor and the `dory.*` GGUF metadata: block partition, loop count, cache mode,
  tensor naming and quant types;
- pin the reference: llama.cpp PR 30306 revision `1c3cdd23787750814aa21a070dc4d6dc451af871` for the
  CPU audit (the earlier `98a74c97ea09ac72f6ceda1559808fc0a2f447ce` pin predates the Dory CPU-placement
  fixes); vLLM PR 60149 for NVFP4, with the exact greedy command, chat template and sampling;
- write the numeric performance gate from measured layer and expert-free size, before optimizing.

**Gate:** a tensor manifest, a reproducible golden command and a recorded greedy baseline at k=1, 2
and 4. No model code is merged before this exists.

**D0 result (2026-11-11):** reference audit complete; the numeric performance gate remains
provisional. The Q5_K_M file is 3,673,288,864 B with SHA-256
`96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`; the BF16 metadata and its
checksums are under `bench-logs/dory-d0/bf16/`. The 507-tensor manifest and 44-key metadata are in
`bench-logs/dory-d0/tensor-inventory.json` (252 Q5_K, 36 Q6_K, 219 BF16). The GGUF declares
`dory.num_loops=4` while the BF16 config declares `n_recurrent_loops: 2`; every golden run therefore
passes an explicit `dory.num_loops` override. The tested llama.cpp PR 30306 tip is `1c3cdd…` (the
older `98a74…` pin predates CPU-placement fixes). A four-prompt, 32-token greedy suite at k=1/2/4 is
in `bench-logs/dory-d0/goldens/goldens.json`; all 12 runs exited 0 and the k=2 repeat matched
token-for-token. This is reference evidence, not Kindi support; D1 is still pending.

### D1 — GGUF import and CPU reference

- extend `kindi/convert.py` with a Dory tensor map and pack manifest (no new quant format yet);
- implement a straightforward CPU reference for the nGPT sublayer, the layer schedule, the loop
  boundaries and greedy decode;
- verify against the pinned llama.cpp golden at k=1 on short prompts
  (`bench-logs/dory-d0/goldens/goldens.json`).

**Gate:** 4/4 greedy prompts token-exact at k=1, 1K-2K context; peak RSS within the BF16 or Q5_K_M
envelope; no runtime disk reads.

### D2 — GPU nGPT core

- add kernels for L2 normalization (eps 1e-6), the per-channel residual blend, the gated attention
  output and the pre-projection normalizations;
- implement the 3:1 sliding/full schedule with a position-indexed sliding cache and full-attention
  KV plane;
- implement GQA 16/4 with `head_dim` 256 and per-layer RoPE bases;
- run k=1 end to end on the GPU.

**Gate:** output identical to D1 at k=1; `/health` reports VRAM, host RSS and the fitted context.

### D3 — loop counts and KV modes

- make k a build/request parameter for 1..4, with the graph built per k and cached;
- implement per-loop KV (exact default) and last-loop KV (opt-in), including sliding-window handling;
- compare each k against the pinned reference; measure the last-loop delta on the project's prompts.

**Gate:** k=1, 2, 3 and 4 are 4/4 greedy-exact versus the reference; last-loop is measured, named as
lossy, off by default, and never mixed with an exactness claim.

### D4 — long context

- extend the full-attention KV pool and the sliding cache to 32K, 128K and 262,144 tokens;
- add retrieval tests at the beginning, middle and end of the context;
- record the context that fits per configuration and fail loudly above it.

**Gate:** 32K and 128K retrieval pass; 256K passes on at least the 16 GB last-loop configuration;
memory and latency are recorded; no silent truncation or context shift.

### D5 — server and chat contract

- wire the engine into `kindi serve`: non-stream, stream, cancellation and prompt cache;
- implement the Dory chat template, ` thinking` blocks and the tool-call format, with a documented
  reasoning-parser behaviour;
- expose loop count as a config key and a request field, with a per-model default of 2;
- add model/loop metrics to `/health` and `/metrics`.

**Gate:** a real server cold/warm/restart cycle, stream `[DONE]`, prompt-cache resume, cancellation,
no 5xx, and identical greedy output between CLI and server at the same k.

### D6 — quantization

- ship Q5_K_M first (it is the only official GGUF);
- add a BF16 converter for the reference path;
- add an NVFP4 dequantizing import path (4-bit data + FP8 block scales); sm_89 has no native FP4, so
  the first implementation dequantizes to BF16 for compute and is measured as such;
- MXFP8 is optional and later.

**Gate:** Q5_K_M and BF16 are token-exact against their own reference; NVFP4 is compared against
BF16 with the divergence measured and documented, not called exact.

### D7 — performance and support claim

- benchmark on the 8 GB target and at least one larger card, at k=1, 2 and 4;
- compare against the pinned llama.cpp CUDA build on the same GPU, file and prompt;
- publish loop-count scaling, KV footprint and the chosen cache mode next to tokens/s;
- add catalog, installer profile and support-matrix rows only after all gates pass.

**Gate:** the numeric gate below passes, the acceptance matrix is green and the measured result is in
the docs. Until then the model remains "planned".

## Provisional performance gate

Fixed properly after D0. The provisional floor on the 8 GB RTX 4060 Ti with Q5_K_M at a 4K prompt:

- **k=1: ≥ 20 tok/s** decode; **k=2: ≥ 12 tok/s**; **k=4: ≥ 8 tok/s**;
- **loop scaling:** k=2 ≥ 0.55x k=1 and k=4 ≥ 0.30x k=1 (ideal compute ratios are 0.64x and 0.38x);
- **≥ 0.75x** the pinned llama.cpp CUDA build at k=2 on the same file and prompt;
- read ≥ 200 tok/s at 4K with the prompt processed once;
- greedy repeat is deterministic; no OOM, no silent fallback, no output change between runs.

A lower number is allowed only by changing the recorded gate before the measurement.

## Acceptance matrix

All cells are pending. A checked cell requires a measured result and a link to its evidence; a
"builds" result is not support.

| setup | loops | 1K exact | 8K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU reference, BF16 | 1 | pending | pending | pending | — | — | — |
| 8 GB, Q5_K_M, last-loop | 2 | pending | pending | pending | pending | no fit | no fit |
| 16 GB, Q5_K_M, last-loop | 2 | pending | pending | pending | pending | pending | pending |
| 24 GB, Q5_K_M, per-loop | 2 | pending | pending | pending | pending | pending | pending |
| 32 GB, BF16, per-loop | 4 | pending | pending | pending | pending | pending | pending |

## Risks and stop conditions

| risk | consequence | mitigation / stop condition |
|---|---|---|
| nGPT residual approximated as `h += f(x)` | output diverges immediately; no verify passes | implement the released blend and L2 rules; unit-test each sublayer against the reference |
| normalization folded into RMSNorm | silent quality and token drift | keep L2 (`justnorm`) separate; exactness tests per layer |
| loop count treated as a flag on a fixed graph | wrong depth or stale graph cache | build and cache a graph per k; test all four counts |
| last-loop cache claimed exact | unverifiable support claim | keep it opt-in and lossy, with the measured delta recorded |
| sliding-window off-by-one | long-context answers change subtly | pin the window boundary with a dedicated test against the reference |
| 256K on an 8/16 GB card | OOM or silent truncation | publish the per-configuration fit; fail loudly above it |
| NVFP4 needs dequant on Ada | slow path or wrong scales | dequantize to BF16, measure, and do not call it native FP4 |
| reference is an unreleased PR | golden command can disappear | pin a revision and keep its outputs and checksums in `bench-logs/` |
| chat template and reasoning blocks | answers differ from the reference harness | use the released template verbatim; test thinking and tool-call paths |
| untied 131K head | extra VRAM and logits cost | budget the LM head separately; do not tie embeddings |

## Open questions

1. What are the exact nGPT residual formula and per-channel parameter names, and is the update a
   blend or a spherical interpolation (slerp) in the released code?
2. Where exactly is the Q/K normalization applied relative to RoPE, and what form does it take?
3. What does `attention_output_gate_logit_scale_init: null` mean for the released checkpoint?
4. What is the exact tokenizer and chat template, and how are reasoning blocks delimited?
5. **Answered by D0:** 507 tensors (252 Q5_K, 36 Q6_K, 219 BF16) and 44 metadata keys; see
   `bench-logs/dory-d0/tensor-inventory.json`.
6. What is the NVFP4 block-scale layout in the NVIDIA export, and what reference reproduces it?
7. **Partly answered by D0:** k=1, 2 and 4 run with `per_loop` on CPU and k=2 repeated
   token-for-token; k=3, `last_loop`, GPU and a second rebuild remain untested.
8. Which configuration does NVIDIA validate at 256K, per-loop or last-loop?

No answer here should be inferred from the Bonsai engine. D0 resolves 5 and part of 7; the rest
remain open.

## Related design records

- [HOW-IT-WORKS.md](HOW-IT-WORKS.md) — the current dense Bonsai execution path.
- [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here) — the support contract.
- [RAON-1-MINI.md](RAON-1-MINI.md) — the other long-term model goal; an MoE cache problem, not a
  looped-depth problem.
- [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) — the dense CPU/GPU layer split.
- [ROADMAP.md](ROADMAP.md) — the project index and archived no-go decisions.
- [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) — generated capability table; Dory is absent until it has a
  catalog entry and a verified status.
