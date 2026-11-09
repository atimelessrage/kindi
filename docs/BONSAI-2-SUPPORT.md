# Bonsai 2 family: support matrix, optimization plan, and projection

Status: research and planning, 2026-11-09. This document does not change engine code and is not a
support claim for anything marked **No** or **Untested** below.

Scope: PrismML's Ternary Bonsai 2 family (2B, 4B, 9B, 27B) and Bonsai Image 4B. Bonsai Image is a
separate diffusion model, not another size of the language model.

Sources:

- [Bonsai 2 family announcement](https://prismml.com/news/bonsai-2-family)
- [Formats and runtime matrix](https://docs.prismml.com/download/formats)
- [Bonsai 2 introduction](https://docs.prismml.com/get-started/introduction)
- [Bonsai 2 Small collection](https://huggingface.co/collections/prism-ml/bonsai-2-small)
- [Bonsai 2 collection](https://huggingface.co/collections/prism-ml/bonsai-2)
- [Bonsai Image](https://docs.prismml.com/models/bonsai-image)

## TL;DR

1. Kindi's text path already has the right shape for this family: all four PTQ1_0 language-model
   profiles are in `kindi/convert.py`, the ternary/Hadamard kernels are implemented, and the
   2B/4B/9B use the `qwen35` chat template while the 27B has its own template. The family is not a
   from-scratch port.
2. The largest missing capability is **image input**. Bonsai 2 is vision-language, but Kindi has no
   mmproj loader, no vision tower, and no image-token insertion. Worse, an OpenAI request containing
   an `image_url` part is currently accepted and the image is silently ignored by `chat.content_of()`;
   the model answers as if no image was sent.
3. The highest-value work is not a new ternary kernel. It is a **2B vision MVP**: mmproj loader,
   Q8_0/BF16 vision encoder, image preprocessing, image tokens, an OpenAI/Ollama image contract, and
   an explicit 400 when the selected pack has no vision weights.
4. **262K context is already demonstrated** for the 27B on two 16 GB P100s (`docs/BENCHMARKS.md`:
   full 262,144 tokens, 8.6 GB free per card, 67.6 tok/s prompt read, 15.8 tok/s decode). Do not
   present 262K as missing. The work is to document and test per-size memory limits; 8 GB cannot
   hold 27B at 262K.
5. **PQ2_0/Q2_0 import is compatibility work, not a speed feature.** PTQ1_0 is the smallest packing
   and the faster decode format on Ada/L4, and it is the format Kindi's five-trit lane is built for.
   Add the other packings only after vision, unless a user specifically has a Q2_0-only download.
6. Prism's 4B/9B MTP and DFlash2 drafters are a later optimization. The 2B has no publisher drafter
   for a reason, and Kindi's own 2B MTP head is already neutral at 24K (336.1 vs 349.0 tok/s
   on an RTX 3080, 0.96x).
7. Bonsai Image generation is a separate 20-30 engineer-day subsystem. It should not block the text
   or vision work.

## 1. What Bonsai 2 is

The current family is text + image input, text output, with a hybrid attention stack and a 262,144-token
context. The publisher describes 2B/4B/9B as Qwen3.5-based and the 27B as Qwen3.8-based; all four
are shipped as ternary models with FP16 scales.

| Size | Language model | Vision tower | Context | Notes |
|---|---:|---:|---:|---|
| 2B | ~1.88B | ~0.33B | 262,144 | smallest; fast decode target |
| 4B | ~4.21B | ~0.33B | 262,144 | middle tier |
| 9B | ~8.95B | ~0.46B | 262,144 | larger single-card tier |
| 27B | 27.36B total | 0.46B | 262,144 | 24.35B language + 2.54B embedding/head, vision tower included |

Architecture points that matter to Kindi:

- Three linear-attention (GDN) blocks for every full-attention block. Only the full-attention layers
  carry a growing KV cache, which is why 262K is feasible at all on large-memory cards.
- Ternary weights in blocks of 128 with FP16 scales, plus Prism's rotated-basis Hadamard transform
  and sign flips. Loading the ternary values without the matching activation transform produces
  gibberish; Kindi already implements the transform and stores the sign vectors.
- Text and image tokens share one vocabulary and one decoder. The vision tower is a separate mmproj
  artifact, not a second language model.
- Apache 2.0 weights. The repository's AGPL-3.0-only license is a distribution question, not a
  blocker for running the weights locally.

## 2. Formats: what to support and in what order

| Artifact | What it is | Kindi today | Recommendation |
|---|---|---|---|
| `PTQ1_0` | Dense trits, ~1.75-1.83 bpw; 5 trits/byte | **Supported** by the converter and runtime | Keep as the primary format |
| `PQ2_0` | 2-bit slots, ~2.13-2.20 bpw; cheaper prompt processing | **Rejected** by `kindi/convert.py` | P2: import/convert to Kindi's pack; no decode speed claim |
| `Q2_0` | Upstream llama.cpp 2-bit type, ~2.28-2.33 bpw | **Rejected** | P2: import for users who already downloaded it; upstream compatibility only |
| mmproj | Q8_0 or BF16 vision tower + projector | **No loader** | P1: required for image input |
| MTP | Publisher MTP head/checkpoint | **No 4B/9B support**; own 2B head only | P2/P3: measure before integrating |
| DFlash2 | Block-diffusion drafter | **No support** | P3: copy/edit workloads only |
| MLX 2-bit | Apple MLX conversion | **No support** | Out of scope for this CUDA/CuPy engine |

The publisher's own guidance is that `PTQ1_0` is the smallest and the faster decode format on Ada/L4,
`PQ2_0` reduces prompt-processing cost, and `Q2_0` exists for stock llama.cpp. Kindi is not stock
llama.cpp, so there is no compatibility reason to make `Q2_0` the default. The practical reason to
add either format is a user who already has that file and does not want to download PTQ1_0 again.

## 3. Current Kindi support

Verified in the code and docs on this branch:

| Capability | 2B | 4B | 9B | 27B | Evidence / caveat |
|---|---|---|---|---|---|
| Text, PTQ1_0 | Yes | Yes | Yes | Yes | Four profiles in `kindi/convert.py`; references and verifies exist |
| Text, PQ2_0/Q2_0 | No | No | No | No | `convert.py` accepts only types 143, 0, 30 |
| Image input | No | No | No | No | No mmproj loader; `chat.content_of()` takes only `text` parts |
| Image part rejected | No | No | No | No | The server validates the list shape but does not reject unknown part types; the image is dropped |
| 262,144 context | Untested | Untested | Untested | Yes, 2x P100 | `docs/BENCHMARKS.md` full-context 27B run; `docs/CARDS.md` shows 110,592 tokens on a 10 GB RTX 3080 |
| Tools / thinking | Yes | Yes | Yes | Yes | `kindi/chat.py`; 27B has reasoning-effort template |
| Publisher sampling | Partial | Partial | Partial | Partial | Temperature/top-p/top-k exist; no presence penalty; thinking off by default |
| CPU text | Yes | Yes | Yes | No | `docs/CPU-SUPPORT.md`; CPU targets are not all measured |
| Multi-GPU text | No | No | No | Yes (2 cards) | 27B 2x P100 full-context result; no vision TP |
| Publisher MTP | No | No | No | No | Prism 4B/9B MTP is not loadable |
| Kindi MTP / draft | 2B head | No | No | 2B drafter | `docs/MTP.md`; 2B head is neutral at 24K |
| Bonsai Image | No | No | No | No | Separate FLUX.2 Klein 4B diffusion pipeline |

Two distinctions matter:

- **Bonsai 2 image input** means the language model reads an image and writes text. That is the P1
  vision work in this document.
- **Bonsai Image** is a text-to-image diffusion model (FLUX.2 Klein 4B, ternary/binary transformer,
  separate text encoder and VAE, Gemlite CUDA or MLX). It is a different product surface and a
  separate backend.

## 4. Gaps and opportunities

### 4.1 Vision input is the main missing capability

The current request boundary is permissive: `_validate_request()` accepts a content list containing
objects, and `chat.content_of()` concatenates only parts with a `text` key. An OpenAI request with

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

is therefore accepted and the image is discarded. The model answers from the text alone. That is
worse than a clear refusal because it looks like vision worked.

Minimum work:

1. Add an `mmproj` entry to the catalog and a `--vision`/`--mmproj` selection path. Download the
   vision artifact separately; it is not part of the language-model pack.
2. Parse the mmproj GGUF. It carries a ViT (24-27 blocks depending on size), a projector, image
   normalization metadata, and the image-token ids. It is Q8_0 or BF16, not ternary.
3. Add a float/int8 GEMM path for the ViT. Do not try to feed the vision tower through the five-trit
   lane. Q8_0 weights can be dequantized once at load or consumed by an int8 tensor-core path; the
   existing `tc.py` int8 machinery is the closest starting point.
4. Preprocess images on the host: decode, orient, resize to the model's expected grid, normalize,
   patchify. Cache the encoded embedding by a hash of the normalized image bytes.
5. Insert the vision special tokens and image embeddings into the prompt at the position of the
   image part, then run the normal decoder. The language model still produces text only.
6. Add `image_url` support to `/v1/chat/completions` and `images` to Ollama `/api/chat`. Accept data
   URLs first; make remote HTTP fetching opt-in and guarded against SSRF. Cap bytes and pixels.
7. Return **400** with "this pack has no vision weights" when an image is supplied without an mmproj.
   Do this even before the vision path exists, so the current silent drop stops.

Performance notes:

- Encode each image once and cache the result. A repeated image in a conversation should cost zero
  vision work on the second turn.
- Image tokens are prefilling work, not decode work. Feed them through the existing chunked prompt
  path; do not build a separate long-prompt path just for vision.
- The 2B vision tower is small relative to the language model. Measure the first image encode and
  the image-token prefill separately; do not assume either is negligible.
- Keep the image embedding cache bounded; a server that accepts many images must not grow host RAM
  without limit.

### 4.2 Publisher sampling and thinking defaults are a cheap quality win

Kindi's defaults are:

- `ctx=24576`
- `max_tokens=1024`
- `temperature=0.7`, `top_p=0.8`, `top_k=20`
- `repeat_penalty=1.0`
- thinking disabled unless the request asks for it

That is close to the publisher's **non-thinking** profile, but it is missing the publisher's
presence penalty, and the thinking profile is different. The publisher recommends:

- thinking: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.0
- non-thinking: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5
- up to 8K output tokens for thinking; avoid greedy decoding and do not treat repetition as a
  sampler substitute

Kindi has `repeat_penalty`/`repeat_last_n` but no `presence_penalty`. This is a small, testable
server/config change and it affects answer quality immediately. Add a per-model generation profile
instead of changing every user's defaults:

- expose `presence_penalty` in the request schema and sampler;
- add a `thinking` profile selected by the same `enable_thinking` flag;
- keep the existing non-thinking defaults for compatibility, but document the publisher profile and
  make it selectable;
- allow a larger default output budget for thinking requests without making 8K the global default.

### 4.3 Long context is mostly a documentation and memory-profile job

The 27B already reads and writes at the full 262,144-token context on two P100s, and fits 110,592
tokens on a 10 GB RTX 3080. The 2B reaches 65K on a 2 GB Quadro K420 and 24K on the 8 GB 4060 Ti.
The remaining work is:

- a per-size table of context versus KV memory, with f16 and q8_0 columns;
- an explicit "requested / granted" context report (the engine already clamps to free memory);
- long-context retrieval tests for 2B/4B/9B at the largest context that fits;
- a clear statement that 27B at 262K needs roughly 8 GiB of q8_0 KV plus 5.95 GB of weights, so an
  8 GB card cannot do it. Two 16 GB cards can.

Do not rewrite the attention path before measuring. The hybrid 3:1 stack is the reason this works.

### 4.4 Drafters: measure before integrating

Kindi already has:

- a custom 2B MTP head;
- a 2B drafter for the 27B (1.8x on an RTX 3060 in the existing docs);
- n-gram speculative decoding tied to the plan engine.

The 2B MTP head is not a win at 24K (336.1 vs 349.0 tok/s on an RTX 3080, 0.96x). N-gram speculation is highly
workload-dependent: 86% acceptance on a copy task but only 8-18% on fresh prose. Prism's 4B/9B MTP
and DFlash2 are different artifacts and would need their own conversion and acceptance tests.

Order:

1. Keep the 27B 2B-drafter path; it is the proven win.
2. Keep 2B MTP off by default above roughly 8K, or require an explicit flag.
3. Add 4B/9B publisher MTP only if the files are distributed separately and can be converted without
   changing the base pack. Measure acceptance at 512/8K/24K before enabling it.
4. Treat DFlash2 as a copy/edit optimization, not a default chat mode.

### 4.5 Tensor-core plans are the next text performance lever

The 2B plan engine is 302.4 tok/s at 24K versus 170.6 plain (1.77x, exact head, no speculation).
That is the strongest measured text optimization in this worktree. Extending it is more valuable than
another prompt-read micro-optimization.

- 4B and 9B plans are the next targets. The 9B HB=4096 path is currently refused before graph build;
  supporting it needs either a different split or a verified HB=4096 tensor-core graph.
- The 27B has a two-card text path; a plan/tensor-core version needs a two-card plan, not a copy of
  the one-card graph.
- The 2B plain prompt read is CPU-launch-bound (CPU time 95.6-104.5% of wall, GPU wait under
  1.1 ms/token in the diagnostic runs). Do not chase GPU clocks for that number; batch the launch
  path or use the plan engine.

## 5. Support projection

Effort is a rough engineering estimate for one person already familiar with the engine, CUDA/CuPy,
and the existing test harness. It is not a schedule commitment. It assumes no new model training,
no new hardware, and access to the publisher's mmproj and reference outputs.

| Milestone | Scope | Effort | Exit criterion |
|---|---|---:|---|
| M0. Text parity and honest API | Verify 2B/4B/9B/27B text on the current PTQ1_0 files; reject image parts with 400 when no vision pack; document exact supported pack | 2-4 days | `kindi verify` passes for each available pack; image request gets 400, not a text-only answer |
| M1. Publisher text profile | Presence penalty; thinking/non-thinking profile; output-budget guidance; per-model defaults | 2-4 days | Deterministic server tests; sample quality comparison on fixed prompts; no regression for existing clients |
| M2. 2B vision MVP | mmproj loader, ViT, projector, image preprocessing, image tokens, data-URL `image_url`, image cache | 10-15 days | Fixed image set answered with image-dependent questions; same image twice reuses cache; image request without mmproj gets 400 |
| M3. Vision for 4B/9B/27B | Reuse the 2B path; per-size projector/token metadata; memory checks; Ollama `images`; remote URL guard | 5-10 days | Each size passes the fixed image set; 27B two-card vision run; no OOM at documented context |
| M4. Context matrix | Per-size largest-fit tests up to 262K; f16/q8_0 KV table; request/grant reporting | 2-4 days | 27B 262K regression; 2B/4B/9B tested at the largest context that fits; docs state the limit |
| M5. Formats and drafters | PQ2_0/Q2_0 import; 4B/9B MTP; DFlash2 evaluation | 8-14 days | Converted pack is bit-identical to PTQ1_0 reference; MTP/DFlash2 measured, not assumed |
| M6. Plan-engine expansion | 4B/9B HB=4096 plan; two-card 27B plan; launch-path cleanup | 10-20 days | Plan output remains exact; measured speedup; no regression on the 2B |
| M7. Bonsai Image | Separate diffusion backend: text encoder, VAE, scheduler, Gemlite/MLX-style path, CLI/server | 20-30 days | 512x512 image generation from fixed prompts; separate from the LM server; no shared-pack regression |

Suggested order: **M0 -> M1 -> M2 -> M3 -> M4 -> M5/M6 -> M7**. M0 and M1 are low-risk and improve
the current product immediately. M2 is the capability milestone. M7 is a separate product and should
not block anything above it.

## 6. Acceptance tests

Text:

- `python -m kindi verify` for every available 2B/4B/9B/27B PTQ1_0 pack.
- Greedy output remains token-exact against the stored references.
- The 27B 262K full-context run remains a regression target; do not accept a context rewrite that
  loses it.

Vision:

- A fixed set of images with questions whose answers depend on the image (count, color, text/OCR,
  spatial relation). Store the prompt, image hash, and expected answer.
- The same image twice in one conversation must not re-encode; verify with a counter/metric.
- Without an mmproj: HTTP 400 with a clear message. With an mmproj but an invalid image: 400.
- A remote image URL is refused unless explicitly enabled; a data URL over the size cap is 413.
- Ollama `/api/chat` with `images` produces the same answer as the OpenAI path for the same image.
- Text-only requests are byte-identical before and after the vision change.

Context:

- 2B at 65K, 27B at 110,592 and 262,144, and the largest context that fits for 4B/9B.
- A retrieval prompt whose answer is at the beginning, middle, and end of the context.
- Report allocated context, KV bytes, and free memory; fail loudly rather than silently truncating.

Performance:

- 2B plan at 24K stays at or above the 1.77x baseline.
- 4B/9B MTP and DFlash2 are enabled only with a positive measured result at the target context.
- Image encoding time and image-token prefill time are reported separately from text prefill.

## 7. Non-goals and risks

- **MLX/Apple support.** The MLX 2-bit path is a separate runtime; Kindi is CUDA/CuPy with a CPU
  fallback. Point Mac users at the MLX ecosystem unless a dedicated port is funded.
- **Training or fine-tuning.** Kindi is an inference engine.
- **Video/audio input.** The current family is text + image; do not generalize the API to media
  types without a model that consumes them.
- **Upstream llama.cpp compatibility.** `Q2_0` is useful for stock llama.cpp, not a requirement for
  Kindi. Do not make it the default.
- **262K on 8 GB.** The 27B's q8_0 KV cache alone is about 8 GiB at 262K; the weights are about
  5.95 GB. This is physically out of reach on an 8 GB card, regardless of kernel work.
- **Bonsai Image in the language server.** Diffusion has a different scheduler, VAE, and output
  contract. Keep it a separate command/service.
- **Exact token parity for vision.** The vision tower is a new numeric path; compare answers and
  top-1 tokens on fixed images, not raw logits, unless a bit-exact reference is available.
- **Licensing.** The weights are Apache 2.0; the repository is AGPL-3.0-only plus a commercial
  option. Distribution of a combined work needs a license review, not an engine change.

## 8. Recommended next actions

1. **This week:** add the 400 for image parts without a vision pack, and verify the four text packs.
   This closes the silent-failure hole without waiting for the vision tower.
2. **Next:** add the publisher non-thinking/thinking profiles and `presence_penalty`. This is a
   small change with a direct quality effect.
3. **Then:** build the 2B vision MVP. Do not start with 4B/9B or Bonsai Image.
4. **After the 2B vision path is exact on the fixed image set:** extend to 4B/9B/27B, then run the
   context matrix.
5. **Only after that:** evaluate PQ2_0/Q2_0 import, Prism MTP/DFlash2, and a separate Bonsai Image
   backend.

The project should describe the current text path as **text-only PTQ1_0** until the vision milestone
passes. The family is vision-language; the engine is not there yet.
