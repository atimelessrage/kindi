# Bonsai 2 family: support matrix, optimization plan, and projection

Where this sits: [CARDS.md](CARDS.md) and [SUPPORT-MATRIX.md](SUPPORT-MATRIX.md) are authoritative
for what a card can run today; this page is research and projection for the family, and
[README.md](README.md) is the index.

Status: research, planning and the shipped Bonsai 2.5 facts, 2026-11-09; updated 2026-11-25 (section 1a
carries the file-level facts, the code that now accepts it, the template parity check, the CPU-engine
run and the `kindi verify --reference` route; section 4.2 the publisher sampling sets and
`reasoning_effort`). Sections 1a and 4.2 describe what
the tree does today; the rest is projection. This document does not change engine code and is not a
support claim for anything marked **No** or **Untested** below.

Scope: PrismML's Ternary Bonsai 2 family (2B, 4B, 9B, 27B) and Bonsai Image 4B. Bonsai Image is a
separate diffusion model, not another size of the language model.

There is no 7B size in this family, in any release: the 2 family is 2B/4B/9B/27B and the 2026-11-24 2.5
release is a 27B-only upgrade, so a "Bonsai 2.5 7B" plan item is a publisher question, not a Kindi gap
(two independent 2026-11-25 reviews each spent time on it; recorded here so it is not re-opened).

Sources:

- [Bonsai 2 family announcement](https://prismml.com/news/bonsai-2-family)
- [Bonsai 2.5 27B announcement](https://prismml.com/news/bonsai-2-5-27b) (2026-11-24)
- [Ternary-Bonsai-2.5-27B-GGUF model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)
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
   mmproj loader, no vision tower, and no image-token insertion. The request boundary now rejects an
   OpenAI `image_url` part or Ollama `images` with HTTP 400 when the selected pack has no vision
   weights, instead of silently dropping the image; text-only requests are unchanged.
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

## 1a. Bonsai 2.5 27B (released 2026-11-24)

Prism ML shipped Ternary Bonsai 2.5 27B as an upgrade of the 27B rather than a new family: the same
Qwen3.8-27B base and ternary hybrid-attention geometry, the same footprint, retrained for agentic
coding, multi-turn tool use and multi-step task execution, with `-mtp` speculative variants and
`xhigh`/`medium` reasoning efforts. The card's own benchmarks are the publisher's; nothing below
repeats them as a Kindi measurement.

The published GGUF header was read on this host without downloading the weights (a 12 MB HTTP range
request over the 6 GB file; evidence and parser in
`docs/benchmarks/2026-11-24-bonsai25-27b-header/`), and it says the 27B engine path applies unchanged:

| header field | value | consequence |
|---|---|---|
| `general.architecture` | `qwen35` | the existing Bonsai qwen35 branch in `kindi/convert.py` |
| blocks / width / ffn / heads / kv heads / groups | 64 / 5120 / 17408 / 24 / 4 / 16 | the same tuple as the 27B profile (`convert_profile_match` is true) |
| `general.file_type` | 143 (`PTQ1_0`) | the primary pack format, no new tensor type |
| tensors | 851: 402 of type 143 and 449 of type 0 (`f32`), all `blk.N.*` plus `output`, `output_norm`, `token_embd` | no unknown or extra tensors to map |
| `prism.hadamard.block_size` | 1024 | stored in the pack manifest; the engine already reads it from there |
| vocabulary / context | 248320 / 262144 | the 27B tokenizer and context, unchanged |
| chat template | 8952 characters, contains `reasoning_effort` with `xhigh` and `medium` | `chat.template_variant()` selects `bonsai27`, and the `reasoning_effort` plumbing already exists |

The geometry that decides the per-layer work is in the same header. The 2026-11-24 range read
recorded `full_attention_interval`, `rope.dimension_count`, `rope.freq_base`, the head counts and
`key_length`; the rest of this table, and the per-layer tensor lists below, were read on 2026-11-25
from both GGUF files themselves (header parse: 851 tensors in the base file, 866 in the `-mtp` file).

| header field | value | consequence |
|---|---|---|
| `qwen35.full_attention_interval` | 4 | 16 full-attention blocks (3, 7, 11, ... 63) and 48 Gated-DeltaNet blocks, counted by tensor presence: 16 blocks carry the 11-tensor attention set, 48 carry `ssm_*` and 14 tensors |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` | mRoPE sections; for text the three streams coincide, so the engine's partial RoPE over the first 64 dims is the same rotation |
| `qwen35.rope.freq_base` / `qwen35.attention.layer_norm_rms_epsilon` | `1e7` / `1e-6` | the engine defaults; no per-model override needed |
| `qwen35.attention.key_length` / `qwen35.rope.dimension_count` | 256 / 64 | `HD = 256`, `RD = 64`; 24 Q heads and 4 KV heads (16 groups) |

The per-layer tensor list is what a loader keys on, and it was read rather than assumed: a
full-attention block has 11 tensors (`attn_q.weight`, `attn_k.weight`, `attn_v.weight`,
`attn_q_norm.weight`, `attn_k_norm.weight`, `attn_output.weight`, `attn_norm.weight`,
`post_attention_norm.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`) and a
Gated-DeltaNet block has 14 (`attn_qkv.weight`, `attn_gate.weight`, `ssm_a`, `ssm_alpha.weight`,
`ssm_beta.weight`, `ssm_conv1d.weight`, `ssm_dt.bias`, `ssm_norm.weight`, `ssm_out.weight`,
`attn_norm.weight`, `post_attention_norm.weight`, `ffn_gate.weight`, `ffn_up.weight`,
`ffn_down.weight`). The `-mtp` file adds one more block, `blk.64`, with 15 tensors: a full-attention
block plus the four `nextn` tensors. It is **not** a second copy of the model: it borrows
`token_embd.weight` and `output.weight`, and its eight matrices are Q4_0 (about 239 MB), not PTQ1_0.
`docs/MTP.md` carries the head's table; the `-mtp` file is 6,232,818,944 B against the base file's
5,993,834,848 B.

What Kindi now does with it:

- `kindi use bonsai-2.5-27b` downloads `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` (5993834848 bytes, SHA-256
  pinned in `kindi/catalog.py`) from `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`, builds a pack and selects
  it. The catalog's fit estimate is the 27B's: one 7600 MiB card, or two at 3800 MiB. The `-mtp` file
  has its own id, `bonsai-2.5-27b-mtp`, and the same fit estimate (see the next bullet). **Download the
  base file unless you are keeping the head for future drafting:** today both files build the same
  64-layer pack and read the same weight stream, so the larger file buys no speed and does not draft.
- The registry id is `bonsai2.5-27b`. A 2.5 pack resolves through the Bonsai adapter's
  `model_id_for_manifest` (from `general.name`, or from `source_file` for a file that names itself
  `Hf`), and the release is part of the id so the two releases cannot share a sampling profile.
- On a host with no card the CLI now points at the engine that actually runs here: `kindi models`
  lists `bonsai-2.5-27b` and `bonsai-2.5-27b-mtp` as `CPU engine` instead of `GPU only` (the CPU
  engine is what loaded both files), and `kindi plan` prints `KINDI_DEVICE=cpu` follow-up commands
  instead of a `--tandem` split, which exits 1 without a CUDA/ROCm device (2026-11-26).
- `sampling_profile = publisher` now has a 2.5 entry: thinking temperature 1.0 / top-p 0.95 / top-k 20 /
  presence 0.0 (Bonsai 2's card says 1.0 there), non-thinking 0.7 / 0.80 / 20 / 1.5. Everything else in
  the profile is unchanged, and a request's own values still win.
- The `bonsai27` template, its `xhigh` default effort and the `medium` fallback apply as-is;
  `enable_thinking` is still off unless the request asks for it.

- The prompt renderer was compared with the published template, not just against a memory of it:
  jinja2 renders the template string from the 2.5 GGUF header and from a converted 2B pack, and
  `kindi.chat.render()` renders the same conversations - **26 cases**, **26 of 26** byte-identical, 0
  mismatches (`docs/benchmarks/2026-11-24-bonsai-chat-template-parity/`). The comparison found a real
  divergence: tool schemas and non-string tool arguments were serialized in insertion order while the
  template uses `tojson`, so every tool-calling prompt Kindi built had its keys in another order and
  left `<`/`&`/`'` unescaped. `kindi/chat.py` now writes both through one `_tojson()` helper, and
  `tests/test_chat_template.py` pins the result against goldens rendered by jinja2. The comparison is
  prompt text only: no 2.5 weights were run to show the corrected prompt changes an answer.

What was run (2026-11-25, i7-8086K, CPU engine; evidence in
`docs/benchmarks/2026-11-25-i7-8086k-bonsai25-27b-cpu/`):

- The 6 GB GGUF was downloaded and its SHA-256 matched the catalog, then converted to a **7,242 MB in 370 s**
  pack: `packcheck` full passed on 10 files, and the manifest reports `851 tensors`, vocabulary
  `248320`, context `262144`, hadamard block 1024 and the same 8952-character template the header run saw.
- `KINDI_DEVICE=cpu` loads that pack in **339.9 s** (avx2-fma-f16c, 6 threads, ctx 2048; the pack
  decoder was 96% of that and got 5.5x faster on 2026-11-26 - measured on the 2B/4B/9B packs, a
  projection for this one, `docs/benchmarks/2026-11-26-i7-8086k-cpu-pack-load/`) and generates
  greedily at **4.02 tok/s** (best request) and **3.08 tok/s** (worst), over **2 prompts** of
  **16 tokens**; the first reply is on topic, which is what shows the pack is loaded coherently. The
  two processes produced the same ids, and a repeated generation in one process is identical too, so
  CPU decode is reproducible.
- `kindi verify` on the pack prints `no llama.cpp reference for this model in this version of Kindi`
  and exits 2, because references are keyed by release and there is none for 2.5 here. The refusal now
  names the way in: `kindi verify --reference FILE` runs the same matrix against a reference made with
  llama.cpp for this exact GGUF. The file is what the bundled ones are - a JSON `{"prompts": [{"ids":
  [...], "expected": [...]}, ...]}` with optional `source` and `prompt` text - and Kindi checks that
  shape before it loads the engine, sizes the context from the longest row, and prints `reference FILE:
  user-supplied` so a PASS cannot be mistaken for the bundled matrix's. It cannot invent the numbers:
  the expected tokens still have to come from llama.cpp's greedy run on the same file.
- The `-mtp` file was run the same way, from its own conversion: the pack manifest names the same
  64-layer model with the head's block and tensor names under `mtp_head`, and the CPU engine loaded
  and generated from it with the same kinds of replies as the base file (the first answer on topic,
  the arithmetic prompt answering the number) and identical ids on an in-process repeat. This is a
  load-and-generate check, not a second process run and not a drafting result: nothing drafted.
- `kindi verify --reference` was run on the 2.5 pack with a two-prompt reference the pack itself
  produced (11 and 1,057 prompt tokens, 8 expected each; `bench-logs/make_self_reference.py`). The file
  was accepted, the engine came up with `ctx 1073` from the 1,057-token row rather than the hard-coded
  1024, both prompts matched 8 of 8 and the run passed (pack load 340 s, 3.3 tok/s over the run). The
  expectations are the pack's own greedy output, so this checks the route and the matrix on the real
  model, not exactness: only a llama.cpp run on this GGUF can settle that, and `--reference` is how its
  tokens would be scored.

That is the new, deliberate behaviour: `verify.reference_for` keys the reference file by
  release as well as by size, so a 2.5 pack is no longer scored against Bonsai 2's tokens
  (`reference-27b.json`) and reported as a port failure.

What is **not** verified and must not be claimed:

- No exactness result: there is no llama.cpp reference row for this release here, so nothing above is a
  token-for-token comparison with another engine. The route now exists (`--reference`, above) and was
  exercised on the pack with a reference the pack itself produced; no llama.cpp build for this GGUF is
  on this host, so the flag is checked and the exactness comparison is not. The support matrix row `bonsai-2.5-27b` reads
  `load+generate, no reference` for CPU, `not run` for GPU, and `not converted` for PQ2_0/Q2_0.
- No GPU run of any kind: the Arc A770 toolchain is still probe-only (`kindi intel-sycl`) and this host
  has no CUDA device, so the GPU cells stay `not run`. Sampling beyond greedy, tool calling, long
  context (>2048 tokens) and a 2-vs-2.5 quality comparison are all unmeasured.
- The `-mtp` files are converted, but their head is not used to draft. Prism ships the head as a
  65th block (`blk.64`): one decoder layer plus its `nextn.*` tensors, after the 64-block language
  model. `kindi convert` recognises that shape (a profile one block past a known model whose extra
  block carries `nextn.eh_proj`), builds the base model's pack, and records the block and its tensor
  names in `kindi_pack.json` under `mtp_head`, so the pack is the same model the base file produces.
  `KINDI_SPEC=mtp` still drafts only with the 2B's registered `.npz` head: Kindi's `MTPDraft` is
  written for that head's geometry on the tensor-core plan engine, and the 27B's embedded head is not
  wired to it. The catalog therefore lists the file (so `kindi use`/`kindi models` handle it) without
  implying a speedup.
- The `mmproj` vision artifacts are unsupported as before (section 4.1): the 2.5 ships the same separate
  BF16/Q8_0 mmproj files, and an image request against a text pack is answered with 400.
- The 2.5 `PQ2_0` and `Q2_0` files use the same container types the converter handles, but none was
  downloaded here, so the byte-identical-pack claim stays a 2B result (section 2).

## 2. Formats: what to support and in what order

| Artifact | What it is | Kindi today | Recommendation |
|---|---|---|---|
| `PTQ1_0` | Dense trits, ~1.75-1.83 bpw; 5 trits/byte | **Supported** by the converter and runtime | Keep as the primary format |
| `PQ2_0` | 2-bit slots, ~2.13-2.20 bpw; cheaper prompt processing | **Supported** by `kindi/convert.py` (type 142) | Convert to Kindi's pack; the result is byte-identical to the PTQ1_0 pack |
| `Q2_0` | Upstream llama.cpp 2-bit type, ~2.28-2.33 bpw | **Supported** by `kindi/convert.py` (type 42) | Import for users who already downloaded it; the result is byte-identical to the PTQ1_0 pack |
| mmproj | Q8_0 or BF16 vision tower + projector | **No loader** | P1: required for image input |
| MTP | Publisher MTP head/checkpoint | **No 4B/9B support**; own 2B head only | P2/P3: measure before integrating |
| DFlash2 | Block-diffusion drafter | **No support** | P3: copy/edit workloads only |
| MLX 2-bit | Apple MLX conversion | **No support** | Out of scope for this CUDA/CuPy engine |

The publisher's own guidance is that `PTQ1_0` is the smallest and the faster decode format on Ada/L4,
`PQ2_0` reduces prompt-processing cost, and `Q2_0` exists for stock llama.cpp. Kindi is not stock
llama.cpp, so there is no compatibility reason to make `Q2_0` the default. The practical reason to
add either format is a user who already has that file and does not want to download PTQ1_0 again.

Since 2026-11-10 the 2B `PQ2_0` and `Q2_0` files convert to packs that are byte-identical to the
PTQ1_0 pack for every stream (`bcodes.bin`, `dpm.bin`, `f32.bin`, both sign files, the tokenizer and
both indexes); only `index.json`'s recorded source type and `kindi_pack.json`'s `source_format`
differ. Because the pack is identical, the memory footprint and every benchmark number are the
PTQ1_0 numbers: converting `PQ2_0`/`Q2_0` buys download compatibility, not speed. 2026-11-15:
the 4B pack on this host is also reproducible byte-for-byte from its downloaded `PTQ1_0` GGUF
(weights identical, converted pack verifies 3+tie exactly like the reference; `bench-logs/convert-4b-verify/RESULTS.md`).
The 4B/9B/27B `PQ2_0`/`Q2_0` **files** themselves remain undownloaded here (Hugging Face unreachable
from this host), so the type-142/42 import claim stays verified for the 2B only.

## 3. Current Kindi support

Verified in the code and docs on this branch:

| Capability | 2B | 4B | 9B | 27B | Evidence / caveat |
|---|---|---|---|---|---|
| Text, PTQ1_0 | Yes | Yes | Yes | Yes | Four profiles in `kindi/convert.py`; references and verifies exist |
| Text, PQ2_0/Q2_0 | Yes | Not tested | Not tested | Not tested | Types 142/42/143 in `kindi/convert.py`; the 2B pack is byte-identical to the PTQ1_0 pack and `kindi verify` passes 4/4 (`bench-logs/pq2-q2-2b/RESULTS.md`). The 4B PTQ1_0 conversion is provenance-verified (reconvert → byte-identical weights, verify 3+tie meeting the reference pack; `bench-logs/convert-4b-verify/RESULTS.md`), but the 4B PQ2_0/Q2_0 **files** were never downloaded here (HF unreachable from this host), so the 142/42 claim for 4B stays untested |
| Image input | No | No | No | No | No mmproj loader; `chat.content_of()` takes only `text` parts |
| Image part rejected without vision weights | Yes (400) | Yes (400) | Yes (400) | Yes (400) | `kindi/server.py` checks `image_url`/`image`/`input_image` parts and Ollama `images`; a pack with `vision` or `mmproj` in `kindi_pack.json` is allowed through |
| 262,144 context | Untested | Untested | Untested | Yes, 2x P100 | `docs/BENCHMARKS.md` full-context 27B run; `docs/CARDS.md` shows 110,592 tokens on a 10 GB RTX 3080 |
| Tools / thinking | Yes | Yes | Yes | Yes | `kindi/chat.py`; 27B has reasoning-effort template |
| Publisher sampling | Partial | Partial | Partial | Partial | Temperature/top-p/top-k exist; no presence penalty; thinking off by default |
| CPU text | Yes | Yes | Yes | No | `docs/CPU-SUPPORT.md`; i7-8086K engine verified 2026-11-09 (2B/4B/9B 4/4); W-2133/i9 unmeasured |
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

The current request boundary now rejects image input when the selected pack has no vision weights:
`_validate_request()` returns 400 for an `image_url`/`image`/`input_image` content part and for
Ollama `images` when `kindi_pack.json` has neither `vision` nor `mmproj`. `chat.content_of()` still
concatenates only parts with a `text` key, so a future vision pack must add the actual image path;
the text-only behavior is unchanged.

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
7. **[Done for the boundary]** Return **400** with "this pack has no vision weights" when an image is
   supplied without an mmproj. The server now checks OpenAI image parts and Ollama `images` before
   the engine; the actual vision path remains open.

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

Bonsai 2.5 27B changes one value: its card recommends presence 0.0 in thinking mode, where Bonsai 2
says 1.0, and `kindi/profiles.py` keys that on the 2.5 model id (`bonsai2.5-27b`). The 2.5 card also
states the off values (`min_p` 0.0, `repetition_penalty` 1.0) and its reasoning-effort rules, so the
profile and Kindi's template handle all three:

| mode | temperature | top_p | top_k | min_p | presence | repetition |
|---|---:|---:|---:|---:|---:|---:|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

- **Reasoning effort:** `xhigh` is the default and the most accurate. `medium` is for shorter
  responses and a balance of speed and accuracy. `low` is not supported by the model: when it is
  selected the model behaves close to `xhigh`, so Kindi renders the `xhigh` instruction for it rather
  than the template's own `low` text (which the model was not trained on). `high` is OpenAI's name for
  that same behavior and is accepted as a second alias of the one measured instruction, so a client's
  legal `high` is not a 400 while `low` is; any other name is a 400 that names `param:
  reasoning_effort`. `reasoning_effort` is a config key and a request field; it applies only when
  thinking is on, and the response does not echo the effective value (one line at `kindi serve`
  startup does).
- **Which request gets which set:** thinking requests (`enable_thinking`) take the first row,
  instruct requests the second. `sampling_profile = publisher` makes the profile active; a request's
  own values, or values the user set in the config, still win.
- **Repeatable benchmark:** `tools/bonsai_sampling_bench.py` runs both rows on a pack (and thinking
  at `xhigh`, `medium` and `low`), on the CPU engine or a card, and writes the ids, text, settings and
  timings to JSON. The 2026-11-25 CPU run on both 2.5 27B packs is
  [the sampling benchmark page](benchmarks/2026-11-25-i7-8086k-bonsai25-27b-sampling/README.md); that
  computer was busy with other work while it ran, so the page is a settings-and-engine check and its
  timings are not a performance measurement.

**Implemented (2026-11-10).** `presence_penalty` is a request field (OpenAI semantics, -2..2) and a
config key; `kindi/profiles.py` carries the publisher profile above, and `sampling_profile = publisher`
makes it active. A request's own values win, and so does a value the user set in the config file, the
environment or a command line: the profile only fills built-in defaults. The thinking profile also
raises the default output to 8192 tokens for that request only. The default `sampling_profile =
default` keeps the old values, so existing users see no change. The Bonsai backends (2B/4B/9B/27B)
apply the presence penalty; Gemma 4 and Nanbeige still do not. Tests: `tests/test_profiles.py`,
`tests/test_sampling.py`, `tests/test_server.py`.

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
| M5. Formats and drafters | PQ2_0/Q2_0 import (**2B done**); 4B/9B MTP; DFlash2 evaluation | 8-14 days | Converted pack is bit-identical to PTQ1_0 reference; MTP/DFlash2 measured, not assumed |
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

1. **This week:** verify the four text packs with the new image-400 boundary (OpenAI image parts and
   Ollama `images`). This closes the silent-failure hole without waiting for the vision tower.
2. **Next:** add the publisher non-thinking/thinking profiles and `presence_penalty`. This is a
   small change with a direct quality effect.
3. **Then:** build the 2B vision MVP. Do not start with 4B/9B or Bonsai Image.
4. **After the 2B vision path is exact on the fixed image set:** extend to 4B/9B/27B, then run the
   context matrix.
5. **Only after that:** evaluate PQ2_0/Q2_0 import, Prism MTP/DFlash2, and a separate Bonsai Image
   backend.
6. **The 2.5 27B CPU half is done (2026-11-25):** the file was downloaded, converted, packchecked and
   run on the CPU engine, and `kindi verify` refuses it for lack of a llama.cpp reference instead of
   scoring it against Bonsai 2. What is still missing is a **GPU** run on a card (the Arc toolchain is
   probe-only) and a reference configuration to make `kindi verify` meaningful. The header and CPU
   checks do not substitute for that. The `-mtp` file is no longer a separate intake problem:
   it converts as its base model with the head recorded (DEC-0012), but drafting with that head
   remains open work and needs a GPU run.

The project should describe the current text path as **text-only PTQ1_0** until the vision milestone
passes. The family is vision-language; the engine is not there yet.
