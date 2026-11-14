# Raon-1 mini R0 evidence

Status: **header/manifest audit complete, 2026-11-11. No full checkpoint was downloaded and no
reference generation was run. This is not a Kindi model support claim.**

The design record is [`docs/RAON-1-MINI-R0.md`](../../docs/RAON-1-MINI-R0.md); the broader goal is
[`docs/RAON-1-MINI.md`](../../docs/RAON-1-MINI.md).

## What was fetched

| item | source | evidence |
|---|---|---|
| config, index, model code, tokenizer config, chat template, license | `KRAFTON/Raon-1-mini-preview` (revision `8c9b802a...`) | `upstream/`, hashes in `source-pins.json` |
| seven safetensors headers | same repo | first 8 MiB of each shard in `headers/`; only 404-420 KiB headers are parsed |
| Q4_K_M GGUF header | `KRAFTON/Raon-1-mini-preview-gguf` (revision `cd61e81a...`) | first 128 MiB in `headers/raon-q4km.head`; tensor table ends at 12,048,512 B |
| llama.cpp command surface | `krafton-ai/llama.cpp` branch `raon-1-mini-preview` at `fb89566...` | `upstream/llama-common-arg.cpp` |

The full base shards (32,702,145,088 file bytes), `tokenizer.json`, and the 19,686,434,240-byte
Q4_K_M payload were **not** downloaded. The published Q4 SHA-256 is from the repository's
`SHA256SUMS`; the header itself is hashed in `source-pins.json`.

## Reproduce

From the repository root:

```bash
# Headers only; rerun these range requests if the local headers are absent.
curl -fsSL -r 0-8388607 \
  https://huggingface.co/KRAFTON/Raon-1-mini-preview/resolve/main/model-00001-of-00007.safetensors \
  -o bench-logs/raon-r0/headers/model-00001-of-00007.safetensors.head
# ... repeat for shards 2-7 ...
curl -fsSL -r 0-134217727 \
  https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf/resolve/main/raon-1-mini-preview-Q4_K_M.gguf \
  -o bench-logs/raon-r0/headers/raon-q4km.head

python bench-logs/raon-r0/build_manifest.py
```

The builder verifies the downloaded upstream files against `source-pins.json`, parses the seven
header lengths, checks the index-to-shard mapping, every expert slot, the GGUF metadata, and the
GGUF tensor-offset byte sum. It writes:

- `base-tensor-inventory.json` — 19,434 tensors
- `gguf-q4km-tensor-inventory.json` — 731 tensors
- `manifest-summary.json` — counts, layer bytes, cache formulas, comparisons
- `source-pins.json` — revisions, file hashes, license and access notes
- `build.log` — the successful run (last run: 19,434 base / 731 GGUF / 8,064 B per token)

## Results

| item | measured |
|---|---:|
| base tensors | 19,434 |
| base tensor bytes | 32,699,649,368 |
| base shard file bytes | 32,702,145,088 |
| header parameter elements | 28,783,218,602 |
| HF summary parameters (200 scalars excluded) | 28,783,218,402 |
| active parameters | 4,317,319,082 (15.0%) |
| routed experts | 3,072 slots, 128 per MoE layer × 24 layers |
| routed-expert tensors | 18,432 (6 per slot) |
| routed-expert bytes | 26,471,301,120 (80.95%) |
| core/dense bytes | 6,228,348,248 |
| MXFP8 bytes per expert | 8,616,960 |
| MXFP8 six cold experts | 51,701,760 B |
| Q4_K_M file | 19,686,434,240 B; 19,674,385,728 B payload |
| Q4_K_M tensors | 731; 72 packed expert tensors |
| Q4 bytes per expert slot | 5,048,320-6,092,800 |
| Q4 six cold experts | 30,289,920-36,556,800 B |
| compressed MLA cache | 8,064 B/token (7.875 KiB); 1.96875 GiB at 262,144 |
| HF expanded MLA cache | 179,200 B/token (175 KiB); 43.75 GiB at 262,144 |
| RDA state | 48,844,800 B (46.58 MiB) per sequence |

The base is **18 RDA + 7 MLA** layers; layer 0 is dense and layers 1-24 are MoE. The exact
zero-based schedules, per-layer bytes, dtypes and GGUF metadata are in `manifest-summary.json`.
The active count is 6/128 of routed-expert elements plus all non-routed elements; the model card's
“30B-A4B” is rounded.

## Findings that change the plan

1. **Q4_K_M is not an exactness reference.** It is a 19.7 GB mixed Q4_K/Q5_0/Q6_K/Q8_0 file. The
   exact reference is the base MXFP8 checkpoint with greedy Transformers, not Q4.
2. **The published 8 KiB/token cache and the HF cache are different.** The card's 8,064 B/token
   is the compressed 576-value latent per MLA layer; `Raon1MiniDynamicCache` stores expanded
   `[B,40,T,192]` + `[B,40,T,128]`, 179,200 B/token (175 KiB). At 262K that is 2.014 GiB versus
   43.795 GiB including the RDA state. The exact llama.cpp layout remains to be read from the
   pinned branch.
3. **Router ties are unspecified.** Selection is `sigmoid(logits) + e_score_correction_bias` with
   `torch.topk(sorted=True)`; the selected weights are the unbiassed sigmoid values, renormalized;
   the shared expert is added after. There is no explicit tie-break.
4. **The header/HF scalar delta is real.** HF's `safetensors.parameters` omits 200 scalar tensors
   (50 F32 `eta`, 150 BF16 `alpha`); the manifest uses the complete header count and records both.
5. **Access is requested.** The card says approval is automatic and asks for a Hugging Face
   login/access request; the API's `gated` flag was false during this audit. The installer should
   honor the card's request/login flow and must not silently fetch the 32.7 GB base or the 19.7 GB
   Q4 file.

## Not run

- No base checkpoint load, no Q4 load, no GPU generation, no 4/4 golden result.
- No Raon llama.cpp binary; the pinned build command and flags were checked in source only.
- No converter, pack, Kindi model code, catalog entry, installer offer or support claim.

The exact reference harness is `reference_greedy.py`; it writes prompt/output token IDs and hashes.
The Q4 runtime command is a separate sanity check and must be labelled lossy.
