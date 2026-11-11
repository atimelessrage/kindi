# Tandem CPU+GPU runtime

Status: **dense runtime implemented and token-exact on the 2B, 2026-11-11. The MoE expert-cache
approach is designed but not implemented, and no model larger than one GPU has been measured yet.**
The runtime is `kindi/tandem.py`, with the per-layer GPU API in `kindi/engine.py`. The recovery
history and the cleanup gate for the interrupted branch are in [TANDEM-STATUS.md](TANDEM-STATUS.md);
the original design is [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md).

## Two dense placements

A dense model needs every matrix on every token, so the placement unit is a complete layer, not an
expert. Both backends keep their own recurrent state and K/V cache; one hidden-state vector crosses
between them per token.

| mode | embedding | layers | output head | best use |
|---|---|---|---|---|
| `cpu-first` | CPU | CPU prefix `0..n-1`, GPU suffix `n..NL-1` | GPU | a small GPU that can hold the expensive suffix and the LM head |
| `gpu-first` | GPU | GPU prefix `0..NL-n-1`, CPU suffix `NL-n..NL-1` | CPU | keeping the embedding and early layers on the GPU while the last layers and LM head stay off it |

`n` is `--tandem-cpu-layers` (default: a quarter of the layers).

The GPU engine now accepts a layer range and uploads only the tensors for that range
(`Engine(layers=..., embed=..., head=...)`). This is a real VRAM reduction: unselected layer
weights are never copied to the card. The CPU engine is still the existing whole-model CPU engine,
so a tandem run holds the full model in host RAM as well as the GPU-side subset. Reducing host RAM
is a separate change.

## Usage

The flags are on `verify`, `bench`, `serve` and `chat`:

```sh
kindi --pack ~/.kindi/packs/bonsai-2b verify --tandem cpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b bench  --tandem gpu-first --tandem-cpu-layers 6
kindi --pack ~/.kindi/packs/bonsai-2b serve  --tandem cpu-first --tandem-cpu-layers 6
```

The CPU library must be buildable on the host (`gcc` with OpenMP, or clang with libomp on Windows).
The tandem path requires a CUDA/ROCm GPU; `KINDI_DEVICE=cpu` is rejected.

## Verification

Measured on the i7-8086K (6 CPU threads) + RTX 4060 Ti 8 GB, native Windows, 2B pack, 1,024-token
context. `kindi verify` compares greedy output token for token with llama.cpp's reference.

| engine | placement | prompts token-exact | tokens/s over the run |
|---|---|---:|---:|
| full GPU | all 24 layers | 4/4, 40/40 tokens each | 189.9 |
| tandem | CPU 0..5, GPU 6..23, GPU head | 4/4, 40/40 tokens each | 70.2 |
| tandem | GPU 0..17, CPU 18..23, CPU head | 4/4, 40/40 tokens each | 70.9 |

The server path was exercised at the default 24,576-token context. A CPU-first chat request with 18
prompt tokens generated 9 tokens (`"Hello! How can I help you today?"`) at 91.7 tok/s; the same
request through the GPU-first placement generated the same reply at 66.7 tok/s. Both reported zero
cached prompt tokens: the tandem runner deliberately does not use the normal prompt cache yet.

Free card memory after the engines loaded at that context was 6,213 MiB for the full GPU, 6,411 MiB
for CPU-first and 6,455 MiB for GPU-first (198-242 MiB released on this 2B). This is free memory
after load, not a peak; the release is the stored tensors of the layers that moved to the CPU. The
2B fits either way, so this is a correctness of the loading path, not the larger-model result.

These are one-run functional measurements, not a speed claim: the CPU layers are much slower than
the GPU and the split has not been tuned. The useful result is that the hidden-state handoff and
both ownership directions are token-exact.

## Current limits

- **Dense Bonsai only.** The manifest check accepts a Bonsai pack and rejects the Gemma 4 /
  Nemotron and Nanbeige backends.
- **No prompt-prefix cache.** `TandemRunner` reads the prompt from position 0 each request. The
  normal cache snapshots one engine's `M`, `conv`, `Kc` and `Vc`; a joint two-backend snapshot is
  not implemented.
- **No fast head, speculative decoding or `feed_batch`.** The tandem runner is exact-head and one
  token per backend call.
- **One GPU.** The split composes with the CPU engine only; the two-card `TPEngine` is not a tandem
  owner.
- **No measured model that does not fit the card.** The 2B fits easily; the gate that matters is a
  27B (or larger) run with a recorded peak VRAM and a comparison against the GPU-only baseline.

## The MoE approach is different

tandem.cpp's expert cache does work that a dense layer split cannot: a MoE token routes to a few
experts, so the hot experts can stay in VRAM, a cache miss can be computed by the CPU while the
GPU continues, and lookahead can prefetch the next token's predicted experts. That policy needs a
real MoE backend with per-expert tensors and router logits. Kindi's Bonsai packs are dense, and the
current repository has no MoE routing path for them (Nemotron is a separate Gemma 4 backend, not a
Bonsai MoE engine). Adding an LRU expert cache here would therefore cache nothing or pretend that
`kindi/engine.py` is a MoE engine.

The MoE design, when a backend exists, is: keep the router and hot experts on the GPU; keep the
remaining experts in host RAM; on a miss, run the expert's GEMV/FFN on the CPU and add its exact
weighted output to the GPU's residual; prefetch only experts the router can predict. It must remain
exact and must be gated on a real MoE pack and token-exact verification, not on the tandem.cpp
claims. It is not part of this runtime.

## Next gates

1. **Larger-than-VRAM exactness:** a 27B-class model through both placements, token-exact against
   the full-GPU reference where that reference fits.
2. **Peak VRAM:** record `nvidia-smi`/CuPy peak memory for full GPU, CPU-first and GPU-first at the
   same context.
3. **Joint cache:** decide whether to checkpoint both engines' state together or keep the tandem
   path cache-free; do not make the cache silently reuse a mismatched state.
4. **Tuning:** measure the layer boundary (`n`) and prompt reading on the actual small-card target.
5. **MoE:** only after a MoE backend with per-expert tensors and router logits exists.
