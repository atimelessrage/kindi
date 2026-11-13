# Tandem CPU+GPU runtime

Status: **dense runtime implemented and token-exact on the 2B and 9B, 2026-11-12. Prefix reuse
checkpoints both owners jointly, and the real 2B and 9B joint-cache checks pass in both placements,
including saved-opening reloads. The MoE expert-cache path has a model-free state-machine prototype
and a Gemma 4 26B backend adaptation in [MOE-TANDEM.md](MOE-TANDEM.md) and
[GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md), but no MoE backend or integration. The 27B `cpu-first`
split is token-exact at 8,192 context and peaked below the full-GPU baseline (7,005 vs 7,238 MiB;
2026-11-12).**
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

## Prompt cache

`TandemRunner` subclasses the normal `cache.Runner` over `TandemEngine.cache_view()`. The view
exposes both owners' `M`/`conv` arrays and the K/V arrays of every attention layer as one list, so a
checkpoint, a parked conversation or a saved opening covers the whole split. K/V stays
position-indexed; a checkpoint restores the two owners' recurrent states. Saved openings are
fingerprinted by the placement, layer ranges and CPU ISA/thread count, so a state made by one split
is never reused by another. The tandem path has no VRAM checkpoint slot: a checkpoint is a host copy
of both owners' state.

A saved opening also contains both owners' K/V. A real check found that `Openings.load` assigned
NumPy K/V into CuPy views with a plain `eng.Kc[i][:p] = ...`, which CuPy treated as a scalar fill;
it now uses `_to_dev`, the same in-place copy used for parked conversations. Both placements then
loaded the disk opening and produced the same reply as a cache-off server.

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
request through the GPU-first placement generated the same reply at 66.7 tok/s. That run reported
zero cached prompt tokens because it predates the joint checkpoints. A repeated prefix now resumes
from a checkpoint that holds both owners' recurrent state; the K/V arrays of both owners stay
position-indexed on their devices.

Free card memory after the engines loaded at that context was 6,213 MiB for the full GPU, 6,411 MiB
for CPU-first and 6,455 MiB for GPU-first (198-242 MiB released on this 2B). This is free memory
after load, not a peak; the release is the stored tensors of the layers that moved to the CPU. The
2B fits either way, so this is a correctness of the loading path, not the larger-model result.

These are one-run functional measurements, not a speed claim: the CPU layers are much slower than
the GPU and the split has not been tuned. The useful result is that the hidden-state handoff and
both ownership directions are token-exact.

## 2B joint-cache check (2026-11-12)

A real 2B check exercised `TandemRunner` in both placements with a 320-token `T1` and 8 greedy
continuation tokens. The extension case resumed from the end state (320 tokens); the branch case
restored the joint checkpoint at token 256 after a different suffix. With
`KINDI_PROMPT_CACHE=0`, the same continuation prompt was re-read from token 0.

| placement | extension (cached / cache-off) | branch (cached / cache-off) | token-exact |
|---|---:|---:|---|
| `cpu-first` | 320 / 0 | 256 / 0 | **yes, both** |
| `gpu-first` | 320 / 0 | 256 / 0 | **yes, both** |

The cached and cache-off token IDs were identical in all six in-process cells. The cache-on times
were 0.186-0.313 s and the cache-off times 3.534-4.514 s for the same eight tokens; these are
one-run harness timings, not a speed claim. The CPU engine was built with clang 18.1.8/libomp,
`avx2-fma-f16c`, 6 threads; the GPU was the RTX 4060 Ti and the context was 512 tokens.

The 2B runner chose 256-token checkpoints under the 512 MB host budget, so the branch point is 256.
The same run enabled saved openings (`prefix_max=1`, `prefix_min=256`): after the branch wrote a
256-token opening, a new `TandemRunner` loaded it from disk and matched cache-off token-for-token
in both placements. Full evidence: `bench-logs/tandem-cache/RESULTS.md`.

A prepare-only run on the same 2B environment measured the reusable prefixes at **320/325**
tokens (extension) and **256/261** tokens (branch) in both placements. The joint recurrent-state
checkpoint is **19.27 MiB**; five-sample medians were **8.7-8.9 ms** to save and **3.3-3.5 ms** to
restore. The checkpoint contains both owners' `M`/`conv` arrays. K/V (6.00 MiB at this context)
stays in the two engines and is not part of the measured copy; saved openings store K/V. See
[PREFIX-RESULTS.md](../bench-logs/tandem-cache/PREFIX-RESULTS.md).

### HTTP server restart

A real `kindi serve` check used a 362-token request whose system opening ends at token 344. The
first server read a warm prompt, then a branch request wrote the 344-token opening. After a restart
over the same state directory, the next request reported `cached_tokens: 344` and
`loaded a saved opening of 344 tokens`; a fresh server with `KINDI_PROMPT_CACHE=0` reported 0 and
produced the identical reply. This passed in both `cpu-first` and `gpu-first`. The save and load
each took about 0.1 s. Evidence: `bench-logs/tandem-cache/server-results-*.json` and
`server-runs/`.

### 4K and 8K context

With the checkpoint interval set explicitly, both placements passed the equality check at 4,096
and 8,192 context. The 4K run used 1,024-token checkpoints, a 4,080-token `T1`, and a 3,072-token
branch; the 8K run used 2,048-token checkpoints, an 8,176-token `T1`, and a 6,144-token branch.
Extension reuse was 4,080/8,176 tokens and branch restore was 3,072/6,144 tokens in both
placements. All cached output IDs matched cache-off. Saved openings were skipped in these runs;
the 512-context saved-opening and server-restart checks remain above. Full commands and timings:
`bench-logs/tandem-cache/LONG-CONTEXT-RESULTS.md`.

### 9B check

The 9B pack passed the same 512-context joint-cache check in both placements with 6 CPU layers. The
CPU owner loaded in 105-108 s and the GPU owner in 5.8-6.0 s. `cpu-first` used CPU layers 0..5 and
GPU 6..31; `gpu-first` used GPU 0..25 and CPU 26..31. Extension reuse was 320 tokens, branch
restore was 256, and a fresh runner loaded the 256-token saved opening in both placements. All
cached outputs matched cache-off. The 50 MB joint checkpoint was spaced every 256 tokens. Full
commands, timings and IDs: `bench-logs/tandem-cache/RESULTS-9B.md`.

The 9B saved-opening reloads were then measured at 4,096 and 8,192 context in both placements
(2026-11-13), with the owned-slice CPU load:

| context | placement | extension cached | branch cached | saved opening cached | exact | load | cache-off read | cached read |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 4,096 | `cpu-first` | 4,080 / 4,085 | 3,072 | 3,072 | yes | 39.5 s | 82 s | 0.39 s |
| 4,096 | `gpu-first` | 4,080 / 4,085 | 3,072 | 3,072 | yes | 33.6 s | 103 s | 0.57 s |
| 8,192 | `cpu-first` | 8,176 / 8,181 | 6,144 | 6,144 | yes | 33.3 s | 179 s | 0.44 s |
| 8,192 | `gpu-first` | 8,176 / 8,181 | 6,144 | 6,144 | yes | 33.6 s | 221 s | 0.72 s |

Every cached continuation produced the same token IDs as its cache-off control, and each saved
opening was written by the branch and then loaded into a fresh runner. Evidence:
`bench-logs/tandem-cache/saved-9b-4k.json` and `saved-9b-8k.json`.

## Current limits

- **Dense Bonsai only.** The manifest check accepts a Bonsai pack and rejects the Gemma 4 /
  Nemotron and Nanbeige backends.
- **Host-only checkpoints.** `TandemRunner` has no VRAM checkpoint slot (one slot cannot hold two
  owners), so every tandem checkpoint is a host copy; prefix reuse is exact but pays that copy cost.
  On the 2B the joint recurrent checkpoint is 19.27 MiB, with 8.7-8.9 ms median save and 3.3-3.5 ms
  median restore.
- **No fast head, speculative decoding or `feed_batch`.** The tandem runner is exact-head and one
  token per backend call.
- **One GPU.** The split composes with the CPU engine only; the two-card `TPEngine` is not a tandem
  owner.
- **27B: both placements exact; 2B/4B/9B/27B throughput and peaks recorded.** The 2026-11-12
  gate records `cpu-first` and `gpu-first` exactness with sampled peaks. The `cpu-first` cells
  in [CROSS-MODE.md](benchmarks/CROSS-MODE.md) have real `bench` throughput runs at a
  24,576-token context (2B read 80.9, generation 65.7/71.3/55.5/43.4/35.8; 4B read 62.5,
  generation 52.1/51.4/33.5/24.8/20.0; 9B read 39.9, generation 28.5/35.8/28.2/20.1/16.0
  tok/s). The 27B row with sampled peak is in
  [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md): 7.2-8.4 tok/s at 100
  against 34.9 full-GPU, 446 MiB (8K) and 193 MiB (24K) less peak.
- **The capacity premise needs restating.** On this card the full-GPU 27B loads at 8,192 and at
  24,576 context (peaks 7,473 and 7,783 MiB, 477 and 167 MiB free), so the split is not what
  makes the model runnable here. It buys headroom at a 4-6x throughput cost; a real
  larger-than-VRAM row still needs a context or a card where the full-GPU engine cannot load.

## 27B gate (2026-11-12)

At an 8,192-token context on the RTX 4060 Ti, the full-GPU plan engine passed **4/4 prompts,
40/40 tokens each**, with a sampled peak of **7,238 MiB total** (6,101 MiB over idle; 950 MiB
left on the card). Load took 20.5 s.

The 27B has 24 query heads and 4 KV heads (6:1). The plain GPU `Engine` used to require exactly
4:1 and rejected the pack (`unsupported grouped head dimensions`); the full-GPU path uses the
separate `TPEngine` plan engine for this model. The plain engine now validates `G = NH // NHKV`
generically (`Engine._gqa_groups`, `G <= 8`, head size 256 for non-4 groups) and, for `G != 4`,
launches the generic-GQA kernels that already existed in `kindi.cu` (`k_attn_sg`, `k_attn_vgn`,
`k_attn_combn`). The 4:1 fast kernels are unchanged (the 2B/4B/9B `kindi verify` remain ALL PASS
on this card), and no new kernels were written. `tests/test_gqa_heads.py` pins the ratios and the
kernel selection.

With that, the `cpu-first` split passed **4/4 prompts, 40/40 tokens each**: the CPU owns layers
0..15, the GPU owns 16..63 and the head. Combined load was 385.6 s (CPU 364.8 s + GPU 20.3 s;
that run loaded the whole pack on the CPU) and
the sampled peak was **7,005 MiB total** (5,827 MiB over idle; 1,183 MiB left) — **233 MiB (3.2%)
below the full-GPU baseline**. The context stayed at 8,192 (`reduced: false`). The dump is
`bench-logs/tandem-27b-gate/cpu-first-8192.json`; the full record is `RESULTS.md` there.

The `gpu-first` placement also passed **4/4 prompts, 40/40 tokens each** (GPU 0..47 with the
embedding, CPU 48..63, CPU head; combined load 413.3 s; sampled peak **7,017 MiB total**, 5,857
MiB over idle, 1,171 MiB left) — 221 MiB below the full-GPU baseline; the context was never
reduced. The 2026-11-13 rerun with the owned-slice CPU load reproduces both placements exactly
with 105.8-114.6 s loads; see [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md). The 2026-11-11 attempt (CPU ready in 435.4 s, then the GPU owner rejected the pack) is
kept as history in `RESULTS.md`. The fixed `--probe-gpu-owner` preflight constructs both 27B GPU
ranges at 8,192 context.

Those load times are the old full-model CPU loads. The CPU engine now accepts
`layers=...`/`embed=...`/`head=...` and `TandemEngine` passes its owned CPU range, so a tandem
start converts only the CPU-owned layers plus the embedding/head it owns (16 of 64 layers on
the 27B splits) instead of the whole pack. The new load time is unmeasured here; do not quote
the old 365 s as the current cost.

## The MoE approach is different

tandem.cpp's expert cache does work that a dense layer split cannot: a MoE token routes to a few
experts, so the hot experts can stay in VRAM, a cache miss can be computed by the CPU while the
GPU continues, and lookahead can prefetch the next token's predicted experts. That policy needs a
real MoE backend with per-expert tensors and router logits. Kindi's Bonsai packs are dense, and the
current repository has no MoE routing path for them (Nemotron is a separate Gemma 4 backend, not a
Bonsai MoE engine). Adding an LRU expert cache here would therefore cache nothing or pretend that
`kindi/engine.py` is a MoE engine.

The detailed design is [MOE-TANDEM.md](MOE-TANDEM.md): the dense core, router and shared expert stay
on the GPU; routed experts stay resident in host RAM; a byte-budgeted cache holds hot experts. The
first exact path computes every selected expert on the GPU, with cache misses copying weights. CPU
miss compute is a separate lane, enabled only after bitwise parity. The Gemma 4 26B profile is in
[GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md); it uses top-8 experts, fused gate/up tensors, a parallel
dense MLP and layer-scoped plans. This runtime remains dense-only.

## 27B rate, peak and checkpoint cost (2026-11-13)

Measured on the same host with `kindi bench --sample-vram` (one fresh process per cell, exact
head) and `measure_prefix.py --checkpoint-only --vram` at 8,192 context. Full table, commands and
the 4K/8K saved-opening evidence: [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md).

| context | engine | read | 100 | 2K | 8K | 16K | 24K | peak | free at peak |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | tandem `cpu-first` | 10.5 | 7.2 | 7.9 | 5.6 | - | - | 7,027 MiB | 923 MiB |
| 8,192 | tandem `gpu-first` | 9.4 | 7.6 | 6.6 | 5.3 | - | - | 7,027 MiB | 923 MiB |
| 8,192 | full GPU (plan engine) | 37.2 | 34.8 | 34.3 | 33.0 | - | - | 7,473 MiB | 477 MiB |
| 24,576 | tandem `cpu-first` | 10.5 | 8.4 | 8.0 | 6.0 | 3.8 | 3.0 | 7,699 MiB | 251 MiB |
| 24,576 | full GPU (plan engine) | 37.4 | 34.9 | 34.2 | 33.0 | 31.6 | 30.2 | 7,783 MiB | 167 MiB |

The joint checkpoint is **149.6 MiB** (37.4 MiB CPU owner, 112.2 MiB GPU owner) and does not grow
with context, since it holds recurrent state rather than K/V: 54.5-61.5 ms median save and
27.0-28.5 ms median restore at 4K and 8K in both placements, five samples each. Saved-opening
reloads are exact at 4K and 8K (extension 4,080/8,176 tokens, branch and saved opening
3,072/6,144 cached). The 512 MB budget keeps one copy and logs one dropped copy; K/V at 8K is
512 MiB per split and stays in the two engines.

The CPU-layer boundary is now measured too: on the 2B at 24,576 context the knee is 4-6 CPU
layers, and 24K decode falls from 118.9 tok/s (2 layers) to 18.2 (12 layers) while every split
stays below GPU-only. See the [tandem benchmark README](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/README.md).

## Next gates

1. **Larger-than-VRAM exactness:** both 27B placements are exact at 8K (2026-11-12).
2. **Peak VRAM and rate:** recorded for the 27B at 8K and 24K in both placements against a
   full-GPU baseline, with `kindi bench --sample-vram` (2026-11-13). The remaining gate is a
   context or card where the full-GPU engine genuinely cannot load; on this card even 24K fits.
3. **Joint cache:** end-state reuse, joint-checkpoint restore and saved-opening reloads through the
   HTTP server are measured on both placements at 512 tokens; in-process equality and saved-opening
   reloads pass at 4K/8K on the 9B and 27B, and the 27B joint checkpoint costs 54.5-61.5 ms to save
   and 27.0-28.5 ms to restore in both placements.
4. **Tuning:** the 2B CPU-layer boundary is measured (knee at 4-6 layers; every split is below
   GPU-only). Prompt reading on the small-card target and a 27B boundary sweep remain open.
5. **MoE:** design in [MOE-TANDEM.md](MOE-TANDEM.md); implementation only after a real MoE backend
   with per-expert tensors and router logits exists.
