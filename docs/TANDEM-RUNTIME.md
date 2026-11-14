# Tandem CPU+GPU runtime

Status: **dense runtime implemented and token-exact on the 2B, 9B and 27B, with a joint prompt cache.
Placement matters more than layer count: moving gated-delta layers and leaving attention on the GPU
keeps long-context decode flat (9B at 24K: 23.6 vs 11.0 tok/s; 27B at 24K: 7.9 vs 2.9; medians of
three, 2026-11-14).** Prefix-reuse checkpoints cover both owners jointly, and the real 2B and 9B
joint-cache checks pass in both placements, including saved-opening reloads. The MoE expert-cache path
is a model-free state-machine prototype with a Gemma 4 26B backend adaptation in
[MOE-TANDEM.md](MOE-TANDEM.md) and [GEMMA4-26B-MOE.md](GEMMA4-26B-MOE.md); there is no MoE backend or
integration. Every split is still slower than GPU-only on this card.
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

`--tandem-cpu-layers` takes one of three forms (default: a quarter of the layers):

- a **count** — `6`: a contiguous CPU prefix (`cpu-first`) or suffix (`gpu-first`), the original behavior;
- an **explicit layer set** — `0,1,2,4,5,6` or `0-2,4-6`: any sorted subset, so the two backends can interleave;
- **`gdn:N`** — the first `N` gated-delta (non-attention) layers, ascending. Attention layers stay on the
  GPU. This is the placement to prefer; see [where to put the CPU layers](#where-to-put-the-cpu-layers);
- **`auto`** (or **`auto:MiB`** to also leave that much room) — the smallest `gdn:`-style set that makes
  the requested context fit, computed from the pack's own tensor sizes, the fp16 K/V per attention layer
  at that context, the head, the f32 file and a measured scratch allowance. It **refuses instead of
  splitting** when the model already fits, and refuses with the missing amount when no split can fit.
  `KINDI_TANDEM_RESERVE_MB` (default 512) sets the free memory it leaves behind.

`kindi bench --json` records the resolved `cpu_range`/`gpu_range` of a run, so a recorded split is
unambiguous.

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

## Where to put the CPU layers (2026-11-14)

Not every layer costs the same on the CPU. A **full-attention** layer on the CPU is the worst choice
available: its per-token cost grows with the context (it scans the whole K/V cache on the CPU) and the
K/V it would have held is exactly the memory a split is trying to free. A **gated-delta (GDN)** layer is
the opposite: O(1) work per token regardless of position and a device state of ~1 MiB. Moving GDN layers
and leaving attention on the GPU is what keeps a split's decode rate flat with position.

All rows below are `kindi bench --sample-vram`, exact head, one fresh process per run, medians of three
runs in one session on the i7-8086K + RTX 4060 Ti 8 GB. The 27B `0-15` control is one run (the earlier
recorded row reproduced to within 5%).

### 9B, 24,576-token context

| placement | read | 100 | 2K | 8K | 16K | 24K | sampled peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| full GPU (plan engine) | 99.6 | 87.1 | 86.5 | 81.1 | 76.7 | 73.2 | 3,276 MiB |
| `cpu-first` 0..5 (has attention layer 3) | 27.9 | 24.4 | 23.7 | 18.7 | 14.4 | **11.0** | 3,452 MiB |
| `gdn:6` (0,1,2,4,5,6, attention on the GPU) | 23.2 | 23.3 | 27.1 | 22.6 | 25.5 | **23.6** | 3,543 MiB |

The contiguous split loses 55% of its decode rate between 2K and 24K; the GDN-first split loses 13%.
At 24K `gdn:6` is **2.1x** the contiguous split's rate for the same number of CPU layers, and 1.55x the
9B tandem row previously recorded in [CROSS-MODE.md](benchmarks/CROSS-MODE.md) (15.2 tok/s at 24K). At
8K the same effect is smaller (23.1 vs 17.9 tok/s at position 8K, medians of three), which is why the
contiguous split looked acceptable while it was only measured at short context.

### 9B, the split costs more than it saves

| placement | read | 100 | 2K | 8K | peak | CPU layers |
|---|---:|---:|---:|---:|---:|---:|
| full GPU | 106.0 | 88.5 | 87.9 | 83.9 | 3,098 MiB | 0 |
| `cpu-first` 0..5 | 24.4 | 25.2 | 21.5 | 17.9 | 3,072 MiB | 6 |
| `gdn:6` | 23.8 | 19.8 | 24.6 | 23.1 | 3,092 MiB | 6 |
| `gdn:3` (0,1,2) | 36.5 | 35.6 | 35.6 | 32.5 | 3,201 MiB | 3 |

At 8,192 context the 9B leaves ~5 GiB free on this card, and the split's peak is **not lower** than the
full-GPU peak. A 9B-ternary pack is 2.3 GiB, so there is nothing to win here: on a card this size the
right placement for the 9B is **no split at all**. The split only becomes interesting when the model is
genuinely larger than the card, which is the 27B case below.

### 27B, 24,576-token context

| placement | read | 100 | 2K | 8K | 16K | 24K | sampled peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| full GPU (plan engine) | 37.4 | 34.9 | 34.2 | 33.0 | 31.6 | 30.2 | 7,783 MiB |
| `cpu-first` 0-15 (attention layers 3,7,11,15 on the CPU) | 9.9 | 8.4 | 7.2 | 5.3 | 3.7 | **2.9** | 7,434 MiB |
| `gdn:16` (0,1,2,4,...,20) | 9.2 | 8.4 | 6.7 | 6.4 | 7.7 | **7.9** | 7,744 MiB |

Same layer count off the GPU, same short-context rate, and **2.7x** the 24K decode rate. The GDN-first
placement keeps 384 MiB more K/V on the card at 24K (the 4 attention layers it does not offload), which
is the price of the flat curve.

### `auto`: the split is only as big as it has to be (2026-11-14)

Hand-picking a layer count is easy to get wrong in both directions, so `--tandem-cpu-layers auto`
computes the minimum set. It uses the pack's real tensor sizes (`kindi/tandem.py:pack_bytes`: bcodes
plus one fp16 scale per 128-column group, or 4 bytes per f32 value), the fp16 K/V of each attention
layer the GPU keeps, the output head, the whole f32 file and a 192 MiB scratch allowance, checked
against the free VRAM before the model loads. The estimate is accurate: on this 27B at 24,576 it
predicts 7,778 MiB against a measured plan-engine peak of 7,783 MiB.

| model | request | `auto` result |
|---|---|---|
| 9B | 24,576 | refuses: "already fits at 24576 tokens (~3282 MiB needed, 7075 MiB free); a split would only make it slower" |
| 27B | 24,576 | 19 CPU layers (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24), context kept at 24,576 unreduced, 522 MiB free after load, sampled peak 7,257 MiB, read 8.3 and decode 6.3-7.0 tok/s |
| 27B | 24,576, 5,100 MiB free | 39 CPU layers (still GDN-only) |

A hand-picked `gdn:16` is faster at 24K (7.9 tok/s) because it leaves only 283 MiB free; `auto` keeps
the 512 MiB reserve by default (`KINDI_TANDEM_RESERVE_MB`).

`auto:1024` asks for **1,024 MiB of headroom freed**, not merely for a model that leaves that much, so
it splits even when the model would fit unsplit. Measured on the 9B at 8,192: 14 GDN layers on the CPU,
5,523 MiB free after load, read 17.5 and 8K decode 16.9 tok/s — useful when something else needs the
card, not a throughput choice. The freed number the planner reports is against the **plain** engine's
allocation, which is ~600 MiB above the plan engine's row at 8K (q8_0 K/V vs fp16), so do not compare
it with a full-GPU row taken on the plan engine.

### The split frees much less VRAM than the layer count suggests

Offloading 16 of 64 layers does not free 16/64 of the pack on the 27B: the measured peak only fell from
7,783 to 7,434 MiB. The reason is that a tandem's GPU owner is the **plain** engine, and on the 27B the
plain engine is much heavier per layer than the plan engine: `KINDI_TP=0` cannot load the full 27B at
24,576 context at all (`not enough GPU memory for this model: 0 MiB free after the weights`), while the
plan engine loads it with 7,783 MiB peak and runs it at 30 tok/s. The plan engine's 27B plan also uses
the q8_0 K/V cache, which halves K/V bytes; the plain engine has no K/V mode and is always fp16.

So the tandem's VRAM saving is real but small on this card, and its rate cost is large. The two ways to
make the split pay off are engineering, not tuning:

1. **Give the tandem a plan-engine GPU owner.** `kindi/tp.py:TPEngine` has no per-layer API; it fuses
   layers into a plan. Extending its plan to start at an arbitrary layer is the change that would make an
offloaded layer free what it should.
2. **A q8_0 K/V cache in the plain engine.** At 24K the 27B's 16 attention layers hold 1.5 GiB of fp16
   K/V; q8_0 would return ~0.75 GiB without touching a single weight.

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
  24,576-token context, medians of three fresh-process runs (2B read 84.3, generation 79.8/72.3/58.8/43.2/34.9;
  4B read 49.3, generation 47.0/43.8/31.7/23.7/16.7; 9B read 38.1, generation 37.5/34.7/25.4/20.5/15.2
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
| 24,576 | tandem `gpu-first` | 8.5 | 7.4 | 6.2 | 4.5 | 3.3 | 2.7 | 7,339 MiB | 611 MiB |
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

On the 27B the boundary is different: 8 CPU layers cannot hold a 24K context (auto-reduced to
17,664 tokens with 23 MiB free), 16 layers keep it (7,699 MiB peak), and going to 24/32 layers
lowers the peak to 6,103/5,115 MiB at a 35-50% rate cost. At 8K, stepping 8 -> 28 layers in
steps of 4 lowers the peak 7,427 -> 5,277 MiB at a flat 428-432 MiB saved per step and
15-37% decode cost per step (read 16.8 -> 6.5, 8K decode 10.7 -> 3.2): the exchange rate is
~27-29 MiB saved per percent of decode rate, with no knee — the flat trade is the finding, and
beyond 28 layers it is dominated by rate. For 24K the boundary is instead the capacity
threshold at 16 layers (loads and keeps 24,576 with 687 MiB free, while 8 layers reduce to
17,664 with 23 free). The capacity question was settled the
same evening: the loader silently reduces context to fit, and on this card a full-GPU 27B holds
~37K tokens while the 16-layer tandem holds ~30K (7,812/7,518 MiB peaks). The split never gains
usable context here; it trades throughput for headroom. Full table: [27B-8K.md](benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md).

## Next gates

1. **Larger-than-VRAM exactness:** both 27B placements are exact at 8K (2026-11-12).
2. **Peak VRAM and rate:** recorded for the 27B at 8K and 24K in both placements against a
   full-GPU baseline, with `kindi bench --sample-vram` (2026-11-13). The capacity question is
   settled on this card: requesting 65,536 or 131,072 tokens reduces to 37,120 (full GPU) and
   30,464 (tandem), and the tandem is always the more reduced engine here. A genuinely closing
   case needs a device where the full-GPU engine cannot run at a usable context yet a split can.
3. **Joint cache:** end-state reuse, joint-checkpoint restore and saved-opening reloads through the
   HTTP server are measured on both placements at 512 tokens; in-process equality and saved-opening
   reloads pass at 4K/8K on the 9B and 27B, and the 27B joint checkpoint costs 54.5-61.5 ms to save
   and 27.0-28.5 ms to restore in both placements.
4. **Tuning:** the 2B boundary is measured (knee at 4-6 layers) and the 27B boundary is too
   (24K needs at least 16 CPU layers here; 24-32 layers buy memory at a steep rate cost). The
   placement is now measured as well: `gdn:N` keeps attention on the GPU and flattens the
   long-context decay (2026-11-14). Every split is still below GPU-only on this card, and the split's
   VRAM saving is limited by the plain engine's footprint, so the next real step is a plan-engine
   owner or a q8_0 K/V cache rather than more layer sweeps.
5. **Context fit on a split:** fixed 2026-11-14. The GPU owner's context estimate charged it for every
   attention layer of the model instead of the ones it loads, so a split reduced context too eagerly
   (the 27B with 8 CPU layers at 24,576 requested now holds 19,712 tokens instead of 17,664).
6. **MoE:** design in [MOE-TANDEM.md](MOE-TANDEM.md); implementation only after a real MoE backend
   with per-expert tensors and router logits exists.
