# Tandem with a plan-engine GPU owner: 27B VRAM, rate and a 48K context (2026-11-14)

The tandem GPU half used to be the plain engine, so an offloaded layer freed its weights but the
layers that stayed cost fp16 K/V, the whole f32 file was uploaded, and on the 27B the plain engine
was too heavy to load the full model at 24K at all. `--tandem-gpu plan` (the default) makes the plan
engine (`kindi/tp.py:TPEngine(layers=...)`) the owner: it loads only the GPU-owned tensors and steps
one prebuilt plan per block.

Machine: i7-8086K (6 CPU threads, `avx2-fma-f16c`, native Windows, clang 18.1.8), RTX 4060 Ti 8,187 MiB,
driver as of 2026-11-14. Pack: `bonsai-27b` (64 layers, 16 attention, Hadamard 1024). Raw dumps are
local (ignored) `bench-logs/tandem-plan-owner/*.json`; this page is the tracked summary.

## Exactness

`kindi verify` (4 prompts against the llama.cpp reference), plan owner, token for token:

| model | placement | result |
|---|---|---|
| 2B | `cpu-first`, 8 CPU layers (GPU head) | **ALL PASS** 4/4, 65.5 tok/s |
| 2B | `gpu-first`, 8 CPU layers (CPU head) | **ALL PASS** 4/4, 47.4 tok/s |
| 9B | `cpu-first`, `gdn:6` | **ALL PASS** 4/4, 25.1 tok/s |

## Joint prompt cache with the plan owner

`bench-logs/tandem-cache/measure_prefix.py --pack bonsai-2b --ctx 512 --cpu-layers 6 --modes cpu-first`
(plan owner, the default): extension request reused **320 of 325** prompt tokens and the branch **256 of
261**, the cache-off controls reused 0, and every continuation matched token for token. The joint
checkpoint is 19.27 MiB (14.59 MiB GPU + 5.61 MiB CPU), median save 29.2 ms, restore 4.0 ms. The 512-token
context also exposed a bug the fix covers: the plan engine's 1024-token floor rejected a tandem owner's
small context, and a partial engine now only requires 256 (it has no chunked or batched prompt path).

## 27B at 24,576 tokens: owner A/B

`kindi bench --tandem cpu-first --tandem-cpu-layers gdn:16 --tandem-gpu ...`, one fresh process per
run; `gdn:16` is the same placement for both owners.

| owner | runs | free after load | sampled peak | read tok/s | decode at 24K tok/s |
|---|---:|---:|---:|---|---|
| plain | 1 | 197 MiB | 7,721 MiB | 7.5 | 4.5 |
| plan | 5 | **1,985 MiB** | **6,004 MiB** (5,862-6,083) | 7.2 (5.5-9.0) | 5.8 (4.7-6.4) |
| plan, `auto` (11 layers) | 3 | 1,319 MiB | 6,449 MiB (6,444-6,756) | 9.2 (8.4-11.2) | 7.9 (4.5-9.3) |
| full GPU (no split) | 1 | 404 MiB | 7,783 MiB | 37.4 | 30.2 |

- Same placement, **1.7 GiB lower peak**: an offloaded layer now frees ~107 MiB of weights plus its
  share of the heavy per-layer state, and the layers that remain use half the K/V (q8_0).
- Decode at 24K is within this machine's run-to-run noise for one cell (5.5-9.0 tok/s across five
  runs). Do not read a rate difference between the owners out of these numbers; the VRAM difference
  is deterministic and 15x the noise band.
- The split is still much slower than the unsplit plan engine (30.2 tok/s at 24K): the CPU-owned
  layers dominate the token.
- `auto` with the plan owner estimates 6,548 MiB and measured 6,444-6,756 MiB, so the 512 MiB reserve
  it promises is kept.
- Load: 11 CPU layers 87 s, 16 layers 122 s (owned-slice CPU conversion); the GPU side ranged-loads
  its tensors out of the existing 5-trit cache, so no second multi-GB cache file is written.
- `KINDI_DP4A=1` on this path is much worse (2.0-4.0 tok/s): the int8 score path is not the default
  on CUDA for a reason.

## 27B at 49,152 tokens: what the freed memory buys

| request | placement | context held | free after load | peak | read | decode (100/2K/8K/16K/24K) |
|---|---|---|---:|---:|---:|---|
| 49,152 | unsplit plan engine | 30,976 | 122 MiB | 7,872 MiB | 34.2 | 21.0/28.5/29.7/28.0/27.3 |
| 49,152 | `auto` = 19 GDN layers, plan owner | **49,152** | 1,240 MiB | 6,876 MiB | 5.3 | 4.9/4.9/2.8/2.7/2.0 |

The unsplit engine reduces a 49,152-token request to 30,976 on this card; the split holds the full
request. That is the larger-than-VRAM case the tandem existed for, and it is 10-15x slower per token.

## Commands

```sh
python -m kindi --pack ~/.kindi/packs/bonsai-2b verify --tandem cpu-first --tandem-cpu-layers 8 --tandem-gpu plan
python -m kindi --pack $HOME/.kindi/packs/bonsai-27b --ctx 24576 bench \
  --tandem cpu-first --tandem-cpu-layers gdn:16 --tandem-gpu plan --sample-vram
python -m kindi --pack $HOME/.kindi/packs/bonsai-27b --ctx 49152 bench \
  --tandem cpu-first --tandem-cpu-layers auto --tandem-gpu plan --sample-vram
```

## Not verified

- No mixed-card, P106, Arc, Blackwell/FP8 or ROCm run: this is one Ada card.
- No joint-cache run on the 9B or 27B with the plan owner (the 2B cell covers the mechanism).
- No `KINDI_TC=1` tensor-core plan on this path: a tandem owner builds the plan engine's own plans.
- A split across two GPU cards is not a tandem placement (the plan owner is one card).
- No 4B run with the plan owner (the 2B, 9B and 27B cover both head placements and the 27B geometry).
- Throughput per cell is not separable from this machine's noise at n=3-5; only the VRAM numbers are
  quoted as medians with ranges.
