# Context growth across Kindi's supported models

Kindi's 0.6 line serves five model families: Ternary Bonsai (2B, 4B, 9B and 27B),
AH-Nanbeige4.3-33B, Nemotron 3.5 Lightning 30B-A3B, DeepSeek-V4.1-Flash and Gemma 4 E4B.
This page derives what each engine allocates as the context grows. It is arithmetic from
the allocation code and the model geometry, not a benchmark: no new model run was made for
this page, and the numbers are rounded binary units (KiB, MiB, GiB).

Two different things are often mixed together:

- **Fixed state.** A recurrent layer keeps one state per sequence, allocated once. Its size
  does not change from 4K to 64K. The Gated-DeltaNet (GDN) layers in Bonsai, the Mamba-2
  layers in Lightning and the local rings in DeepSeek are in this group.
- **Growing cache.** A full-attention layer stores one K/V entry per position, so its cache
  is `positions * bytes_per_token`. The attention layers in Bonsai, every Nanbeige layer,
  the six Lightning attention layers and the global layers in Gemma 4 are in this group.

A hybrid model is therefore not "flat". It is **fixed state plus a smaller linear cache**.
Kindi's context fit adds both, then the weights, workspaces and a reserve.

## At a glance

"Growing cache" is the cache at 4K and 64K positions only; add the fixed state for the
per-sequence total. The per-token column is the slope between those two points.

| family | architecture | layers with a growing cache | per-token cache | fixed state | 4K -> 64K cache |
|---|---|---|---:|---:|---:|
| Bonsai 2B | 18 GDN + 6 attention (24 layers) | 6 / 24 | 12 KiB f16; 6.4 KiB q8 | 19.3 MiB | 48 -> 768 MiB |
| Bonsai 4B | 24 GDN + 8 attention (32 layers) | 8 / 32 | 32 KiB f16; 17 KiB q8 | 25.7 MiB | 128 -> 2,048 MiB |
| Bonsai 9B | 24 GDN + 8 attention (32 layers) | 8 / 32 | 32 KiB f16; 17 KiB q8 | 25.7 MiB | 128 -> 2,048 MiB |
| Bonsai 27B | 48 GDN + 16 attention (64 layers) | 16 / 64 | 64 KiB f16; 34 KiB q8 | 149.6 MiB | 256 -> 4,096 MiB |
| Nanbeige 33B | 52 full-attention layers | 52 / 52 | 208 KiB f16; 110.5 KiB q8 | none | 832 -> 13,312 MiB |
| Lightning 30B-A3B | 23 Mamba-2 + 23 MoE + 6 attention | 6 / 52 | 6 KiB f16 | 47.6 MiB | 24 -> 384 MiB |
| DeepSeek-V4.1-Flash | 40 layers, 4 compressed-KV sources | 4 sources; all 40 have fixed 128-token rings | 8.25 KiB (5 KiB KV + 3.25 KiB index + 4 B token map) | ~10-50 MiB of rings | 33 -> 528 MiB |
| Gemma 4 E4B | 35 local + 7 global; last 18 share KV | 4 global caches / 42 | 16 KiB f16 | 40 MiB local rings | 64 -> 1,024 MiB |

The Nanbeige row is the steepest by a wide margin: its 4K cache alone is larger than the
27B's 4K cache plus fixed state. Lightning's per-token slope is the smallest of the dense
families, but its 47.6 MiB Mamba/conv state is not zero. DeepSeek is different in kind: its
local windows are fixed and only four layers own the compressed global KV.

## Ternary Bonsai (2B, 4B, 9B, 27B)

The publisher describes the 2B/4B/9B as Qwen3.5-based and the 27B as Qwen3.8-based. All
four interleave three GDN blocks with one full-attention block: `full_attention_interval = 4`,
so `NATT = NL / 4` attention layers and the other `G = NL - NATT` layers are GDN.

The growing cache is only the attention K/V:

```
Kf16 = NATT * 2 * NHKV * HD * 2
Kq8  = NATT * 2 * NHKV * (HD + (HD / 32) * 2)
```

`NHKV` is the number of K/V heads, not the query heads. The GDN state is separate:

```
M       = G * NV * 128 * 128 * 4          # the delta matrix, fp32
CONV_CH = 2 * 16 * 128 + NV * 128
CONV    = G * 3 * CONV_CH * 4             # the three convolution channels
fixed   = M + CONV
NV      = gdn_inner / 128
```

| size | layers | GDN `G` | attention `NATT` | K/V heads | head dim | f16/token | q8_0/token | GDN value heads `NV` | fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | 24 | 18 | 6 | 2 | 256 | 12 KiB | 6.375 KiB | 16 | 19.3 MiB |
| 4B | 32 | 24 | 8 | 4 | 256 | 32 KiB | 17 KiB | 16 | 25.7 MiB |
| 9B | 32 | 24 | 8 | 4 | 256 | 32 KiB | 17 KiB | 16 | 25.7 MiB |
| 27B | 64 | 48 | 16 | 4 | 256 | 64 KiB | 34 KiB | 48 | 149.6 MiB |

The fixed column is the live GDN state (the same state a prompt-cache checkpoint copies):
19.27 MiB on the 2B and 149.6 MiB on the 27B. The engine also allocates a convolution slot
on every layer, including the attention layers that do not use it, which adds about 0.4,
0.6 and 1.9 MiB on the 2B, 4B/9B and 27B respectively.

Cache only, in MiB:

| cache | 4K | 16K | 64K | 262,144 |
|---|---:|---:|---:|---:|
| 2B f16 | 48 | 192 | 768 | 3,072 |
| 2B q8_0 | 25.5 | 102 | 408 | 1,632 |
| 4B f16 | 128 | 512 | 2,048 | 8,192 |
| 4B q8_0 | 68 | 272 | 1,088 | 4,352 |
| 9B f16 | 128 | 512 | 2,048 | 8,192 |
| 9B q8_0 | 68 | 272 | 1,088 | 4,352 |
| 27B f16 | 256 | 1,024 | 4,096 | 16,384 |
| 27B q8_0 | 136 | 544 | 2,176 | 8,704 |

For scale: at 16K the 2B cache is 192 MiB and its fixed state is 19.3 MiB; the 27B cache is
1 GiB and its fixed state is 149.6 MiB. From 4K to 16K the growing part adds 144 MiB on the
2B, 384 MiB on the 4B/9B and 768 MiB on the 27B. That is smaller than a full-attention model
of the same width, but it is not flat: only `G` of the layers are recurrent.

Kindi's fit also charges `NH * 4 + (NH * HD * 4) / 256 + rope_dim * 4` bytes per position for
attention scores, partials and RoPE tables: 320 B (2B), 384 B (4B/9B) and 448 B (27B). The
tables above count K/V only, so they understate the fit by 0.31-0.44 KiB per position.

### Why Bonsai needs checkpoints

A GDN state cannot be rewound to an earlier position, so prefix caching has to copy the
recurrent state rather than just truncate a K/V array. Kindi copies it at `checkpoint_every`
(default 256 tokens; `KINDI_DENSE_SPACING` controls the dense-opening spacing) and keeps
saved openings and parked conversations through `prefix_min`, `prefix_max` and `park_max`
(see [CONFIGURATION.md](CONFIGURATION.md)). These are cache-policy copies of the fixed state,
not another per-token cache. A 2B checkpoint is about 19.3 MiB and a 27B checkpoint about
149.6 MiB.

## AH-Nanbeige4.3-33B

Nanbeige is the opposite end of the spectrum: a dense 52-layer Llama-style decoder with
full attention on every layer, no recurrent state and no sliding window. The published
config has 56 query heads and 8 K/V heads (7:1 GQA, not 8:1) at head dimension 128; the
query heads do not multiply the K/V bytes.

```
Kf16 = 52 * 2 * 8 * 128 * 2       = 212,992 B/token = 208 KiB/token
Kq8  = 52 * 2 * 8 * (128 + 8)     = 113,152 B/token = 110.5 KiB/token
```

| cache | 4K | 16K | 64K | 262,144 |
|---|---:|---:|---:|---:|
| f16 | 832 MiB | 3,328 MiB | 13,312 MiB | 53,248 MiB |
| q8_0 | 442 MiB | 1,768 MiB | 7,072 MiB | 28,288 MiB |

There is no fixed recurrent state to add; the cache is the whole context cost apart from
workspace, alignment and the weights. Kindi's engine reads `n_layer`, `n_embd`, `n_head`
and `n_kv` from the GGUF and asserts a 128-wide head and a 5:1 or 7:1 group
(`kindi/nanbeige.py`). The published checkpoint declares 524,288 maximum positions, but no
Kindi run at that length is claimed here.

The catalog Nanbeige GGUF on the development host is corrupt as served (its tensor-info
names are NUL-garbled), so this page does not claim a Nanbeige verify/bench result. The
geometry above comes from the published `config.json`; the formulas come from the engine.

## Nemotron 3.5 Lightning 30B-A3B

The 52 layers are 23 Mamba-2 mixers, 23 MoE feed-forward layers (128 routed experts, top-6,
plus one shared expert) and 6 attention layers. The GGUF's per-layer K/V-head list marks
the attention layers (`Spec.kind` in `kindi/lightning.py`); they use 2 K/V heads at head
dimension 128, stored as f16.

```
Kf16 = 6 * 2 * 2 * 128 * 2        = 6,144 B/token = 6 KiB/token
S    = 23 * (64 * 64 * 128 * 4 + (4096 + 2 * 8 * 128) * 3 * 4)
     = 49,930,240 B              = 47.6 MiB
```

`S` is the Mamba-2 state plus the convolution state, allocated once for the 23 Mamba layers.
The MoE layers add no cache. Only the six attention layers grow:

| | 4K | 16K | 64K | 262,144 | 1,048,576 |
|---|---:|---:|---:|---:|---:|
| attention K/V | 24 MiB | 96 MiB | 384 MiB | 1,536 MiB | 6,144 MiB |
| K/V + fixed | 71.6 MiB | 143.6 MiB | 431.6 MiB | 1,583.6 MiB | 6,191.6 MiB |

The Mamba state is constant-size, not free: it is about 47.6 MiB per sequence whether the
context is 4K or 1M. The engine's fit in `kindi/lightning.py` subtracts this state and the
dense weights before dividing the remaining room by `per_pos`. There is no q8_0 K/V path in
the Lightning engine; the cache is f16.

## DeepSeek-V4.1-Flash

Kindi's DeepSeek engine is a separate host/CPU engine with optional CUDA forward kernels
(`kindi_dsv41`), not a normal Bonsai-style pack. It has 40 layers. Layers 0 and 1 have only
the local window; the compressed global KV is owned by four source layers, and every layer
from 2 to 39 reuses one of those sources:

```
source layer   layers that reuse it
2              2-7
8              8-13
14             14-19
20             20-39
```

The four sources have compression ratios 2, 2, 2 and 1: three store one entry every two
tokens, and layer 20 stores one per token. The index has eight sources: 2, 8, 14, 20, 24,
28, 32 and 36. The growing buffers are:

```
CKV = (1/2 + 1/2 + 1/2 + 1) * 512 * 4             = 5,120 B/token
IDX = (1/2 + 1/2 + 1/2 + 1 + 1 + 1 + 1 + 1) * 128 * 4
                                                    = 3,328 B/token
TOK = 4 B/token                                    # int32 token map for the engram
total = 8,452 B/token                              = 8.25 KiB/token
```

| cache | 4K | 16K | 64K | 262,144 | 1,048,576 |
|---|---:|---:|---:|---:|---:|
| compressed KV + index + token map | 33.0 MiB | 132.1 MiB | 528.3 MiB | 2,113 MiB | 8,452 MiB |

The one-million-token column is arithmetic, not a Kindi run: the backend's default context is
8,192 and the installer's example is 65,536. The engine takes whatever `--ctx` it is given
and that fits the host.

Each of the 40 layers also owns a fixed local ring of 128 positions plus the prompt chunk:

```
ring = 40 * (128 + tmax) * 512 * 4
```

At `tmax = 1` that is about 10.1 MiB; at the documented default prompt chunk of 512 it is
about 50 MiB. It does not grow with the number of positions processed. Layer 20 also keeps a
candidate-block buffer sized from the configured maximum context, not from the tokens seen
so far.

The reference numerics round the compressed KV and index to FP4 and the window KV and
attention output to FP8, but Kindi's host/CPU buffers are float32 (the allocations in
`kindi/dsv41_cpu.c` are `... * HD * 4` and `... * IHD * 4`). The byte figures above are the
buffer sizes Kindi actually allocates, not a packed-4-bit size. Model-card comparisons that
quote a packing ratio are not Kindi measurements and are not restated here.

## Gemma 4 E4B

Gemma 4 E4B is also served by Kindi, but it is easy to miss in a four-family
Bonsai/Nanbeige/Lightning/DeepSeek comparison. The published model has 42 layers: 35 local
sliding-window layers (window 512, 2 K/V
heads, head dimension 256) and 7 global layers (2 K/V heads, head dimension 512). The last
18 layers share a KV cache with an earlier layer, so only 24 layers own a cache: 20 local
rings and 4 global caches. The local rings are fixed; the global caches grow.

```
fixed  = 20 * 2 * 2 * 256 * 1024 * 2   = 41,943,040 B = 40 MiB
Kf16   = 4 * 2 * 2 * 512 * 2           = 16,384 B/token = 16 KiB/token
```

| | 4K | 16K | 64K | 128K |
|---|---:|---:|---:|---:|
| global K/V | 64 MiB | 256 MiB | 1,024 MiB | 2,048 MiB |
| K/V + local rings | 104 MiB | 296 MiB | 1,064 MiB | 2,088 MiB |

The local ring is 1,024 positions per owning local layer in `kindi/gemma4.py` (the 512-token
window plus room for one prompt batch), not 512. No Gemma 4 E4B pack is present on the
development host, so this section is an allocation derivation from the engine and the
published config, not a measured run.

## What this page does not claim

- These are allocation formulas, not benchmark results. No model was loaded or benchmarked
  to produce this page.
- Bonsai is not described as "flat": six, eight or sixteen attention layers still grow
  linearly. Only a model with no full-attention layers would be flat.
- Nanbeige is 7:1 GQA (56 query / 8 K/V heads), not 8:1.
- Nanbeige's catalog GGUF is corrupt on this host; its geometry is from the published config
  and the engine reads it dynamically. No Nanbeige parity run is claimed.
- Lightning is not "mostly attention": 46 of its 52 layers are Mamba-2 or MoE, and only six
  hold attention K/V.
- DeepSeek's four compressed-KV sources are 2, 8, 14 and 20. The engine is one 40-layer
  stack; this page does not assign an encoder/decoder split that the code does not allocate.
- DeepSeek's FP4/FP8 model-card packing comparisons are not repeated as Kindi measurements.
- The large-context columns (Bonsai 262,144, Nanbeige 524,288, Lightning 1,048,576 and
  DeepSeek 1,048,576) are arithmetic references, not supported-context claims for a
  particular machine.
- The tables exclude weights, activation workspaces, attention score buffers beyond the
  Bonsai `per_pos` note, prompt-cache checkpoint copies, parked conversations, CUDA graphs
  and allocator padding. Kindi's fit can reduce the requested context to whatever the free
  memory allows.

## Where the numbers come from

| family | source of the geometry and formulas |
|---|---|
| Bonsai | `kindi/engine.py` (`kv_per_pos`, `per_pos`, `M`, `conv`), `kindi/convert.py` profiles, `tests/test_plan_engine.py` and `docs/MIXED-CARD-ROADMAP.md` for the 27B GDN geometry, `docs/CONFIGURATION.md` for the checkpoint policy |
| Nanbeige | `kindi/nanbeige.py`; [published config](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/main/config.json) |
| Lightning | `kindi/lightning.py` (`Spec.kind`, `per_pos`, fixed-state fit); 23/23/6 split in `docs/TERNARY_MOE.md` and `docs/CARDS.md` |
| DeepSeek-V4.1 | `kindi/dsv41_cpu.c` (constants, source layers, allocations), `kindi/dsv41_gpu_fwd.cuh` (rounding flags); `docs/DEEPSEEK.md` |
| Gemma 4 E4B | `kindi/gemma4.py` (cache allocation, local/global sets, shared-KV tail); [published config](https://huggingface.co/google/gemma-4-E4B/blob/main/config.json) |

The prompt-cache policy itself is in [PROMPT-CACHE.md](PROMPT-CACHE.md); family installation
and performance notes are in [BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md),
[DEEPSEEK.md](DEEPSEEK.md) and [CARDS.md](CARDS.md).
