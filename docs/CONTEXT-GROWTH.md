# Context growth across Kindi's supported models

Kindi's 0.6 line serves five model families: Ternary Bonsai (2B, 4B, 9B and 27B),
AH-Nanbeige4.3-33B, Nemotron 3.5 Lightning 30B-A3B, DeepSeek-V4.1-Flash and Gemma 4 E4B.
This page sizes the per-sequence state each engine allocates as context grows. Every byte
figure is derived arithmetic, not a benchmark; no new model run was made for this page.
Units are binary (KiB, MiB, GiB).

Two things add up:

- **Fixed allocation.** A recurrent layer keeps one state per sequence, allocated once. Its
  size does not change from 4K to 64K. The Gated-DeltaNet (GDN) layers in Bonsai, the
  Mamba-2 layers in Lightning and the local rings in DeepSeek are in this group. Bonsai's
  fixed column also includes the convolution slot the engine allocates on every layer,
  including attention layers that do not use it.
- **Growing cache.** A full-attention layer stores one K/V entry per position, so its cache
  is `positions * bytes_per_token`. The attention layers in Bonsai, every Nanbeige layer,
  the six Lightning attention layers and the global layers in Gemma 4 are in this group.

A hybrid model's per-sequence footprint is therefore fixed allocation plus a smaller linear
cache. The tables add both; weights, workspaces and the engine's reserve are separate.

## At a glance

"Fixed" is the allocation described above; "cache" is only the growing K/V. The 4K and 64K
columns add both for f16 K/V. The token column gives f16 and q8_0 sizes where a q8_0 path
exists.

| family | fixed (MiB) | cache/token (f16 / q8) | total 4K (MiB) | total 64K (MiB) | geometry basis |
|---|---:|---|---:|---:|---|
| Bonsai 2B | 19.7 | 12 / 6.4 KiB | 67.7 | 787.7 | local pack/GGUF |
| Bonsai 4B | 51.0 | 32 / 17 KiB | 179 | 2,099 | local pack/GGUF |
| Bonsai 9B | 51.0 | 32 / 17 KiB | 179 | 2,099 | local pack/GGUF |
| Bonsai 27B | 151.5 | 64 / 34 KiB | 407.5 | 4,247.5 | test geometry |
| Nanbeige 33B | 0 | 208 / 110.5 KiB | 832 | 13,312 | published config |
| Lightning 30B-A3B | 47.6 | 6 / — KiB | 71.6 | 431.6 | published config |
| DeepSeek-V4.1-Flash | 10-50 | 8.25 / — KiB | 43-83 | 538-578 | code |
| Gemma 4 E4B | 40 | 16 / — KiB | 104 | 1,064 | published config |

"Local pack/GGUF" means the dimensions were read from local `kindi_pack.json` and GGUF
files; "test geometry", "published config" and "code" mean the row is inferred. The byte
figures are calculated in every row, not measured. The family sections give the exact
geometry and formulas.

Nanbeige is the steepest: its 4K cache alone is larger than the 27B's 4K cache plus fixed
allocation. Lightning's per-token slope is the smallest, but its 47.6 MiB state is not zero.
DeepSeek's fixed rings make only four layers own the compressed global KV.

## Ternary Bonsai (2B, 4B, 9B, 27B)

All four use `full_attention_interval = 4`: `NATT = NL / 4` layers have full attention and
the other `G = NL - NATT` layers are GDN. The 4B and 9B geometry was read from their local
pack manifests and GGUF headers: `qwen35.ssm.inner_size = 4096`, so `NV = 32`. The 27B row
uses recorded test geometry (`gdn_inner = 6144`, `NV = 48`), not a local header read.

The growing cache is only the attention K/V:

```
Kf16 = NATT * 2 * NHKV * HD * 2
Kq8  = NATT * 2 * NHKV * (HD + (HD / 32) * 2)
```

`NHKV` is the number of K/V heads, not the query heads. The fixed allocation is separate:

```
M       = G * NV * 128 * 128 * 4          # the delta matrix, fp32
CONV_CH = 2 * 16 * 128 + NV * 128
CONV    = NL * 3 * CONV_CH * 4             # a convolution slot on every layer
fixed   = M + CONV + (NL - G) * 4          # 1-float M slot on attention layers
NV      = gdn_inner / 128
```

| size | layers | GDN `G` | attention `NATT` | K/V heads | head dim | f16/token | q8_0/token | GDN value heads `NV` | live GDN | fixed allocation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | 24 | 18 | 6 | 2 | 256 | 12 KiB | 6.375 KiB | 16 | 19.3 MiB | 19.7 MiB |
| 4B | 32 | 24 | 8 | 4 | 256 | 32 KiB | 17 KiB | 32 | 50.3 MiB | 51.0 MiB |
| 9B | 32 | 24 | 8 | 4 | 256 | 32 KiB | 17 KiB | 32 | 50.3 MiB | 51.0 MiB |
| 27B | 64 | 48 | 16 | 4 | 256 | 64 KiB | 34 KiB | 48 | 149.6 MiB | 151.5 MiB |

The live GDN state alone is 19.27, 50.25 and 149.6 MiB on the 2B, 4B/9B and 27B. The fixed
column adds the 0.4, 0.75 and 1.9 MiB of convolution slots on layers that do not use them.

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

At 16K the 2B cache is 192 MiB and the 27B cache is 1 GiB; their fixed allocations are 19.7
and 151.5 MiB. Only `G` layers grow with context.

Kindi's fit also charges `NH * 4 + (NH * HD * 4) / 256 + rope_dim * 4` bytes per position for
attention scores, partials and RoPE tables: 320 B (2B), 384 B (4B/9B) and 448 B (27B). The
tables above count K/V only, so add 0.31-0.44 KiB per position for the fit.

## AH-Nanbeige4.3-33B

Nanbeige is a dense 52-layer Llama-style decoder with full attention on every layer, no
recurrent state and no sliding window. The [published config](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/main/config.json)
has 56 query heads and 8 K/V heads (7:1 GQA) at head dimension 128; the query heads do not
multiply the K/V bytes. The local catalog GGUF is corrupt as served (its tensor-info names
are NUL-garbled), so this geometry comes from the published config and no Kindi run is
claimed.

```
Kf16 = 52 * 2 * 8 * 128 * 2       = 212,992 B/token = 208 KiB/token
Kq8  = 52 * 2 * 8 * (128 + 8)     = 113,152 B/token = 110.5 KiB/token
```

| cache | 4K | 16K | 64K | 262,144 |
|---|---:|---:|---:|---:|
| f16 | 832 MiB | 3,328 MiB | 13,312 MiB | 53,248 MiB |
| q8_0 | 442 MiB | 1,768 MiB | 7,072 MiB | 28,288 MiB |

There is no fixed recurrent state to add; the cache is the whole context cost apart from
workspace, alignment and the weights. The config declares 524,288 maximum positions; the
262,144 column is an arithmetic reference, not a Kindi run.

## Nemotron 3.5 Lightning 30B-A3B

The 52 layers are 23 Mamba-2 mixers, 23 MoE feed-forward layers (128 routed experts, top-6,
plus one shared expert) and 6 attention layers. NVIDIA's [published config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json)
gives 2 K/V heads at head dimension 128; Kindi stores the six attention caches as f16. No
Lightning GGUF or `kindi_lightning` pack is present on this host, so this row is inferred
from that config and the engine's assertions.

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
context is 4K or 1M. The engine's plan subtracts this state and the dense weights before
fitting the K/V and expert placement. There is no q8_0 K/V path in the Lightning engine; the
cache is f16.

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

Each of the 40 layers also owns a fixed local ring of 128 positions plus the prompt chunk:

```
ring = 40 * (128 + tmax) * 512 * 4
```

At `tmax = 1` that is about 10.1 MiB; at the documented 512-token prompt chunk it is about
50 MiB. It does not grow with the number of positions processed. The 1,048,576 column is
arithmetic; the backend's default context is 8,192.

The reference numerics round the compressed KV and index to FP4 and the window KV and
attention output to FP8, but Kindi's host/CPU buffers are float32 (the allocations in
`kindi/dsv41_cpu.c` are `... * HD * 4` and `... * IHD * 4`). The byte figures above are the
buffer sizes Kindi actually allocates.

## Gemma 4 E4B

The [published model](https://huggingface.co/google/gemma-4-E4B/blob/main/config.json) has
42 layers: 35 local sliding-window layers (window 512, 2 K/V heads, head dimension 256) and
7 global layers (2 K/V heads, head dimension 512). The last 18 layers
share a KV cache with an earlier layer, so only 24 layers own a cache: 20 local rings and 4
global caches. The local rings are fixed; the global caches grow.

```
fixed  = 20 * 2 * 2 * 256 * 1024 * 2   = 41,943,040 B = 40 MiB
Kf16   = 4 * 2 * 2 * 512 * 2           = 16,384 B/token = 16 KiB/token
```

| | 4K | 16K | 64K | 128K |
|---|---:|---:|---:|---:|
| global K/V | 64 MiB | 256 MiB | 1,024 MiB | 2,048 MiB |
| K/V + local rings | 104 MiB | 296 MiB | 1,064 MiB | 2,088 MiB |

The local ring is 1,024 positions per owning local layer in `kindi/gemma4.py` (the 512-token
window plus room for one prompt batch), not 512. No Gemma 4 E4B pack is present on this
host, so this section is an allocation derivation from the engine and the published config.

## Scope and status

- All byte figures are derived arithmetic; no model was loaded or benchmarked. The
  "geometry basis" column names where the dimensions came from, not a measured allocation.
- Tables count per-sequence cache and fixed allocation only. They exclude weights,
  activation workspaces, attention score buffers beyond the Bonsai `per_pos` note,
  prompt-cache checkpoint copies, parked conversations, CUDA graphs and allocator padding.
- Columns beyond 64K are arithmetic references, not supported-context claims for a
  particular machine. Kindi's fit can reduce the requested context to whatever the free
  memory allows.

The allocation formulas are in `kindi/engine.py`, `kindi/nanbeige.py`, `kindi/lightning.py`,
`kindi/dsv41_cpu.c` and `kindi/gemma4.py`. Prompt-cache policy is in [PROMPT-CACHE.md](PROMPT-CACHE.md);
family installation and performance notes are in [BONSAI-2-SUPPORT.md](BONSAI-2-SUPPORT.md),
[DEEPSEEK.md](DEEPSEEK.md) and [CARDS.md](CARDS.md).
