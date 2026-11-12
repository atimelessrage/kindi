# Nanbeige 33B CPU+GPU proposal: triage and next gates

Status: **review only, 2026-11-11. No engine, kernel, installer or test code was changed. The
shared proposal is unverified input; only claims checked against this tree are treated as facts.
This page is not a speed or support claim.**

The proposal targets AH-Nanbeige4.3-33B IQ3_XXS on a 12 GB Quadro K6000 and suggests full
intra-layer CPU/GPU slicing, a CPU drafter, CPU K/V attention, parallel vocabulary scoring and
several speculative-decoding schemes. Some of that is already in the Nanbeige backend in a
different form; some is not feasible as a drop-in.

## 1. Current tree: two different CPU+GPU paths

Kindi does not have one generic tandem engine. The paths must not be conflated.

| path | current behaviour | evidence |
|---|---|---|
| `kindi/nanbeige.py` | The 33B backend. On a K6000 at 8,448 context it keeps **13 layers** in pinned host memory, copies them into two device slots while computing, and computes **55% of the FFN rows** of each host layer on the CPU. Attention, the output head and all K/V stay on the card. | `CPU_F`, `HOST_SLOTS`, `_plan`, `_hcopy`, `_mv_split`, `_layers`; [BENCHMARKS.md](BENCHMARKS.md) K6000 section |
| `kindi/tandem.py` | Dense **Bonsai** layer split only: CPU prefix/suffix and one GPU owner, one hidden-state handoff per token. It rejects `kindi_nanbeige` and `kindi_gemma4` manifests. | manifest guard in `TandemEngine.__init__`; [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md) |
| `kindi/spec.py` | The 2B Bonsai drafter is a **GPU** drafter on the same card; the 27B target uses it. This is not a CPU drafter and is not the Nanbeige path. | `KINDI_SPEC_PACK`, `draft2b`; [MTP.md](MTP.md) |
| `kindi/nanbeige_gen.py` | Prompt lookup and a Nanbeige MTP-head draft source; the released 33B GGUF has no MTP head, so prompt lookup is the free default. There is no CPU draft engine. | `lookup`, `mtp_head`; [CARDS.md](CARDS.md) Nanbeige section |

The measured K6000 result is **3.56 tok/s at 8,160 prompt tokens**, 3.55 at 2,016 and 3.85 at
480, with 55% of the host-layer FFN rows on the CPU (8 threads). At 1,024 context with 6 host
layers it is 4.4-5.0 tok/s. These are the baseline for any proposed speedup.

## 2. Claim triage

| proposal claim | status against this tree | correction |
|---|---|---|
| K6000 keeps 13 layers in host memory | **true** at 8,448 context | Context-dependent; it is not always 13. |
| The CPU runs those 13 layers sequentially and the GPU is idle | **false** | The card copies host layers into slots while it computes; `_mv_split` runs the GPU rows and CPU FFN rows concurrently. The CPU is a bottleneck, not the only worker. |
| Split every matrix 80-85% VRAM / 15-20% RAM | **not implemented** | Only FFN rows of **host layers** are split today. The other layers are fully on the card; host-layer attention is still copied and computed on the card. |
| Only activations cross PCIe | **false for the current path** | Host-layer **weights** are copied into the two device slots on each pass. Only the split FFN inputs/outputs use the pinned activation buffers. Eliminating weight copies is the main architectural difference in the proposal. |
| Reuse llama.cpp AVX2/AVX-512 dequantization | **false** | `nbq_cpu.c` is Kindi's own IQ3_XXS codebook implementation, built with `gcc -O3 -march=native -fopenmp`; it uses AVX2/FMA and OpenMP, not llama.cpp. There is no explicit AVX-512 path. |
| CPU 2B drafter + GPU 33B verification | **not implemented** | The 2B drafter is a GPU Bonsai engine; `nanbeige.py` has its own draft protocol and no CPU drafter adapter. Cross-tokenizer drafting is not a supported mode. |
| Batch verification is compute-bound, so 3-5 tokens are nearly free | **false at small T** | The measured MTP pass costs are 1.80x/2.61x/3.42x one step at T=2/3/4 on a K1200 and 1.82x/2.81x/3.65x on a GTX 1080. Verification still reads the weights. |
| CPU K/V attention for old context | **not implemented** | K/V stays on the card in `nanbeige.py`; CPU attention would need partial-softmax reduction and an exactness contract. |
| Parallel vocabulary scoring on CPU | **not implemented** | The head is on the card; `nbq_argmax_part`/`nbq_argmax_final` split the search across GPU blocks, not CPU. |
| DFlash 360M, EAGLE-3, Medusa, PLE offload | **not in this tree** | No checkpoints, CPU kernels, tokenizer adapter, training path or tests. They are separate projects, not flags. |
| 7-9 tok/s on K6000; 12-18 on a 10 GB 3080; 1.5-1.8x on a 3090 | **unsupported** | No measurement or implementation supports these numbers. Treat them as hypotheses, not targets. |

## 3. Corrected bandwidth arithmetic

For a single memory-bound matrix-vector pass, with model bytes `M`, GPU bandwidth `B_g`, CPU
bandwidth `B_c`, GPU fraction `R`, CPU fraction `1-R`, and a synchronization cost `s`:

```
T_gpu(R) = R * M / B_g
T_cpu(R) = (1 - R) * M / B_c
T(R)    = max(T_gpu(R), T_cpu(R)) + s
R_max   = (V_usable - KV - overhead) / M
```

Ignoring compute and synchronization, the balance point is

```
R* = B_g / (B_g + B_c),     R_actual = min(R*, R_max)
```

This is a **ceiling**. It does not include CPU dequantization, kernel launch, PCIe activation
exchange, all-gather latency, or the current per-host-layer weight copies.

For the K6000 case (`M = 13.1 GB`, `B_g = 288 GB/s`, `B_c = 45 GB/s`, about 10.5 GB usable after
K/V and overhead):

- `R_max = 10.5 / 13.1 = 0.80`; CPU share is 2.6 GB, GPU share is 10.5 GB.
- CPU weight-read time is about `2.6 / 45 = 58 ms`; GPU time is about `10.5 / 288 = 36 ms`.
- The CPU is the long pole. The ideal ceiling is about `1 / 0.058 = 17 tok/s`, before compute and
  synchronization. The measured 3.56 tok/s is about 21% of that ceiling.
- The proposal's 14-16 tok/s is therefore in the right order as an **upper bound**, not a
  forecast. The 80/20 split is set by VRAM, not by bandwidth equilibrium: the bandwidth-optimal
  CPU share would be about 14%, but it would exceed the 10.5 GB usable VRAM budget.

The same calculation for the examples in the proposal:

| setup | usable weights | GPU / CPU bytes | ideal CPU read time | ideal ceiling | measured 33B |
|---|---:|---:|---:|---:|---:|
| K6000 12 GB, DDR4 | 10.5 GB | 10.5 / 2.6 GB | 58 ms | ~17 tok/s | **3.56 tok/s** at 8K |
| RTX 3080 10 GB, DDR5 | 8.5 GB | 8.5 / 4.6 GB | 54 ms | ~18 tok/s | not measured on 10 GB |
| GTX 1080 8 GB, DDR4 | 6.5 GB | 6.5 / 6.6 GB | 147 ms | ~6.8 tok/s | not measured |
| RTX 3090 24 GB, DDR5 | 13.1 GB | about 12.05 / 1.05 GB | 12 ms | about 80 tok/s | **59.7 tok/s** on one card |

The 3090 row shows why a theoretical split is not enough: the model already fits, and the CPU
slice, synchronization and CPU compute can erase the small ideal gain. The 3090's measured 59.7
tok/s is below the ideal 80 tok/s ceiling even without a CPU split.

## 4. What is worth doing

### P0: measure the existing K6000 path before adding an engine

This is the highest-value step and needs no new inference code. Sweep the existing
`KINDI_NB_CPUF` and `KINDI_NB_SLOTS` settings at 480, 2,016 and 8,160 tokens, with the same
prompts and f16 K/V as the recorded run.

- `KINDI_NB_CPUF`: 0.35, 0.45, 0.55, 0.65, 0.75.
- `KINDI_NB_SLOTS`: 2 and, where memory permits, 3.
- Record tok/s, CPU and GPU busy time, host-to-device copy bytes, and peak VRAM.

The gate is diagnostic: identify whether the current 3.56 tok/s is CPU-FFN-bound, PCIe-copy-bound
or GPU-bound. If a simple `CPUF`/slot change reaches a meaningful gain, a new 80/20 engine is not
the first fix.

### P1: resident FFN slices, only if P0 shows copy or host-layer overhead

The closest feasible version of the proposal is not "split every matrix at once":

1. Keep the attention, output head and K/V on the GPU.
2. Keep a fixed GPU/CPU row split of **FFN gate/up/down** for **all layers**, sized to the measured
   CPU/GPU balance, so host layers no longer need a full weight copy each token.
3. Use the existing pinned activation exchange and CPU `nbq_cpu.c` kernels first; add CPU
   attention only after the FFN slice works.

This would still be a new loader/layout and a new execution schedule; it cannot be enabled by
changing `KINDI_NB_CPUF`.

Gate: token-exact on the 33B verify set (ties allowed as recorded), at least 1.2x the current 3.56
tok/s at 8K (about 4.3 tok/s), and peak VRAM no higher than the current K6000 profile.

### P2: attention and output-head split, only after P1

A CPU attention slice needs Q/K/V and output projections, partial softmax or an exact reduction,
and a second exchange per layer. A CPU output-head slice needs a merge of top-k candidates, not
just an argmax. Both are separate kernels and exactness gates, not part of the FFN prototype.

### P3: CPU drafting, only if P0/P1 show idle CPU capacity

The existing free option is prompt lookup, already on by default for Nanbeige. A CPU neural
drafter needs:

- a CPU engine for the draft model;
- a tokenizer/vocabulary compatibility check with the 33B;
- the same state/check API used by `nanbeige.py`'s draft protocol; and
- a net-speed gate that includes the CPU time taken away from the host-layer FFN work.

The current 2B Bonsai GPU drafter does not satisfy these conditions. DFlash, EAGLE-3 and Medusa
are separate model/training projects and should not be mixed into this gate.

### P4: CPU K/V offload, last

Streaming old K/V blocks to CPU attention requires exact partial-softmax reduction and a new
attention path. It is not the first bottleneck on K6000 with the current 8K context, and it should
not be started until the weight path is measured.

## 5. Acceptance matrix

| item | current | first gate |
|---|---|---|
| K6000 33B baseline | 3.56 tok/s at 8K | reproduce with the same prompts and settings |
| `KINDI_NB_CPUF`/`KINDI_NB_SLOTS` sweep | not recorded | identify the bottleneck; no code change |
| resident FFN slice | not implemented | token-exact; ≥1.2x baseline at 8K; peak VRAM recorded |
| CPU attention/head | not implemented | exactness and a measured gain after P1 |
| CPU drafter | not implemented | same tokenizer, net tok/s gain, no quality change |
| CPU K/V attention | not implemented | exact partial-softmax gate at long context |
| DFlash/EAGLE/Medusa | not in tree | separate checkpoint, tokenizer, training and support decision |

## 6. What to drop from the proposal

- Do not describe the current path as sequential CPU layers with an idle GPU.
- Do not claim the current code reuses llama.cpp dequantization routines.
- Do not treat 80/20 as bandwidth-optimal; it is a VRAM-capped layout.
- Do not use 7-9, 12-18 or 1.5-1.8x as targets without a measurement.
- Do not start with CPU K/V, output-head splitting or a CPU drafter; the current K6000 path is
  already an FFN split with a measurable bottleneck.
- Do not add these ideas to `tandem.py`: that engine is Bonsai-only and its 27B gate is still
  blocked by the GPU owner's head-ratio check.

## 7. Evidence in this tree

- `kindi/nanbeige.py`: `CPU_F` (default 0.55), `HOST_SLOTS` (2), `_plan`, `_hcopy`, `_mv_split`,
  `_layers`; the host-layer FFN row split and the per-pass weight copies.
- `kindi/nbq_cpu.c`: AVX2/FMA/OpenMP IQ3_XXS kernels; `kindi/nanbeige_cpu.py` builds it with
  `-march=native`.
- `kindi/tandem.py`: Bonsai-only manifest guard; `cpu-first`/`gpu-first`; one hidden handoff per
  token.
- `kindi/spec.py`: GPU 2B drafter and prompt lookup; `kindi/nanbeige.py`: separate `generate_spec`
  with a draft object, no CPU adapter.
- `kindi/nanbeige_gen.py`: prompt lookup default and optional Nanbeige MTP head.
- [BENCHMARKS.md](BENCHMARKS.md), [CARDS.md](CARDS.md), [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md),
  [MTP.md](MTP.md), [CPU_GPU_TANDEM_DESIGN.md](CPU_GPU_TANDEM_DESIGN.md).

The first required artifact is a reproducible profile of the current K6000 path. Until that
exists, the proposal remains a hypothesis.
