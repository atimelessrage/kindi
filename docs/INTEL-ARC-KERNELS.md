# A770: the kernels still missing before a 27B decode step, ranked cheapest-first

Status: **planning review, not a claim.** This is the work list that follows the measured probes in
[INTEL-ARC.md](INTEL-ARC.md), derived by reading `kindi/kcpu.c` (the spec) against the six
`kindi/intel_*.py` probes (what the card actually does). Every token figure quoted here comes from a
linked benchmark page; the token times for kernels that do not exist yet are marked as estimates,
and nothing in this file is a support claim.

Review date 2026-11-26. Scope: **one Arc A770 16 GB, Windows, the OpenCL C path the existing `kindi intel-*`
probes already use**. Target: Bonsai 2.5 27B - 64 layers (48 GDN + 16 attention), D 5120, FFN 17408,
NH 24 / NKV 4 / HD 256, NK 16 / HK 128 / NV 48, conv 10240, hadamard block 1024, 7,242 MB pack.

The question answered here: *what has to exist before one real decode step runs on the card, and in what
order do we build it so a measured tok/s replaces the projected ternary floor.*

## 1. Where the Arc code actually is

| module | device kernels | what it is |
|---|---|---|
| `intel_cl.py` | `vecadd`, `gemm` (f32) | exactness probes |
| `intel_ternary.py` | `ternary_gemv_part`, `_reduce`, `_part4` | synthetic + real codes, bit-exact vs host |
| `intel_gemm.py` | `q8x`, `gemm_tile`, `gemm_tile_local` | the engine's q8 arithmetic, bit-exact |
| `intel_pack.py` | `pack_gemv4`, `pack_gemv_persist`, `pack_gemv_reduce` | every ternary tensor of the real 27B pack, bit-exact on `blk.3.attn_k` |
| `intel_l0.py` | none | Level Zero USM copy smoke |
| `intel_sycl.py` | none | icpx/toolchain detection + vecadd |

Exactly one op family exists - **the ternary matvec** - and it is good: 219-228 GB/s persistently against a
measured 385 GB/s read ceiling ([pack page](benchmarks/2026-11-25-i7-8086k-arc-a770-pack/README.md)). There is
no layer runner, no state, no token loop. `INTEL-ARC.md` says this plainly ("no backend, model or support
claim"), so nothing below is a correction of the docs, it is the work list.

## 2. One 27B decode token, as `kindi/kcpu.c` really executes it

The CPU engine is the spec. Verbatim from `gdn()`, `attn()`, `ffn_input()`, `head_body()`, `embed()`.

### GDN layer (48 of 64) - 6 ternary matvecs

| # | op | 27B shape / size | notes |
|---|---|---|---|
| 1 | `rmsfold(attn_norm, add)` | 5120 | residual add, RMS, x w, then **fold** = sign x FWHT_1024 |
| 2 | GEMV `qkv` + `gate` | 10240x5120, 6144x5120 | one folded input `xp`, two matrices |
| 3 | alpha/beta projections | 2 x 48x5120 **f32** | on the *unfolded* `xn`; sigmoid(beta), softplus(alpha+dt) x ssm_a |
| 4 | conv1d + silu | 10240 ch, 4 taps, state 3x10240 | causal |
| 5 | L2 norm | 2x16 heads x 128 | q and k |
| 6 | **delta rule** | 48 heads x 128x128 M | `sk=M^T(k e)`, `dd=(v-sk) beta`, `M=M e + k dd^T`, `o=Mq`, x 1/sqrt(128), then per-head RMS(128) |
| 7 | gate + permute | 6144 | `silu(z[perm]) outn[perm]`, rep = 48/16 = 3 |
| 8 | fold(6144) + GEMV `ssm_out` | 5120x6144 | |

### Attention layer (16 of 64) - 7 ternary matvecs

| # | op | 27B shape / size | notes |
|---|---|---|---|
| 1 | `rmsfold(attn_norm, add)` | 5120 | |
| 2 | GEMV q, k, v | 12288x5120, 1024x5120, 1024x5120 | q carries a **gate half** per head |
| 3 | per-head RMS + partial RoPE | 24+4 heads x 256, 32 pairs | q first half, k |
| 4 | KV write | 2 KB/token/layer | **f16** (`uint16`), K roped, V raw |
| 5 | scores | ctx x 6 q-heads/kv x 256, x 1/16 | GQA |
| 6 | softmax | per q head over `pos+1` | |
| 7 | weighted V + sigmoid gate | 24x256 | gate = sigmoid of q's second half |
| 8 | fold(6144) + GEMV o | 5120x6144 | |

### Every layer, then the head

- FFN: `rmsfold(post_norm, add)` -> GEMV `ffn_gate`+`ffn_up` (17408x5120 each) -> `silu(g) u` -> fold(17408) ->
  GEMV `ffn_down` (5120x17408).
- Head: `rmsfold(output_norm, add)` -> GEMV (248320x5120) -> argmax.
- Embed: one ternary **row** decode (40 group words + 8 scales = 168 B).

Totals per token: **400 layer matvecs + 1 head = 401**, 193 norm+fold passes, 48 delta-rule blocks,
16 attention blocks, 48 convs.

## 3. The gap list

Status: **have** = works and is bit-exact today; **missing** = nothing on device.

| # | kernel | must do | 27B work | calls/token | status | est. |
|---|---|---|---|---|---|---|
| K1 | `norm_fold` | residual add, RMS(n), x w, sign x FWHT(HB) -> `xn` and folded `xp`; rows for per-head modes | 5120 / 17408 / 6144 / 256 / 128 | **193** | **have** (`kindi intel-norm`, 2 kernels) | done |
| K2 | `embed_row` | decode one token's trits (128-col groups, 8 row scales) | 5120 | 1 | **closed** - host row + K1's fold | 0 |
| K3 | ternary GEMV | 401 dependent calls on a resident pack, 1-3 matrices per launch | 6.80 GB read | 257 if 4/layer | **have** (needs wiring) | 1.5 d |
| K4 | `ewise` | silu-mul, residual, sigmoid-gate-mul, group permute - one kernel, mode switch | <=17408 | ~150 | missing | 1 d |
| K5 | `ssm_gates` | alpha/beta f32 matvec `[2 NV x D]`, softplus/sigmoid, x ssm_a | 94 MB/token | 48 | missing | 0.5 d |
| K6 | `conv1d_silu` | 4-tap depthwise + state rotate | 10240 | 48 | missing | 0.5 d |
| K7 | `l2norm_heads` | 2 NK rows x HK | 32x128 | 48 | missing | 0.3 d |
| K8 | `gdn_delta` | decay, `M^T k`, rank-1 update, `Mq`, per-head RMS + gamma | 48x128x128 (3.1 MB/layer) | 48 | missing | 2-3 d |
| K9 | `qk_norm_rope` | per-head RMS(256), partial RoPE(32), KV append f16 | 28 heads | 16 | missing | 1 d |
| K10 | `attn_decode` | GQA scores, softmax, weighted V, sigmoid gate | ctx<=2048 first | 16 | missing | 2 d |
| K11 | `argmax` | reduce 248,320 logits on device | 338 MB read | 1 | missing | 0.3 d |
| K12 | sampling | top-k / temperature / greedy | 993 KB logits | 1 | **not a kernel** | 0 |

Device state to allocate once: weights **7.24 GB** (6.72 codes + 0.42 scales + 0.11 f32), GDN M 151 MB,
conv 8 MB, KV 67 MB at ctx 2,048 (1.07 GB at 32k), residual/`xn`/`xp` about 0.1 MB. Around 7.4 GB of 16 GB,
so the card holds the model with room for a 32k context.

## 4. Three corrections to the brief

1. **There is no f32 output head to build.** In the pack, `output.weight` is `PTQ1_0` ternary
   (`type 143`, `folded: true`, 248,320x5,120) - 338 MB of codes+scales, the single largest read of the
   token (1.53 ms measured) but the *same* `pack_gemv` kernel. Only norms and the SSM scalars are f32:
   29 MB of the 9B pack's 2.4 GB, and about 105 MB of the 27B's 7,242 MB - 94 MB of that is the 48 GDN
   layers' alpha/beta projections, not the head. "f32 head" is a CUDA-era assumption, not a Bonsai property.
2. **The 26.87 ms persistent work list cannot run a real decode.** It streams all 401 matrices back to back
   in one launch, which is only valid if every input is known up front. Layer *l*+1 consumes layer *l*'s
   output, so a real token is 401 *dependent* launches. The runnable baseline is the same page's per-matrix
   pattern: **47.4 ms/token (21 tok/s)**
   ([overhead page](benchmarks/2026-11-25-i7-8086k-arc-a770-gemv-overhead/README.md)). Fusion - not a new
   kernel - is the lever back toward 30 ms.
3. **Sampling needs no kernel.** 248,320 logits is 993 KB; a host download plus numpy greedy/top-k is
   about 0.05 ms of traffic. Put `argmax` on device (K11) only because it is nearly free, not because it is
   needed.

Also: the embedding is a **row**, never a stream - the 6.80 GB/token figure already excludes it. Keep it that
way; a GEMV over the token-embedding matrix would add 338 MB/token for nothing.

## 5. What a real token should measure

| path | ms/token | tok/s | source |
|---|---:|---:|---|
| spec peak read (560 GB/s) | 12.1 | 83 | arithmetic only |
| measured peak read (385 GB/s) | 17.7 | 56 | measured read probe |
| one persistent launch (not runnable) | 26.9 | 37 | measured, invalid for decode |
| **401 dependent launches** | **47.4** | **21** | measured, runnable |
| + ~500 small launches for the missing ops | ~53 | 19 | estimate |
| fused, 4 launches/layer | 30-33 | 30-33 | estimate |
| CPU engine, 2.5 27B | 249 | 4.0 | measured 2026-11-25 |

First honest Arc number: **19-21 tok/s, about 5x the CPU**, not 56. Say it that way in the report when it lands.

**Measured update, 2026-11-26 - the last estimate in that table is now a measurement, and it moves the
lever.** The packed GEMV was A/B'd on the 401 shapes with distinct bytes: 27.44 ms/token in the pack's own
layout and 26.31 ms/token over a 256-row-blocked one, and - the part that matters - with the decode deleted
the *pack layout* already reads at 405.0 GB/s, above the 385 GB/s anchor. The read pattern was never the
limiter; about 8 ms of the 26.3 ms token is `TRI_DOT`'s decode, and 16.8 ms/token is the measured ceiling if
it were free ([layout page](benchmarks/2026-11-26-i7-8086k-arc-a770-layout/README.md)). So Phase D's fusion
target stays as written, but the per-matvec kernel has a bigger headroom than this file first assumed, and it
is on the arithmetic side.

## 6. Build order (cheapest first, by dependency depth and reuse)

### Phase A - first honest 27B token (2-3 d)
1. ~~**K1 `norm_fold`**~~ - **done 2026-11-26**: `kindi/intel_layer.py` (`kindi intel-norm`), checked
   against a numpy transcription of `kcpu.c`'s `rmsfold()`. One call is 43.09 us at 2,048 wide and
   45.99 / 45.50 / 54.33 us at the 27B's 5,120 / 6,144 / 17,408 as two enqueues; 193 passes would be
   9.38 ms/token, of which the 10.99 us enqueue floor is ~4.2 ms - so the shipped shape has to fuse
   the fold into the layer's launch, not call it twice. The driver has no fp64, so the CPU's double
   sum is a two-float pair: the scale is bit-equal in 8 of 12 (2,048) and 10 of 12 (6,144) random
   vectors and the folded vector moves by at most 2.86e-06 / 7.15e-07 when it is not, which is inside
   this file's 1e-5 bar - the Arc gate is `kindi verify`, not bit-equality. Page:
   [norm+fold](benchmarks/2026-11-26-i7-8086k-arc-a770-norm-fold/README.md), claim
   `arc-a770-norm-fold-20261126`.
2. ~~**K2 `embed_row`**~~ - **closed 2026-11-26**: one row is 544 B (16 phases x 32 bytes + one fp16
   scale per phase), a host read beside the token id; its fold is K1's second kernel with unit weight
   and unit scale, 27.99 us and 0 differing elements against `kcpu.c`'s embed.
3. **K3 wiring** - a layer loop with a resident pack, 1-3 matrices per launch (`pack_gemv_persist` already
   takes `nmats`), host does conv/delta/attention/softmax/silu in numpy.

   *Milestone: a real 27B decode, measured. Around 50 ms/token expected.*

### Phase B - take the 48 GDN layers off the host (4-5 d)
4. **K4 `ewise`**, 5. **K5 `ssm_gates`**, 6. **K6 `conv1d_silu`**, 7. **K7 `l2norm_heads`**,
   8. **K8 `gdn_delta`** (the only hard one: 128x128 state per head, 48 workgroups, M in global with 32x128
   LDS tiles because 64 KB of LDS does not fit alongside the rest).

   *Milestone: no host math for 48 of 64 layers; around 40 ms expected.*

### Phase C - attention (3 d)
9. **K9 `qk_norm_rope`** (+ KV append), 10. **K10 `attn_decode`** (start at ctx 2,048; two-pass softmax in
   LDS is fine at that width), then widen to 32k with online softmax.

   *Milestone: all 64 layers on device; host sees only the token id.*

### Phase D - head and fusion (2-3 d)
11. **K11 `argmax`**, then fuse: one launch per layer (norm -> 4-7 matvecs -> ewise), keep `h`, `xn`, `xp`
    device resident, kill the per-matrix launch tax.

   *Milestone: 30-33 tok/s, the realizable ceiling.*

Out of scope here: prefill/batching (needs a batched ternary GEMV; the existing q8 GEMM does not apply to pair codes). Roadmap CP3 (2026-11-26) prototyped the CPU-side batched GEMV's byte-lane
formulations (shared sign extraction, T-row lanes, group-outer order, both): all bit-identical, all slower
or no better, and the 9B engine's 5-row pass still costs 5.02 plain rows against the <=1.5 gate, so no
batched GEMV lands on this host (evidence `benchmarks/2026-11-26-i7-8086k-cpu-batch-byte-lane/`, claim
`cpu-batch-byte-lane-20261126`). The Arc token loop stays 401 dependent launches, and the MTP draft head
(not on the decode path) stays out too.

**Why this order and not "attention first":** K1 is called 193 times a token and is a dependency of every
matvec; attention is called 16 times and is worth about 0.8 GFLOP at ctx 2048 - 3% of the token. GDN comes
before attention because it is 48 layers and its cost is one 128x128 matvec per head, which is the same shape
family the GEMV kernel already handles. State-free kernels first, stateful ones last, because state needs
allocation, reset and a context loop that you do not want to debug while also debugging a new reduction.

## 7. Toolchain: none of the three paths is needed for this plan

Every existing probe compiles OpenCL C **at runtime through the driver** (`clBuildProgram`), so kernels can be
written and measured today with no clang, no MSVC and no oneAPI.

- `S:\oneAPI` + `T:\VisualStudio` are needed only for the **SYCL** route (`icpx` + `vcvars`), which the
  roadmap's M3 wants because the support target is Linux. `intel_sycl.py` already finds them; treat SYCL as a
  port of the same C99 kernels, not as a prerequisite.
- `C:\Program Files\clang+llvm-18.1.8` is used by nothing in the current Arc path.

So the fastest route to a benchmark is to extend the working OpenCL harness, then port for the supported
backend - not to stand up SYCL first.

## 8. Verification

- **Oracle:** the CPU engine. `kindi verify` has references for 2B/4B/9B (40/40 and 13/13) and none for 2.5
  27B, so for the 27B compare greedy tokens on one prompt, Arc against `KINDI_DEVICE=cpu`, and say so honestly.
- **Build the first kernels against the 2B pack** (1.1 GB, verifiable in seconds). A wrong norm or fold shows
  up as a token diff immediately instead of inside a 30 s 27B layer.
- **Per-kernel exactness:** the ternary matvec is already bit-exact against the host pair-table reference.
  Keep a host numpy reference for every new kernel - tolerance 0 for matvec/argmax, about 1e-5 for the
  FWHT/norm chain (fp32 reassociation only).
- Record ReBAR state, PCIe link and clock with every number, as the pack page already does.

## 9. Risks

- **OpenCL 1.2 on this driver**: no subgroups, no fp16 arithmetic (store KV as `ushort`, convert in-kernel, as
  the CPU does), 64 KB LDS per workgroup (K8 must tile).
- **Memory compression**: zero-filled weights read up to 2.4x faster; benchmark real pack bytes only.
- **Launch tax**: 8.78 us per enqueue, about 22 ms of ramp/idle across 401 small launches. Fusion is where the
  last 40% lives, not in a faster matvec.
- **ReBAR off** invalidates comparisons; it is a required field in the report, not a footnote.

The A/B harness for the blocked layout is `bench-logs/arc_layout_probe.py`, which is not tracked, as with the
overhead sweep's harness.
