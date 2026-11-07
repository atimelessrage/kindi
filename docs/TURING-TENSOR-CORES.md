# Turing tensor-core support for Kindi

This document is the as-built description of the Turing tensor-core backend for
NVIDIA Turing cards (compute capability 7.5, `sm_75`), such as the Quadro RTX
4000. It adds `KINDI_TC=1` support for these cards without changing the Ampere
path (`kindi/tc.cu`, `kindi/tc.py`) or the non-tensor 5-trit path.

Status: the code is complete, its numerics are verified against the Ampere path
on a newer card, and the `sm_75` PTX assembles in the driver's JIT. A run on
Turing hardware is still pending (see [Verification](#verification)); the
installer's `turing` profile is the opt-in and `KINDI_TC=0` remains the
non-tensor fallback.

## What was added

New files:

```text
kindi/tc_turing.cu    Turing int8 tensor-core products (decode and prompt)
kindi/tc_turing.py    the loader: supported(), load_module(), TCTuringKernels
kindi/tca_turing.cu   Turing prompt-chunk attention (int8 scores, fp16 P V)
```

Modified files:

```text
kindi/spec.py         _tc_backend() picks Ampere, then Turing; SpecKernels
                      compiles the Turing sources and loads compute_75 PTX
install.sh            card class "turing" (cc 7.5, >= 7.5 GB) and its profile
docs/CARDS.md         the RTX 4000 profile
README.md             the supported-cards row
```

`kindi/chunk.py` and `kindi/tp.py` needed no changes: the Turing module keeps
the existing `tc` weight layout, kernel names, signatures, grids and buffer
sizes, so the plans, `gemm_tc` and the attention gate work unchanged.

## Selection

`spec.install()` is installed only when `KINDI_TC=1` (`spec.wanted()`). When the
engine loads a card it calls `spec._tc_backend(card)`:

```python
def _tc_backend(card):
    if _tc.supported(card):          # major >= 8: Ampere and later
        return _tc
    if _tc_turing.supported(card):   # exactly 7.5
        return _tc_turing
    return None
```

`tc_turing.supported()` requires `major == 7 and minor == 5` **and** a
tensor-core Turing device name (TU102/TU104/TU106: RTX 20, Titan RTX, Quadro
RTX, Tesla T4, CMP 40HX). Volta (7.0) is excluded because the int8
`m8n8k16` MMA needs `sm_75`; Ada/Hopper/Blackwell keep the Ampere `m16n8k32`
path. The other `sm_75` parts - TU116/TU117 (GTX 16 series, Quadro
T400/T600/T1000/T2000, MX4xx/MX5xx, CMP 30HX) - have no tensor cores, and an
`mma` instruction on them traps. Compute capability cannot tell the chips
apart, so the name decides; an unknown `sm_75` name stays on the plain engine.
If no backend applies, the existing 5-trit path runs, so `KINDI_TC=1` on an
older or tensor-core-less card is harmless.

The backend is used only when the model is the one-card 5-trit (`g5l`) path; the
same condition already guards the Ampere backend.

## Instruction

Ampere uses `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. Turing does not
have that shape, so the Turing module uses its int8 IMMA:

```cuda
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
```

Per lane it takes one `.b32` for A (four int8), one `.b32` for B (four int8) and
two `.s32` for C/D. Fragment layouts (PTX ISA, `m8n8k16`):

```text
A: row = lane >> 2,              col = 4 * (lane & 3) + i
B: col = lane >> 2,              row = 4 * (lane & 3) + i
C: row = lane >> 2,              col = 2 * (lane & 3) + i
```

A warp still owns a 16-row tile: two `m8n8k16` MMAs cover the two 8-row halves
(`rowsel` 0 and 1), and eight k-blocks of 16 cover the 128-column group. That is
16 MMAs per group per warp tile instead of Ampere's 4.

## Weight layout: unchanged

The Turing module keeps the existing `tc` packed layout, `k_tc_pack` and
`repack()` byte-for-byte. This is possible because one Turing A register is
exactly one Ampere A register:

- Ampere register `rho` of a lane belongs to column group `G = rho >> 4`, k-step
  `s = (rho >> 2) & 3` and fragment `f = rho & 3`; fragment `(s, f)` covers rows
  `g8 + 8 * (f & 1)` and columns `32 s + 16 (f >> 1) + 4 t + byte`.
- Turing k-block `c` covers columns `16 c + 4 t + byte`. So the Ampere fragment
  with `f = 2 * (c & 1) + rowsel` covers the same columns, and its register
  number is `16 G + 2 c + rowsel`.
- Therefore the Turing A register for k-block `c` and 8-row half `rowsel` is
  `tc_r<16 * G + 2 * c + rowsel>` - the same unpacking helper and the same
  packed words as Ampere.
- The Turing B word for k-block `c` is the staged activation word for k-step
  `s = c >> 1`, half `h = c & 1`, i.e. word `2 s + h = c` of the uint4 pair. The
  words are consumed in the order `b0.x, b0.y, b0.z, b0.w, b1.x .. b1.w`.
- The D pair `(c0, c1)` of a Turing MMA is row `g8 + 8 * rowsel`, tokens
  `2 t` and `2 t + 1`, so the results land in the same `acc[0 .. 3]` as the
  Ampere path.

`repack()` still requires `R % 16 == 0` and `C % 1024 == 0` and writes
`R * (C / 128) * 7` words. Python grids (`R // 16` tiles of 16 rows), buffer
sizes and the tuning rules are unchanged.

Because every MMA is an exact integer sum of the same 16 products, and the
per-token scale and group-sum correction are applied in the same order in fp32,
a token's result is bit-identical to the Ampere path and to the 5-trit path.
The extra MMAs change speed, not arithmetic.

## Shared-memory staging: no cp.async

`cp.async` is a PTX ISA 7.0 / `sm_80` instruction. It is not available on
Turing, and PTX that contains it cannot be loaded for `.target sm_75`. The
Turing module therefore stages with ordinary loads and stores:

```cuda
__device__ __forceinline__ void tc_cp16(void* s, const void* g){ *(uint4*)s = __ldg((const uint4*)g); }
__device__ __forceinline__ void tc_cp4(void* s, const void* g){ *(unsigned int*)s = __ldg((const unsigned int*)g); }
__device__ __forceinline__ void tc_commit(){}
__device__ __forceinline__ void tc_wait1(){}
__device__ __forceinline__ void tc_wait0(){}
```

The double-buffered loop, the `__syncthreads()` placement and the shared-memory
structures (`TCS`, `TCM`) are the same as in `tc.cu`, so the staged data is
visible at the same point. The only cost is that a block waits for the copy
instead of overlapping it; the layout is chosen so lanes read consecutive
16-byte words (no bank conflicts).

Shared-memory sizes stay under Turing's 48 KB static limit:

```text
TCS (decode)   16.9 KB   (uint4 b[2][8][2][8][4] + float xs[2][8][8])
TCM (k_tc_mm2) 16.9 KB   (uint4 b[2][2][64][4] + float xs[2][64])
tca_turing     38.0 KB   (2 x 32 x 272 + 2 x 32 x 8 halves + 256 x 40 halves)
```

## Decode path

`tc_turing.cu` provides the same names and signatures as `tc.cu`:

```text
k_tc_pack, k_tc_q8
k_tc_mv2, k_tc_mv4, k_tc_mv8
k_tc_mv3_2, k_tc_mv3_4, k_tc_mv3_8
k_tc_sum
k_tc_mm4, k_tc_mm8, k_tc_mm16
k_tc_mm2
```

The decode kernels keep one 16-row tile per warp and the same unit of eight
column groups. `k_tc_mv*` splits are the same. `k_tc_q8` and `k_tc_sum` are
identical jobs; `k_tc_q8` writes the same int8 layout, scales and integer sums.

## Prompt path

`k_tc_mm*` runs `NB` blocks of 8 tokens against one unpacked set of A
fragments, and `k_tc_mm2` stages 64 tokens per block in shared memory. Both
write the same `Y[n * ldy + r]` layout. With `m8n8k16`, one MMA covers eight
tokens, so a "block of 8 tokens" is one MMA column set rather than half of an
Ampere 8-token tile; the loop order over groups and token blocks is preserved.

`k_tc_mm2` keeps `__launch_bounds__(256, 2)` and the same grid as Ampere. If the
shared-memory size limits occupancy to one block on a particular Turing part,
the kernel still runs; only occupancy changes.

## Attention

`kindi/spec.py` concatenates `tca_turing.cu` instead of `tca.cu` when the Turing
backend is selected. `k_att_tc_T` has the interface and job of
`k_att_flash_T`, and the same online softmax, masking, staging, epilogue and
output layout. Only the two MMA shapes and the f16 pair packing change.

The score MMA:

- a 32-value q8_0 K block is two k-blocks of 16, and a 16-row group is two
  8-row halves;
- the four A registers are `qa[b][0..3]` in the order (k-block, half) =
  `(0,0), (0,1), (1,0), (1,1)`;
- the B words are `b0` (values 0..15) and `b1` (values 16..31);
- the four results come back as `lo[0], lo[1], hi[0], hi[1]`.

The value product cannot use Ampere's shape: ptxas rejects `mma.sync.aligned.m16n8k16` (with either fp32 or fp16
accumulation) and `m8n8k4` for `.target sm_75`. Turing's widest f16 MMA is the 8-wide k:

```cuda
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

So a 32-value probability tile is four k-steps of 8. The A registers of step `nt` are `pa[nt][0]` (rows `g8`) and
`pa[nt][1]` (rows `g8 + 8`), whose columns `2t, 2t + 1` are the position pair the mask and softmax already use; B is
one word `VT[f][nt * 8 + 2 t]`. The C fragment (rows `g8` / `g8 + 8`, features `2t`, `2t + 1`) is the same shape, so
`o[16][4]` and the epilogue are unchanged.

Turing also lacks the two-source `cvt.rn.f16x2.f32` (PTX reports it as `sm_80`+), so `atc_h2` converts each float with
`cvt.rn.f16.f32` and packs the pair with `mov.b32`, the same round-to-nearest-even result.

The K tile is staged with ordinary loads (`tc_cp16`/`tc_cp8`), ordered by the same barriers.

`k_att_tc_T` is fetched unconditionally for CUDA in `SpecKernels`; both
`tca.cu` and `tca_turing.cu` define it, so `chunk.py`'s
`KINDI_ATT_TC=1` gate works for either backend.

## Compilation and loading

`tc_turing.load_module()` compiles the concatenated sources with NVRTC
`-arch=compute_75` and loads the resulting PTX as produced:

- the image's NVRTC (CUDA 10.2) targets `compute_75` natively and emits PTX ISA
  6.5, the first version that knows `mma.m8n8k16`, so no header patching is
  needed (the Ampere module still raises `.version` to 7.1 and retargets to
  `sm_8x`);
- `load_module()` raises `.version` to 6.5 only if some other NVRTC emits an
  older one;
- the PTX target is `sm_75`, which matches the card;
- the driver JIT compiles the PTX for the actual Turing part and caches the
  cubin in the driver's JIT cache (`CUDA_CACHE_PATH`), as it already does for the
  Ampere path.

Inline PTX is passed through NVRTC unchecked, so a CUDA 10.2 NVRTC accepts the
`m8n8k16` and `m16n8k8` text even though that toolkit does not emit the
instructions itself. The driver JIT then assembles it.

## Installer

`install.sh` adds a `turing` card class:

```sh
card_tc() {       # 0 for TU102/TU104/TU106; 1 for TU116/TU117 and the like
  case "$name" in *"GTX 16"*|*"QUADRO T4"*|*"QUADRO T6"*|*"QUADRO T10"*|*"QUADRO T20"*|*"CMP 30HX"*|*MX*) return 1 ;; esac
  return 0
}
...
elif [ "$c" -eq 75 ] && [ "$V" -ge 7500 ] && card_tc "$k"; then echo turing
```

It is checked after `tc` and before `pascal8`, so tensor-core Turing cards of
7.5 GB or more (such as the 8 GB RTX 4000) take the Turing profile instead of
the Pascal 8 GB profile. The name check keeps the tensor-core-less `sm_75` parts
(GTX 16 series and the like) on the plain profile, where `KINDI_TC=1` is never
set. The profile (`profile_of`) sets:

```text
KINDI_CARDS=1, KINDI_CTX=65536 (131072 on 11 GB and more), KINDI_MAX_TOKENS=65536
KINDI_TC=1, KINDI_FD=1, KINDI_ATT_TC=1, KINDI_TC_TMAX=4
KINDI_CKPT_VRAM=1, KINDI_MEM_RESERVE_MB=160 (11 GB and more only)
KINDI_CHECKPOINT_BUDGET_MB=1280, KINDI_PARK_MAX=1, KINDI_DENSE_SPACING=2048, KINDI_PREFIX_MAX=2
```

The tensor-core settings apply to every model size, not only the 27B as in the
`tc` profile: the Turing kernels are size-independent and numerically identical,
and on an 8 GB card the 2B / 4B / 9B are common choices. The 8 GB cards keep the
engine's default memory reserve (256 MB) and a 65,536-token context, like the
GTX 1080 profile.

For the 27B it also enables the 2B draft model (`KINDI_SPEC=draft2b`,
`KINDI_SPEC_K=3`, `KINDI_SPEC_PACK=/data/pack-2b`, `KINDI_SPEC_WINDOW=8192`) and
adds the 2B pack, as the RTX 3080 profile does. Single-card servers keep
`KINDI_BLOCKING_SYNC=0` (the installer's rule for every profile except `tc`).

`KINDI_CHUNK_Q8` stays at its default (on for `sm_61` and later), so prompt
products use int8 activations on Turing too.

## Turning it off

- `KINDI_TC=0` (or unset) runs the 5-trit path; the Turing sources are never
  compiled or loaded.
- `KINDI_ATT_TC=0` keeps `KINDI_TC=1` but uses the non-tensor-core prompt
  attention.
- `KINDI_TC=0` on a card where the backend is not wanted makes `spec.install()`
  a no-op.

If verification fails, setting `KINDI_TC=0` in the container is enough to return
to the existing path; no source change is required.

## Verification

The full record, the exact commands and the pending sm_75 checks are in
[VERIFICATION.md](VERIFICATION.md). In short: the int8 products are exact integer
sums, so the Turing kernels can be checked without a Turing card. An independent
NumPy model of the layout and arithmetic was compared with `SpecKernels(
backend=tc_turing)` (built for `compute_75` and JIT-compiled by the sm_89 driver),
and the same packed weights, quantized activations, scales and integer sums were
fed to `tc.cu` and to `tc_turing.cu` and compared byte for byte.

Done locally (RTX 4060 Ti, driver 13.4, no Turing GPU):

- `k_tc_pack`, `k_tc_q8` (words, scales, sums) identical;
- `k_tc_mv2/4/8` for 1, 2, 3, 4, 5, 7 and 8 tokens, splits 1, 2 and 3 identical;
- `k_tc_mv3_2/4/8` identical;
- `k_tc_mm4/8/16` for 8, 17, 32, 40 and 64 tokens identical;
- `k_tc_mm2` for 7 and 64 tokens identical;
- `k_tc_sum` identical;
- `k_att_tc_T` identical (att and the `Z > 1` partials) for six shapes, including
  a 300-position context and `Z = 3`;
- the full `SpecKernels` source compiles for `compute_75` **and** the `sm_75` PTX
  assembles in the driver's JIT (the check a real Turing card performs) - this is
  what caught the `m16n8k16` / `cvt.f16x2.f32` features, which the first version
  of `tca_turing.cu` used;
- `.version 6.5` (CUDA 10.2's PTX) with `.target sm_75` assembles, so the image's
  NVRTC output needs no patching;
- `tc_turing.py` and `spec.py` pass `py_compile`;
- `install.sh` passes `bash -n`;
- the CPU-reference suite ([tests/test_tc_turing_reference.py](../tests/test_tc_turing_reference.py),
  9 tests) passes: 180/180 vector products, 80/80 token counts, 27/27
  shapes and merges, and the whole chain is bit-identical to the fused model
  768/768 for split 1 and split 4;
- cppcheck and clang-tidy (CUDA device mode, `sm_75`) report no defect in
  `tc_turing.cu` or `tca_turing.cu`; see [VERIFICATION.md](VERIFICATION.md).

Pending on `sm_75` hardware (RTX 4000 or Quadro RTX 4000):

1. `KINDI_TC=1 kindi verify` must match the reference output token for token.
2. `KINDI_TC=1 kindi bench` must beat the 5-trit path on generation and prompt
   reading.
3. Long-context tests (2K, 8K, 16K, 24K, 64K) must stay stable.
4. The Ampere path must still pass `kindi verify` unchanged.

Until then, treat the Turing numbers as unmeasured.

## Risks

- The driver JIT may reject the compute_75 PTX on some driver versions even
  though NVRTC accepts it; a later driver is the workaround.
- The extra 16-MMA-per-group loop may be limited by instruction issue rather
  than tensor-core throughput, reducing the speedup on the smallest Turing
  parts.
- No `cp.async` means the staging is synchronous; the double buffer hides the
  latency only partially.
- The backend selects on the device name as well as compute capability, because
  TU116/TU117 has no tensor cores. An sm_75 card whose driver reports an
  unexpected name stays on the plain engine (safe, but no tensor cores) until
  the name is added to `tc_turing._TENSOR`.
- The prompt attention value product uses `m16n8k8` where Ampere uses
  `m16n8k16`. Both are fp16 products with fp32 accumulation, but the association
  of the 32-term sum differs in principle. The local comparison came out
  bit-identical for every shape tested; if a long-context case ever disagrees,
  it differs only in the last bits, the same class of difference that
  `KINDI_ATT_TC` itself introduces against the non-tensor path.
