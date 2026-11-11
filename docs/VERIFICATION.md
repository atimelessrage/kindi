# Verification of the Turing tensor-core backend

This page records what has been checked for the Turing (`sm_75`) tensor-core backend
([TURING-TENSOR-CORES.md](TURING-TENSOR-CORES.md)), with the exact commands, and what still has to run on real
`sm_75` hardware. It was written on 2026-11-07 against the 0.4.0 Turing merge; it is retained as a
historical, Turing-specific record and is not the general verification contract for every backend or card.

**Status:** the arithmetic is verified against an independent CPU model, and the sources pass static analysis; the
runtime checks on a Turing card are still pending (no `sm_75` card was available).

| item | result |
|---|---|
| CPU-reference suite (`tests/test_tc_turing_reference.py`) | 9/9 tests pass on an RTX 4060 Ti (sm_89; the compute_75 PTX is JIT-compiled) |
| int8 products vs the independent model | vector 180/180, token counts 80/80, shapes/merges 27/27, `k_tc_sum`, `k_tc_q8`, `k_tc_pack` all exact |
| whole decode chain | bit-identical to the fused fp32 emission 768/768 (split 1) and 768/768 (split 4) |
| cppcheck 2.22.0 | only reports shared with the Ampere / 0.4 baseline (`uninitvar` false positives in `kindi.cu`, `tc_turing.cu`/`tc.cu`, `tcf.cu`, plus cast style notes) |
| clang-tidy 18.1.8, CUDA device mode, `sm_75` | parses the full Turing and Ampere translation units; only the two baseline `kindi.cu` dead-store reports |
| `kindi verify` / `kindi bench` on `sm_75` | **pending** |

## Why a CPU model is a valid check

Every Turing product is an exact integer sum: the A operand is the trit (0..2), the B operand is the quantized
activation, and the MMA accumulates `sum((trit - 1) * q)` in int32. The scale `XS * d` is applied in fp32 afterwards.
There is no instruction-level rounding inside the product, so an independent integer model can predict the result
exactly. The only floating-point association that matters is the final split accumulation, and the model reproduces
the grouping of `k_tc_mv` + `k_tc_sum`.

The NumPy model in `tests/test_tc_turing_reference.py` is written from the documented format, not transcribed from
the CUDA: it rebuilds the packed bytes from the base-3 definition, decodes the activations from
`q = rn(x * 127 / amax)`, and forms the products as Python-level integer sums. A typo in either implementation changes
one of the byte-exact comparisons.

## CPU-reference suite

File: `tests/test_tc_turing_reference.py`. It builds the kernels through the production path,
`kindi.spec.SpecKernels(backend=tc_turing)`, so `tc_turing.load_module()`, the generated token-batched wrappers, the
`tc` layout and the kernel signatures are all exercised.

```sh
# from the repository root; numpy and cupy (with any CUDA device, sm_75 or newer) are required
PYTHONPATH=. python -m pytest tests/test_tc_turing_reference.py -q
PYTHONPATH=. python tests/test_tc_turing_reference.py     # same tests, per-test summary
```

Result on the verification machine (RTX 4060 Ti, driver 13.4, Python 3.13, NumPy 2.5.1, CuPy 14.2.0):

```text
device: b'NVIDIA GeForce RTX 4060 Ti' (sm_89; the Turing module is compute_75 PTX, JIT-compiled)
ok     test_turing_names
ok     test_reference_layout
ok     test_activations
k_tc_pack: every data byte decodes to a base-3 number below 243 and every scale is the fp16 scale
ok     test_pack_layout
vector products: 180/180 cases match the model
ok     test_products
token counts: 80/80 cases match the model
ok     test_token_counts
shapes and merged products: 27/27 cases match the model
ok     test_shapes_and_merges
ok     test_sum
whole chain sp1: bit-identical to the fused emission 768/768, to the rounded one 231/768, max err 9.13e-07
whole chain sp4: bit-identical to the fused emission 768/768, to the rounded one 315/768, max err 4.23e-07
ok     test_whole_chain
all tests passed
```

What each test covers:

| test | coverage |
|---|---|
| `test_turing_names` | `tc_turing.supported()` selects TU102/TU104/TU106 names and rejects TU116/TU117 (GTX 16, Quadro T400/T600/T1000/T2000, MX4xx/MX5xx, CMP 30HX) |
| `test_reference_layout` | the model's own 5-trit encoder/decoder round-trips |
| `test_activations` | `k_tc_q8`: XQ, XS, XA for round-half-even ties, normal, huge and all-zero inputs |
| `test_pack_layout` | `k_tc_pack`: every byte decodes below 243, every scale is the fp16 scale, end to end |
| `test_products` | `k_tc_mv2/4/8` and `k_tc_mv3_2/4/8`, 1..8 tokens, 1..3 column splits, several units, short tiles, strided rows, all-zero/all-two weights |
| `test_token_counts` | odd token counts and every plan size 1..8 |
| `test_shapes_and_merges` | `k_tc_mm2/4/8/16` and the merge kernels for 8..64 tokens |
| `test_sum` | `k_tc_sum` split order and rounding |
| `test_whole_chain` | decode chain end to end, split 1 and split 4, compared with the fused and separately-rounded fp32 emissions |

The whole-chain result is the one that matters for the requirement "token-identical to the non-tensor-core path": the
GPU output is bit-identical to the fused emission in all 768 elements for both split counts. The separately-rounded
variant is *not* expected to match (231/768 and 315/768): nvcc contracts `acc += corr * (xs * d)` into an `fma`, and
the model uses the same contraction.

## Static analysis

The verification machines have no CUDA toolkit and no CUDA headers, so the kernels are checked as source. The helper
`tests/static/cuda_stub.h` provides the execution-space qualifiers, vector types, math/warp builtins and the inline
PTX constraints; the CUDA sources themselves are unchanged.

```sh
python tests/static/run_static_analysis.py              # Turing sm_75 and Ampere sm_86
python tests/static/run_static_analysis.py --only turing
python tests/static/run_static_analysis.py --keep out/  # keep the generated TUs
```

The script builds the same concatenation `SpecKernels` compiles: `kindi.cu` + generated wrappers + `spec.cu` +
`tc_turing.cu` + `tcf.cu` + `tca_turing.cu` (and the Ampere equivalent), then runs:

```sh
cppcheck --enable=warning,style,performance,portability --language=c++ --std=c++14 \
    --suppress=missingIncludeSystem --inline-suppr --include=tests/static/cuda_stub.h \
    --template='{file}:{line}: {severity}: {message} [{id}]' TU.cu

clang-tidy --checks='-*,clang-analyzer-*' --header-filter='^$' \
    --extra-arg=-x --extra-arg=cuda --extra-arg=--cuda-device-only --extra-arg=--cuda-gpu-arch=sm_75 \
    --extra-arg=-nocudainc --extra-arg=-nocudalib --extra-arg=-std=c++14 \
    --extra-arg=-include --extra-arg=tests/static/cuda_stub.h TU.cu
```

Tools: Cppcheck 2.22.0 and clang/LLVM 18.1.8 on Windows. `--cuda-device-only` makes clang-tidy parse the device pass
for `nvptx64`, so the inline-PTX constraints (`"=f"`, `"h"`, `"r"`, ...) are the ones that actually apply;
`-nocudainc -nocudalib` keep it independent of NVIDIA's headers and libdevice.

Findings on 2026-11-07:

- **clang-analyzer:** two dead-store reports in the TU, both the same pattern in `kindi.cu`
  (`const int CW = C >> 4, NPH = C >> 9;` at TU lines 139 and 234), present in the Turing and the Ampere TU alike.
  No report in `tc_turing.cu` or `tca_turing.cu`.
- **cppcheck:** 21 `uninitvar` / 1 `legacyUninitvar` / 2 `rw` reports, identical in the Turing and the Ampere TU,
  so they are baseline, not Turing regressions:
  - `bscale`, `pmax`, `lo_s`/`hi_s`, `ok`, `ep` and `scl` in `kindi.cu`, the generated wrappers and `spec.cu`:
    values written through inline PTX or by one lane, which cppcheck cannot follow;
  - `legacyUninitvar` `wm` in `tc_turing.cu` / `tc.cu` (lane 0 writes `wm[w]`, all lanes read it after
    `__syncthreads()`; the same code in both);
  - `rw` on `q.y[rw]` in `kindi.cu` (a store that feeds the next loop iteration).
  The Turing files add only `cstyleCast` pointer-cast style notes (5 in the Turing TU vs 2 in Ampere: two in
  `tc_turing.cu` and one in `tca_turing.cu`); no `warning`, `performance` or `portability` defect.
- **compilation:** both full TUs parse in CUDA device mode for their targets (`sm_75`, `sm_86`) with no compiler
  error (the first run of this check caught a missing `rintf` in the stub, not in the kernels).

Apart from those shared false positives and style notes there is no finding in `tc_turing.cu` or `tca_turing.cu`;
they are not Turing defects. The static check proves that the code parses and that the analysers find nothing new;
it does not prove the numerics - that is the CPU-reference suite's job.

## Register and local-memory footprint

Measured from the compiled kernels (`cuFuncGetAttribute`):

| kernel | registers / thread | local memory / thread |
|---|---|---|
| `k_tc_mm16` (Turing) | 255 | 304 B |
| `k_tc_mm16` (Ampere, reference) | 255 | 464 B |

Both kernels are at the architectural register limit (255), so occupancy is low on both paths; Turing spills less
(304 B vs 464 B). This is one of the reasons the Turing numbers are expected to be below Ampere's, and it does not
affect correctness. The real `sm_75` run should confirm the numbers with the image's CUDA 10.2 NVRTC (the local check
used NVRTC 12.9). The other kernels are well under the limit, for example `k_tc_mv8` and `k_tc_mm8` at 80 and 255
registers, and `k_tc_mm2` at 128.

## Tools available locally

| tool | status |
|---|---|
| Python 3.13, NumPy 2.5.1, pytest 9.1.1 | available |
| CuPy 14.2.0 (CUDA 12.x), cuda-python + NVRTC 12.9 | available |
| Cppcheck 2.22.0 | available |
| clang / clang-tidy 18.1.8 | available |
| `nvcc`, `compute-sanitizer`, `nsys` | **not available** on the Windows host |
| WSL2 Ubuntu | Python and `nvidia-smi` only; **no CUDA toolkit**, so memcheck / initcheck / synccheck cannot run there |

Consequences: the kernels can be compiled (NVRTC, from CuPy) and executed (the sm_89 card JITs the compute_75 PTX),
but they cannot be sanitized locally. `compute-sanitizer --tool memcheck|initcheck|synccheck` is part of the pending
`sm_75` checklist.

## Pending on sm_75 hardware

Run these on an RTX 4000 / Quadro RTX 4000 (or another TU102/TU104/TU106 card) with the CUDA 10.2 image:

1. `KINDI_TC=1 kindi verify` must produce the reference output token for token; also run the same pack with
   `KINDI_TC=0` and compare the greedy output files byte for byte.
2. `KINDI_TC=1 kindi bench` for generation and prompt reading, compared with `KINDI_TC=0` and with the numbers in
   [CARDS.md](CARDS.md).
3. Long-context tests at 2K, 8K, 16K, 24K and 64K tokens: stable output, no restart, flat memory.
4. The Ampere path must still pass `kindi verify` unchanged (the merge touched `spec.py`, `cache.py` and
   `install.sh`).
5. `compute-sanitizer --tool memcheck` and `--tool initcheck` on a decode and a prompt pass.
6. Confirm the register/local-memory numbers above with the image's NVRTC; if `k_tc_mm16` spills more than the
   304 B seen locally, lower `KINDI_TC_MM_NB` or switch to `k_tc_mm2`.

Until these pass, treat the Turing speed numbers as unmeasured (the arithmetic already has the local evidence above).

## Reproducing the checks

```sh
# reference suite (needs a CUDA device with a driver new enough for compute_75 PTX)
PYTHONPATH=. python tests/test_tc_turing_reference.py

# static analysis (needs cppcheck and/or clang-tidy)
python tests/static/run_static_analysis.py

# installer syntax (either shell has no bash on Windows, so convert CRLF first or run it on Linux/WSL)
bash -n install.sh

# the name -> profile matrix used by the new Turing checks (see docs/CARDS.md)
#   RTX 2060 / Titan RTX / Quadro RTX / T4 / CMP 40HX -> turing
#   GTX 16 / Quadro T400-T2000 / MX4xx-MX5xx / CMP 30HX -> maxwell (plain engine)
```

See also [tests/README.md](../tests/README.md) for the short test guide.
