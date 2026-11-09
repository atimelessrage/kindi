# Tests

These tests check the Turing (`sm_75`) tensor-core backend without a Turing GPU. The full verification record is in
[docs/VERIFICATION.md](../docs/VERIFICATION.md); this page is the short guide.

| file | what it does |
|---|---|
| `test_tc_turing_reference.py` | Compares the Turing kernels with an independent NumPy model: the packed weight layout, the int8 activations, every product kernel, the merge/sum kernels and the whole decode chain. |
| `static/run_static_analysis.py` | Builds the full CUDA translation units that `kindi.spec.SpecKernels` compiles (Turing and Ampere) and runs cppcheck and clang-tidy on them. |
| `static/cuda_stub.h` | The minimal CUDA declarations (execution-space qualifiers, vector types, builtins) that let the analysers parse the kernels without a CUDA toolkit. Never compiled into Kindi. |

## Requirements

- Python 3 with `numpy` and `cupy` (the suite needs a CUDA device; any sm_75+ card works because the Turing module is
  `compute_75` PTX that the driver JIT-compiles, and the suite is run on an RTX 4060 Ti / sm_89).
- `pytest` for the pytest entry point; `python tests/test_tc_turing_reference.py` also runs without pytest.
- `cppcheck` and/or `clang-tidy` for the static script (found on `PATH` or set with `CPPCHECK` / `CLANG_TIDY`).

## Run the reference suite

```sh
PYTHONPATH=. python -m pytest tests/test_tc_turing_reference.py -q
PYTHONPATH=. python tests/test_tc_turing_reference.py      # same, with the per-test summary
```

It builds the kernels through the production path (`kindi.spec.SpecKernels(backend=tc_turing)`), so the loader and
the full translation unit are exercised too. All 9 tests passed on 2026-11-07 (sm_89).

## Why the reference is independent

The NumPy model does not call or transcribe the CUDA code. It works from the documented format:

- the `tc` packed layout is rebuilt byte by byte from the base-3 trit definition (`b = ceil(256 v / 243)`, scales in
  word 26/27, group `G = 2t + (wk - 26)`, row `g8 + 8 * (bq >> 1)`);
- `k_tc_q8` is re-derived from `q = rn(x * 127 / amax)`, `XS = amax / 127`, `XA = sum(q)`;
- each product is the exact integer sum `sum((trit - 1) * q)` scaled by `XS * d`, with the accumulation split the
  kernels use;
- the whole chain reproduces the split-accumulation grouping of `k_tc_mv` + `k_tc_sum` and the fused `fma` emission.

A typo in either the CUDA or the model changes one of the byte-exact comparisons.

## Run the static analysis

```sh
python tests/static/run_static_analysis.py              # Turing (sm_75) and Ampere (sm_86)
python tests/static/run_static_analysis.py --only turing
python tests/static/run_static_analysis.py --keep out/  # keep the generated translation units
```

The expected findings are the shared baseline ones listed in
[docs/VERIFICATION.md](../docs/VERIFICATION.md#static-analysis): two `clang-analyzer` dead-store reports in
`kindi.cu`, the cppcheck `uninitvar` false positives (same in the Ampere TU) and the C-style-cast style notes. A new
finding in `tc_turing.cu` or `tca_turing.cu` is a regression.

## Not covered here

Runtime checks on a real `sm_75` card (`kindi verify`, `kindi bench`, long contexts) need Turing hardware; they are
listed as pending in [docs/VERIFICATION.md](../docs/VERIFICATION.md).
