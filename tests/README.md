# Tests

The repository has a fast CPU-only gate and an optional CUDA reference suite. Neither is a substitute
for `kindi verify` on the target card, which needs a model pack and a GPU.

## CPU-only gate

Run from the repository root:

```sh
python -m compileall -q kindi
python tools/check_python_quality.py
python -m pytest tests/ -q
# Optional local coverage gate (matches the CPU CI job):
python -m pytest tests/ -q --cov=kindi --cov-branch --cov-report=term --cov-report=json:coverage.json --cov-fail-under=27
python tools/check_coverage.py
bash -n install.sh
```

The Python-quality gate ratchets Ruff 0.16.0 (`F`, `E9`, `B`, `I`), mypy 2.4.0, Pyright 1.1.414
(basic, all platforms) and Pylint 4.1.3 (errors and fatal messages) against
`tools/python_quality_baseline.json`: existing debt is recorded, and only a new or increased finding
fails. Ruff covers the Python tree; the other three cover production code. `--update --reason
"..."` lowers the baseline only after review; it refuses any new or increased finding and records
the reason and delta in the baseline.

The suite uses stub engines and stand-in CuPy where needed. It covers config precedence, the chat
template and tool-call round trips, tokenizer behavior, sampling, cache decisions, the HTTP boundary
(validation, auth, body caps, queue, Ollama endpoints, image rejection), startup pack checks, the
plan-engine guards, the tracked-file secret/personal-data guard and detect-secrets baseline checks, the installer's GRUB fixture and
CPU ISA preflight, and GPU-driver selection. No CUDA device or model pack is required.

## The kernel launch contract is partly measured, and says how much

`test_kernel_arity.py` reads every `__global__` declaration in `kindi/*.cu` (including
`__launch_bounds__` forms and the generated `k_*_T` wrappers) and compares each graph launch's
literal argument tuple with it. Since 2026-11-24 it also holds the launch's **block size** against
the kernel's `__launch_bounds__` ceiling: 28 (site, kernel) pairs are measured and 135 launches are
unresolved (96 kernels declare no bound; the rest name the kernel or the block dynamically), with
both counts pinned. A block larger than the declaration (`k_big asks 1024 > 256`) fails the suite
instead of failing at launch time on a card this host does not have. Current split: **71 call sites checked across 40 kernels** (`spec.py`
26, `dspine.py` 4, `mtp.py` 3, `tp.py` 38) and **71 call sites skipped** because their kernel is a
variable, list element, dict lookup, `getattr` or conditional, or their tuple is assembled at run
time. The skipped ones are pinned in `UNRESOLVED_SITES` with a reason each - a new dynamic launch
fails the suite until somebody classifies it - and the checked count is pinned too, so neither
number can move silently. Running `pytest -s tests/test_kernel_arity.py` prints the per-module
split.

`tests/requirements-test.txt` lists the test dependencies. The same commands run in the CPU job of
`.github/workflows/ci.yml`. A separate `cpu-engine-smoke` job compiles `kindi/kcpu.c` with gcc for
the AVX2/FMA/F16C tier and runs `test_cpu_build.py`, `test_cpu_threads.py` and `test_cpu_smoke.py`;
`test_cpu_build.py` also builds and loads the AVX2/F16C no-FMA and AVX-only fallbacks. The smoke
test loads the library and checks the ctypes ABI without a model pack. The manual
self-hosted GPU lane is described in [docs/CI.md](../docs/CI.md); it is not run by hosted CI.

### Skipped-test inventory (2026-11-24 full CPU run)

`python -m pytest -q -rs` with the Anthropic endpoint (REQ-0028), the selftest's Anthropic check,
the block-size contract (REQ-0023 A2), the Intel q8 GEMM probe and its staged/pinned additions in:
**1,563 passed, 58 skipped, 0 failed** in 195.17 s
(`bench-logs/full-suite-20261124-arc-tiled-final.log`). The 1,560-passed run before it is
`bench-logs/full-suite-20261124-intel-gemm.log`; the 1,551-passed run is
`bench-logs/full-suite-20261124-blocksize.log`. Every skip is one
of ten groups; pytest groups `-rs` output by location, so the count per group is what to watch. A
skip is not a pass: a new group, or a changed count, must be reviewed instead of hidden by the total.

| # | file | count | category and exact reason |
|---:|---|---:|---|
| 1 | `tests/test_gemma4_fp8_mma.py` | 14 | CUDA/cuBLAS absent: `cudaErrorNoDevice: no CUDA-capable device is detected`, and `Failure finding "known cublasLt DLL"` |
| 2 | `tests/test_gemma4_fp8.py` | 11 | CUDA absent: `CUDA is unavailable: cudaErrorNoDevice: no CUDA-capable device is detected` |
| 3 | `tests/test_tc_turing_reference.py` | 10 | `needs a CUDA device for the Turing PTX kernels` |
| 4 | `tests/test_cpu_build.py` | 9 | `neither gcc nor clang is installed` (6), `clang is not installed; set KINDI_CC to ...` (3) |
| 5 | `tests/backends/test_contract.py` | 5 | `<Backend> needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 6 | `tests/test_cpu_avx_fallback.py` | 2 | `set KINDI_CPU_AVX_PACKS` / `set KINDI_CPU_NO_FMA_PACKS to one or more Bonsai pack directories` |
| 7 | `tests/test_cpu_layers.py` | 2 | `no gcc/clang to build the ISA tier libraries`, `set KINDI_CPU_PACK to a Bonsai pack ...` |
| 8 | `tests/test_cpu_smoke.py` | 2 | `set KINDI_CPU_SMOKE=1 with a prebuilt kcpu library` |
| 9 | `tests/test_server_cpu.py` | 2 | `no cached CPU library and no gcc/clang to build it` |
| 10 | `tests/test_calib.py` | 1 | `chmod does not make a dir read-only on Windows` |

58 skips: 35 need a CUDA device, 12 need a C compiler, 10 need a model pack or an explicit opt-in
(`KINDI_CPU_SMOKE`, `KINDI_CC`), and 1 is a Windows filesystem difference. None is evidence that the corresponding GPU, model-pack or
optional-dependency path passed.

#### Per-node baseline (2026-11-19, superseded)

The 28-row table below is the 2026-11-19 run (`3002a40` plus the local changes: **1,312 passed, 28
skipped, 0 failed** in 217.78 s, `bench-logs/full-suite-head-20261119.txt`, with the JUnit/`-rs`
evidence `bench-logs/full-suite-final-20261119.{xml,txt}`). It is kept because the suite roughly
doubled its skip count since (28 -> 58) and the node IDs show which rows are new; regenerate it with
`python -m pytest -q --junitxml=bench-logs/full-suite.xml` when a per-node list is needed.

| # | test | category | exact skip reason |
|---:|---|---|---|
| 1 | `tests/backends/test_contract.py::test_text_primitives_round_trip[BonsaiBackend]` | model pack | `BonsaiBackend needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 2 | `tests/backends/test_contract.py::test_text_primitives_round_trip[NanbeigeBackend]` | model pack | `NanbeigeBackend needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 3 | `tests/backends/test_contract.py::test_text_primitives_round_trip[LightningBackend]` | model pack | `LightningBackend needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 4 | `tests/backends/test_contract.py::test_text_primitives_round_trip[Gemma4Backend]` | model pack | `Gemma4Backend needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 5 | `tests/backends/test_contract.py::test_text_primitives_round_trip[DeepseekV41Backend]` | model pack | `DeepseekV41Backend needs a model pack to tokenize; its text primitives are verified through the kindi verify/bench gate, not this harness` |
| 6 | `tests/test_calib.py::test_write_failure_is_best_effort` | platform/Windows | `chmod does not make a dir read-only on Windows` |
| 7 | `tests/test_cpu_avx_fallback.py::test_no_fma_verify_is_token_exact[None]` | model pack | `set KINDI_CPU_AVX_PACKS to one or more Bonsai pack directories` |
| 8 | `tests/test_cpu_avx_fallback.py::test_avx2_no_fma_verify_is_token_exact[None]` | model pack | `set KINDI_CPU_NO_FMA_PACKS to one or more Bonsai pack directories` |
| 9 | `tests/test_cpu_build.py::test_avx_only_build_preprocesses` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 10 | `tests/test_cpu_build.py::test_avx_only_library_builds_and_loads` | optional compiler/toolchain | `clang is not installed; set KINDI_CC to build the AVX-only fallback` |
| 11 | `tests/test_cpu_build.py::test_avx2_f16c_without_fma_preprocesses` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 12 | `tests/test_cpu_build.py::test_avx2_f16c_no_fma_library_builds_and_loads` | optional compiler/toolchain | `clang is not installed; set KINDI_CC to build the no-FMA AVX2 fallback` |
| 13 | `tests/test_cpu_build.py::test_full_isa_library_builds_and_loads` | optional compiler/toolchain | `clang is not installed; set KINDI_CC to build the native path` |
| 14 | `tests/test_cpu_build.py::test_non_avx_build_is_rejected_with_the_required_isa` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 15 | `tests/test_cpu_build.py::test_supported_isa_source_preprocesses` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 16 | `tests/test_cpu_build.py::test_other_avx2_variants_preprocess[flags0]` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 17 | `tests/test_cpu_build.py::test_other_avx2_variants_preprocess[flags1]` | optional compiler/toolchain | `neither gcc nor clang is installed` |
| 18 | `tests/test_cpu_layers.py::test_cpu_isa_tiers_agree_on_tokens_and_hidden_close` | optional compiler/toolchain | `no gcc/clang to build the ISA tier libraries` |
| 19 | `tests/test_cpu_layers.py::test_layerwise_matches_whole_model_step` | model pack | `set KINDI_CPU_PACK to a Bonsai pack to run the real exactness check` |
| 20 | `tests/test_cpu_smoke.py::test_compiled_library_loads_with_the_python_abi` | optional prebuilt CPU library | `set KINDI_CPU_SMOKE=1 with a prebuilt kcpu library` |
| 21 | `tests/test_cpu_smoke.py::test_compiled_library_initializes` | optional prebuilt CPU library | `set KINDI_CPU_SMOKE=1 with a prebuilt kcpu library` |
| 22 | `tests/test_gemma4_fp8.py::test_ops_dense_batch_dispatch[30]` | optional GPU library (cuBLASLt) | `cuBLAS is unavailable: Failure finding "known cublasLt DLL": No such file: known cublasLt DLL, No such file: known cublasLt DLL` |
| 23 | `tests/test_gemma4_fp8.py::test_ops_dense_batch_dispatch[43]` | optional GPU library (cuBLASLt) | `cuBLAS is unavailable: Failure finding "known cublasLt DLL": No such file: known cublasLt DLL, No such file: known cublasLt DLL` |
| 24 | `tests/test_gemma4_fp8.py::test_ops_dense_batch_dispatch[44]` | optional GPU library (cuBLASLt) | `cuBLAS is unavailable: Failure finding "known cublasLt DLL": No such file: known cublasLt DLL, No such file: known cublasLt DLL` |
| 25 | `tests/test_gemma4_fp8_mma.py::test_ops_linear_batch_falls_back_without_the_native_kernel[43]` | optional GPU library (cuBLASLt) | `cuBLAS is unavailable: Failure finding "known cublasLt DLL": No such file: known cublasLt DLL, No such file: known cublasLt DLL` |
| 26 | `tests/test_gemma4_fp8_mma.py::test_ops_linear_batch_falls_back_without_the_native_kernel[44]` | optional GPU library (cuBLASLt) | `cuBLAS is unavailable: Failure finding "known cublasLt DLL": No such file: known cublasLt DLL, No such file: known cublasLt DLL` |
| 27 | `tests/test_server_cpu.py::test_cpu_server_boots_and_answers` | optional CPU library/compiler | `no cached CPU library and no gcc/clang to build it` |
| 28 | `tests/test_server_cpu.py::test_cpu_server_reports_the_pack_in_health` | optional CPU library/compiler | `no cached CPU library and no gcc/clang to build it` |

The FP8 rows are **not** a "no GPU" result: this host has an RTX 4060 Ti, but CuPy cannot load
`cublasLt`, so the CUDA FP8 path is not exercised here. The CPU-build and CPU-server rows need a
compiler or a cached `kcpu` library; the `KINDI_CPU_SMOKE` rows are explicit opt-ins. None of these
skips is evidence that the corresponding GPU, model-pack or optional-dependency path passed.

## Coverage: where the 27% floor comes from

Measured at final commit `539a504` (Windows, line+branch; coverage rerun 2026-11-19):
**1,241 passed, 28 skipped, 0 failed; 46.36% total** (46.3603%; 23,504 statements and 8,394
branches). The floor
in CI stays 27%, not 46%, because the number moves with the platform and a gate that fails on a
fresh Linux checkout is worse than a loose gate. The distribution is what matters: the CPU suite
covers the modules it can reach, and the heavy GPU modules are covered by `kindi verify` and
`kindi bench` on hardware, not by pytest. The run is recorded with its revision, command and
counts in [full-suite.json](../docs/benchmarks/2026-11-18-i7-8086k-windows-cpu/full-suite.json).

| module | cover | who checks it |
|---|---|---|
| `tandem.py`, `vram.py`, `expert_cache.py` | 90%, 89%, 89% | the CPU suite (byte accounting, device mapping, cache state machine) |
| `bench.py`, `cache.py`, `server.py` | 75%, 52%, 69% | the CPU suite plus the HTTP boundary tests |
| `engine.py`, `spec.py`, `tp.py`, `small mtp.py` | 24%, 20%, 13%, 19% | `kindi verify` (token-exact against llama.cpp) and the perf gate; pytest reaches the loaders, guards and CPU-side helpers |
| `convert.py`, `gemma4_tp.py` | 16%, 11% | `kindi convert` plus the mixer/support tests; the CUDA half needs a card |

The final `539a504` coverage figures round to these same twelve module values; only the overall
totals changed. The 27% total floor is the noisy-gate guard. On top of it, `tests/coverage_floors.json`
pins a per-module floor for **46** modules the CPU suite reaches (audit M4), including the
planner, MoE policy, grammar/chat parsers, CLIs and the new cold/I/O/autotune modules.
`tools/check_coverage.py` fails when one of those modules drops below its floor, so a
CPU-exercised module cannot rot while the total holds. The new floors sit 10 points below the
measured value; the graph builders (`spec.py`, `dspine.py`, `mtp.py`, `tp.py`) get only a low
CPU-reachable floor, because their CUDA paths are `kindi verify`'s job on hardware. A GPU run
should raise those floors, not a hand edit. `moe_cpu.py` stays out until its C library is built
in CI. Do not read "46%" as "the engine is half-tested"; read the table.

## Turing reference suite (optional)

These tests check the Turing (`sm_75`) tensor-core backend without a Turing GPU. The full verification
record is in [docs/VERIFICATION.md](../docs/VERIFICATION.md).

| file | what it does |
|---|---|
| `test_tc_turing_reference.py` | Compares the Turing kernels with an independent NumPy model: the packed weight layout, the int8 activations, every product kernel, the merge/sum kernels and the whole decode chain. |
| `static/run_static_analysis.py` | Builds the full CUDA translation units that `kindi.spec.SpecKernels` compiles (Turing and Ampere) and runs cppcheck and clang-tidy on them. |
| `static/cuda_stub.h` | The minimal CUDA declarations that let the analysers parse the kernels without a CUDA toolkit. Never compiled into Kindi. |

### Requirements

- Python 3 with `numpy` and `cupy`; the reference suite needs a CUDA device. Any sm_75+ card works because
  the Turing module is `compute_75` PTX that the driver JIT-compiles.
- `pytest` for the pytest entry point; `python tests/test_tc_turing_reference.py` also runs without pytest.
- `cppcheck` and/or `clang-tidy` for the static script (found on `PATH` or set with `CPPCHECK` / `CLANG_TIDY`).

### Run the reference suite

```sh
PYTHONPATH=. python -m pytest tests/test_tc_turing_reference.py -q
PYTHONPATH=. python tests/test_tc_turing_reference.py      # same, with the per-test summary
```

It builds the kernels through the production path (`kindi.spec.SpecKernels(backend=tc_turing)`), so
the loader and the full translation unit are exercised too. All 9 tests passed on 2026-11-07 (sm_89).

On a host with no CUDA device the module skips as a unit with the reason `needs a CUDA device for
the Turing PTX kernels`; the 10 node IDs are `test_reference_layout`, `test_turing_names`,
`test_activations`, `test_int8_shift_compensation_is_required`, `test_pack_layout`, `test_products`,
`test_token_counts`, `test_shapes_and_merges`, `test_sum` and `test_whole_chain`. That is an
environment skip, not a pass.

### Why the reference is independent

The NumPy model does not call or transcribe the CUDA code. It works from the documented format:

- the `tc` packed layout is rebuilt byte by byte from the base-3 trit definition (`b = ceil(256 v / 243)`,
  scales in word 26/27, group `G = 2t + (wk - 26)`, row `g8 + 8 * (bq >> 1)`);
- `k_tc_q8` is re-derived from `q = rn(x * 127 / amax)`, `XS = amax / 127`, `XA = sum(q)`;
- each product is the exact integer sum `sum((trit - 1) * q)` scaled by `XS * d`, with the accumulation
  split the kernels use;
- the whole chain reproduces the split-accumulation grouping of `k_tc_mv` + `k_tc_sum` and the fused `fma`
  emission.

A typo in either the CUDA or the model changes one of the byte-exact comparisons.

## Static analysis

```sh
python tests/static/run_static_analysis.py              # Turing (sm_75) and Ampere (sm_86)
python tests/static/run_static_analysis.py --only turing
python tests/static/run_static_analysis.py --keep out/  # keep the generated translation units
```

The expected findings are the shared baseline ones listed in
[docs/VERIFICATION.md](../docs/VERIFICATION.md#static-analysis): two `clang-analyzer` dead-store reports
in `kindi.cu`, the cppcheck `uninitvar` false positives and the C-style-cast style notes. A new finding
in `tc_turing.cu` or `tca_turing.cu` is a regression.

## Not covered here

Runtime checks on a real `sm_75` card (`kindi verify`, `kindi bench`, long contexts) need Turing hardware;
they are listed as pending in [docs/VERIFICATION.md](../docs/VERIFICATION.md).
