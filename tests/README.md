# Tests

The repository has a fast CPU-only gate and an optional CUDA reference suite. Neither is a substitute
for `kindi verify` on the target card, which needs a model pack and a GPU.

## CPU-only gate

Run from the repository root:

```sh
python -m compileall -q kindi
python -m pytest tests/ -q
# Optional local coverage gate (matches the CPU CI job):
python -m pytest tests/ -q --cov=kindi --cov-branch --cov-report=term --cov-report=json:coverage.json --cov-fail-under=27
python tools/check_coverage.py
bash -n install.sh
```

The suite uses stub engines and stand-in CuPy where needed. It covers config precedence, the chat
template and tool-call round trips, tokenizer behavior, sampling, cache decisions, the HTTP boundary
(validation, auth, body caps, queue, Ollama endpoints, image rejection), startup pack checks, the
plan-engine guards, the tracked-file secret/personal-data guard and detect-secrets baseline checks, the installer's GRUB fixture and
CPU ISA preflight, and GPU-driver selection. No CUDA device or model pack is required.

## The kernel launch contract is partly measured, and says how much

`test_kernel_arity.py` reads every `__global__` declaration in `kindi/*.cu` (including
`__launch_bounds__` forms and the generated `k_*_T` wrappers) and compares each graph launch's
literal argument tuple with it. Current split: **71 call sites checked across 40 kernels** (`spec.py`
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

## Coverage: where the 27% floor comes from

Measured on 2026-11-18 (Windows, 1,196 passed, 28 skipped, line+branch): **45.4% total**. The floor
in CI stays 27%, not 45%, because the number moves with the platform and a gate that fails on a
fresh Linux checkout is worse than a loose gate. The distribution is what matters: the CPU suite
covers the modules it can reach, and the heavy GPU modules are covered by `kindi verify` and
`kindi bench` on hardware, not by pytest.

| module | cover | who checks it |
|---|---|---|
| `tandem.py`, `vram.py`, `expert_cache.py` | 90%, 89%, 88% | the CPU suite (byte accounting, device mapping, cache state machine) |
| `bench.py`, `cache.py`, `server.py` | 67%, 56%, 46% | the CPU suite plus the HTTP boundary tests |
| `engine.py`, `spec.py`, `tp.py`, `small mtp.py` | 24%, 20%, 13%, 19% | `kindi verify` (token-exact against llama.cpp) and the perf gate; pytest reaches the loaders, guards and CPU-side helpers |
| `convert.py`, `gemma4_tp.py` | 18%, 0% | `kindi convert` plus the mixer/support tests; the CUDA half needs a card |

The 27% total floor is the noisy-gate guard. On top of it, `tests/coverage_floors.json`
pins a per-module floor for **46** modules the CPU suite reaches (audit M4), including the
planner, MoE policy, grammar/chat parsers, CLIs and the new cold/I/O/autotune modules.
`tools/check_coverage.py` fails when one of those modules drops below its floor, so a
CPU-exercised module cannot rot while the total holds. The new floors sit 10 points below the
measured value; the graph builders (`spec.py`, `dspine.py`, `mtp.py`, `tp.py`) get only a low
CPU-reachable floor, because their CUDA paths are `kindi verify`'s job on hardware. A GPU run
should raise those floors, not a hand edit. `moe_cpu.py` stays out until its C library is built
in CI. Do not read "45%" as "the engine is half-tested"; read the table.

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
