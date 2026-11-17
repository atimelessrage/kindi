# CI and the GPU lane

The normal GitHub-hosted workflow is
[.github/workflows/ci.yml](../.github/workflows/ci.yml):

- `CPU tests`: the CPU-only pytest suite on Ubuntu, with line and branch coverage for `kindi` and a
  27% regression floor;
- `CPU engine build and smoke`: compiles `kindi/kcpu.c` with AVX2/FMA/F16C and loads it;
- `Windows CPU and config boundary`: the Windows-safe config, CPU, startup and tokenizer tests;
- `Installer shell checks`: `bash -n` and ShellCheck on the committed LF installer.
- `Secret scan`: detect-secrets 1.5.0 scans every tracked file against `.secrets.baseline`, and
  [tools/check_secret_baseline.py](../tools/check_secret_baseline.py) rejects a stale baseline.

These jobs do **not** run CUDA kernels, a model pack or `kindi verify` on a real card.

## Coverage floor

The CPU job runs `pytest` with `--cov=kindi --cov-branch --cov-report=term --cov-fail-under=27`.
The current local baseline is **27.17%** line-and-branch coverage on the CPU-only suite
(2026-11-12); the floor is deliberately at that measured level so platform and dependency
differences do not make the job noisy. New tests in `tests/test_requirement_coverage.py` are
written against requirement and user-visible contracts, not implementation internals. Coverage
is an outcome of those tests, not a reason to test private code paths. This is an
anti-regression floor, not a quality claim: most CUDA, CuPy, model-loading and vendor paths are
uncovered because hosted CI has no GPU or model pack. Raise the floor only with measured coverage,
and do not exclude hardware modules merely to make the number look better.

On top of the total floor, the CPU job runs `tools/check_coverage.py` against
`tests/coverage_floors.json` (audit M4): a per-module floor for every module the CPU suite
can actually reach (server, cache, bench, tandem, vram, expert_cache, tokenizer, config,
hostram, catalog, sched, sampling, spm, lookup, profiles, gpudrv). A total floor alone lets a
CPU-exercised module rot while the total holds; the per-module floors pin the modules the
requirement tests cover. The heavy CUDA modules (`engine.py`, `spec.py`, `tp.py`) are
deliberately excluded — they are verified by `kindi verify` on hardware, and a floor for them
needs GPU coverage first (the GPU lane above). Floors sit ~15 points below the measured value
so platform variance does not make CI noisy; re-record with `--update` only after a measured run.

The performance half of the roadmap gate is `tools/check_perf_baseline.py`: it compares the
recorded `docs/benchmarks/CROSS-MODE.csv` against `docs/benchmarks/perf-baseline.json` and fails
when a measured cell disappears or regresses by more than 15%. The CPU CI job runs this gate, so
an accidental overwrite of the recorded numbers fails CI; the GPU lane should regenerate the
matrix on its runner and run the same gate for a real throughput check. A tok/s number is only
comparable on the same runner, card, driver and CuPy/CUDA build, so re-recording the baseline
(`--update`) is a deliberate, reviewed step — never a drive-by. `tests/test_perf_baseline.py`
pins the gate behaviour.

The same job runs `python tools/check_claims.py`: every number quoted in `docs/BACKENDS.md`,
`docs/ROADMAP.md` and `CURRENT-VS-KINDI-0.6.md` is re-derived from the tracked artifact of its run
(a JSON row, a `CROSS-MODE.csv` cell, a ratio of two cells, a max gap between two files, or a
verbatim line in the run's `RESULTS.md`), and a doc that stops quoting the number fails. It also
refuses a `docs/benchmarks/<dir>` citation that no `docs/claims/<gate>.json` (and no reasoned
`unclaimed` entry in `docs/claims/REGISTRY.json`) accounts for, so new evidence cannot enter a doc
without an owner. Schema, check kinds and the add/change procedure are in
[docs/claims/README.md](claims/README.md); `tests/test_claims.py` pins both the repository's claims
and the checker's own failure modes (a drifted doc number, a moved artifact, untracked evidence, an
unregistered citation).

## CUDA static analysis

The `static-analysis` job installs `cppcheck` and `clang-tidy-18` on the hosted runner and runs
`tests/static/run_static_analysis.py`, which builds the Turing (sm_75) and Ampere (sm_86)
translation units the way `kindi.spec.SpecKernels` does and analyses them as source only
(`tests/static/cuda_stub.h` supplies the CUDA declarations; no CUDA toolkit or GPU is needed).
Hosted runners have no CuPy, so the harness installs a minimal stub `cupy` module when the real
one is unavailable — it only has to survive module import while the translation units are built.
The script exits non-zero when an analyser fails to run; warnings are reported, not gated.
Validated locally (tools absent): the Turing TU builds and the script skips cleanly with exit 0.

`tests/requirements-test.txt` is deliberately bounded (`pytest>=7,<10`, `fastapi>=0.100,<1`,
`pydantic>=2,<3`, …): the suite is verified on those majors, and an open-ended spec would let
CI break with no commit. `tests/test_ci_config.py` pins both contracts.

## Secret baseline and local hook

The secret scan is intentionally tracked-file based. CI runs
[tools/check_secret_baseline.py](../tools/check_secret_baseline.py) to compare the baseline with a
fresh scan of `git ls-files`, then runs `detect-secrets-hook --baseline .secrets.baseline` to reject
new findings. The pre-commit hook uses the same baseline and scans all text files; only
`.secrets.baseline` is excluded so the hashes in the baseline are not reported as secrets.

Install the local hook once:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run detect-secrets --all-files
```

To review and accept a new false positive (for example a published checksum), update the baseline
from tracked files only:

```bash
python tools/check_secret_baseline.py --check
python tools/check_secret_baseline.py --update
```

Do not use `detect-secrets scan --all-files` for this repository: it includes untracked benchmark
logs and test caches. Review the flagged line before updating; the baseline stores a hash, not the
value, but it still marks the line as known. Never commit a real credential to make the baseline
match.

## GPU lane

[.github/workflows/gpu-smoke.yml](../.github/workflows/gpu-smoke.yml) is a manual
`workflow_dispatch` lane. It requires a self-hosted runner with both labels:

- `self-hosted`
- `gpu`

The runner needs a working NVIDIA driver, the CuPy version used by the chosen image, and one or
more converted Kindi packs. The workflow has three independent jobs:

1. **GPU verify and server boundary** starts from the `pack` input, runs `python -m kindi verify`,
   then starts the server, waits for `/health` and sends one greedy chat request.
2. **GPU token exactness and performance regression** uses `pack_dir` and the selected `models`
   and `modes`. It runs [tools/verify_matrix.py](../tools/verify_matrix.py) as fresh `kindi verify`
   processes with a clean `KINDI_*` environment. If `baseline` is supplied, the same run also
   fails on a status loss, a lower exact pass count, a higher fail count, a baseline cell missing
   from the run, or a throughput drop greater than `max_tps_drop_pct`.
3. **GPU long-context 24K** runs [tools/check_long_context.py](../tools/check_long_context.py) on
   the `pack` input with `--ctx 24576`. It fails unless the engine reports at least the requested
   context and the 24,000-token generation cell exists with non-zero throughput. Change `ctx`
   when testing a different long-context size.

The default matrix is `models: 2b,4b,9b`, `modes: plain`. Add `tc` explicitly when the card and
packs are known to support it: a 9B tc refusal is a valid recorded result and deliberately makes
the matrix fail. The matrix and long-context jobs upload their logs and JSON results as workflow
artifacts even when they fail.

This is intentionally not an automatic job: a GPU and a 0.4-19 GB model pack are not available on
GitHub-hosted runners, and a skipped job is not verification. Record the runner, card, driver,
CuPy/CUDA versions and pack hash with the run. Until that run exists, the dual-card rows remain
unverified as described in [CARD-SUPPORT.md](CARD-SUPPORT.md).

### Registering the runner (M6)

The lane is ready; the missing piece is a registered runner. On the GPU host (a working NVIDIA
driver, the CuPy version used by the image, and the packs under `~/.kindi/packs/`):

1. In the fork's GitHub settings go to **Actions → Runners → New self-hosted runner** and follow
   the printed download/configure commands.
2. At configure time pass the labels `self-hosted` and `gpu` (the Blackwell lane additionally
   needs `sm_100` or `sm_120` — one runner per architecture, or one runner with both labels).
3. Install the runner as a service (`./svc.sh install`) so it survives reboots, then dispatch
   **GPU smoke** from the Actions tab with the pack path and record the run (runner, card,
   driver, CuPy/CUDA, pack hash) in [CARD-SUPPORT.md](CARD-SUPPORT.md).
4. Then run `python tools/check_perf_baseline.py` against the regenerated CROSS-MODE.csv on
   that runner for the real throughput gate.

Until step 3 has run at least once, every GPU-verified card row in the docs stays a manual
measurement, and `gpu-smoke.yml` remains an unused lane.

## Blackwell FP8 lane

[.github/workflows/blackwell-fp8.yml](../.github/workflows/blackwell-fp8.yml) is a separate manual
`workflow_dispatch` lane for the native Gemma 4 FP8 kernels. It needs a self-hosted runner labelled
`self-hosted`, `gpu` and the selected architecture (`sm_100` or `sm_120`), plus a working NVIDIA
driver and CuPy; it does not need a Kindi pack. The workflow routes the job to the matching
architecture label from the `arch` input, which requires `sm_100` or `sm_120`. Dispatch once per
architecture to cover both. The Python tool still accepts `--arch auto` for local runs, but the
workflow does not use it, so a job cannot land on the wrong Blackwell architecture.

The lane runs [tools/verify_blackwell_fp8.py](../tools/verify_blackwell_fp8.py), which refuses to
pass on any device whose compute capability is not exactly 100 or 120. A non-Blackwell device or an
arch mismatch is written as `unsupported` and exits 2; a loaded-kernel or numerical mismatch exits
1. Both outcomes upload `results.json`, `RESULTS.md` and `run.log`. A green workflow therefore
means the synthetic E4M3/E5M2 quantization, GEMM and `Ops.linear_batch` checks passed on that
Blackwell card, not that a real FP8 model was validated. No Blackwell runner is registered for
this worktree, so the workflow has not run; a local gate on the Ada-only host returned
`unsupported` (exit 2) on 2026-11-12, recorded in
[benchmarks/blackwell-fp8-ada-refusal-20261112/RESULTS.md](benchmarks/blackwell-fp8-ada-refusal-20261112/RESULTS.md).
It is not a Blackwell result.

## Performance baseline

The performance gate is opt-in through the `baseline` input because a throughput number is only
comparable on the same runner, card, driver and CuPy/CUDA build. Create the baseline from a clean
run on that machine:

```bash
python tools/verify_matrix.py \
  --pack-dir /data/kindi/packs \
  --models 2b,4b,9b \
  --modes plain \
  --out /data/kindi/ci/gpu-baseline-run \
  --save-baseline /data/kindi/ci/gpu-baseline.json
```

Then pass `/data/kindi/ci/gpu-baseline.json` as `baseline` on later dispatches. The default
`max_tps_drop_pct` is 10%. Status, exactness and missing-cell regressions fail regardless of the
throughput threshold. If `baseline` is empty, the job still checks token exactness but prints a
warning and makes no performance-regression claim.

Regenerate the baseline deliberately, on the same hardware, after a measured performance change;
do not commit it. Upload the run's `comparison.json` alongside the baseline so a failure can be
reviewed.

## Adding a pack to a runner

The pack is produced once with `python -m kindi convert MODEL.gguf`; point the runner at that
directory, not at the GGUF, and keep it outside the workspace. Do not commit the pack or any model
file. The workflow uses `KINDI_PACK` and `KINDI_PACK_DIR`, so the same scripts work with a path
containing spaces.
