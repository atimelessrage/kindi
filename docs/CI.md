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

The performance half of the roadmap gate remains opt-in in the manual GPU lane. A tok/s number is
only comparable on the same runner, card, driver and CuPy/CUDA build, so it needs a recorded
baseline rather than a hosted-CPU threshold.

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
[bench-logs/blackwell-fp8-ada-refusal-20261112/RESULTS.md](../bench-logs/blackwell-fp8-ada-refusal-20261112/RESULTS.md).
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
