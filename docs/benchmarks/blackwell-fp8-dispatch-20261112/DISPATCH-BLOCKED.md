# Blackwell FP8 dispatch attempt — 2026-11-12

Result: **blocked before any runner job; no sm_100 or sm_120 hardware result.**

## Requested

Dispatch `.github/workflows/blackwell-fp8.yml` twice, once with `arch=sm_100` and once with
`arch=sm_120`, on self-hosted runners labelled `self-hosted`, `gpu` and `blackwell`; record the
resulting `results.json`, `RESULTS.md` and `run.log` artifacts.

## Evidence collected

- Local GPU: `NVIDIA GeForce RTX 4060 Ti`, compute capability `8.9` (Ada), driver `617.42`.
  This is not an sm_100 or sm_120 device.
- GitHub CLI: `gh: command not found`.
- GitHub token variables: `GH_TOKEN`, `GITHUB_TOKEN` and `GITHUB_PAT` are all unset.
- Local workflow state: `.github/workflows/blackwell-fp8.yml` is untracked (`??`) and
  `git log --all -- .github/workflows/blackwell-fp8.yml` has no commits.
- Remote workflows for `atimelessrage/kindi` (public default branch `main`) are only `CI` and
  `GPU smoke`; there is no Blackwell workflow.
- GitHub contents API for `.github/workflows/blackwell-fp8.yml`:
  - `atimelessrage/kindi@main`: HTTP 404
  - `atimelessrage/kindi@freebuff/tandem-cpu-recovery`: HTTP 404
  - `ahtx/kindi@main`: HTTP 404
- GitHub runners API without authentication: HTTP 401 `Requires authentication` for both
  `atimelessrage/kindi` and `ahtx/kindi`. Runner registration therefore cannot be inspected with
  the available access.

## Why no dispatch was made

`workflow_dispatch` needs the workflow on the repository's default branch, and the API call also
needs an authenticated client with Actions write access. Neither condition is met here: the
workflow is not committed or pushed, and no `gh` login/token is available. The branch is also ten
commits ahead of its remote and has many unrelated uncommitted changes, so pushing or merging the
whole worktree would not be a safe way to expose this lane. No commit or push was made.

The local Ada refusal is already recorded separately in
`bench-logs/blackwell-fp8-ada-refusal/`; it is an `unsupported` exit 2, not a Blackwell pass.

## What is needed to run the lane

1. Authenticate a GitHub CLI or token with Actions write access for the target repository.
2. Put `blackwell-fp8.yml`, `tools/verify_blackwell_fp8.py` and `kindi/gemma4_fp8.cu` on the
   repository's default branch (or an authorized branch that can be selected for the dispatch).
3. Register at least one runner with labels `self-hosted`, `gpu`, `blackwell` and an sm_100 device,
   and one with the same labels and an sm_120 device. Both need a working NVIDIA driver and CuPy.
4. Dispatch once per architecture and preserve the uploaded artifacts here:
   - `kindi-blackwell-fp8-sm_100`
   - `kindi-blackwell-fp8-sm_120`

Do not mark either architecture as verified until its own artifact reports `status: pass` with
device compute capability 100 or 120.
