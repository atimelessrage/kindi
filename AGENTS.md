# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Worktree and shared-file protocol (2026-11-08)

This session runs in a **linked worktree**, not the main checkout. The two working trees share the
Git object database, not the index or the files on disk:

| worktree | branch | HEAD | role |
|---|---|---|---|
| `T:/kindi` | `main` | `0fc024f` | main checkout; another agent is active here |
| `C:/Users/ICH/.freebuff/worktrees/f0e51299/kindi` | `freebuff/project-review-audit-report` | `4b90ca9` | this session |

The main worktree has a bridge note at `T:/kindi/AGENTS.md`; it is not merged into this worktree.
Read it before editing a shared file.

**Shared files:** `AGENTS.md`, `README.md`, `WHY.md`, `CHANGELOG.md`, `install.sh`,
`kindi.conf.example`, `docs/**`, `.github/**` and every file named in the lease table below.
`AGENTS.md` is per-worktree until it is committed; the main agent cannot see an untracked copy here.

**Rules**

1. Before editing a shared file, run `git worktree list --porcelain`, `git status --short -- <file>`
   and `git -C <other-worktree> status --short -- <file>`. Read the other copy if it differs.
2. A task owns the shared files in its row. Only the owner edits them; other tasks read only. Add a
   row before the first edit of a new task.
3. If the other worktree has uncommitted changes to a shared file, do not overwrite it, copy over
   it or merge it. Wait for the owner to commit or hand off. When both worktrees are clean,
   synchronize with `git merge` or `git rebase` on the owning branch; never copy working files
   between worktrees.
4. Do not run `git reset --hard`, `git clean`, `git checkout --`, `git stash` or a rebase in a
   worktree you do not own. Do not use `git add -A`; commit only the task's files.
5. Re-read the file immediately before writing and after another agent's commit. Git is the source
   of truth; a chat message is not a lock.
6. At handoff: commit the task, record its branch, HEAD and remaining files here, and leave the
   worktree clean or explicitly listed. Do not delete another worktree's files.

### Current leases (checked 2026-11-08)

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| Main-worktree task (other agent) | `T:/kindi` | `main` | `README.md`, `docs/CARDS.md`, `docs/CONFIGURATION.md`, `docs/INSTALL.md`, `docs/TURING-TENSOR-CORES.md`, `docs/VERIFICATION.md`, `docs/PAIR-SETUPS.md`, `docs/CPU_GPU_TANDEM_DESIGN.md`, `docs/GPU_PROFILE_ANALYSIS.md`, `docs/dev/**`, `install.sh`, `kindi/spec.py`, `.github/workflows/ci.yml` | dirty in the main worktree; owner: other agent; do not edit |
| Documentation drift (P1-3, P1-4, P1-5, P1-6) | this worktree | `freebuff/project-review-audit-report` | `WHY.md`, `docs/CONFIGURATION.md`, `docs/INSTALL.md`, `docs/BENCHMARKS.md`, `docs/CARDS.md`, `docs/MTP.md`, `kindi.conf.example`, `kindi/spec.py` | partial: `WHY.md` and `kindi.conf.example` updated; `docs/CONFIGURATION.md`, `docs/INSTALL.md`, `docs/CARDS.md`, `docs/BENCHMARKS.md` and `docs/MTP.md` remain blocked by the main worktree's dirty copies |
| Continuous-batching investigation | this worktree | `freebuff/project-review-audit-report` | `docs/CONTINUOUS-BATCHING.md`, `AGENTS.md` | archived NO-GO; Phase-0 gate failed (0.69x); top banner and section 8 updated; no production code changed |
| Server request validation + boundary hardening (P1-1/P2-1) | this worktree | `freebuff/project-review-audit-report` | `kindi/server.py`, `tests/test_server.py`, `tests/test_config_example.py` | done; 10 server tests (validation, auth, 413 body cap) + 1 example-config test + 9 installer fixture tests = 20 passed; invalid requests and 413 keep the cache |
| GRUB edit guard (plan 1) | this worktree | `freebuff/project-review-audit-report` | `install.sh`, `tests/test_installer_grub.py` | done; 9 fixture tests pass, guard verified non-vacuous. **Release blocker: docs/INSTALL.md still pins the buggy 592a... hash; the fixed copy hashes 38fb3417...; main owns docs/INSTALL.md. `T:/kindi/install.sh` also remains dirty; reconcile by hand before release** |
| Per-slot plain-engine plan (Phase 1) | this worktree | `freebuff/project-review-audit-report` | `docs/PER-SLOT-REFACTOR.md`, `docs/CONTINUOUS-BATCHING.md` | archived NO-GO with `docs/CONTINUOUS-BATCHING.md`; no engine code changed |
| 2B CB-vs-spec measurement | this worktree | `freebuff/project-review-audit-report` | `bench-logs/compare-2b-cb-spec/**`, `AGENTS.md` | measured; no code changed; results in `bench-logs/compare-2b-cb-spec/RESULTS.md` |
| Phase-0 CB spike | this worktree | `freebuff/project-review-audit-report` | `bench-logs/spike_batch_gemm.py`, `bench-logs/phase0-spike/**`, `docs/CONTINUOUS-BATCHING.md` | implemented; B=4 gate FAIL (0.69x vs single decode); no production code changed |
| Plan-engine exactness guard | this worktree | `freebuff/project-review-audit-report` | `kindi/spec.py`, `AGENTS.md`, `bench-logs/diag_exact_plan.py`, `bench-logs/diag_verify8.py`, `bench-logs/diag_spec_long.py` | 2B sp=1 (no k_tc_sum) + 9B HB guard; long spec 160/160 exact; **main worktree has uncommitted `kindi/spec.py`; merge by hand** |
| Long-term card support (single/dual) | this worktree | `freebuff/project-review-audit-report` | `docs/CARD-SUPPORT.md`, `AGENTS.md` | roadmap and acceptance contract written; no engine/installer code changed; the 4060 Ti 8 GB classification and all non-Maxwell pairings are blocked by the main worktree's dirty `install.sh` |
| Long-term CPU support (i5-2500K, W-2133, i9-13900K, i7-8086K) | this worktree | `freebuff/project-review-audit-report` | `docs/CPU-SUPPORT.md`, `AGENTS.md` | roadmap and acceptance contract written; no CPU/installer code changed; `kindi/cpu.py`, `kindi/kcpu.c` and `install.sh` are main-owned/dirty; the i5-2500K needs an AVX-only/scalar fallback |
| `install.sh` / `.github` collision | both | both | `install.sh`, `.github/**` | modified or present in both worktrees; do not merge or overwrite until the two changes are reconciled by hand (the main copy was empty on 2026-11-08) |

That main-worktree lease covers its **whole dirty set**, not only the docs and installer. It also
has changes in `GOALS.md`, `IMPLEMENTATION_PLAN.md`, `setup.py`, `kindi/config.py`,
`kindi/convert.py`, `kindi/cpu.py`, `kindi/engine.py`, `kindi/hostram.py`, `kindi/kcpu.c`,
`kindi/layout.py`, `kindi/nanbeige_cpu.py`, `kindi/nbq_cpu.c`, `kindi/tp.py`, `tests/README.md`,
new tests, and its own `AGENT_COLLABORATION_PROTOCOL.md` (a review-findings list, not a worktree
lease). `git -C T:/kindi status --short` is the authoritative lease; treat every path there as
main-owned except `T:/kindi/AGENTS.md` (this bridge note). Re-run the status command; this snapshot
is not a lock.

## Snapshot (2026-11-08)

- Version 0.4.0 (`kindi/__init__.py`), model id `bonsai2-2b`, AGPL-3.0-only plus a commercial
  option. New files carry the SPDX header of their neighbours.
- Audited and committed in this worktree: `5061af2` (server boundary hardening, WHY/config example,
  No-Go docs, CPU tests) and `2b84658` (installer + plan-engine fixes, full CPU suite). Left untracked:
  `.github/`, `bench-logs/`, the two audit reports. The main worktree's central docs, CI, `setup.py`,
  `install.sh` and `kindi/spec.py`/`engine.py`/`tp.py`/`cpu.py` remain dirty and are untouched here;
  reconcile by hand before merging. The documentation drift task is complete for `WHY.md` and
  `kindi.conf.example`; its remaining files are blocked by the main worktree.
- Kernels are not compiled at build time: CuPy compiles the `*.cu` sources with NVRTC on first
  use, and the driver caches them under `state_dir`, so the first start is slow.
- The server serves one request at a time. Greedy output is token-for-token identical to
  llama.cpp for the same file (`kindi verify`). Do not change that casually.
- Configuration precedence: built-in defaults < config file (`kindi.conf`, `~/.kindi/kindi.conf`)
  < environment (`KINDI_<KEY>`) < command-line flags.

## Handoff (2026-11-09)

- Branch `freebuff/project-review-audit-report`; audit follow-up commits `5061af2`, `2b84658`,
  `4b90ca9` and this roadmap commit.
- Verified: `python -m pytest tests/ -q` -> 58 passed; focused server/config/installer run -> 20
  passed; `bash -n install.sh` -> OK; `python -m compileall -q kindi/server.py` -> OK. No GPU run.
- Still blocked: the 0.4.0 installer checksum in `docs/INSTALL.md`; the central docs/CI, packaging
  (`setup.py`) and engine/tp/cpu edits are owned by the dirty main worktree. Do not release until
  those are reconciled and the checksum is updated to the fixed `install.sh` hash.
- Long-term single/dual card target is in `docs/CARD-SUPPORT.md`. The current installer misclassifies
  the 8 GB RTX 4060 Ti as `pascal8` because `tc` also requires at least 9,500 MiB, and `pair_ok()`
  rejects every pair above sm_60. Both fixes and the `tc2` profile are main-worktree work; the card
  target is not a support claim until a GPU verify is recorded.
- Long-term CPU target is in `docs/CPU-SUPPORT.md`. The CPU engine requires AVX2, FMA and F16C but
  checks none of them; the i5-2500K (AVX only) therefore cannot build the current `kcpu.c`. The Xeon
  W-2133, i9-13900K and i7-8086K have the required instructions but have no measured CPU run here.
  `kindi/cpu.py`, `kindi/kcpu.c` and `install.sh` remain main-owned; no CPU code changed.

## Layout

| path | what |
|---|---|
| `kindi/` | Python package: `engine.py`, `server.py`, `cache.py`, `spec.py`, `cpu.py`, `convert.py`, `config.py`; CLI in `__main__.py` |
| `kindi/*.cu` | CUDA kernels, compiled at runtime; `kcpu.c` is the AVX2 CPU engine |
| `install.sh` | the installer: machine check, Docker, models, `kindi verify`, services |
| `docs/` | user docs; `docs/CONFIGURATION.md` is the key reference |
| `tests/` | CPU-only unit tests (config, chat, tools, tokenizer, sampling, cache) and the installer GRUB fixture test |
| `docker/` | `Dockerfile` (CUDA 10.2) and `Dockerfile.rocm` |

## Setup

Native, on Linux with the driver and the CUDA 10.2 toolkit:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" \
            "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Docker, the recommended path for the old cards:

```sh
docker build -t kindi:0.4.0 -f docker/Dockerfile .
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.4.0 info
```

Models and packs live in `~/.kindi` (`pack`, `state_dir`). See `docs/INSTALL.md` for the pinned
GGUF files and hashes.

## Build and run

There is no compile step for the engine. "Build" means: convert a model once, optionally build
the fast head, and start the server.

```sh
python3 -m kindi convert ~/.kindi/Ternary-Bonsai-2-2B-PTQ1_0.gguf   # once, ~1 minute
python3 -m kindi fast-head                                        # optional, ~2 minutes
python3 -m kindi serve --host 127.0.0.1 --port 8093
curl -s localhost:8093/health
```

Docker build/run commands are in `README.md` and `docs/DOCKER.md`; the installer wraps all of
this (`bash install.sh --dry-run` prints every step without changing anything, Linux only).

## Test

Run from the repository root. The no-GPU gate is cheap and must pass before any handoff.

| command | needs | proves |
|---|---|---|
| `bash -n install.sh` | bash | the installer parses |
| `python3 -m compileall -q kindi` | python | every module parses |
| `python3 tests/test_installer_grub.py` | python, `sed` | the installer's GRUB sed keeps existing kernel parameters |
| `python3 -m pytest tests/ -q` | `pip install pytest` | the same tests under pytest |
| `python3 -m kindi --help` | python | the CLI imports without a GPU |
| `python3 -m kindi info` | python | config, pack and GPU report |

The GRUB test extracts the real `sed` script out of `install.sh` and runs it against
`tests/fixtures/grub.default`; it never touches `/etc`. If you change the edit, reintroduce the
old bug on a copy and confirm the test fails, so the test is not vacuous.

## Verify (GPU, model pack and reference)

```sh
python3 -m kindi convert MODEL.gguf     # once; the pack must match the reference file
python3 -m kindi verify                 # greedy output vs llama.cpp, token for token; must PASS
python3 -m kindi bench                  # tokens/s at several positions, prompt read, memory
KINDI_DEVICE=cpu python3 -m kindi bench # CPU engine only, no GPU
curl -s localhost:8093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-2b","messages":[{"role":"user","content":"Say hello."}]}'
```

Without a GPU you can run the no-GPU gate only. Say so in your report; never present a compile
or a syntax check as a runtime verification.

## Definition of done

- The no-GPU gate passes (first four rows above).
- If the server or cache changed: an invalid request returns 400 (not 500) and does not clear
  the prompt cache; a valid request right after it still hits the cache.
- If `install.sh` changed: `bash -n`, the fixture test, and `--dry-run` on Linux. Kernel edits
  are never tested against the real `/etc/default/grub`.
- Docs updated when a config key or environment variable is added (`docs/CONFIGURATION.md`,
  `kindi.conf.example`).
- The report names what was not verified.

## Current plan (2026-11-08)

Done:

- [x] Audit of code, docs, harness and design; report in
  `Buffy_ReviewAudit_202611082102.txt` (findings P0-P3, handoff section).
- [x] P0-1: the amdgpu GRUB edit in `install.sh` (`check_cwsr()`) used a literal 0x01 byte
  instead of the sed backreference `\1`, so it wiped every existing kernel parameter.
  Fixed; `tests/test_installer_grub.py` + `tests/fixtures/grub.default` pin the behaviour.
- [x] CPU-only pytest suite (`tests/test_*.py`) and `.github/workflows/ci.yml`: pytest, `bash -n
  install.sh` and shellcheck, with no GPU and no model pack.
- [x] Continuous-batching investigation: architecture and implementation plan in
  `docs/CONTINUOUS-BATCHING.md` (MVP 26-38 person-days, Phase-0 spike first; no engine code changed).
- [x] 2B comparison CB vs. Speculative Decoding on the RTX 4060 Ti: spec (plan T=4 + n-gram
  K=3) is 1.8x (engine) and 3.0-3.5x (copy/edit end-to-end) faster than plain; the CB proxy
  at B=4 is ~plain (199 vs. 191.5 tok/s), B=8-16 aggregates 273-755 tok/s but 21-30 ms per
  token per stream; CB is not implemented, so this is a proxy, not a CB system. Evidence:
  `bench-logs/compare-2b-cb-spec/RESULTS.md`.
- [x] Phase-0 CB spike: `bench-logs/spike_batch_gemm.py` runs one GDN block, one attention
  block and the head with B=1/2/4 independent rows; rows are exact vs B=1 (max diff 0.0,
  head argmax 1/1, 2/2, 4/4) and the GEMM scales 3.90x relative to its own B=1. Absolute
  B=4 is 131.8 tok/s = 0.69x of the 191.5 single decode, so the 1.5x gate FAILS. No-Go for
  the 4-slot MVP on the current chunk-GEMM path. Evidence:
  `bench-logs/phase0-spike/RESULTS.md`.
- [x] Plan-engine correctness: the 2B's TC plan uses one column split (`sp=1`, no `k_tc_sum`)
  for HB=512, so T=1/4/8 are token-exact (4/4 prompts, 373.5-375.7 tok/s vs 190.8 plain; the
  T=8 verify plan matches the plain engine row for row). A longer n-gram spec run (k=7, 160
  tokens, repeated prompt) matched plain 160/160; every one of its 20 verify passes was T=8
  (140/140 drafts accepted; 593.0 vs 143.7 tok/s). HB=4096 (9B) is refused before graph build
  with a clear error; the 9B plain engine passes 4/4 at 87.8 tok/s. No 27B pack was present,
  so its unchanged HB=1024 path was not re-run. Evidence:
  `bench-logs/verify-2b-plan-tc1-tmax4-fixed.log`, `...tmax8-fixed.log`, `diag-verify8.log`,
  `diag-spec-long.log`, `verify-9b-plan-tc1-guard.log`, `verify-9b-plain-fixed.log`; `pytest tests/ -q` 58 passed.

Next, in order:

1. [x] Guard the GRUB edit. `cwsr_edit_grub()` backs the file up, rewrites it, re-reads
   `GRUB_CMDLINE_LINUX_DEFAULT`, and restores the backup and stops when the sed failed, added nothing,
   or dropped a parameter of the old line. `check_cwsr` calls it; the raw backup/sed lines are gone.
   Evidence: `python -m pytest tests/test_installer_grub.py -q` -> 9 passed, `bash -n install.sh` OK;
   with the guard block removed on a copy (`KINDI_INSTALLER=<copy>`) the two failure-path tests fail,
   so they are not vacuous. `--dry-run` needs Linux and was not run here.
2. [x] Request validation and boundary hardening in `kindi/server.py`, plus a complete
   `kindi.conf.example` regression test (`tests/test_config_example.py`). Malformed bodies and
   out-of-range fields return 400 from `_validate_request` before `K.check_prompt`/`K.generate`;
   a configured API key is compared with `hmac.compare_digest`; bodies over 16 MiB
   (`KINDI_MAX_BODY_MB`) return 413 without touching the prompt cache. Evidence: `python -m pytest
   tests/test_server.py tests/test_config_example.py tests/test_installer_grub.py -q` -> 20 passed;
   `python -m pytest tests/ -q` -> 58 passed; with `_validate_request` stubbed to a no-op the earlier
   validation tests fail, so they are not vacuous.
3. [~] Documentation sync. `WHY.md` now names 0.4.0 and no longer says "one CUDA file";
   `kindi.conf.example` now lists every key in `config.DEFAULTS` (including the five that were
   missing). **Still blocked:** `T:/kindi` has uncommitted changes in `docs/CONFIGURATION.md`,
   `docs/INSTALL.md`, `docs/CARDS.md`, `docs/BENCHMARKS.md`, `docs/MTP.md` and `docs/dev/`, so
   the central config reference, installer/checksum, support matrix and dead links cannot be
   changed here until that work is committed or handed off.
   Done when: every key in `config.DEFAULTS` appears in `docs/CONFIGURATION.md` and no doc link 404s.
4. [ ] Packaging. **Blocked/owned by the main worktree:** it has `setup.py` staged and is editing
   the packaging/docs surface. Do not add a competing `pyproject.toml` here; reconcile with the
   main copy first. CI (`.github/workflows/ci.yml`) is also staged there, so the fastapi test
   dependency and GPU-job changes must go into that file, not a second pipeline; use
   `tests/requirements-test.txt` (pytest, fastapi, pydantic, httpx, numpy) as the dependency list.
5. [ ] Feature work is tracked separately: `GOALS.md` / `IMPLEMENTATION_PLAN.md` when present
   (Turing/CPU SIMD/tandem). Do not fork that roadmap here; update those files.
6. [x] Long-term single/dual card target written to `docs/CARD-SUPPORT.md`: seven requested cards, current
   installer/engine gates, one- and two-card acceptance matrix, and P0-P3 work. It records two blockers:
   the 8 GB RTX 4060 Ti is classified `pascal8` because `tc` also demands 9,500 MiB, and `pair_ok()` rejects
   every pair above sm_60. No installer/engine code changed here; the implementation is main-worktree work
   and no card is a support claim until a GPU verify is recorded.
7. [x] Long-term CPU target written to `docs/CPU-SUPPORT.md`: the four requested CPUs, the AVX2/FMA/F16C
   requirement, per-CPU single-socket targets, the i5-2500K fallback work, hybrid i9 thread tuning, and
   the CPU acceptance matrix. No CPU/installer code changed here; no CPU is a support claim until a
   native build and `kindi verify` are recorded.

Decisions not to undo without new measurements:

- CPU as drafter for speculative decoding on Pascal is rejected: a verify pass costs ~0.9 plain
  steps per token there, and the CPU 2B is slower than the GPU one. See the audit appendix.
- The GRUB sed must keep `\1`; the fixture test is the guard against a control byte creeping
  back in.

## Resume protocol

1. `git worktree list --porcelain`, `git status --short` and `git diff` - assume the previous agent
   stopped mid-change. Check the lease table and `T:/kindi/AGENTS.md` before touching a shared file.
2. Read this file, then the audit report if it is present.
3. Run the no-GPU gate; if it fails, fix that before starting new work.
4. Take the first unchecked action above, or the user's new instruction if it overrides it. Do not
   start a task whose shared files are owned by another worktree.
5. When done: commit the task's files on its branch, tick the action with one line of evidence,
   update the snapshot and lease table, and state what could not be verified.
