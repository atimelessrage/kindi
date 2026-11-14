# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Current state (live — read this first)

- **HEAD:** `9cb4887` on both `main` and `freebuff/tandem-cpu-recovery` (identical, clean;
  guarded by `tests/test_agents_handoff.py`). Full suite last measured there:
  494 passed, 19 skipped, 1 warning; the 27% line-and-branch coverage floor is unchanged.
- **Shipped on main:** dense tandem runtime (2B/4B/9B/27B rate, peak and exactness at
  512/4K/8K; 2B CPU-layer sweep; joint checkpoint 149.6 MiB on the 27B at 54.5-61.5 ms
  save), owned-slice CPU load, `kindi bench --sample-vram` sampled-peak VRAM, multi-ISA CPU
  dispatch, FP8 `mma.sync` kernels (no real model), MoE expert-cache prototype (no backend),
  rate limiting, JSON access log, coverage floor, perf-baseline gate, CI static analysis.
- **Open gates:** installer pin/version (release owner), tandem-beats-GPU (tuning), a
  device where full-GPU cannot run yet a split can (on the 4060 Ti the loader reduces the 27B to
  ~37K full-GPU / ~30K tandem tokens, so the split buys headroom, not context), 9B plan engine
  (Hadamard 4096), GPU CI runner registration (steps in `docs/CI.md`), mixed-card/P106/Arc
  hardware runs, real FP8/BF16 models, Dory D1, Raon/Gemma 26B backends, AVX-512. See
  `docs/ROADMAP.md`. 27B boundary: 24K needs >=16 CPU layers here; 24-32 layers buy memory at
  a 35-50% rate cost.
- **Shared-file owners:** the newest lease row in the table below owns its files until
  marked done; `T:/kindi/AGENTS.md` is the main worktree's bridge note, not this file.

## Project memory (permanent)

Requirements, architecture, design decisions, lessons and operational knowledge live in
`docs/development/`; start at `docs/development/README.md`. The project-local workflow is
`.agents/skills/project-memory/SKILL.md`. Each requirement is one stable
`docs/development/requirements/REQ-NNNN.md` file; `python tools/project_memory.py` records
its added, last-modified and deleted commit hashes in
`docs/development/REQUIREMENTS-HISTORY.md`.

Update the internal record and the relevant user-facing docs/examples when behavior changes.
For roadmap work, reconcile `docs/ROADMAP.md` against requirement status and
`REQUIREMENTS-HISTORY.md`: an item is not done until its requirement evidence and commit
hashes are recorded; keep deleted/retired history instead of rewriting it. After a context
reset, start at `docs/development/README.md`, read `REQUIREMENTS-HISTORY.md`, inspect old
requirement states with `git show <hash>:docs/development/requirements/REQ-NNNN.md`, and run
`python tools/project_memory.py --check` before making new claims. Before handoff run
`python tools/project_memory.py --check`,
`python -m pytest tests/test_project_memory.py tests/test_docs_links.py -q`, the normal
CPU gate, and `git diff --check`. Do not leave durable knowledge only in chat.

## Worktree and shared-file protocol (2026-11-08)

This session runs in a **linked worktree**, not the main checkout. The two working trees share the
Git object database, not the index or the files on disk:

| worktree | branch | HEAD | role |
|---|---|---|---|
| `T:/kindi` | `main` | `9cb4887` | main checkout; clean; best-of main |
| `<worktree>` | `freebuff/tandem-cpu-recovery` | `9cb4887` | this session; best-of main (identical to `main`) |

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

### Reconciliation (2026-11-09)

The main worktree was interrupted mid-edit: `install.sh` was empty, `kindi/tandem.py` did not parse,
and `tandem_new.py`/`hostram.py` were placeholders. Its full tracked and untracked state is preserved
in `bench-logs/main-backup-20261109-045352/` and on the branches
`wip/main-interrupted-snapshot-20261109-045352` and `wip/main-interrupted-backup-20261109-045352`.
The main worktree is clean now; `main` was fast-forwarded to this branch after its three Turing
commits were merged. The corrupt tandem WIP was not merged into `main`; restart it from
`docs/CPU_GPU_TANDEM_DESIGN.md` and the WIP branch if it is resumed.

### Current leases

The full dated lease table (one row per task since 2026-11-08, with its evidence, ~110 rows) is in
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md). Keep this table to the handoffs
the next agent needs in front of them; when a task finishes, move its row to the history file.

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| Cross-mode tc rows + 27B gpu-first 24K row | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/build_cross_mode.py`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `docs/benchmarks/perf-baseline.json`, `docs/benchmarks/2026-11-13-i7-8086k-rtx4060ti-tc/**`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `tests/test_cross_mode_benchmarks.py`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; tc rows are medians of three (2B read 471.7, 4B 214.1, 9B 133.9; 24K gen 299.2/127.8/97.0, peaks 1,645/3,311/3,257 MiB) and the perf gate now guards 12 cells; 27B gpu-first 24K (read 8.5, gen 7.4/6.2/4.5/3.3/2.7, peak 7,339 MiB) completes the 27B table |
| verify_matrix tc-hp mode | this worktree | `freebuff/tandem-cpu-recovery` | `tools/verify_matrix.py`, `tests/test_verify_matrix.py`, `docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; the matrix gained a `tc-hp` mode (tc + `KINDI_TC_HP=1`) because without it the 9B tc cell can only ever record `refused`; measured 9B tc-hp PASS 4/4 at 123.9 tok/s (docs: ROADMAP plan-engine bullet now names tc-hp) |
| Audit findings: dispatch, JSON ctx, tc-hp defaults | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/engine.py`, `kindi/bench.py`, `kindi/spec.py`, `tests/test_bench_json.py`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; plan engine is now the default for every model it supports (`KINDI_TP=0` keeps plain) and the 9B log line states the HP requirement; bench JSON carries ctx_requested + ctx_reduced. Evidence: 2B default verify 4/4 at 253.4 tok/s via the plan engine; new test pins the reduced-ctx JSON; full focused set 17 passed; config_reference --check, perf gate, git diff --check pass |
| 27B boundary sweep 8K (flat exchange) | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; 27B cpu-first at 8K, 8/12/16/20/24/28 CPU layers: peak 7,427 -> 5,277 MiB (428-432 MiB saved per 4-layer step) for read 16.8 -> 6.5 and 8K decode 10.7 -> 3.2 (15-37% cost per step); no knee — the exchange rate is ~27-29 MiB per percent of decode; 24K boundary is the capacity threshold at 16 layers (687 MiB free, kept) vs 8 (reduced to 17,664, 23 free) |
| Tandem placement: GDN-first CPU layers | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/tandem.py`, `kindi/engine.py`, `kindi/bench.py`, `kindi/__main__.py`, `tests/test_tandem.py`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; `--tandem-cpu-layers` accepts a count, an explicit set (`0,1,2,4` / `0-2,4-6`) or `gdn:N` (first N gated-delta layers, attention stays on GPU), so the CPU runs the layers whose per-token cost does not grow with context. Medians of three at 24K: 9B 23.6 vs 11.0 tok/s and 27B 7.9 vs 2.9 for the same CPU layer count (2.1x / 2.7x), because the contiguous split loses 55% of its rate from 2K to 24K and GDN-first loses 13%. Both exact: 2B 4/4 with the integer and `gdn:` forms, 9B 4/4 with an explicit set. Also fixed: the partial-range engine charged the GPU owner for every attention layer in the context fit, so the 27B with 8 CPU layers now holds 19,712 tokens instead of 17,664. Two measured limits documented: the 9B split saves no peak VRAM on this card (~5 GiB free, 2.3 GiB pack) so "no split" is right there, and a split frees little VRAM because its GPU owner is the plain engine (`KINDI_TP=0` cannot load the full 27B at 24K at all, the plan engine can) — next step is a plan-engine owner or q8_0 K/V in the plain engine. Evidence: `bench-logs/tandem-improve/`; 503 passed, 19 skipped; config reference regenerated; perf gate 12 cells; support_matrix --check, git diff --check pass |
| Tandem `auto` placement planner | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/tandem.py`, `kindi/__main__.py`, `tests/test_tandem.py`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; `--tandem-cpu-layers auto[:MiB]` computes the smallest GDN-first CPU set that makes the context fit from the pack's own tensor sizes (`pack_bytes`), fp16 K/V per kept attention layer, the head, the whole f32 file and a measured 192 MiB scratch allowance, checked against free VRAM before loading; it refuses instead of splitting when the model already fits and names the shortfall when nothing can fit (`KINDI_TANDEM_RESERVE_MB`, default 512). Estimate accurate to 5 MiB (27B at 24K: predicts 7,778, measured plan-engine peak 7,783). Real runs: 27B at 24,576 picks 19 GDN layers, keeps 24,576 unreduced, 522 MiB free, peak 7,257 MiB, read 8.3 / decode 6.3-7.0; 9B at 24,576 refuses with "~3282 MiB needed, 7075 MiB free". 6 new CPU-only tests (byte accounting, K/V charging, GDN preference, budget, refusal, wiring). Evidence: `bench-logs/tandem-improve/27b24-auto-r1.{json,log}`; 509 passed, 19 skipped; config reference regenerated (129 env reads); perf gate 12 cells; support_matrix --check, git diff --check pass |

## Layout

| path | what |
|---|---|
| `kindi/` | Python package: `engine.py`, `server.py`, `cache.py`, `spec.py`, `cpu.py`, `convert.py`, `config.py`; CLI in `__main__.py` |
| `kindi/*.cu` | CUDA kernels, compiled at runtime; `kcpu.c` is the AVX2 CPU engine |
| `install.sh` | the installer: machine check, Docker, models, `kindi verify`, services |
| `docs/` | user docs; `docs/CONFIGURATION.md` is the key reference |
| `docs/SUPPORT-MATRIX.md` | generated by `python -m kindi.support_matrix`; models, formats, K/V modes and CPU ISA from the code |
| `docs/ROADMAP.md` | the single index of archived no-go work, its gates and the live work |
| `tests/` | CPU-only unit tests (config, chat, tools, tokenizer, sampling, cache), the installer GRUB fixture test and the plan-engine contract test (`test_plan_engine.py`, stand-in `cupy`) |
| `docker/` | `Dockerfile` (CUDA 10.2) and `Dockerfile.rocm` |

## Setup

Native, on Linux with the driver and the CUDA 10.2 toolkit:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" \
            "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Docker, the recommended path for the old cards:

```sh
docker build -t kindi:0.5.0 -f docker/Dockerfile .
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.5.0 info
```

The image tag follows `kindi/__init__.py:VERSION`; `docs/DOCKER.md` and the Dockerfiles are
 authoritative when they disagree.

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
| `python -m compileall -q kindi` | python | every module parses |
| `python -m pytest -q` | `pip install -r tests/requirements-test.txt` | the whole CPU suite (`pytest.ini` sets `testpaths = tests`, so bare `pytest` is safe) |
| `python -m pytest -q tests/test_installer_grub.py` | python, `sed` | the installer's GRUB sed keeps existing kernel parameters |
| `python -m kindi.support_matrix --check` | python | the generated support matrix matches the code |
| `python -m kindi.config_reference --check` | python | the generated configuration reference matches the code |
| `python tools/project_memory.py --check` | python | the requirement history matches git |
| `python tools/check_perf_baseline.py` | python | the committed CROSS-MODE rows are within 15% of the perf baseline |
| `git diff --check` | git | no whitespace damage |
| `python -m kindi --help` / `python -m kindi info` | python | the CLI imports without a GPU; config, pack and GPU report |

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
  `kindi.conf.example`, and the regenerated `docs/CONFIGURATION-REFERENCE.md`).
- `python tools/project_memory.py --check` passes, and any new requirement or decision is recorded in
  `docs/development/` rather than only in chat.
- The AGENTS.md worktree table names the current HEAD (guarded by `tests/test_agents_handoff.py`).
- The report names what was not verified.


## History

The dated Snapshot, Handoff, lease history, Current plan and Resume protocol that used to live here were
moved verbatim to [docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-14
(this file had grown to 1,057 lines and the audit flagged it). Read them when you need the reasoning behind
an older decision; `docs/development/` is the durable record and this file is the live handoff.
