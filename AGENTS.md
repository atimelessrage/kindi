# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Current state (live — read this first)

- **HEAD:** `bcfa482` on both `main` and `freebuff/tandem-cpu-recovery` (identical, clean; guarded by `tests/test_agents_handoff.py`). Full suite last measured there: 564 passed, 10 skipped, 1 warning; the 27% line-and-branch coverage floor is unchanged.
- **Shipped on main:** dense tandem runtime (2B/4B/9B/27B rate, peak and exactness at
  512/4K/8K; 2B CPU-layer sweep; joint checkpoint 149.6 MiB on the 27B at 54.5-61.5 ms
  save), owned-slice CPU load, `kindi bench --sample-vram` sampled-peak VRAM, multi-ISA CPU
  dispatch, FP8 `mma.sync` kernels (no real model), MoE expert-cache prototype (no backend),
  rate limiting, JSON access log, coverage floor, perf-baseline gate, CI static analysis.
- **Open gates:** installer pin/version (release owner), tandem-beats-GPU (tuning), a
  device where full-GPU cannot run yet a split can (on the 4060 Ti the loader reduces the 27B to
  ~37K full-GPU / ~30K tandem tokens, so the split buys headroom, not context), GPU CI runner registration (steps in `docs/CI.md`), mixed-card/P106/Arc
  hardware runs, whether the tandem helps pre-tensor-core cards (Pascal, M4000 — measured cells
  defined in `docs/ROADMAP.md` Cards), the remaining eight prior-art items in
  `docs/EXTERNAL-ENGINES.md` (the first - the measured CPU/PCIe ratio and the `rate` planner - is
  done; its beat clause measured 8.0 vs 7.9 tok/s, not met),
  real FP8/BF16 models, Dory D1, Raon/Gemma 26B backends, AVX-512. See
  `docs/ROADMAP.md`. 27B boundary: 24K needs >=12 CPU layers here with the plan owner (the `rate`
  pick); 24-32 layers buy memory at a 35-50% rate cost.
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
| `T:/kindi` | `main` | `bcfa482` | main checkout; clean; best-of main |
| `<worktree>` | `freebuff/tandem-cpu-recovery` | `bcfa482` | this session; best-of main (identical to `main`) |

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
5a. Every commit that touches `AGENTS.md` refreshes the worktree table hashes (to HEAD after the
   commit, i.e. the commit's own parent) in the same commit. A bump commit cannot name itself, so the
   parent is the value; a content commit must not leave the table older than its parent.
   `tests/test_agents_handoff.py` enforces both the table value and the bump in the diff.
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
| Merge upstream Kindi 0.6.0 into main (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `install.sh`, `kindi/server.py`, `kindi/spec.py`, `kindi/config.py`, `kindi/bench.py`, `kindi/support_matrix.py`, `kindi.conf.example`, `docker/Dockerfile*`, `README.md`, `WHY.md`, `SECURITY.md`, `docs/{CARDS,CONFIGURATION,DOCKER,INSTALL,MTP,WINDOWS,AMD}.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/SUPPORT-MATRIX.md`, `tests/{test_server,test_installer_cards,test_installer_source,test_no_secrets}.py`, `AGENTS.md` | done. `origin/main` (ahtx/kindi 0.6.0, 3 commits, 105 files, +52,683 lines: MoE backend `kindi/moe*.py`, DeepSeek-V4.1-Flash `kindi/dsv41*`, Nemotron Lightning `kindi/lightning*`, DSpine drafter `kindi/dspine*`, container entrypoint + compose, Kindi Bridge, `install.ps1`, turn-save cache, priority scheduler, `/v1/tokenize`+`/detokenize`, `dry_run`) merged into the fork (172 commits diverged at 0.5, `d9fce60`). Only 18 files conflicted (32 hunks). Resolutions: server completions take upstream SCHED/priority/`run_generate`/`after_turn` flow wrapped in the fork's `_logged` JSON access log and `gate` queue bound (released in relay/`finally`, incl. the keepalive path); stats keep the fork's superset keys under an RLock; `/health` unions both queue views + `drafting`; `presence_penalty` kept as a config float key (upstream moved validation to the server; the config default and `profiles.py` still need it); spec keeps the fork's missing-head fallback plus upstream DSpine; installer takes 0.6.0, upstream card classes (M5000 sm_52 fix, Lightning offers, DSpine profile) plus the fork's pinned source archive re-pinned to the published v0.6.0 tarball sha256 `613f7320...` (downloaded and hashed), P100 sm_60 and GT1030 classes, and the verify-speed pipefail guard; Dockerfiles take upstream `kindi.container` entrypoint/labels with the fork's `/health` HEALTHCHECK restored; WINDOWS.md concatenates upstream WSL2 with the fork's native-CPU quickstart (corrected: installer is no longer Linux-only). Fallout fixed: `lightning-30b` STATUS row, `kv_cache` example key, regenerated config reference (36 keys, 185 reads) and support matrix (7 models), image tags/version strings to 0.6.0, `/home/kindi` container-user allowlist in `test_no_secrets`, test doubles gained `after_turn`/`drafting`/`progress`. Checks: `pytest -q` **564 passed, 10 skipped**; `bash -n install.sh`, `compileall`, `config_reference --check`, `support_matrix --check`, `project_memory --check`, perf gate, `git diff --check` pass. Not verified: no GPU run of the new 0.6 backends (MoE/DeepSeek/Lightning/DSpine) and no Windows `install.ps1` run in this session. The split-ratio beat clause from the row below is recorded **not met** (8.0 vs 7.9 tok/s at 24K, 6,151 MiB peak). |
| CPU-vs-PCIe ratio and the `rate` split planner (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `tools/measure_split_ratio.py` (new), `kindi/tandem.py`, `tests/test_tandem.py`, `tests/test_measure_split_ratio.py` (new), `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-split-ratio/` (new), `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done. The ratio is measured live on the 4060 Ti + 8086K with the 27B pack: PCIe pageable-staged 4.5/2.3 GiB/s (256 MiB), hidden handover 0.009 ms (never the constraint), raw CPU GDN layer 4-6 ms, and the **effective** CPU GDN layer in the live tandem pipeline **11.7 ms** (~22x the GPU's 0.517 ms/layer, so the pipeline dispatch is half the split's cost, not the PCIe boundary). `tools/measure_split_ratio.py` writes a pack-hash-keyed `kindi-split-ratio` JSON (PCIe, CPU layers, a `TandemEngine` step probe per N in a fresh subprocess, effective CPU ms, a predicted tok/s table at 8K/24K, validation vs the recorded sweeps) and `--track`s it into `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-split-ratio/split-ratio.json`. `kindi/tandem.py` gains `load_ratio()` (pack-hash match, `KINDI_TANDEM_RATIO` path override, `~/.cache/kindi/tandem/` scan, and since the run: a live measurement with a non-null `gpu` outranks a stale no-card file - the first bench run picked up the stale `nocard` file, so the cache now prefers measured files by `gpu` presence then `measured_utc`; test in `test_tandem.py`), `plan_cpu_layers_rate()` and the `--tandem-cpu-layers rate[:frac]` form: bare `rate` takes the fastest fitting `gdn:` count, `rate:0.25` the largest count the calibration still predicts at >=25% of the full-GPU rate; without a calibration it falls back to the fit-only pick and says so (never fake numbers), and an unkeepable floor refuses. TandemEngine wires `rate`/`rate:frac` with the pack hash. Prediction gate: `rate` predicts the smallest fitting count (12 at 27B/24K with the plan owner), matching the recorded `auto` best-fit (11), instead of the hand-set 16 (5 steps further, slightly slower) - **predicts the best gdn:N within one step: met**. Validation errors vs the recorded 8K sweep are −10%...−25% (the model is systematically a bit pessimistic). Live 27B@24K rate-pick bench at 12 CPU layers: read 8.4 tok/s, generation 8.0 tok/s at 24K, sampled peak **6,151 MiB** (298 MiB less than plan `auto` 7.9 @ 6,449, 1,593 MiB less than plain `gdn:16` 7.9 @ 7,744) - **beat clause NOT met** (+1.3% over 7.9 at a lower peak, not +10%); ROADMAP item 1 marked met (prediction) / not met (beat) with the measured verdict in RESULTS.md. 10 new CPU-only tests (6 `rate` planner + parse/floor/fallback/`load_ratio` with the live-over-nocard preference, 4 tool round-number); `pytest -q` **564 passed, 10 skipped**; `config_reference --check` (regenerated 131 reads; `KINDI_TANDEM_RATIO` added), `project_memory --check`, perf gate 12 cells pass. |
| Support matrix takes a verify_matrix results.json (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/support_matrix.py`, `tests/test_support_matrix.py`, `docs/benchmarks/card-results.json` (new), `docs/SUPPORT-MATRIX.md`, `docs/OLD-CARD-RIG.md`, `docs/CARD-SUPPORT.md`, `AGENTS.md` | done. **The rig run now updates the card rows without hand-copying numbers.** `python -m kindi.support_matrix --from-verify docs/benchmarks/<date>-<card>/results.json` folds a `tools/verify_matrix.py` run into the tracked registry `docs/benchmarks/card-results.json` (per card: cells, evidence path, started_utc, git revision) and regenerates `docs/SUPPORT-MATRIX.md`, which gains a `## Cards` section (P106/P4000/M4000, arch, installer class, per-model cells, evidence). Cells are derived from the run's own rows (`plain 4/4 @ 30.1 tok/s · tc refused`, tie and first-diff forms, `not run` for skipped models); a refusal is recorded as a result, never an error. The tool refuses: a source under `bench-logs/` (the link would fail the docs-link guard - the runbook now says to copy the run into `docs/benchmarks/<date>-<card>/` first), an unknown recorded card name, and a multi-GPU run without `--card <slug>`. `--card` also covers a CPU container with no recorded GPU. 6 new tests; `tests/test_support_matrix.py` fails until the file is regenerated, as before. `docs/OLD-CARD-RIG.md` "What to send back" is now a 3-line command block ending in `--from-verify`; `docs/CARD-SUPPORT.md` names the matrix section as where measured numbers land. **Not done: still no rig run exists** - the registry is empty (`{"cards": {}}`) and all three cards render `not measured` in SUPPORT-MATRIX.md. Checks: `pytest -q` **554 passed, 10 skipped**; `support_matrix --check`, `project_memory --check`, `git diff --check` pass. |
| P4000 and M4000 card pages (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `docs/P4000.md` (new), `docs/M4000.md` (new), `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/README.md`, `AGENTS.md` | done. Both cards get a per-card page modeled on `docs/P106.md`: status banner (installer support implemented and tested, not measured, not a support claim), official specs, a what-Kindi-does-today table, the installer classification from `install.sh` source (`M4000` pattern -> CC 5.2 -> `maxwell`, offers 9b nemotron 4b 2b, no 27B; `Quadro P` -> CC 6.1 with 8,192 MiB >= 7,500 -> `pascal8`, offers 27b), the one-card/pair commands and the `verify_matrix` one-command record (`tools/verify_p106.py` refuses on both cards by name), the shared tandem gate, and a verification-record table with the checks still to run. `docs/CARDS.md` gains a proper `## Quadro P4000` section next to the M4000 one (silicon, class, pairs, `KINDI_TC=1` as the dp4a experiment, page pointer) and the M4000 section gains the silicon/class/pair/refusal rows; `docs/README.md` and `docs/CARD-SUPPORT.md` link both pages. **Not done: still no run on either card** - the pages are records to fill in, and their tables say so. One invented claim ("27B/KV at 8 GB: claim 24K") was written and deleted before commit; CARDS makes no context claim for either card. Checks: `pytest -q`, `test_docs_links`, `project_memory --check`, `support_matrix --check` (numbers in the commit message). |
| P106/P4000/M4000 rig preparation (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `tools/verify_matrix.py`, `tests/test_verify_matrix.py`, `docs/OLD-CARD-RIG.md` (new), `docs/README.md`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/P106.md`, `AGENTS.md` | done. The three unmeasured cards now have a session script instead of only installer routing: `docs/OLD-CARD-RIG.md` lists the host/toolchain prerequisites (CUDA 10.2.89 + CuPy 9.6 image, driver >= 440.33), per-card commands (matrix, bench with `--sample-vram`, serving/cache, 27B on the 8 GB cards, mixed pair with `--cards 2`), what to send back, the five-criterion acceptance rule, expected-vs-unconfirmed outcomes per card (sm_52 refuses `KINDI_TC=1` by design; P4000 dp4a is the interesting experiment; 27B on 8 GB is the likely first failure) and the ordered fallbacks (`KINDI_TC`, `KINDI_TP`, `KINDI_G5`, `KINDI_CHUNK_Q8`, `KINDI_KV`). `tools/verify_matrix.py` now records an `environment` block in `results.json` - card name, compute capability, driver, VRAM, CuPy/CUDA runtime, and per pack `source_sha256`/fold/size - and prints a `recorded: ...` line; without `nvidia-smi` it records "no card detected" instead of failing, so the same command works in a CPU container. 3 new tests. **Not done: no run on any of the three cards** - this is preparation, and CARD-SUPPORT stays unmeasured for all three. Checks: `pytest -q` **548 passed, 10 skipped**; docs links, `project_memory --check`, `support_matrix --check` pass. |
| Current-vs-0.5 report refresh + 9B tc default (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/engine.py`, `tests/test_plan_engine.py`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done. (1) `CURRENT-VS-KINDI-0.5.md` gains §21: 21.1 new/better (9B default, plan-engine GPU owner, plain q8 K/V, prior-art gates, the speculation rule, harness), 21.2 the honest worse list (installer pin and version identity unchanged and release-blocked, tandem still slower than GPU-only, `KINDI_TC_HP` T=4 cap, 4B tc tie, k8v4, no GPU runner, single-session rows), 21.3 what was fixed rather than listed. Two §19.2 rows are marked stale: `/health` still returns the full detail to an authenticated caller, and the split now holds 49,152 tokens, so it no longer "buys headroom, not context". (2) **9B tensor-core path is exact by default**: `engine.tc_hp_for(man)` returns `"1"` for an HB-4096 pack when `KINDI_TC_HP` is unset, and `make_engine` sets it before importing `kindi.spec` (which reads it once and caps T at 4); an explicit `KINDI_TC_HP=0` still refuses. Verified on the RTX 4060 Ti with no variable in the environment: `kindi verify` **4/4, 122.4 tok/s** (previously a hard refusal; plain engine 74.6-80.0). 1 new test; CARDS/CARD-SUPPORT/ROADMAP wording updated from "refused by default" to the automatic selection. Checks: `pytest -q` **545 passed, 10 skipped**; `config_reference --check` (regenerated: engine.py is now a read site for `KINDI_TC_HP`), `support_matrix --check`, `project_memory --check`, perf gate 12 cells, `git diff --check` all pass. Not verified: the 4B/27B paths and other HB-4096 packs (the 9B is the only one here), and no CI run. |
| Audit fixes: optional accelerators, dead knob, bench JSON (2026-11-14) | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/spec.py`, `kindi/bench.py`, `tests/test_plan_engine.py`, `tests/test_bench_json.py`, `tests/test_config_reference.py`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/MTP.md`, `AGENTS.md` | done. (1) An optional speculation artifact no longer aborts startup: `spec.py` gains module-level `optional_path()`/`head_paths()` (a missing/unreadable head clears its variable with one `speculation off: ...` line), the two `MTPDraft` constructions and the 2B drafter build are wrapped (log and continue; `_building_drafter` now resets in `finally`), and the reserve's `os.path.getsize` is guarded. 4 new CPU-only tests in `test_plan_engine.py` (missing head, present head, mode-specific draft head, unset is silent). (2) `KINDI_STREAM_LOAD` was documented in `docs/CONFIGURATION.md:113` but read nowhere since streaming became unconditional (`tp.py:262 _stream_to_card`) - row removed, and the new `test_every_documented_switch_is_read_by_the_code` fails on any documented-but-unread switch (it failed on exactly this one before the row was dropped). (3) `bench_json` creates a missing parent directory and `run()` logs an OSError instead of a traceback, so a typo cannot cost a 27B run; one test. (4) `docs/MTP.md` gains the 27B-vs-9B drafter table (40.0 -> 62.5 tok/s at 43% acceptance vs 115 -> 113 at 41%) and the missing-file fallback. Verified at this commit: `pytest -q` **544 passed, 10 skipped** (6 new tests, was 538); `config_reference --check` (regenerated after the spec.py env-read line numbers moved), `support_matrix --check`, `project_memory --check`, perf gate 12 cells and `git diff --check` all pass. Evidence: the new config test failed on exactly `KINDI_STREAM_LOAD` before the row was dropped, so it is not vacuous. |
| Review/audit findings (2026-11-14 two-agent audit) | this worktree | `freebuff/tandem-cpu-recovery` | `tests/test_docs_links.py`, `tests/README.md`, `docs/CI.md`, `docs/FP8-BF16.md`, `docs/PROMPT-CACHE.md`, `docs/ROADMAP.md`, `docs/RAON-1-MINI-R0.md`, `docs/CARD-SUPPORT.md`, `docs/BONSAI-2-SUPPORT.md`, `docs/CPU-SUPPORT.md`, `docs/TANDEM-STATUS.md`, `docs/MIXED-CARD-ROADMAP.md`, `docs/benchmarks/{dory-d0,raon-r0,tandem-cache}/**`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`, `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md`, `docs/development/LESSONS.md`, `.pre-commit-config.yaml`, `AGENTS.md` | done. The biggest find is a **vacuous guard**: `tests/test_docs_links.py` skipped any path part named `.freebuff`, and this worktree lives under `~/.freebuff/...`, so it collected **0 of 92** files and stayed green. It now compares `relative_to(ROOT)` parts; on its first honest run it found 6 link labels naming `bench-logs/` while opening `benchmarks/` (relabelled) and 7 broken links in two pages promoted out of `bench-logs/` (dory-d0: `goldens/goldens.json`, `bf16/SHA256SUMS`, `harness/dory_golden.cpp`, `SHA256SUMS` promoted; the 1.3 MB `tensor-inventory.json` stays local raw and is now plain text; raon-r0: two `../../docs/` links one level too deep). Two new guards: link labels must point where the link goes, and no Markdown link may open gitignored evidence. Harness/reproducibility: the three `tandem-cache` scripts the tracked pages tell you to run are now tracked next to them, and their `D:/kindi-27b` pack/state paths became `$HOME/.kindi/...` (2 pages). Drift: the four sibling status pages that had no forward pointer (`CARD-SUPPORT`, `BONSAI-2-SUPPORT`, `CPU-SUPPORT`, `TANDEM-STATUS`) now name the authoritative page, as `docs/README.md` already required; `MIXED-CARD-ROADMAP` says its `PAIR-SETUPS` file exists only on the WIP branch. `tests/README.md` gains the measured per-module coverage table (31% total, floor stays 27% until a Linux number exists). Pre-commit gains `check-merge-conflict`/`check-case-conflict` (ShellCheck still CI-only: CRLF noise). VERIFIED ALREADY FIXED earlier today, not re-done: N1 handoff guard (rule 5a), N2 per-device VRAM sampling, N3 estimate head charging/`state_bytes`, N4 `gdn:` SystemExit, N5 `ctx_reduced`, F1/F2 generic audit-file ignores (0 tracked), and the tandem.cpp goal is now recorded (`docs/EXTERNAL-ENGINES.md`). REJECTED as misreads: the "absent bench-logs evidence" list is mostly example commands and WIP-branch references (LESSONS L-0009); `optimizing-mixed-gpu-rig.json` and moving `CURRENT-VS-KINDI-0.5.md` stay as decided on 2026-11-14. |
| Prior-art review: llama.cpp 1.0.0, Strata 0.2.13, tandem.cpp | this worktree | `freebuff/tandem-cpu-recovery` | `docs/EXTERNAL-ENGINES.md` (new), `docs/ROADMAP.md`, `docs/README.md`, `AGENTS.md` | done, planning only (no code): nine ranked items with numeric gates in ROADMAP "Prior-art follow-ups (2026-11-14)" — measured CPU/PCIe split rule, calibration cache, `kindi kld` quality gate, determinism gate, `kindi plan` dry-run, K/V prefix sharing, CPU-set thread count, FP8 K/V (gated on KLD), Windows unbuffered load; MoE cache semantics recorded as design parity. Notable correction: llama.cpp's refcounted checkpoints do not apply to our checkpoints (ours already copy state only) — the duplicates are parked/persisted K/V |
| Review finding fixes (2026-11-14 third audit) | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/vram.py`, `kindi/bench.py`, `kindi/tandem.py`, `kindi/convert.py`, `tests/test_agents_handoff.py`, `tests/test_vram_monitor.py`, `tests/test_tandem.py`, `tests/test_bench_json.py`, `docs/TANDEM-RUNTIME.md`, `docs/BENCHMARKS.md`, `AGENTS.md` | done. CONFIRMED and fixed: (1) handoff guard excused a stale table - the parent rule let be6e3a9 (a content commit) keep its parent's hash, so the test now also requires every commit that touches `AGENTS.md` to refresh the hashes (protocol rule 5a); (2) `--sample-vram` read GPU 0 only - it now parses the index column and maps the engine's ordinal through `CUDA_VISIBLE_DEVICES` (`vram.visible_device`); (3) the estimate charged the head even in gpu-first (322 MiB over-count) and hid the per-layer GDN state in a flat scratch allowance - `state_bytes()` now charges ~3.1 MiB per owned GDN layer and the head only when the GPU owns the last layer (27B 24K `gdn:16` plain estimate 6,208 MiB vs ~6,240 measured, was ~100 MiB under); (4) `gdn:` with no count is a `SystemExit`, not a `ValueError` traceback; (5) `ctx_reduced` lost its 47-token fudge (any `eng.ctx < ctx_requested` is a reduction); (6) `convert.py` closes its four multi-GB files through an `ExitStack` on a mid-convert error. REJECTED as a measurement error: "the plan-owner peak exceeds its estimate by 208 MiB" compares device-wide `peak_used` (which includes the ~950 MiB the desktop holds) with an engine-bytes estimate; against `peak_delta` the plan estimate is ~890 MiB conservative. STALE: the problems report is not tracked in either worktree (`git ls-files` is empty) and the audit files are already ignored (`.gitignore:34-35`). SKIPPED with reason: the 34 `json.load(open(...))` sites are short-lived pack reads (closed at GC, no locked files observed in convert/install); a sweep would touch 13 modules for no measured failure. |
| Old-card tandem measurement on the roadmap | this worktree | `freebuff/tandem-cpu-recovery` | `docs/ROADMAP.md`, `AGENTS.md` | done; Cards bullet + open gates: measure whether (and which placement of) the CPU+GPU tandem helps Pascal/M4000 cards, cells defined |
| Plain-engine q8_0 K/V + 27B split re-measure | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/engine.py`, `kindi/tandem.py`, `kindi/__main__.py`, `tests/test_tandem.py`, `tests/test_plain_kv.py`, `docs/TANDEM-RUNTIME.md`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/ROADMAP.md`, `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-plain-q8/`, `docs/development/requirements/REQ-0013.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; plain engine `KINDI_KV=q8_0`/`kv=` reuses the plan kernels (`k_store_q8`/`k_q8row`/`k_attn_sq8t`/`k_attn_vq8w`/`k_attn_combn`), chunked prompt path off in q8 mode, K/V exposed as `tp.Dist` views (cache round-trip exact); standalone stays fp16; tandem plain owner always q8. Exact: 2B/9B/27B verify 4/4 (standalone q8 and tandem). 27B 24K `gdn:16`: plain-q8 peak ~7,110 MiB (995 free) vs 7,721 fp16, plan re-run 6,091 (1,985 free). Evidence: `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-plain-q8/RESULTS.md`; 529 passed, 10 skipped |
| External consultation, safe subset | this worktree | `freebuff/tandem-cpu-recovery` | `.gitignore`, `docs/README.md`, `docs/GOALS.md`, `docs/ROADMAP.md`, `docs/CI.md`, `docs/FP8-BF16.md`, `docs/PROMPT-CACHE.md`, `docs/RAON-1-MINI-R0.md`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `docs/benchmarks/{blackwell-fp8-ada-refusal-20261112,blackwell-fp8-dispatch-20261112,dory-d0,raon-r0,tandem-cache}/`, `README.md`, `AGENTS.md` | done (P0-1 root cleanup, losslessly): `*_ReviewAudit_*.txt`/`*_ProblemsOnly_*.txt` replace the per-model ignore patterns and `git rm --cached` untracked the tracked problems report (blob stays in history; file stays on disk, ignored). NOT done and why: `optimizing-mixed-gpu-rig.json` (3.1 MB, already ignored, sole copy of the mixed-gpu measurement — deleting it loses knowledge) and moving `CURRENT-VS-KINDI-0.5.md` (breaks ~20 links and user workflow). Doc consolidation reduced to an index (`docs/README.md`, names the authoritative page per subject; linking from root README) plus `docs/GOALS.md` (pointers only, no duplicated claims); no page was moved or demoted, and `docs/AGENTS.md` was not renamed (many links and lease rows cite it). P0-2 fixed by promotion, not by deletion: the 13 evidence files docs linked to from the gitignored `bench-logs/` are copied into `docs/benchmarks/<same subdir>/` (originals untouched) and all current-doc links re-pointed; `docs/development/AGENTS-HISTORY.md` is append-only so its citations keep the original local locations, and this row records the new locations. ROADMAP gains the CPU+RAM tandem scheduler goal with the >=25% gate (P1-3) and MoE marked design-only (P2-5 was otherwise already recorded). Skipped: GPU CI + the four-card benchmarks + AVX-512/Blackwell (need a registered runner or hardware), pre-commit shellcheck (CRLF false positives on Windows; CI already runs it), coverage floor raise (would redden CI — per-module numbers not raised). |
| Tandem plan-engine GPU owner | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/tp.py`, `kindi/tandem.py`, `kindi/engine.py`, `kindi/kindi.cu`, `kindi/bench.py`, `kindi/verify.py`, `kindi/server.py`, `kindi/__main__.py`, `tests/test_tandem.py`, `tests/test_plan_engine.py`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/development/**`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; `TPEngine(layers=,embed=,head=)` owns an arbitrary subset (only the owned tensors load, K/V only for the owned attention layers, f32 per layer) and steps one prebuilt plan per block with the residual add inside the plan (`k_padd`; a CuPy add per block cost 17% of a 2B token). `--tandem-gpu auto|plan|plain` (`KINDI_TANDEM_GPU`) picks the owner, plan on one card; `gpu_estimate(..., owner=)` charges the owner's real allocation. Exact: 2B cpu-first and gpu-first 4/4, 9B `gdn:6` 4/4, 2B joint cache 320/325 + 256/261. 27B 24K `gdn:16`: peak **6,004** MiB (n=5, 5,862-6,083) vs **7,721** plain, 1,985 MiB free, decode inside noise. 48K: 19 GDN layers hold all 49,152 tokens (peak 6,876) where unsplit reduces to 30,976. Fixed the plan engine's 1024-token floor for partial engines (256). Evidence: `bench-logs/tandem-plan-owner/`, `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-plan-owner/RESULTS.md`; 516 passed, 19 skipped |
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

- The no-GPU gate passes: `bash -n install.sh`, `compileall`, `pytest -q` and the GRUB fixture test
  (the rows above that need no GPU).
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
