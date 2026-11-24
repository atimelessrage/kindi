# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Current state (live — read this first)

### Handoff

- **HEAD:** `e362a43` on `merge/upstream-0.7.0` (0.7.1 merge; 0.7.0 merge `35e4697`; guarded by `tests/test_agents_handoff.py`).
  **2026-11-24 (latest): the generic fallback backend landed, and the CPU half of REQ-0030 is measured.**
  REQ-0019 T4.4's recommended option is in the tree: `kindi/backends/llamacpp.py` (registered `llamacpp`) is a
  wildcard that declares no model ids/architectures, so `resolve` picks it only for models nothing specific
  serves, and forwards generation to a running `llama-server`'s `/completion`
  (`KINDI_LLAMACPP_BASE_URL`, `KINDI_LLAMACPP_API_KEY`); ChatML by Kindi, one non-streaming request, no
  constrained decoding, all stated in the module. 12 CPU tests with an injectable opener; **the live
  `llama-server` gate is open** (no server on this host), so there is still no arbitrary-GGUF claim.
  REQ-0030 A1's native-CPU half: 2B on the CPU engine keeps the mask as the only native candidate
  (11.2 ms/token short / 1.1 long), tokenizer 0.034 ms and framing 0.0027 ms/chunk stay removed
  (`docs/benchmarks/2026-11-24-i7-8086k-cpu-req0030/`, claim `docs/claims/req0030-cpu-profile.json`).
  Fixed `tools/grammar_cost.py` `walk()` (broke when the mask cache became a bounded `StateMaskCache`), and
  the root of six adapters' mypy `assignment` finding: `Backend.capabilities` is now declared, not `None`
  (mypy 59 -> 53, baseline lowered with a reason). Suite: **1,633 passed / 47 skipped / 0 failed (241 s)**.
  **2026-11-24:** the 27B verifies on the CPU engine, and the "missing" compiler was installed.**
  `KINDI_DEVICE=cpu kindi verify` on the 27B pack is **4/4 token-exact** (40/40 per prompt) at 3.7 tok/s after a 340 s load;
  packcheck full (10 files, 18.4 s) and a structural audit (851 tensors, byte ranges tiling every file, reference ids in range)
  passed first, and `kindi plan` ran on it (8 GiB assumed: no split, 7,266 MiB at ctx 8192; the default max context plans 2 GDN CPU layers). Because
  the blocker was detection, not hardware: clang+LLVM 18.1.8 sits under Program Files while `clang` is not on PATH, so `_find_cc` resolves `KINDI_CC`, then
  PATH, then a standard LLVM install (newest first); the cache key names the compiler that will run, and `kindi caps` asks the same
  resolver. That unblocked REQ-0028 A4's live half (`kindi selftest`, 2B CPU: `dialects agree`, exit 0) and the
  `test_cpu_build.py` compile-and-load checks; `kindi plan` names the resolved pack and `kindi models` drops "too large" for
  the 2B/4B/9B without a card. GPU rows (tc/plain verify, K8/V4, bench, tandem) still need a card; `MANUAL-HELP-NEEDED.md` §3c holds the one open repo gate (4 unbaselined rows on `pr/07-intel-probe`), and that file is the only manual-help tracker (the local scratch copy is folded in and deleted).
  **2026-11-24:** `POST /v1/messages` answers Anthropic's dialect on the same `run_generate` path (`kindi/api/anthropic.py`;
  `x-api-key` or bearer; envelope and SSE order; one `_relay` for both streaming routes), and `kindi selftest` drives it in
  process and with `--url`. REQ-0023's block half landed too (`tests/test_kernel_arity.py`: 28 block/bound pairs, 135 pinned).
  The package and `install.sh` said `0.7.1-fork`; all three match now, pinned by
  `tests/test_version_consistency.py`. The `v0.7.1-fork` release is **still unpublished** (only `v0.5.0-installer`),
  so README's one-liners 404 until the owner runs `docs/RELEASE-CHECKLIST.md`. The mask cache is a bounded 64 MiB
  LRU (`StateMaskCache`, `KINDI_MASK_CACHE_MB`); A4/A5-rate open. Suite: 1,598 passed / 47 skipped / 0 failed (226 s).
  The v0.7.1 (`1cb59d2`) and v0.7.0 (`8298a0f`) merge details (Inkling 2 Mini, Flash-Next drafting,
  one CPU thread per physical core, the fork's own `tools/check_warp_masks.py`, the 2026-11-23 installer
  slice, and what stayed unrun) moved to
  [docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md), "The v0.7.1 and v0.7.0 merges".

### Python quality ratchet

Audit F9's Python quality ratchet is `tools/check_python_quality.py`: Ruff 0.16.0 (`F`, `E9`, `B`, `I`)
over the Python tree, mypy 2.4.0 over first-party production
code, Pyright 1.1.414 (`basic`, all platforms) and Pylint 4.1.3 (`E`, `F`, import-error
disabled) at Python 3.10 over the same production targets, recorded as **207 Ruff findings / 59 mypy
errors / 863 Pyright errors / 245 Pylint findings** in
`tools/python_quality_baseline.json`. CI runs it; a new or increased file/rule count fails, while
existing debt is tolerated and `--update --reason "..."` lowers the baseline after review; it refuses growth and records the reason and delta, so it cannot launder a new finding. Pyright and Pylint skip `VENDORED_UNWIRED` (qwen4exp/qx) and `UPSTREAM_UNVERIFIED` (`kindi/inkling2*`, merged but never run here); `kindi/server.py` stays in scope.
`tests/test_python_quality.py` + `tests/test_ci_config.py` **15 passed**; no GPU run. Dependency scanning is report-only in CI (`tools/check_dependencies.py`, `requirements-runtime.txt`); the gate waits for a reviewed baseline. The prior test task is below.

### Full CPU gate

The 2026-11-24 27B-CPU gate at `32075c4` is **1,598 passed, 47 skipped, 0 failed** in 226.48 s (the CPU build checks now compile and load every tier here instead of skipping); the 2026-11-24 grammar-mask gate at `ebf871f` is 1,513 passed, 58 skipped, 0 failed in 180.66 s (plain CPU suite, no coverage rerun); the 2026-11-22 prewarm gate at `4bab483` was 1,423 / 58 / 0 in 138.50 s and the 2026-11-20 integration gate at `82a72bf` 1,412 / 28 / 0 in 250.33 s. The last coverage-measured run remains `539a504`: **1,241 passed, 28 skipped, 0 failed at 46.36%** line+branch (23,504 statements, 8,394 branches; plain 1,241 / 28 / 0 in 227.62 s), and the consolidated 2026-11-19 commits (`f75253b` + `84b6686`) 1,357 / 28 / 0 in 229.53 s. Every skip is inventoried by node ID and reason in `tests/README.md`.

### Upstream v0.7.0 and comparison

**2026-11-23:** upstream `v0.7.0` (`8298a0f`) is merged into this branch; the current comparison is
`CURRENT-VS-KINDI-0.7.md` (0 commits behind, 702 ahead, 931 changed files). The PR status rechecked the same day:
**33 fork PRs (#47-#79) open, 0 merged, 7 drafts**, external #37 open and dirty; #48, #49, #72 and #76 are in v0.7.0 by
content and can be closed as included. The paragraphs below are the 0.6.1-era record that produced this tree.

Stock moved to upstream `v0.6.1` (`7fececb`, fetched 2026-11-17) and one of its fixes was a bug this fork had: `kindi/mtp.py` never passed `k_rmsfold_qT`'s
`hp`, so the MTP head's CUDA graph did not build and mode A speculation was dead on the tensor-core layout (fixed in `f934a55`, with
`tests/test_kernel_arity.py`); measured after the fix, mode A at the old default `KINDI_SPEC_K=4` costs 0.66x on the 2B; at
`KINDI_SPEC_K=1` it is 1.12-1.13x at 512 and 24K tokens of context and a wash at 8K (4-run medians, 2026-11-17, `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-mtp-context/RESULTS.md`), so mode A now
defaults to k=1 (`docs/MTP.md`), and `CURRENT-VS-KINDI-0.7.md` records the current official v0.6.1 comparison (664-commit / 827-file
delta at `53198e5`; the 2026-11-19 round adds request-shape validation, Windows secret-scan batching, MTP k=1, the Q2 expert lane, the controlled pairs and the chunk sweep; upstream `main` now points at the release commit `7fececb` (no post-release commit to port); round history in
`detailed-fork-changelog.md`); §4.6 closes the same-settings A/B with stock `v0.6.1` (tag source, loader shim) and this
tree at `a79137f` on the same 2B/9B packs at ctx 8192, plain engine and exact head, five passes each: every decode median is within 2.6% and the ranges overlap,
while the 9B prompt-reading gap (83.9 vs 72.4) was rechecked at 20 passes and closed as noise (70.7 vs 75.7; paired -1.4, p=0.648); a separate stock harness
adds literal position-512 medians (190.3 on the 2B, 84.5 on the 9B) outside the bench sweep, after a 2026-11-18 paired run showed its position-100 anchor
matching the immediately preceding bench (+1.2%/0.0%) and replaced the earlier cold-clock-heavy 171.2/80.7 values, and the four sweep cells now carry three-pass
medians with ranges (the pages note that the 9B prompt-reading ranges overlap, so no reading shape is claimed). §4 is re-based at `0184bb8` into separate worse
/ missing / unverified tables with a closure condition per row; it distinguishes the 16-file `v0.6.0..v0.6.1` range from the 15-file 0.6.1 commit (the compose
surface is present from `dce89df`) and records the MoE backend's status (exercised on GPU through Lightning). Upstream `v0.6.0` (`179710c`) + `origin/main` head
(`dce89df`, v0.6.1's parent) both contained, while `v0.6.1` itself is one commit past `dce89df`; vs `v0.6.0` the tree is a strict superset; comparison report in
`CURRENT-VS-KINDI-0.7.md`.

Dated notes (MTP smoke guard, bench split, audit follow-up, commit-history redaction, Ollama links) moved to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-23.

### Engine-liveness roadmap

The general lessons from that bug are on the roadmap, not started: `docs/ROADMAP.md` "Engine-liveness follow-ups (2026-11-17)" ranks eleven follow-ups and takes
four with their gates first — a launch-shape contract for every kernel call site (REQ-0023 — A1+A3 done 2026-11-17: 105 launches measured across
`spec.py`/`dspine.py`/`mtp.py`/`tp.py`, 37 unresolvable sites pinned with reasons, every measured site proven by mutation; the dspine `kf`/`self.k`/`self.k8`
tables are resolved from their literals and only two dspine sites remain pinned; the grid/block half stays open), report the effective speculation state plus a
`mtp_heads.json` digest check (REQ-0024 - A1+A2+A3+A4 done 2026-11-17: one mode resolver behind `/health`, the startup log and `kindi info`, the head's sha256
and size checked before `np.load`, and `/metrics`' `kindi_spec_mode_effective` naming the effective mode), `kindi verify --spec` as an exactness gate (REQ-0025
- A1/A2/A4 done and A3 four of five, 2026-11-17: the mode is printed first, a missing drafter exits 2, and the 2B is token-exact 4/4 in plain, `ngram`, `mtp`,
`draft2b` and `both`; `dspine` waits on weights, `docs/MANUAL-HELP-NEEDED.md` row 7) and `kindi bench --spec` (REQ-0026; REQ-0025 A6 now also runs the four
modes as `tc-<spec>` cells in `tools/verify_matrix.py`, scoring requested/effective and drafts and skipping absent files so a baseline gates them); the seven
parked ones name the measurement that would reopen them.

### I/O and cold tier

`kindi io` now measures this host's memory tiers (RAM copy, unbuffered disk read, pinned PCIe both ways) so a cache or streaming budget is an input, not an
assumption; the mixed-precision tiers stay parked, while the CPU-compute lane has an exact expert-level gate for the Lightning Q2_0 format and is still off by default (the copy lane won on this host, and the model row needs the absent pack) in `docs/ROADMAP.md` "Memory-tier follow-ups (2026-11-17)"; I/O-aware slot sizing now
consumes those rates in `kindi plan`/`kindi autotune`; the plan's follow-up bench carries the planned context and count with the global `--ctx` and portable
`--dcache` flags and a test pins both, the printed `bench` and `verify` commands quote spaced interpreter and pack paths, `kindi autotune` adds an `expert-cache-<MiB>` cold-tier candidate that sets `KINDI_LT_COLD=1` + `KINDI_LT_HOST_MB` from
the same rate-sized working set and caps it by the host experts, and `kindi io --json FILE` reports can be reused by either command through
`--io-report`/`KINDI_IO_REPORT` (a missing or unusable report names itself and falls back to one probe), with the arithmetic unit-tested and its Lightning-pack
comparison still open; the NVMe cold-expert tier is **started 2026-11-17** behind `KINDI_LT_COLD=1` (`kindi/cold_experts.py`: the hottest host experts stay
pinned under `KINDI_LT_HOST_MB` (default available RAM less 10%/1 GiB), only the overflow is file-backed, unbuffered aligned reads, coalesced gaps, a bounded
lookahead prefetch queue (a daemon thread fills it while the current layer's kernels run; the next cold layer joins before its demand read), Q2_0 de-interleave,
requested/transferred and exact pinned/disk byte accounting; prompts stage the host block with each pinned expert at its host-tier slot, `cold_stats()` counts
each pinned expert once while its byte totals stay per matrix, decode keeps a per-layer pinned slot cache (`cold_slot_plan`: a routed expert's slot is never
evicted and a repeated expert takes no read; `cold_stats()` reports `slot_hits`), the decode graph is disabled on this opt-in path, and the per-layer stream
sync plus blocking current-layer read stay open (C9; no tok/s claimed), unit-tested only with **no larger-than-RAM pack, verify, GPU run or tok/s claimed**, and
the Strata/KTransformers/BigMoeOnEdge triage is in `docs/EXTERNAL-ENGINES.md`.

### Suite and recent additions

The coverage-measured full suite at `539a504` is **1,241 passed, 28 skipped, 0 failed** in 277.17 s at
**46.36%** (23,504 statements, 8,394 branches; the plain run is 1,241 / 28 / 0 in 227.62 s). The per-round
list of new tests (kernel-arity resolvers, CLI comma values, saved I/O reports, autotune/cold-expert checks,
effective-mode metrics, speculation cells, I/O probes, MTP registry, secret-history guards) moved verbatim to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) on 2026-11-24.

### Secret guards

The secret guards are now three: entropy baseline over tracked files, pattern scan over tracked+untracked files, and a history gate against
`tools/secret_history_baseline.json` (160 rows / 3,318 legacy hits from an old benchmark dump, old AGENTS/log revisions, the 27B median transcripts and one commit message that names a
cloud storage folder, recorded not endorsed; the gate scans commit messages - subject and body - with the same rules as file content, so a path in a message is
caught instead of invisible).

### Plan slot arithmetic

Thirteen tests pinning `kindi plan`'s decode-slot arithmetic (the fit subtracts the slots before dividing over `n_moe - 2` layers, so 10 slots cost 20 experts
per layer on the 256-expert test pack; the line used to claim one per slot, `kindi/ltfit.py` is now the one implementation both `plan._lt_fit` and
`lightning._plan` call, and the weighted branch prices the slots against the same room with `dcache=0` - 9 of 128 card experts and 5.8 points of served share on
the test pack's tight card - in the line and in `slots_cost`).

### Coverage gate

The coverage gate `python tools/check_coverage.py` passes on 46 modules from a measured run (`coverage run --branch --source=kindi -m pytest` -> `coverage
json`, since `pytest-cov` hangs on this host; the 2026-11-18 historical run 1,196 passed / 28 skipped / 45.4% total is superseded by the final 539a504
run: **1,241 passed, 28 skipped, 0 failed at 46.36%**, 23,504 statements and 8,394 branches), and `tests/test_no_secrets.py` now scans untracked files too (pinned by
`test_untracked_files_are_scanned`), with `project_memory --check`, `config_reference --check` and `support_matrix --check` all clean.
The per-module rows in `tests/README.md` were checked against the final run's `.coverage` data; their rounded figures are unchanged, and the total and
per-module floors are untouched. Its tracked revision, command and counts are in
[docs/benchmarks/2026-11-18-i7-8086k-windows-cpu/full-suite.json](docs/benchmarks/2026-11-18-i7-8086k-windows-cpu/full-suite.json).

### Grammar and claims

[REQ-0027](docs/development/requirements/REQ-0027.md) passed its 10x gate on 2026-11-22 (token-class table + startup prewarm) and again on 2026-11-24 (edge-derived
safe sets, mask unions); its A4/A5 GPU rows stay open. The CPU set-equality guard against the pre-token-class `vocab_allowed` sweep is what keeps the accepted
language fixed. GPU grammar/JSON-schema gate closed on the 2B (§39); recursive GBNF is opt-in (`KINDI_GBNF_MAX_DEPTH`, bounded unroll cross-checked against an
independent matcher - see the newest rows in `docs/development/AGENTS-HISTORY.md`) and measured as a diagnostic only: it once cost 2,720 ms/token against 231 for
`json_schema` and *fell* to 463 at 200 tokens (cache hits; a whitespace-permitting constraint lets the greedy 2B spend the budget on 9 characters) (`docs/API.md`,
`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md`). The harnesses are tracked as `tools/grammar_cost.py` and
`tools/grammar_cost_serve.py`; every quoted number is re-derived, not prose:
`python tools/check_claims.py` re-derives every one of them from the tracked artifact of its run (38 gates, 270 numbers across `docs/claims/*.json`, six
claimed docs and zero citations listed as debt), and refuses a benchmark citation no claim accounts for - schema and procedure in `docs/claims/README.md`.

### Dated evidence

The 27B A/B, fail-closed gate, startup-warning, MTP-server and cold-expert bullets
(`Latest` through `Older still`) were moved verbatim to
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) under
"Current-state bullets moved from AGENTS.md (2026-11-18)".

- **Shipped on main:** dense tandem runtime (2B/4B/9B/27B rate, peak and exactness at
  512/4K/8K; 2B CPU-layer sweep; joint checkpoint 149.6 MiB on the 27B at 54.5-61.5 ms
  save), owned-slice CPU load, `kindi bench --sample-vram` sampled-peak VRAM, multi-ISA CPU
  dispatch, FP8 `mma.sync` kernels (no real model), MoE expert-cache prototype (no backend),
  rate limiting, JSON access log, coverage floor (per-module, including `engine.py` at its
  CPU-reachable coverage and every `kindi/backends/*` module), perf-baseline gate, CI static analysis.
- **Open gates:** installer pin/version (release owner), tandem-beats-GPU (tuning), a
  device where full-GPU cannot run yet a split can (on the 4060 Ti the loader reduces the 27B to
  ~37K full-GPU / ~30K tandem tokens, so the split buys headroom, not context; the 9B never
  reduces — measured 2026-11-17 up to 131,072 tokens on both placements, so a split there is
  2.8-4.6x cost for <0.9 GiB), GPU CI runner registration (steps in `docs/CI.md`), mixed-card/P106/Arc
  hardware runs, whether the tandem helps pre-tensor-core cards (Pascal, M4000 — measured cells
  defined in `docs/ROADMAP.md` Cards), the remaining eight prior-art items in
  `docs/EXTERNAL-ENGINES.md` (the first - the measured CPU/PCIe ratio and the `rate` planner - is
  done; its beat clause measured 8.0 vs 7.9 tok/s, not met),
  real FP8/BF16 models, Dory D1, Raon/Gemma 26B backends, AVX-512. See
  `docs/ROADMAP.md`. 27B boundary: 24K needs >=12 CPU layers here with the plan owner (the `rate`
  pick); 24-32 layers buy memory at a 35-50% rate cost.

### Owner actions

Owner actions pending: Nanbeige 33B GGUF replacement, Gemma 4 E4B pack, DeepSeek V4.1 shards/gate data, PQ2_0/Q2_0 quant downloads, a pinned llama-server, a Linux A770 host for the M2 target, other-hardware test rigs, push/installer/CI publications, T4.4/T4.5 + D3/D5 decisions — `docs/MANUAL-HELP-NEEDED.md`.
**A770 (2026-11-24, no claim):** `kindi intel-ternary` splits the packed GEMV over the weight phases (0 mismatches
at six shapes, 5.75x-14.14x) and `kindi intel-gemm` runs the engine's q8 GEMM bit-exact; pinning its work-group to `(256,1)` bought 1.71x (555->323 ms at 512x512x4096, and it was wrong across tiles before), `--staging local` another 1.02-1.05x, and all three OpenCL probes now take `--device-name Arc` because the driver's device order is not stable
(the iGPU has come back first). **`kindi intel-sycl` now passes on Level Zero**: oneAPI 2026.2 `icpx`, backend `ext_oneapi_level_zero`, driver `1.17.39894+6`, 0 mismatches at 1,048,576/1/1,000,003 elements, and `--selector opencl:gpu` exits 1 instead of falling back (`docs/benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/`). The earlier "no L0" was PATH: the adapter loads only from oneAPI's `bin` dirs, `intel-info` finds the install via `KINDI_ONEAPI_ROOT`, and the `sycl-ls` parser now accepts both bracket layouts. M2 still needs the Linux target; no SYCL backend or support claim.
- **Shared-file owners:** the newest lease row in the table below owns its files until
  marked done; `T:/kindi/AGENTS.md` is the main worktree's bridge note, not this file.

## Project memory (permanent)

Requirements, architecture, design decisions, lessons and operational knowledge live in
`docs/development/`; start at `docs/development/README.md`. The project-local workflow is
`.agents/skills/project-memory/SKILL.md`. Each requirement is one stable
`docs/development/requirements/REQ-NNNN.md` file. The external local-AI knowledge base at
The external knowledge base (its path comes only from `$KINDI_AI_INVENTORY`, because a synced-folder
path is personal data) is read-only inspiration (`.agents/skills/ai-inventory/SKILL.md`):
its numbers are hypotheses about someone else's engine, so an idea from there becomes roadmap work
only with a gate, and is a win only once measured here; what was read is recorded in
`docs/development/AI-INVENTORY-SCAN.md`; `python tools/project_memory.py` records
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
| `T:/kindi` | `main` | `a101b7d` | main checkout; clean; fast-forwarded to this round |
| `<worktree>` | `merge/upstream-0.7.0` | `67e1468` | this session; 2026-11-24 v0.7.1 merge, grammar-mask edges, Anthropic endpoint + A4 dialects, Arc SYCL smoke on Level Zero, Arc GEMM, bench --spec, 27B CPU verify + compiler detection |

Read its copy only while that worktree has uncommitted changes; otherwise this file is the source of truth.

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

### Consolidation audit (2026-11-17)

`main` and this worktree are the same commit and both clean, the local untracked comparison render is
now gitignored, and every remote ref is already contained: `origin/main` (`dce89df`), `myfork/main`
(`fd54148`) and `myfork/freebuff/tandem-cpu-recovery` (`437891e`) are ancestors of `main` (0 ahead,
358-415 behind). Two branches stay outside `main` on purpose, audited rather than merged because
merging either would import superseded or corrupt code:

- `wip/main-interrupted-snapshot-20261109-045352` — and the stash-format
  `wip/main-interrupted-backup-20261109-045352`, whose files are a subset of it — holds 38 paths that
  exist nowhere else: a doorbell/async-copy `tandem_kernels.cu` plus a `tandem_new.py` placeholder, an
  AVX-512/VNNI/AMX `cpuid` detection draft, the 2026-11-07/08 planning set (`GOALS.md`,
  `IMPLEMENTATION_PLAN.md`, `docs/GPU_PROFILE_ANALYSIS.md`, `docs/PAIR-SETUPS.md`,
  `researchAndComparisons.md`, `docs/dev/spec-measurement-plan-2026-11-08.md`) and one-off
  `fix_*.py`, `update_*.py`, `test_brace*.sh`, `test_pair_ok.sh` scripts. The async design it
  implements was measured unnecessary on 2026-11-15 (no dispatch gap; see
  `docs/TANDEM-OVERLAP-DESIGN.md`), the ISA draft has no AVX-512 kernels behind it and AVX-512 is
  already a named gate in `docs/ROADMAP.md`, the `pair_ok` bugs its analysis targets were fixed by
  `pair_fits` on 2026-11-14, and the planning docs are 0.4.0-era drafts superseded by
  `docs/ROADMAP.md`, `docs/GOALS.md`, `docs/CARDS.md`, `docs/MIXED-CARD-ROADMAP.md`,
  `docs/OLD-CARD-RIG.md` and `docs/EXTERNAL-ENGINES.md`. Nothing was merged; both stay archival.
- `clean/kindi-0.5` (1 commit, 633 behind) is the 0.5 comparison baseline, not a candidate. A
  2026-11-18 re-check at `f6e73ae` found 38 snapshot-only paths, 13 in the backup (a subset), and
  none in `clean/kindi-0.5`; the candidates are the same broken/unbuilt drafts, so nothing was
  extracted and no new test was added.

There is no TODO/FIXME or unfinished code path in `kindi/` or `tools/`; the open work is the
owner/hardware-bound gate list above.

#### Verified again and finished where possible (2026-11-17, second pass)

`main`, this session branch and both worktrees are the same commit and clean; `git stash list` is
empty; `origin/main` (`dce89df`), `myfork/main` (`fd54148`) and
`myfork/freebuff/tandem-cpu-recovery` (`437891e`) are all ancestors (423/366/367 behind). Nothing
was left to merge: the session branch had no commit that `main` did not have.

The one piece of the interrupted snapshot worth taking over was taken: its CPUID/feature detection
idea (`kindi/kcpu_cpuid.c`, which is where the AVX-512/AMX reporting came from) now answers
`kindi caps` on Windows through `kernel32.IsProcessorFeaturePresent` (39/40/41) with no compiler
needed, merged only where the engine probe has no answer. Measured on the i7-8086K:
`AVX=yes, AVX2=yes, AVX-512=no` instead of three `unknown`s. Its AVX-512 kernels
(`fwht_block_avx512`, `rows16_avx512`) stay archived: no host here has AVX-512, so they cannot be
compiled or verified, and shipping an unverifiable kernel would be a claim rather than a feature
(recorded in [docs/AUTOTUNE.md](docs/AUTOTUNE.md)). `clean/kindi-0.5` remains the 0.5 baseline.
Nothing was deleted, rebased or force-moved; both `wip/` branches stay as the audit-sized archive.

### Current leases

The full dated lease table (one row per task since 2026-11-08, with its evidence) is in
[docs/development/AGENTS-HISTORY.md](docs/development/AGENTS-HISTORY.md) under "Leases moved out of the
live table (2026-11-17)". **Keep this table to open work.** A new task adds one row here with its
worktree, branch, shared files and state; when it finishes, update the state and move the row to that
file. This file is capped at 450 lines by `tests/test_agents_handoff.py`, so a finished row left inline
is what makes it trip.

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| _(none open)_ | — | — | — | — |

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
| `python tools/check_python_quality.py` | python, `ruff==0.16.0`, `mypy==2.4.0`, `pyright==1.1.414`, `pylint==4.1.3` | no new or increased Ruff, mypy, Pyright or Pylint findings against the measured baseline |
| `python -m pytest -q tests/test_installer_grub.py` | python, `sed` | the installer's GRUB sed keeps existing kernel parameters |
| `python -m kindi.support_matrix --check` | python | the generated support matrix matches the code |
| `python -m kindi.config_reference --check` | python | the generated configuration reference matches the code |
| `python tools/project_memory.py --check` | python | the requirement history matches git |
| `python tools/check_perf_baseline.py` | python | the committed CROSS-MODE rows are within 15% of the perf baseline |
| `python tools/check_claims.py` | python, git | every number the six claimed docs quote is re-derived from its tracked run, and every benchmark citation has a claim |
| `python tools/check_warp_masks.py` | python | no new unmasked warp shuffle/barrier in `kindi/*.cu` (7 reviewed sub-warp masks in `tools/warp_mask_baseline.json`) |
| `python tools/check_secret_history.py` | python, git | no new credential or personal-data row in git history (tracked-file scans cannot see a deleted line) |
| `python tools/check_secret_baseline.py` | python, `detect-secrets==1.5.0` | `.secrets.baseline` still matches the tracked tree |
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
an older decision; `docs/development/` is the durable record and this file is the live handoff. The
finished lease rows were moved there again on 2026-11-17 (its last section), leaving the live table with
no open lease and this file well under the 450-line guard.
