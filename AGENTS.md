# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

## Current state (live — read this first)

- **HEAD:** `0dbff91` on `freebuff/tandem-cpu-recovery` (`main` receives the parent-of-this-commit bump and a FF at a safe handoff, not while another lease is active; guarded by `tests/test_agents_handoff.py`). The 2026-11-18 audit follow-up moved local machine roots out of tracked docs/evidence and benchmark logs, closed the DSV4.1 body-error keep-alive path, added a Bridge smoke suite plus CI job, and pinned every workflow's actions. This is focused-test evidence, not a full-suite rerun. Stock moved to upstream `v0.6.1` (`7fececb`, fetched 2026-11-17) and one of its fixes was a bug this fork had: `kindi/mtp.py` never passed `k_rmsfold_qT`'s `hp`, so the MTP head's CUDA graph did not build and mode A speculation was dead on the tensor-core layout (fixed in `f934a55`, with `tests/test_kernel_arity.py`); measured after the fix, mode A costs 0.66x on the 2B at the default `KINDI_SPEC_K=4`, but at `KINDI_SPEC_K=1` it is 1.12-1.13x at 512 and 24K tokens of context and a wash at 8K (4-run medians, 2026-11-17, `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-mtp-context/RESULTS.md`), so it is off at the default k and a small win at k=1 (`docs/MTP.md`), and `CURRENT-VS-KINDI-0.6.md` records the current official v0.6.1 comparison (576-commit / 704-file delta at `0184bb8`; round history in `detailed-fork-changelog.md`); §4.6 closes the same-settings A/B with stock `v0.6.1` (tag source, loader shim) and this tree at `a79137f` on the same 2B/9B packs at ctx 8192, plain engine and exact head, five passes each: every decode median is within 2.6% and the ranges overlap, while the 9B prompt-reading gap (83.9 vs 72.4) was rechecked at 20 passes and closed as noise (70.7 vs 75.7; paired -1.4, p=0.648); a separate stock harness adds literal position-512 medians (190.3 on the 2B, 84.5 on the 9B) outside the bench sweep, after a 2026-11-18 paired run showed its position-100 anchor matching the immediately preceding bench (+1.2%/0.0%) and replaced the earlier cold-clock-heavy 171.2/80.7 values, and the four sweep cells now carry three-pass medians with ranges (the pages note that the 9B prompt-reading ranges overlap, so no reading shape is claimed). §4 is re-based at `0184bb8` into separate worse / missing / unverified tables with a closure condition per row; it distinguishes the 16-file `v0.6.0..v0.6.1` range from the 15-file 0.6.1 commit (the compose surface is present from `dce89df`) and records the MoE backend's status (exercised on GPU through Lightning). `kindi bench` now reports the prefill/decode share of one whole request (`--split-at`, default 4096,8192): on the 2B with a 128-token reply it is 13.6% prefill at 512 prompt tokens, 64.1% at 4K and 82.2% at 8K, and the chunked prefill runs at 4,349/3,215 tok/s against the 302 tok/s single-token read line, so the old pair of rates was not a request decomposition (`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-bench-split/RESULTS.md`). The general lessons from that bug are on the roadmap, not started: `docs/ROADMAP.md` "Engine-liveness follow-ups (2026-11-17)" ranks eleven follow-ups and takes four with their gates first — a launch-shape contract for every kernel call site (REQ-0023 — A1+A3 done 2026-11-17: 105 launches measured across `spec.py`/`dspine.py`/`mtp.py`/`tp.py`, 37 unresolvable sites pinned with reasons, every measured site proven by mutation; the dspine `kf`/`self.k`/`self.k8` tables are resolved from their literals and only two dspine sites remain pinned; the grid/block half stays open), report the effective speculation state plus a `mtp_heads.json` digest check (REQ-0024 - A1+A2+A3+A4 done 2026-11-17: one mode resolver behind `/health`, the startup log and `kindi info`, the head's sha256 and size checked before `np.load`, and `/metrics`' `kindi_spec_mode_effective` naming the effective mode), `kindi verify --spec` as an exactness gate (REQ-0025 - A1/A2/A4 done and A3 four of five, 2026-11-17: the mode is printed first, a missing drafter exits 2, and the 2B is token-exact 4/4 in plain, `ngram`, `mtp`, `draft2b` and `both`; `dspine` waits on weights, `docs/MANUAL-HELP-NEEDED.md` row 7) and `kindi bench --spec` (REQ-0026; REQ-0025 A6 now also runs the four modes as `tc-<spec>` cells in `tools/verify_matrix.py`, scoring requested/effective and drafts and skipping absent files so a baseline gates them); the seven parked ones name the measurement that would reopen them. `kindi io` now measures this host's memory tiers (RAM copy, unbuffered disk read, pinned PCIe both ways) so a cache or streaming budget is an input, not an assumption; the CPU-compute lane and mixed-precision tiers are parked in `docs/ROADMAP.md` "Memory-tier follow-ups (2026-11-17)"; I/O-aware slot sizing now consumes those rates in `kindi plan`/`kindi autotune`; the plan's follow-up bench carries the planned context and count with the global `--ctx` and portable `--dcache` flags and a test pins both, `kindi autotune` adds an `expert-cache-<MiB>` cold-tier candidate that sets `KINDI_LT_COLD=1` + `KINDI_LT_HOST_MB` from the same rate-sized working set and caps it by the host experts, and `kindi io --json FILE` reports can be reused by either command through `--io-report`/`KINDI_IO_REPORT` (a missing or unusable report names itself and falls back to one probe), with the arithmetic unit-tested and its Lightning-pack comparison still open; the NVMe cold-expert tier is **started 2026-11-17** behind `KINDI_LT_COLD=1` (`kindi/cold_experts.py`: the hottest host experts stay pinned under `KINDI_LT_HOST_MB` (default available RAM less 10%/1 GiB), only the overflow is file-backed, unbuffered aligned reads, coalesced gaps, a bounded lookahead prefetch queue (a daemon thread fills it while the current layer's kernels run; the next cold layer joins before its demand read), Q2_0 de-interleave, requested/transferred and exact pinned/disk byte accounting; prompts stage the host block with each pinned expert at its host-tier slot, `cold_stats()` counts each pinned expert once while its byte totals stay per matrix, decode keeps a per-layer pinned slot cache (`cold_slot_plan`: a routed expert's slot is never evicted and a repeated expert takes no read; `cold_stats()` reports `slot_hits`), the decode graph is disabled on this opt-in path, and the per-layer stream sync plus blocking current-layer read stay open (C9; no tok/s claimed), unit-tested only with **no larger-than-RAM pack, verify, GPU run or tok/s claimed**, and the Strata/KTransformers/BigMoeOnEdge triage is in `docs/EXTERNAL-ENGINES.md`. The one commit message that named the knowledge-base folder in full was rewritten on 2026-11-17 to `$KINDI_AI_INVENTORY`; that commit and the 32 after it were re-hashed with identical trees and dates, both worktree branches were moved together, and every short hash cited below, in `CURRENT-VS-KINDI-0.6.md`, in the requirements history and in `docs/claims/` now names the rewritten commit. Last full suite recorded (before the 2026-11-18 audit round; no full rerun since): **1204 passed, 28 skipped, 0 failed**, including two new kernel-arity resolver checks (only tables rooted at `self` are followed; every branch of a conditional kernel must accept the call's argument count) and the two CLI comma-value checks (unknown `--split-at`/`--tandem-cpu-layers` tokens are rejected at parse time with the valid forms; the full suite re-ran after the generated reference was refreshed), and the newest twenty-six being seven saved-I/O-report checks (save/reload, missing and unusable fallback, and the plan/autotune wiring) plus the Lightning plan's command-reproduction test, four autotune host-cache candidate checks (env pair, rate size, host cap, default run budget and missing inputs) plus the cold-expert flag, host/disk split and pinned/disk byte accounting, exact-byte/coalescing/prefetch accounting, Q2_0 split, ordering, warm-slot skip and short-read checks in `tests/test_plan.py`/`tests/test_autotune.py`/`tests/test_cold_experts.py`, before them the six rate-normalization, rate-slot prediction/cap, plan-rate report and autotune candidate checks in `tests/test_io_probe.py`/`test_lt_fit_equivalence.py`/`test_plan.py`/`test_autotune.py`, then the `kindi io` invalid-`--skip` rejection, then the three effective-mode metrics states in `tests/test_server.py` (loaded, refused and drafter fallback), before them the eight speculation cells in `tests/test_verify_matrix.py` (the `tc-<spec>` env, skip reasons, requested/effective/draft parsing, refusal ordering and dry run), then `kindi io`'s RAM/disk/PCIe probes (PCIe reports unavailable when CuPy/CUDA is absent; the disk probe labels a buffered fallback instead of passing it off as the device's), before them the seventeen MTP head registry/effective-speculation and `verify --spec` checks, then the eight MTP head and effective-state tests, and before those the kernel-arity checks, and the previous seven covering `tools/check_secret_history.py` (a key deleted from the tree is still found in a diff, the count gate flags a new row, the baseline carries no matched text, the last 20 commits add no history hit, and a message-only leak is caught in both a subject and a body with no file carrying the path); the secret guards are now three: entropy baseline over tracked files, pattern scan over tracked+untracked files, and a history gate against `tools/secret_history_baseline.json` (128 rows / 3,247 legacy hits from an old benchmark dump, old AGENTS/log revisions and one commit message that names a cloud storage folder, recorded not endorsed; the gate scans commit messages - subject and body - with the same rules as file content, so a path in a message is caught instead of invisible); thirteen tests pinning `kindi plan`'s decode-slot arithmetic (the fit subtracts the slots before dividing over `n_moe - 2` layers, so 10 slots cost 20 experts per layer on the 256-expert test pack; the line used to claim one per slot, `kindi/ltfit.py` is now the one implementation both `plan._lt_fit` and `lightning._plan` call, and the weighted branch prices the slots against the same room with `dcache=0` - 9 of 128 card experts and 5.8 points of served share on the test pack's tight card - in the line and in `slots_cost`); `tools/ollama_links.py` links the GGUF models Ollama already downloaded under readable names (`$OLLAMA_MODELS`, else `~/.ollama/models`, `-m` to point elsewhere; the manifest's largest model layer, with a blob scan for the `GGUF` magic when there are no manifests), using a symlink when the host allows it and a same-volume hard link otherwise into `~/.kindi/models/ollamaLinks` (`-o` to move it) - `tools/ollama_links.ps1` is now a thin wrapper over the same script and twelve tests cover the naming, the fallbacks, the environment variable, dry run, idempotence and the failure hints; the coverage gate `python tools/check_coverage.py` passes on 46 modules from a measured run (`coverage run --branch --source=kindi -m pytest` -> `coverage json`, since `pytest-cov` hangs on this host; 2026-11-18 full suite 1,196 passed / 28 skipped, 45.4% total), and `tests/test_no_secrets.py` now scans untracked files too (pinned by `test_untracked_files_are_scanned`), with `project_memory --check`, `config_reference --check` and `support_matrix --check` all clean; GPU grammar/JSON-schema gate closed on the 2B (§39); recursive GBNF is now opt-in (`KINDI_GBNF_MAX_DEPTH`, bounded unroll cross-checked against an independent matcher - see the newest rows in `docs/development/AGENTS-HISTORY.md`) and measured as a diagnostic only: on the 2B's 248,320-entry vocabulary the unrolled llama.cpp `json.gbnf` (1,737 states at depth 1) costs 2,720 ms per token against 231 ms for `json_schema` (0.42 vs 4.9 tok/s end to end), and 9,483 states cost the same within 3%, because the filter sweeps the vocabulary once per new state - and at a 200-token document the average *falls* to 463 ms/token against 57 for the schema (8x) because repeated states are cache hits, while a whitespace-permitting constraint lets the greedy 2B spend the whole budget on whitespace (200 tokens, 9 characters) (`docs/API.md`, `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md`), and the two harnesses behind it are tracked as `tools/grammar_cost.py` and `tools/grammar_cost_serve.py` so a clean checkout re-derives the rows instead of trusting the JSON artifact. The mask itself is now a gated roadmap item, not started: [REQ-0027](docs/development/requirements/REQ-0027.md) takes the 231 and 2,720 ms/token baselines as the floor its gate must beat by 10x, with a token-class table or an on-card mask named only as direction and a CPU set-equality test against today's `vocab_allowed` as the semantics guard (`docs/ROADMAP.md` "Constrained-decoding cost"). Upstream `v0.6.0` (`179710c`) + `origin/main` head (`dce89df`, v0.6.1's parent) both contained, while `v0.6.1` itself is one commit past `dce89df`; vs `v0.6.0` the tree is a strict superset; comparison report in `CURRENT-VS-KINDI-0.6.md`. The numbers those docs quote are no longer prose: `python tools/check_claims.py` re-derives every one of them from the tracked artifact of its run (17 gates, 95 numbers across `docs/claims/*.json`, 4 citations listed as reasoned debt), and refuses a benchmark citation no claim accounts for - schema and procedure in `docs/claims/README.md`. Owner actions pending: Nanbeige 33B GGUF replacement, Gemma 4 E4B pack, DeepSeek V4.1 shards/gate data, PQ2_0/Q2_0 quant downloads, a pinned llama-server, other-hardware test rigs, push/installer/CI publications, T4.4/T4.5 + D3/D5 decisions — `docs/MANUAL-HELP-NEEDED.md`.
- **Latest:** The latest evidence is at `0184bb8`. A direct 27B A/B grounds the byte-identical
  claim: `tools/spec_ab.py` builds one 27B tensor-core engine with the 2B drafter and head,
  generates the four reference prompts plain and under `ngram`, `draft2b` and `both`, and compares
  token ids and decoded bytes directly. The run is **12/12 prompt/mode pairs identical** (40 greedy
  tokens each, no mismatch), recorded in `docs/MTP.md` and
  `docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-27b-spec-ab/`. The tracked 27B verify baseline now
  stores the three-pass medians (45.7/71.0/62.0 tok/s) instead of the faster first pass, and its
  `started_utc` (`2026-11-17T21:15:21Z`) and `git_revision` (`cf81a82`) name that median set,
  while each cell's `duration_s` and summary name the actual pass whose tok/s is the median
  (not a median duration from another pass), so a gate receipt copies the medians' revision
  instead of the first pass's `53003a9`; the 2026-11-18 median-baseline gate passed at 49.8/75.7/66.9 tok/s
  (+9.0/+6.6/+7.9%), with all four matrix cells passing.
  Checks: `tests/test_spec_ab.py` + `tests/test_verify_matrix.py` **43 passed**,
  `tests/test_verify_matrix.py` + docs/links/claims **58 passed**, docs/claims/memory **24 passed**,
  direct A/B exit 0, `check_claims` 16 gates / 97 numbers, `project_memory --check`, `compileall`,
  `git diff --check` green. The same three median rows are now in `docs/CARDS.md`'s RTX 4060 Ti
  verify-matrix table, linked to the baseline page; the last recorded full CPU gate is **1204 passed, 28 skipped** (not re-run in the audit round)
  and the docs/handoff/memory checks **12 passed**. The stock position-512 harness was rechecked on
  2026-11-18 with five paired bench-then-harness passes: its position-100 anchor was 191.3 vs 189.1
  on the 2B and 86.3 vs 86.3 on the 9B, so the earlier 171.2/80.7 standalone medians were cold-clock
  runs and are superseded by **190.3/84.5 tok/s** at position 512. §4.6 now also carries
  separate, explicitly non-same-settings tensor-core rows: 2B 484.9 read / 354.9 @100 / 333.8 @2000 /
  331.0 @8000; 9B 137.2 / 118.3 / 115.2 / 111.8, with the stock reference rows repeated and all
  ten raw logs force-added past the `*.log` rule. The five-pass 9B reading gap (83.9 vs 72.4) is
  now closed as noise: 20 alternating-order pairs give 70.7 stock vs 75.7 fork, a paired median of
  -1.4 tok/s, and an exact sign test of p=0.648; the 40 raw clock/power logs are tracked in the
  reading recheck page. The full gate then found ten TC JSONs carrying a double-escaped local pack
  path; they are sanitized at `0184bb8`, with the historical occurrence recorded in the reviewed
  secret baseline. The prior task is below.
- **Previous:** The 2B gate's fail-closed proof is now a test, not prose, at `ae9c861`:
  `tests/test_verify_matrix.py::test_2b_gate_fails_closed_when_the_optional_head_and_pack_are_absent`
  runs the documented gate with `--mtp-head`/`--spec-pack` absent and `ngram` deselected; the three
  optional cells are skipped, all four baseline spec cells (including `tc-ngram`) are reported
  missing, and the gate exits 1. `tests/test_verify_matrix.py` **39 passed**; docs/claims/memory
  **24 passed**; `compileall`, `git diff --check` green. No GPU run. The prior task is below.
- **Earlier:** A server started with `KINDI_SPEC` set but without `KINDI_TC=1` now logs
  `warning: speculation off: KINDI_SPEC=... requires the tensor-core plan engine...` at startup,
  built by the same helper as `/health`, so the gap is visible before anything queries health. At
  `6ea3524`: `tests/test_server.py` **86 passed**, including the warning test over `mtp` and
  `draft2b`; `config_reference` regenerated (198 environment reads) and checked; docs/claims/memory
  **24 passed**; `compileall` and `git diff --check` green. No GPU run. The prior task is below.
- **Older:** The open audit findings are closed at `2b41b31` where the code/tree owns them, and
  recorded as open hardware risks where it does not. `kindi serve` now has one family table: Nanbeige,
  Lightning and Gemma 4 are served, dsv41 directs to `python -m kindi.dsv41_server`, Cascade keeps its
  converter-only message, and an external `KINDI_BACKEND_MODULES` family is refused instead of falling
  into the dense builder; tests pin a registry adapter for every served family. `Usage` validates
  count/order shape and `split_host_tiers` reports when it falls back to the usage order; autotune's
  `expert-cache-<MiB>` candidate now sizes through `rate_slots(room=..., n_moe=...)` and the host-tier
  cap; `card_features()` is silicon-only again (FA2 dropped). CI runs `check_claims`, `project_memory
  --check`, `config_reference --check` and `support_matrix --check`; coverage floors grew from 26 to
  the measured **46** modules (`engine.py` 20%; `spec/tp/dspine/mtp` only at CPU-reachable levels) from
  a **1,196 passed / 28 skipped / 45.4%** full run whose only failure was the then-stale generated
  reference (regenerated; gate green). The installer docs lead with the checksum path and a reviewed
  Windows download. C9/R1 (cold-tier sync/blocking reads and pointer freshness) and R2-R4 stay open
  and are now stated in `docs/MOE-TANDEM.md`/`docs/BACKENDS.md`; no GPU or cold-tier run is claimed.
  Focused tests after the final edits: **253 passed**; no full-suite rerun after the reference
  regeneration. The follow-up at `e2879de` records a live 2B check: `drafting.method` `mtp` with
  `drafting.last` 3 passes / 12 drafted / 8 accepted, in `docs/MTP.md` and REQ-0024, so the
  attached-versus-drafted distinction is durable. That run's context, `KINDI_SPEC_K`, pack/head
  and command were not captured; four drafts per pass fits the engine default k=4 and rules out
  the smoke's k=1, and the session's request count is unknown (`drafting.last` covers one request).
  A 2026-11-18 repeat with `tools/mtp_smoke.py --ctx 4096 --spec-k 4` passed on the same 2B/head,
  printing `drafting.last.method=mtp` with 2 passes / 8 drafted / 3 accepted (metrics 8/3, five
  tokens): the tool now asserts the per-request method too, and the run reproduces the
  attached-and-drafting path, not the earlier counts. The prior task is below.
- **Older still:** `drafting.last` now names the effective speculation mode that produced its drafts. The
  engine's per-request `spec_mode` is copied into the last-request record when generation ends, and
  `/health` exposes it as `last.method` alongside `passes`/`drafted`/`accepted`/`acceptance`; the
  sibling `method` still describes what is effective now. At `d9f3fe3`: focused server tests
  **8 passed** (81 in `tests/test_server.py`), full suite **1185 passed, 28 skipped**; no GPU run
  for this field. `docs/AGENTS.md` and `CHANGELOG.md` document it. The prior task is below.
- **Oldest:** The 2B MTP server check is now `tools/mtp_smoke.py`. It boots a real `kindi serve`
  child with the registered head, a clean `KINDI_*` environment, an empty config, a temporary
  state directory and a free loopback port; it sends one greedy completion and fails unless
  `drafting.method` is `mtp`, `drafting.last.drafted` and `kindi_spec_drafted_tokens_total` are both
  above zero, and the accepted counters are consistent. At `f958a1f` on the RTX 4060 Ti (sm_89),
  pack `bonsai-2b`, head sha256 `1b5c0c790a59`, ctx 4096, k=1: PASS with `drafting.last` 3 passes /
  3 drafted / 2 accepted and the same 3/2 metrics, completion 5 tokens. It is not under `tests/`,
  so the CPU suite does not collect it; full suite **1184 passed, 28 skipped**. The prior task is
  below.
- **Older still:** The cold-expert reader now separates coalescing from a real lookahead prefetch
  queue: `prefetch()` fills a bounded exact-span cache, a later `read_many()` consumes those
  bytes with no second file read, and Lightning queues the next cold MoE layer's most-used
  experts after preparing the current one (`tests/test_cold_experts.py`). `kindi bench --dcache N`
  applies the Lightning plan's slot count to one run (and
  restores the caller's `KINDI_LT_DCACHE`), while `kindi plan` now emits that portable flag instead
  of a POSIX-only `KINDI_LT_DCACHE=` prefix (`tests/test_bench_json.py`, `tests/test_plan.py`). The
  Lightning plan's follow-up now also carries its planned context as the global `--ctx N` before
  `bench`, so the fit and slot count are reproduced at the same context (`tests/test_plan.py`).
  With `KINDI_LT_COLD` set, the plan's `host ram` line and `host_tier` report show pinned versus
  file-backed host experts under the same budget as the engine (`tests/test_plan.py`). The
  kernel-arity resolver checks every branch of a conditional argument tuple, so a
  6-arg branch cannot pass a 7-parameter kernel on the 7-arg branch (`tests/test_kernel_arity.py`, 8
  tests; coverage unchanged at 105 measured / 37 pinned). The full-suite rerun also found local
  machine paths in the three tracked 27B median JSONs; they are placeholder-only now and the
  reviewed history hits are in `tools/secret_history_baseline.json`. The standalone DSV4.1 CLIs now
  reject malformed values at argparse time: `--ctx` is a positive-integer list, `--auto` and
  `--cal` are finite `a,b` pairs, server `--ctx` is a positive integer, and `--range` is two
  non-negative integers `a-b` with `a <= b`, instead of a bare split after the model loads
  (`tests/test_dsv41_cli.py`), and `python -m kindi.dsv41_server` caps request bodies at
  `KINDI_MAX_BODY_MB` (16 MiB default) with a 413 (`tests/test_dsv41_server.py`). The autotune
  expert-cache proposal is capped by the host's currently
  available RAM after the same 10% / 1 GiB headroom as the default `KINDI_LT_HOST_MB` budget
  (`tests/test_autotune.py`). The Qwen3.6 roadmap row now records both community ternary builds
  as external Apple-Silicon quantizations, not llama.cpp/NVIDIA artifacts or Q0 shortcuts, and
  keeps the pinned-GGUF Q0 gate in front of any asymmetric-precision experiment
  (`docs/ROADMAP.md`).
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
| `T:/kindi` | `main` | `0184bb8` | main checkout; clean; behind this session while another lease is active |
| `<worktree>` | `freebuff/tandem-cpu-recovery` | `0dbff91` | this session; not fast-forwarded while `main` is active |

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

### Consolidation audit (2026-11-17)

`main` and this worktree are the same commit and both clean, the local untracked comparison render is
now gitignored, and every remote ref is already contained: `origin/main` (`dce89df`), `myfork/main`
(`fd54148`) and `myfork/freebuff/tandem-cpu-recovery` (`437891e`) are ancestors of `main` (0 ahead,
358-415 behind). Two branches stay outside `main` on purpose, audited rather than merged because
merging either would import superseded or corrupt code:

- `wip/main-interrupted-snapshot-20261109-045352` — and the stash-format
  `wip/main-interrupted-backup-20261109-045352`, whose files are a subset of it — holds 40 paths that
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
- `clean/kindi-0.5` (1 commit, 418 behind) is the 0.5 comparison baseline, not a candidate.

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
| `python -m pytest -q tests/test_installer_grub.py` | python, `sed` | the installer's GRUB sed keeps existing kernel parameters |
| `python -m kindi.support_matrix --check` | python | the generated support matrix matches the code |
| `python -m kindi.config_reference --check` | python | the generated configuration reference matches the code |
| `python tools/project_memory.py --check` | python | the requirement history matches git |
| `python tools/check_perf_baseline.py` | python | the committed CROSS-MODE rows are within 15% of the perf baseline |
| `python tools/check_claims.py` | python, git | every number the three claimed docs quote is re-derived from its tracked run, and every benchmark citation has a claim or a reasoned debt entry |
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
