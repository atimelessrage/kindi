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
| `T:/kindi` | `main` | `46b63f8` | main checkout; clean apart from the untracked `optimizing-mixed-gpu-rig.json` |
| `<worktree>` | `freebuff/tandem-cpu-recovery` | `50d1d56` + uncommitted audit fixes | this session; includes main's Turing work |

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

### Current leases (checked 2026-11-11)

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| Main-worktree task (other agent) | `T:/kindi` | `main` | `README.md`, `docs/CARDS.md`, `docs/CONFIGURATION.md`, `docs/INSTALL.md`, `docs/TURING-TENSOR-CORES.md`, `docs/VERIFICATION.md`, `docs/CPU_GPU_TANDEM_DESIGN.md`, `install.sh`, `kindi/spec.py`, `.github/workflows/ci.yml` | clean; no active shared-file edit; only untracked `optimizing-mixed-gpu-rig.json`; interrupted WIP preserved on `wip/main-interrupted-snapshot-20261109-045352` |
| Documentation drift (P1-3, P1-4, P1-5, P1-6) | this worktree | `freebuff/project-review-audit-report` | `WHY.md`, `docs/CONFIGURATION.md`, `docs/INSTALL.md`, `docs/BENCHMARKS.md`, `docs/CARDS.md`, `docs/MTP.md`, `kindi.conf.example`, `kindi/spec.py` | partial: `WHY.md` and `kindi.conf.example` updated; `docs/INSTALL.md` checksum updated by the release task; `docs/CONFIGURATION.md`, `docs/CARDS.md`, `docs/BENCHMARKS.md` and `docs/MTP.md` remain blocked by the main worktree's dirty copies |
| Continuous-batching investigation | this worktree | `freebuff/project-review-audit-report` | `docs/CONTINUOUS-BATCHING.md`, `AGENTS.md` | archived NO-GO; Phase-0 gate failed (0.69x); top banner and section 8 updated; no production code changed |
| Server request validation + boundary hardening (P1-1/P2-1) | this worktree | `freebuff/project-review-audit-report` | `kindi/server.py`, `tests/test_server.py`, `tests/test_config_example.py` | done; 10 server tests (validation, auth, 413 body cap) + 1 example-config test + 9 installer fixture tests = 20 passed; invalid requests and 413 keep the cache |
| GRUB edit guard (plan 1) | this worktree | `freebuff/project-review-audit-report` | `install.sh`, `tests/test_installer_grub.py` | done; guard committed in `2b84658`; 9 fixture tests pass and the non-vacuous check is recorded. Release commit updates `docs/INSTALL.md` to the fixed artifact hash `c05d0190...`; the merged `install.sh` now carries both the Turing profile and the GRUB guard; the pinned hashes were refreshed after the merge |
| Installer safety release (GRUB guard) | this worktree | `freebuff/project-review-audit-report` | `docs/INSTALL.md` (checksum only), `docs/RELEASE-CHECKLIST.md`, `AGENTS.md` | release docs committed; guard already in `2b84658`; no tag, push or upload. Verified: `bash -n` OK, 9 guard tests, 58 full tests, `compileall` OK, artifact hash `c05d0190...` check OK, no control bytes. Release artifact is the CRLF worktree file; the LF Git blob is `2ae8f960...`; docs must name the exact published file |
| Per-slot plain-engine plan (Phase 1) | this worktree | `freebuff/project-review-audit-report` | `docs/PER-SLOT-REFACTOR.md`, `docs/CONTINUOUS-BATCHING.md` | archived NO-GO with `docs/CONTINUOUS-BATCHING.md`; no engine code changed |
| 2B CB-vs-spec measurement | this worktree | `freebuff/project-review-audit-report` | `bench-logs/compare-2b-cb-spec/**`, `AGENTS.md` | measured; no code changed; results in `bench-logs/compare-2b-cb-spec/RESULTS.md` |
| Phase-0 CB spike | this worktree | `freebuff/project-review-audit-report` | `bench-logs/spike_batch_gemm.py`, `bench-logs/phase0-spike/**`, `docs/CONTINUOUS-BATCHING.md` | implemented; B=4 gate FAIL (0.69x vs single decode); no production code changed |
| Plan-engine exactness guard | this worktree | `freebuff/project-review-audit-report` | `kindi/spec.py`, `AGENTS.md`, `bench-logs/diag_exact_plan.py`, `bench-logs/diag_verify8.py`, `bench-logs/diag_spec_long.py` | 2B sp=1 (no k_tc_sum) + 9B HB guard; long spec 160/160 exact; merged with main's Turing changes; the plan-engine tests pass |
| Long-term card support (single/dual) | this worktree | `freebuff/project-review-audit-report` | `docs/CARD-SUPPORT.md`, `AGENTS.md` | roadmap and acceptance contract written; no engine/installer code changed; the 4060 Ti 8 GB classification and all non-Maxwell pairings are blocked by the main worktree's dirty `install.sh` |
| Long-term CPU support (i5-2500K, W-2133, i9-13900K, i7-8086K) | this worktree | `freebuff/project-review-audit-report` | `docs/CPU-SUPPORT.md`, `AGENTS.md` | roadmap and acceptance contract written; no CPU/installer code changed; `kindi/cpu.py`, `kindi/kcpu.c` and `install.sh` are main-owned/dirty; the i5-2500K needs an AVX-only/scalar fallback |
| CI + plan-engine regression test | this worktree | `freebuff/project-review-audit-report` | `.github/workflows/ci.yml`, `tests/test_plan_engine.py`, `AGENTS.md` | done; CI installs `tests/requirements-test.txt` (FastAPI/httpx were missing, so collection failed) and ShellChecks the committed LF blob; 20 new CPU-only plan tests; 78 full tests pass. The main worktree's own `.github/workflows/ci.yml` is still dirty: reconcile by hand, do not overwrite it |
| Generated configuration reference | this worktree | `freebuff/project-review-audit-report` | `kindi/config_reference.py`, `tests/test_config_reference.py`, `docs/CONFIGURATION-REFERENCE.md`, `kindi.conf.example`, `AGENTS.md` | done; `python -m kindi.config_reference` renders 25 config keys and 104 environment rows from the code, and `--check` plus 6 tests fail on drift. `docs/CONFIGURATION.md` stays main-owned/dirty; the generated file is new, so merging it is additive |
| 2B spec acceptance on non-repetitive prompts | this worktree | `freebuff/project-review-audit-report` | `bench-logs/measure_2b_spec_norepeat.py`, `bench-logs/measure-2b-spec-norepeat/**`, `AGENTS.md` | measured; no code changed; n-gram acceptance is 8-18% on fresh prose vs. 86% on the copy control, throughput 364.9 vs. 328.9 tok/s (plan) and 168.1 (plain); results in `bench-logs/measure-2b-spec-norepeat/RESULTS.md` |
| Agent serving layer (task_0013 slice) | this worktree | `freebuff/project-review-audit-report` | `kindi/server.py`, `tests/test_server.py`, `docs/AGENTS.md`, `docs/CONFIGURATION-REFERENCE.md`, `AGENTS.md` | done; bounded queue with 429 + Retry-After, `/metrics`, and Ollama `/api/tags`, `/api/chat`, `/api/generate`; 18 server tests and 104 full tests pass after the Turing merge and the native-Windows driver loader; no engine code changed; multi-GPU replicas remain (needs `tp.py`, main-owned) |
| `install.sh` / `.github` collision | both | both | `install.sh`, `.github/**` | reconciled on 2026-11-09: the merged `install.sh` has the GRUB guard and the Turing profile; the WIP's empty `install.sh` was discarded, and main's Turing commits were merged |
| Tandem/CPU recovery audit | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CPU_GPU_TANDEM_DESIGN.md`, `docs/TANDEM-STATUS.md`, `AGENTS.md`, `kindi/cpu.py`, `tests/test_cpu_threads.py`, `docs/CPU-SUPPORT.md`, `docs/CARDS.md`, `docs/BENCHMARKS.md` | design restored; no tandem runtime merged; WIP `tandem.py` is unparsable, CPU changes are uncompiled and the WIP branches are preserved; see `docs/TANDEM-STATUS.md` for the cleanup gate. CPU thread default added 2026-11-09 (see Handoff) |
| K8/V4 chunk + tensor-core plan | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/chunk.py`, `kindi/chunk.cu`, `kindi/spec.py`, `kindi/tcf.cu`, `kindi/tp.py`, `tests/test_plan_engine.py`, `bench-logs/check_k8v4_chunk.py`, `bench-logs/27b-matrix/REPORT.md`, `AGENTS.md` | done; V4 store/one-pass/exact chunk kernels and V4 decode/verify plan kernels selected from `tp.kv`; `KINDI_TC=1` no longer refused for k8v4; ATT_TC/FD fall back with a log; 140 CPU tests, both modules compile on the 4060 Ti, direct GPU check passes (store vs NumPy, chunk V4 == decode V4 bit for bit); 27B 8K/24K retrieval matches q8_0 (5/5 codes + follow-up); the 49K read was aborted at >3x slower, see `bench-logs/27b-longctx/RESULTS.md` |
| K8/V4 two-card + cache verification and quarantine | this worktree | `freebuff/tandem-cpu-recovery` | `AGENTS.md`, `kindi/tp.py`, `tests/test_k8v4.py`, `tests/test_plan_engine.py`, `docs/CONFIGURATION-REFERENCE.md`, `bench-logs/k8v4-cache/**`, `bench-logs/test_k8v4_cache.py`, `bench-logs/test_k8v4_dist_cache.py`, `manual_help_needed.txt` | one-card cache save/restart/restore PASS (867-token opening reloaded after a restart, reply bit-exact, 867/888 cached); two-card run blocked (one GPU, `TPEngine` needs two); 2B k8v4 verify 2/4 vs the fp16 reference, 4B 4/4; 9B k8v4 unavailable (plan engine refuses HB 4096, plain engine ignores `KINDI_KV`) |
| PQ2_0 / Q2_0 conversion | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/convert.py`, `kindi/__main__.py`, `tests/test_convert.py`, `docs/TROUBLESHOOTING.md`, `docs/BONSAI-2-SUPPORT.md`, `bench-logs/convert_pq2_q2.py`, `bench-logs/pq2-q2-2b/**`, `AGENTS.md` | done; GGUF types 142/42 import to packs byte-identical to the PTQ1_0 pack for the 2B; `kindi verify` 4/4 for both new packs; 24,576-token bench (3 runs each) medians PTQ1_0 170.8 / PQ2_0 162.1 / Q2_0 160.8 tok/s at position 24,000, read 207.7 / 214.7 / 160.1; deltas are noise because the packs are identical. 4B/9B/27B PQ2_0/Q2_0 files not downloaded, so not verified |
| Native Windows CPU build | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/cpu.py`, `kindi/kcpu.c`, `tests/test_cpu_build.py`, `tests/test_cpu_threads.py`, `docs/CPU-SUPPORT.md`, `AGENTS.md` | done; clang 18.1.8 + MSVC 14.51 build `libkcpu.dll` without a vcvars shell; 2B/4B/9B verify 4/4 token-exact at 33.7/16.7/9.1 tok/s with 6 auto-detected threads |
| Generated support matrix + roadmap | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/support_matrix.py`, `tests/test_support_matrix.py`, `docs/SUPPORT-MATRIX.md`, `docs/ROADMAP.md`, `docs/CONTINUOUS-BATCHING.md`, `docs/PER-SLOT-REFACTOR.md`, `docs/TANDEM-STATUS.md`, `AGENTS.md` | done; `python -m kindi.support_matrix` renders models/formats/K-V modes/CPU ISA from the code and `--check` fails on drift; `docs/ROADMAP.md` is the single index of the archived no-go work (batching, per-slot, tandem, k8v4) |
| Sampling profiles + presence penalty | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/config.py`, `kindi/profiles.py`, `kindi/sampling.py`, `kindi/server.py`, `kindi.conf.example`, `tests/test_config.py`, `tests/test_profiles.py`, `tests/test_sampling.py`, `tests/test_server.py`, `docs/AGENTS.md`, `docs/CONFIGURATION.md`, `docs/BONSAI-2-SUPPORT.md`, `docs/CONFIGURATION-REFERENCE.md`, `AGENTS.md` | done; publisher thinking/non-thinking profiles behind `sampling_profile = publisher`, `presence_penalty` in the API/config; 189 tests, both generated-file checks and `compileall` pass; no GPU run. `docs/CONFIGURATION.md` was edited with `T:/kindi` clean (checked) |
| Installer card classification + dual pairing | this worktree | `freebuff/tandem-cpu-recovery` | `install.sh`, `tests/test_installer_cards.py`, `docs/CARD-SUPPORT.md`, `docs/CARDS.md`, `AGENTS.md` | done; every sm_80+ card is `tc` regardless of memory (the 8 GB RTX 4060 Ti was `pascal8`), same-class pairs with at least 3,800 MiB pair, `tc2` for two sm_80 cards; 12 fake-nvidia-smi tests pass and 7 fail on the old installer; 201 tests, `bash -n`, generated checks and `compileall` pass; no GPU run. `install.sh` was edited with `T:/kindi` clean (checked) |
| Cross-mode benchmarks (CPU / GPU / tandem) | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/build_cross_mode.py`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `tests/test_cross_mode_benchmarks.py`, `docs/BENCHMARKS.md`, `docs/TANDEM-STATUS.md`, `AGENTS.md` | done; CPU-only and GPU-only 2B/4B/9B at positions 100-24,000 from the measured i7-8086K runs; tandem rows are `n/a` because there is no runtime; generated `--check`, 203 tests and the usual gates pass; no new timing run |
| i7-8086K CPU support row + server chat | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CARDS.md`, `docs/CPU-SUPPORT.md`, `docs/ROADMAP.md`, `docs/SUPPORT-MATRIX.md`, `docs/BENCHMARKS.md`, `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md`, `docs/benchmarks/2026-11-09-i7-8086k-cpu/README.md`, `kindi/support_matrix.py`, `AGENTS.md` | done; i7-8086K row added to `docs/CARDS.md`; native 2B CPU server answered `/v1/chat/completions` non-stream and stream at 32.9 tok/s, 6 threads, 0 errors; support matrix regenerated; 203 tests and generated checks pass |
| CPU server chat 4B/9B | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CPU-SUPPORT.md`, `docs/CARDS.md`, `docs/BENCHMARKS.md`, `docs/ROADMAP.md`, `docs/SUPPORT-MATRIX.md`, `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md`, `docs/benchmarks/2026-11-09-i7-8086k-cpu/README.md`, `kindi/support_matrix.py`, `AGENTS.md` | done; native CPU server chat 4B/9B non-stream + stream, 0 errors, 0 MiB; 2B/4B/9B claim documented; 214 tests and generated checks pass |
| CI CPU build-and-smoke job | this worktree | `freebuff/tandem-cpu-recovery` | `.github/workflows/ci.yml`, `tests/test_cpu_smoke.py`, `docs/CPU-SUPPORT.md`, `AGENTS.md` | done; `cpu-engine-smoke` compiles `kcpu.c` with gcc for AVX2/FMA/F16C and loads it through `kindi.cpu._lib()`; 203 tests pass and the gated smoke test passes with a compiled library; no GPU or pack |
| RTX 4060 Ti tc verify | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `AGENTS.md` | done; 2B tc 4/4 at 370.2 tok/s, 4B tc 2/4 (prompts 2/3), 9B tc refused (Hadamard 4096), plain 4B/9B 4/4 at 114.9/72.5 tok/s; no inference code changed; verify_matrix.py now covers 27B |
| 27B tc 24K + server/cache | this worktree | `freebuff/tandem-cpu-recovery` | `bench-logs/test_27b_tc_cache.py`, `bench-logs/27b-tc-24k/**`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `AGENTS.md` | done; 27B tc at ctx 24,576 read 40.5 and generated 37.6/37.1/36.6/34.7/33.9 tok/s at 100/2K/8K/16K/24K, 724 nodes/decode, 351 MiB free; three real servers saved/loaded an 867-token opening (cached 867/888), replies bit-exact, stream `[DONE]`; no inference code changed |
| Native Windows 24K CPU benchmark + cross-mode matrix | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/{README.md,runs.csv,raw/bench-24k-*.log}`, `docs/benchmarks/build_cross_mode.py`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `docs/BENCHMARKS.md`, `docs/CARDS.md`, `docs/CPU-SUPPORT.md`, `AGENTS.md` | done; native 2B/4B/9B 24K generation 8.2/3.1/2.6 tok/s, read 45.1/19.2/9.9; replaced the WSL2 CPU cells; GPU/tandem rows unchanged |
| CPU thread-count sweep (6/8/10/12) | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/cpu.py`, `tests/test_cpu_threads.py`, `docs/CPU-SUPPORT.md`, `docs/CARDS.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/benchmarks/2026-11-09-i7-8086k-cpu/README.md`, `AGENTS.md` | done; WSL2 2B 6/8/10 read 38.2/44.3/25.5 and gen 28.6/31.2/24.1; native 2B 6/8/10 read 43.2/28.4/13.5 and gen 31.7/21.6/9.8; 8 is a WSL2/libgomp tuning, 10/12 are slower, default stays 6; no auto-tune |
| i7-8086K processor details | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CPU-SUPPORT.md`, `docs/CARDS.md`, `AGENTS.md` | done; added Coffee Lake 14 nm, 6C/12T, 4.00 GHz base, 5.00 GHz single-core turbo, typical 4.3-4.4 GHz all-core, 12 MB L3, unlocked multiplier, 2018 40th-anniversary positioning and the stock-clock caveat; no code changed |
| 9B CPU queue load test | this worktree | `freebuff/tandem-cpu-recovery` | `docs/AGENTS.md`, `AGENTS.md` | done; native 9B CPU, 6 threads, ctx 1024, queue max 8, 12 clients: 9x200 + 3x429 with Retry-After 1, max waiting 8, 0 errors; 429s returned in 8-10 ms; raw evidence untracked in `bench-logs/9b-queue-load/` |
| Intel Arc A770 support goal | this worktree | `freebuff/tandem-cpu-recovery` | `docs/INTEL-ARC.md`, `docs/ROADMAP.md`, `docs/CARD-SUPPORT.md`, `AGENTS.md` | planning/audit done; no Intel backend code; no Arc hardware (RTX 4060 Ti only, no `sycl-ls`); one/two/CPU-tandem goals, gates and milestones written; not a support claim |
| P106-100 single/dual support + CPU tandem gate | this worktree | `freebuff/tandem-cpu-recovery` | `docs/P106.md`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/ROADMAP.md`, `tests/test_installer_cards.py`, `AGENTS.md` | installer single/dual routing tested; no P106 hardware; general Bonsai CPU tandem blocked on the missing `TandemEngine` (Nanbeige host-layer path is separate); not a support claim |
| P106 verification runner | this worktree | `freebuff/tandem-cpu-recovery` | `tools/verify_p106.py`, `tests/test_verify_p106.py`, `docs/P106.md`, `docs/CARDS.md`, `AGENTS.md` | `tools/verify_p106.py` records the single-card 2B/4B/9B verify, the real server/cache restart, and the two-card 27B verify in one run; it pins the P106 indices, drops inherited `KINDI_*`, and refuses a full run without exactly two P106-100s. 10 CPU-only tests and a dry-run pass on this host; no P106 hardware, so the record is not run and not a support claim |
| Review/audit follow-up: version, config and link consistency | this worktree | `freebuff/tandem-cpu-recovery` | `README.md`, `WHY.md`, `SECURITY.md`, `docker/Dockerfile`, `docs/CARDS.md`, `docs/CONFIGURATION.md`, `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md`, `kindi/tokenizer.py`, `tests/test_version_consistency.py`, `tests/test_config_example.py`, `tests/test_tokenizer.py`, `tests/test_docs_links.py`, `AGENTS.md` | done; current-facing 0.4.0 tags/archive names, WHY/SECURITY version statements and the stale card tag fixed; `docs/CONFIGURATION.md` now names every default key; missing BPE pieces raise instead of silently encoding as token 0; local Markdown links are checked by a new test. Evidence: 14 focused tests pass, `config_reference --check` and `support_matrix --check` exit 0, `compileall` and `git diff --check` pass. The new `Buffy_ReviewAudit_202611111200.txt` is only an executive summary (1,012 bytes, no findings); the concrete older findings were re-triaged. The installer source tarball is still not hash-pinned; changing it requires a new published installer, so it is left for the release owner |

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

- 2026-11-11 audit follow-up: the new report at `T:/kindi/Buffy_ReviewAudit_202611111200.txt` is incomplete
  (only the executive summary; 1,012 bytes, no findings), so the two earlier reports were checked against the
  current tree instead. Fixed the current-facing 0.4.0 Docker tags and release archive names in `README.md`,
  `docs/CARDS.md` and `docker/Dockerfile`, the stale `WHY.md`/`SECURITY.md` version statements, the missing
  default keys in `docs/CONFIGURATION.md`, and the broken `../../CROSS-MODE.md` link. `kindi/tokenizer.py` now
  raises on a missing BPE piece instead of encoding it as token 0. New guards: image tags/archive names must
  match `kindi/__init__.py`, every `config.DEFAULTS` key must appear in `docs/CONFIGURATION.md`, and every local
  Markdown link must resolve. Evidence: the 14 focused tests (`test_version_consistency`, `test_config_example`,
  `test_tokenizer`, `test_docs_links`) pass; `python -m kindi.config_reference --check` and
  `python -m kindi.support_matrix --check` exit 0; `compileall -q kindi` and `git diff --check` pass. The
  published installer and the v0.5.0 archive both hash to `b84df4fe...`, matching `docs/INSTALL.md`; no
  installer change was made. The source tarball still has no pinned hash in `install.sh`; it is a release-level
  follow-up (the downloaded v0.5.0 archive hashes to `26424ec0...`).
- Native Windows runs without an external shim: `gpudrv.driver_name()` selects `nvcuda.dll` on
  Windows and `libcuda.so.1` on Linux, and `sched.py`, `chunk1.py` and `nanbeige.py` now use
  `gpudrv.driver()` instead of a hardcoded library. Verified natively through `python -m kindi`:
  9B 4/4 token-exact and 2B plan 4/4, with 104 CPU tests passing.
- 2026-11-09 reconciliation: main's three Turing commits were merged into this branch
  (`d21e566`); the interrupted main worktree was snapshotted to
  `wip/main-interrupted-snapshot-20261109-045352` and
  `bench-logs/main-backup-20261109-045352/`, then cleaned. The corrupt tandem WIP was not
  merged. Post-merge verification: 101 CPU tests, `compileall`, `bash -n install.sh`, generated
  reference check, 9B verify (4/4, 87.6 tok/s) and 2B plan verify (4/4, 377.1 tok/s) on the
  4060 Ti.
- Tandem/CPU recovery audit: `main` has only the host-RAM preflight, the baseline CPU engine and
  `layout.repack`; it has no tandem runtime. The WIP `kindi/tandem.py` is unparsable and the only
  historical blob is the corrupt one, so a branch merge would lose the audit/server work and still
  not provide a working engine. The intact design was restored at
  `docs/CPU_GPU_TANDEM_DESIGN.md`; the evidence and cleanup gate are in
  `docs/TANDEM-STATUS.md`. The WIP branches remain preserved.
- Branch `freebuff/project-review-audit-report`; audit follow-up commits `5061af2`, `2b84658`,
  `4b90ca9` and this roadmap commit.
- Verified: `python -m pytest tests/ -q` -> 84 passed (58 before the CI/plan-engine task, plus 20
  in `tests/test_plan_engine.py` and 6 in `tests/test_config_reference.py`); focused
  server/config/installer run -> 20 passed; `bash -n install.sh` -> OK; `python -m compileall -q
  kindi` -> OK; ShellCheck on the committed LF blob -> exit 0. No GPU run, so no plan was built or
  launched.
- CI is now runnable from this worktree (`.github/workflows/ci.yml`): the CPU job installs
  `tests/requirements-test.txt` and compiles every module; the shell job parses `install.sh` and
  ShellChecks `git show HEAD:install.sh`, which is LF even on a CRLF checkout. The main worktree
  has its own dirty copy of this file: reconcile by hand, do not overwrite it.
- The configuration reference is generated from the code (`python -m kindi.config_reference`):
  `docs/CONFIGURATION-REFERENCE.md` lists every `config.DEFAULTS` key and every `KINDI_*` read in
  `kindi/`, and `tests/test_config_reference.py` fails on drift. `docs/CONFIGURATION.md` is still
  main-owned and dirty, so its hand-written tables and the generated file must be reconciled by hand.
- The agent serving layer has a bounded queue (`KINDI_QUEUE_MAX`, default 8 waiting) that answers 429 with
  `Retry-After`, a Prometheus `/metrics` endpoint, and Ollama-compatible `/api/tags`, `/api/chat` and
  `/api/generate`. The API key protects all of them. `docs/AGENTS.md` describes the contract. No engine code
  changed; multi-GPU replicas are still open (they need the main-owned `kindi/tp.py`).
- Spec on non-repetitive prompts was measured on the 4060 Ti: acceptance falls from 86.1%
  (copy-task control) to 8.4% (k=3) / 17.9% (k=1) on three fresh prose prompts, and throughput is
  364.9 tok/s vs. 328.9 plan-only and 168.1 plain. `generate_spec` still equals `generate` on the
  plan engine; the plan engine itself diverges from plain on two of the three prompts. Evidence:
  `bench-logs/measure-2b-spec-norepeat/RESULTS.md`. No code changed.
- 2B/4B/9B benchmark at 24,576 tokens on the i7-8086K + RTX 4060 Ti is stored in
  `docs/benchmarks/2026-11-09-i7-8086k-rtx4060ti/` (README, `runs.csv`, `stats.csv`,
  `summary.json`, 18 raw logs). Current vs `v0.4.0` + native loader only: 24K generation
  160.6/79.8/62.3 vs 165.5/82.1/62.7 tok/s (-3.0/-2.8/-0.6%). `kindi/bench.py` and
  `kindi/engine.py` are byte-identical to the tag, so this is not a benchmark-path speedup or
  regression. 4B `kindi verify` 4/4. A separate 2B plan-vs-plain run at 24K is 302.4 vs
  170.6 tok/s (1.77x; T=4 exact plan, no drafting), stored as `2b-plan-vs-plain-24k.csv` and
  `.json`. No speculative/Turing/server/CPU measurement. Evidence:
  `docs/benchmarks/2026-11-09-i7-8086k-rtx4060ti/summary.json` and
  `.../2b-plan-vs-plain-24k.json`; all 24 runs exited 0.
- Release docs are prepared on this branch: `docs/INSTALL.md` pins the fixed release artifact
  `c05d0190...` (the LF Git blob is `2ae8f960...`; do not mix them), and
  `docs/RELEASE-CHECKLIST.md` records the release steps. The guard is in `2b84658`; the Turing
  commits were merged and the hashes refreshed on 2026-11-09. No tag, push or upload was made.
- Long-term single/dual card target is in `docs/CARD-SUPPORT.md`. The 2026-11-10 installer fix
  classifies every sm_80 card as `tc` (the 8 GB RTX 4060 Ti was `pascal8`), pairs same-class cards with
  at least 3,800 MiB each, and adds the `tc2` profile; `tests/test_installer_cards.py` covers it. The
  card target is still not a support claim until a GPU verify is recorded.
- Long-term CPU target is in `docs/CPU-SUPPORT.md`. The CPU engine requires AVX2, FMA and F16C but
  checks none of them; the i5-2500K (AVX only) therefore cannot build the current `kcpu.c`. The i7-8086K
  is verified in WSL2 (2B/4B/9B 4/4 token-exact; 6-thread 24K benchmark; 12 threads not viable); the
  Xeon W-2133 and i9-13900K remain unverified, and the native Windows CPU build is still unbuilt.
  Evidence: `docs/benchmarks/2026-11-09-i7-8086k-cpu/README.md`. `kindi/kcpu.c` and `install.sh`
  remain main-owned; the only CPU code changed here is the thread default below.
- CPU thread default (2026-11-09): `kindi/cpu.py` no longer lets OpenMP fall back to every logical CPU.
  `_configure_threads()` takes an explicit `threads=`, then `OMP_NUM_THREADS` already in the
  environment, then the physical-core count of the CPUs the process may run on (sysfs
  `thread_siblings_list`, `/proc/cpuinfo` fallback, logical count as the last resort), and sets
  `OMP_PROC_BIND=close` / `OMP_PLACES=cores` only in that default case. Cause of the 12-thread
  collapse, measured on the 2B at ctx 512: binding is irrelevant (12 threads read 1.3-1.4 tok/s with
  `close`, `spread` and no binding) while `OMP_WAIT_POLICY=passive` or `GOMP_SPINCOUNT=0` recover 12.5
  — libgomp spins both hyperthread siblings of a core at the per-row-block barriers of `kcpu.c`.
  Same probe: 6 threads `close`/`cores` 39.2 read / 28.9 generation (39.2-40.0 over four runs),
  8 threads 47.9 / 37.3 repeated at 46.4-49.1 / 36.0-36.1 (faster, not the default), 12 threads 1.3. Verified with no `OMP_NUM_THREADS`: engine reports 6 threads,
  39.8 read / 29.7 generation.  `tests/test_cpu_threads.py` pins the CPU-list parser, sibling counting,
  the cpuinfo fallback, explicit-env preservation and the physical-core default.
- K8/V4 chunk + tensor-core support (2026-11-10): with `KINDI_KV=k8v4`, prompts no longer fall back to
  the per-token plan. `chunk.kv_plan` / `spec.kv_plan` select the store and value kernels from the
  engine's kv mode: `k_store_k8v4_T` plus `k_att_flash4_T` / `k_att_pv4_T` for the chunked path (V row
  stride halved), and `k_attn_vq4w` / `k_attn_vq4w_m` for tensor-core decode and verify. `tp.py` no
  longer refuses `KINDI_TC=1` with k8v4 (it still refuses `KINDI_SPEC`) and refuses k8v4 on any head
  size other than 256. `KINDI_ATT_TC=1` and `KINDI_FD=1` remain q8-only: with k8v4 they log a fallback
  to the V4 flash chunk / the standard verify path instead of being refused. q8_0 kernels, buffers and
  arithmetic are untouched. Evidence: `python -m pytest tests/ -q` -> 140 passed;
  `python -m kindi.config_reference --check` -> 0; `python -m compileall -q kindi` -> 0; `ChunkKernels()`
  and `SpecKernels()` compile on the 4060 Ti; `bench-logs/check_k8v4_chunk.py` -> store matches NumPy,
  the K bytes are identical to `k_store_q8`, chunk V4 values equal decode `k_attn_vq4w` + `k_attn_combn`
  bit for bit (0.0), flash V4 vs the exact path 1.49e-07 abs. End-to-end on the 27B pack (RTX 4060 Ti,
  8 GB): retrieval at 7,496 and 23,319 prompt tokens matches q8_0, 5/5 codes and the single follow-up
  in both formats; a 44,625-token run was aborted after 36 min at ~21 tok/s against 151 tok/s at 24K
  (>3x rule), so 49K quality is unmeasured and the long-context path needs root-causing. q8_0 fits
  only 35,328-35,840 tokens on the card. Evidence: `bench-logs/27b-longctx/RESULTS.md`. The cache
  remains lossy (2/4 prompts exact, see `bench-logs/27b-matrix/REPORT.md`).
- origin/main v0.5.0 merged (2026-11-10): merge commit `78a66d3` brings `origin/main` (`d9fce60`,
  v0.5.0) into this branch on top of `858eac5` and `fc23913`. Resolutions keep the
  Turing/Pascal/Maxwell card sections with the AMD 840M/8060S sections, the 0.5.0 `fd_ch`/`kexpf`
  fingerprint keys plus `att_tc`, and the vq4w and B2 kernel names. The v0.5.0 one-card HB 512/4096
  guard is null-safe for the plan-engine probe; the B2 maps are built only for `tc.cu` because
  `tc_turing.cu` has no B2 kernels; `docs/AMD.md` now says 0.5.0. `docs/CONFIGURATION-REFERENCE.md`
  regenerated (25 keys, 122 env reads). Gate: 140 tests, `config_reference --check`, `compileall`,
  `bash -n install.sh`. `origin` (`ahtx/kindi`) rejected the push with 403 for `atimelessrage`;
  `main`, `freebuff/tandem-cpu-recovery` and the annotated tag `v0.4.0-k8v4` (-> `858eac5`) are on
  `myfork` (`atimelessrage/kindi`). The merge commit is `78a66d3`; this handoff commit follows it.
- v0.5.0 merge verified on the i7-8086K + RTX 4060 Ti (2026-11-10): `kindi verify`
  passed 4/4 prompts token-exact for 2B, 4B and 9B plain and for the 2B T=4 plan
  (`KINDI_TC=1 KINDI_TP=1 KINDI_CARDS=1 KINDI_TC_TMAX=4 KINDI_ATT_TC=1 KINDI_FD=1
  KINDI_SPEC=0`), 40/40 each except the 9B prompt 3 stop (13/13). Three 24K
  `kindi bench` runs per config: medians 2B 172.8, 4B 86.4, 9B 66.4 and 2B plan
  301.3 tok/s at position 24,000, against the pre-merge 160.6 / 79.8 / 62.3 /
  302.4; reads 224.1 / 128.6 / 92.5 / 474.3. No 27B pack is installed, so the
  K8/V4 long-context path was not rerun after the merge. The pre-merge checkpoint
  is tagged `v0.4.0-cpu-threads` (`fc23913`); the merged, verified state is
  tagged `v0.5.0-k8v4` at this handoff commit.
- K8/V4 two-card + cache verification (2026-11-10, `f7de617`): **the two-card run was not done** - this
  machine has one RTX 4060 Ti and `TPEngine` refuses `W=2` without two devices, so the K/V split, the
  per-card exchange and a two-card save/restore are unverified. One-card K8/V4 cache
  save/restart/restore passes end to end (`bench-logs/test_k8v4_cache.py` -> ALL PASS): a 867-token
  opening is saved when two requests part after it, a fresh process logs `saved openings: 1` and
  `loaded a saved opening of 867 tokens (0.1s)`, the reply is identical to the cold run, and
  `cached_tokens` is 867 of 888. The two-card four-part K/V serialization layout round-trips
  byte-exact with only the device selection stubbed (`bench-logs/test_k8v4_dist_cache.py`) - a layout
  check, not a two-card run. `kindi verify` with k8v4: 2B **2/4 token-exact** (first differences at
  tokens 36 and 35, 36/40 and 35/40) while the q8_0 control is 4/4, and the 4B is 4/4; K is q8_0 in
  both modes, so the 2B difference is the 4-bit V loss, and forcing (`KINDI_CHUNK_MIN=1`) or disabling
  (`KINDI_CHUNK=0`) the chunk path gives the same tokens. The 9B cannot run k8v4: the plan engine
  refuses its Hadamard block 4096 and the plain engine ignores `KINDI_KV`. No 27B pack is installed.
  Gate: 140 tests, `compileall -q kindi`, `config_reference --check`. Evidence:
  `bench-logs/k8v4-cache/RESULTS.md`. k8v4 cannot be made token-exact (4-bit V), so it is quarantined:
  `tp.kv_mode()` refuses it unless `KINDI_KV_EXPERIMENTAL=1` and the message names the measured loss;
  `tests/test_k8v4.py` pins the gate, the exact opt-in, the V4 kernel selection, the cache-fingerprint
  isolation and the generated-reference row, and `docs/CONFIGURATION-REFERENCE.md` is regenerated.
- PQ2_0 / Q2_0 conversion support (2026-11-10): `kindi/convert.py` accepts GGUF type 142 (PQ2_0, 128-weight
  groups, 34-byte blocks: fp16 `d` then 32 bytes of 2-bit slots) and type 42 (Q2_0, the same slots in 64-weight
  18-byte blocks), records `source_format` in `kindi_pack.json`, and gates on the pack's one-scale-per-128-weights
  rule (the two 64-halves of a Q2_0 group must share the fp16 scale, else it refuses). Both 2B GGUFs were
  downloaded and `sha256sum -c`-verified (PQ2_0 517,312,992 B, Q2_0 546,689,504 B). The conformance gate is
  byte identity with the PTQ1_0 pack: `bench-logs/convert_pq2_q2.py` reports `ALL PASS` for both (bcodes, dpm,
  f32, both sign files, tokenizer, `bcodes_index.json` identical; only the recorded source type and
  `source_format` differ), and `kindi verify` is 4/4 token-exact for both new packs. Two format-only bugs were
  fixed: `q2_groups` now handles several 128-groups per row, and the pack's `bc_off`/`dpm_off` accumulators now
  advance by `ne0/4` bytes and `ne0/128` scales instead of the source group size (Q2_0 has twice as many
  64-groups, which had corrupted `bcodes_index.json` and the manifest sizes even though the stream bytes were
  right). Benchmark at the common 24,576-token context (all three packs fit, 6213 MiB free, three runs): median
  generation at position 24,000 PTQ1_0 170.8, PQ2_0 162.1, Q2_0 160.8 tok/s; read 207.7 / 214.7 / 160.1. The
  spread within one pack (148.8-192.1 tok/s at 16K) is wider than every between-pack difference, and the packs are
  byte-identical, so no format speed claim is made. Evidence: `bench-logs/pq2-q2-2b/RESULTS.md`. Gate: 161 tests,
  `compileall -q kindi`, `config_reference --check`. Only the 2B was converted; 4B/9B/27B PQ2_0/Q2_0 are untested
  and no 27B pack is installed.
- Native Windows CPU build (2026-11-10): `kindi/cpu.py` builds `libkcpu.dll` with clang and libomp when
  `os.name == "nt"`, with no vcvars shell (clang finds MSVC 14.51 and the Windows SDK through vswhere), and
  `kindi/kcpu.c` exports its four entry points with `__declspec(dllexport)`. `KINDI_CC`/`KINDI_LLVM`
  point at the clang 18.1.8 install. Threads default to the physical cores from
  `GetLogicalProcessorInformationEx` (6, not 12). Native `KINDI_DEVICE=cpu` verify: 2B/4B/9B 4/4
  token-exact at 33.7/16.7/9.1 tok/s, against 28.1/13.0/5.6 on WSL2. `tests/test_cpu_build.py` pins both command lines and
  `tests/test_cpu_threads.py` the topology parser; `docs/CPU-SUPPORT.md`, `docs/CARDS.md`,
  `docs/BENCHMARKS.md` and `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/` record the result.
- Generated support matrix and roadmap (2026-11-10): `kindi/support_matrix.py` renders
  `docs/SUPPORT-MATRIX.md` from `catalog.MODELS`, `convert.TERNARY`, `tp.kv_mode()` and the
  `kcpu.c` intrinsics, plus a curated verification table that must have a row for every catalog
  model; `python -m kindi.support_matrix --check` and 8 tests fail on drift. `docs/ROADMAP.md` is
  now the single index of the archived no-go work (continuous batching 0.69x, per-slot refactor,
  tandem runtime, quarantined k8v4) with the gate that would reopen each, and the three archived
  documents point at it. Gate: 172 tests, `config_reference --check`, `support_matrix --check`.
- Sampling profiles and presence penalty (2026-11-10): `presence_penalty` (OpenAI's additive form, -2..2,
  applied once per token already in the prompt or reply) is a request field and a config key;
  `sampling_profile = publisher` selects the Bonsai publisher's values from `kindi/profiles.py`
  (non-thinking 0.7 / 0.8 / 20 with presence 1.5; thinking 1.0 / 0.95 / 20 with presence 1.0 and an
  8192-token default output). Precedence: request > explicitly configured key (`config.load` records
  `cfg["explicit"]`) > profile > built-in default; `sampling_profile = default` keeps existing behaviour.
  Presence is applied to the full logits before top-k/top-p, like `repeat_penalty`, and disables
  speculative drafting. The Bonsai backends (2B/4B/9B/27B) apply it; Gemma 4 and Nanbeige do not, as with
  `repeat_penalty`. Gate: 189 tests (17 new: profile selection/precedence, additive/negative/zero presence,
  greedy presence, explicit-key tracking, HTTP range validation and Ollama mapping), `config_reference
  --check`, `support_matrix --check`, `compileall -q kindi` all pass. No GPU run: sampled output and the
  profile values are unmeasured; the default profile leaves token-exactness unchanged.
- Installer card classification + dual pairing (2026-11-10): `card_class()` now classifies every sm_80+
  card as `tc` from compute capability alone (the 8 GB RTX 4060 Ti was `pascal8`), and `profile_of()`
  gives an 8 GB tc card a 24,576-token context. `pair_ok()` accepts two cards of the same class with at
  least 3,800 MiB each (maxwell, pascal8, turing, tc, p100), so dual P106, P4000, RTX 4000, 4060 Ti and
  5060 Ti setups are offered; 2 GB cards and mixed classes stay out. Two sm_80+ cards get the new `tc2`
  profile (KINDI_CARDS=2, KINDI_TC/FD/ATT_TC, TMAX=4, 512 MiB reserve, context from the smaller card,
  spec off until the two-card verify). The 27B 8 GB tc single-card profile leaves the 2B drafter off.
  Evidence: `tests/test_installer_cards.py` runs the installer's own functions against a fake nvidia-smi
  (12 pass; 7 fail with `KINDI_INSTALLER` on the old copy), 201 tests pass, `bash -n install.sh`,
  `config_reference --check`, `support_matrix --check` and `compileall` pass. No GPU verify and no Linux
  installer run, so this is not a support claim yet.
- Cross-mode benchmark matrix (2026-11-10): `docs/benchmarks/build_cross_mode.py` reads the measured native
  Windows CPU-only (`2026-11-10-i7-8086k-windows-cpu`, 6 threads, one run) and GPU-only
  (`2026-11-09-i7-8086k-rtx4060ti`, median of 3) runs and writes `CROSS-MODE.md`/`.csv` for the 2B, 4B and 9B
  at positions 100-24,000 plus the 1,000-token read. CPU-only at 24K is 8.2/3.1/2.6 tok/s; GPU-only/CPU-only
  is 19.6x/25.7x/24.0x. The WSL2 CPU cells are archived. The GPU+CPU tandem rows are present and `n/a`: there
  is no `TandemEngine`, so no tandem number is claimed; the page names the TANDEM-STATUS gates that must pass
  first. Gate: `python docs/benchmarks/build_cross_mode.py --check`, 203 tests, `compileall`, config/support
  checks; no new GPU run.
- i7-8086K CPU support row and CPU server chat (2026-11-10): `docs/CARDS.md` now lists the i7-8086K
  (6 threads, 2B/4B/9B, native Windows 33.7/16.7/9.1 tok/s verify, native 24K table). The support gate
  is verified on the native build: `KINDI_DEVICE=cpu` served the 2B/4B/9B on 127.0.0.1:8199 at a 1,024
  context, answering `POST /v1/chat/completions` non-stream and stream with 0 errors, 0 MiB GPU memory
  and `kindi.cpu.Engine` at 6 threads. Non-stream rates were 32.9/19.2/11.5 tok/s; the streams were
  17/7 (2B), 17/1 (4B) and 15/9 (9B), each ending in `[DONE]`. The 4B loaded in 55.8 s, the 9B in
  123.7 s. `docs/CPU-SUPPORT.md`, `docs/ROADMAP.md`, `docs/BENCHMARKS.md`, both CPU benchmark pages and
  the generated support matrix now say so. Gate: 214 tests, `support_matrix --check`,
  `config_reference --check`, `compileall`, `bash -n`, `git diff --check`.
- CI CPU build-and-smoke job (2026-11-10): `.github/workflows/ci.yml` adds `cpu-engine-smoke`, which
  compiles `kindi/kcpu.c` with gcc for the documented AVX2/FMA/F16C tier and runs the CPU build and
  thread tests plus `tests/test_cpu_smoke.py`. The smoke test loads the library through
  `kindi.cpu._lib()`, checks `kc_sizeof_model`/`kc_sizeof_layer` against `ctypes.sizeof(Model/Layer)`
  and calls `kc_init`; it skips unless `KINDI_CPU_SMOKE=1` and `KINDI_CPU_CACHE` hold a prebuilt
  library, so the default suite needs no C toolchain. No model pack, GPU or full token is used.
  Gate: 203 passed + 2 skipped, the three CPU test files pass with `KINDI_CPU_SMOKE=1`, `compileall`,
  `bash -n`, config/support checks and `git diff --check`.
- RTX 4060 Ti tc verify (2026-11-11 one-pass matrix): `kindi verify` with `KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1
  KINDI_TC_TMAX=4 KINDI_SPEC=0 KINDI_HEAD=exact` on the 4060 Ti (sm_89, 8 GB, Windows) gives 2B **4/4 token-exact** at
  370.2 tok/s, 4B **2/4** (prompts 2 and 3; first differences at tokens 6 and 37), and 9B **refused before
  graph build** (Hadamard block 4096); the 27B tc plan passes with 3 exact prompts + 1 llama.cpp tie (token 14, gap 0.0064) at 38.9 tok/s in the 2026-11-11 one-pass matrix, using 724 nodes/decode token and leaving 1,128 MiB free. The plain `KINDI_TP=0` 4B/9B paths pass 4/4 at 114.9/72.5 tok/s.
  `docs/CARDS.md` and `docs/CARD-SUPPORT.md` record this as a partial result, not a tc support claim. Raw logs (untracked): `bench-logs/verify-matrix-tc-4060ti/` (one-pass matrix); earlier `bench-logs/verify-4060ti-tc/` and `bench-logs/27b-tc-4060ti/` logs remain. The 27B pack and GGUF are at `D:/kindi-27b` (C: had 3.8 GB free); the follow-up 24K benchmark and server/cache run are recorded in `bench-logs/27b-tc-24k/RESULTS.md`. No inference code changed.
- 27B tc 24K + server/cache (2026-11-11): the tc profile above on the 4060 Ti at `--ctx 24576` built
  T=1..4 (724 nodes/decode token), read 40.5 tokens/s, generated 37.6/37.1/36.6/34.7/33.9 tokens/s at
  positions 100/2,000/8,000/16,000/24,000, and left 351 MiB free. Three real servers ran a fresh-state
  cold/save/restart cycle: 888 prompt tokens, an 867-token opening saved in 0.2 s and loaded in 0.3 s,
  cached 867/888 in process and after restart, bit-identical replies, and a stream ending in `[DONE]`;
  350-398 MiB stayed free. Cold loads were 56.4/54.6 s and the restart was 6.0 s. Evidence:
  `bench-logs/27b-tc-24k/` and `bench-logs/test_27b_tc_cache.py`; `docs/CARDS.md` and
  `docs/CARD-SUPPORT.md` record the qualification. No inference code changed.
- Native Windows 24K CPU benchmark (2026-11-10): `python -m kindi --pack ... --ctx 24576 bench` on the
  native Windows engine (6 threads auto-detected, exact head, one fresh process per model) measured 2B
  45.1 read / 35.9 / 29.3 / 19.4 / 11.9 / 8.2; 4B 19.2 / 15.2 / 13.4 / 6.8 / 4.3 / 3.1; 9B 9.9 / 8.7 /
  7.7 / 4.9 / 3.5 / 2.6 tokens/s. All three `bench` processes exited 0. The rows are in
  `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/runs.csv`; the generator now reads them and
  `CROSS-MODE.md`/`.csv` use 8.2/3.1/2.6 at 24K (GPU/CPU 19.6x/25.7x/24.0x). The GPU and tandem rows are
  unchanged. Raw logs: `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/raw/bench-24k-*.log`. Gate: 203 tests,
  `python docs/benchmarks/build_cross_mode.py --check`, config/support checks, `compileall`, `bash -n`,
  `git diff --check`.
- CPU thread-count sweep (2026-11-10): 6, 8, 10 and 12 threads were measured on the i7-8086K. The
  affinity probe shows 8 uses six cores with two SMT pairs, 10 uses four pairs and 12 doubles all six.
  WSL2/libgomp 2B at 512 tokens: 6/8/10 read 38.2/44.3/25.5 and generation 28.6/31.2/24.1 tok/s; 12
  collapses to 1.7. WSL2 2B at 24K: 6/8 generation at 24,000 is 4.5/6.2 and read is 38.3/40.0. Native
  Windows/clang 2B: 6/8/10 read 43.2/28.4/13.5 and generation 31.7/21.6/9.8; native passive waiting
  lifts 10 to 20.0/14.7 but not past 6. Stage timers (`kc_times`) localize the WSL2 gain to GEMV
  (read 0.480 to 0.404 s, generation 0.323 to 0.275 s); on native the same GEMV worsens (0.406 to
  0.506 and 0.278 to 0.452). The barrier-heavy GDN/attention stages do not improve, so the effect is
  SMT latency hiding in `rows8`, not barrier load balance. 4B read regresses under WSL2 8
  (14.8 to 10.2) while generation is +3%; 9B is 7.3/6.4 at 6 and 7.5/7.3 at 8. Decision: keep the
  physical-core default; 8 is an explicit WSL2/libgomp tuning, 10/12 are not viable, and auto-tuning
  is not justified by these mixed model/read-vs-generation results. Evidence (untracked):
  `bench-logs/thread-sweep-20261110/`; docs updated in `docs/CPU-SUPPORT.md`, `docs/CARDS.md`,
  `docs/CONFIGURATION-REFERENCE.md`, `docs/benchmarks/2026-11-09-i7-8086k-cpu/README.md` and
  `kindi/cpu.py`. Gate: 203 tests, `compileall`, `bash -n`, config/support checks, `git diff --check`.
- i7-8086K processor details (2026-11-10): `docs/CPU-SUPPORT.md` and `docs/CARDS.md` now record the
  Coffee Lake 14 nm, 6-core/12-thread part, 4.00 GHz base, up to 5.00 GHz single-core turbo, typical
  4.3-4.4 GHz all-core turbo (board/cooling/power dependent), 12 MB L3, unlocked multiplier, and the
  2018 40th-anniversary/i7-8700K positioning. The benchmark caveat is explicit: no explicit overclock
  or thermal/power clamp was applied, but per-core clocks and turbo residency were not logged. No code changed.
- 9B CPU queue load test (2026-11-10): the native 9B server (6 threads, ctx 1024, `KINDI_QUEUE_MAX=8`)
  with 12 concurrent clients returned 9 HTTP 200 and 3 HTTP 429; all 429s carried `Retry-After: 1`
  and returned in 8-10 ms. `/health` peaked at 8 waiters, metrics recorded 3 rejects and 0 errors, and
  the eight waiters drained serially. Recorded in `docs/AGENTS.md`; raw evidence is untracked in
  `bench-logs/9b-queue-load/`. No code changed.
- Intel Arc A770 support goal (2026-11-10): `docs/INTEL-ARC.md` records the one-card, two-card and
  CPU+Arc tandem goal, the 22-kernel/CuPy audit, the SYCL/oneAPI backend design, acceptance matrix,
  performance gates and M0-M8 milestones. M0 is done; M1 is blocked because the test host has only
  an RTX 4060 Ti and no `sycl-ls`/Arc device. No Intel backend code exists and this is not a support
  claim. `docs/ROADMAP.md` and `docs/CARD-SUPPORT.md` link the plan.
- P106-100 single/dual support and CPU-tandem gate (2026-11-10): `docs/P106.md` records the official
  1,280-core/6 GB/192-bit specifications, the single-card 2B/4B/9B path, the two-card 27B path, the
  headless/PCIe caveats and the CPU+tandem gate. The installer already classifies the 6 GB card as the
  `maxwell` 24,576-token profile, offers no single-card 27B, pairs two P106s and gives the pair the
  `k1200x2` profile; new fake-GPU tests pin the single profile, the no-27B rule and the dual profile.
  No P106 hardware is attached, so no `verify`/server/cache result exists. There is no general Bonsai
  CPU+GPU tandem runtime (`docs/TANDEM-STATUS.md`); the Nanbeige host-layer path is separate.
  `docs/CARDS.md`, `docs/CARD-SUPPORT.md` and `docs/ROADMAP.md` link the page. Gate:
  `pytest tests/test_installer_cards.py -q` -> 15 passed; `pytest tests/ -q` -> 206 passed, 2 skipped;
  `compileall`, the support/config checks and `bash -n install.sh` pass. No GPU run, no P106 hardware.

- CPU per-layer tandem entry point (2026-11-11): `kindi/kcpu.c` exports `kc_embed`, `kc_layer` and
  `kc_head`, and `kindi/cpu.py` exposes `begin`, `layer`, `head`, `feed_layers`, `step_layers`, plus
  `hidden`/`get_hidden`/`set_hidden`. `kc_layer` takes the complete residual in `hidden` and leaves the
  complete residual there; `step`/`feed` keep the original single-OpenMP-region path. `tests/test_cpu_layers.py`
  has fake-library ordering/validation tests and an opt-in `KINDI_CPU_PACK` exactness check. Native
  Windows i7-8086K (6 threads, clang 18.1.8): the five-token layerwise sequence is token- and
  hidden-state-exact against `step` on the 2B/4B/9B packs (5 passed per pack); no pack: 4 passed,
  1 skipped; compiled CPU smoke: 16 passed. No `TandemEngine`, layer placement or GPU handoff exists;
  the RTX 4060 Ti is not used by this slice.

- P106 verification runner (2026-11-11): `tools/verify_p106.py` writes `results.json`, `RESULTS.md`,
  a generated `kindi.conf` and raw logs for the single-card 2B/4B/9B verify, the real server/cache
  save/in-process-hit/restart/invalid-request cycle, and the two-card 27B verify. It requires exactly two
  visible P106-100s for a full run (or `--single-only`), pins `CUDA_VISIBLE_DEVICES` to their nvidia-smi
  indices, removes inherited `KINDI_*` settings, and requires `plan engine ready: 2 cards` for the dual
  result. Evidence: `pytest tests/test_verify_p106.py -q` -> 10 passed; `python tools/verify_p106.py --dry-run`
  -> exit 0 on the RTX 4060 Ti host (no P106). Not run on P106 hardware; not a support claim.

- Secret/personal-data guard (2026-11-10): `tests/test_no_secrets.py` scans every tracked file
  (binary files skipped) for private keys, cloud tokens, JWTs, email addresses and local home paths;
  `.gitignore` now covers raw evidence, audit scratch files and key material. 39 local home-path
  occurrences in 33 tracked benchmark files were redacted to `<home>`. The scan and the full gate
  pass; commit metadata still uses the repository's configured author identity and was not rewritten.

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
| `python3 -m kindi.support_matrix --check` | python | the generated support matrix matches the code |
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
   missing). **Unblocked on 2026-11-09:** `T:/kindi` is clean, so the central config reference,
   installer/checksum, support matrix and dead links can be reconciled here; run the worktree
   status check immediately before a shared edit.
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
   the CPU acceptance matrix. The i7-8086K engine is now verified in WSL2 (2B/4B/9B 4/4; 6-thread
   24K benchmark); the Xeon/i9 and full server/native-Windows claims remain open.
8. [x] Installer safety release prepared: the GRUB guard is committed in `2b84658`, `docs/INSTALL.md`
   pins the fixed artifact hash `c05d0190...`, and `docs/RELEASE-CHECKLIST.md` records reconciliation,
   tests, exact-artifact hashing and publication checks. Evidence: `bash -n install.sh` OK;
   `pytest tests/test_installer_grub.py -q` -> 9 passed; `pytest tests/ -q` -> 58 passed;
   `compileall -q kindi` OK; `sha256sum -c` -> `install.sh: OK`; no control bytes. No tag, push or
   upload was made. The main worktree's empty `install.sh` and dirty `docs/INSTALL.md` remain a
   publication blocker; the release checklist says so.
9. [x] CI made runnable and the plan engine got a CPU-only regression test: `.github/workflows/ci.yml`
   installs `tests/requirements-test.txt` (the old `numpy pytest` install left `test_server.py`
   uncollectable) and ShellChecks the committed LF blob (a Windows CRLF checkout fails SC1017).
   `tests/test_plan_engine.py` imports `kindi/spec.py` and `kindi/tp.py` with a stand-in `cupy` and
   pins TMAX = 8 and the `k -> TMAX - 1` draft clamp (T = 8 verify), the 2B HB-512 one-split rule,
   and both HB-4096 (9B) guards. Evidence: `pytest tests/test_plan_engine.py -q` -> 20 passed;
   `pytest tests/ -q` -> 78 passed; `bash -n install.sh`, `compileall -q kindi` and ShellCheck on
   `git show HEAD:install.sh` -> exit 0; five mutations of the guarded lines each fail their test.
   No GPU run: the plans themselves were not built or executed.
10. [x] Configuration reference generated from the code: `kindi/config_reference.py` reads
    `config.DEFAULTS` (key, type, default and the inline comment) plus every `os.environ` read of a
    `KINDI_*` name under `kindi/`, and writes `docs/CONFIGURATION-REFERENCE.md`.
    `python -m kindi.config_reference --check` and `tests/test_config_reference.py` fail when a key
    or switch is added without regenerating. Evidence: `--check` exit 0; 6 new tests and 84 full
    tests pass; adding `KINDI_BRAND_NEW` or a new DEFAULTS key to a copy makes `--check` exit 1 and
    the list tests fail. `docs/CONFIGURATION.md` itself is not touched: it is main-owned and dirty,
    so its hand-written tables and the generated file must be reconciled by hand.
11. [x] 2B spec acceptance and throughput on non-repetitive prompts measured (RTX 4060 Ti,
    greedy, ctx 1024, N=192, best of 3): n-gram drafting accepts 17.9% at k=1 and 8.4% at k=3,
    with 1.06-1.08 tokens per pass; 364.9 tok/s at k=3 vs. 328.9 plan-only and 168.1 plain
    (2.17x plain, +11% over plan). The copy-task control reads 86.1% with the same counters, and
    spec == plan on every prompt. `plan != plain` from token 30/140 on two prompts (q8_0 K/V), a
    known numerical divergence, not a spec error. Evidence:
    `bench-logs/measure-2b-spec-norepeat/RESULTS.md`, `bench-logs/measure-2b-spec-norepeat.log`,
    `bench-logs/measure-2b-spec-norepeat.json`. No code changed; `bench-logs/` stays untracked.
12. [x] Agent serving layer (`task_0013`) advanced without touching leased files: `kindi/server.py` now
    admits at most `KINDI_QUEUE_MAX` (default 8) waiting requests and answers the next with **429** and
    `Retry-After: 1` instead of queueing without a bound; `GET /metrics` exports Prometheus counters for
    requests, prompt/cached/completion tokens, speculative drafted/accepted tokens, generation seconds and
    the queue; `GET /api/tags`, `POST /api/chat` and `POST /api/generate` speak Ollama's NDJSON API (streaming
    and not), so Ollama clients work without an OpenAI shim. `docs/AGENTS.md` documents the endpoints and the
    429 contract. Evidence: `python -m pytest tests/test_server.py -q` -> 18 passed; `pytest tests/ -q` -> 92
    passed; `python -m kindi.config_reference --check` -> 0; `compileall -q kindi` -> 0. Remaining in task_0013:
    multi-GPU replicas (needs `tp.py`, main-owned) and a deployment note in the main-owned README.

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
