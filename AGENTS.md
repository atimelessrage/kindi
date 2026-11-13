# AGENTS.md - working on Kindi

Kindi is a custom inference engine for ternary Bonsai models (plus Gemma 4 and Nanbeige), with a
CUDA/CuPy backend, a CPU backend, an OpenAI-compatible server and a one-command installer.
`docs/AGENTS.md` is the guide for agents that *use* the API; this file is for agents that
*change the repository*.

Keep this file current: it is the handoff document. Tick a step when it is done and add one
line of evidence (command + result). A task must be resumable from this file alone.

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
| `T:/kindi` | `main` | `1f2c682` | main checkout; clean; best-of main |
| `<worktree>` | `freebuff/tandem-cpu-recovery` | `1f2c682` | this session; best-of main (identical to `main`) |

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

### Current leases (checked 2026-11-12)

| task | worktree | branch | shared files | state |
|---|---|---|---|---|
| AVX2/F16C no-FMA fallback | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/kcpu.c`, `kindi/cpu.py`, `install.sh`, `README.md`, `docs/CPU-SUPPORT.md`, `docs/SUPPORT-MATRIX.md`, `docs/CONFIGURATION-REFERENCE.md`, `tests/test_cpu_build.py`, `tests/test_cpu_features.py`, `tests/test_installer_cpu.py`, `tests/test_cpu_avx_fallback.py`, `tests/test_support_matrix.py`, `tests/README.md`, `kindi/tandem.py`, `AGENTS.md` | done; `rows8()` is `__AVX2__`-gated with an FMA branch and a separate mul+add branch, F16C and the AVX-only path are unchanged; `kindi.cpu.cpu_tier()` and the installer print the selected tier, and the Windows CPU cache key separates FMA from no-FMA. Forced clang 18.1.8 `-mavx2 -mno-fma -mf16c` build: 2B `kindi verify` 4/4 token-exact at 31.7 tok/s (`bench-logs/cpu-nofma-verify.log`); 334 passed, 8 skipped; generated checks, `bash -n`, LF ShellCheck and `git diff --check` pass. No native no-FMA hardware and no 4B/9B no-FMA run |
| Dory1.0-4B long-term looped-model goal | this worktree | `freebuff/tandem-cpu-recovery` | `docs/DORY.md`, `docs/ROADMAP.md`, `AGENTS.md`, `bench-logs/dory-d0/**`, `tools/dory_gguf_inventory.py` | D0 reference evidence complete; Q5_K_M 3,673,288,864 B SHA-256 `96080be4...`; BF16 metadata hashed; 507-tensor manifest (252 Q5_K/36 Q6_K/219 BF16) in `bench-logs/dory-d0/tensor-inventory.json`; GGUF `dory.num_loops=4` vs BF16 config 2; llama.cpp PR 30306 tip `1c3cdd...` built CPU-only; four prompts x k=1/2/4, 32 greedy tokens each, 12/12 runs exit 0 in `bench-logs/dory-d0/goldens/goldens.json`, k=2 repeat exact; no Kindi converter/engine/catalog/installer/support claim; D1 pending; `docs/CARDS.md` untouched because `T:/kindi` was dirty at that time |
| Multi-ISA CPU dispatch | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/cpu.py`, `kindi/kcpu.c`, `kindi/__main__.py`, `kindi/support_matrix.py`, `tests/test_cpu_build.py`, `tests/test_cpu_features.py`, `README.md`, `docs/CPU-SUPPORT.md`, `docs/SUPPORT-MATRIX.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/ROADMAP.md`, `AGENTS.md` | done; five explicit tiers (`avx2-fma-f16c`, `avx2-f16c-no-fma`, `avx2-fma`, `avx2`, `avx-only`) are selected from the host flags or `KINDI_CPU_ISA`, built with explicit flags into per-tier cache paths, and checked against `kcpu.c:kc_isa_name()`; 2B verify is 4/4 token-exact at 43.6/43.9/8.9 tok/s for FMA/F16C, no-FMA and AVX-only (`bench-logs/cpu-dispatch-*.log`); 355 passed, 8 skipped, generated checks pass; `avx2-fma`/`avx2` are compile-tested only, with no native no-F16C or Sandy Bridge hardware |
| BF16/FP8 real-model validation | this worktree | `freebuff/tandem-cpu-recovery` | `docs/FP8-BF16.md`, `docs/ROADMAP.md`, `AGENTS.md` | blocked 2026-11-11; `D:\ollamaModels` holds only `gemma4:26b` Q4_K_M (30 layers, width 2816, MoE type 7 rejected by `inspect_model`) and `T:\strata\models\models` holds Qwen3.6-35B Q4_K_XL; no BF16/FP8 E4B GGUF, no llama.cpp CLI and no matching reference JSON, so no token-exactness run was possible; needed: an E4B GGUF with types 30/43/44 plus a pinned llama.cpp reference |
| 4B/9B ISA-tier verification | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CPU-SUPPORT.md`, `bench-logs/cpu-isa-4b-9b/**`, `AGENTS.md` | done; all five forced tiers are 4/4 token-exact on the 4B and 9B (i7-8086K, native Windows, clang 18.1.8, 6 threads, ctx 1024, one run per tier/model). 4B tok/s 18.1/15.8/17.8/19.2/3.9 and 9B 10.7/10.4/10.9/11.0/2.0 for `avx2-fma-f16c`/`avx2-f16c-no-fma`/`avx2-fma`/`avx2`/`avx-only`; all ten exited 0 and reported ALL PASS. Evidence: `bench-logs/cpu-isa-4b-9b/{RESULTS.md,summary.json,logs/}` and a fresh `KINDI_CPU_CACHE`. Still forced tiers on an AVX2 host, not native Sandy Bridge or native no-FMA support |
| Native FP8 tensor-core GEMM | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/gemma4_fp8.cu`, `kindi/gemma4.py`, `tests/test_gemma4_fp8_mma.py`, `bench-logs/bench_fp8_mma.py`, `docs/FP8-BF16.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION-REFERENCE.md`, `AGENTS.md` | done; E4M3/E5M2 `mma.sync` m16n8k32 kernels on sm_89+ with per-token activation scaling, selected in `linear_batch` for cc >= 8.9 with `KINDI_G4_FP8=0` as the opt-out; 20 focused tests pass and the full suite is 367 passed, 10 skipped; synthetic Ada timings are 0.11-0.76 ms (15.1-35.5 TFLOPS, `bench-logs/fp8-mma-ada.log`); no real FP8 model and no Blackwell hardware; the manual Blackwell sm_100/sm_120 lane now exists; a local gate on the Ada-only host returned unsupported (exit 2) on 2026-11-12, so this is a kernel path, not a model/support claim; `docs/CARD-SUPPORT.md` was not edited because `T:/kindi` was dirty at that time |
| Structured access log and dependency updates | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/server.py`, `kindi/config.py`, `kindi.conf.example`, `tests/test_server.py`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/AGENTS.md`, `docs/ROADMAP.md`, `.github/dependabot.yml`, `AGENTS.md` | done; `log_json = 1` writes one JSON line per inference request (request id, status, duration, queue wait, token counts and tok/s) and returns `X-Request-ID`; rejected and failed requests are logged, headers and keys are not; Dependabot opens weekly reviewed PRs for `tests/requirements-test.txt` and the workflow actions. Evidence: new server tests, `test_config_example` and generated `config_reference --check` pass. Actions are pinned to the v4/v5 commit SHAs the tags resolved to; Dependabot keeps them current |
| Blackwell FP8 verification lane | this worktree | `freebuff/tandem-cpu-recovery` | `tools/verify_blackwell_fp8.py`, `tests/test_verify_blackwell_fp8.py`, `.github/workflows/blackwell-fp8.yml`, `docs/FP8-BF16.md`, `docs/CI.md`, `docs/ROADMAP.md`, `bench-logs/blackwell-fp8-dispatch-20261112/**`, `AGENTS.md` | done; `tools/verify_blackwell_fp8.py` accepts only compute capability 100/sm_100 or 120/sm_120 and exercises both E4M3 and E5M2 quantization, native GEMM at 64x64x64, 37x96x5 and 130x128x70, and `Ops.linear_batch`, recording device/driver/CuPy/CUDA, errors and timings in `results.json`, `RESULTS.md` and `run.log`; non-Blackwell or arch mismatch is `unsupported` exit 2, never a pass. Manual `.github/workflows/blackwell-fp8.yml` needs `[self-hosted, gpu, <selected arch>]` and an explicit `sm_100`/`sm_120` input, so each job lands on an architecture-specific runner. Evidence: 4 CPU-only tests, 20 focused FP8 tests (5 skipped because cuBLASLt is absent), docs/config/support checks pass; local RTX 4060 Ti Ada sm_89 run is `unsupported` exit 2 in `bench-logs/blackwell-fp8-ada-refusal/`. No Blackwell hardware, so no Blackwell pass exists; a fresh local gate run on 2026-11-12 again returned unsupported exit 2 in `bench-logs/blackwell-fp8-ada-refusal-20261112/`. Dispatch attempt 2026-11-12 was blocked before a runner job: the local GPU is sm_89, `gh` and all GitHub token variables are absent, and the untracked workflow is absent from both remote branches (contents API 404; remote workflows are only CI and GPU smoke). No sm_100/sm_120 hardware result exists; see `bench-logs/blackwell-fp8-dispatch-20261112/DISPATCH-BLOCKED.md` |
| Raon-1 mini R0 and long-term MoE goal | this worktree | `freebuff/tandem-cpu-recovery` | `docs/RAON-1-MINI.md`, `docs/RAON-1-MINI-R0.md`, `docs/MOE-TANDEM.md`, `docs/ROADMAP.md`, `AGENTS.md`, `bench-logs/raon-r0/**` | R0 artifacts complete 2026-11-11; base `8c9b802...`, 7 shards / 32,702,145,088 file bytes, 19,434 tensors / 32,699,649,368 tensor bytes; Q4_K_M 19,686,434,240 B `ba56ad4b...`; 731 GGUF tensors; 18 RDA + 7 MLA, layer 0 dense + 24 MoE, 128 experts top-6 + 1 shared; 8,616,960 B per MXFP8 expert; Apache-2.0; exact greedy harness and Q4 sanity command pinned; no base/Q4 load, no reference goldens, no backend/catalog/installer/support claim; cache is 8,064 B/token compressed vs 179,200 B/token expanded in HF; router tie and llama.cpp cache layout still open; `docs/CARDS.md` untouched |
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
| Tandem joint prompt cache | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/tandem.py`, `kindi/cache.py`, `kindi/server.py`, `tests/test_tandem.py`, `tests/test_cache_decisions.py`, `bench-logs/tandem-cache/**`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `docs/PROMPT-CACHE.md`, `docs/ROADMAP.md`, `AGENTS.md` | done; `TandemRunner` now subclasses `cache.Runner` over a joint `_TandemCacheView`; checkpoints cover both owners' M/conv; parked conversations and saved openings also cover both owners' K/V; saved openings are fingerprinted by placement, layer ranges, CPU ISA/threads and `tandem.py`; no VRAM checkpoint slot. Full suite: `python -m pytest -q tests/` 362 passed, 19 skipped; docs links and `config_reference --check` pass. A real 2B tandem cache check now passes in both placements: the extension reused 320 tokens, the branch restored the 256-token joint checkpoint, and a new runner loaded the 256-token disk opening; all six cached continuations matched cache-off token-for-token. A real HTTP `serve` run then saved a 344-token opening, restarted, loaded it (344 cached versus 0 with the cache off) and returned the identical reply in both placements. The same harness also passed at 4K and 8K context with 1,024/2,048-token checkpoints: extension reuse 4,080/8,176 tokens and branch restore 3,072/6,144 tokens, all exact. The 9B pack then passed the 512-context check in both placements with 6 CPU layers: extension 320, branch and saved opening 256, all exact; CPU load was 105-108 s and GPU load 5.8-6.0 s; the HTTP server save/restart/reload path also passed in both placements with 344 cached after restart versus 0 cache-off. This found and fixed `Openings.load` assigning NumPy K/V into CuPy views with a plain assignment; it now uses `_to_dev`; a GPU-free regression in `tests/test_cache_decisions.py` uses a CuPy-like K/V array that rejects plain assignment and checks the restored prefix and untouched tail; evidence in `bench-logs/tandem-cache/`, including `RESULTS.md`, `RESULTS-9B.md`, `LONG-CONTEXT-RESULTS.md`, `server-results-cpu.json`/`server-results-gpu.json`, `server-results-9b.json`/`server-results-9b.md`, `checkpoint-cost-9b-512.json` and server logs. Checkpoint-only 9B/512 measurement: 50.25 MiB joint recurrent state, 15.9/17.4 ms median save and 10.5-11.1 ms median restore, device peak 3,063/3,065 MiB (1.8/1.7 GiB over the 1,184/1,293 MiB idle baseline), K/V 16 MiB excluded. The 9B pack also passed 4K/8K equality in both placements: extension 4,080/8,176 and branch 3,072/6,144 cached, cache-off 0, exact; evidence `context-9b-4k.json`/`context-9b-8k.json`. Prepare-only measurement: 320/325 and 256/261 tokens reused, 19.27 MiB joint recurrent checkpoint, 8.7-8.9 ms median save and 3.3-3.5 ms median restore (`bench-logs/tandem-cache/PREFIX-RESULTS.md`) |
| 27B tandem gate attempt | this worktree | `freebuff/tandem-cpu-recovery` | `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `bench-logs/tandem-27b-gate/**`, `AGENTS.md` | 2026-11-12: plain `Engine` gained generic GQA (`Engine._gqa_groups`, G≤8 at head size 256; `Engine._attn_core` launches `k_attn_sg`/`k_attn_vgn`/`k_attn_combn` for G≠4, no new kernels); `tests/test_gqa_heads.py` (8 tests); `run_gate.py --probe-gpu-owner` exit code fixed (was inverted). Tandem `cpu-first` (CPU 0..15, GPU 16..63) and `gpu-first` (GPU 0..47 + embedding, CPU 48..63) each 4/4 prompts, 40/40 each, peaks 7,005 / 7,017 MiB vs the 7,238 MiB full-GPU baseline at 8,192 context (never reduced); suite 444 passed, 19 skipped; 2B/4B/9B verify ALL PASS after the refactor; docs and `CURRENT-VS-KINDI-0.5.md` §9 updated; harness in `bench-logs/tandem-27b-gate/run_gate.py` |
| MoE expert-cache tandem design | this worktree | `freebuff/tandem-cpu-recovery` | `docs/MOE-TANDEM.md`, `docs/EXPERT-CACHE-COPY-INTERFACE.md`, `docs/RAON-1-MINI.md`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `docs/CPU_GPU_TANDEM_DESIGN.md`, `docs/ROADMAP.md`, `AGENTS.md` | design plus model-free cache prototype; no model/backend/GPU copy; pilot is the planned Raon-1 mini; first exact lane is GPU compute with copy misses, CPU miss compute is parity-gated; the cache-to-copy seam is policy-firewalled: immutable jobs/tickets/completions only, cache owns every slot and eviction decision, copy worker never calls the cache; no support claim |
| Model-free expert cache prototype | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/expert_cache.py`, `tests/test_moe_cache.py`, `tests/test_moe_cache_properties.py`, `docs/MOE-TANDEM.md`, `docs/ROADMAP.md`, `docs/TANDEM-STATUS.md`, `docs/CPU_GPU_TANDEM_DESIGN.md`, `AGENTS.md` | done; copy-only slot state machine (`FREE`/`LOADING`/`READY`/`IN_USE`), token-scoped pins, LRU over unpinned READY cache slots, transient fallback, failed-copy discard and JSON-safe metrics; 11 focused tests plus 32 seeded randomized route/acquire traces pass; the traces assert capacity, pin ownership, slot-state, atomic-rejection and metric invariants after every operation; no model, GPU copy, backend, converter, catalog/installer entry or support claim |
| Gemma 4 26B-A4B cache adaptation | this worktree | `freebuff/tandem-cpu-recovery` | `docs/GEMMA4-26B-MOE.md`, `docs/GEMMA4-26B-COPY-COMBINE.md`, `docs/MOE-TANDEM.md`, `docs/FP8-BF16.md`, `docs/ROADMAP.md`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `docs/RAON-1-MINI.md`, `AGENTS.md` | design adaptation only; local Q4_K_M header verified 30 layers / 2816 width / 128 experts top-8 + dense shared MLP / 704 expert width; 3,717,120-4,336,640 B per expert slot, 14,479,917,056 B expert weights, 2,578,661,496 B dense; layer-scoped 8-slot plans, fused gate/up, expert-id combine and parallel dense MLP; the execution contract in `GEMMA4-26B-COPY-COMBINE.md` fixes the eight-slot + transient arena, two-buffer pinned staging, one-event-per-expert copy, deterministic ascending-id combine, copy-failure cleanup and the seven-eviction-plus-transient forced-eviction trace; 25 sliding + 5 global layers, 256K KV is 5.19 GiB and does not fit 8 GB; no 26B backend, reference golden, copy-path implementation or support claim |
| Installer source checksum pin | this worktree | `freebuff/tandem-cpu-recovery` | `install.sh`, `tests/test_installer_source.py`, `docs/RELEASE-CHECKLIST.md`, `AGENTS.md` | done; the default v0.5.0 source archive is downloaded to a `.part` file and checked against `26424ec0...` before extraction; `KINDI_SOURCE_SHA256` pins a custom URL; 4 source tests and the installer suite pass, `bash -n` and LF ShellCheck pass; candidate LF `54f76308...`, CRLF `e49e1f42...`; that candidate was superseded by the 2026-11-11 AVX refresh; `docs/INSTALL.md` now names the published `43766d8b...` |
| Problems-only review triage + CPU ISA guard | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/kcpu.c`, `kindi/cpu.py`, `tests/test_cpu_build.py`, `tests/test_tc_turing_reference.py`, `tests/test_server.py`, `docs/CPU-SUPPORT.md`, `docs/CARDS.md`, `Buffy_ProblemsOnly_202611111200.txt`, `AGENTS.md` | done; report #1-3/#6 cite `kcpu_cpuid.*`/`kc_set_isa` that exist only in the preserved WIP backup, not this tree; #7-9/#58/#60/#61/#63 are already fixed/tested; #10 imports pass; added the AVX2/FMA/F16C guard and a 5060 Ti status row; the 2026-11-12 closure pass added a targeted INT8 shift-compensation regression in `tests/test_tc_turing_reference.py` (passes on the 4060 Ti) and a concurrent admission-gate regression in `tests/test_server.py`, and updated `Buffy_ProblemsOnly_202611111200.txt` to the current-worktree state; remaining items are feature/design or hardware-verification gaps, not silently claimed as fixed |
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
| Mixed-card pairing/asymmetric TP roadmap | this worktree | `freebuff/tandem-cpu-recovery` | `docs/MIXED-CARD-ROADMAP.md`, `docs/ROADMAP.md`, `AGENTS.md` | proposal plus the later opt-in engine layout, not an installer profile; WIP `PAIR-SETUPS` claims a `mixedx2` mode, but WIP `install.sh` is zero bytes and WIP `tp.py` only adjusts FFN/head; current mixed offers use equal-split `k1200x2`/`tc2` by default (`KINDI_TP_SPLIT=auto` is opt-in), `KINDI_G5=0` is global, and no mixed-card run exists; 19 installer tests pass; no WIP code merged |
| Nanbeige 33B host-layer optimization review | this worktree | `freebuff/tandem-cpu-recovery` | `docs/TANDEM-PROPOSAL-REVIEW.md`, `docs/ROADMAP.md`, `AGENTS.md` | triage only; `nanbeige.py` already overlaps 55% of host-layer FFN rows on the CPU but copies host layers; the proposal's resident 80/20 slice, CPU drafter, CPU K/V and output-head split are not implemented; no code or K6000 run |
| P106-100 single/dual support + CPU tandem gate | this worktree | `freebuff/tandem-cpu-recovery` | `docs/P106.md`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/ROADMAP.md`, `tests/test_installer_cards.py`, `AGENTS.md` | installer single/dual routing tested; no P106 hardware; general Bonsai CPU tandem blocked on the missing `TandemEngine` (Nanbeige host-layer path is separate); not a support claim |
| P106 verification runner | this worktree | `freebuff/tandem-cpu-recovery` | `tools/verify_p106.py`, `tests/test_verify_p106.py`, `docs/P106.md`, `docs/CARDS.md`, `AGENTS.md` | `tools/verify_p106.py` records the single-card 2B/4B/9B verify, the real server/cache restart, and the two-card 27B verify in one run; it pins the P106 indices, drops inherited `KINDI_*`, and refuses a full run without exactly two P106-100s. 10 CPU-only tests and a dry-run pass on this host; no P106 hardware, so the record is not run and not a support claim |
| Review/audit follow-up: version, config and link consistency | this worktree | `freebuff/tandem-cpu-recovery` | `README.md`, `WHY.md`, `SECURITY.md`, `docker/Dockerfile`, `docs/CARDS.md`, `docs/CONFIGURATION.md`, `docs/benchmarks/2026-11-10-i7-8086k-windows-cpu/README.md`, `kindi/tokenizer.py`, `tests/test_version_consistency.py`, `tests/test_config_example.py`, `tests/test_tokenizer.py`, `tests/test_docs_links.py`, `AGENTS.md` | done; current-facing 0.4.0 tags/archive names, WHY/SECURITY version statements and the stale card tag fixed; `docs/CONFIGURATION.md` now names every default key; missing BPE pieces raise instead of silently encoding as token 0; local Markdown links are checked by a new test. Evidence: 14 focused tests pass, `config_reference --check` and `support_matrix --check` exit 0, `compileall` and `git diff --check` pass. The new `Buffy_ReviewAudit_202611111200.txt` is only an executive summary (1,012 bytes, no findings); the concrete older findings were re-triaged. The installer source tarball is still not hash-pinned; changing it requires a new published installer, so it is left for the release owner |
| AVX feature preflight + installer refresh | this worktree | `freebuff/tandem-cpu-recovery` | `install.sh`, `tests/test_installer_cpu.py`, `docs/INSTALL.md`, `docs/RELEASE-CHECKLIST.md`, `docs/CPU-SUPPORT.md`, `AGENTS.md` | done; CPU setups require AVX; `--gpu cpu` and a machine with no usable GPU fail early, while GPU installs remain available. 4 fixture tests pass; 33 installer tests and 254 full tests (7 skipped) pass with `bash -n`, LF ShellCheck, `compileall`, generated config/support checks and `git diff --check`. Published LF artifact 65,112 bytes, sha256 `43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99`; the release `install.sh` and `install.sh.sha256` assets were refreshed and the public URLs and release notes verified. Previous pair retained as `install.sh.pre-avx-preflight-1794385057` and `install.sh.sha256.pre-avx-preflight-1794385057`; no native Linux dry run |
| Heterogeneous two-card pairing | this worktree | `freebuff/tandem-cpu-recovery` | `install.sh`, `tests/test_installer_cards.py`, `docs/CARD-SUPPORT.md`, `docs/CARDS.md`, `docs/INSTALL.md`, `docs/P106.md`, `AGENTS.md` | done; `pair_ok()` pairs any two cards from the supported pair classes (maxwell, pascal8, turing, tc, p100), including mixed generations; mixed pairs use plain `k1200x2`, a pre-6.0 + 6.0-or-newer pair adds `KINDI_G5=0`, and Kepler stays single-card. 19 card tests and 257 full tests (7 skipped) pass; generated config/support checks, `compileall`, `bash -n` and `git diff --check` pass. ShellCheck reports 5 warnings at unchanged lines 185, 215, 221, 294 and 594; the changed pairing block is clean. No mixed-card GPU run, no P106/P4000/M4000/RTX 4000 hardware, and the installer artifact was not rebuilt or republished |

| Gemma 4 host prompt cache byte budget + telemetry | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/gemma4_cache.py`, `kindi/server.py`, `tests/test_gemma4_cache.py`, `docs/PROMPT-CACHE.md`, `AGENTS.md` | done; `G4Cache` now enforces the class-level 1 GiB host byte budget (`DEFAULT_BUDGET_BYTES`) after every checkpoint/park, evicts oldest parked conversations then oldest active checkpoints, refuses an individually oversized snapshot, and records `evictions`/`evicted_bytes`/`checkpoints_dropped`/`parked_dropped`/`peak_bytes` in `metrics()`; `/metrics` exposes the Gemma 4 budget/usage/eviction counters. Evidence: 8 focused CPU-only tests pass (bounded active checkpoints, oldest-first parked eviction, room-making before parking, oversized snapshots, cache-off release, zero checkpoint limit, checkpoint-count telemetry and no active/parked double-count), full suite 386 passed/19 skipped, `config_reference --check`, `support_matrix --check`, `py_compile` and `git -c core.whitespace=cr-at-eol diff --check` pass. No real Gemma 4 model/GPU run; `docs/CONFIGURATION-REFERENCE.md` was not touched because the main-worktree copy is dirty |
| Proportional VRAM-aware two-card split | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/tp.py`, `kindi/cache.py`, `tests/test_plan_engine.py`, `docs/MIXED-CARD-ROADMAP.md`, `docs/ROADMAP.md`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `AGENTS.md` | done; opt-in `KINDI_TP_SPLIT=auto` derives a complete per-card layout from free VRAM after a 512 MiB reserve (10-point deadband; `equal` remains the default): GDN q/k groups with tiled v heads, attention q/KV heads, FFN rows and columns on the same whole-HB interval, and 16-row-aligned output-head rows. Local widths stay HB-aligned; too-small components fail before load. The layout is in the `g5` cache key and saved-opening fingerprint. Tests: 39 focused plan-engine tests (27B-aligned 8/8 GDN (24/24 v heads) and 12/12 q, 11264/6144 FFN, 165552/82768 head; a synthetic rep-2 geometry exercises GDN 12/4; Shard offsets; invalid alignment; fingerprint), full suite 394 passed/19 skipped, generated config/support checks and py_compile. At that time `T:/kindi` had uncommitted `kindi/tp.py`, `kindi/spec.py` and `tests/test_plan_engine.py` (FD/g5-cache fixes); merge by hand, do not overwrite those copies (now merged; `T:/kindi` clean). No mixed-card GPU exists here, so no two-card token-exactness or peak-VRAM record |
| Remaining review findings: health details, installer port/verify robustness | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/server.py`, `tests/test_server.py`, `install.sh`, `tests/test_installer_source.py`, `docs/AGENTS.md`, `README.md`, `docs/INSTALL.md`, `docs/CONFIGURATION.md`, `docs/TROUBLESHOOTING.md`, `docker/Dockerfile`, `docker/Dockerfile.rocm`, `docker/Dockerfile.rocm7`, `docs/DOCKER.md`, `.gitattributes`, `AGENTS.md` | done; `/health` now always returns only `ok`, `kindi`, `commit`, `model` and `ctx`, and hides queue/GPU/uptime/usage details unless the API key is supplied (the installer still gets its five liveness fields); `install.sh` no longer assumes `ss` (localhost `/dev/tcp` fallback, timeout treated as occupied) or a `tokens/s over the run` line, since `pipefail` would otherwise abort; all three Dockerfiles have a `/health` `HEALTHCHECK`; `docs/AGENTS.md`, README, CONFIGURATION, TROUBLESHOOTING and INSTALL document the contract and the installer-test vintage; `.gitattributes` keeps shell scripts LF. Evidence: 398 passed, 19 skipped; `config_reference --check`, `support_matrix --check`, `py_compile`, `bash -n install.sh` and `git diff --check` pass. Design/feature and hardware-verification findings (GPU CI, AVX-512/AMX, FP8 dequantization, Windows CI, etc.) remain scheduled, not silently claimed as fixed. No Docker build or real keyed-server run was possible here |
| 2B forced-tier completion | this worktree | `freebuff/tandem-cpu-recovery` | `docs/CPU-SUPPORT.md`, `docs/ROADMAP.md`, `bench-logs/cpu-isa-2b/**`, `AGENTS.md` | done; the remaining forced `avx2-fma` and `avx2` 2B runs are 4/4 token-exact at 31.9 and 29.1 tok/s on the i7-8086K (native Windows, clang 18.1.8, 6 threads, ctx 1024, fresh dedicated cache); all five tiers now have 2B evidence (43.6/43.9/31.9/29.1/8.9 tok/s for `avx2-fma-f16c`/`avx2-f16c-no-fma`/`avx2-fma`/`avx2`/`avx-only`). Full logs and pack hash in `bench-logs/cpu-isa-2b/`; still forced tiers on an AVX2 host, not native no-F16C hardware |
| Server per-client rate limit and CORS decision | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/server.py`, `kindi/config.py`, `kindi.conf.example`, `tests/test_server.py`, `tests/test_config.py`, `docs/AGENTS.md`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/ROADMAP.md`, `Buffy_ProblemsOnly_202611111200.txt`, `AGENTS.md` | done; `KINDI_RATE_LIMIT_PER_MIN` (config `rate_limit_per_min`, default 0 = off) is an in-process fixed-window limit per source address for generation requests, with 429, `Retry-After`, health/metrics/model-listing exemptions, `/metrics` counters and a bounded client map; `docs/AGENTS.md` states that no CORS headers are sent and why, so browser apps need a same-origin proxy. Evidence: 401 passed, 19 skipped; new limiter, endpoint, config and generated-reference tests pass; `config_reference --check`, `py_compile`, `bash -n install.sh` and `git diff --check` pass. Proxy deployments share one source address; no distributed quota is claimed |
| Tandem throughput benchmark | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/build_cross_mode.py`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/**`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `tests/test_cross_mode_benchmarks.py`, `AGENTS.md` | done; the cross-mode builder now consumes complete tandem CSV rows and leaves incomplete models `n/a`; a real 2B `cpu-first` bench at 24,576 context records read 80.9 and generation 65.7/71.3/55.5/43.4/35.8 tok/s with 6411 MiB free after load; 4B/9B remain `n/a` and no peak or GPU-beat claim is made. Evidence: full suite 460 passed, 10 skipped; `project_memory --check`, cross-mode/config/support `--check`, `compileall` and `git diff --check` pass |
| Address 2026-11-12 review findings | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/server.py`, `kindi/cache.py`, `tests/test_server.py`, `tests/test_cache_decisions.py`, `docs/CONFIGURATION.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/BENCHMARKS.md`, `docs/CARD-SUPPORT.md`, `docs/CARDS.md`, `docs/P106.md`, `docs/INTEL-ARC.md`, `docs/MOE-TANDEM.md`, `docs/GEMMA4-26B-MOE.md`, `docs/ROADMAP.md`, `docs/RELEASE-CHECKLIST.md`, `docs/TANDEM-STATUS.md`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/README.md`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/bench-2b-cpu-first-24k.log`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; S1 non-loopback keyless `/health` now returns only liveness fields, B7 opening restore validates fingerprint/count/shape/dtype, B8 saved-opening save failures are counted in `/metrics`, B9 documents process-cwd-relative `pack`/`state_dir`, B2 tandem/P106/Arc/roadmap staleness is corrected, B4 MoE pages carry explicit not-a-backend banners, and the B1 installer pin is documented as a release-owner publication step. Evidence: 456 passed, 19 skipped; `project_memory`, cross-mode/config/support `--check`, `compileall` and `git diff --check` pass |
| Permanent project memory and roadmap/git reconciliation | this worktree | `freebuff/tandem-cpu-recovery` | `.agents/skills/project-memory/SKILL.md`, `docs/development/README.md`, `tests/test_project_memory.py`, `AGENTS.md` | done; the skill now has a context-compaction reset recipe, roadmap-vs-git reconciliation step, deleted-requirement recovery with `git show`, and a rule that roadmap items need requirement evidence; `docs/development/README.md` mirrors the reconciliation gate and `AGENTS.md` points agents to it. Evidence: 456 passed, 19 skipped; `project_memory --check`, docs links, config/support `--check`, `compileall`, `bash -n install.sh` and `git diff --check` pass |
| Current vs Kindi 0.5 report refresh (round 2) | this worktree | `freebuff/tandem-cpu-recovery` | `CURRENT-VS-KINDI-0.5.md`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/bench-4b-cpu-first-24k.log`, `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/bench-9b-cpu-first-24k.log`, `AGENTS.md` | done; report refreshed to `082c5da` (301 files, 245 changed vs stock, 2B/4B/9B tandem rows, owned-slice CPU loads 17.2/32.4 s), and the force-added 4B/9B bench logs sanitized after they failed `test_no_secrets` with a local home path. Evidence: full suite 462 passed/19 skipped at 27.15% line-and-branch coverage, `project_memory --check`, docs/project-memory 6 passed, `git diff --check` pass |
| Buffy audit remediation (202611122247) | this worktree | `freebuff/tandem-cpu-recovery` | `tests/requirements-test.txt`, `.github/workflows/ci.yml`, `docs/CI.md`, `tests/test_ci_config.py`, `tests/static/run_static_analysis.py`, `AGENTS.md` | done; verified each Buffy_ReviewAudit_202611122247 finding against HEAD 71df9ed. Fixed: D6 test deps now bounded (pytest>=7,<10, fastapi>=0.100,<1, pydantic>=2,<3, installed suite versions verified), D5 static analysis wired into CI as a `static-analysis` job (cppcheck + clang-tidy-18 on ubuntu-latest; harness now stubs cupy when unavailable so TU building works GPU-free — verified locally: Turing TU builds 216,472 bytes, tools skip exit 0), AGENTS Snapshot version line marked historical. Verified not-a-problem / stale report claims: M5 proportional VRAM split already exists as opt-in `KINDI_TP_SPLIT=auto` (kindi/tp.py:848, docs correctly state no mixed-card hardware run); INSTALL.md already fork-hosted (atimelessrage, sha 43766d8b); D7 broad-except sites carry noqa+rationale. Blocked, unchanged: push to myfork/main + stale public branch (no push without authorization), self-hosted GPU runner registration (needs runner setup), AVX-512/VNNI (multi-day kernel work, roadmap). Evidence: tests/test_ci_config+secret_scan+cross_mode 10 passed, py_compile ok, static script --only turing exit 0
| 27B tandem saved-opening + joint checkpoint at 4K/8K | this worktree | `freebuff/tandem-cpu-recovery` | `bench-logs/tandem-cache/**`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `docs/ROADMAP.md`, `AGENTS.md` | in progress; 27B (local 27B pack, 16 CPU layers) in-process joint-checkpoint exactness PASS at 512/4K/8K in both placements (extension 320/4080/8176, branch 256/3072/6144 cached, exact; context-27b-4k.json, context-27b-8k.json, results-27b-512.json) and saved-opening reload PASS at 512 (256/256 both) and 4K (3072 both; cpu-first first attempt hit C: disk-full Errno 28, rerun on D: state root passed) — harness gained --state-root and checkpoint-every/budget flags. Remaining: saved-opening at 8K and checkpoint-only cost/VRAM, then docs update |
| Address audit findings round 2 | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/build_cross_mode.py`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `tests/test_cross_mode_benchmarks.py`, `docs/CPU-SUPPORT.md`, `docs/P106.md`, `docs/INTEL-ARC.md`, `AGENTS.md`, `docs/TANDEM-STATUS.md` | done; verified and fixed the real findings: builder list-vs-tuple (`measured_tandem == list(MODELS)`, CROSS-MODE intro regenerated + regression test in `tests/test_cross_mode_benchmarks.py`), AGENTS worktree table bumped to 1f2c682 and stale "dirty" lease claims marked historical, CPU-SUPPORT/P106/ARC runtime-dispatch and tandem lines corrected, scope note added above the dated Snapshot/Handoff history. Installer pin (B1) and GPU-CI/perf-gate items verified as documented-blocked (release owner / self-hosted runner, docs/CI.md:28,64,115) — no code change possible here. Evidence: focused tests 9 passed (`test_cross_mode_benchmarks`, `test_docs_links`, `test_project_memory`, `test_no_secrets`), `project_memory --check`, `build_cross_mode --check` pass |
| Tandem CROSS-MODE throughput completion | this worktree | `freebuff/tandem-cpu-recovery` | `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/**`, `docs/benchmarks/CROSS-MODE.md`, `docs/benchmarks/CROSS-MODE.csv`, `docs/BENCHMARKS.md`, `docs/TANDEM-STATUS.md`, `docs/TANDEM-RUNTIME.md`, `docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; real 4B/9B `cpu-first` bench runs at 24,576 context recorded in `runs.csv`, the matrix regenerated with no tandem `n/a`, and dependent docs updated. 4B read 62.5, generation 52.1/51.4/33.5/24.8/20.0; 9B read 39.9, generation 28.5/35.8/28.2/20.1/16.0 tok/s. Tandem still does not beat GPU-only; the larger-than-VRAM throughput-plus-peak gate remains open. Evidence: `build_cross_mode.py --check`, cross-mode/docs/project-memory tests 8 passed, `project_memory --check`, `git diff --check` pass |
| 27B tandem CPU load reduction | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/cpu.py`, `kindi/tandem.py`, `tests/test_tandem.py`, `tests/test_cpu_build.py`, `docs/TANDEM-RUNTIME.md`, `docs/TANDEM-STATUS.md`, `CURRENT-VS-KINDI-0.5.md`, `AGENTS.md` | done; `cpu.Engine` accepts `layers`/`embed`/`head` and converts only owned ternary weights, and `TandemEngine` passes its CPU range, cutting the 27B CPU conversion from 64 layers to the owned 16 plus the owned embed/head; the fix also exposed and corrected `_TandemCacheView` owner counters that incremented for GPU-owned layers. Docs record the old 365 s as the full-load measurement and the new slice load as unmeasured. Evidence: `tests/test_tandem.py` 12 passed, CPU build/feature/thread/smoke/sampling 48 passed/11 skipped, cache/docs/project-memory 31 passed, `project_memory --check` and `git diff --check` pass |
| Current vs Kindi 0.5 report refresh | this worktree | `freebuff/tandem-cpu-recovery` | `CURRENT-VS-KINDI-0.5.md`, `.gitignore`, `AGENTS.md` | done; report now records `main` merged at `2d56478`, 299 tracked files, 47 test modules, 462 passed/19 skipped with 27.20% line-and-branch coverage, the recorded 2B tandem throughput row, the 4B tc 3-exact-plus-tie state, and the remaining installer/version release blockers; `.coverage`/`coverage.json` are ignored so coverage runs no longer dirty the tree. Evidence: `project_memory --check`, `test_project_memory`/`test_docs_links` 6 passed, full suite 462 passed/19 skipped, `git diff --check` pass |
| Tandem cache-loss pick coverage | this worktree | `freebuff/tandem-cpu-recovery` | `tests/test_tandem.py`, `AGENTS.md` | done; focused test proves the tandem runner recomputes a fresh next-token pick after its prompt checkpoint cache is deleted (committed `2d56478`; `tests/test_tandem.py` 12 passed) |
| Requirement and structural coverage improvement | this worktree | `freebuff/tandem-cpu-recovery` | `tests/test_requirement_coverage.py`, `tests/test_server.py`, `tests/test_cache_decisions.py`, `tests/test_config.py`, `tests/test_sampling.py`, `tests/test_cpu_features.py`, `.github/workflows/ci.yml`, `docs/CI.md`, `docs/development/README.md`, `AGENTS.md` | done; requirement-based tests added (`tests/test_requirement_coverage.py`), CI now measures branch coverage, and open review findings were verified with focused tests (committed `70a6f76`/`dfb36f9`; full suite 462 passed, 19 skipped, 27.20% line-and-branch coverage) |
| External MuseEmber review remediation | this worktree | `freebuff/tandem-cpu-recovery` | `kindi/cpu.py`, `tests/test_cpu_build.py`, `docs/TROUBLESHOOTING.md`, `docs/AGENTS.md`, `docs/CARDS.md`, `docs/CARD-SUPPORT.md`, `docs/P106.md`, `docs/CONFIGURATION-REFERENCE.md`, `docs/RELEASE-CHECKLIST.md`, `docs/development/README.md`, `AGENTS.md` | done; exact CPU compiler command is logged, troubleshooting now covers health/auth and CPU-build recovery (`KINDI_CPU_CACHE`, `KINDI_CC`, `KINDI_LLVM`, `KINDI_CPU_ISA`), P106 top status names the missing P106 tandem result, card targets are labeled unverified, release checklist now requires a history secret scan, and development README lists authoritative sources. Evidence: full suite 456 passed, 19 skipped; `project_memory`, cross-mode/config/support `--check`, `compileall`, `bash -n install.sh` and `git diff --check` pass |

That main-worktree lease covers its **whole dirty set**, not only the docs and installer. It also
has changes in `GOALS.md`, `IMPLEMENTATION_PLAN.md`, `setup.py`, `kindi/config.py`,
`kindi/convert.py`, `kindi/cpu.py`, `kindi/engine.py`, `kindi/hostram.py`, `kindi/kcpu.c`,
`kindi/layout.py`, `kindi/nanbeige_cpu.py`, `kindi/nbq_cpu.c`, `kindi/tp.py`, `tests/README.md`,
new tests, and its own `AGENT_COLLABORATION_PROTOCOL.md` (a review-findings list, not a worktree
lease). `git -C T:/kindi status --short` is the authoritative lease; treat every path there as
main-owned except `T:/kindi/AGENTS.md` (this bridge note). Re-run the status command; this snapshot
is not a lock.

> **Scope note (added 2026-11-12):** the Snapshot/Handoff sections below are dated history.
> For the current HEAD, worktree state and open leases, see the worktree table and the newest
> lease rows above; do not treat historic claims (e.g. "main merge pending") as current.

## Snapshot (2026-11-08)

- (Historical, 2026-11-08) Version 0.4.0 (`kindi/__init__.py`), model id `bonsai2-2b`, AGPL-3.0-only plus a commercial
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
- Mixed-card pairing/asymmetric TP roadmap (2026-11-11): reviewed the preserved WIP against the
  current installer and engine. WIP `install.sh` is **0 bytes**, so its `mixedx2`, `KINDI_TP_SPLIT`,
  per-card TC and per-card reserve claims have no installer implementation. WIP `kindi/tp.py` only
  proportionally changes FFN/head and admits the remaining splits need kernel work; WIP `spec.py`
  removes the Turing `W=2` guard without a test. Current `pair_ok()` offers mixed pairs, but
  `profile_of()` uses equal-split `k1200x2`/`tc2`, `KINDI_G5=0` is a global 2-bit fallback, and no
  mixed-card GPU run exists. Corrected phased proposal: `docs/MIXED-CARD-ROADMAP.md`; no code merged.
- Nanbeige 33B CPU+GPU proposal review (2026-11-11): triaged the shared unverified proposal
  against the actual Nanbeige host-layer path. The K6000 baseline is 3.56 tok/s at 8K, with 13
  host layers copied into two slots and 55% of their FFN rows on the CPU; it is not a sequential
  CPU-only phase. Current `tandem.py` remains Bonsai-only, the 2B drafter is GPU-side and the
  Nanbeige path has no CPU drafter or CPU K/V. The proposal's 80/20 resident slice is a plausible
  next architecture, but the first gate is a `KINDI_NB_CPUF`/slot profile. Corrected review:
  `docs/TANDEM-PROPOSAL-REVIEW.md`; no code changed.
- 27B tandem gate attempt (2026-11-11): full-GPU plan baseline at 8K passed 4/4 with a sampled
  **7,238 MiB** peak (6,101 over idle, 950 MiB left); load 20.5 s. The tandem `cpu-first` CPU engine
  loaded in 435.4 s and then the GPU owner failed with `unsupported grouped head dimensions`: the
  27B has 24 q / 4 KV heads (6:1) and plain `Engine` requires 4:1. A direct probe of both placement
  GPU ranges failed identically. No tandem peak or token result exists; 1,265 MiB was desktop plus
  CPU load, not a split result. The gate is blocked until a partial plan-engine owner or 6:1 plain
  GPU owner exists. Evidence: `bench-logs/tandem-27b-gate/RESULTS.md`.
- 27B tandem grouped-query support (2026-11-12): the plain GPU `Engine` now validates grouped-query ratios generically (`Engine._gqa_groups`: G≤8, head size 256 except the untouched 4:1 fast path) and `Engine._attn_core` launches the existing generic-GQA kernels (`k_attn_sg`, `k_attn_vgn`, `k_attn_combn`) for the 27B's 6:1 attention; no new CUDA kernels. `tests/test_gqa_heads.py` (8 CPU-only tests) pins ratios, rejections and kernel arguments; the suite is 444 passed, 19 skipped. `kindi verify` after the refactor is ALL PASS on the 2B (4/4, 153.7 tok/s), 4B (4/4, 108.9) and 9B (4/4, 74.6), so the 4:1 path did not regress. The tandem 27B gate is exact 4/4 (40/40 each) in both placements, peaks 7,005 MiB (`cpu-first`, load 385.6 s) and 7,017 MiB (`gpu-first`, load 413.3 s) vs the 7,238 MiB full-GPU baseline at 8,192 context (`reduced: false`). `run_gate.py --probe-gpu-owner` exits 0 when both 27B GPU ranges construct. At that date CROSS-MODE tandem throughput rows were `n/a`; the 2B row was later recorded on 2026-11-12 and 4B/9B remain `n/a` (see `docs/TANDEM-STATUS.md`). `docs/CONFIGURATION-REFERENCE.md` was regenerated (`config_reference --check` passes). No commit, push or merge.
- MoE expert-cache design + model-free prototype (2026-11-11): `docs/MOE-TANDEM.md` defines the
  copy-only lane and the future backend contract. `kindi/expert_cache.py` now implements the slot
  state machine: `(layer, expert)` keys, token-scoped pins, LRU over unpinned READY cache slots,
  transient fallback when the cache is pinned, failed-copy discard, and JSON-safe metrics.
  `tests/test_moe_cache.py` covers cold/hit transitions, LRU, pin rejection, transient reuse safety,
  failed copies and metrics without a model or GPU. Gate: 11 tests pass; `py_compile` passes. This
  is not a backend: no tensor copy, VRAM allocation, engine wiring, converter, pack, catalog entry
  or support claim; the Raon-1 mini goldens and MoE backend remain open. A new property harness
  (`tests/test_moe_cache_properties.py`) runs 32 seeded traces x 90 mixed route/acquire/copy/use/release
  operations and checks capacity, pin ownership, slot states, atomic rejects and metric consistency
  after every step; all pass. The generic cache-to-copy seam is defined in
  `docs/EXPERT-CACHE-COPY-INTERFACE.md`: immutable copy tickets/jobs/completions, a bounded copy
  queue, stale-generation fencing, and a strict rule that only the scheduler calls `mark_ready`,
  `fail`, or `release`; the copy worker has no cache handle and cannot choose an eviction victim.
- Gemma 4 26B-A4B cache adaptation (2026-11-11): `docs/GEMMA4-26B-MOE.md` records the local
  Q4_K_M contract and the cache changes: 30 layers, 2,816 width, 128 experts with top-8 plus a
  dense shared MLP, 704 expert width, fused `ffn_gate_up_exps`, 3,717,120-4,336,640 B slots and
  14,479,917,056 B of routed weights. Per-layer plans pin eight experts and release after combine;
  the fixed slot is 4,336,640 B and eight slots need 34,693,120 B. The official combine is in
  ascending expert-id order, not router order, and the dense MLP runs in parallel. The 5.19 GiB
  KV at 256K does not fit 8 GB with the dense core and reserve; the exact reference needs BF16 or a same-Q4 llama.cpp run, which
  is not local. `kindi/gemma4.py` remains E4B-only and rejects Q5_1; no backend, cache integration,
  reference golden or support claim was added. The execution contract in
  `docs/GEMMA4-26B-COPY-COMBINE.md` (2026-11-12) fixes the eight cache slots plus one transient
  slot, the two-image pinned staging ring, one H2D/event per complete expert, ascending-expert-id
  combine independent of slot/copy order, hard copy/kernel failure handling, and the required
  seven-eviction-plus-transient trace; it is design-only.
- Tandem joint prompt cache (2026-11-11): `TandemRunner` subclasses `cache.Runner` over
  `TandemEngine.cache_view()`, a virtual engine exposing both owners' M/conv arrays and the K/V
  arrays of every attention layer. Checkpoints therefore
  cover both owners' M/conv; parked conversations and saved openings also cover both owners' K/V. Saved-opening fingerprints include placement, layer ranges, CPU ISA and
  thread count, and `TandemEngine.cache_fingerprint()` hashes `tandem.py`. The view is host-only (no
  VRAM checkpoint slot). Evidence: `tests/test_tandem.py` 11 passed, `tests/test_cache_decisions.py`
  8 passed, and the full `python -m pytest -q tests/` is 362 passed, 19 skipped; docs link and
  generated-reference checks pass. A real 2B cache check now passes in both placements.
  The 320-token extension and 256-token branch matched cache-off token-for-token (clang 18.1.8/libomp,
  avx2-fma-f16c, 6 threads, 512 ctx, RTX 4060 Ti); evidence in `bench-logs/tandem-cache/RESULTS.md`;
  the saved-opening and server checks are now recorded (2026-11-12): a new runner loaded the
  256-token opening from disk, and a real HTTP server saved/restarted/loaded a 344-token opening
  with identical replies and 344 cached versus 0 cache-off tokens in both placements. The run
  found and fixed `Openings.load` using a plain NumPy-to-CuPy K/V assignment (CuPy treated it as a
  scalar fill); it now uses `_to_dev`. A prepare-only run measured 320/325 and 256/261 tokens reused
  and a 19.27 MiB joint recurrent checkpoint, with 8.7-8.9 ms median save and 3.3-3.5 ms median
  restore (`bench-logs/tandem-cache/PREFIX-RESULTS.md`). A checkpoint-only follow-up measured joint save/restore at 4K/8K: cpu-first 14.734/3.675 ms and 10.873/3.322 ms, gpu-first 9.781/3.296 ms and 10.223/3.416 ms (branch restore medians; K/V excluded). `kindi/server.py` passes the config and
  pack id into `TandemRunner`; `docs/TANDEM-RUNTIME.md` and `docs/PROMPT-CACHE.md` describe the
  contract.
- 2B tandem saved-opening + HTTP cache check (2026-11-12): both placements saved a 256-token
  opening and a fresh `TandemRunner` loaded it from disk, matching cache-off token-for-token. A real
  `kindi serve` process then warmed a 362-token request, wrote a 344-token opening, restarted over
  the same state directory, and reported 344 cached tokens versus 0 on a fresh cache-off server;
  the replies were identical. The harness also passed at 4K and 8K context with 1,024/2,048-token
  checkpoints: extension reuse 4,080/8,176 and branch restore 3,072/6,144 tokens, all exact. The 9B
  pack then passed the 512-context check in both placements with 6 CPU layers: extension 320,
  branch and saved opening 256, all exact; CPU load 105-108 s and GPU load 5.8-6.0 s. The 9B HTTP
  server save/restart/reload path then passed in both placements with a 344-token opening (344
  cached after restart, 0 cache-off, exact replies). The run
  exposed a CuPy-only restore bug in `Openings.load`: a plain NumPy K/V assignment was treated as a
  scalar fill. It now uses `_to_dev`; 19 focused tandem/cache tests pass. Commands and logs:
  `bench-logs/tandem-cache/run_check.py`, `run_server_check.py`, `RESULTS.md`, `RESULTS-9B.md`,
  `LONG-CONTEXT-RESULTS.md`, `server-results-9b.json`, `server-results-9b.md` and
  `server-results-9b.log`. A checkpoint-only 9B/512 run measured 50.25 MiB joint recurrent state,
  15.9/17.4 ms median save and 10.5-11.1 ms median restore; device-wide sampled peak was
  3,063/3,065 MiB (1.8/1.7 GiB over idle), with K/V 16 MiB excluded. The 9B pack also passed 4,096/8,192-context exactness in both placements:
  extension 4,080/8,176 and branch 3,072/6,144 cached, cache-off 0, exact. No commit or push.
- BF16/FP8 real-model validation check (2026-11-11): blocked, no token-exactness run. `D:\ollamaModels`
  contains `gemma4:26b` (Ollama blob `a8bdd07a49b...`, Q4_K_M, 30 layers, width 2816, MoE expert
  tensors including type 7/Q5_1; no type 30/43/44 language tensors; `kindi.gemma4.inspect_model`
  rejects it on the unsupported type 7) and the T: store contains Qwen3.6-35B-A3B UD-Q4_K_XL
  (`qwen35moe`, not Gemma 4). No llama.cpp CLI exists in the workspace or on PATH, and there is no
  `reference-*.json` for a Gemma 4 E4B file; `reference-nemotron-bolt-e4b.json` is a different
  checkpoint. The engine is E4B-only (42 layers, 2560 width) and the 26B weighs 15.9 GiB, more than
  the 8 GB card. A future run needs an E4B GGUF whose language matrices are BF16/E4M3/E5M2 and a
  pinned llama.cpp greedy reference for that exact file. Evidence and the blockers are recorded in
  `docs/FP8-BF16.md` (Real-model check). `docs/CARD-SUPPORT.md` remains blocked by the dirty main copy.
- Native FP8 tensor-core prompt GEMM (2026-11-11): `kindi/gemma4_fp8.cu` adds E4M3 and E5M2
  `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` kernels (`.e5m2.e5m2` for type 44) with
  fp32 accumulators, a 64-row x 64-token block and 32-byte shared-memory K steps. `Ops` loads the
  module when CUDA compute capability is >= 8.9; `linear_batch` quantizes activations per token
  (`max|x|/448` E4M3, `max|x|/57344` E5M2, hardware `cvt.rn.satfinite`) and scales the epilogue.
  This replaces the dequantize-to-f32 + cuBLAS prompt path for types 43/44 on Ada/Hopper/Blackwell;
  pre-Ada, HIP, widths not divisible by 32, a failed module load and `KINDI_G4_FP8=0` keep the old
  path. Decode is unchanged. It is W8A8, so it is not bit-exact with the f32 path. Evidence: 20
  focused tests pass (`tests/test_gemma4_fp8_mma.py` and `test_gemma4_fp8.py`) and the full suite is
  367 passed, 10 skipped; synthetic RTX 4060 Ti timings are 0.11 ms/15.1 TFLOPS at 2,560x2,560x128,
  0.24 ms/28.2 at 512 tokens, 0.23 ms/29.0 at 10,240x2,560x128 and 0.76 ms/35.5 at 512 tokens
  (`bench-logs/fp8-mma-ada.log`). No real FP8/BF16
  Gemma 4 GGUF, so no end-to-end token agreement or model throughput; Blackwell is enabled by
  capability but unverified, and no speedup over the cuBLAS fallback is claimed because cuBLASLt is
  absent on this host. The sm_89 PTX is the portable form, so Blackwell's driver JITs it; no native
  FP4/FP6 or BF16 MMA was added. `docs/CARD-SUPPORT.md` still says there is no native FP8
  tensor-core path and was not edited: `T:/kindi` was dirty at that time (uncommitted card-verify changes),
  so its owner must update that sentence when the file is handed off.
- Multi-ISA CPU dispatch (2026-11-11): `kindi/cpu.py` defines five ISA tiers and `select_isa()`
  picks the best one the host reports (`/proc/cpuinfo`, or the compiler's `-march=native` macros on
  Windows), with `KINDI_CPU_ISA` as the override; an explicitly unsupported forced tier exits with
  an actionable error. Each tier builds with explicit `-mavx2/-mfma/-mf16c` flags into a cache key
  that includes the tier, compiler and OS, so one cache can serve different CPUs. `kindi/kcpu.c`
  exports `kc_isa()` and `kc_isa_name()`, and `_lib()` refuses a library whose reported tier does
  not match the selection. `kindi info` prints the detected path and the build; the engine log
  includes the tier. Evidence: 37 build/feature tests with clang 18.1.8; 2B `kindi verify` 4/4
  token-exact at 43.6 (`avx2-fma-f16c`), 43.9 (`avx2-f16c-no-fma`) and 8.9 (`avx-only`) tok/s
  (`bench-logs/cpu-dispatch-*.log`); full suite 355 passed, 8 skipped; `compileall`,
  `config_reference --check`, `support_matrix --check` and `git diff --check` pass. The `avx2-fma`
  and `avx2` tiers are compile-tested but not run end to end, and no native Sandy Bridge,
  no-F16C or 4B/9B per-tier run exists. `docs/CARDS.md` was not edited because `T:/kindi` was dirty at that time.
- Review follow-ups: structured logging + dependency updates (2026-11-11): `kindi/server.py` has an
  opt-in JSON access log (`log_json = 1`, `KINDI_LOG_JSON=1`): one line per inference request with
  `request_id` (also `X-Request-ID`), method, path, status, `duration_ms`, `queue_wait_ms` (time
  waiting for the engine lock), `stream`, and on success prompt/cached/completion tokens, `gen_tok_s`
  and `finish`. Rejected, cancelled and failed requests are logged; `/health`, `/metrics` and
  `/v1/models` are not; headers and the API key never are. `kindi/config.py` validates 0/1 and the
  example/reference docs are regenerated. `.github/dependabot.yml` adds weekly reviewed PRs for the
  test requirements and the workflow actions; the actions are also pinned to the commit SHAs that
  `actions/checkout@v4`, `actions/setup-python@v5` and `actions/upload-artifact@v4` resolved to,
  closing review D6 and D8. This also closes the roadmap's structured
  logging follow-up. Gate: `pytest -q tests/test_server.py tests/test_config_example.py
  tests/test_config_reference.py` and `git diff --check` pass.
- Dory1.0-4B long-term looped-model goal (2026-11-11): `docs/DORY.md` records the released
  `config.json` (4 B, `model_type: dory`, 72 physical layers = 16 input + 40 shared recurrent + 16
  output, 27 sliding-window + 9 full-attention layers at a 3:1 ratio, window 1,024, GQA 16 query/4
  KV heads, `head_dim` 256, hidden 2,560, 131,072 vocab, 262,144 context, `n_recurrent_loops: 2`,
  loops 1-4), the nGPT L2-only normalization and per-channel residual blend, the loop-boundary
  recurrence `u = N(s + x); s = N(F(u))` and `32 + 40k` physical-layer applications, the KV envelope
  (36/56/96 KiB per token for k=1/2/4 per-loop; 8.8 GiB at 256K for one loop), the per-loop exact vs
  last-loop lossy cache modes, the D0-D7 phases, the provisional 20/12/8 tok/s gate at k=1/2/4, and
  an all-pending acceptance matrix. The official formats are BF16, MXFP8, NVFP4 and Q5_K_M GGUF
  (3,673,288,864 B, sha256 `96080be4...`), license OpenMDW 1.1, default sampling 1.0/0.95, reasoning
  blocks on. `docs/ROADMAP.md` has a Dory row in `## Long-term model goals`. No catalog entry,
  converter, engine, installer offer or support claim was added; the checkpoint is public and the
  Q5_K_M GGUF plus BF16 metadata are now downloaded and pinned in the D0 handoff below, while NVFP4
  was not downloaded; the llama.cpp/vLLM references are unreleased PRs pinned in the doc. Gate:
  `python -m pytest -q tests/test_docs_links.py` -> 1 passed; `git diff --check` -> 0. `docs/CARDS.md`
  was not edited because `T:/kindi` was dirty at that time.
- Dory1.0-4B D0 reference audit (2026-11-11): downloaded the Q5_K_M GGUF (3,673,288,864 B,
  sha256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`) and the BF16
  config/tokenizer metadata (five files; hashes in `bench-logs/dory-d0/bf16/SHA256SUMS`).
  `tools/dory_gguf_inventory.py` records 507 tensors (252 Q5_K, 36 Q6_K, 219 BF16) and 44 metadata
  keys; `dory.num_loops` is 4 in the GGUF while the BF16 config has 2, so k is explicit. Built
  llama.cpp PR 30306 tip `1c3cdd...` CPU-only (MSVC 14.51, Release, CUDA off) and pinned four
  prompts x k=1/2/4 at 32 greedy tokens in `bench-logs/dory-d0/goldens/goldens.json`; all 12 runs
  exit 0 and the k=2 repeat matches exactly. The earlier `98a...` pin was not used because later PR
  commits fix Dory CPU placement. No Kindi Dory code, catalog/installer entry or support claim was
  added; D1 is pending. Gate: `python -m pytest -q tests/test_docs_links.py` -> 1 passed;
  `python -m compileall -q tools/dory_gguf_inventory.py bench-logs/dory-d0/run_goldens.py` -> OK.
- Raon-1 mini R0 (2026-11-11): found the public KRAFTON preview and recorded the actual
  manifest instead of the old planning contract. Base `8c9b802...`: 7 shards, 32,702,145,088
  file bytes, 19,434 tensors / 32,699,649,368 tensor bytes, 28,783,218,602 header parameters
  (HF summary 28,783,218,402 excludes 200 scalars) and 4,317,319,082 active. Q4_K_M is
  19,686,434,240 B, SHA-256 `ba56ad4b...`, 731 GGUF tensors, 72 packed expert tensors.
  Geometry: 18 RDA + 7 MLA, layer 0 dense + 24 MoE, 128 experts top-6 + 1 shared, NoPE,
  mHC-4, 8,616,960 B per MXFP8 expert, Apache-2.0. Published compressed MLA cache is
  8,064 B/token; the HF reference expands to 179,200 B/token; the router tie and exact
  llama.cpp cache layout remain open. Evidence: `docs/RAON-1-MINI-R0.md`,
  `bench-logs/raon-r0/README.md`, `manifest-summary.json`, `source-pins.json`,
  `build_manifest.py` and `reference_greedy.py`. No base/Q4 load, no reference goldens,
  no backend/catalog/installer/support claim; Q4 is lossy and not the exactness reference.
  Gate: `python -m pytest -q tests/test_docs_links.py` -> 1 passed; `py_compile` for both
  scripts -> 0; `build_manifest.py` -> 19,434/731; `git diff --check` for tracked docs -> 0.
  `docs/CARDS.md` untouched; `T:/kindi/AGENTS.md` was not edited.
- AVX2/F16C no-FMA fallback (2026-11-11): `kindi/kcpu.c` now gates `rows8()` on `__AVX2__` and uses
  `_mm256_fmadd_ps` when `__FMA__` is set, otherwise a separate `_mm256_mul_ps` + `_mm256_add_ps`;
  F16C and the AVX-only scalar path are unchanged. `kindi/cpu.py` adds `cpu_tier()` /
  `CPU_TIER_LABELS`, includes the no-FMA state in the Windows cache key, and makes `kindi info` report
  the selected path; `install.sh` adds `cpu_flag()`/`cpu_isa_tier()` and labels the CPU setup with the
  tier. Tests: `test_cpu_build.py` compiles and loads AVX2/F16C no-FMA and AVX-only variants,
  `test_cpu_features.py` pins tier selection, `test_installer_cpu.py` the installer classification, and
  `test_cpu_avx_fallback.py` is the opt-in `KINDI_CPU_NO_FMA_PACKS` token-exact check. Forced clang
  18.1.8 `-avx2 -mno-fma -mf16c` build: 2B `kindi verify` 4/4 token-exact at 31.7 tok/s
  (`bench-logs/cpu-nofma-verify.log`); the opt-in test passed on the 2B pack. Full gate: 334 passed,
  8 skipped; `compileall`, `support_matrix --check`, `config_reference --check`, `bash -n`, LF
  ShellCheck and `git diff --check` pass. No native no-FMA hardware and no 4B/9B no-FMA run.
- Heterogeneous two-card pairing (2026-11-11): `pair_ok()` now accepts any two cards from `maxwell`,
  `pascal8`, `turing`, `tc` or `p100` with at least 3,800 MiB each; Kepler stays single-card. Mixed
  generations use the plain `k1200x2` engine, and `profile_of()` adds `KINDI_G5=0` when one card is
  below compute capability 6.0 and the other is 6.0 or newer, so both use the 2-bit path. Examples now routed:
  P106+P4000 (no G5 override), M4000+P4000 (`KINDI_G5=0`), RTX 4000+4060 Ti (plain, no TC). Evidence:
  `tests/test_installer_cards.py` 19 passed; full suite 257 passed, 7 skipped; `python -m
  kindi.config_reference --check`, `python -m kindi.support_matrix --check`, `python -m compileall -q
  kindi`, `bash -n install.sh` and `git diff --check` pass. ShellCheck on the worktree copy reports 5
  warnings at unchanged lines 185, 215, 221, 294 and 594; the changed pairing block is clean. No mixed-card
  GPU run and no P106/P4000/M4000/RTX 4000 hardware; the published installer remains the AVX refresh
  (`43766d8b...`) and was not republished. This supersedes the same-class-only statements in the earlier
  installer rows and the 2026-11-10 handoff.
- AVX feature preflight and installer refresh (2026-11-11): `install.sh` checks `avx` in
  `/proc/cpuinfo` (or `KINDI_CPUINFO`) before offering CPU setups. CPU setups are hidden without AVX;
  `--gpu cpu` and a machine with no usable GPU fail with an actionable message, while GPU installs remain
  available. `tests/test_installer_cpu.py` covers AVX, AVX2/FMA-only, a missing table and the guard.
  Gate: 33 installer tests and 254 full tests (7 skipped), `bash -n`, LF ShellCheck, `compileall`,
  generated config/support checks and `git diff --check` pass. Published LF artifact is 65,112 bytes,
  sha256 `43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99`; the release `install.sh` and `install.sh.sha256` assets were refreshed and the
  public URLs and release notes verified. The previous pair is retained as
  `install.sh.pre-avx-preflight-1794385057` and `install.sh.sha256.pre-avx-preflight-1794385057`.
  No native Linux dry run; the shell/fixture gates are the local evidence. `docs/INSTALL.md` and
  `docs/RELEASE-CHECKLIST.md` carry the published hash.
- Installer source checksum pin (2026-11-11): `install.sh` now downloads the default v0.5.0 source archive
to `$SRC.tar.gz.part`, compares its sha256 with `SOURCE_SHA256` (`26424ec08caf07a06a4f57547aa50d5b62a85ec8f7e8c231dcadc5c81519b1e2`), and extracts only on a match;
a custom `--source` URL is checked when `KINDI_SOURCE_SHA256` is set. `tests/test_installer_source.py`
executes the success and mismatch paths with fake `curl`/`sha256sum`/`tar` and guards against `curl | tar`.
Candidate installer hashes are LF `54f76308e7461a6799dcd1c7126f5a24699ba8c40aa5019f74b01bf500fe523a` and
CRLF `e49e1f42b57c190ca988d1627ae4f58e0f095db249dab25cc96efe2edef2dbb5`; the live `docs/INSTALL.md`
checksum was superseded by the 2026-11-11 AVX refresh, which published `43766d8b...`. Gate: 4 source tests, the
installer test files, the full 242-test CPU suite (3 skipped), `bash -n`, LF ShellCheck and
`git diff --check` pass. No artifact was published.

- Problems-only review triage (2026-11-11): `T:/kindi/Buffy_ProblemsOnly_202611111200.txt` is partly stale.
  Its critical CPU findings (#1-3, #6) name `kcpu_cpuid.c`, `kcpu_cpuid.py`, `kc_set_isa` and
  `rows16_avx512_vnni`, which are not in this tree; the only copies are in
  `bench-logs/main-backup-20261109-045352/`. The live `kindi/kcpu.c` is AVX2/FMA/F16C-only and now fails
  compilation with an actionable `#error`; the AVX-only fallback remains P1. `kindi.engine`, `kindi.tp`,
  `kindi.spec` and `kindi.chunk` import cleanly (no cycle). The installer card report is stale: the 4060 Ti
  and 5060 Ti classify as `tc`, P106/P106 and RTX 4000/RTX 4000 pair, and mixed classes such as
  P106/P4000 are rejected by the documented same-class policy and the existing fake-GPU tests. Server body caps,
  constant-time auth, bounded queue/429 and
  token-exact tests already exist. Added `tests/test_cpu_build.py` guard tests and a 5060 Ti status row in
  `docs/CARDS.md`. No FP8, tandem, AVX-512/VNNI/AMX, GPU CI or Windows CI work was added; those are feature
  or verification gaps, not regressions found in this tree.

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
  published installer and the v0.5.0 archive both hash to `b84df4fe...`; that was the pre-AVX artifact, superseded on 2026-11-11 by the AVX refresh (`43766d8b...`) now recorded in `docs/INSTALL.md`; no
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
  is 19.6x/25.7x/24.0x. The WSL2 CPU cells are archived. At that date the GPU+CPU tandem rows were present and `n/a` because there
  was no `TandemEngine`; the dense runtime now exists and the 2B tandem row is recorded, while 4B/9B
  remain `n/a` (see `docs/TANDEM-STATUS.md`). Gate: `python docs/benchmarks/build_cross_mode.py --check`, 203 tests, `compileall`, config/support
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

- 4B tensor-core FD fallback + streamed g5 cache repair (2026-11-11): `kindi/tp.py` disables
  `KINDI_FD` for the 4B's 2560-wide plan (the split decode attention diverges) and logs the fallback;
  `kindi/spec.py` rejects a streamed g5 cache whose key set predates the tied-head alias, so
  `Shard._load` rebuilds it. `tests/test_plan_engine.py` covers both. On the RTX 4060 Ti, 4B tc is
  3 exact + 1 accepted tie (token 24, gap 0.0157) at 164.6 tok/s; 2B tc is 4/4 and 27B tc is
  3 exact + 1 tie. Full CPU-enabled suite: 283 passed. `compileall`, `config_reference --check`,
  `support_matrix --check`, `cross_mode --check`, `bash -n install.sh`, the GRUB test and
  `git diff --check` pass. The 9B tc (Hadamard 4096) and 27B plain (`unsupported grouped head
  dimensions`) refusals remain unsupported combinations, not support claims.

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

## Review audit follow-up (2026-11-11, local)

Branch `freebuff/tandem-cpu-recovery`; merge commit `d5a0dc2` absorbs `main` at `fd54148`
(server boundary/startup validation, `logprobs`, Windows CI and GPU-smoke workflow) while keeping
this branch's AVX-only CPU fallback and mixed-generation installer routing.

- **D1 installer source pin.** `install.sh` now downloads the fork archive at
  `fd5414889e4acfd72b46bba3138a908b8e97c5be`
  (`https://github.com/atimelessrage/kindi/archive/${SOURCE_COMMIT}.tar.gz`) and checks
  `3ca1e79d371c87bea5471841db10b0509b90d3bd557512cf59fc83854b929a94` before extraction.
  `tests/test_installer_source.py` also rejects an `ahtx/kindi/archive` URL. This is fork `main`,
  not this branch; publish a new installer after this branch is merged and tagged.
- **D3 CPU cache identity.** `kindi/cpu.py` now writes `libkcpu-<key>.so` / `.dll`, keyed by OS,
  architecture, compiler choice, host CPU identity/flags and the compiler command line. A shared
  `KINDI_CPU_CACHE` can no longer reuse a `-march=native` binary from a different machine/toolchain.
  `tests/test_cpu_build.py` covers the key; `tests/test_cpu_smoke.py`, the AVX-fallback test and the
  CI smoke job use `_so_path()`.
- **D4 config-expressible server limits.** `max_body_mb` (16) and `queue_max` (8) joined
  `config.DEFAULTS`/`INT_KEYS`, validation, `kindi.conf.example`, `docs/CONFIGURATION.md` and the
  generated reference. `create_app` stores the body cap on `app.state` and passes `queue_max` to
  `Gate`; explicit config values beat the module import-time env fallback.
- **Roadmap.** The review's worthwhile feature ideas are planned, not implemented, in
  `docs/ROADMAP.md` under “Review follow-ups (2026-11-11)”: multi-ISA dispatch, tandem runtime,
  asymmetric TP, GPU CI runner, static/secret analysis, coverage/perf gates, Windows install docs,
  structured logging, AGENTS ownership/index trimming, and dependency/Actions pinning. The tandem
  design page now carries an explicit “design only” banner.

Evidence: `python -m pytest tests/ -q` -> **288 passed, 7 skipped**;
`python -m kindi.config_reference --check` -> 0; `python -m kindi.support_matrix --check` -> 0;
`python -m compileall -q kindi` -> 0; `bash -n install.sh` -> 0; `git diff --check` -> 0. No GPU run,
no installer artifact republished, and no push. The public
`myfork/freebuff/tandem-cpu-recovery` remains at `437891e`; it does not contain this merge and needs an
authorized push before a release can use it. The source archive pinned by the installer is fork `main`,
so it also does not contain this branch's fixes until a later commit is published.

### Interrupted-WIP recovery (2026-11-11)

Recovered only the snapshot's secret-scanning work: `.secrets.baseline` was regenerated from the
current tracked tree, `.pre-commit-config.yaml` runs `detect-secrets` locally, and the CI
`secret-scan` job runs `detect-secrets-hook` so the baseline is checked without rewriting it. The
rest of `wip/main-interrupted-snapshot-20261109-045352` was rejected: its `install.sh` is empty, its
`tandem.py`/streaming layout/CPU worker are placeholders, its asymmetric TP and Turing two-card
paths are unverified, its CPUID binding has undefined functions and an ABI mismatch, and its
benchmark/verification harnesses are broken (zero baselines, invalid CLI calls). The WIP planning
docs contain stale or contradictory support claims and were not merged into the current docs.

Evidence: `detect-secrets-hook --baseline .secrets.baseline` on all tracked files, `pre-commit
validate-config .pre-commit-config.yaml`, and the usual test/config/support/compile/shell gates.
