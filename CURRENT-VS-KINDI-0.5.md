# Current worktree vs Kindi 0.5.0

Audit date: **2026-11-12**

> This file records the pre-improvement audit. Section 8 lists the low-risk fixes
> applied afterward, and section 9 records the 2026-11-12 evening 27B tandem
> implementation and gate; unresolved release blockers are called out there.

This report compares the current `freebuff/tandem-cpu-recovery` worktree with the
stock Kindi **0.5.0** release. It separates verified behavior from recorded benchmark
evidence, incomplete features, and release-hygiene problems. It is not a claim that
everything described here has been re-run on current hardware.

## 1. Baselines and current state

| | Stock “normal” Kindi 0.5 | Clean local 0.5 snapshot | Current worktree |
|---|---|---|---|
| Ref | `v0.5.0` tag, `1c4cc3e01732f0fada4ab7030ccf4f7651e01053` | `clean/kindi-0.5`, `1b0cbf780b57d464cdb3e5e9acf75ed49256d622` | `freebuff/tandem-cpu-recovery`, `b6d35c52e08fab8ee7b73bb45c061355ca5c8a5e` plus uncommitted work |
| Date | 2026-11-09 | clean snapshot of local 0.5 work | 2026-11-12 |
| Tracked files | 102 | 216 | 242 |
| Test modules | **0** (`tests/` does not exist) | 25 | 34 tracked, 43 test files including 9 untracked |
| Docs files | 12 | 84 | 94 tracked, 104 including 10 untracked |
| Git state | release tag | clean | **57 modified + 28 untracked = 85 entries at audit start** (this report adds one untracked file) |

The primary comparison below is against the **stock `v0.5.0` tag**. `clean/kindi-0.5`
already contains earlier local audit, CPU, server and documentation work, so it is a
narrower comparison, not the stock release. Relative to that clean snapshot, the
current worktree has 88 changed tracked paths plus the same 28 untracked paths.

Relative to the stock tag, the current tracked tree changes **186 files with
+22,353/−342 lines**; the committed part alone (`v0.5.0..HEAD`) is 179 files with
+19,776/−267. The 28 untracked files are not included in those figures.

Evidence labels used below:

- **Verified now:** the current worktree was executed during this audit.
- **Recorded measurement:** the repository contains the logs/results, but the test was
  not re-run in this audit.
- **Unit/CPU only:** covered by tests without a model or GPU run.
- **Design/blocked:** the feature is explicitly not working or not integrated.
- **Unverified:** the code or installer path exists, but no matching hardware result
  was recorded.

## 2. Bottom line

The current worktree is a substantial functional expansion over stock 0.5, not a clean
replacement release.

**Genuinely better:**

- A real dense CPU+GPU tandem runtime that is token-exact on the 2B and 9B in both
  placements, with a joint prompt cache and saved-opening reload.
- A portably built CPU engine with five ISA tiers, an AVX-only fallback, native Windows
  compilation, and exactness runs on 2B/4B/9B.
- A much more operationally complete server: bounded queue, 429 responses, optional
  per-source rate limiting, metrics, JSON request logs, Ollama-compatible endpoints,
  request validation and body limits.
- A real test suite (stock 0.5 has none), generated config/support references, CI jobs,
  secret scanning, installer guards and much more documentation.
- Native FP8 prompt-GEMM kernels, PQ2_0/Q2_0 import, and additional tensor-core/card
  plumbing — but several of these are partial or synthetic-only, not support claims.

**Still not working or not supportable:**

- **27B tandem** is implemented and measured in both placements (`cpu-first` peak 7,005 MiB,
  `gpu-first` 7,017 MiB, vs the 7,238 MiB full-GPU baseline, at 8,192 context, 2026-11-12).
  There is still no tandem throughput benchmark.
- **9B tensor-core plan** is refused (Hadamard block 4096); **4B tc** is only 2/4 exact;
  **k8v4** is absent for 9B and only 2/4 on 2B.
- **Blackwell FP8/BF16 and real FP8 models** are not validated; the local Blackwell gate
  correctly reports `unsupported` on the available Ada card.
- **MoE/Gemma 4 26B** has a design and a model-free cache prototype, not a backend.
- **Mixed-card pairs, P106, RTX 5060 Ti and Intel Arc** are installer/design targets,
  not hardware-verified support.
- **Release packaging is not ready:** the installer still fetches the older `main`
  commit, 28 feature/test/doc files are untracked, the version is still 0.5.0, and a
  bare `pytest -q` fails collection because of a copied benchmark tree.

## 3. What is new or better

### 3.1 Dense CPU+GPU tandem runtime — headline addition

Stock 0.5 has no dense tandem runtime. The current tree adds `kindi/tandem.py`, a
layer-range API on the GPU engine, and `TandemRunner` over a joint cache view.

Recorded results:

| Check | Result | Evidence |
|---|---|---|
| 2B, full GPU | 4/4 prompts token-exact, 40/40 tokens each, 189.9 tok/s | `docs/TANDEM-RUNTIME.md` |
| 2B, `cpu-first` (CPU 0..5, GPU 6..23, GPU head) | 4/4 exact, 70.2 tok/s | same |
| 2B, `gpu-first` (GPU 0..17, CPU 18..23, CPU head) | 4/4 exact, 70.9 tok/s | same |
| 2B joint cache, both placements | extension 320 tokens and branch 256 tokens reused; all six continuations token-exact against cache-off | `bench-logs/tandem-cache/RESULTS.md` |
| 2B HTTP restart | 344-token saved opening loaded after restart; `cached_tokens: 344` versus 0 cache-off; identical reply in both placements | `bench-logs/tandem-cache/server-results-*.json` |
| 2B 4K/8K cache | 4,080/8,176 extension tokens; 3,072/6,144 branch tokens; exact in both placements | `LONG-CONTEXT-RESULTS.md` |
| 9B, 512 context | 320 extension / 256 branch / 256 saved-opening tokens, exact in both placements; CPU load 105–108 s, GPU load 5.8–6.0 s | `RESULTS-9B.md` |
| 9B 4K/8K cache | 4,080/8,176 and 3,072/6,144 cached; exact in both placements | `context-9b-4k.json`, `context-9b-8k.json` |
| 9B checkpoint cost | 50.25 MiB joint recurrent state; 15.9/17.4 ms median save; 10.5/11.1 ms median restore; ~3.06 GiB device peak | `checkpoint-cost-9b-512.json` |
| 27B `cpu-first` (CPU 0..15, GPU 16..63, GPU head), 8,192 context | 4/4 exact, 40/40 tokens each; combined load 385.6 s; peak **7,005 MiB** vs 7,238 MiB full-GPU baseline | `bench-logs/tandem-27b-gate/cpu-first-8192.json` |
| 27B `gpu-first` (GPU 0..47 + embedding, CPU 48..63, CPU head), 8,192 context | 4/4 exact, 40/40 tokens each; combined load 413.3 s; peak **7,017 MiB** vs 7,238 MiB full-GPU baseline | `bench-logs/tandem-27b-gate/gpu-first-8192.json` |

The cache work also fixed a real bug: `Openings.load` assigned a NumPy K/V array into a
CuPy view with a plain assignment, which CuPy treated as a scalar fill. It now uses the
in-place `_to_dev` copy. A GPU-free regression test reproduces the scalar-fill failure.
This is a concrete correctness improvement over the pre-fix state.

**Not covered:** cumulative-B (33B), MoE, long-context *saved-opening reload* on
9B, and tandem throughput at the CROSS-MODE positions. The 105–108 s 9B CPU load — and the
≈365 s 27B CPU load — are also practical costs, not a speed claim.

### 3.2 CPU portability and CPU serving

Stock 0.5 builds `kcpu.c` with `gcc -O3 -march=native` and has no explicit tier
selection. The current tree adds five explicit tiers:

`avx2-fma-f16c`, `avx2-f16c-no-fma`, `avx2-fma`, `avx2`, `avx-only`.

The engine selects the best tier from host flags, supports `KINDI_CPU_ISA` to pin a
tier, builds per-tier cached libraries with explicit flags, and checks the loaded
library’s reported ISA. The installer reports the selected path and refuses CPU-only
installs on a CPU without AVX.

Recorded exactness (i7-8086K, native Windows, clang 18.1.8, 6 threads, ctx 1024):

| Tier | 2B | 4B | 9B |
|---|---:|---:|---:|
| `avx2-fma-f16c` | 4/4, 43.6 tok/s | 4/4, 18.1 tok/s | 4/4, 10.7 tok/s |
| `avx2-f16c-no-fma` | 4/4, 43.9 tok/s | 4/4, 15.8 tok/s | 4/4, 10.4 tok/s |
| `avx2-fma` | 4/4, 31.9 tok/s | 4/4, 17.8 tok/s | 4/4, 10.9 tok/s |
| `avx2` | 4/4, 29.1 tok/s | 4/4, 19.2 tok/s | 4/4, 11.0 tok/s |
| `avx-only` | 4/4, 8.9 tok/s | 4/4, 3.9 tok/s | 4/4, 2.0 tok/s |

Also added:

- Native Windows CPU builds with clang + MSVC and libomp, without a vcvars shell.
- CPU server chat on 2B/4B/9B, non-stream and stream, 0 errors.
- 24K CPU generation/read benchmarks for 2B/4B/9B.
- A compile-and-load CI smoke job for the CPU engine.
- Installer AVX preflight and a clear error for non-AVX CPU-only machines.

**Caveat:** the AVX-only and no-FMA runs are forced tiers on an AVX2 host, not native
Sandy Bridge or native no-FMA hardware. The Xeon W-2133 and i9-13900K remain unverified.

### 3.3 Server and API operations

`kindi/server.py` grew from a basic one-request server into a more operable service:

- Bounded queue with `429` and `Retry-After` when full.
- Optional per-source-address fixed-window rate limit, off by default, with metrics.
- `/metrics` with queue, cache and request counters.
- `log_json = 1` structured request logs with request ID, status, duration, queue wait,
  token counts and tok/s; returned as `X-Request-ID`.
- Ollama-compatible `/api/tags`, `/api/chat`, `/api/generate`.
- Request validation, authentication checks, a maximum request-body size, and a 413
  response that preserves the prompt cache.
- `/health` now hides busy/GPU/uptime/usage details unless the API key is supplied;
  the installer still receives the five liveness fields it needs.
- Sampling profiles and `presence_penalty` for the Bonsai backends.

The rate limiter and queue are process-local; a reverse proxy still shares one source
address. This is not a distributed quota.

### 3.4 Prompt-cache improvements beyond tandem

- `G4Cache` now enforces the documented 1 GiB host byte budget after every checkpoint
  or park, evicts oldest parked conversations then active checkpoints, refuses an
  individually oversized snapshot, and records eviction/usage/peak counters exposed by
  `/metrics`.
- Tandem checkpoints cover both owners’ recurrent state and K/V, and saved openings are
  fingerprinted by placement, layer ranges, CPU ISA/threads and `tandem.py`, preventing
  a cache made by one split from being reused by another.
- The saved-opening scalar-fill bug fix described above.

Stock 0.5 had a prompt cache, but not the joint tandem ownership, the Gemma 4 byte
budget/telemetry, or the saved-opening fingerprinting described here.

### 3.5 Formats, quantization and tensor cores

- **PQ2_0 / Q2_0 import:** GGUF types 142/42 import to packs byte-identical to the
  PTQ1_0 pack for the 2B. Both verify 4/4, and the 24K benchmark showed no meaningful
  speed difference because the packs are identical. 4B/9B/27B files were not available.
- **K8/V4:** V4 store/one-pass/exact chunk kernels and V4 decode/verify plan kernels
  are implemented and selected from `tp.kv`; `KINDI_TC=1` is no longer refused for
  k8v4. Direct GPU checks store against NumPy and compare chunk V4 with decode V4 bit
  for bit. This remains an experimental lossy path; see the gaps below.
- **Native FP8 prompt GEMM:** `gemma4_fp8.cu` adds E4M3/E5M2 `mma.sync` m16n8k32 kernels
  with per-token activation scaling, selected in `linear_batch` for compute capability
  >= 8.9, with `KINDI_G4_FP8=0` as an opt-out. Synthetic Ada timings are 0.11–0.76 ms
  (15.1–35.5 TFLOPS). The path is **not** bit-exact with the old f32 dequantize +
  cuBLAS path.
- **Blackwell verification lane:** `tools/verify_blackwell_fp8.py` accepts only
  sm_100/sm_120, exercises both FP8 formats and tail shapes, and records a machine-
  readable result. Non-Blackwell hardware is `unsupported` exit 2, never a pass.
- **Turing tensor cores:** `tc_turing.py`, `tc_turing.cu` and `tca_turing.cu` add the
  sm_75 path; the installer classifies every sm_80+ card as `tc`, and support docs
  record 2B/4B/9B/27B tc and plain results.
- **Card plumbing:** installer card classification, dual pairing, mixed-generation
  pairs and a P106-100 support document are present. These are described as targets
  pending hardware verification where appropriate.

### 3.6 Installer, CI, testing and documentation

- Installer source archive is pinned by SHA-256 before extraction; custom sources can
  be pinned with `KINDI_SOURCE_SHA256`.
- GRUB edit guard, AVX preflight, card classification and mixed/dual pair handling.
- Optional `/health` checks in all three Dockerfiles.
- CI workflow, CPU build/smoke job, GPU smoke lane, dependency pinning and Dependabot.
- Pre-commit secret scanning and a committed `.secrets.baseline`.
- Generated configuration reference and support matrix, both with `--check` drift
  detection.
- 34 tracked test modules (+9 untracked), a 436-test CPU/unit pass, and much more
  benchmark/limitation documentation than stock 0.5.
- Cross-mode CPU/GPU benchmark tables and native Windows benchmark runs.
- The 27B tc path has a 24K server/cache record: 40.5 tok/s read, 34.7–37.6 tok/s
  generation at 100–24K positions, 351 MiB free, and three real server save/load cycles
  with bit-exact replies.

Stock 0.5 had **no test suite at all**, so the testing and generated-reference work is
a categorical improvement even where individual features remain incomplete.

## 4. What is worse, incomplete, or does not work

### 4.1 Functional gaps

These are not necessarily regressions from stock 0.5 — several are new features that do
not yet meet their own gates. They should not be presented as supported.

| Area | Current state | Why it matters |
|---|---|---|
| **27B tandem** | implemented and measured in both placements: `cpu-first` 4/4 (peak 7,005 MiB) and `gpu-first` 4/4 (peak 7,017 MiB), all 40/40 tokens each, vs the 7,238 MiB full-GPU baseline (8,192 context, 2026-11-12); CROSS-MODE tandem throughput rows are still `n/a`. | The “larger than VRAM” use case works in one of two placements. The plain GPU engine now accepts grouped-query ratios up to 8:1 at head size 256, reusing the existing generic-GQA kernels; the 4:1 fast paths are unchanged. |
| **9B tensor-core plan** | Refused before graph build because the Hadamard block is 4096. | The 9B tc cell is not available; plain 9B is 4/4. |
| **4B tensor-core plan** | 2/4 exact at 178.2 tok/s (prompts 2 and 3 fail); plain 4B is 4/4 at 114.9 tok/s. | The tc path is not a general replacement; it is an uneven, partial optimization. |
| **2B/9B k8v4** | 2B k8v4 is 2/4 against q8_0; 9B k8v4 is unavailable. 4B k8v4 is 4/4. | Experimental lossy cache; do not enable by default or describe as 9B-ready. |
| **Blackwell FP8/BF16** | No Blackwell hardware result. Local sm_89 gate returns `unsupported` exit 2. | “Blackwell FP8 support” is not established; only the verification lane exists. |
| **Real FP8/BF16 model** | No Gemma 4 E4B BF16/FP8 GGUF available. The local `gemma4:26b` is Q4_K_M and is rejected at unsupported GGUF type 7; the Gemma 4 engine is E4B-only. | Kernel tests do not prove model-level token agreement or throughput. |
| **MoE / Gemma 4 26B** | Model-free expert-cache state machine and design documents; no router, pack, backend or GPU copy integration. | The 26B MoE path is not implemented; do not advertise it as runnable. |
| **Mixed cards, P106, RTX 5060 Ti, Turing hardware** | Installer offers mixed pairs and P106 single/dual profiles; docs mark them unverified. No mixed-card execution record. | Installer acceptance is not hardware validation. A user can be offered a pair that has never produced a token. |
| **Intel Arc A770** | Separate SYCL/oneAPI backend; M0 audit only, M1 blocked on hardware. | Design-only; not part of the current support surface. |
| **Continuous batching** | Archived as a no-go. The server serves one active request and queues the rest; the queue does not increase throughput. | The server scales operationally, not computationally. Multi-GPU replicas still need `tp.py` work. |
| **Long-context tandem evidence** | 9B saved-opening reload at 4K/8K and 9B peak VRAM were not measured. Tandem rows in the cross-mode benchmark remain `n/a`. | Do not generalize the 512-context 9B results to long-context serving or memory. |

### 4.2 Current-state breakage and release blockers

These were problems with the worktree at audit time, separate from feature completeness.
Some were fixed afterward; see section 8.

1. **A bare `pytest -q` did not work at audit time.** It failed collection with **26 errors**.
   The cause is `bench-logs/scan-tree-f7de617/`, an untracked copy of the repository
   containing duplicate `tests/` and `kindi/` packages. Pytest imports the copied
   modules, then reports basename mismatches and imports the wrong `kindi` package
   (for example, `ModuleNotFoundError: No module named 'kindi.dtypes'`).
   The canonical command is fine:

   ```text
   python -m pytest -q tests
   436 passed, 19 skipped, 1 warning in 19.72s
   ```

   This was a repository-hygiene failure, not a code failure: any CI or contributor
   that ran plain `pytest` at the root got a red result for the wrong reason. It is
   fixed by the new `pytest.ini`; see section 8.

2. **The installer does not install the current state.** `install.sh` still pins:

   ```text
   KINDI_VERSION="0.5.0"
   SOURCE_COMMIT="fd5414889e4acfd72b46bba3138a908b8e97c5be"
   SOURCE_SHA256="3ca1e79d..."
   ```

   `fd54148` is `main`, while the audited branch HEAD is `b6d35c5`. The installer’s
   default source therefore omits the nine branch commits (including the dense tandem
   runtime and AVX-only recovery) and all 85 uncommitted changes. Its image and
   install metadata still say `0.5.0`. The installer artifact documented elsewhere was
   not rebuilt for this branch.

3. **The current worktree is not self-contained.** At audit start it had 57 modified
   and 28 untracked files (plus this report). The untracked set includes runtime code
   and tests that are required for the current feature set:

   - `kindi/dtypes.py`, `kindi/expert_cache.py`, `kindi/gemma4_fp8.cu`
   - 9 test modules for FP8, MoE cache, Gemma 4 cache, secret scanning, long-context
     checking and Blackwell verification
   - `tools/verify_blackwell_fp8.py`, `tools/check_secret_baseline.py`,
     `tools/check_long_context.py`, `tools/dory_gguf_inventory.py`
   - 10 design/support documents and two `.github` workflow files

   A clean checkout of `b6d35c5` is therefore not the state that produced the 436-test
   pass. The 436-pass result applies only to this dirty worktree.

4. **Version and changelog identity are stale.** `kindi/__init__.py` still says
   `VERSION = "0.5.0"`; `README.md` and `CHANGELOG.md` still describe stock 0.5.0.
   There is no changelog entry or tag for the tandem, CPU, FP8, server, test or
   installer work. A user cannot distinguish a stock 0.5 install from this tree by
   version, and the installer builds `kindi:0.5.0`.

5. **Documentation is partly stale or internally inconsistent.**
   `docs/TANDEM-STATUS.md` says near the top that the 9B check passed on 2026-11-12,
   but its cleanup gate still says “2B passed ... 9B is still open.” It also says CPU
   ISA detection/dispatch is absent, although the five-tier dispatch is implemented and
   tested. `docs/TANDEM-RUNTIME.md`/`TANDEM-STATUS.md` correctly say tandem throughput
   rows are `n/a`, while other docs describe the runtime as implemented. Treat the docs
   as a map with stale regions, not a single consistent contract.

6. **`/health` is a breaking change for existing monitors.** Stock 0.5 returned busy
   progress, uptime, stats and GPU free/total memory unauthenticated. The current
   server returns only `ok`, `kindi`, `commit`, `model` and `ctx` without a key. This
   is a defensible security improvement, but an existing unauthenticated monitoring
   integration will lose its data.

7. **GPU behavior was revalidated after the audit (2026-11-12 evening).** `kindi verify`
   reports ALL PASS on the 2B (4/4, 153.7 tok/s), 4B (4/4, 108.9) and 9B (4/4, 74.6), and
   the 27B tandem `cpu-first` gate is exact 4/4 — all with the refactored attention code.
   What remains unverified is listed in §9.

8. **Shared-worktree merge risk remains.** The main checkout `T:/kindi` has its own
   uncommitted copies of shared files (`kindi/tp.py`, `kindi/spec.py`, docs and tests).
   The current branch’s changes must be reconciled by hand; blindly copying either
   worktree over the other will lose work. This is a delivery risk, not a runtime bug.

## 5. What this means

### If “normal Kindi 0.5” means the stock release

The current worktree is clearly ahead in engineering breadth: tests, CPU portability,
server operations, tandem 2B/9B, cache correctness, installer guards and documentation.
It is also substantially less release-ready: the installer points at an older commit,
28 required files are untracked, the version is unchanged, and the default test command
is broken by an accidental tree copy. The right description is **“feature branch with
several verified improvements and a long list of explicit non-goals,”** not “Kindi 0.5
plus fixes.”

### If “normal Kindi 0.5” means the clean local snapshot

The delta is narrower, but the same conclusions hold. The current tree adds the dense
tandem runtime, the latest CPU ISA fallback work, FP8/MoE experiments, mixed-split
plumbing and the newer server/installer changes beyond `clean/kindi-0.5`; it also adds
the same untracked-file and installer-source problems.

## 6. Recommended order of work

1. **Make the tree publishable before calling it a release:** add the 28 untracked
   files, commit the 57 modifications, decide the version, and point the installer’s
   `SOURCE_COMMIT`/SHA-256 at the exact release commit. Do not ship an installer that
   fetches `fd54148` while claiming the current branch.
2. **Fix the test entry point:** exclude `bench-logs/` from pytest collection or remove
   the duplicate `scan-tree-f7de617` copy, then make the documented command and CI use
   the same invocation. Keep `python -m pytest -q tests` as the canonical command until
   then.
3. **Turn the known partials into explicit no-gos in user-facing docs:** 27B tandem,
   9B tc, 4B tc, 9B k8v4, Blackwell FP8, real FP8/BF16 models, MoE and mixed cards.
   The repository already has honest limitation pages; make those the support contract,
   not the target matrix.
4. **Re-run a GPU smoke/model check** on the current commit after the dirty files are
   committed. The 2B/4B/9B `verify` and 27B tandem `cpu-first` results above were taken
   with the current uncommitted attention code (§9); a committed-tree rerun is still
   the release gate.
5. **Resolve the `T:/kindi` shared-file conflicts by hand** before any merge or release
   build.

## 7. Verification performed for this report

Run in the current worktree on 2026-11-12:

| Command | Result |
|---|---|
| `python -m pytest -q tests` | 436 passed, 19 skipped, exit 0 (444 passed after §9) |
| `python -m pytest -q` (after fix) | **436 passed, 19 skipped**, exit 0 |
| `python -m pytest -q` (audit-time, before fix) | **26 collection errors**, exit 2, caused by `bench-logs/scan-tree-f7de617/` shadowing |
| `python -m kindi.config_reference --check` | exit 0 |
| `python -m kindi.support_matrix --check` | exit 0 |
| `git -c core.whitespace=cr-at-eol diff --check` | exit 0 |
| `git status --short` | 57 modified, 28 code/docs files untracked (+ this report) |
| `git diff --stat v0.5.0` | 186 tracked paths, +22,353/−342 (audit-time; §9 extends it) |
| `kindi verify` 2B/4B/9B (2026-11-12 evening) | ALL PASS, 153.7 / 108.9 / 74.6 tok/s |
| 27B tandem gate, `cpu-first` (2026-11-12) | **4/4 prompts, 40/40 each; peak 7,005 MiB; 8,192 ctx; exit 0** |

No commit, push, merge, installer publication, GPU benchmark or hardware validation was
performed for this report. The benchmark numbers above are quoted from the repository’s
recorded logs and docs, not re-measured here.

## 8. Improvements applied after this audit

Two low-risk weak points were fixed without changing unverified GPU code:

1. **Default test discovery.** Added `pytest.ini` with `testpaths = tests` and
   `norecursedirs = bench-logs`. Bare `python -m pytest -q` now collects only the real
   suite and passes with **436 passed, 19 skipped**. Focused cross-mode and docs-link
   tests pass, and `build_cross_mode.py --check` exits 0.
2. **Tandem documentation drift.** `docs/TANDEM-STATUS.md` no longer says CPU ISA
   detection is absent, no longer calls the 9B tandem gate open, and now says the page
   describes this branch. `docs/benchmarks/build_cross_mode.py` and its generated
   `CROSS-MODE.md`/`CROSS-MODE.csv` now distinguish “runtime exists but no throughput
   benchmark” from “runtime absent”; the rows remain `n/a` because no new timing run was
   made.
3. **Coverage regression floor.** The CI CPU job now runs
   `pytest tests/ -q --cov=kindi --cov-report=term --cov-fail-under=27`, with
   `pytest-cov` in the test requirements. The measured local total is 27.90%; the
   floor is deliberately conservative and documented as an anti-regression gate, not a
   quality claim. The performance half remains the existing opt-in GPU baseline gate.
4. **Windows quickstart.** Added `docs/WINDOWS.md`, linked from README and INSTALL: the
   verified native CPU path, clang/libomp setup, model/verify/serve commands, and explicit
   limits. Native Windows GPU/AMD execution and the Linux installer are not claimed.

Still unresolved and unchanged: the installer source commit, the untracked release
payload, 9B tc, k8v4, Blackwell/FP8 hardware validation, and the 27B `gpu-first` and
tandem-throughput measurements. The 27B head-ratio blocker and the lack of a fresh
GPU/model run were resolved as §9 item 5 and §9 item 6 below.

## 9. 27B tandem grouped-query support and the gate (2026-11-12 evening)

The §4.1 head-ratio blocker is fixed. The plain GPU `Engine` now validates grouped-query
attention generically (`Engine._gqa_groups` in `kindi/engine.py`: `G = NH/KV` must divide
and be `<= 8`; groups other than 4 require `head_dim == 256`), and `Engine._attn_core`
dispatches to the generic-GQA kernels that already existed in `kindi/kindi.cu`
(`k_attn_sg`, `k_attn_vgn`, `k_attn_combn`) when `G != 4`. No new CUDA kernels were
written. The 4:1 fast kernels are untouched: the 2B/4B/9B `kindi verify` results above
ran *after* the refactor, which is the regression check for code that moved.
8 CPU-only tests in `tests/test_gqa_heads.py` pin the ratios and kernel arguments, and
the full suite is now **444 passed, 19 skipped**.

The measured 27B result is recorded in §7 and in
`bench-logs/tandem-27b-gate/RESULTS.md`: `TandemEngine` with CPU layers 0..15, GPU layers
16..63 and the GPU head is token-exact on all four gate prompts at an 8,192-token
context (never reduced), and its sampled peak is 233 MiB below the full-GPU plan-engine
baseline. The gate harness's `--probe-gpu-owner` exit code was inverted (it exited 0 only
when both GPU ranges *failed*); it now exits 0 when both construct, and both 27B ranges
construct at 8,192 context.

Updated "what is worse or no longer works" list (as of 2026-11-12 evening):

- **27B `gpu-first`** — measured, and `gpu-first` now works in both placements, so the
  partial-plan owner design the report used to call for is no longer needed on either path.
- **Tandem throughput** — still no number at the CROSS-MODE positions. `bench` does not
  accept tandem, and the CROSS-MODE rows stay `n/a` until a real tandem `bench` run is
  recorded.
- **Installer pin, untracked payload, version** — unchanged: `SOURCE_COMMIT` still names
  `fd54148`, `kindi/__init__.py` still says `0.5.0`, 33 files are untracked and no commit
  exists for them. Committing and publishing is an explicit release decision, left for
  the release owner.
- **9B tc** (Hadamard 4096 refused), **4B tc** (2/4 exact), **k8v4** (2B 2/4, 9B unavailable),
  **Blackwell/FP8/BF16 validation**, **MoE/Gemma-4 26B backend**, **mixed-card/P106/5060 Ti/
  Turing/Arc hardware** — all unchanged, all the same explicit non-goals as §4.1.
- **`git diff --check` without the project's `cr-at-eol`** still flags trailing CRs in
  `kindi/gemma4_cache.py` from earlier uncommitted edits; with `-c core.whitespace=cr-at-eol`
  it exits 0.
