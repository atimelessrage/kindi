# Current worktree vs Kindi 0.5.0

Audit date: **2026-11-12**

> This file records the pre-improvement audit and the fixes that followed.
> Sections 8–11 record the post-audit improvements, the 2026-11-12 evening 27B
> tandem implementation/gate, the later requirement/structural-coverage work, and
> the 2026-11-13 measurement rounds (sampled-peak VRAM, the 27B rate/peak and
> checkpoint-cost rows, the 2B CPU-layer sweep). **Section 11 is the current
> "what is better / what is worse" list; sections 10 and earlier are kept as the
> history at their own dates.** The best-of branch is merged into `main`: both
> `main` and `freebuff/tandem-cpu-recovery` are at `0cd58bcd223bf1afba2d8319601e64b014a97082`. The installer source
> pin and the `0.5.0` version identity remain a release-owner publication step, not
> a code-complete claim.

This report compares the current `freebuff/tandem-cpu-recovery` worktree with the
stock Kindi **0.5.0** release. It separates verified behavior from recorded benchmark
evidence, incomplete features, and release-hygiene problems. It is not a claim that
everything described here has been re-run on current hardware.

## 1. Baselines and current state

| | Stock “normal” Kindi 0.5 | Clean local 0.5 snapshot | Current worktree |
|---|---|---|---|
| Ref | `v0.5.0` tag, `1c4cc3e01732f0fada4ab7030ccf4f7651e01053` | `clean/kindi-0.5`, `1b0cbf780b57d464cdb3e5e9acf75ed49256d622` | `main` and `freebuff/tandem-cpu-recovery`, `0cd58bcd223bf1afba2d8319601e64b014a97082` |
| Date | 2026-11-09 | clean snapshot of local 0.5 work | 2026-11-13 (merged state) |
| Tracked files | 102 | 216 | 322 |
| Test modules | **0** (`tests/` does not exist) | 25 | 53 tracked test modules |
| Docs files | 12 | 84 | 139 tracked paths under `docs/` |
| Git state | release tag | clean | `main` fast-forwarded to the best-of branch at `0cd58bcd223bf1afba2d8319601e64b014a97082`; generated coverage artifacts are ignored; `bench-logs/` is ignored local evidence |

The primary comparison below is against the **stock `v0.5.0` tag**. `clean/kindi-0.5`
already contains earlier local audit, CPU, server and documentation work, so it is a
narrower comparison, not the stock release. Relative to that clean snapshot, the committed branch has **181 changed tracked paths
with +18,511/−620 lines**.

Relative to the stock tag, the committed branch (`v0.5.0..HEAD`) changes **268 tracked
files with +32,869/−372 lines**. The generated `.coverage`/`coverage.json` artifacts
are now ignored; `bench-logs/` remains ignored local evidence.

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
- A real test suite (stock 0.5 has none), generated config/support references, CI jobs
  with a 27% line-and-branch coverage floor, secret scanning, installer guards and
  much more documentation.
- Native FP8 prompt-GEMM kernels, PQ2_0/Q2_0 import, and additional tensor-core/card
  plumbing — but several of these are partial or synthetic-only, not support claims.

**Still not working or not supportable:**

- **27B tandem** is implemented and measured in both placements (`cpu-first` peak 7,005 MiB,
  `gpu-first` 7,017 MiB, vs the 7,238 MiB full-GPU baseline, at 8,192 context, 2026-11-12).
  Its rate and sampled peak are now measured at 8K and 24K against a full-GPU baseline
  (7.2-8.4 vs 34.8-34.9 tok/s at position 100; 446 MiB less peak at 8K), and the joint
  checkpoint costs 54.5-61.5 ms to save. Tandem still does not beat GPU-only throughput,
  and the full-GPU engine also loads at 24K on this card, so the "larger than VRAM"
  premise is not yet demonstrated.
- **9B tensor-core plan** is refused (Hadamard block 4096); **4B tc** is 3 exact + 1 accepted
  tie with the FD fallback (forced FD was 2/4); **k8v4** is absent for 9B and only 2/4 on 2B.
- **Blackwell FP8/BF16 and real FP8 models** are not validated; the local Blackwell gate
  correctly reports `unsupported` on the available Ada card.
- **MoE/Gemma 4 26B** has a design and a model-free cache prototype, not a backend.
- **Mixed-card pairs, P106, RTX 5060 Ti and Intel Arc** are installer/design targets,
  not hardware-verified support.
- **Release packaging is not ready:** the installer still fetches the older `fd54148`
  commit and the version is still 0.5.0. The best-of branch is merged into `main`, but
  publishing a new installer artifact and retargeting `SOURCE_COMMIT`/`SOURCE_SHA256`
  remain release-owner steps. The bare-`pytest` collection failure is fixed and the
  generated coverage artifacts are ignored.

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
| 27B `cpu-first` (CPU 0..15, GPU 16..63, GPU head), 8,192 context | 4/4 exact, 40/40 tokens each; combined load 385.6 s (full-model CPU load); peak **7,005 MiB** vs 7,238 MiB full-GPU baseline | `bench-logs/tandem-27b-gate/cpu-first-8192.json` |
| 27B `gpu-first` (GPU 0..47 + embedding, CPU 48..63, CPU head), 8,192 context | 4/4 exact, 40/40 tokens each; combined load 413.3 s (full-model CPU load); peak **7,017 MiB** vs 7,238 MiB full-GPU baseline | `bench-logs/tandem-27b-gate/gpu-first-8192.json` |
| 2B/4B/9B `cpu-first` CROSS-MODE throughput at 24,576 ctx (medians of three, 2026-11-13) | 2B read 84.3, gen 79.8/72.3/58.8/43.2/34.9; 4B read 49.3, gen 47.0/43.8/31.7/23.7/16.7; 9B read 38.1, gen 37.5/34.7/25.4/20.5/15.2 tok/s; free-after-load 6411/5431/4635 MiB | `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/` (2026-11-12 single runs kept as `bench-<model>-cpu-first-24k.log`; see §12) |
| 27B rate + sampled peak, 8K and 24K, both placements and full GPU | tandem 7.2-8.4 tok/s at position 100 vs 34.8-34.9 full GPU; peaks 7,027 vs 7,473 MiB at 8K and 7,699 vs 7,783 MiB at 24K | [27B-8K.md](docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md) |
| 27B joint cache at 4K/8K | saved-opening reloads exact in both placements; 149.6 MiB checkpoint, 54.5-61.5 ms save, 27.0-28.5 ms restore; checkpoint-only peaks 6,842/6,833 MiB (4K) and 7,030/7,027 MiB (8K) | same |
| 9B saved opening at 4K/8K | extension 4,080/8,176, branch and saved opening 3,072/6,144, cache-off 0, exact in both placements; loads 33.3-39.5 s | `bench-logs/tandem-cache/saved-9b-4k.json`, `saved-9b-8k.json` |
| 2B CPU-layer sweep, 24,576 ctx | 2/4/6/8/12 CPU layers: 118.9/41.7/39.5/22.6/18.2 tok/s at 24K; knee at 4-6 layers | `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/README.md` |

The cache work also fixed a real bug: `Openings.load` assigned a NumPy K/V array into a
CuPy view with a plain assignment, which CuPy treated as a scalar fill. It now uses the
in-place `_to_dev` copy. A GPU-free regression test reproduces the scalar-fill failure.
This is a concrete correctness improvement over the pre-fix state.

**Not covered:** cumulative-B (33B) and MoE. The 2B/4B/9B CROSS-MODE rows, the 27B
rate/peak rows and the 2B layer sweep are recorded, and none of them beats GPU-only.
The 105–108 s 9B CPU load was a full-model load; the CPU engine now loads only its
owned slice (measured owned-slice loads: 17.2 s for 4B, 32.4 s for 9B, 105.8-114.6 s
for the 27B), so the old ≈365 s 27B full-load figure no longer describes a tandem start.

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
- 44 tracked test modules, a 444-test CPU/unit pass, and much more
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
| **27B tandem** | exact 4/4 in both placements (`cpu-first` peak 7,005 MiB, `gpu-first` 7,017 MiB, vs 7,238 MiB full-GPU at 8,192 context, 2026-11-12); 2026-11-13 rate plus sampled peak at 8K and 24K for both placements and full GPU (7.2-8.4 vs 34.8-34.9 tok/s at 100; 446 MiB less peak at 8K, 193 MiB at 24K); joint cache exact at 4K/8K with saved-opening reloads in both placements. | The split is exact and it saves peak VRAM, but it is **4-6x slower**, and the full-GPU engine still loads at 24K on the 4060 Ti (7,783 MiB peak, 167 MiB free), so this is a headroom trade, not a capacity requirement. A genuinely larger-than-VRAM row still needs a longer context or a smaller card. |
| **9B tensor-core plan** | Refused before graph build because the Hadamard block is 4096. | The 9B tc cell is not available; plain 9B is 4/4. |
| **4B tensor-core plan** | 3 exact + 1 accepted llama.cpp tie with the FD fallback at 164.6 tok/s (the forced-FD run was 2/4); plain 4B is 4/4 at 114.9 tok/s. | The tc path is a conditional pass, not a general replacement; forced-FD exactness remains open. |
| **2B/9B k8v4** | 2B k8v4 is 2/4 against q8_0; 9B k8v4 is unavailable. 4B k8v4 is 4/4. | Experimental lossy cache; do not enable by default or describe as 9B-ready. |
| **Blackwell FP8/BF16** | No Blackwell hardware result. Local sm_89 gate returns `unsupported` exit 2. | “Blackwell FP8 support” is not established; only the verification lane exists. |
| **Real FP8/BF16 model** | No Gemma 4 E4B BF16/FP8 GGUF available. The local `gemma4:26b` is Q4_K_M and is rejected at unsupported GGUF type 7; the Gemma 4 engine is E4B-only. | Kernel tests do not prove model-level token agreement or throughput. |
| **MoE / Gemma 4 26B** | Model-free expert-cache state machine and design documents; no router, pack, backend or GPU copy integration. | The 26B MoE path is not implemented; do not advertise it as runnable. |
| **Mixed cards, P106, RTX 5060 Ti, Turing hardware** | Installer offers mixed pairs and P106 single/dual profiles; docs mark them unverified. No mixed-card execution record. | Installer acceptance is not hardware validation. A user can be offered a pair that has never produced a token. |
| **Intel Arc A770** | Separate SYCL/oneAPI backend; M0 audit only, M1 blocked on hardware. | Design-only; not part of the current support surface. |
| **Continuous batching** | Archived as a no-go. The server serves one active request and queues the rest; the queue does not increase throughput. | The server scales operationally, not computationally. Multi-GPU replicas still need `tp.py` work. |
| **Long-context tandem evidence** | complete for the 9B and 27B: saved-opening reloads at 4K/8K exact in both placements (9B 4,080/8,176 extension and 3,072/6,144 branch/opening, 33.3-39.5 s loads, 82-221 s cache-off read versus 0.39-0.72 s cached; 27B 8,176/6,144 with 105.8-114.6 s loads). 2B/4B/9B CROSS-MODE 24K rows and a 2B CPU-layer sweep are recorded. | Checkpoint cost grows with layer count (19.3 MiB on the 2B, 50.3 on the 9B, 149.6 on the 27B) even though context does not change it; budget host RAM accordingly. The cached-read times are harness timings, not a serving-rate claim. |

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

   `fd54148` is the old stock-0.5-era `main`, while the merged best-of `main` is now
   `082c5da`. The installer’s default source therefore still omits the dense tandem
   runtime, AVX-only recovery, generic-GQA support and the other verified work.
   Its image and install metadata still say `0.5.0`. This is a release-publication
   blocker, documented as a release-owner step in `docs/RELEASE-CHECKLIST.md`; it
   cannot be fixed by retargeting the pin without publishing the matching artifact.

3. **The branch was not self-contained at audit time (now committed and merged into
   `main`).** At audit start it had 57 modified and 28 untracked files
   (plus this report). The untracked set included runtime code and tests that
   are required for the current feature set:

   - `kindi/dtypes.py`, `kindi/expert_cache.py`, `kindi/gemma4_fp8.cu`
   - 9 test modules for FP8, MoE cache, Gemma 4 cache, secret scanning, long-context
     checking and Blackwell verification
   - `tools/verify_blackwell_fp8.py`, `tools/check_secret_baseline.py`,
     `tools/check_long_context.py`, `tools/dory_gguf_inventory.py`
   - 10 design/support documents and two `.github` workflow files

   The branch has since been committed and merged into `main` at `082c5da`, so
   the verified state is no longer only a dirty worktree.

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
   the 27B tandem gate is exact 4/4 in both placements — all with the refactored
   attention code. The 2B/4B/9B CROSS-MODE tandem throughput rows are recorded. The
   27B throughput-plus-peak row landed on 2026-11-13 (§11).

8. **Shared-worktree merge risk is resolved for this branch.** Both worktrees are
   at `0cd58bcd223bf1afba2d8319601e64b014a97082` and `T:/kindi` is clean. Future shared-file work still needs the
   AGENTS.md lease protocol; blindly copying either worktree over the other will
   lose work.

## 5. What this means

### If “normal Kindi 0.5” means the stock release

The merged tree is clearly ahead in engineering breadth: tests, CPU portability,
server operations, tandem 2B/9B/27B, cache correctness, installer guards and
documentation. It is also substantially less release-ready: the installer points at
an older commit and the version is unchanged. The best-of branch is merged into
`main`; the default test command is fixed and coverage artifacts are ignored.
The right description is **“best-of main with several verified improvements and a
long list of explicit non-goals,”** not “Kindi 0.5 plus fixes.”

### If “normal Kindi 0.5” means the clean local snapshot

The delta is narrower, but the same conclusions hold. The current tree adds the dense
tandem runtime, the latest CPU ISA fallback work, FP8/MoE experiments, mixed-split
plumbing and the newer server/installer changes beyond `clean/kindi-0.5`; it also
retains the same main-merge and installer-source problems.

## 6. Recommended order of work

1. **Finish the release decision:** the best-of branch is merged into `main` at
   `0cd58bcd223bf1afba2d8319601e64b014a97082`. Decide the version, publish the matching installer artifact, and
   point `SOURCE_COMMIT`/SHA-256 at the exact release commit. Do not ship an
   installer that fetches `fd54148` while claiming the current tree.
2. **Fix the test entry point:** exclude `bench-logs/` from pytest collection or remove
   the duplicate `scan-tree-f7de617` copy, then make the documented command and CI use
   the same invocation. Keep `python -m pytest -q tests` as the canonical command until
   then.
3. **Turn the known partials into explicit no-gos in user-facing docs:** 9B tc,
   4B tc, 9B k8v4, Blackwell FP8, real FP8/BF16 models, MoE and mixed cards.
   The repository already has honest limitation pages; make those the support contract,
   not the target matrix.
4. **Confirm the release commit:** the 2B/4B/9B `verify`, both 27B tandem gate
   placements and the 494-test suite are recorded for the merged state; a rerun
   on the exact release commit remains the release gate.
5. **Keep the AGENTS.md lease protocol for any future shared-file work** so the
   main worktree and feature worktrees do not drift again.

## 7. Verification performed for this report

Run in the current worktree on 2026-11-12:

| Command | Result |
|---|---|
| `python -m pytest tests/ -q --cov=kindi --cov-branch --cov-fail-under=27` | **462 passed, 19 skipped**, total line-and-branch coverage **27.15%**, exit 0 |
| `python -m pytest -q` (after fix) | **462 passed, 19 skipped**, exit 0 |
| `python -m pytest -q` (audit-time, before fix) | **26 collection errors**, exit 2, caused by `bench-logs/scan-tree-f7de617/` shadowing |
| `python -m kindi.config_reference --check` | exit 0 |
| `python -m kindi.support_matrix --check` | exit 0 |
| `git -c core.whitespace=cr-at-eol diff --check` | exit 0 |
| `git status --short` | `main` and the feature branch identical at `0cd58bcd223bf1afba2d8319601e64b014a97082`; coverage artifacts ignored; `bench-logs/` ignored |
| `git diff --stat v0.5.0..HEAD` | 268 tracked paths, +32,869/−372 |
| `kindi verify` 2B/4B/9B (2026-11-12 evening) | ALL PASS, 153.7 / 108.9 / 74.6 tok/s |
| 27B tandem gate, `cpu-first` (2026-11-12) | **4/4 prompts, 40/40 each; peak 7,005 MiB; 8,192 ctx; exit 0** |
| 27B tandem gate, `gpu-first` (2026-11-12) | **4/4 prompts, 40/40 each; peak 7,017 MiB; 8,192 ctx; exit 0** |
| `python -m pytest -q tests/` (2026-11-13, `0cd58bcd223bf1afba2d8319601e64b014a97082`) | **494 passed, 19 skipped**, exit 0 |
| `python -m pytest -q tests/test_vram_monitor.py tests/test_verify_bench_contracts.py` | 9 passed (sampled-peak VRAM and verify/bench contracts) |
| `python tools/project_memory.py --check`, `build_cross_mode.py --check`, `support_matrix --check`, `config_reference --check`, `compileall`, `git diff --check` | all exit 0 |

The best-of work is merged into `main` at `0cd58bcd223bf1afba2d8319601e64b014a97082`. No push was performed. The
benchmark numbers are quoted from the repository’s recorded logs and docs except
where a run is stated.

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
   `pytest tests/ -q --cov=kindi --cov-branch --cov-report=term --cov-fail-under=27`,
   with `pytest-cov` in the test requirements. The measured local total is 27.15%
   line-and-branch coverage (462 passed, 19 skipped); the floor is deliberately
   conservative and documented as an anti-regression gate, not a quality claim.
   Requirement-based tests were added in `tests/test_requirement_coverage.py`,
   including cache-loss pick recomputation for the tandem runner. The performance
   half remains the existing opt-in GPU baseline gate.
4. **Windows quickstart.** Added `docs/WINDOWS.md`, linked from README and INSTALL: the
   verified native CPU path, clang/libomp setup, model/verify/serve commands, and explicit
   limits. Native Windows GPU/AMD execution and the Linux installer are not claimed.

Still unresolved and unchanged: the installer source commit/version publication,
9B tc, k8v4, Blackwell/FP8 hardware validation, MoE/Gemma-4 26B backend, mixed-card
and P106 hardware verification, and 4B/9B tandem-throughput measurements. The 27B
head-ratio blocker, the 27B `gpu-first` gate, and the lack of a fresh GPU/model run
were resolved as in §9 below. The 2B tandem throughput row is recorded; tandem
still does not beat GPU-only.

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
- **Tandem throughput** — 2B/4B/9B `cpu-first` have recorded numbers at the CROSS-MODE
  positions, and the 27B rate-plus-sampled-peak row followed on 2026-11-13 (§11). `bench`
  accepts `--tandem cpu-first|gpu-first --tandem-cpu-layers N` and `--sample-vram`.
- **Installer pin and version** — the installer `SOURCE_COMMIT` names
  `fd54148`, `kindi/__init__.py` still says `0.5.0`, while `main` is merged at
  `0cd58bcd223bf1afba2d8319601e64b014a97082`. Publishing and retargeting the installer are explicit release decisions
  left for the release owner.
- **9B tc** (Hadamard 4096 refused), **4B tc** (3 exact + 1 accepted tie with FD
  fallback; forced-FD 2/4), **k8v4** (2B 2/4, 9B unavailable),
  **Blackwell/FP8/BF16 validation**, **MoE/Gemma-4 26B backend**, **mixed-card/P106/5060 Ti/
  Turing/Arc hardware** — all unchanged, all the same explicit non-goals as §4.1.
- **Generated test artifacts** — `.coverage` and `coverage.json` are now ignored,
  so normal coverage runs no longer dirty the tree.
- **Tracked bench logs** — the force-added 4B/9B tandem raw logs initially leaked the
  local Windows home path in the CuPy warning line and failed `test_no_secrets.py`;
  both logs are now sanitized like the 2B log, and the full suite passes again.
- **`git diff --check` trailing CRs** — resolved: with a clean tree both plain
  `git diff --check` and the `cr-at-eol` form exit 0 (verified 2026-11-13).

## 10. Audit-remediation and harness rounds (2026-11-12/13, `082c5da..15913b6`)

Eleven commits since the `082c5da` state above. The genuinely new or better items:

- **Cross-mode builder bug fixed** — `measured_tandem == MODELS` compared a list to a
  tuple, so the generated `CROSS-MODE.md` always claimed unmeasured tandem rows remain
  `n/a` even with all three models measured. Fixed to `list(MODELS)`, matrix
  regenerated, regression test added. The CSV was always correct; the headline
  markdown intro was wrong.
- **Perf-regression gate** — `tools/check_perf_baseline.py` compares the recorded
  CROSS-MODE numbers against `docs/benchmarks/perf-baseline.json` (9 cells, 15%
  tolerance) and runs in the CPU CI job; `tests/test_perf_baseline.py` pins pass,
  regression-fail and missing-cell behavior.
- **CI static analysis** — the CUDA static-analysis harness now runs in CI
  (`static-analysis` job: cppcheck + clang-tidy-18) via a GPU-free CuPy stub for
  translation-unit building; verified locally (Turing TU builds, tools skip exit 0).
- **Pinned test dependencies** — `tests/requirements-test.txt` is bounded
  (`pytest>=7,<10`, `fastapi>=0.100,<1`, `pydantic>=2,<3`, …) with
  `tests/test_ci_config.py` guarding the pins, the static job and SHA-pinned actions.
- **Zero-coverage modules covered** — `kindi/catalog.py` (user-visible `kindi models`
  fit logic) and `kindi/sched.py` (blocking-sync opt-out), then `kindi/lookup.py`
  (prompt-lookup drafting) and `kindi/spm.py` (SentencePiece tokenizer), all with
  requirement-based tests.
- **Broad-except audit** — all 27 `except Exception` sites in `kindi/*.py` now carry
  a `noqa` rationale (best-effort probes, logged fallbacks, error-response mapping);
  no behavior changes.
- **Handoff hygiene** — `AGENTS.md` opens with a live Current-state section, the
  worktree table HEAD is guarded by `tests/test_agents_handoff.py` (it caught the
  drift twice), and `docs/ROADMAP.md` was corrected to match the landed state.
- **Secret baseline refreshed** — after reviewing every flagged line (public SHA-256
  checksums and `API_KEY=none` only), and a new local pack path in `AGENTS.md` was
  generalized so no machine-specific path is tracked.

Updated "what is worse or no longer works" list (as of `15913b6`):

- **Installer pin and version** — still `SOURCE_COMMIT=fd54148`, `KINDI_VERSION=0.5.0`;
  the published installer therefore still delivers stock-0.5-era code without tandem,
  owned-slice CPU load, or the server hardening. Release-owner publication step;
  no push has been performed.
- **Tandem still does not beat GPU-only** (2B 35.8 vs 160.6 tok/s at 24K; 4B 20.0 vs
  79.8; 9B 16.0 vs 62.3) and the **larger-than-VRAM (27B) throughput row with sampled
  peak** remains unmeasured. The 27B 8K saved-opening measurement is in flight
  (512/4K saved-opening reloads pass in both placements).
- **9B tc** (Hadamard 4096 refused), **4B tc** (3 exact + 1 accepted tie; forced-FD
  2/4), **k8v4** (2B 2/4, 9B unavailable), **Blackwell/FP8/BF16 validation**,
  **MoE/Gemma-4 26B backend**, **mixed-card/P106/5060 Ti/Turing/Arc hardware** —
  unchanged explicit non-goals (§4.1). The GPU CI lane exists but no runner is
  registered, so none of these can be verified automatically yet.
- **Rate limiting** remains process-local per source address (documented; behind a
  proxy all clients share one address).

## 11. Measurement rounds (2026-11-13, `15913b6..0cd58bcd223bf1afba2d8319601e64b014a97082`)

Six commits since §10. This section is the current statement of what is better and what is
still worse; §10 and earlier keep their own dates.

### 11.1 New or better

- **`kindi bench --sample-vram` — sampled device-wide peak.** `kindi/vram.py` samples
  `nvidia-smi` memory on a background thread and logs the peak with its idle delta and the
  phase it happened in (`--vram-sample-s` sets the interval). A host without `nvidia-smi`
  logs that and the benchmark still finishes. Pinned by `tests/test_vram_monitor.py`. Until
  now the only peak sampler lived in an ignored `bench-logs/` harness, so a long-context or
  split-memory claim could not be reproduced from the product.
- **27B rate and sampled peak, 8K and 24K, both placements plus a full-GPU baseline.**

  | ctx | engine | read | 100 | 2K | 8K | 16K | 24K | sampled peak | free at peak |
  |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 8K | tandem `cpu-first` | 10.5 | 7.2 | 7.9 | 5.6 | - | - | 7,027 MiB | 923 MiB |
  | 8K | tandem `gpu-first` | 9.4 | 7.6 | 6.6 | 5.3 | - | - | 7,027 MiB | 923 MiB |
  | 8K | full GPU | 37.2 | 34.8 | 34.3 | 33.0 | - | - | 7,473 MiB | 477 MiB |
  | 24K | tandem `cpu-first` | 10.5 | 8.4 | 8.0 | 6.0 | 3.8 | 3.0 | 7,699 MiB | 251 MiB |
  | 24K | full GPU | 37.4 | 34.9 | 34.2 | 33.0 | 31.6 | 30.2 | 7,783 MiB | 167 MiB |

  Same session, same idle baseline, exact head. The split saves **446 MiB of peak at 8K**
  and **193 MiB at 24K**, and costs **4-6x** decode throughput.
- **27B joint cache at 4K/8K.** Checkpoint 149.6 MiB (37.4 CPU owner + 112.2 GPU owner),
  54.5-61.5 ms median save and 27.0-28.5 ms median restore in both placements; saved-opening
  reloads exact (extension 8,176, branch and opening 6,144 cached); checkpoint-only peaks
  6,842/6,833 MiB at 4K and 7,030/7,027 MiB at 8K. The checkpoint does not grow with
  context, because it holds recurrent state, not K/V.
- **9B saved-opening reloads at 4K/8K — the last open cache cell.** Both placements are exact:
  4,080/8,176 extension tokens, 3,072/6,144 branch and saved-opening tokens, cache-off 0, and the
  cached continuation matches cache-off token for token. The owned-slice CPU load cuts the 9B
  start to 33.3-39.5 s (from 105-108 s), and the same eight tokens take 82-221 s cache-off against
  0.39-0.72 s cached. Evidence: `bench-logs/tandem-cache/saved-9b-4k.json`, `saved-9b-8k.json`.
- **2B CPU-layer boundary sweep**, 24,576 context, `cpu-first`: 2 layers 122.8 read / 118.9
  at 24K; 4 layers 110.5 / 41.7; 6 layers 94.6 / 39.5; 8 layers 85.6 / 22.6; 12 layers
  80.0 / 18.2 tok/s. The knee is 4-6 layers; every split stays below GPU-only (160.6 at
  24K), so the knob buys memory, not speed.
- **Tracked benchmark evidence page.** `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`
  records the 27B cache, checkpoint, rate and peak numbers with commands, so the evidence is
  reviewable in the repository instead of only in ignored logs.
- **Contract tests for two previously untested user-visible paths.** `tests/test_verify_bench_contracts.py`
  pins `kindi verify` reference routing (2B/4B/9B/27B, 2B default, `None` for an unknown
  size) and the `kindi bench` refusal of the Nanbeige backend on a CPU host; both modules
  were at 0% coverage.
- **Handoff document fixed rather than stale.** `AGENTS.md`'s live current-state block still
  named `f8c81b7` and 462 tests; it now names the real HEAD, 494 passed / 19 skipped, the new
  bench flag and the revised open gates.

### 11.2 Still worse or not working (as of `0cd58bcd223bf1afba2d8319601e64b014a97082`)

- **Installer pin and version — unchanged.** `SOURCE_COMMIT=fd54148`, `KINDI_VERSION=0.5.0`;
  the published installer still delivers stock-0.5-era code without tandem, owned-slice CPU
  load or the server hardening. Release-owner publication step; no push was performed.
- **Tandem still does not beat GPU-only, at any size.** 2B 35.8 vs 160.6 tok/s at 24K, 4B 20.0
  vs 79.8, 9B 16.0 vs 62.3, 27B 7.2-8.4 vs 34.8-34.9 at position 100. The split is exact and
  cheaper in peak VRAM; it is not faster.
- **The "larger than VRAM" premise is not demonstrated on this card.** The full-GPU 27B loads
  at 8K and at 24K (7,473 and 7,783 MiB peaks, 477 and 167 MiB free). A real capacity row
  needs a longer context or a smaller card; the documents now say this instead of implying a
  capacity win.
- **Measurement variance is not controlled.** The 2B 6-layer tandem row reads 94.6 tok/s in
  the 2026-11-13 session against 80.9 recorded on 2026-11-12 (about 17%), which is larger than
  the perf gate's 15% tolerance. All matrix rows remain single-session, single-run numbers;
  the gate compares the committed CSV with the committed baseline and cannot detect drift
  between sessions.
- **Unchanged explicit non-goals:** 9B tensor-core plan (Hadamard 4096 refused), 4B tc
  (3 exact + 1 accepted tie), k8v4 (2B 2/4, 9B unavailable), Blackwell FP8/BF16 and real FP8
  models, MoE/Gemma 4 26B backend, mixed-card/P106/RTX 5060 Ti/Turing/Intel Arc hardware,
  native Sandy Bridge/no-FMA, AVX-512/VNNI, Dory D1, Raon/Gemma 26B backends.
- **GPU CI lane has no runner**, so none of the GPU-backed gates above run automatically;
  registration steps are in `docs/CI.md`.
- **Rate limiting stays process-local** per source address; behind a proxy every client shares
  one address.
- **`/health` is still a breaking change** for unauthenticated monitors (intentional security
  fix; §4.2 item 6).

### 11.3 What was fixed in this round rather than only listed

| was | now |
|---|---|
| No product way to measure a transient VRAM peak; only an ignored harness | `kindi bench --sample-vram` with unit tests and docs |
| 27B "throughput row with sampled peak" open for two rounds | measured at 8K and 24K, both placements, against a full-GPU baseline |
| 27B saved-opening reload and checkpoint cost at long context open | exact at 4K/8K; cost and peak recorded |
| 9B saved-opening reload at 4K/8K is the last open cache cell | measured exact in both placements |
| CPU-layer boundary unmeasured ("tune n") | 2B sweep with the knee identified and documented |
| Capacity premise overstated in `ROADMAP`/`TANDEM-*` | restated with the measured full-GPU 24K peak |
| `kindi/verify.py`, `kindi/bench.py` at 0% coverage | contract tests for both |
| `AGENTS.md` live state stale by five commits | updated with the real HEAD, suite count and gates |

## 12. Tandem rows re-recorded as medians (2026-11-13)

§11.2 listed one measurement fault: the tandem rows in the matrix were single runs, and a
re-measurement came back ~17% away from the 2B row. I have now replaced them with the same
methodology as the GPU-only cells — **the median of three fresh-process runs per model**,
one session, same machine, exact head, 24,576-token context, `cpu-first`, 6 CPU layers:

| model | runs (read) | median read | median gen 100/2K/8K/16K/24K | was (single run) |
|---|---:|---:|---|---|---:|
| 2B | 77.3 / 84.4 / 84.3 | 84.3 | 79.8 / 72.3 / 58.8 / 43.2 / 34.9 | 80.9, 65.7/71.3/55.5/43.4/35.8 |
| 4B | 52.8 / 49.3 / 47.8 | 49.3 | 47.0 / 43.8 / 31.7 / 23.7 / 16.7 | 62.5, 52.1/51.4/33.5/24.8/20.0 |
| 9B | 40.1 / 38.1 / 37.0 | 38.1 | 37.5 / 34.7 / 25.4 / 20.5 / 15.2 | 39.9, 28.5/35.8/28.2/20.1/16.0 |

The single runs were off by **up to 32%** (9B at position 100) and 4B read by **-21%**.
Adopting the medians made the 15%-tolerance perf gate flag exactly those two 4B cells
(-21% read, -17% at 24K) before I re-recorded the baseline with
`tools/check_perf_baseline.py --update`; only the three tandem cells changed, and the
baseline now holds the medians. `docs/benchmarks/build_cross_mode.py` generates the new
intro (“All tandem rows below are measured on 2026-11-13, three fresh-process runs per
model; the table uses their median”) and `runs.csv` keeps one row per run. The 2026-11-12
single-run logs, with the old numbers, remain in the same folder as the record they were;
§3.1 above is updated to the adopted values.

Updated worse status: **the "measurement variance is not controlled" item from §11.2 is
closed for the tandem rows** — they now carry the same median-of-three standard as the
GPU-only rows, and the discrepancy is bounded by the observed run-to-run spread
(±5-25% across cells) rather than hidden inside one lucky invocation.

| was | now |
|---|---|
| Tandem matrix rows are single runs, up to 32% off a re-measurement | medians of three fresh-process runs per model; the perf baseline holds them and only the tandem cells were re-recorded |

## 13. Capacity and the 27B boundary (2026-11-13)

The report's remaining capacity question — a context where full-GPU cannot run but the split
can — is now answered on the 4060 Ti, and the answer is not what the project hoped.

**The loader reduces context instead of failing.** Requesting 65,536 or 131,072 tokens on the
27B loads with fewer tokens:

| requested | engine | actual context | peak | free after load | free at peak |
|---:|---|---:|---:|---:|---:|
| 24,576 | tandem `cpu-first` (16 layers) | 24,576 | 7,699 MiB | 687 MiB | 251 MiB |
| 24,576 | full GPU | 24,576 | 7,783 MiB | 463 MiB | 167 MiB |
| 65,536 | tandem `cpu-first` (16 layers) | 30,464 | 7,518 MiB | 383 MiB | 432 MiB |
| 65,536 | full GPU | 37,120 | 7,812 MiB | 23 MiB | 138 MiB |
| 131,072 | tandem `cpu-first` (16 layers) | 30,464 | 7,448 MiB | 383 MiB | 502 MiB |
| 131,072 | full GPU | 37,120 | 7,806 MiB | 23 MiB | 144 MiB |

Full GPU caps near **37K** and the 16-layer tandem near **30K**, and both snap to the same
reduced size regardless of the request, so the split buys headroom, not usable context. A truly
closing case needs a device where the full-GPU engine cannot run at a usable context yet a split
can. Evidence: `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`.

**The 27B layer boundary was swept.** At 24,576 requested context (`cpu-first`):

| CPU layers | actual ctx | read | 100 | 2K | 8K | 16K | 24K | peak | free after load |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 17,664 (reduced) | 16.1 | 14.8 | 13.6 | 9.9 | 7.6 | – | 7,723 MiB | 23 MiB |
| 16 | 24,576 | 10.5 | 8.4 | 8.0 | 6.0 | 3.8 | 3.0 | 7,699 MiB | 687 MiB |
| 24 | 24,576 | 6.8 | 5.6 | 5.3 | 3.6 | 2.6 | 2.1 | 6,103 MiB | 1,689 MiB |
| 32 | 24,576 | 5.2 | 4.6 | 4.1 | 2.9 | 2.0 | 1.7 | 5,115 MiB | 2,677 MiB |

8 CPU layers cannot hold a 24K context and are reduced to 17,664 tokens with 23 MiB free. From
16 layers the requested context holds, but each added 8 layers costs 35-50% of decode rate for
another ~1.0-1.6 GiB of peak. **16 layers is the 24K boundary** on this card; 24-32 layers are
a memory purchase at a steep rate cost, and every split stays far below full GPU (read 37.4, 24K
decode 30.2). A smaller card gets a different, measured answer with the same command; the tuning
gate from §11 is closed for the 2B and 27B on this host.

Also new in this round: **`kindi bench --json FILE`** writes a run's numbers (read rate, per-
position generation, placement, context, sampled peak) in machine-readable form, so recorded
runs no longer need log scraping (`kindi/bench.py`, `tests/test_bench_json.py`).

## 14. The 9B tensor-core plan (HB 4096) (2026-11-13)

The plan engine's HB 4096 refusal was stale: the kernels have dispatched the 4096 folds since the
27B work (`ff_rmsfold<8,8>`, `ff_smfold<8>`, `ff_fold<8>`), and only the Python gate still
refused. Lifting it and running the real verify showed:

| 9B run | result | tok/s |
|---|---|---|
| plain (KINDI_TP=0) | 4/4 | 76.3 - 80.0 |
| tc, default int8 activations | 3/4 (prompt 1 flips at token 36) | 125 |
| tc, `KINDI_TC_HP=1` | 4/4 | 122.7 |

The flip was isolated to activation precision: fold-split, FF, FD, ATT_TC and TMAX toggles all
left it unchanged and only HP (two int8 rows per token) made it exact. `kindi/spec.py` now
refuses HB 4096 unless `KINDI_TC_HP=1` (before the repack, not after), so the default tc path
keeps refusing with the measured reason and the exact path is opt-in — the same standing rule as
the quarantined `k8v4` cache and the 4B's FD-fallback caveat now documented in `docs/CARDS.md`,
`docs/CARD-SUPPORT.md` and `docs/ROADMAP.md`. No regression: 2B tc is still 4/4 (388.3 tok/s)
and the 9B tc cell now records `refused` in `tools/verify_matrix.py`. Evidence:
`bench-logs/verify-matrix-9b-tc/`, `bench-logs/verify-matrix-9b-tc-refused2/`.

## 15. Tensor-core rows in the cross-mode matrix + the 27B `gpu-first` 24K row (2026-11-13)

The plan engine is now the fourth mode of the cross-mode matrix (`cpu`/`gpu`/`tc`/`tandem`), with
three new perf-baseline cells (`tc/2b`, `tc/4b`, `tc/9b`) guarding its throughput in CI. The
measurements are medians of three fresh-process `kindi bench` runs per model at 24,576 context
with the exact head (`KINDI_TP=1`, `KINDI_TC=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`,
`KINDI_KV=q8_0`; `KINDI_TC_HP=1` for the 9B), `--sample-vram` for the sampled peak:

| model | read | 100 | 2K | 8K | 16K | 24K | peak | vs GPU-only |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2B | 471.7 | 349.1 | 327.4 | 325.2 | 319.4 | 299.2 | 1,645 MiB | 1.9x |
| 4B | 214.1 | 174.0 | 173.6 | 156.6 | 141.7 | 127.8 | 3,311 MiB | 1.6x |
| 9B | 133.9 | 114.7 | 111.3 | 108.7 | 103.2 | 97.0 | 3,257 MiB | 1.6x |

Evidence: `docs/benchmarks/2026-11-13-i7-8086k-rtx4060ti-tc/` (runs.csv + sanitized bench JSONs/logs).
The generator (`docs/benchmarks/build_cross_mode.py`) gained `tc_rows()`, a tensor-core table
section and a speedup-over-GPU-only section; `tests/test_cross_mode_benchmarks.py` now asserts
four modes.

Also measured the missing 27B matrix cell: `gpu-first` at 24,576 context reads 8.5 and generates
7.4/6.2/4.5/3.3/2.7 tok/s at 100/2K/8K/16K/24K, with a sampled peak of 7,339 MiB (633 MiB free
after load). The 27B table in `docs/benchmarks/2026-11-12-i7-8086k-rtx4060ti-tandem/27B-8K.md`
and `docs/TANDEM-RUNTIME.md` is now complete in both placements at both contexts.
