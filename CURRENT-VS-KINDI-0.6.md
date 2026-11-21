# Current fork vs official Kindi 0.6.1

## TLDR - too long didnt read

**The comparison:** official Kindi **v0.6.1** (still the latest release when rechecked 2026-11-21; upstream `main` is the release commit `7fececb`) against this fork at committed baseline **`66eebd3`** (code checkpoint `e57f6bd`). The fork is **678 commits / 897 files ahead**, with **28 numbered requirements**, **998 tracked files vs 172 upstream**, and **zero upstream file deletions**. Nine pull requests (**#47-#55**) are open upstream; together they cover **35 of the 897 changed paths**.

**Biggest measured wins**

- **Tensor-core plan path:** in the separate TC configuration, the 2B `gen @100` cell is **354.9 vs 180.5 tok/s** for stock (~1.97x); the 9B is **118.3 vs 83.8 tok/s** (~1.41x).
- **Same-settings plain decode is at parity:** every fork median is within **2.6%** of stock on the 2B and 9B, and the stock-parity check is **12/12 token-exact**.
- **27B tandem memory:** peak drops **7,721 -> 6,091 MiB** (~21%) and holds 49,152 tokens where the unsplit engine reduces to 30,976. Treat this as a memory win, not a speed win.
- **Adaptive draft policy:** **+15.8%** on the 2B (368.4 vs 318.2 tok/s) with exactness unchanged; parity on the 27B.
- **Mode A k=1 default:** **1.12-1.13x** at 512/24K, versus an explicit `KINDI_SPEC_K=4` that still costs **0.66x**.

**Biggest new features**

- **Planning and tuning:** `kindi plan`, `kindi caps`, `kindi autotune`, `kindi kld`, `kindi io`.
- **Sessions:** persistent store plus CLI and HTTP endpoints, resume, prune and stats.
- **API and constrained decoding:** backend plugin contract with adapters, OpenAI error envelope and conformance harness, JSON-schema/GBNF decoding.
- **Release and runtime:** committed **0.7.0-fork** identity, local archive builder, passing Python 3.6 container smoke, per-card tandem VRAM ceiling/reserve controls.
- **Platform:** Windows install and CPU capability paths; a bit-identical Q2_0 CPU expert-lane gate (implemented but not wired).

**Caveats:** the **0.7.0-fork** release is still unpublished (no fork images or installer assets), local `/health` still reports `commit: null`, hardware coverage is incomplete, and DeepSeek V4.1, DSpine, Nanbeige 4.3, Gemma 4 E4B, non-dense GPU grammar gates and GPU CI remain unverified.

**Upstream status (2026-11-21):** nine pull requests (**#47-#55**) are open, one commit each, all `mergeable: clean`, every title carries the CLA 1.1 statement, and the repository has **no CI check suites**; no maintainer review has landed. They cover the small self-contained fixes only. The rest of the `0.7.0-fork` surface is verified here but **not proposed**; §3a maps covered against uncovered.

Report date: **2026-11-21** (latest official release rechecked 2026-11-21). Baseline: official **Kindi 0.6.1** (tag `v0.6.1`, commit
`7fececb`, published 2026-11-16T21:49:20Z) against this fork's committed baseline, **`66eebd3`** (code checkpoint `e57f6bd`; the previous report baseline was **`ee2ba46`**, before it **`60a039b`**; the round-3 audit baseline was **`3fed704`**). The `0.7.0-fork` version string, installers and image tags are committed in this baseline. The dated speed and quality measurements below remain the `60a039b` runs; the release integration, VRAM controls and the 2026-11-21 checkpoint were not re-benchmarked for this refresh.
Upstream `main` still points at the release commit itself (`7fececb`), so there is no
post-release upstream commit left to port. This page keeps the current comparison only;
the round-by-round history has moved verbatim to `detailed-fork-changelog.md`. The 2026-11-19 engine/test/evidence round is committed through `53198e5`, the round-3 audit fixes are in `3fed704`, the 2026-11-20 integration runs through `ee2ba46`, and the 2026-11-21 checkpoint is `e57f6bd` (`66eebd3` refreshes AGENTS only).

> **Changed since the last report:** re-based from `ee2ba46` to `66eebd3`; the committed tree adds **2 commits / 28 files / +1,424/-131** (the 2026-11-20 checkpoint `e57f6bd`, then the AGENTS-only `66eebd3`). The speed and quality tables were not re-run. The new content in this refresh is the **upstream pull-request status** in §3a: nine open PRs covering 35 of 897 changed paths.

**Short answer.** This fork is **678 commits and 897 files ahead** of `v0.6.1` at the committed
baseline `66eebd3` (**+127,084/−1,417** lines), with **zero file deletions** against the official tree.
That is **two commits and 28 files** beyond the previous report's `ee2ba46` baseline. The measured advantages are the fork's own plan engine and tensor-core path, the tandem
memory split, the KLD quality gate, mode A's corrected k=1 default, `kindi plan` /
`kindi caps` / `kindi autotune`, and the session and API work. The release identity is committed:
package, both installers and image tags say **0.7.0-fork**, not official 0.6.1. The older **649**-commit figure was the `3002a40` snapshot in
`detailed-fork-changelog.md`. What remains
open is hardware evidence, the unpublished fork release, plus a few deliberately
slower opt-in paths. Nothing upstream is missing: both code changes in 0.6.1 are ported.

| question | answer |
|---|---|
| **What is different** | 678 commits / 897 files ahead at the committed baseline `66eebd3`, 998 tracked files against 172 upstream; 28 numbered requirements; no upstream file deleted. Nine upstream PRs (#47-#55) cover 35 of the changed paths; the rest is not proposed (§3a). The 2026-11-20 integration adds the committed `0.7.0-fork` release surface, local archive builder, read-only Intel probe, dense placement portfolio, process-shared token-class grammar masks, stricter request validation, autotune/parity fixes, the container-smoke and stock-parity evidence, and the per-card tandem VRAM ceiling/reserve. The earlier 2026-11-19 round adds the k=1 default, the Q2_0 expert lane, the chunk sweep and the controlled 9B pairs, plus the round-3 cold-reader teardown and quality-ratchet fixes. |
| **Better, measured** | Dense tandem + plan-owner placement; the default tensor-core plan path (2B and 9B generation far ahead of stock's plain engine, in the separate TC configuration); 2B/9B same-settings decode parity; 27B peak reduced 7,721 -> 6,091 MiB; KLD quality gate; adaptive draft policy (+15.8% on the 2B, parity on the 27B); mode A's new k=1 default (4/4 token-exact on the live 2B check); physical-core thread count. Details in §3 and §4.6. |
| **New here** | `kindi plan`, `kindi kld`, `kindi caps`, `kindi autotune`, `kindi io`; sessions store/CLI/HTTP/resume/prune/stats; backend plugin contract + adapters; OpenAI error envelope and conformance harness; JSON-schema/GBNF constrained decoding; Windows install and CPU capability paths; a bit-identical Q2_0 CPU expert-lane gate (not wired). |
| **Worse** | The release bundle is prepared locally but not published: no fork `0.7.0-fork` images or installer assets exist yet. The local `kindi:0.7.0-fork` image, rebuilt on 2026-11-20 with Python 3.6.9, passes `/health` and one non-streaming chat on the **unmodified** image; `/health` still reports `commit: null` from the `kindi/COMMIT` placeholder, and no image has been pushed. An explicit `KINDI_SPEC_K=4` still costs **0.66x**, but it is no longer the default (mode A now defaults to k=1, measured 1.12-1.13x at 512/24K); the tandem split remains a memory feature, not a speed win; FP8 K/V is lossy; recursive GBNF is expensive by design. Details in §4.1-§4.4 and `docs/RELEASE-CHECKLIST.md`. |
| **Unverified** | DeepSeek V4.1, DSpine, Nanbeige 4.3, Gemma 4 E4B, non-dense GPU grammar gates, a complete Windows install, two-card TP/mixed cards, the five legacy cards, Arc A770, Intel iGPU, AVX-512/AMX, the calibration >=20 s clause, the Q2_0 expert-lane model row, and GPU CI. Details in §4.3. |

## What changed since the last report update

The last report update was `b04cca8`, which still used `60a039b` for its counts. Since `60a039b`, the committed tree has **7 commits, 161 files, +19,341/−917** and **63 more tracked files**:

| commit | change |
|---|---|
| `1627b96` | widened claims coverage to the six registered docs |
| `82a72bf` | landed the `0.7.0-fork` runtime, planning and evidence: release surface, local archive builder, read-only Intel probe, dense placement portfolio, process-shared token-class grammar masks, stricter request validation and autotune/parity fixes; tracked the container-smoke and stock-parity evidence |
| `a2c450f` | recorded the final integration handoff |
| `b04cca8` | pinned the local archive to the integrated tree |
| `20e2ea0` | refreshed the handoff and kept it out of release archives |
| `3b9224a` | finalized the `0.7.0-fork` release pins |
| `ee2ba46` | added the per-card dense-tandem VRAM ceiling and display reserve |

The report numbers move with the baseline: **669 → 676 commits**, **827 → 891 files**, **+107,320/−1,370 → +125,781/−1,407**, **929 → 992 tracked files**. The substantive changes since the last report are the now-committed release identity and archive pins, the Python 3.6 container fix with a passing unmodified-image `/health` + chat smoke, the tracked stock-parity and controlled-pair evidence, and the new `--vram-budget-mib` / `--vram-reserve-mib` controls. The speed and quality table below was measured at `60a039b` and was not re-run for this refresh.

**2026-11-21 checkpoint and PR refresh.** Two commits follow `ee2ba46`: `e57f6bd` (28 files, +1,424/−131) is the 2026-11-20 checkpoint, and `66eebd3` refreshes the handoff. Totals move **676 → 678 commits**, **891 → 897 files**, **+125,781/−1,407 → +127,084/−1,417**, **992 → 998 tracked files**. This refresh adds no measurement; its new content is the upstream pull-request status in §3a.

## 1. Baseline

Official upstream: [**Kindi 0.6.1**](https://github.com/ahtx/kindi/releases/tag/v0.6.1), tag `v0.6.1` (annotated tag `116532a`, peeled to
`7fececb`, 2026-11-16), GitHub release published 2026-11-16T21:49:20Z. The GitHub
release API was checked again on 2026-11-20; **v0.6.1 remains the latest official
release**. `git ls-remote origin refs/heads/main` returns `7fececb` on 2026-11-20, the
release commit itself, so there is no post-release upstream commit left to port; `v0.6.0` (`179710c`) is still
an ancestor of `main`, and the shared base remains `dce89df`. The comparison range is
therefore `v0.6.1..66eebd3`. **`66eebd3` is the committed baseline**: the range and the
figures below use that commit's tree (the code checkpoint is `e57f6bd`; `66eebd3` refreshes
AGENTS only). The `0.7.0-fork` version string, installers and image
tags are committed as part of the 2026-11-20 integration (`82a72bf`..`3b9224a`); the local
image was built from that working tree, not an older release tree. Section 4 keeps the dated
`3fed704`/`ee2ba46` analysis and says so. The 2026-11-20 commits since `60a039b` are listed
above; `detailed-fork-changelog.md` §44-§46 list the earlier rounds.

```sh
git merge-base --is-ancestor v0.6.0 HEAD       # yes
git merge-base --is-ancestor v0.6.1 HEAD       # no: v0.6.1 is one commit past dce89df
git merge-base v0.6.1 HEAD                     # dce89df
git log --oneline v0.6.1..66eebd3 | wc -l       # 678 commits
git diff --shortstat v0.6.1..66eebd3           # 897 files, +127,084/-1,417
git diff --diff-filter=D --name-only v0.6.1..66eebd3  # (empty: zero file deletions)
git ls-tree -r --name-only v0.6.1 | wc -l      # 172 upstream files
git ls-tree -r --name-only 66eebd3 | wc -l     # 998 tracked files at the committed baseline
ls docs/development/requirements/REQ-*.md | wc -l  # 28 numbered requirements
```

The published `v0.6.0..v0.6.1` range is **16 files, +336/−43**: it includes
`dce89df`'s container packaging commit and `a6f4891`'s README acknowledgement as
well as the 0.6.1 commit. The 0.6.1 commit itself (`dce89df..v0.6.1`) is **15 files,
+52/−36**: `compose.yaml`, the three Dockerfiles, `docs/CONTAINERS.md`, the
Docker/install docs, README/NOTICE/CHANGELOG, the 0.6.1 version strings,
`install.sh`/`install.ps1`, a graph-build diagnostic in `kindi/spec.py`, and one
real code fix in `kindi/mtp.py`. It does not change `compose.amd.yaml`; that came
with `dce89df`. Both 0.6.1 code changes are in this fork (`f934a55`): the MTP
`hp` argument and the graph diagnostic.

The 0.6.0 merge landed on 2026-11-14 (commit `f635513`, 18 conflicted files over
172 diverged upstream commits) and has been carried forward since. The tree vs
`v0.6.0` is a **strict superset**: every upstream file is present, and the
zero-deletion check against `v0.6.1` shows none of its paths was dropped either.
A line-count drop in a shortstat is edits inside files, not file deletions.

The 2026-11-19 round is committed through `53198e5`; the round-3 audit fixes are in `3fed704`. `f75253b` adds DSV4.1 request-shape
validation and the batched Windows secret scan; `84b6686` commits the controlled 2B/4B/9B pairs,
the 9B chunk sweep, the context-growth and ternary-MoE pages and the Q2_0 expert-lane/combine
evidence. The lane module is `tools/moe_expert_lane.py`, an experiment that is not wired into the
engine; the mixed card/CPU combine contract is separate and exact.

Full suite at the current baseline: **1,420 passed, 28 skipped, 0 failed** (recorded at
`ee2ba46`; not re-run for this refresh); the earlier `60a039b`
**1,391** and `f75253b` + `84b6686` **1,357** runs are historical. On 2026-11-21 the generated
config reference was found stale (missing `KINDI_TANDEM_HEAD_SPLIT`, `spec.py` line drift); it
was regenerated and `tests/test_config_reference.py` is **8 passed**. The generated-artifact
gates (`project_memory --check`, `config_reference --check`, `support_matrix --check`) are
clean. The skip count is the hardware/model/clang-only set, so the CPU and stub-engine
surface is the part measured; the GPU and model paths were not re-run for this suite.
`python tools/check_claims.py` re-derives **37 gates / 265 numbers** from tracked evidence across six docs.
A previously clean coverage run at `539a504` remains **1,241 passed / 28 skipped / 0 failed**.
The preview chips are computed at render time from the committed baseline and its shortstat, and show the working-tree delta separately when it differs.

## 2. Upstream 0.6.x features present here

| 0.6.0 item | in this tree | status here |
|---|---|---|
| MoE backend (`kindi/moe*.py`, `moe_cpu.c`) | present (merged) | **exercised on GPU through Lightning (5 exact + 7 ties, 0 fail; `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`)**, with no separate expert-tier run; the fork adds expert-tier *planning* (`kindi plan` MoE strategy, `kindi/moe.py plan()`); small-MoE generalization tracked in §5 |
| DeepSeek-V4.1-Flash backend (`kindi/dsv41*`) | present (merged) | **GPU-unverified here** |
| Nemotron Lightning (`kindi/lightning*`) | present (merged) | **GPU-gated here 2026-11-16**: the T4.2 adapter is ported, and a clean catalog GGUF (size/SHA-256/GGUF verified, fresh pack) verifies 5 exact + 7 llama.cpp ties, 0 fail on the RTX 4060 Ti, with bench ctx 8192 read 788.3–821.3 and write 165.96–167.41 tok/s. `kindi plan` describes its tiering header-only |
| **DSpine drafter** (`kindi/dspine.py`, `kindi/dspine.cu`) | present (merged); `spec.py` hooks wired | **GPU-unverified here**; fork wraps the build in `spec._optional` so a broken DSpine degrades to no-speculation instead of aborting startup |
| Container entrypoint + compose, Kindi Bridge, `install.ps1` | present (merged) | `install.ps1` has one real Windows run (installer steps only; Docker/model/request skipped, §4.3) |
| Turn-save cache (`turn_keep`, `turn_disk_mb`), priority scheduler, `/v1/tokenize` + `/detokenize`, `dry_run` | present (merged) | CPU-tested; the Windows run covered the installer, not this path (§4.3) |
| New card classes (M5000 sm_52 fix, Lightning offers, DSpine profile) | present (merged) | rig-measured cards remain `not measured` except the 4060 Ti (§4) |

## 3. What this fork has *beyond* stock 0.6.1 (better, verified)

| fork item | evidence |
|---|---|
| Dense tandem runtime (2B/4B/9B/27B, plan-engine GPU owner, GDN-first `gdn:N`, `auto`/`rate` planners) | `docs/benchmarks/`; plan owner cut the 27B 24K peak 7,721 → 6,091 MiB and holds 49,152 tokens where the unsplit engine reduces to 30,976 |
| `kindi kld` quality gate | q8_0 2.07–2.74e-07 nats, k8v4 quarantined on numbers (~470–1,090× q8_0); `docs/benchmarks/2026-11-14-i7-8086k-rtx4060ti-kld-2b/` |
| Calibration cache (`kindi/calib.py`, `kindi calibrate`) | 27B parity ≤0.3% fresh-tuned vs cached; the ≥20 s startup clause needs a HIP rig (g5 tuning is 0.2 s on NVIDIA) |
| `kindi plan`: dry-run placement report (dense + MoE tiers) | live agreement 12 layers both ways on the 4060 Ti |
| Physical-core thread count | 104.6 vs 86.1 tok/s at 6 vs 12 threads (measured, not assumed) |
| Parked K/V prefix blocks shared once and refcounted | prior-art #6 mechanism; restores byte-identical |
| Legacy five-card + triple-server hardening | `install.sh --gpu 0,1,2`, M5000/RTX 4000 registry rows, `profile_of()` W=2 guard; no physical card runs, so no new performance claim |
| FP8 host reference codec (`kindi/dtypes.py` `f32_to_fp8`) | round-to-nearest ties-to-even, all-pattern round-trip tests |
| **FP8 E4M3 K/V decode cache** (`KINDI_KV=fp8`, REQ-0014) | implemented decode-only 2026-11-15 and **measured lossy**: KLD 2.48e-06 (2K) / 2.80e-06 (8K), ~9–14× q8_0; verify 3/4; same cache size as q8_0 (8.5 bits/value). Experimental opt-in, not a default |
| **Adaptive draft policy** (`kindi/draft_policy.py`, REQ-0015) | 2B GPU gate 2026-11-14: 368.4 vs 318.2 tok/s (+15.8%), exactness unchanged; **27B gate 2026-11-15: fixed-k median 40.1 vs policy-on 40.1 tok/s, identical exactness** (3 exact + same tie). Token stream is verify-only, so exactness cannot change by construction |
| **Tool-call conformance suite** (`tests/test_tool_call_conformance.py`, REQ-0018) | 18 tests over Gemma 4 native, Nanbeige/Lightning XML+JSON and DeepSeek DSML truncated blocks (2026-11-15); all parsers hold the contract, no leak bugs found. 23 tool-call tests total pass |
| **Backend plugin foundations** (`kindi/backends/`, REQ-0019) | T4.1 contract/registry/FakeBackend + T4.3 `kindi backends [--json]` listing (2026-11-15); CPU-only, importable on Python 3.6. T4.2 adapters for all five families landed 2026-11-16; Bonsai is GPU-gated on all four packs and Lightning for plain verify/bench on the 4060 Ti; Nanbeige, Gemma 4 and DeepSeek V4.1 are CPU-tested with blocked GPU gates — so still no arbitrary-GGUF claim |
| **OpenAI error envelope + API conformance harness** (`kindi/api/errors.py`, `tests/api/`) | T5.1 2026-11-15: every server error is the OpenAI envelope; harness drives the real app with the official `openai` client, GPU-free |
| **Intel Arc A770 purchased** (REQ-0020) | one A770 16 GB bought 2026-11-15, in transit (not installed); M0 audit done, M1 arrival checklist ready, `arc-a770` registered as `not measured`. No backend code, no support claim |
| **Intel iGPU G0 slice** (`kindi/intel.py`, REQ-0021) | 2026-11-15: DRM-sysfs discovery, oneAPI-floor eligibility, shared-memory budget, tandem GPU contract; `KINDI_DEVICE=intel` refuses loudly. G1+ SYCL kernels need an 11th-gen+ iGPU + oneAPI to compile/verify |
| **Tandem worker-thread verdict** | step-0 re-measured 2026-11-15 (27B/24K, RTX 4060 Ti + 8086K): CPU 74.7 ms/token (6.2 ms/layer) at raw-kernel rate; head 17.6 ms + transfer 3.09 ms come first. **Worker thread not justified, should not be built** |
| **4B conversion provenance** | fresh `kindi convert` of the downloaded PTQ1_0 GGUF reproduces the 4B pack byte-identically (sha256); converted pack verifies 3+tie like the reference; 9B/27B pack verify re-run 4/4 (2026-11-15) |
| Requirement-based coverage + per-module floors | `tools/check_coverage.py` + `tests/coverage_floors.json` in CPU CI (audit M4); requirement-traceable CPU tests |
| **Server-side `/sessions` endpoints** (`kindi/server.py`, REQ-0016) | 2026-11-15: `GET /sessions` (status/tag filters), `GET /sessions/<id>`, `GET /sessions/<id>/export` (raw markdown), `POST /sessions/<id>/status`, `POST /sessions/<id>/tags` over the existing `SessionStore` — host-only (no engine/cache/GPU), API-key gated, enveloped 400/404, strict 12-hex id check so path-shaped ids never reach a file read. Pinned by `tests/test_server_sessions.py` (stub engine, no GPU) |
| **Loud rejection of unsupported requests** (`kindi/server.py`, REQ-0018) | `n` values other than 1 return an enveloped 400 before the engine or prompt cache is touched; structured output is constrained on all four served families (dense and Gemma 4 GPU-gated; Lightning/Nanbeige CPU end-to-end, their GPU grammar gate still open); malformed schemas, `grammar`, and `tools` combined with structured output still return enveloped 400s. Pinned by `tests/test_server.py` and `tests/test_jsongrammar.py` |
| **Config reference scans subpackages** (`kindi/config_reference.py`) | 2026-11-15: `env_vars()` scans `kindi/**/*.py`, so env vars read in subpackages (`KINDI_BACKEND_MODULES` in `kindi/backends/`) appear in the generated reference with subpackage site paths. Audit triage of `Buffy_ReviewAudit_202611141517.txt`: the rest of its findings (N1-N6, D1/D3/D4, M3/M4) were verified fixed on the current tree; M1/M2 stay hardware-bound |
| Audit dispositions (2026-11-15) | the 202611150853 audit verified item-by-item on the current tree: handoff-guard findings already fixed, coverage floors already landed, hardware items already recorded unverified; hash-bump hook declined by design, fork gap unfixable (push 403) |
| Audit fixes: optional-accelerator degradation, `kindi_cache_*` telemetry, reference-config section | 2026-11-14 audit round (N1/N3–N5/D3/D4) |
| **`kindi caps`: per-card and per-CPU capability report** (`kindi/caps.py`, REQ-0022) | 2026-11-17: nvidia-smi → pynvml → CuPy layered identification (name, UUID, PCI bus, compute capability, total/free VRAM, driver), hardware facts kept apart from the Kindi paths they enable, unknown capacity reported as unknown; CPU half reuses the engine probe plus model name, cores, L2/L3, AVX-512/AMX and the CPU-build toolchain. On Windows the AVX/AVX2/AVX-512F flags come from `kernel32.IsProcessorFeaturePresent` (no compiler needed). `tests/test_caps.py` 19 tests; live run on the 4060 Ti rig |
| **`kindi autotune`: measured placements** (`kindi/autotune.py`, REQ-0022) | 2026-11-17: candidates from the hardware report + pack manifest (full card, two-card TP split, tandem `auto`/`gdn:N`, Lightning decode cache, CPU), each one a real `kindi bench --sample-vram` child run, ranked by decode tok/s with reduced-context runs ranked below full-context ones and failed runs unscored. Live 2B/ctx-8192 run: 97.4–97.6 tok/s decode, 111.7–113.0 read, 3,050 MiB peak; `tandem-auto` refused as unnecessary and `tandem-gdn:8` blocked by a missing compiler, both reported as such. `tests/test_autotune.py` 18 tests |
| **Image input is refused, not silently dropped** (`kindi/server.py`) | 2026-11-17: a pack manifest that declares `vision`/`mmproj` no longer unlocks acceptance — no build implements a projector, so an image part or Ollama `images` is always an enveloped 400, and the message names the missing projector when a pack claims one. Pinned by an app-level test built on a vision-declaring manifest |
| **One grammar compile per structured request** (`kindi/server.py`) | 2026-11-17: validation already built the automaton to answer an invalid schema/grammar with a 400; the decode path built it again. The compiled object is now parked on the request and reused (GBNF and `json_schema`), so a structured request compiles once: llama.cpp's canonical `json.gbnf` at nesting 2 is 7,936 states / 7.8 ms median build, ~15.6 → ~7.8 ms per request. Pinned by a call-counting test |
| **GBNF size honesty** (`kindi/gbnf.py`) | 2026-11-17: `Grammar.states` exposes the compiled size, and the over-budget refusal names `*`/`+` and `json_schema` instead of only a state count. Measured: exact per-use expansion costs ~143 states per enumerated copy (a 40-pair record referencing `ws` 82 times is 5,915 states / ~6 ms), the loop form is ~330 states at any length, and the default 20,000-state cap holds ~130 enumerated copies. The pre-`221d652` rule cache was 20× smaller because it widened the language (an "exactly 3" grammar accepted 0–5 copies) |
| **Python quality ratchet** (`tools/check_python_quality.py`, audit M6) | 2026-11-18/19: Ruff 0.16.0 (`F`, `E9`, `B`, `I`) over the Python tree, mypy 2.4.0, Pyright 1.1.414 (`basic`, all platforms) and Pylint 4.1.3 (`E`, `F`, import-error off) over production code, currently **315 Ruff / 59 mypy / 1,035 Pyright / 255 Pylint** findings, against a `tools/python_quality_baseline.json` that allows 321 Ruff; CI fails only on a new or increased finding, so existing debt is measured rather than hidden. `tests/test_python_quality.py` + `tests/test_ci_config.py` pin the comparison, parser shape and exact tool versions. Dependency scanning stays open |
| **Every standalone CLI validates before a model is mapped** (`kindi/dsv41*.py`, `tools/*`, 2026-11-19) | `python -m kindi.dsv41`, `dsv41_server`, `dsv41_dspark` and the gate/harness tools now require readable inputs, a writable output parent, a complete DSpark directory, token ids in range and finite/positive counts at argparse time; commands that need `--prompts`/`--ref` fail with their required set first. `tests/test_dsv41_cli.py` (24 tests) pins the parsers and every pre-load exit |
| **DSV4.1 server body cap** (`kindi/dsv41_server.py`, 2026-11-18) | the standalone server shares `KINDI_MAX_BODY_MB` (16 MiB), refuses an over-cap body with 413 `request_too_large` before reading it, answers 400 for a malformed length/JSON, and half-closes plus drains the refused body so the 413 cannot be lost to a reset; a socket-level test sends a 1 MiB body past a small cap followed by a valid request |
| **Cold-expert slots stay pinned across decode steps** (`kindi/cold_experts.py`, 2026-11-18) | the decode hit test now uses a per-layer slot table from `cold_slot_plan`: a repeated expert is served from its unchanged slot, a routed expert's slot is never stolen, and `cold_stats()` reports `slot_hits`. Unit-tested only; the per-layer stream sync and blocking current-layer read stay open (C9), with no GPU, larger-than-RAM pack or tok/s claim |
| **Mode A now defaults to its measured k=1 win** (`kindi/spec.py`, 2026-11-19) | explicit `KINDI_SPEC=mtp` now defaults to `KINDI_SPEC_K=1`; the 2B path is 4/4 token-exact, 160/160 tokens with 65/94 drafts accepted on the live ctx-1024 check. k=4 remains available but no longer silently selected; the other drafters keep k=4 (`docs/MTP.md`) |
| **Controlled 9B CPU-only vs CPU+GPU tandem pairs** (2026-11-19/20) | three controlled `cpu-first` six-layer pairs at ctx 24,576: three-run median CPU-only **2.4** vs tandem **14.8** tok/s at 24K (**6.2x** over CPU-only), **4.1x** at 100 with runs spanning **3.0x-4.2x**. Run 3 after a passive >7 h cooldown reproduced run 2's tandem row within 8% and its CPU row within 4-10% except CPU-only 24K (1.8 -> 2.4, +33%), so run 2 was not a run-wide outlier; run 1 is the high side. Run 1 verified both placements 4/4, 133/133; runs 2-3 were throughput only. Corrects the old cross-run shallow-context 4.3x; the split stays below the full-GPU 9B row and buys headroom only. `docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-9b-controlled/` |

## 3a. Upstream pull-request coverage (2026-11-21)

Nine pull requests are open against `ahtx/kindi` from this fork. Each is one commit based on
`v0.6.1` (`7fececb`), each reports `mergeable: clean`, and each title carries the CLA 1.1
statement. The repository has **no CI check suites**, and no maintainer review has landed.
Together they touch **35 unique paths** of the **897** changed paths; the other **862 paths
are not proposed**.

| PR | proposal | head | files | lines | focused evidence |
|---:|---|---|---:|---:|---|
| #47 | load the CUDA driver on native Windows | `a556ef3` | 5 | +48/−7 | `tests/test_gpudrv.py` 3 passed |
| #48 | cap host RAM accounting at the container cgroup limit | `17d4116` | 2 | +152/−4 | `tests/test_hostram_limit.py` 10 passed |
| #49 | reject an inconsistent tokenizer pack | `55fb576` | 2 | +29/−1 | `tests/test_tokenizer.py` 1 passed |
| #50 | close pack/cache JSON handles before returning | `5c2797c` | 6 | +73/−18 | `tests/test_jsonio.py` 2 passed |
| #51 | sample device-wide VRAM peak in `kindi bench` | `ee2cdc1` | 4 | +241/−1 | `tests/test_vram_monitor.py` 7 passed |
| #52 | measure host memory tiers with `kindi io` | `a79cd28` | 3 | +579/−1 | `tests/test_io_probe.py` 13 passed |
| #53 | read-only Intel probe and loud `KINDI_DEVICE=intel` refusal | `21f731c` | 4 | +896/−1 | `tests/test_intel.py` 12 passed |
| #54 | GBNF + JSON-schema + token-class grammar engine | `c412b79` | 8 | +2,186/−3 | 58 passed; llama.cpp fixture source/license in `NOTICE` |
| #55 | opt-in adaptive draft-length policy | `e880da2` | 4 | +194/−0 | 8 passed; tensor-core spec path only |

**Covered by that set:** Windows driver loading, the cgroup RAM cap, tokenizer pack consistency,
JSON handle closure, sampled VRAM peaks, the host memory-tier probe, the Intel probe/refusal, the
GBNF/JSON-schema/token-class engine, and the adaptive draft-length policy. Two limits matter:
#54 ships the grammar **engine only** (no request path calls it yet), and #55 runs only on the
tensor-core spec path (`tc_step`, `KINDI_TC=1`).

**Verified in the fork but not covered by any PR:**

| workstream | recorded evidence | why it is not a PR yet |
|---|---|---|
| `kindi caps` + CPU feature/tier report | `tests/test_caps.py` 20 passed; live i7-8086K report (`AVX=yes, AVX2=yes, AVX-512=no`) | needs the `feature_report`/`cpu_tier`/`select_isa` block from `kindi/cpu.py`; that file's +603-line diff also carries multi-ISA dispatch, so split it |
| `kindi plan` + `kindi autotune` + `--portfolio` + per-card VRAM ceiling/reserve | `test_plan.py`/`test_autotune.py`; 27B 24K peak 6,091 MiB; `--vram-budget-mib`/`--vram-reserve-mib` | ~1,300 lines across `plan.py`, `autotune.py`, `ltfit.py`; the planner contract needs maintainer design feedback |
| `support_matrix` + `config_reference` generators | `test_support_matrix.py`; `test_config_reference.py` 8 passed after the 2026-11-21 regeneration | self-contained AST scanners; the generated reference was stale at the checkpoint and had to be fixed first |
| backend plugin interface + five adapters | Bonsai gate closed 2B/4B/9B/27B; Lightning gate closed; Nanbeige/Gemma 4/DSV4.1 adapters CPU-tested | ~2,000 lines across registry and adapters; three GPU gates still blocked by packs |
| server/API hardening | `test_server*`, session tests; bounded queue/429, rate limit, JSON access log, `/metrics`, body caps, OpenAI error envelope, sessions | +1,408 lines in `server.py` alone; split by endpoint family |
| speculation harnesses | `verify --spec`, `bench --spec`, `tools/verify_matrix.py`, `tools/spec_ab.py`; 12/12 byte-identical A/B | depends on `spec.py` and model packs; #55 covers only the policy |
| grammar server wiring | `kindi/constrained.py` + server `response_format` tests | #54 ships the engine only; wiring is a separate PR after it |
| dense tandem + joint cache + owned-slice CPU load | `test_tandem.py`; 2B/4B/9B/27B exactness, rate and peak rows | measured conclusion is “tandem does not beat GPU-only”; needs a maintainer capacity decision |
| MoE expert cache / cold-expert tier | `test_expert_cache.py`/`test_cold_experts.py`; prefetch, slots and byte accounting | unit-tested only: no larger-than-RAM pack, GPU run or tok/s claim |
| model backends (Gemma 4, Nanbeige, Lightning, DSV4.1) | family tests, CLI/server validation, Lightning GPU gate | packs absent or corrupt; each backend is a large PR on its own |
| CPU/TC/FP8 kernels | multi-ISA dispatch, AVX2/F16C fallback, Turing TC, FP8 `mma.sync` tests | no AVX-512 host, no real FP8 model and no Blackwell run |
| quality and evidence tooling | 107 test files; 1,420 passed recorded; coverage/perf/claims/secret/quality gates | baselines are repo-specific and need adaptation and review |
| release and docs surface | `0.7.0-fork` identity, installers, archive builder, 616 doc files | depends on the code PRs; docs-only PRs would conflict heavily |

§4.3 lists the work that is not verified at all. The table above lists only work with recorded
tests or measurements, so it is an inventory of what could be proposed next, not a promise that
each item should be.

## 4. What is worse, missing, or unverified vs official Kindi 0.6.1

Re-derived at the committed baseline `ee2ba46` (2026-11-20) against official `v0.6.1` (`7fececb`,
published 2026-11-16), with `v0.6.0` (`179710c`) and the shared base `dce89df` kept for the release
ancestry. The 2026-11-21 checkpoint (`e57f6bd`) is test/runtime work and does not re-run these rows. These figures are the committed tree's, not the working tree's. The speed and quality
measurements were taken at `60a039b`; the seven later commits add the release integration, evidence,
claims and VRAM controls, not a re-run of the benchmark matrix. The 2026-11-19 analysis was taken at
`3fed704` (666 commits, 827 files, `+107,317/−1,370`). The
title folds three different questions together, so they are separate below; each row ends with what
would move it, and the linked benchmark pages and `docs/MTP.md` hold the dated runs behind them.

**The short answer.** Nothing upstream is deleted or unported: every file of `v0.6.1` is still present
(`git diff --diff-filter=D --name-only v0.6.1..66eebd3` is empty), and the release's two code changes are
in `f934a55`. The official release publishes `0.6.1-cuda`, `0.6.1-rocm6` and `0.6.1-rocm7` plus the
source and installer hashes. This fork now prepares its own `0.7.0-fork` package, installer and
container tags, explicitly marked as a fork release rather than official 0.6.1. What remains
open is **evidence**: measurements that need weights, a
second card, another ISA or a clean Windows box. The only measured regression against the simpler path
is mode A speculation at an explicit k=4, and it is no longer the default (mode A now defaults to k=1); the other §4.1 rows are deliberate trade-offs.

### 4.1 Worse than the simpler path (measured; not a regression against stock)

| surface | measured here | why it stays | what would move it |
|---|---|---|---|
| Mode A at explicit `KINDI_SPEC_K=4` (opt-in; the default is now k=1) | **0.66x** at k=4 on the 2B / RTX 4060 Ti at 24K: 361.2/363.7 tok/s plain vs 236.9/239.6 with the head, 378 of 1,680 drafted tokens accepted (22.5%); forcing the guard on from position 0 gives 230.9/247.2/246.6. At the current default k=1 it measured 1.12–1.13x at 512/24K and 1.01x at 8K, and a live default-path check passed 4/4, 160/160 tokens | the head's graph had not built since 0.5.0 (fixed in `f934a55`; details in `docs/MTP.md`), and the old k=4 setting loses: the first guard windows are speculation windows, so a short reply pays the loss first | the current default follows the winning k=1 row; k=4 stays opt-in for measurement. `docs/MTP.md` |
| Tandem CPU+GPU split on one card | 24K decode **~8 vs ~31–34 tok/s** against the equivalent full-GPU rows; step-0 is 17.6 ms/token of head plus 3.09 ms of boundary transfer, ahead of any worker thread | it is a **memory** feature: a 49,152-token request keeps all 49,152 where the unsplit plan engine reduces to 30,976 | a second card or a much faster CPU head, not a code fix; `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-tandem-9b/RESULTS.md` |
| 9B tandem `gdn:8` / `gdn:16` | saves **486–488 MiB** for 2.8–3.0x decode, or **872–874 MiB** for 4.5–4.6x, flat from 8K to 131,072, while full GPU already holds every request at 5,149 MiB peak | buys headroom, never context; the 27B is the only pack here that needs the split | a pack that does not fit full-GPU; otherwise no split |
| FP8 E4M3 K/V (`KINDI_KV=fp8`) | KLD **2.48e-06** (2K) / **2.80e-06** (8K), ~9–14x q8_0, verify 3/4, same cache size as q8_0 (8.5 bits/value) | experimental opt-in, never a default | a lossless 8-bit format; keep it off otherwise |
| Recursive GBNF (`KINDI_GBNF_MAX_DEPTH`) | opt-in and bounded: llama.cpp's `json.gbnf` reaches array nesting 2 inside the 20,000-state cap, and depth 7 is 398,268 states; `json_schema` is the unbounded path | exact expansion is what keeps the accepted language honest (the old rule cache was 20x smaller by widening it) | shared states or a token-class mask (REQ-0027) |

### 4.2 Missing vs official 0.6.1

| item | official release | here | consequence / closure |
|---|---|---|---|
| Version identity | `0.6.1`; `install.sh` sha256 `31c4a860…` | committed at `ee2ba46`: `kindi/__init__.py`, both installers and the image tags say `0.7.0-fork`; `install.sh` pins the fork source archive hash | a fork release, not an engine change; the bundle is prepared but not published (`docs/MANUAL-HELP-NEEDED.md`, `docs/RELEASE-CHECKLIST.md`) |
| Container packaging | `ghcr.io/ahtx/kindi:0.6.1-cuda`, `0.6.1-rocm6`, `0.6.1-rocm7`; compose/Docker docs updated | the fork's compose files and three Dockerfiles name `0.7.0-fork` tags under `ghcr.io/atimelessrage/kindi`; the Python 3.6 blocker is fixed and the rebuilt local CUDA image (`sha256:b08916d7…`, Python 3.6.9) passes the **unmodified-image** `/health` (200) and one chat (200 in 0.2337 s), with GPU passthrough and the 2B pack load; `/health` still reports `commit: null` from the `kindi/COMMIT` placeholder | no fork images are published; the local smoke is not an official 0.6.1 image-parity claim (`docs/RELEASE-CHECKLIST.md`) |
| Upstream code | `kindi/mtp.py`'s `hp` argument and `kindi/spec.py`'s graph diagnostic | both ported in `f934a55`; the remaining v0.6.1 changes are version strings, docs, `compose.yaml`, the Dockerfiles and install scripts | nothing to port; no file of the `v0.6.1` tree is missing here |
| Not-behind-stock absences | stock lacks them too | dense standard-attention llama/qwen engines; an image projector (an image part is an enveloped 400); the llama.cpp proxy (REQ-0019 T4.4) and real concurrency (T4.5), whose positioning is an open maintainer call (MANUAL-HELP §6); Metal | listed so "missing" is not read as "behind stock" |
| MoE expert-cache tandem | stock had no such path | **not a backend**: `kindi/expert_cache.py` is a model-free slot/LRU prototype, with no router, pack, GPU copy path or measured inference (`docs/MOE-TANDEM.md`). The 2026-11-19 Q2_0 CPU expert-lane gate is bit-identical to `lt_mv_q2`/`lt_mv_q2_down` (0 differing elements with device-resident or host-mapped weights), but the CPU expert is **2.395 ms**, against **0.783 ms** for the copy lane (**3.06x** faster) and **0.662 ms** for the engine's direct host-mapped GPU read (**3.62x** faster than the CPU lane; **1.18x** faster than the copy lane), and the lane is not wired into `lightning.Engine`, but the mixed card/CPU combine contract is implemented: `Q2Matrix.partials`, `lt_mv_q2_down_parts`, `lt_q2_shared_parts` and `lt_q2_combine` match `lt_mv_q2_down` bit for bit in the synthetic combine gate (**0 differing elements**); no Lightning pack exists here, so there is no model row | Raon-1 mini R0 reference goldens or the Gemma 4 26B contract; the host-mapped GPU read wins on this host, so the CPU lane is not the default. `docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-expert-lane/`; combine evidence `docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-q2-combine/` |

### 4.3 Unverified (in the tree, no recorded run)

| # | surface | why there is no run here | what closes it |
|---|---|---|---|
| 1 | DeepSeek V4.1 Flash backend | shards and gate data are not on this host | weights + `kindi verify` on the 4060 Ti |
| 2 | DSpine drafter | no dspine profile or weights; `kindi verify --spec dspine` exits non-zero when the drafter does not attach | `docs/MANUAL-HELP-NEEDED.md` §7 |
| 3 | Nanbeige 4.3 33B | the served GGUF is corrupt (full size, wrong sha256, NUL-garbled tensor names) | a fixed upload, or bartowski's quant with regenerated references (MANUAL-HELP §1) |
| 4 | Gemma 4 E4B pack | pack not on this host | download, then the adapter's GPU gate (MANUAL-HELP) |
| 5 | Structured output on the non-dense families (Gemma 4, Lightning, Nanbeige) | all four families pass the CPU end-to-end gate, but only the dense 2B has had the GPU grammar gate; Lightning's GPU verify/bench gate is closed (5 exact + 7 ties, 0 fail; read 788.3–821.3, write 165.96–167.41 tok/s at ctx 8192; `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`) | the same grammar gate that closed for the dense 2B, on each family's GPU path |
| 6 | `install.ps1` complete install | a real run on 2026-11-17 covered PC checks, rootfs download with a SHA256SUMS match, `wsl --import`, systemd, the GPU check and the Start-menu/uninstall entries (94 s); Docker integration, the model download and a served request were skipped | a clean box with Docker; then `/health` and one chat completion |
| 7 | `kindi autotune` two-card and MoE candidates | one card and no Lightning pack; `tp-2-*` and `dcache-*` are unit-tested only | a second card and a Lightning pack; a placement-changing winner still needs `kindi verify` |
| 8 | Five legacy cards, Arc A770, Intel iGPU | no physical cards (the A770 is in transit); iGPU G1+ needs an 11th-gen+ iGPU and oneAPI | MANUAL-HELP §4 |
| 9 | AVX-512 / AMX CPU kernels | no host can compile or verify; the 2026-11-09 draft stays archived and `kindi caps` reports the flags only | an AVX-512 host, then kernels and tests |
| 10 | Calibration's `>=20 s` startup clause | NVIDIA tuning is 0.2 s; the clause needs HIP | a HIP rig |
| 11 | PQ2_0 / Q2_0 4B/9B/27B import | only the 2B variants are on this host | MANUAL-HELP §2 downloads |
| 12 | Arbitrary-draft-GGUF speculation | generalized harness not built | that harness's gate |
| 13 | Two-card TP and mixed-card pairs | one card only | a second card, same or mixed vendor |
| 14 | GPU CI (the agreed-cards gate) | hand-run and hand-recorded; the runner is not registered | the runner registration in `docs/CI.md` §5.1 |
| 15 | Q2_0 CPU expert lane in a running Lightning layer | the per-expert gate is exact and the mixed card/CPU combine contract is implemented (0 differing elements in the synthetic gate), but no `kindi_lightning` pack exists here and the lane is not wired into the engine | a clean Lightning pack, Engine wiring, then `kindi verify` 4/4 and decode tok/s with the lane enabled |

### 4.4 Deliberate divergences (not defects)

| divergence | why |
|---|---|
| `presence_penalty` stays a config float key (upstream moved validation to the server; config defaults and `profiles.py` still read it) | local coherence; documented in the merge commit |
| Server stats keep the fork's superset keys; `/health` unions both queue views and `drafting` | documented merge resolution |
| Installer pins the fork's `0.7.0-fork` source archive | the archive is built locally and is not uploaded yet; the official v0.6.1 tarball (`kindi-0.6.1.tar.gz`, 2,013,311 bytes, sha256 `63f20abf…`) remains upstream's, so a fork installer release remains a release-owner step |
| Turn-save entries' stored rows live under different keys than `Turns` blocks | cosmetic; no upstream rows are read by this fork |
| The fork's version is `0.7.0-fork`, not official `0.6.1` | a fork release must not claim the official patch version without an official artifact; the `-fork` suffix, fork registry and fork release URLs keep it distinct |

### 4.5 The measurements that would change the ranking

In order of how much of this table they would move: a **second card** (two-card TP, mixed cards, the
autotune `tp-2-*` rows, tandem-vs-GPU on a pair), **weights** (a clean Nanbeige upload, DeepSeek shards,
Gemma 4 E4B, the DSpine profile, a Raon/Gemma MoE pack), an **AVX-512 host** (the CPU half tops out at AVX2
today) and a **HIP or Metal rig** (the calibration clause and a non-CUDA backend). None is addressable from
this machine, which is why `docs/MANUAL-HELP-NEEDED.md` lists them as owner actions rather than planned work.
The same-settings A/B is no longer on that list: §4.6 records it on the 2B and 9B. The
controlled 9B CPU/tandem pair in §3 is a placement correction, not a stock A/B.

### 4.6 Same-settings A/B against official v0.6.1 (2026-11-17)

The comparison is no longer only code-level. Stock `v0.6.1` (`7fececb`, extracted with
`git archive v0.6.1`, not the release tarball or a container image) and this tree at
`a79137f` ran `kindi bench` on the same 2B and 9B packs, on the same RTX 4060 Ti, at
`--ctx 8192`. Both sides used the plain engine and the exact head: stock has no plan
engine, so the fork used its plain path (`KINDI_TP=0`) and `--split-at off`, and no
tensor-core, speculation, tandem or cache setting was enabled on either side. Stock
still needs the native-Windows `libcuda.so.1` -> `nvcuda.dll` loader shim recorded on
the earlier stock page; it changes no engine path.

Five interleaved passes per cell, medians (tok/s), except the 9B prompt-reading
cells marked `‡`, which are the 20-pass 2026-11-18 recheck:

| model | engine | prompt reading | gen @100 | @2,000 | @8,000 |
|---|---|---:|---:|---:|---:|
| 2B | stock v0.6.1 | 215.5 | 180.5 | 182.9 | 176.4 |
| 2B | this fork | 211.1 | 178.8 | 179.9 | 178.6 |
| 9B | stock v0.6.1 | 70.7 `‡` | 83.8 | 81.3 | 77.4 |
| 9B | this fork | 75.7 `‡` | 81.7 | 81.5 | 77.2 |

Decode is the same on both models: every fork median is within 2.6% of stock's, and
the five-pass ranges overlap (the 2B at position 8,000 is 176.4 with a 148.2-182.1
range against 178.6 with 174.0-180.9). The 2B prompt-reading cell is still not
resolved: 215.5 and 211.1 with overlapping ranges. The 9B five-pass values were
**83.9** stock and **72.4** fork; their ranges overlapped (stock 52.0-92.7, fork
56.2-89.7), and the gap did not reproduce. The `‡` recheck used 20 fresh-process
passes per side with alternating order: 70.7 stock vs 75.7 fork, a paired
stock-minus-fork median of **-1.4 tok/s**, a 95% bootstrap interval spanning zero,
and an exact sign test of **p = 0.648**. The read window is a sub-second, 20-token
burst; 50 ms telemetry found it mostly in P5/P8 at low clock/power, while generation
reached the boost clock, and the second run in each pair was faster on median. The
9B prompt-reading gap is therefore **closed as noise**; no reading difference is
claimed. Details and the 40 clock logs are in the recheck page.

The fork's **default tensor-core plan path** was measured separately on the same two
packs at the same context. This is a **different configuration, not a same-settings
row**: stock has no plan engine or tensor-core path, and the fork used
`KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1 KINDI_TC_TMAX=4`, its q8_0 K/V
default and the exact head; the 9B automatically selects `KINDI_TC_HP=1` for its
Hadamard-4096 pack. No speculation, tandem, cache or split request was enabled. Five
interleaved passes per model (2026-11-18):

| model | engine | prompt reading | gen @100 | @2,000 | @8,000 |
|---|---|---:|---:|---:|---:|
| 2B | stock v0.6.1 (reference, 2026-11-17) | 215.5 | 180.5 | 182.9 | 176.4 |
| 2B | **this fork, tc plan - different configuration** | 484.9 | 354.9 | 333.8 | 331.0 |
| 9B | stock v0.6.1 (reference, 2026-11-17) | 83.9 | 83.8 | 81.3 | 77.4 |
| 9B | **this fork, tc plan - different configuration** | 137.2 | 118.3 | 115.2 | 111.8 |

These are absolute rates for the TC path against the same bench cells, not a
controlled estimate of what tensor cores or the plan engine change; the fork's plain
rows above are the controlled comparison. The stock rows are repeated from the
table above, not re-run in the TC capture; the stock 9B reading reference is the
five-pass value, while the same-settings row above is the 20-pass recheck. Ranges
and per-run records are in the TC benchmark page.

The same-settings rows above do not measure the fork's plan engine or tensor cores;
the separate TC rows do, but they are intentionally not presented as same-settings
numbers. Neither table measures speculation, tandem, server/cache paths or token
exactness; the separate token-golden gate is §4.7. Evidence:
`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-ab-0.6.1/RESULTS.md` (20 logs, 10 fork
JSONs, `summarize.py` -> `ab.json`),
`docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-fork-tc-vs-stock/RESULTS.md` (10 logs,
JSONs, `summary.json`) and
`docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-9b-reading/RESULTS.md` (40 logs, 40
clock CSVs, `summary.json`, `runs.json`).

### 4.7 Token-exact stock goldens (2026-11-20)

§4.6 measures speed, not token identity. The separate
[stock-parity page](docs/benchmarks/2026-11-20-i7-8086k-rtx4060ti-stock-parity/RESULTS.md)
pins official v0.6.1 greedy token ids for the 2B and 9B at three prefix lengths,
with the fixed prompt ids in `prompts.json` and the official outputs in
`goldens.json`. The gate is one command:

```sh
python tools/stock_parity.py --check
```

It verifies the local `v0.6.1` tag, extracts the stock tree from it, runs both
trees, exits non-zero on the first differing token, and refreshes the tracked
`parity-report.json`. The captured run is **12/12 exact**. The page records the plain-engine/exact-head settings and the
limits; it makes no speed claim and does not replace the §4.6 A/B.

## 5. Model and hardware status

**Rig.** i7-8086K (6 cores / 12 threads), RTX 4060 Ti 8 GB with the full **BAR1 8192 MiB**
window, 2026-11-18. This is the only GPU used for the current measurements; the legacy
cards, Arc and Intel iGPU rows below are not physical runs here.

**Small MoEs (2026-11-14, `kindi/moe_compat.py` + `kindi/cascade.py`).**
Recognition-only, not backends: tool routing for Nemotron Cascade 2 30B-A3B
(converter/validator: header + exact F32/Q5_0/Q8_0/Q4_K dequant + hardened
pinned-reference gate; GPU K-quant kernels pending), Gemma 4 26B-A4B
(E4B-only today), Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, North Mini Code 1.0,
and Recursive Flint 40B. Full table: `docs/ROADMAP.md` (Small-MoE
compatibility). No catalog/installer/support-matrix entry before a pinned
reference + token-exact `kindi verify`.

**Legacy cards.** Installer/profile routing verified with fake-GPU tests for
P4000 (`pascal8`), P106-100 (single 2B/4B/9B, dual 27B), M4000/M5000
(`maxwell`), and RTX 4000 (`turing`, TC profile). Rig sessions now cover
M5000 and RTX 4000 Turing (`docs/OLD-CARD-RIG.md`); **no physical run** on
any of the five cards, so no new performance claim. Intel Arc: M0 audit
done, A770 16 GB purchased and in transit, M1 (hardware qualification)
unblocked on arrival (`docs/INTEL-ARC.md`, REQ-0020); Intel iGPU tandem
investigation complete with G0 implementation (`docs/INTEL-IGPU-TANDEM.md`,
REQ-0021). Mixed-card pairs: same-vendor layout implemented, GPU-unverified
(`docs/MIXED-CARD-ROADMAP.md`).

**Speculation.** 2B drafter pays on the 27B (1.56x at 8K), does not pay on
the 9B (0.98x) or the 4B (0.88x), and does not fit at 24K on 8 GB — all
measured, not assumed (`docs/MTP.md`). The adaptive draft-policy controller
(REQ-0015) is now gated on both sizes: 2B +15.8% (368.4 vs 318.2 tok/s) and
27B parity (40.1 vs 40.1 tok/s), exactness identical in both. The generalized
arbitrary-draft-GGUF harness stays a separate gate.

## 6. Evidence

- §1's git commands (`git merge-base --is-ancestor`, `rev-list --count`,
  `diff --shortstat`, `diff --diff-filter=D`, `ls-tree`, `ls-files`).
- Latest official release check (2026-11-21): GitHub `/repos/ahtx/kindi/releases/latest` -> [`v0.6.1`](https://github.com/ahtx/kindi/releases/tag/v0.6.1),
  published 2026-11-16T21:49:20Z; `git ls-remote origin refs/heads/main` -> `7fececb`.
- Baseline counts at `66eebd3`: **678 commits**, **897 files**, `+127,084/−1,417`, **998 tracked files**,
  **28 numbered requirements**, zero file deletions (`git rev-list`, `git diff --shortstat`,
  `git diff --diff-filter=D`, `git ls-tree`).
- Upstream PR status (2026-11-21): #47-#55 open, one commit each, `mergeable: clean`, no CI
  check suites configured; the union of their `git diff --name-only v0.6.1..<head>` is **35 paths**.
  The CLA 1.1 statement is in every title; per-commit author/committer is
  `Kindi Maintainers <noreply@kindi.invalid>`.
- Container smoke status (2026-11-20): the rebuilt `kindi:0.7.0-fork` image
  (`sha256:b08916d7…`, Python 3.6.9) passes the **unmodified-image** smoke on the RTX 4060 Ti:
  `/health` 200, one non-streaming chat 200 in **0.2337 s**, GPU passthrough and 2B pack load
  passed. The Python 3.6 blocker is fixed; no image has been published and `/health` still reports
  `commit: null` from the `kindi/COMMIT` placeholder. Evidence:
  `docs/benchmarks/2026-11-20-i7-8086k-rtx4060ti-container-smoke/RESULTS.md`.
- Focused checks at this refresh: `python tools/check_claims.py` (37 gates / 265 numbers),
  `python -m pytest -q tests/test_claims.py tests/test_docs_links.py` (20 passed).
- Full suite on the current baseline: **1,420 passed, 28 skipped, 0 failed** (recorded at
  `ee2ba46`; not re-run for this refresh); the earlier **1,391** at `60a039b` and **1,357** at
  `f75253b` + `84b6686` are historical. On 2026-11-21 `config_reference --check` was stale
  (missing `KINDI_TANDEM_HEAD_SPLIT`); after regeneration, `tests/test_config_reference.py` is
  **8 passed**. Generated-artifact checks (`project_memory --check`, `config_reference --check`,
  `support_matrix --check`) clean.
- Claims registry and procedure: `docs/claims/README.md`, `docs/claims/REGISTRY.json`.
- Requirement evidence: `docs/development/REQUIREMENTS-HISTORY.md`; live handoff and leases:
  `AGENTS.md`, `docs/development/AGENTS-HISTORY.md`.
- Benchmark pages linked from §3, §4.2 and §4.6 (each tracked under `docs/benchmarks/`).
- GPU evidence that is gitignored: `bench-logs/req0015-gate-27b/` (draft-policy 27B gate),
  `bench-logs/convert-4b-verify/` (4B provenance).
