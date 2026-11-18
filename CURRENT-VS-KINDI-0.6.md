# Current worktree vs official Kindi 0.6.1

Report date: **2026-11-18**. Baseline: official **Kindi 0.6.1** (tag `v0.6.1`, commit
`7fececb`, published 2026-11-16T21:49:20Z) against this fork's `main` at `0184bb8`.
This page keeps the current comparison only; the round-by-round history has moved
verbatim to `detailed-fork-changelog.md`.

**Short answer.** This fork is **576 commits and 704 files ahead** of `v0.6.1`, with
**zero file deletions** against the official tree. The measured advantages are the
fork's own plan engine and tensor-core path, the tandem memory split, the KLD quality
gate, `kindi plan` / `kindi caps` / `kindi autotune`, and the session and API work. The
real disadvantages are release identity (the fork still reports and installs **0.6.0**,
and publishes no 0.6.1 images), a few deliberately slower opt-in paths, and evidence
that still needs hardware this machine does not have. Nothing upstream is missing:
both code changes in 0.6.1 are ported.

| question | answer |
|---|---|
| **What is different** | 576 commits / 704 files ahead, 812 tracked files here against 172 upstream; 28 numbered requirements; no upstream file deleted. The fork adds its own runtime, commands, API surface and tests rather than changing 0.6.1's structure. |
| **Better, measured** | Dense tandem + plan-owner placement; the default tensor-core plan path (2B and 9B generation far ahead of stock's plain engine, in the separate TC configuration); 2B/9B same-settings decode parity; 27B peak reduced 7,721 -> 6,091 MiB; KLD quality gate; adaptive draft policy (+15.8% on the 2B, parity on the 27B); physical-core thread count. Details in §3 and §4.6. |
| **New here** | `kindi plan`, `kindi kld`, `kindi caps`, `kindi autotune`, `kindi io`; sessions store/CLI/HTTP/resume/prune/stats; backend plugin contract + adapters; OpenAI error envelope and conformance harness; JSON-schema/GBNF constrained decoding; Windows install and CPU capability paths. |
| **Worse** | Version identity stays 0.6.0 while official is 0.6.1, and there are no fork 0.6.1 images or fork installer artifact; mode A MTP speculation costs **0.66x** at its default k (off by default); the tandem split is **~8 vs ~31-34 tok/s** at 24K but buys memory, not speed; FP8 K/V is lossy; recursive GBNF is expensive by design. Details in §4.1-§4.4. |
| **Unverified** | DeepSeek V4.1, DSpine, Nanbeige 4.3, Gemma 4 E4B, non-dense GPU grammar gates, a complete Windows install, two-card TP/mixed cards, the five legacy cards, Arc A770, Intel iGPU, AVX-512/AMX, the calibration >=20 s clause, and GPU CI. Details in §4.3. |

## 1. Baseline

Official upstream: **Kindi 0.6.1**, tag `v0.6.1` (`7fececb`, 2026-11-16), GitHub
release published 2026-11-16T21:49:20Z. The release notes name the 2B MTP fix and
the `0.6.1-cuda` / `0.6.1-rocm6` / `0.6.1-rocm7` images. `v0.6.0` (`179710c`) is an
ancestor of `main`; `v0.6.1` is one commit past the shared base `dce89df` and is not
an ancestor of this fork. The comparison range is therefore `v0.6.1..HEAD`:

```sh
git merge-base --is-ancestor v0.6.0 HEAD       # yes
git merge-base --is-ancestor v0.6.1 HEAD       # no: v0.6.1 is one commit past dce89df
git merge-base v0.6.1 HEAD                     # dce89df
git log --oneline v0.6.1..HEAD | wc -l         # 576 commits
git diff --shortstat v0.6.1..HEAD              # 704 files, +93,047/-1,229
git diff --diff-filter=D --name-only v0.6.1..HEAD  # (empty: zero file deletions)
git ls-tree -r --name-only v0.6.1 | wc -l      # 172 upstream files
git ls-files | wc -l                           # 812 tracked files here
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

Full suite at `0184bb8` (this refresh): **1204 passed, 28 skipped, 0 failed**, with
`project_memory --check`, `config_reference --check` and `support_matrix --check`
all clean. The skip count is the hardware/model/clang-only set, so the CPU and
stub-engine surface is the part measured; the GPU and model paths were not re-run
here. `python tools/check_claims.py` re-derives **17 gates / 95 numbers** from
tracked evidence. The preview's HEAD chip includes the documentation commit that
follows this snapshot, so its shortstat can be slightly larger than the numbers
above.

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
| **`kindi autotune`: measured placements** (`kindi/autotune.py`, REQ-0022) | 2026-11-17: candidates from the hardware report + pack manifest (full card, two-card TP split, tandem `auto`/`gdn:N`, Lightning decode cache, CPU), each one a real `kindi bench --sample-vram` child run, ranked by decode tok/s with reduced-context runs ranked below full-context ones and failed runs unscored. Live 2B/ctx-8192 run: 97.4–97.8 tok/s decode, 111.7–113.0 read, 3,050 MiB peak; `tandem-auto` refused as unnecessary and `tandem-gdn:8` blocked by a missing compiler, both reported as such. `tests/test_autotune.py` 18 tests |
| **Image input is refused, not silently dropped** (`kindi/server.py`) | 2026-11-17: a pack manifest that declares `vision`/`mmproj` no longer unlocks acceptance — no build implements a projector, so an image part or Ollama `images` is always an enveloped 400, and the message names the missing projector when a pack claims one. Pinned by an app-level test built on a vision-declaring manifest |
| **One grammar compile per structured request** (`kindi/server.py`) | 2026-11-17: validation already built the automaton to answer an invalid schema/grammar with a 400; the decode path built it again. The compiled object is now parked on the request and reused (GBNF and `json_schema`), so a structured request compiles once: llama.cpp's canonical `json.gbnf` at nesting 2 is 7,936 states / 7.8 ms median build, ~15.6 → ~7.8 ms per request. Pinned by a call-counting test |
| **GBNF size honesty** (`kindi/gbnf.py`) | 2026-11-17: `Grammar.states` exposes the compiled size, and the over-budget refusal names `*`/`+` and `json_schema` instead of only a state count. Measured: exact per-use expansion costs ~143 states per enumerated copy (a 40-pair record referencing `ws` 82 times is 5,915 states / ~6 ms), the loop form is ~330 states at any length, and the default 20,000-state cap holds ~130 enumerated copies. The pre-`221d652` rule cache was 20× smaller because it widened the language (an "exactly 3" grammar accepted 0–5 copies) |

## 4. What is worse, missing, or unverified vs official Kindi 0.6.1

Re-derived at `0184bb8` (2026-11-18) against the official `v0.6.1` release (`7fececb`, published
2026-11-16), with `v0.6.0` (`179710c` plus the post-tag `dce89df`) kept for the release ancestry. The
title folds three different questions together, so they are separate below; each row ends with what
would move it, and the linked benchmark pages and `docs/MTP.md` hold the dated runs behind them.

**The short answer.** Nothing upstream is deleted or unported: every file of `v0.6.1` is still present
(`git diff --diff-filter=D --name-only v0.6.1..HEAD` is empty), and the release's two code changes are
in `f934a55`. The official release now publishes `0.6.1-cuda`, `0.6.1-rocm6` and `0.6.1-rocm7` plus
the source and installer hashes; this fork deliberately keeps its 0.6.0 version and container tags
until it has a matching release. What remains open is **evidence**: measurements that need weights, a
second card, another ISA or a clean Windows box. The only measured regression against the simpler path
is mode A speculation, and it is off by default; the other §4.1 rows are deliberate trade-offs.

### 4.1 Worse than the simpler path (measured; not a regression against stock)

| surface | measured here | why it stays | what would move it |
|---|---|---|---|
| Mode A speculation (the model's own MTP head, `KINDI_SPEC=mtp` / `KINDI_MTP_HEAD`) | **0.66x** on the 2B / RTX 4060 Ti at 24K: 361.2/363.7 tok/s plain vs 236.9/239.6 with the head, 378 of 1,680 drafted tokens accepted (22.5%); forcing the guard on from position 0 gives 230.9/247.2/246.6 | the head's graph had not built since 0.5.0 (fixed in `f934a55`; details in `docs/MTP.md`) and now that it builds it loses: the first guard windows are speculation windows, so a short reply pays the loss first | a per-model acceptance gate that stays off when acceptance is low, or a guard that reacts inside a few tokens; at `KINDI_SPEC_K=1` it measured 1.12–1.13x. `docs/MTP.md` |
| Tandem CPU+GPU split on one card | 24K decode **~8 vs ~31–34 tok/s** against the equivalent full-GPU rows; step-0 is 17.6 ms/token of head plus 3.09 ms of boundary transfer, ahead of any worker thread | it is a **memory** feature: a 49,152-token request keeps all 49,152 where the unsplit plan engine reduces to 30,976 | a second card or a much faster CPU head, not a code fix; `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-tandem-9b/RESULTS.md` |
| 9B tandem `gdn:8` / `gdn:16` | saves **486–488 MiB** for 2.8–3.0x decode, or **872–874 MiB** for 4.5–4.6x, flat from 8K to 131,072, while full GPU already holds every request at 5,149 MiB peak | buys headroom, never context; the 27B is the only pack here that needs the split | a pack that does not fit full-GPU; otherwise no split |
| FP8 E4M3 K/V (`KINDI_KV=fp8`) | KLD **2.48e-06** (2K) / **2.80e-06** (8K), ~9–14x q8_0, verify 3/4, same cache size as q8_0 (8.5 bits/value) | experimental opt-in, never a default | a lossless 8-bit format; keep it off otherwise |
| Recursive GBNF (`KINDI_GBNF_MAX_DEPTH`) | opt-in and bounded: llama.cpp's `json.gbnf` reaches array nesting 2 inside the 20,000-state cap, and depth 7 is 398,268 states; `json_schema` is the unbounded path | exact expansion is what keeps the accepted language honest (the old rule cache was 20x smaller by widening it) | shared states or a token-class mask (REQ-0027) |

### 4.2 Missing vs official 0.6.1

| item | official release | here | consequence / closure |
|---|---|---|---|
| Version identity | `0.6.1`; `install.sh` sha256 `31c4a860…` | `0.6.0` in `kindi/__init__.py`; `install.sh` pins the v0.6.0 archive (sha256 `613f7320…`) | the official v0.6.1 installer now exists; this fork still reports and installs 0.6.0. A fork release, not an engine change; `docs/MANUAL-HELP-NEEDED.md` §5 |
| 0.6.1 container packaging | `ghcr.io/ahtx/kindi:0.6.1-cuda`, `0.6.1-rocm6`, `0.6.1-rocm7`; compose/Docker docs updated | `compose.yaml`, `compose.amd.yaml` and `docs/CONTAINERS.md` are present from `dce89df`; the 0.6.1 commit changes `compose.yaml`, the three Dockerfiles and docs; `compose.amd.yaml` came in with `dce89df` | no 0.6.1 images exist from this fork, so pointing at them would be false; publish matching images with a fork release |
| Upstream code | `kindi/mtp.py`'s `hp` argument and `kindi/spec.py`'s graph diagnostic | both ported in `f934a55`; the remaining v0.6.1 changes are version strings, docs, `compose.yaml`, the Dockerfiles and install scripts | nothing to port; no file of the `v0.6.1` tree is missing here |
| Not-behind-stock absences | stock lacks them too | dense standard-attention llama/qwen engines; an image projector (an image part is an enveloped 400); the llama.cpp proxy (REQ-0019 T4.4) and real concurrency (T4.5), whose positioning is an open maintainer call (MANUAL-HELP §6); Metal | listed so "missing" is not read as "behind stock" |
| MoE expert-cache tandem | stock had no such path | **not a backend**: `kindi/expert_cache.py` is a model-free slot/LRU prototype, with no router, pack, GPU copy path or measured inference (`docs/MOE-TANDEM.md`) | Raon-1 mini R0 reference goldens or the Gemma 4 26B contract, plus a GPU copy path; until then no support claim |

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

### 4.4 Deliberate divergences (not defects)

| divergence | why |
|---|---|
| `presence_penalty` stays a config float key (upstream moved validation to the server; config defaults and `profiles.py` still read it) | local coherence; documented in the merge commit |
| Server stats keep the fork's superset keys; `/health` unions both queue views and `drafting` | documented merge resolution |
| Installer pins the published v0.6.0 tarball sha256 `613f7320…` | the official v0.6.1 tarball is now published (`kindi-0.6.1.tar.gz`, 2,013,311 bytes, sha256 `63f20abf…`); the fork cannot republish upstream's artifact, so a fork installer release remains a release-owner step |
| Turn-save entries' stored rows live under different keys than `Turns` blocks | cosmetic; no upstream rows are read by this fork |
| The fork's version stays `0.6.0` while official Kindi is `0.6.1` | a version bump is a release-owner decision; claiming 0.6.1 without a fork artifact would be a false version, and `kindi __version__` would disagree with the installer pin |

### 4.5 The measurements that would change the ranking

In order of how much of this table they would move: a **second card** (two-card TP, mixed cards, the
autotune `tp-2-*` rows, tandem-vs-GPU on a pair), **weights** (a clean Nanbeige upload, DeepSeek shards,
Gemma 4 E4B, the DSpine profile, a Raon/Gemma MoE pack), an **AVX-512 host** (the CPU half tops out at AVX2
today) and a **HIP or Metal rig** (the calibration clause and a non-CUDA backend). None is addressable from
this machine, which is why `docs/MANUAL-HELP-NEEDED.md` lists them as owner actions rather than planned work.
The same-settings A/B is no longer on that list: §4.6 records it on the 2B and 9B.

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
exactness. Evidence:
`docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-ab-0.6.1/RESULTS.md` (20 logs, 10 fork
JSONs, `summarize.py` -> `ab.json`),
`docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-fork-tc-vs-stock/RESULTS.md` (10 logs,
JSONs, `summary.json`) and
`docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-9b-reading/RESULTS.md` (40 logs, 40
clock CSVs, `summary.json`, `runs.json`).

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
- Focused checks at this refresh: `python tools/check_claims.py` (17 gates / 95 numbers),
  `python -m pytest -q tests/test_claims.py tests/test_docs_links.py tests/test_agents_handoff.py`.
- Full suite at `0184bb8`: **1204 passed, 28 skipped, 0 failed**; generated-artifact checks
  (`project_memory --check`, `config_reference --check`, `support_matrix --check`) clean.
- Claims registry and procedure: `docs/claims/README.md`, `docs/claims/REGISTRY.json`.
- Requirement evidence: `docs/development/REQUIREMENTS-HISTORY.md`; live handoff and leases:
  `AGENTS.md`, `docs/development/AGENTS-HISTORY.md`.
- Benchmark pages linked from §3 and §4.6 (each tracked under `docs/benchmarks/`).
- GPU evidence that is gitignored: `bench-logs/req0015-gate-27b/` (draft-policy 27B gate),
  `bench-logs/convert-4b-verify/` (4B provenance).
