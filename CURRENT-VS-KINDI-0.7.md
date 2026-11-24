# Current fork vs official Kindi 0.7.1

## TLDR - too long didnt read

**The comparison:** official Kindi **v0.7.1** (tag `v0.7.1`, commit `1cb59d2`, released 2026-11-23) against this fork at
the **0.7.0 merge commit `35e4697` plus the 2026-11-24 v0.7.1 merge** on `merge/upstream-0.7.0`.

**What 0.7.1 adds (now in the tree):** the **Inkling 2 Mini** engine (`inkling2.py`, `inkling2_gpu.cu`,
`inkling2_cpu.c`, `inkling2_quant.py`, the `kindi inkling` command and the `kindi_inkling2` server backend; the
upstream notes measure 120.9 tok/s read and 19.8 tok/s write on a GTX 1080 at 2,048 context), **Flash-Next drafting
for sampled requests** (`sampling.spec_accept` takes the draft's distribution; `KINDI_FN_DRAFT_SAMPLED=0` opts
out), **one CPU thread per physical core** (`kindi/cores.py`, used by the dense CPU engines, the Flash-Next expert
pool and the DeepSeek pools) and the `check_prompt` engine-lock fix for backend models. **None of these was run on
this host** (no CUDA device, no Inkling pack), so they are present and CPU-reachable-tested only.

**The 0.7.1 changelog names `tools/check_warp_masks.py`, but the tag does not ship it.** The fork wrote it
(`tools/check_warp_masks.py` + 14 tests + `tools/warp_mask_baseline.json`): 404 intrinsic sites classified, 7
reviewed sub-warp masks, 0 unguarded legacy intrinsics. Writing it found two real hazards, both fixed: `chunk.cu`
`k_q8x_T2` bypassed the file's own `KSHFLX` macro and `kindi.cu` `wsum_d`/`wsum_f` used bare `__shfl_xor` on every
arch.

**Measured 2026-11-24 against `v0.7.1` (`1cb59d2`):** **722 commits / 980 changed files ahead**
(+136,509/−17,775), **1,133 tracked files against 232 upstream**, no upstream file missing. The v0.7.0 comparison
that produced this report was 702 commits / 931 files against `8298a0f` at `35e4697`; the numbers below that are
0.6.1-era runs and are unchanged.

**What 0.7.0 brought in (now in the tree):** the Flash-Next engine (`qwen4exp_*`, `qx_cpu.c`, `qx_gpu.cu`), the
multi-slot serving stack (`slots.py`, the `*_R` row kernels in `spec.cu`, `--slots`, `KINDI_SLOTS_*`), placement and
routing tools (`place.py` as `kindi place`, `route.py` as `kindi route`, `cards.py` as `kindi cards`), the request-field
validator (`oai.py`, upstream #38), the pack-size gate (`packcheck.py`), DeepSeek V4.1 expert-card calibration
(`dsv41_xcal.py`), the native Windows launcher and tray monitor (`windows/`), `docs/FLASHNEXT.md`, and the upstream
0.7.0 test and doc updates. It also fixes upstream issue #44 (the Gemma 4 pinned-host leak) the way this fork did, and carries four of this
fork's PRs by content (see the PR status below).

**Still the fork's own, measured on 0.6.1-era hardware:** the tensor-core plan path (**354.9 vs 180.5 tok/s** on the 2B
`gen @100`, **118.3 vs 83.8** on the 9B), same-settings decode parity (every median within **2.6%**, **12/12**
token-exact), the 27B tandem memory split (**7,721 -> 6,091 MiB** peak, 49,152 tokens where unsplit reduces to 30,976),
the adaptive draft policy (**+15.8%** on the 2B), mode A's **k=1** default (**1.12-1.13x**), sessions, `kindi plan` /
`kindi caps` / `kindi autotune` / `kindi kld` / `kindi io`, the JSON-schema/GBNF engine and server wiring, the
process-shared grammar prewarm, and the `0.7.1-fork` release identity. Those numbers were **not** re-benchmarked after
this merge (no CUDA or model packs on this host).

**Caveats:** the `0.7.1-fork` release is still unpublished; no GPU path added by 0.7.0 has been run here, so
**multi-slot serving and the Flash-Next engine are present but not wired into this fork's server** (the fork's server
does not call `slots.py` or the Flash-Next adapter); DeepSeek V4.1, DSpine, Nanbeige 4.3, Gemma 4 E4B and the
Lightning/Nanbeige structured-output paths remain unverified; `/health` still reports `commit: null` from the
placeholder.

**Upstream status (2026-11-23):** **33 fork PRs (#47-#79) are open and 0 are merged**; 7 are drafts (#67-#71, #74, #76)
and external #37 (`JohnUzoka/kindi`, target `dce89df`) is open and dirty. The repository still has no CI check suites
and no maintainer review has landed. Their union is **122 paths**: **92 of the 931 changed paths** in the 0.7.0 delta
and 30 paths that the merge now matches upstream on. **Four are in 0.7.0 by content** - #48 (host RAM capped at the
container cgroup limit), #49 (a tokenizer pack inconsistency fails loudly), #72 (mode A speculation defaults to k=1) and
#76 (Hadamard-4096 packs default to exact tensor-core activations) - because upstream squashes a release into one
commit; they can be closed upstream as included. **2026-11-24 update:** the other 28 PR branches (#50-#71, #73-#75,
#77-#79) were rebased onto `v0.7.0` (`8298a0f`) and force-pushed with `--force-with-lease`, so each PR now diffs
against the official release; local pre-rebase tips are kept at `refs/backup/pr-<name>-pre070`. #47 was left
untouched: 0.7.0 already carries the equivalent Windows loader (`libcuda_name()`/`libcuda()`), so rebasing it
would leave only tests against a renamed API. That leaves 33 open PRs, of which 28 are rebased, 4 are in 0.7.0
by content, and 1 (#47) is superseded.

**Current status (2026-11-24, v0.7.1 merged):** **33 fork PRs (#47-#79) are open and 0 are merged**; 7 are drafts
(#67-#71, #74, #76), and external #37 is open and dirty. After the official v0.7.1 (`1cb59d2`) merge, **31 of the 33
branches were rebased onto it** and force-pushed to `myfork` with `--force-with-lease`; #37 was left untouched.
**#47 (Windows CUDA-driver detection) and #48 (host RAM at the cgroup limit) are now in 0.7.1 by content** - 0.7.1's
`kindi/hostram.py` carries `_cgroup_dirs`/`_cgroup_limit`/`_cgroup_usage` - so both can be closed as included, which
leaves 31 rebased, 2 included and no superseded branch. A merge rehearsal on `1cb59d2` found a 14-PR conflict-free
foundation (#49, #50, #59, #60, #73, #57, #55, #72, #54, #61, #65, #66, #75, #51; #67 and #68 follow); the rest
conflict in `kindi/__main__.py`, `kindi/server.py`, `kindi/backends/__init__.py` or the token-class/engine files.

Report date: **2026-11-23** (PR-status paragraph updated 2026-11-24). Baseline: this fork at **`e19850e`** (the merge commit `35e4697`) against official **Kindi 0.7.0** (tag `v0.7.0`, commit `8298a0f`, released
2026-11-23) against this fork's `merge/upstream-0.7.0` branch. The earlier comparison, its 0.6.1 measurements and the
round-by-round history stay in `detailed-fork-changelog.md`; the speeds and quality figures below are the 0.6.1-era runs
at `60a039b` unless a table says otherwise.

## 0. The 0.7.0 merge (2026-11-23)

| question | answer |
|---|---|
| **What was merged** | `v0.7.0` (`8298a0f`) into `wip/local-round-2026-11-20` (`fa787b4`), branch `merge/upstream-0.7.0`; merge commit **`35e4697`**. Merge base `dce89df`; 701 fork commits and 2 upstream commits. Upstream's side is 106 files, +24,385/-490. |
| **Gemma 4 #44 fix** | v0.7.0's `_to_card` (a plain synchronous upload instead of `cupy.asarray`) and `_give_back` (free CuPy's pinned pool after a drop) are now the fork's too, with 0.7.0's two extra release points (a trimmed checkpoint list, and the checkpoints a resume drops). The fork keeps its host-memory budget and telemetry on top, and `tests/test_gemma4_cache.py` pins all six release points; deleting the two new calls fails two tests. |
| **Conflicts** | 39 files / 69 hunks, all resolved by hand. The fork's implementation wins where it has tests and evidence; upstream's self-contained fixes were ported. |
| **Ported from 0.7.0** | `packcheck` size + sha256 gate (B16), the B16 `g5_<key>.src` source stamp, `bench --split-at` curve positions (#33), `gpudrv.libcuda()`/`libcuda_name()`, the cgroup parent walk and usage in `hostram.py` (#48), `chunk.py`'s `_cublas`/`_gemm16`/`f16_gemm_ok` GPU-path fixes (#4/#33), `tp.py`'s B15 split fixes, `tc_hp_for(man, card)` with the Turing carve-out (#76), `kindi place`/`kindi route`/`kindi cards`, `oai.check` (#38), the per-card installer detection, and the 0.7.0 docs. |
| **Present but not wired** | `slots.py` and the multi-slot server path (`--slots`, `KINDI_SLOTS_*`; `spec.py` has the `*_R` kernels and `build_plan(..., slots=...)`, but the fork's `server.py` still serves one engine), the Flash-Next adapter (`qwen4exp_gen.py`/`qwen4exp_gpu.py`, `KINDI_BACKEND` unchanged), the Windows launcher/tray monitor (built separately), and the `kindi place` portfolio as a server placement. Each needs a GPU run this host cannot do. |
| **Kept from the fork** | The single-engine server path and its structured-output validation (`json_object`/`json_schema`/`grammar` compile to a decode grammar, so upstream's `oai.check` is called with `structured=True` and does not refuse them), the fork's `kindi plan` report (upstream's placement tool is `kindi place`), the fork's `KINDI_TP_SPLIT=auto`, `gemma4_cache.py`, `server.py`'s queue/auth/health surface, and the `0.7.0-fork` version string. |
| **Verification** | `python -m pytest -q`: **1,473 passed, 58 skipped, 0 failed**; `tools/check_python_quality.py`: **207 Ruff / 59 mypy / 884 Pyright / 245 Pylint**, all under the ratchet baseline; `bash -n install.sh`, `compileall`, `kindi --help`/`place`/`route`/`cards`, and the config-reference/support-matrix/project-memory/claims checks pass. **No GPU, pack or Windows run.** |
| **Line endings** | 0.7.0 adds `* text=auto`; `kindi/gemma4_cache.py` keeps its pre-normalization CRLF bytes through a `-text` exception, so the #44 fix and its history show logic lines instead of a whole-file rewrite. |
| **Analyser scope note** | The vendored, unwired Flash-Next modules (`kindi/qwen4exp_*.py`, `kindi/qx_*.py`) are outside Pyright's and Pylint's scope until a GPU run makes them part of the served surface (`pyrightconfig.json`, `VENDORED_UNWIRED` in `tools/check_python_quality.py`); Ruff and mypy still cover them. |
| **Merge defects fixed** | Undefined `slots`/`_slots`/`bc_all`/`dp_all` names that a partial auto-merge left behind, a duplicate `tc_hp_for`, a shadowed `_busy` that turned every 429 into a 500, a streaming path that called a nonexistent `K.acquire`, an unused `mem`/`loop`, `qwen4exp_mtp`'s undefined `hst`, and 43 naked `json.load(open(...))` sites converted to `jsonio.load_json`. |

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

**2026-11-22 PR status refresh (reported baseline `39f9768`).** Five docs-only commits follow `66eebd3`: `741d3f5` (3 files, +109/−45), `96acdea` (`docs/ROADMAP.md`, +38), `69b08d2` (2 files, +69/−51), `c68fb69` (this comparison, +55/−46) and `39f9768` (`AGENTS.md`, +2/−2). The tree lands at **683 commits and +127,213/−1,417**; the net docs-only range is 4 files, +180/−51, so the 897 changed paths and 998 tracked files stay put. No measurement was re-run. §3a now records **34 open PRs** (#37, #47-#79): 27 non-draft, 7 draft, 33 clean and #37 dirty; 122 unique paths, 94 of the 897 changed paths covered and 803 not proposed.

## 1. Baseline

> **At the 0.7.0 merge (2026-11-23):** official `v0.7.0` (`8298a0f`) against this fork on `merge/upstream-0.7.0`:
> **0 commits behind, 702 commits ahead, 931 changed files (+133,113/-1,946), 1,066 tracked files vs 216**, no upstream
> file missing. The paragraphs below are the 0.6.1 baseline they were written at and stay as the measurement record.

Official upstream: [**Kindi 0.6.1**](https://github.com/ahtx/kindi/releases/tag/v0.6.1), tag `v0.6.1` (annotated tag `116532a`, peeled to
`7fececb`, 2026-11-16), GitHub release published 2026-11-16T21:49:20Z. The GitHub
release API was checked again on 2026-11-22; **v0.6.1 remains the latest official
release**. `git ls-remote origin refs/heads/main` returns `7fececb` on 2026-11-22, the
release commit itself, so there is no post-release upstream commit left to port; `v0.6.0` (`179710c`) is still
an ancestor of `main`, and the shared base remains `dce89df`. The comparison range is
therefore `v0.6.1..39f9768`. **`39f9768` is the reported baseline; `66eebd3` is the code baseline**
(five later commits are docs-only: `741d3f5` refreshes this comparison, `96acdea` records
the parked TokenEconomizer note, `69b08d2` refreshes the comparison and PR table, `c68fb69`
adds the stacked grammar-server status, and `39f9768` records the handoff; this refresh follows
them). The range and figures below use the current tree. The `0.7.0-fork` version string, installers and image
tags are committed as part of the 2026-11-20 integration (`82a72bf`..`3b9224a`); the local
image was built from that working tree, not an older release tree. Section 4 keeps the dated
`3fed704`/`ee2ba46` analysis and says so. The 2026-11-20 commits since `60a039b` are listed
above; `detailed-fork-changelog.md` §44-§46 list the earlier rounds.

```sh
git merge-base --is-ancestor v0.6.0 HEAD       # yes
git merge-base --is-ancestor v0.6.1 HEAD       # no: v0.6.1 is one commit past dce89df
git merge-base v0.6.1 HEAD                     # dce89df
git log --oneline v0.6.1..HEAD | wc -l         # 683 commits
git diff --shortstat v0.6.1..HEAD             # 897 files, +127,213/-1,417
git diff --diff-filter=D --name-only v0.6.1..HEAD  # (empty: zero file deletions)
git ls-tree -r --name-only v0.6.1 | wc -l      # 172 upstream files
git ls-tree -r --name-only HEAD | wc -l        # 998 tracked files
ls docs/development/requirements/REQ-*.md | wc -l  # 28 numbered requirements
```

The code baseline `66eebd3` is **678 commits / +127,084**; the five later commits are docs-only
(net 4 files, +180/-51 across them), which is why the file and tracked-file counts do not move.

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

## 3. What this fork has *beyond* stock (better, verified; measured against 0.6.1)

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

## 3a. Upstream pull-request coverage (2026-11-23)

> **Status at the 0.7.0 merge:** **33 fork PRs (#47-#79) open, 0 merged, 7 drafts** (#67-#71, #74, #76); external #37
> open and dirty. Upstream still has no CI check suites and no maintainer review. The union is **122 paths**, of which
> **92 are in the 931-path 0.7.0 delta** and 30 are now identical to upstream after the merge. **#48, #49, #72 and #76
> are in v0.7.0 by content** (upstream squashes a release into one commit) and can be closed upstream as included.
> The per-PR table and the 0.6.1 coverage arithmetic below are unchanged from the 2026-11-22 measurement.
>
> **Superseded by the 2026-11-24 state:** all 31 non-#37 branches were rebased onto v0.7.1 (`1cb59d2`), and **#47**
> joined the included-by-content list (0.7.1 ships the host-RAM cgroup cap and its own Windows loader), so the open
> set is 33 PRs of which 31 rebased and 2 closeable as included. See the current-status paragraph above.

Thirty-four pull requests are open against `ahtx/kindi`: **#37** and **#47-#79**. **33 target
`v0.6.1` (`7fececb`)**; #37 is an external fork (`JohnUzoka/kindi`) based on `dce89df`
(the 0.6.1 parent) and GitHub reports it **`dirty`**. **27 are non-draft and 7 are draft**
(#67-#71, #74, #76); the 33 fork PRs are `mergeable: clean`. Every one of the 33 fork PR titles carries the CLA 1.1
sentence; external #37 does not. The repository has **no CI check suites** and no maintainer
review has landed.
Together the 34 touch **122 unique paths**: **94 are among the 897 changed paths (10.5%)**
and **28 are outside the fork's changed-path set** (27 new files plus `kindi/tc_hip.cu`, which
external #37 modifies); 54 paths are touched by more than one PR. Conversely, **803 of the 897 changed paths are not proposed**. File count is
not feature count: the 803 comprises 606 docs, 92 tests, 37 `kindi/` paths, 31 tools and 37
other paths, while the 94 include whole engine modules.

| group | PRs | state | unique paths | in 897 | outside delta | proposals |
|---|---|---:|---:|---:|---:|---|
| Foundation / runtime | #47-#60 | 14 open, clean | 51 | 51 | 0 | Windows loader, cgroup RAM, I/O hygiene, VRAM sampling, `kindi io`, Intel probe, grammar engine, draft policy, caps, config reference, error envelope, codecs, Ollama links |
| Server + grammar wiring | #61-#65 | 5 open, clean | 25 | 17 | 8 | `response_format`, Responses API, admission bounds, Anthropic Messages, HTTP sessions |
| Backend contract + adapters | #66-#71 | #66 open; #67-#71 draft | 36 | 24 | 12 | Plugin contract/registry and one adapter per family |
| Speculation / TC / tandem / CLI / parity | #72-#79 | 6 open; #74, #76 draft | 38 | 32 | 6 | k=1 default, stock parity gate, session CLI, Turing TC, dense tandem runtime/planner/VRAM |
| External HIP | #37 | open, dirty, base `dce89df` | 12 | 10 | 2 | RX 6600 Windows HIP and opt-in memory fitting |
| **all** | **#37, #47-#79** | **27 open + 7 draft; 33 clean + 1 dirty** | **122** | **94** | **28** | |

| PR | state | proposal | head | paths | +/− | stack / note |
|---:|---|---|---|---:|---:|---|
| #37 | open, dirty | RX 6600 Windows HIP + opt-in memory fitting | `e8163a2` | 12 | +541/−32 | external fork; base `dce89df` |
| #47 | open | load the CUDA driver on native Windows | `a556ef3` | 5 | +48/−7 | base for #67 |
| #48 | open | cap host RAM at the container cgroup limit | `17d4116` | 2 | +152/−4 | |
| #49 | open | reject an inconsistent tokenizer pack | `55fb576` | 2 | +29/−1 | |
| #50 | open | close pack/cache JSON handles before returning | `5c2797c` | 6 | +73/−18 | |
| #51 | open | sample device-wide VRAM peak in `kindi bench` | `ee2cdc1` | 4 | +241/−1 | |
| #52 | open | measure host memory tiers with `kindi io` | `a79cd28` | 3 | +579/−1 | |
| #53 | open | read-only Intel probe (Linux + Windows Level Zero) and loud `KINDI_DEVICE=intel` refusal | `b8d8b70` | 4 | +1,383/−1 | |
| #54 | open | GBNF + JSON-schema + token-class grammar engine | `c412b79` | 8 | +2,186/−3 | base for #61 |
| #55 | open | opt-in adaptive draft-length policy | `e880da2` | 4 | +194/−0 | tensor-core spec path only |
| #56 | open | `kindi caps` + CPU feature/tier report | `feb2cf7` | 7 | +1,836/−42 | |
| #57 | open | generated configuration reference | `feea85e` | 4 | +587/−1 | |
| #58 | open | OpenAI-style API error envelope | `c1ca43b` | 4 | +105/−4 | |
| #59 | open | host-side BF16 / FP8 codecs | `ec52897` | 2 | +192/−0 | |
| #60 | open | link Ollama GGUFs under readable names | `3d83926` | 2 | +376/−0 | |
| #61 | open | wire `response_format`/GBNF into the server | `3fc8428` | 12 | +2,852/−14 | stacked on #54; includes its 8-file change |
| #62 | open | stateless OpenAI Responses API endpoint | `8ba03d2` | 8 | +694/−2 | |
| #63 | open | bound queue, request rate and body size | `61a5c84` | 6 | +334/−13 | |
| #64 | open | serve the Anthropic Messages API | `39c8cba` | 8 | +641/−2 | |
| #65 | open | create/manage chat sessions over HTTP | `735b080` | 4 | +720/−4 | base for #74 |
| #66 | open | backend plugin contract and registry | `381ef95` | 16 | +1,514/−4 | base for #67-#71 |
| #67 | draft | Bonsai backend adapter | `e4996d3` | 24 | +2,066/−11 | on #66/#47 |
| #68 | draft | Gemma 4 backend adapter | `7961e86` | 19 | +2,157/−4 | on #66; test conflict with #69 |
| #69 | draft | Nemotron Lightning backend adapter | `f22530c` | 19 | +2,084/−4 | on #66; test conflict with #68 |
| #70 | draft | Nanbeige backend adapter | `7a837dc` | 19 | +1,999/−4 | on #66 |
| #71 | draft | DeepSeek V4.1 backend adapter | `c20ef01` | 19 | +1,999/−4 | on #66 |
| #72 | open | default mode A speculation to k=1 | `535b15d` | 3 | +43/−1 | |
| #73 | open | reproducible stock-v0.6.1 greedy parity gate | `4f8a6c8` | 6 | +11,402/−0 | 12/12 exact golden capture |
| #74 | draft | session CLI and resumable terminal chats | `c2b525b` | 6 | +1,147/−10 | on #65 |
| #75 | open | Turing (sm_75) tensor-core backend | `40a72fb` | 7 | +1,544/−16 | base for #76 |
| #76 | draft | Hadamard-4096 packs use exact TC activations | `9a999d5` | 9 | +1,714/−29 | on #75 |
| #77 | open | dense CPU+GPU tandem runtime | `f4a06b4` | 12 | +1,025/−51 | base for #78 |
| #78 | open | plan dense tandem placements with `gdn:N` and `auto` | `09c0126` | 12 | +1,356/−52 | on #77; includes its bench fix |
| #79 | open | plan-engine tandem owner and per-card VRAM budgets | `32acc12` | 19 | +2,977/−89 | on #78; head lacks `09c0126` |

Paths and lines are GitHub's PR diff against `main`; a stacked PR's count includes its base
(#61 includes #54's 8-file change; #67 includes #66 and #47; #68-#71 include #66; #74
includes #65; #76 includes #75; #78 includes #77; #79 includes #78). `mergeable: clean` only
means the diff applies; it is not review or a merge recommendation. #68 and #69 each assert
that only their own adapter is registered, so both tests cannot pass in one tree. #79 was cut
before #78's latest `09c0126` bench fix and needs a rebase after #78. #37 is external, targets
`dce89df`, and is dirty; it is not merge-ready.

**Covered at path level (94 of 897).** The union now includes the backend contract/registry
(#66) and all five adapters (#67-#71); the OpenAI error envelope (#58), grammar wiring
(#61), Responses API (#62), admission bounds (#63), Anthropic Messages (#64) and HTTP
sessions (#65); mode A k=1 (#72), the stock parity gate (#73), Turing TC and its HP default
(#75/#76), and dense tandem runtime/planner/VRAM (#77-#79). It also includes the earlier
runtime/tooling batch (#47-#60). Draft or stacked status still applies to each.

**Not covered (803 changed paths).** 606 docs, 92 tests, 37 `kindi/` paths, 31 tools and
37 other paths. Of the 86 changed `kindi/` paths, 49 are proposed and 37 are not; of 616
changed docs, only 10 are. The gap table below is feature-level, not a claim that every
uncovered file is independent work.

| workstream | not covered by any PR | recorded evidence | why it is still fork-only |
|---|---|---|---|
| Planner, autotune, calibration and support matrix | `kindi/plan.py`, `autotune.py`, `ltfit.py`, `calib.py`, `calibrate.py`, `catalog.py`, `cascade.py`, `profiles.py`, `support_matrix.py`, `kld.py` | `test_plan.py`/`test_autotune.py`; 27B 24K peak 6,091 MiB; the `>=20 s` calibration clause; generated support matrix | #77-#79 add tandem placement and VRAM budgets inside the engine, not the `kindi plan`/`autotune` CLIs or the generated matrix; the planner contract needs maintainer design feedback |
| Native family engines and speculators | `kindi/dsv41*`, `gemma4*`, `lightning*`, `nanbeige*`, `mtp.py`, `mtp_heads.py` | family tests, CLI/server validation, Lightning GPU gate | #67-#71 add the adapter boundary only; packs are absent or corrupt, and each engine is a large PR on its own |
| MoE expert cache and cold tier | `kindi/moe.py`, `moe_compat.py`, `moe_tandem.py`, `expert_cache.py`, `expert_chunk.py`, `cold_experts.py` | `test_expert_cache.py`/`test_cold_experts.py`; prefetch, slots and byte accounting | unit-tested only: no larger-than-RAM pack, GPU run or tok/s claim |
| Kernels outside the Turing PRs | `kindi/chunk.cu`, `gemma4.cu`, `gemma4_fp8.cu`, `lightning.cu`, `tcf.cu` | multi-ISA dispatch, FP8 `mma.sync` tests, Q2_0 expert-lane gate | #75/#76 cover Turing TC; no AVX-512 host, real FP8 model or Blackwell run, and the Q2 lane is not wired |
| Speculation and quality harnesses | `tools/verify_matrix.py`, `tools/spec_ab.py`, `tools/mtp_smoke.py`, `tools/check_claims.py`, coverage/perf/secret/quality gates | `verify --spec`/`bench --spec`, a 12/12 byte-identical A/B, 1,420-test suite, 37 claims gates / 265 numbers | #72 covers only the k=1 default and #73 only stock parity; these harnesses depend on model packs and repo-specific baselines |
| Remaining server/API breadth | `kindi/server.py` is touched, but `/metrics`, the JSON access log, route-level 404/422 and the remaining request fields have no PR; session CLI is draft #74 | `test_server*`, session tests; bounded queue/429, rate limit, body caps | #58/#61-#65 split the API surface, and #74 is still a draft; the route-level work is not proposed |
| Release/docs surface | `kindi/__init__.py`, `container.py`, `convert.py`, both installers, the archive builder, 606 docs and 92 tests | `0.7.0-fork` identity, container smoke, release checklist | depends on the code PRs; docs-only PRs would conflict heavily |

§4.3 lists the work that is not verified at all. This table lists only fork work with recorded
tests or measurements that no PR proposes; it is an inventory of what could be proposed next,
not a promise that each item should be.

## 4. What is worse, missing, or unverified vs official Kindi (measured against 0.6.1)

Re-derived at the committed baseline `ee2ba46` (2026-11-20) against official `v0.6.1` (`7fececb`,
published 2026-11-16), with `v0.6.0` (`179710c`) and the shared base `dce89df` kept for the release
ancestry. The 2026-11-21 checkpoint (`e57f6bd`) is test/runtime work and does not re-run these rows. These figures are the committed tree's, not the working tree's. The speed and quality
measurements were taken at `60a039b`; the seven later commits add the release integration, evidence,
claims and VRAM controls, not a re-run of the benchmark matrix. The 2026-11-19 analysis was taken at
`3fed704` (666 commits, 827 files, `+107,317/−1,370`). The
title folds three different questions together, so they are separate below; each row ends with what
would move it, and the linked benchmark pages and `docs/MTP.md` hold the dated runs behind them.

**The short answer.** Nothing upstream is deleted or unported: every file of `v0.6.1` is still present
(`git diff --diff-filter=D --name-only v0.6.1..39f9768` is empty), and the release's two code changes are
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
- Latest official release check (2026-11-22): GitHub `/repos/ahtx/kindi/releases/latest` -> [`v0.6.1`](https://github.com/ahtx/kindi/releases/tag/v0.6.1),
  published 2026-11-16T21:49:20Z; `git ls-remote origin refs/heads/main` -> `7fececb`.
- Counts at reported baseline `39f9768`: **683 commits**, **897 files**, `+127,213/−1,417`, **998 tracked files**,
  **28 numbered requirements**, zero file deletions (`git rev-list`, `git diff --shortstat`,
  `git diff --diff-filter=D`, `git ls-tree`). The code baseline `66eebd3` is 678 commits / +127,084;
  the five later commits are docs-only (net 4 files, +180/−51).
- Upstream PR status (2026-11-22, GitHub API): **34 open** (#37, #47-#79): **27 non-draft and
  7 draft** (#67-#71, #74, #76), **33 `mergeable: clean`** and external #37 `dirty`. All target
  `7fececb` except #37 (`JohnUzoka/kindi`, base `dce89df`, not merge-ready). The union is **122 unique
  paths**: **94 of the 897 changed paths** (49 of 86 `kindi/` paths), 28 paths outside the delta and 54 shared;
  **803 changed paths are not proposed** (606 docs, 92 tests, 37 `kindi/`, 31 tools, 37 other).
  Stacks: #61 on #54, #67 on #66/#47, #68-#71 on #66, #74 on #65, #76 on #75, #78 on #77 and
  #79 on #78; #79 was cut before #78's `09c0126` bench fix. #68 and #69 each assert only their
  own adapter is registered, so both tests cannot pass in one tree. All 33 fork PR titles carry the
  CLA 1.1 statement; external #37 does not. There are **0 CI check runs** and no maintainer review. Per-commit author/committer is
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
