# Current worktree vs stock Kindi 0.6.0

Report date: **2026-11-15** (refresh of the 2026-11-14 report; §7 records
what changed since). This file is the successor to
`CURRENT-VS-KINDI-0.5.md` §25: that file's §§1–24 are the history against
0.5.0 and §25 is the first evaluation against 0.6.0; this file carries the
current one. It separates verified behavior from recorded evidence,
incomplete features, and release-hygiene problems. It is not a claim that
everything described here has been re-run on current hardware.

## 1. Baseline

Upstream: tag `v0.6.0` (`179710c`, 2026-11-12) plus the post-tag upstream
head `dce89df` (container-registry docs, 2026-11-13). Both are ancestors of
HEAD `dd605e3`:

```sh
git merge-base --is-ancestor v0.6.0 HEAD   # yes
git merge-base --is-ancestor origin/main HEAD  # yes
git log --oneline v0.6.0..HEAD | wc -l     # 261 commits
git diff --shortstat v0.6.0..HEAD          # 397 files, +52,364/-993
git diff --diff-filter=D --name-only v0.6.0..HEAD  # (empty: zero file deletions)
```

The 0.6.0 merge landed on 2026-11-14 (commit `f635513`, 18 conflicted files
over 172 diverged upstream commits) and has been carried forward since. The
tree vs `v0.6.0` is a **strict superset**: every upstream file is present,
and the fork adds behavior on top of it. The −993 in the shortstat are line
deletions inside files, not file deletions (the name-only check above is
empty).

Full suite at `dd605e3`: **760 passed, 11 skipped, 0 failed, 0 errors**
(with the host's clang 18.1.8 on PATH for the `libkcpu` build only). Without
clang on PATH the same run is 748 passed + 1 failure (a stale generated
config reference, fixed by regenerating in `dd605e3`) + 2 setup errors in
`test_server_cpu.py` (the CPU-library build needs `clang`; environmental,
not a code bug — the two tests pass with clang present).

## 2. What 0.6.0 brought that 0.5.0 did not have (now in the tree)

| 0.6.0 item | in this tree | status here |
|---|---|---|
| MoE backend (`kindi/moe*.py`, `moe_cpu.c`) | present (merged) | **no GPU run in this session**; fork adds expert-tier *planning* (`kindi plan` MoE strategy, `kindi/moe.py plan()`); small-MoE generalization tracked in §5 |
| DeepSeek-V4.1-Flash backend (`kindi/dsv41*`) | present (merged) | **GPU-unverified here** |
| Nemotron Lightning (`kindi/lightning*`) | present (merged) | **GPU-unverified here**; `kindi plan` describes its tiering header-only |
| **DSpine drafter** (`kindi/dspine.py`, `kindi/dspine.cu`) | present (merged); `spec.py` hooks wired | **GPU-unverified here**; fork wraps the build in `spec._optional` so a broken DSpine degrades to no-speculation instead of aborting startup |
| Container entrypoint + compose, Kindi Bridge, `install.ps1` | present (merged) | `install.ps1` **not run on Windows in this session** |
| Turn-save cache (`turn_keep`, `turn_disk_mb`), priority scheduler, `/v1/tokenize` + `/detokenize`, `dry_run` | present (merged) | CPU-tested; no Windows install run |
| New card classes (M5000 sm_52 fix, Lightning offers, DSpine profile) | present (merged) | rig-measured cards remain `not measured` except the 4060 Ti (§4) |

## 3. What this fork has *beyond* stock 0.6.0 (better, verified)

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
| **Backend plugin foundations** (`kindi/backends/`, REQ-0019) | T4.1 contract/registry/FakeBackend + T4.3 `kindi backends [--json]` listing (2026-11-15); CPU-only, importable on Python 3.6. No engine adapter yet (T4.2), so no arbitrary-GGUF claim |
| **OpenAI error envelope + API conformance harness** (`kindi/api/errors.py`, `tests/api/`) | T5.1 2026-11-15: every server error is the OpenAI envelope; harness drives the real app with the official `openai` client, GPU-free |
| **Intel Arc A770 purchased** (REQ-0020) | one A770 16 GB bought 2026-11-15, in transit (not installed); M0 audit done, M1 arrival checklist ready, `arc-a770` registered as `not measured`. No backend code, no support claim |
| **Intel iGPU G0 slice** (`kindi/intel.py`, REQ-0021) | 2026-11-15: DRM-sysfs discovery, oneAPI-floor eligibility, shared-memory budget, tandem GPU contract; `KINDI_DEVICE=intel` refuses loudly. G1+ SYCL kernels need an 11th-gen+ iGPU + oneAPI to compile/verify |
| **Tandem worker-thread verdict** | step-0 re-measured 2026-11-15 (27B/24K, RTX 4060 Ti + 8086K): CPU 74.7 ms/token (6.2 ms/layer) at raw-kernel rate; head 17.6 ms + transfer 3.09 ms come first. **Worker thread not justified, should not be built** |
| **4B conversion provenance** | fresh `kindi convert` of the downloaded PTQ1_0 GGUF reproduces the 4B pack byte-identically (sha256); converted pack verifies 3+tie like the reference; 9B/27B pack verify re-run 4/4 (2026-11-15) |
| Requirement-based coverage + per-module floors | `tools/check_coverage.py` + `tests/coverage_floors.json` in CPU CI (audit M4); requirement-traceable CPU tests |
| Audit dispositions (2026-11-15) | the 202611150853 audit verified item-by-item on the current tree: handoff-guard findings already fixed, coverage floors already landed, hardware items already recorded unverified; hash-bump hook declined by design, fork gap unfixable (push 403) |
| Audit fixes: optional-accelerator degradation, `kindi_cache_*` telemetry, reference-config section | 2026-11-14 audit round (N1/N3–N5/D3/D4) |

## 4. What is worse, missing, or unverified vs stock 0.6.0

Nothing upstream was deleted. Deliberate divergences kept from the fork:

| divergence | why |
|---|---|
| `presence_penalty` stays a config float key (upstream moved validation to the server; config defaults and `profiles.py` still read it) | local coherence; documented in the merge commit |
| Server stats keep the fork's superset keys; `/health` unions both queue views + `drafting` | documented merge resolution |
| Installer pins the published v0.6.0 tarball sha256 `613f7320…` | the fork cannot republish upstream's artifact; the **installer release asset itself is still unpublished** (release-owner step) |
| Turn-save entries' stored rows live under different keys than `Turns` blocks | cosmetic; no upstream rows are read by this fork |

Still real and orderable against 0.6.0:

- The 0.6 backends (MoE / DeepSeek / Lightning / DSpine) have **no GPU run record in this session** — first check with real hardware: `kindi verify` with `KINDI_USING=dsv41`, `KINDI_BACKEND=lightning`, `KINDI_DRAFTER=dspine` on the rig.
- `install.ps1` has never been run on Windows hardware.
- Tandem is still slower than GPU-only on 1 GPU (24K decode ~8 vs ~31–34 tok/s equivalent rows at 8K/24K); the overlap step-0 (re-measured 2026-11-15) puts the head (17.6 ms/token) and boundary transfers (3.09 ms) ahead of any worker thread.
- FP8 K/V (#8) is measured **lossy, not lossless** (KLD ~9–14× q8_0, verify 3/4) — an experimental opt-in, not a default; #9 (Windows unbuffered load) measured NOT met; #6's ≤1.2× clause recorded unsatisfiable with the dedupe rule.
- Dense standard-attention llama/qwen GGUFs are refused (no engine): see §5.
- Intel Arc: A770 purchased and in transit (M1 unblocked on arrival); no SYCL backend exists, and the iGPU route is G0-only until an 11th-gen+ iGPU + oneAPI toolchain exists to compile/verify kernels.
- Backend plugin interface: T4.2 engine adapters, T4.4 llama.cpp proxy, REQ-0018 structured output (T5.4) and real concurrency remain planned; the conformance suite and error envelope are the landed foundations.
- PQ2_0/Q2_0 (type 142/42) import stays verified for the 2B only — the 4B/9B/27B files were never downloaded and Hugging Face is unreachable from this host.

## 5. Model and hardware status

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

- `git merge-base --is-ancestor v0.6.0 HEAD` and `origin/main HEAD` (§1 commands).
- The full suite at `dd605e3` (760 passed, 11 skipped, with clang on PATH for the CPU-library build).
- Lease rows in `AGENTS.md`; requirement evidence in
  `docs/development/REQUIREMENTS-HISTORY.md`.
- `docs/ROADMAP.md`, `docs/EXTERNAL-ENGINES.md`, `docs/MTP.md`,
  `docs/INTEL-ARC.md`, `docs/INTEL-IGPU-TANDEM.md`,
  `docs/MIXED-CARD-ROADMAP.md` for the per-item gates.
- GPU evidence (gitignored bench-logs): `bench-logs/req0015-gate-27b/`
  (draft-policy 27B gate), `bench-logs/convert-4b-verify/` (4B provenance).

## 7. What changed since the 2026-11-14 report

- Diff vs `v0.6.0` grew 360 → 397 files (+47,505 → +52,364 insertions);
  still zero file deletions. Suite grew 627 → 760 passed (new backends/API/
  tool-call/sessions/coverage tests).
- New beyond-0.6 items: FP8 E4M3 K/V (measured lossy), backend plugin
  foundations + `kindi backends` listing, OpenAI error envelope + API
  conformance harness, tool-call conformance suite, draft-policy 27B gate
  met, Intel Arc A770 purchased + iGPU G0, tandem worker-thread verdict
  (not justified), 4B conversion provenance + 9B/27B verify sweep.
- Corrections to the previous report: the draft-policy GPU gate is now
  measured (was "pending"); FP8 K/V is implemented and measured (was
  "codec only"); Intel Arc moved from "blocked on A770" to "A770 purchased,
  in transit"; tandem numbers refreshed from the 2026-11-15 re-measurement.
