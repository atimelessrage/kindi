# Current worktree vs stock Kindi 0.6.0

Report date: **2026-11-14**. This file is the successor to
`CURRENT-VS-KINDI-0.5.md` §25: that file's §§1–24 are the history against
0.5.0 and §25 is the first evaluation against 0.6.0; this file carries the
current one. It separates verified behavior from recorded evidence,
incomplete features, and release-hygiene problems. It is not a claim that
everything described here has been re-run on current hardware.

## 1. Baseline

Upstream: tag `v0.6.0` (`179710c`, 2026-11-12) plus the post-tag upstream
head `dce89df` (container-registry docs, 2026-11-13). Both are ancestors of
HEAD `84f08a9`:

```sh
git merge-base --is-ancestor v0.6.0 HEAD   # yes
git merge-base --is-ancestor origin/main HEAD  # yes
git log --oneline v0.6.0..HEAD | wc -l     # 209 commits
git diff --shortstat v0.6.0..HEAD          # 360 files, +47,505/-454
git diff --diff-filter=D --name-only v0.6.0..HEAD  # (empty: zero deletions)
```

The 0.6.0 merge landed on 2026-11-14 (commit `f635513`, 18 conflicted files
over 172 diverged upstream commits) and has been carried forward since. The
tree vs `v0.6.0` is a **strict superset**: every upstream file is present,
and the fork adds behavior on top of it. Nothing upstream was deleted.

Full suite last measured at `49952e6`: **627 passed, 20 skipped, 2 setup
errors** (only `clang` missing on the host for the CPU-library build).
Focused checks at `84f08a9`: cascade + draft-policy + dtypes + docs-links
**23 passed**.

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
| FP8 host reference codec (`kindi/dtypes.py` `f32_to_fp8`) | round-to-nearest ties-to-even, all-pattern round-trip tests; groundwork for the #8 `KINDI_KV=fp8` KLD gate (REQ-0014) |
| Adaptive draft policy (`kindi/draft_policy.py`) | pure-python acceptance + confidence controller, 6 unit tests; GPU adoption gate pending (REQ-0015) |
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
- Tandem is still slower than GPU-only on 1 GPU (24K decode ~8 vs ~31–34 tok/s equivalent rows at 8K/24K); the overlap step-0 re-justified the worker thread (CPU compute 62.2 ms/token dominates at 24K on the 27B-24K row).
- FP8 K/V (#8) is gated on KLD ≤ q8_0 and unmeasured (codec only); #9 (Windows unbuffered load) measured NOT met; #6's ≤1.2× clause recorded unsatisfiable with the dedupe rule.
- Dense standard-attention llama/qwen GGUFs are refused (no engine): see §5.

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
done, M1 blocked on an A770 (`docs/INTEL-ARC.md`). Mixed-card pairs:
same-vendor layout implemented, GPU-unverified (`docs/MIXED-CARD-ROADMAP.md`).

**Speculation.** 2B drafter pays on the 27B (1.56x at 8K), does not pay on
the 9B (0.98x) or the 4B (0.88x), and does not fit at 24K on 8 GB — all
measured, not assumed (`docs/MTP.md`). The adaptive draft-policy
controller (REQ-0015) is the next step, gated on a GPU adoption run.

## 6. Evidence

- `git merge-base --is-ancestor v0.6.0 HEAD` and `origin/main HEAD` (§1 commands).
- The full suite at `49952e6` (627 passed, 20 skipped, 2 clang setup errors).
- Lease rows in `AGENTS.md`; requirement evidence in
  `docs/development/REQUIREMENTS-HISTORY.md`.
- `docs/ROADMAP.md`, `docs/EXTERNAL-ENGINES.md`, `docs/MTP.md`,
  `docs/INTEL-ARC.md`, `docs/MIXED-CARD-ROADMAP.md` for the per-item gates.
