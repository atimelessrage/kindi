# Current worktree vs stock Kindi 0.6.0

Report date: **2026-11-16** (refresh of the 2026-11-15 report; §36–§37 record
the T4.2 Gemma 4/DeepSeek port + engine-gateway audit, the Lightning gate
block, Lightning/Nanbeige structured output, and the True-MoE roadmap
labels). This file is the successor to
`CURRENT-VS-KINDI-0.5.md` §25: that file's §§1–24 are the history against
0.5.0 and §25 is the first evaluation against 0.6.0; this file carries the
current one. It separates verified behavior from recorded evidence,
incomplete features, and release-hygiene problems. It is not a claim that
everything described here has been re-run on current hardware.

## 1. Baseline

Upstream: tag `v0.6.0` (`179710c`, 2026-11-12) plus the post-tag upstream
head `dce89df` (container-registry docs, 2026-11-13). Both are ancestors of
HEAD `5b8330b`:

```sh
git merge-base --is-ancestor v0.6.0 HEAD   # yes
git merge-base --is-ancestor origin/main HEAD  # yes
git log --oneline v0.6.0..HEAD | wc -l     # 348 commits
git diff --shortstat v0.6.0..HEAD          # 402 files, +54,482/-1021
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
| Nemotron Lightning (`kindi/lightning*`) | present (merged) | **GPU-gated here 2026-11-16**: the T4.2 adapter is ported, and a clean catalog GGUF (size/SHA-256/GGUF verified, fresh pack) verifies 5 exact + 7 llama.cpp ties, 0 fail on the RTX 4060 Ti, with bench ctx 8192 read 788.3–821.3 and write 165.96–167.41 tok/s (§38). `kindi plan` describes its tiering header-only |
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
| **Backend plugin foundations** (`kindi/backends/`, REQ-0019) | T4.1 contract/registry/FakeBackend + T4.3 `kindi backends [--json]` listing (2026-11-15); CPU-only, importable on Python 3.6. T4.2 adapters for all five families landed 2026-11-16 (§36); Bonsai is GPU-gated on all four packs and Lightning for plain verify/bench on the 4060 Ti (§38); Nanbeige, Gemma 4 and DeepSeek V4.1 are CPU-tested with blocked GPU gates — so still no arbitrary-GGUF claim |
| **OpenAI error envelope + API conformance harness** (`kindi/api/errors.py`, `tests/api/`) | T5.1 2026-11-15: every server error is the OpenAI envelope; harness drives the real app with the official `openai` client, GPU-free |
| **Intel Arc A770 purchased** (REQ-0020) | one A770 16 GB bought 2026-11-15, in transit (not installed); M0 audit done, M1 arrival checklist ready, `arc-a770` registered as `not measured`. No backend code, no support claim |
| **Intel iGPU G0 slice** (`kindi/intel.py`, REQ-0021) | 2026-11-15: DRM-sysfs discovery, oneAPI-floor eligibility, shared-memory budget, tandem GPU contract; `KINDI_DEVICE=intel` refuses loudly. G1+ SYCL kernels need an 11th-gen+ iGPU + oneAPI to compile/verify |
| **Tandem worker-thread verdict** | step-0 re-measured 2026-11-15 (27B/24K, RTX 4060 Ti + 8086K): CPU 74.7 ms/token (6.2 ms/layer) at raw-kernel rate; head 17.6 ms + transfer 3.09 ms come first. **Worker thread not justified, should not be built** |
| **4B conversion provenance** | fresh `kindi convert` of the downloaded PTQ1_0 GGUF reproduces the 4B pack byte-identically (sha256); converted pack verifies 3+tie like the reference; 9B/27B pack verify re-run 4/4 (2026-11-15) |
| Requirement-based coverage + per-module floors | `tools/check_coverage.py` + `tests/coverage_floors.json` in CPU CI (audit M4); requirement-traceable CPU tests |
| **Server-side `/sessions` endpoints** (`kindi/server.py`, REQ-0016) | 2026-11-15: `GET /sessions` (status/tag filters), `GET /sessions/<id>`, `GET /sessions/<id>/export` (raw markdown), `POST /sessions/<id>/status`, `POST /sessions/<id>/tags` over the existing `SessionStore` — host-only (no engine/cache/GPU), API-key gated, enveloped 400/404, strict 12-hex id check so path-shaped ids never reach a file read. Pinned by `tests/test_server_sessions.py` (stub engine, no GPU) |
| **Loud rejection of unsupported `n` / `response_format`** (`kindi/server.py`, REQ-0018 decision D2) | 2026-11-15: `n` values other than 1 and any structured-output `response_format` type (`json_schema`, `json_object`) return an enveloped 400 before the engine or prompt cache is touched, instead of being silently ignored. `n = 1` and `{"type": "text"}` remain accepted. Pinned by malformed-request and default-acceptance tests in `tests/test_server.py` (stub engine, no GPU). Constrained decoding itself is still open (T5.4) |
| **Config reference scans subpackages** (`kindi/config_reference.py`) | 2026-11-15: `env_vars()` scans `kindi/**/*.py`, so env vars read in subpackages (`KINDI_BACKEND_MODULES` in `kindi/backends/`) appear in the generated reference with subpackage site paths. Audit triage of `Buffy_ReviewAudit_202611141517.txt`: the rest of its findings (N1-N6, D1/D3/D4, M3/M4) were verified fixed on the current tree; M1/M2 stay hardware-bound |
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

- The 0.6 backends (MoE / DeepSeek / DSpine) have **no GPU run record in this session** — first check with real hardware: `kindi verify` with `KINDI_USING=dsv41`, `KINDI_BACKEND=lightning`, `KINDI_DRAFTER=dspine` on the rig. **Lightning closed 2026-11-16 (§38); the other three are blocked on weights/corrupt uploads, not code.**
- `install.ps1` has never been run on Windows hardware.
- Tandem is still slower than GPU-only on 1 GPU (24K decode ~8 vs ~31–34 tok/s equivalent rows at 8K/24K); the overlap step-0 (re-measured 2026-11-15) puts the head (17.6 ms/token) and boundary transfers (3.09 ms) ahead of any worker thread.
- FP8 K/V (#8) is measured **lossy, not lossless** (KLD ~9–14× q8_0, verify 3/4) — an experimental opt-in, not a default; #9 (Windows unbuffered load) measured NOT met; #6's ≤1.2× clause recorded unsatisfiable with the dedupe rule.
- Dense standard-attention llama/qwen GGUFs are refused (no engine): see §5.
- Intel Arc: A770 purchased and in transit (M1 unblocked on arrival); no SYCL backend exists, and the iGPU route is G0-only until an 11th-gen+ iGPU + oneAPI toolchain exists to compile/verify kernels.
- Backend plugin interface: T4.2 engine adapters are ported for all five families (Bonsai, Nanbeige, Lightning, Gemma 4, DeepSeek V4.1) with the Bonsai and Lightning gates measured and the other three GPU-gated by missing/corrupt weights; T4.4 llama.cpp proxy and real concurrency remain planned. REQ-0018 structured output is landed for dense, Gemma 4, Lightning and Nanbeige (CPU-tested on the last two, GPU gates unclaimed).
- PQ2_0/Q2_0 (type 142/42) import stays verified for the 2B only — the 4B/9B/27B files were never downloaded (owner action, `docs/MANUAL-HELP-NEEDED.md` §2).
- Recursive GBNF is **opt-in and bounded**, not exact: `KINDI_GBNF_MAX_DEPTH=N` unrolls N levels (a fresh copy per reference, deeper branches pruned) and `tests/test_gbnf_recursion.py` checks the resulting language against an independent matcher at every allowed depth. Because copies are not shared, llama.cpp's `json.gbnf` reaches array nesting 2 inside the default 20,000-state limit (398,268 states at depth 7), so `response_format: json_schema` stays the unbounded path for real JSON.

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

## 12. What changed in the 2026-11-15 improve-and-fix round

- Sessions endpoints: concurrent status/tag mutations are serialized
  (a shared lock around the read-modify-write), writes use unique tmp
  files, and `list()` closes its handles (`08a954c`).
- D2 validation: 400s now name the field in the envelope `param`
  (`n`, `response_format`, `logprobs`) instead of `null` (`1b6177e`).
- Tandem `rate` planner: the log no longer prints a fake
  `predicted ~0.0 tok/s` when the calibration lacks the picked count
  (`8f1587f`).
- Config reference: the scanner also reads `os.getenv(...)` forms;
  recursive scan + subpackage sites were landed in the prior round
  (`ed6a2d9`).
- Docs updated alongside: `docs/API.md`, REQ-0016/REQ-0018 evidence,
  ROADMAP bullets, CHANGELOG.
- Diff vs `v0.6.0` grew 398 files (+52,813 → +52,946 insertions);
  still zero file deletions.
- §13: the second improve-and-fix pass — sessions export filename,
  scanner `setdefault` coverage, `load_ratio` robustness tests.

## 13. What changed in the 2026-11-15 improve-and-fix round 2

- Sessions export: `Content-Disposition` filename (`<date>-<slug>.md`,
  same convention as the CLI `write_export`) (`449f9f2`).
- Config reference: the scanner also reads `os.environ.setdefault(...)`
  (a var introduced only via setdefault would otherwise vanish);
  reference regenerated at 191 environment reads (`54542ed`).
- Tandem: `load_ratio` robustness pinned by tests — wrong-kind and
  corrupt calibration files return `None` (fit-only fallback)
  (`34647cd`).
- Docs updated alongside: `docs/API.md` (export row), REQ-0016
  evidence, CHANGELOG.
- Diff vs `v0.6.0` grew 398 files (+52,946 → +53,008 insertions);
  still zero file deletions.
- §14: the performance deepen round — tandem owner-table dispatch.

## 14. What changed in the 2026-11-15 performance deepen round

- Tandem decode: the per-layer owner table is resolved once at
  construction instead of per layer per token (`frozenset` rebuild +
  set lookup removed); routing behavior identical, pinned by the
  fake-backend call-order tests, the probe-parity test, and a new
  `_owners` invariant test (`f720a5d`).
- Dispatch overhead measured on the fake backends (Python overhead
  only): 19.34 µs → 17.57 µs per token (×0.91); against the measured
  27B @ 24K budget (~103 ms/token) the saving is noise — kernels
  dominate, and the step-0 verdict (CPU at the raw-kernel rate, no
  dispatch gap) stands. Recorded in `docs/TANDEM-STATUS.md` and the
  CHANGELOG.
- Diff vs `v0.6.0` grew 398 files (+53,008 → +53,062 insertions);
  still zero file deletions.
- §15: session resume in the chat loop (REQ-0016), plus a full-suite
  re-measurement (779 passed, 11 skipped).

## 15. What changed in the 2026-11-15 roadmap round (session resume)

- `kindi chat --resume SESSION_ID` lands the last open REQ-0016 queued
  item before storage growth: the REPL is seeded with the stored turns
  and appends new turns back to the same record on exit
  (`SessionStore.append_messages`, one save, bumps `updated`;
  malformed/unknown ids refuse loudly); the seed/append round trip and
  the refusals are pinned by tests (`2b36d19`).
- Full suite re-measured at `24b4a2f` with clang on PATH:
  **779 passed, 11 skipped, 0 failed** (one environment error without
  clang, the documented case); AGENTS.md current-state updated.
- Only REQ-0016 item left open: unbounded session storage growth.
- Diff vs `v0.6.0` grew 398 files (+53,062 → +53,181 insertions);
  still zero file deletions.
- §16: session retention pruning closes REQ-0016 completely.

## 16. What changed in the 2026-11-15 roadmap round (session retention)

- `SessionStore.prune` + `kindi sessions prune --days N [--all]
  [--dry-run]`: by default deletes only `resolved` sessions not
  updated for N days; `--all` widens the scope; `--dry-run` lists
  without deleting; unparseable timestamps are never pruned; negative
  days are refused. Pinned by store and CLI tests (`27590c1`).
- REQ-0016 (chat sessions, experimental) is now fully implemented:
  store + CLI + triage + tags + raw-markdown export/import +
  server-side `/sessions` endpoints + resume + retention.
- Diff vs `v0.6.0` grew 398 files (+53,181 → +53,292 insertions);
  still zero file deletions.
- §17: stop-sequence O(n²) fix + CPU profile baseline.

## 17. What changed in the 2026-11-15 performance deepen round 2

- **Fix:** the generation loop scanned the whole accumulated reply for
  stop strings every token (O(n²) over a generation). It now scans only
  the not-yet-checked tail via a `stop_from` cursor (O(window) per
  token), proved correct by induction (new matches can only appear in
  the tail; thinking freezes the cursor so pre-think matches are still
  found). Isolated check at 20K tokens: 0.088 s → 0.018 s (4.8×);
  spanning and thinking cases pinned by tests driving the real loop
  (`549ac52`).
- **Measured and recorded:** the CPU engine's Python overhead is
  0.000 s over 1.027 s of feeds (all inside the `kc_step` C call;
  27.8 ms/feed on the 2B at 512 ctx, i7-8086K). The CPU path is
  kernel-bound; deeper CPU speed now means `kcpu.c` kernel work
  measured against this baseline (`docs/BENCHMARKS.md`).
- Diff vs `v0.6.0` grew 398 files (+53,292 → +53,420 insertions);
  still zero file deletions.
- §18: the stop fix extended to the Gemma, Lightning and Nanbeige
  generation loops; `dsv41` is already incremental.

## 18. What changed in the 2026-11-15 generation-loop stop fix (round 2)

- The `stop_from` cursor fix from §17 is now applied to the three
  remaining generation loops that had the identical whole-text scan:
  the Gemma path in `kindi/server.py` (with the same
  first-in-configured-order + first-occurrence truncation), and the
  Lightning (`kindi/lightning_gen.py`) and Nanbeige
  (`kindi/nanbeige_gen.py`) loops (boolean check).
- `dsv41_server.py`'s `Stream` already searches incrementally in a
  leftover buffer and was verified not affected.
- Diff vs `v0.6.0` grew 400 files (+53,420 → +53,480 insertions);
  still zero file deletions.
- §19: sessions stats for ops visibility before retention pruning.

## 19. What changed in the 2026-11-15 roadmap round (sessions stats)

- `SessionStore.stats()` + `kindi sessions stats [--json]`: sessions by
  status, message totals, bytes on disk, oldest/newest last update —
  the visibility needed before pruning (`9c59e2b`). Store and CLI tests
  pin the empty store, the counts, and the JSON shape.
- Diff vs `v0.6.0` grew 400 files (+53,480 → +53,567 insertions);
  still zero file deletions.
- §20: tandem split timings on `/metrics`.

## 26. What changed in the 2026-11-15 T4.2 bonsai adapter round

The dense Ternary-Bonsai-2 engine now lives behind the backend contract as
the first T4.2 family adapter (`kindi/backends/bonsai.py`, registered as
`bonsai`; REQ-0019 T4.2 partial). `kindi.engine.make_engine` resolves
dense Bonsai packs through the registry by default and routes to the
pre-port factory as `make_engine_legacy` when `KINDI_LEGACY_ENGINE_PATH=1`
is set — the adapter builds the same engine object the legacy path built,
so serve/verify/bench numbers are unchanged by construction. The contract
harness skips the text-primitives test for backends that declare
`REQUIRES_PACK` (lifecycle/info still run packless); `kindi backends` and
the registry not-found message now list the bonsai adapter. Verified on
the RTX 4060 Ti, 2026-11-15: `kindi verify` 2B is 4/4 on the adapter path
(103.0 tok/s) and the valve path (103.2 tok/s, timing jitter); `kindi bench`
2B through the adapter reads 97.3/91.5/86.3/81.3 tok/s at 2K/8K/16K/24K
positions (exact head). Nanbeige, Nemotron Lightning and DeepSeek V4.1
remain on the legacy path until packs exist to verify against; the
agreed-cards (P4000/M4000/RTX4000) verify gate stays hardware-bound.

- Diff vs `v0.6.0` grew 402 files (+54,055 → +54,482 insertions),
  348 commits (344 → 348); still zero file deletions.

## 27. What changed in the 2026-11-15 JSON-schema constrained decoding round

`response_format` `json_object` and `json_schema` now constrain the dense
decode instead of being rejected (REQ-0018; decision D2's rejection round
made JSON structured output a 400 until this landed). New
`kindi/jsongrammar.py`: a schema-ordered deterministic automaton
(properties in schema order, `required` enforced, strict
`additionalProperties`, canonical integers, prefix-language number
rejection, enum/const primitives) with a per-state full-vocabulary allowed
set (the 248K enumeration is ~0.6 s, cached per state). `kindi/sampling.py`
gains the `allowed` intersection and `kindi/server.py` accepts the two
structured formats (malformed schemas, `grammar` and unknown types, and
`tools` with structured output stay enveloped 400). The GPU gate found
real dead automaton states on the first 2B run (empty allowed sets) —
`otail`-comma with no properties left, number-prefix dead corners, and
required-prefix key buffers — all fixed and pinned by
`tests/test_jsongrammar.py` (23). Verified on the RTX 4060 Ti,
2026-11-15: 2B `{"name": "Alex", "age": 28}` and `json_object`
`{"city": "New York"}`, 9B with an array property — all parse against
the schema with keys in schema order and `finish: stop`. Gemma 4 and
Lightning/Nanbeige decode paths are untouched.

- Diff vs `v0.6.0`: 402 files (+54,482 → +54,512 insertions),
  354 commits (348 → 354); still zero file deletions.

## 28. What changed in the 2026-11-15 Gemma4 constrained-decoding round

The structured-output constraint from §27 now also covers the second served
family: `_generate_gemma` wires the same JSON automaton (`Kindi._g4_json_token`
— the best grammar-legal token from the full Gemma vocabulary, greedy or
tempered; EOS/stop tokens are admitted only once the automaton is done),
and structured output takes over the whole reply (no thought channel is
entered). `supports_json_grammar` names the two constrained backends
(dense + Gemma 4) at the HTTP boundary, so a Lightning/Nanbeige
structured-output request is rejected loudly (400, `param:
response_format`, naming the backend) instead of silently unconstrained —
before this round the validator accepted it globally and the two `*_gen`
loops just ignored it. (2026-11-16: the two `*_gen` loops now advance the
same `JsonGrammar`, so `supports_json_grammar` names all four; see REQ-0018.) CPU end-to-end: a scripted illegal top logit is
skipped and the reply parses as `{"ok": true}`. **No Gemma GPU gate was
run** (no Gemma pack on the host); the dense 2B/9B gate in §27 is the
measured one. `jinja2` (a real Gemma runtime dep) was missing in the cloud
env and installed to run the new test.

- Diff vs `v0.6.0`: 404 files (+54,512 → +55,343 insertions),
  357 commits (354 → 357); still zero file deletions.

## 29. What changed in the 2026-11-15 wire-coverage round

Test-only round: the live-HTTP conformance suite (`tests/api/`, real app
over uvicorn driven by the official `openai` client) never sent a
`response_format`, and covered no `/sessions` route — so the headline
feature of §27/§28 had no wire-level acceptance test and the sessions HTTP
server was stub-only verified. New `tests/api/test_sessions_roundtrip.py`
(list/show/export/status/tags/read + malformed-id 400 over the real app)
and structured-output cases in
`test_openai_client_conformance.py` (openai-client accept + record, and
the loud-400 case for a backend without the grammar, via a stub flag).
The conformance file header still names T5.1 only — a follow-up should
mention the round; not in this commit to keep it test+docs.

- Diff vs `v0.6.0`: 404 files (+55,343 → +55,551 insertions),
  360 commits (357 → 360); still zero file deletions.

## 30. What changed in the 2026-11-15 doc-consistency round

Docs-only: `docs/BACKENDS.md`'s intro still said only the fake backend
resolves (the T4.2 Bonsai adapter shipped 2026-11-15 and `kindi backends`
lists it) and mislabeled the shipped T4.3 authoring guide as future work;
the intro and status now describe the actual registry (fake + bonsai +
external). `docs/AGENTS.md` — the page agent hosts actually read — never
mentioned `response_format`; it now documents structured output, its two
carrying backends, and the loud 400 elsewhere. ROADMAP's T5.1 bullet also
still listed structured output (T5.4) as planned; it now says it landed for
dense + Gemma 4 and stays planned only for Lightning/Nanbeige.

- Diff vs `v0.6.0`: 405 files (+55,551 → +55,754 insertions),
  363 commits (360 → 363); still zero file deletions.

## 31. What changed in the 2026-11-15 T4.2 4B/9B gate round

Measurement only (no code): the Bonsai adapter's GPU gate had covered only
the 2B. Extended to the other two packs on the host (RTX 4060 Ti,
2026-11-15): verify 4B PASS 3+tie on both the adapter and the
`KINDI_LEGACY_ENGINE_PATH` valve (141.0/143.1 tok/s — the tie is the known
prompt-3 llama.cpp top-2 gap 0.0012), 9B ALL PASS 4/4 on both (99.3/99.6
tok/s); adapter bench at ctx 24576 (4B read 161.0, gen
132.9/137.4/125.6/114.1/106.8; 9B read 109.2, gen 92.6/93.8/89.5/83.5/78.7)
agrees with the 9B valve row within ~1%. Recorded in
`docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-t42-bonsai-4b-9b/RESULTS.md`
(+ machine-readable bench JSON). No 27B pack on the host, so no 27B
adapter row is claimed; the other T4.2 families stay unported.

- Diff vs `v0.6.0`: 405 files (+55,754 → +55,787 insertions),
  365 commits (363 → 365); still zero file deletions.

## 32. What changed in the 2026-11-16 Nanbeige adapter round

The T4.2 Nanbeige adapter (`kindi/backends/nanbeige.py`, registered
`nanbeige-33b`) is ported and CPU-tested (registry resolve/load-reject/
manifest-id tests, CLI listing, Python 3.6 grammar) — 27 passed, 2 skipped.
Its GPU gate is **blocked**: the catalog's `ahtx/AH-Nanbeige4.3-33B-IQ3_XXS.gguf`
downloaded in full (13,108,685,952 bytes) is corrupt as served by Hugging Face
(SHA-256 `39d8e2d8…` vs the catalog's `9d4f18e8…`; only 5 of 471 tensor-info
names parse as text, the rest NUL-garbled). The corrupt copy was deleted and
the owner action recorded in `docs/MANUAL-HELP-NEEDED.md`. **No verify/bench
parity is claimed**; `make_engine` previously built a wrong dense engine for
Nanbeige packs and now returns the Nanbeige engine through the adapter.

- Diff vs `v0.6.0`: 405 files (+55,787 → +55,830 insertions),
  368 commits (365 → 368); still zero file deletions.

## 33. What changed in the 2026-11-16 Bonsai-27B adapter-gate round

The Bonsai adapter's local gate is now complete for all four packs on this
host. The earlier record said the 27B pack is not on this host; it is in fact
at `T:/kindi-packs/bonsai-27b`, but its manifest's `model` is the 27B GGUF's
placeholder `Hf`, so `backends/bonsai.model_id_for_manifest` returned `None`
and the pack silently took the legacy path — the adapter already listed
`27b` in `SIZES`/`model_ids`. Fixed: `model_id_for_manifest` now also reads
the size from `source_file` (the fallback `kindi/config.py` uses to name the
served model), pinned by a `tests/backends/test_registry.py` case. With the
fix, `kindi verify` 27B is **4/4 PASS on both the adapter path (34.5 tok/s)
and the valve path (35.7 tok/s)**, and `kindi bench` at ctx 24576 agrees
within ~1% everywhere (read 36.7/37.1; gen identical at 8K/16K/24K).
Evidence: `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-t42-bonsai-27b/RESULTS.md`.

- Diff vs `v0.6.0`: 412 files (+55,830 → +56,331 insertions),
  377 commits (368 → 377); still zero file deletions.

## 34. What changed in the 2026-11-16 Lightning adapter-port round

The T4.2 Lightning adapter (`kindi/backends/lightning.py`, registered
`nemotron-lightning-30b`) is ported and CPU-tested (registry
resolve/load-reject/manifest-id tests, CLI list updated, Python 3.6
grammar) — 32 passed, 3 skipped. Its GPU gate was **pending** at write
time (the catalog GGUF, 9,472,656,320 bytes, was still downloading);
**superseded by §38**, which closes that gate with a clean, hash-checked
copy. `make_engine` previously built a
wrong dense engine for Lightning packs and now returns the Lightning
engine through the adapter. The Bonsai-27B round's REQ-0019 change also
got its missing REQUIREMENTS-HISTORY regen in a separate commit.

- Diff vs `v0.6.0`: 415 files (+56,331 → +56,478 insertions),
  379 commits (377 → 379); still zero file deletions.

## 35. What changed in the 2026-11-16 DEC-0003 re-measurement + tests/api cleanup + MANUAL-HELP-NEEDED rounds

Three smaller rounds on the same day, none of them a feature: (1) **DEC-0003
re-measured:** with the `fd_supported` guard bypassed, `KINDI_TC=1
KINDI_FD=1 KINDI_ATT_TC=1` on the 4B is 2 exact / 2 fails (prompt 2 token 6,
gap 0.0447; prompt 3 token 37) — exactly the 2026-11-11 points, so the
4B flash-decode guard stays and no speed gain was recorded (DECISIONS.md
entry + evidence in `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-fd-4b/`).
(2) **`tests/api` wire-test cleanup:** one `_serve` context manager replaces
the duplicated uvicorn start/stop, the sessions fixture moved into conftest
(patching `_make_engine` itself), `stub`/`no_grammar` fixtures removed the
vacuous assert and inline flag flipping, and the sessions wire file dropped
cases already pinned in-process (net −14 lines; 13 passed). (3)
**`docs/MANUAL-HELP-NEEDED.md` expanded to the full owner-action inventory**:
downloads (Nanbeige corrupt file, PQ2_0/Q2_0 quants, a pinned llama-server),
other-hardware rigs (A770 M1, P106/P4000/M4000/RTX4000 pairs, Blackwell FP8,
AVX-only CPUs, mixed cards), publications (push 403, installer refresh, GPU
CI runner), and maintainer decisions (T4.4/T4.5, D3/D5). The Lightning T4.2
gate was still the agent-side open item at write time (GGUF downloading at
~2.3 MB/s); §38 closes it later the same day.

- Diff vs `v0.6.0`: 416 files (+56,478 → +56,909 insertions),
  384 commits (379 → 384); still zero file deletions.

## 36. What changed in the 2026-11-16 T4.2 Gemma 4 + DeepSeek V4.1 + gateway rounds

The last two T4.2 families are now ported (CPU-tested; **both GPU gates
blocked**): Gemma 4 (`kindi/backends/gemma4.py`, registered
`gemma4`/`gemma4-e4b`, architecture `gemma4`) wraps `gemma4.Engine` /
`gemma4_tp.TPEngine`, the Gemma tokenizer's own stop ids, the GGUF chat
template through `gemma4_chat`, batched prefill at the KV ring's 512, and
greedy or sampled token selection from the full logits; DeepSeek V4.1
(`kindi/backends/dsv41.py`, `deepseek-v41-flash`, architecture
`deepseek41`, device `cpu`) wraps `kindi.dsv41.Model` from a hand-written
pack (`backend: kindi_dsv41`, `model_path`, optional `encoding`) — there is
no converter, so the DSpark drafter, RAM slots, prompt cache and turn saves
stay in the standalone server. The one sampler chain for adapters is now
`sampling.pick_from_logits` (top-k/top-p/min-p/temperature via
`distribution`, masked argmax at 0), and the two-card rule lives in
`backends.gemma4.two_cards()` with `server._two_cards` delegating.
`kindi bench` / `kindi verify` name the Gemma 4 / DeepSeek families they
cannot drive instead of falling into the dense loop or the llama.cpp
matrix. **Gates, NOT run:** no Gemma 4 GGUF/pack on this host and no
DeepSeek V4.1 shards or release gate data (`gate_prompts.jsonl` +
`gate_ref.txt`) on this host; no verify/bench parity is claimed for either.

The engine gateway was audited the same day and now refuses to build the
wrong engine silently: a pack whose manifest names a family with no adapter
(`kindi_cascade`, an unknown one) gets a `SystemExit` naming the pack, the
family and the fact that the dense builder would be the wrong engine; a
manifest claimed by two adapters is refused as contradictory; the registry
refuses an equally-specific two-backend match as ambiguous instead of
picking by registration order. The one legitimate fallback stays visible: a
dense pack its own adapter does not claim (the placeholder-`model` "Hf"
case) still uses the legacy builder but logs `no adapter claims it` and
every adapter build logs `-> the <name> backend (<model id>)`. The
`KINDI_LEGACY_ENGINE_PATH=1` valve now refuses a named non-dense family
instead of handing it to the dense builder. The new
`tests/test_engine_gateway.py` harness asserts every pack under
`~/.kindi/packs` resolves to the adapter its manifest declares (5/5).
Full suite after these rounds: **870 passed, 28 skipped** (no clang on
PATH, so the libkcpu tests skip); 3.6 grammar OK; `project_memory
--check`, `git diff --check` clean.

- Diff vs `v0.6.0` (committed HEAD `f2e16eb`): 416 files,
  +56,988/−1,023, 388 commits; still zero file deletions.

## 38. What changed in the 2026-11-16 Lightning clean-run gate round

The Lightning T4.2 gate is **closed on this host**, superseding §34, §35 and
§37(1). A clean re-download to
`L:/kindi-lightning-clean/models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`
matches the catalog exactly: 9,472,656,320 bytes, SHA-256
`38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3`, valid
GGUF magic, `nemotron_h_moe` with 401 tensors. A fresh pack was converted
from that file and the RTX 4060 Ti ran the gate:

- `plan --ctx 8192`: MoE expert tiers, ~5.57 GiB on the card, 70–106 of 128
experts per layer on the card.
- `verify`: **5 token-exact + 7 llama.cpp ties, 0 fail** over 12 prompts
(74 s).
- `bench --ctx 8192`: read 788.3–821.3 tok/s, write 165.96–167.41 tok/s.

Evidence: `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`.
This is a plain-generation gate (verify + bench); the Lightning
structured-output path stays CPU-tested only, and no other context or
mixed-card parity is claimed. The remaining T4.2 GPU gates are Nanbeige
(corrupt upstream GGUF), Gemma 4 (no E4B pack on this host) and DeepSeek
V4.1 (no shards or release gate data).

## 37. What changed in the 2026-11-16 structured-output + Lightning-block + MoE-label rounds

(1) **Lightning gate moved from "pending download" to blocked:** the local
catalog GGUF (`boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`,
9,472,656,320 bytes) is not usable on the current host. The evidence chain
disagrees three ways: catalog SHA-256 `38854de6…`, current local file
`bc22bc…` (same size, first 1 MiB zero, rejected by the GGUF reader), and
pack-manifest `source_sha256` `2ec6d074…`. The convert/verify attempt is
therefore discarded as parity evidence and no verify/bench claim is made.
This proves the present local copy is bad; it does not prove whether the
upstream bytes or the local transfer/write are at fault. The T4.2 adapter
stays CPU-tested; the next step is a clean re-download, a size/SHA/GGUF
check against the catalog, and a fresh pack conversion from that file.
**Superseded by §38:** that clean re-download matched the catalog hash and
the fresh pack passed the 4060 Ti verify/bench gate.
(2) **Structured output on Lightning/Nanbeige (REQ-0018):** both `{x}_gen`
generation loops now advance the shared `JsonGrammar` — the whole reply is
the document (no thought channel), Nanbeige drafting/speculation is
disabled while structured output is active, and stop tokens are admitted
only once the automaton is done. A scripted illegal top logit (`<`) is
skipped in both loops and the reply parses as `{"ok": true}`;
`supports_json_grammar` now names dense, Gemma 4, Lightning and Nanbeige.
CPU-only evidence (fake engines/tokenizers); no GPU gate claimed.
Checks for this round: tool-call conformance + server + backends +
backends_cli **122 passed**; docs-links + project-memory + handoff
**12 passed**; `project_memory --check`, `git diff --check` clean.
(3) **True-MoE roadmap labels:** the Small-MoE table now names
**Gemma 4 26B-A4B (MoE)** and **Qwen 3.6-35B-A3B (True MoE)** explicitly;
both remain recognition-only/refused (no catalog/installer/support-matrix
entry before a pinned-reference + token-exact `kindi verify` gate).

- Diff vs `v0.6.0` including the uncommitted worktree: 416 files,
  +57,922/−1,076; committed HEAD stays +56,988/−1,023; still zero file
  deletions. The uncommitted slice is 28 tracked files (+1,112/−231) plus
  four untracked files (`kindi/backends/gemma4.py`,
  `kindi/backends/dsv41.py`, `tests/test_engine_gateway.py`, and the
  FD-bisect note).

## 25. What changed in the 2026-11-15 full-suite fix + CPU parity round

A full-suite run (783 passed, 20 skipped, 1 failure, 2 fixture
errors) found: the generated configuration reference was stale
after the sessions edits → regenerated (191 reads); and
`test_server_cpu` errored rather than skipped on hosts without a
compiler → it now skips when the libkcpu cache is cold and no
gcc/clang is present (pass case re-verified with clang). New
cross-ISA parity property test: all five tier libraries built and
loaded run a fixed decode on the 2B — greedy tokens identical
across tiers; hidden states within ~3e-4 summation-order
tolerance (byte-identical hidden is false by construction).

- Diff vs `v0.6.0` grew 401 files (+53,986 → +54,055 insertions);
  still zero file deletions.

## 24. What changed in the 2026-11-15 sessions protection round

Sessions records now carry `protected` and `unread` flags.
`unread` is raised by record/append/import and cleared by
`kindi sessions show` or `POST /sessions/<id>/read`; `protected`
is set by `kindi sessions protect SESSION_ID [--off]` and makes
prune keep the record (flagged in dry runs and in `stats`).
List/triage show `[unread]`/`[protected]` markers; the server
summaries and both rounds of API docs carry the flags. This
closes the REQ-0016 queued `unread` + ownership items.

- Diff vs `v0.6.0` grew 401 files (+53,830 → +53,986 insertions);
  still zero file deletions.

## 23. What changed in the 2026-11-15 ReBAR 9B measurement

ReBAR is on (BAR1 8192 MiB, PCIe Gen3 x8 slot, driver 617.42).
9B verify matrix: plain 4/4 @ 73.1 tok/s (bench 61.4 @ ctx 24576),
tc 4/4 @ 114.7 tok/s. The tc pass is the Hadamard-4096 HP path
landing, not ReBAR; plain is −7.9% cross-day vs the pre-ReBAR
2026-11-10 cell (79.4) — no resident-compute gain, as expected for
a fully resident 2 GB model. A BIOS-toggle A/B is documented as
the follow-up; the 27B spill paths are where ReBAR should show.
Evidence: `docs/benchmarks/2026-11-15-i7-8086k-rtx4060ti-rebar-9b/`.

- Diff vs `v0.6.0` grew 401 files (+53,760 → +53,830 insertions);
  still zero file deletions.

## 22. What changed in the 2026-11-15 legacy Fermi path

Fermi / pre-Kepler cards now classify as `fermi` in the installer
rather than the old `unsupported` bucket. They offer no GPU models
or pairs, skip the NVIDIA container toolkit setup when they are
the only GPUs, and the installer offers the CPU engine on an AVX
CPU. This is a graceful-degradation path: no kernel compiles for
sm_21, so no CUDA support is claimed.

- Diff vs `v0.6.0` grew 400 files (+53,722 → +53,760 insertions);
  still zero file deletions.

## 21. What changed in the 2026-11-15 ten-card rig support pass

Two installer name-fallback fixes (old drivers that do not report
compute capability), found by extending the fake-GPU tests to the
full rig list: RTX 50 (Blackwell) had no rule and fell through to
6.1 — now 12.0; the pre-existing `RTX 40` rule reported the Turing
RTX 4000 as Ada 8.9 — now a dedicated RTX 4000 → 7.5 rule wins
before it. Support-matrix slugs added for the 5060 Ti, 940M and
GT 1030 (rendered not measured; no new measured rows because only
the 4060 Ti is present on this host). The Fermi GT 430 stays
unsupported (below the installer's sm_30 floor; the CUDA 10.2
image cannot run it).

- Diff vs `v0.6.0` grew 400 files (+53,616 → +53,722 insertions);
  still zero file deletions.

## 20. What changed in the 2026-11-15 tandem observability round

- `/metrics` now emits `kindi_tandem_tokens_total` and
  `kindi_tandem_{cpu,gpu,transfer,head,total}_ms_per_token` plus
  `kindi_tandem_transfers_per_token` whenever the tandem probe
  (`KINDI_TANDEM_DEBUG_TIMING`) is on; absent when it is off
  (`3f9245f`).
- Diff vs `v0.6.0` grew 400 files (+53,567 → +53,616 insertions);
  still zero file deletions.

## 11. What changed in the 2026-11-15 second audit triage

- Triaged `Buffy_ReviewAudit_202611151307.txt` (audited `f07d42b`): secrets
  clean, prior dispositions verified. Remaining items are process/hardware:
  fork lag (push 403, `myfork` 219 commits behind), GPU CI runner, T4.2
  adapters, AVX-512, a real FP8/BF16 model, Windows installer.
- One audit claim is stale: the engine no longer splits columns 50/50 — a
  proportional `plan_split` with `KINDI_TP_SPLIT=auto` (installer sets it
  for unequal pairs) is implemented and pinned by tests; only mixed-pair
  GPU verification is hardware-bound.
- Hygiene: `AGENTS.md` Current-state line refreshed — full suite 773 passed,
  11 skipped (clang on PATH) at `ec31c76`; superset 398 files, +52,813/−993.
- Diff vs `v0.6.0`: still 398 files, zero file deletions.

## 10. What changed in the 2026-11-15 audit triage round

- Triaged `Buffy_ReviewAudit_202611141517.txt` (audited HEAD `549c030`)
  against the current tree: N1-N6, D1/D3/D4, M3 and M4 are verified fixed;
  M1 (GPU CI runner) and M2 (P106/P4000/M4000/RTX4000 rows) remain
  hardware/process-bound and are already recorded as such in `docs/CI.md`
  and the support matrix.
- One still-open instance of the audit's N2/N3 class was real and fixed:
  the config-reference scanner missed subpackage reads. Reference
  regenerated (+`KINDI_BACKEND_MODULES`, 190 env reads), regression test
  added, CHANGELOG entry added.
- Diff vs `v0.6.0` grew 398 files (+52,755 → +52,794 insertions);
  still zero file deletions.

## 9. What changed in the 2026-11-15 unsupported-field rejection round

- New beyond-0.6 behavior: `n > 1` and structured-output
  `response_format` are rejected with an enveloped 400 (REQ-0018, decision
  D2); previously both were silently ignored. No generation, sampler or
  kernel code changed.
- Documentation updated alongside: `docs/API.md` request-field table and
  D2 status, REQ-0018 evidence, ROADMAP agent-API bullet, CHANGELOG.
- Diff vs `v0.6.0` grew 398 files (+52,691 → +52,755 insertions);
  still zero file deletions.

## 8. What changed in the 2026-11-15 sessions round

- New beyond-0.6 feature: server-side `/sessions` endpoints (REQ-0016),
  CPU-tested with a stub engine (55 tests with `test_server.py` +
  `test_sessions.py`). No engine, sampler or kernel code changed.
- Documentation updated alongside: `docs/API.md` (sessions contract),
  `docs/AGENTS.md` (native endpoints), REQ-0016 evidence, ROADMAP
  (chat-session section marked landed; stale agent-API conformance bullet
  corrected), CHANGELOG.
- Diff vs `v0.6.0` grew 397 → 398 files (+52,364 → +52,691 insertions);
  still zero file deletions.

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

## 39. What changed in the 2026-11-17 rig review (worse / missing / unverified)

Run on the RTX 4060 Ti rig (`NVIDIA GeForce RTX 4060 Ti`, 8188 MiB, driver 617.42).

| item from §4 | state on 2026-11-17 |
|---|---|
| full suite / handoff count | **953 passed, 28 skipped, 0 failed** (121 s) at `b613264`, `bench-logs/full-suite-2026-11-17.log`. The handoff still said 886/28 from 2026-11-16; the difference is the rounds since (MoE chunk/tandem, GBNF, plan report). |
| **GPU structured-output gate** (previously CPU-only, "GPU gate unclaimed") | **closed on the 2B**: live server, `response_format` GBNF — two ISO-8601 requests returned `2025-01-15T12:00:00Z` and `2026-05-06T14:30:00Z`, both matching the date automaton with `finish: stop`; an enum grammar returned `"blue"` in language; `json_schema` returned `{"name": "thing", "age": 0}` with both required keys in schema order. An unknown rule and the oversized `{1,100000}` both answered **400 `param: response_format`** before generation. Script: `bench-logs/grammar-gate.py` (gitignored, with the server log). |
| tandem slower than GPU-only on one card | unchanged and not a bug: `step-0` overlap puts the head and boundary transfers ahead of any worker. Still the largest known performance gap vs a single-card run; hardware/tuning-bound. |
| `install.ps1` never run on Windows | **still unverified.** Running it replaces the user's installed engine, so it needs a throwaway box/profile rather than this rig. |
| PQ2_0/Q2_0 4B/9B/27B | still owner-bound: only the 2B variants are on this host (`~/.kindi/packs/bonsai-2b{,-pq2,-q2}`); 4B/9B packs here are PTQ1_0. |
| dense standard-attention llama/qwen GGUFs refused | missing engine, not a regression: unchanged, and out of scope for this rig (the 0.6 backend set does not cover them either). |
| MoE / DeepSeek / DSpine GPU gates | blocked on weights (no MoE pack, no DeepSeek shards, no DSpine profile) and the Nanbeige/Lightning upload problems; not addressable on the rig. |
| performance sanity (is anything slower than the 0.6.0-era records?) | no regression found on the 2B: `kindi bench` at ctx 8192 with today's defaults (plan engine, q8_0 K/V) reads **298.2 tok/s** and generates **221.7 / 220.2 / 226.8 tok/s** at positions 100 / 2,000 / 8,000. The 2026-11-09 4060 Ti row for the same pack is 190.0 read and 161.5-165.5 generation; the settings differ, so this is a floor that rules out a regression, not a new record. 6,523 MiB free at 8,192 context. |

Two fork surfaces verified on the rig in earlier rounds remain unchanged: `kindi plan`
(model-free) and the MoE scheduler reports, which move no bytes.
