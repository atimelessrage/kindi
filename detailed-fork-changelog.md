# Detailed fork changelog

Round-by-round history moved verbatim from `CURRENT-VS-KINDI-0.7.md` on 2026-11-18.
The live comparison now keeps only the current 0.6.1 status; these sections retain
their original numbering, order and evidence, including the round notes that were
written against earlier baselines. They are a record, not a second current report.

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
at `~/.kindi/packs/bonsai-27b`, but its manifest's `model` is the 27B GGUF's
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
`<kindi-model-root>/kindi-lightning-clean/models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`
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

## 40. What changed in the 2026-11-17 rounds (refresh)

Nine rounds landed after §39, all committed on `main` (`899a110` at this refresh).
The first two are measurements that changed what we say; the rest are features,
fixes and record-keeping. Every row names the evidence, not the intention.

| round | commit | what it changed | evidence |
|---|---|---|---|
| Repo consolidation (branches, worktrees, uncommitted state) | `70336f6` | `main` == session branch == both worktrees; every remote ref an ancestor; the two `wip/main-interrupted-*` branches and `clean/kindi-0.5` audited file by file and kept archival (their unique content is the async-tandem WIP measured unnecessary, an AVX-512 draft, `pair_ok` analysis fixed as `pair_fits`, and 0.4.0-era planning docs) | `AGENTS.md` "Consolidation audit"; 0 commits missing from `main` |
| 9B tandem vs full GPU at comparable contexts | `8990d5f` | **The split buys headroom, never context, on a pack that fits.** Full GPU holds every request up to 131,072 tokens (5,149 MiB peak); `gdn:8` saves 486–488 MiB at 2.8–3.0× decode cost, `gdn:16` 872–874 MiB at 4.5–4.6×, flat across 8K–131K. The 27B stays the only pack here that needs the split | `docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-tandem-9b/RESULTS.md` |
| GBNF rule-cache result | `6d2a956` | Dropping the pre-`221d652` rule cache is not a regression: sharing a rule widened the language (an "exactly 3" grammar accepted 0–5 pairs) and the cache was 20× smaller *because of that* (302 vs 5,915 states). Added `Grammar.states`, an actionable over-budget refusal naming `*`/`+` and `json_schema`, and the measured ladder in `docs/API.md` | `tests/test_gbnf.py` (40-pair grammar accepts exactly 40, rejects 39/41; loop form ~330 states; 200-copy refusal) |
| First real Windows installer run | `48e875d` | `install.ps1` executed on Windows 10 19045, isolated on `S:` with its own `-AppName`/`-Distro`/`-DistroDir`: real rootfs download with Canonical SHA256SUMS match, `wsl --import` (ext4.vhdx 1.3 GB), systemd, `nvidia-smi` seeing the 4060 Ti inside WSL, Start-menu items, HKCU uninstall entry, transcript and state in `%LOCALAPPDATA%\KindiInstallerTest` (85 KB). Docker, the model download and a served request were deliberately skipped. Findings: published `install.sh` is 0.6.1 vs 0.6.0 here; in-WSL disk check reports vhdx capacity, not host free space | `docs/WINDOWS.md` "Verified installer run"; `Ubuntu`/`docker-desktop` distros, Docker settings mtime and port 8093 verified untouched |
| Evaluation ideas triage | `5c6cc6d` | Fifteen proposed measurements triaged against the tree: seven accepted with their gate written first (task-level equivalence suite, failure-mode catalog, KV/peak table generator, vision-gate honesty, quant comparison, pack licence table, queueing curve, cold-start table), six already tracked and not duplicated, one not this repo's shape. The triage itself found two real gaps | `docs/ROADMAP.md` "Evaluation ideas review (2026-11-17)" |
| Vision gate made honest + one grammar compile | `c7b55a0` | A manifest `vision`/`mmproj` flag used to be the only thing between an image request and a text-only engine, and nothing writes that field or runs a projector — image input is now always a 400 that names the missing projector when a pack claims one. Validation's compiled grammar is now reused by the decode path, so a structured request compiles once (~15.6 → ~7.8 ms on canonical `json.gbnf` at nesting 2, 7,936 states) | `tests/test_server.py` 72 passed, incl. the vision-declaring-manifest 400 and the call-counting test |
| `kindi caps` + `kindi autotune` (REQ-0022) | `9e6405f` | Solid identification and a measured placement search: see §3 rows. Live on the rig: `sm_89`, 8188/6994 MiB, driver 617.42, i7-8086K 6/12, L3 12 MiB; autotune 2B/ctx-8192 `gpu0` 97.4–97.6 tok/s decode / 111.7–113.0 read / 3,050 MiB peak, with the two tandem candidates refused/blocked and reported as such | `docs/AUTOTUNE.md`; `tests/test_caps.py` 19 + `tests/test_autotune.py` 18 |
| Windows CPU flags without a compiler | `78fb27f` | Took over the detection half of the archived 2026-11-09 CPUID draft: `kernel32.IsProcessorFeaturePresent(39/40/41)` answers AVX/AVX2/AVX-512F and is merged only where the engine probe has no answer. Measured on the i7-8086K: `AVX=yes, AVX2=yes, AVX-512=no` (was three `unknown`s); FMA3/F16C have no kernel32 constant and stay unknown with a note naming `KINDI_CC`. The draft's AVX-512 kernels stay archived — no host here can compile or verify them | `kindi/caps.py:windows_features`; `tests/test_caps.py` (kernel32, no-kernel32, merge-precedence) |
| Verification round (this report's baseline) | `893bacd`, `899a110` | `main` verified end to end: **1026 passed, 28 skipped, 0 failed** (109 s) plus the three generated-artifact gates. The run caught two real defects: a stale `docs/CONFIGURATION-REFERENCE.md` (the new `KINDI_CC` probe moved a recorded line) and a handoff table lagging its commit | run log in the round's commit message; `docs/CONFIGURATION-REFERENCE.md` regenerated |

## 41. Stock moved to v0.6.1 — and its MTP fix was our bug too (2026-11-17)

Official upstream `v0.6.1` (`7fececb`) was published 2026-11-16 and fetched here on 2026-11-17.
The published range `v0.6.0..v0.6.1` is **16 files, +336/−43**: it includes `dce89df`'s container
packaging commit (which adds `compose.yaml` and `compose.amd.yaml`) and `a6f4891`'s README
acknowledgement as well as the 0.6.1 commit. The 0.6.1 commit itself (`dce89df..v0.6.1`) is **15
files, +52/−36**: `compose.yaml`, the three Dockerfiles, `docs/CONTAINERS.md`, the Docker/install
docs, README/NOTICE/CHANGELOG, the 0.6.1 version strings, `install.sh`/`install.ps1`, a graph-build
diagnostic in `kindi/spec.py`, and one real code fix in `kindi/mtp.py`. It does **not** change
`compose.amd.yaml`; that file came in with `dce89df`. An earlier version of this section gave the
16-file total but attributed it to the 0.6.1 commit and listed `compose.amd.yaml` as its change,
both corrected here.

| 0.6.1 item | our status |
|---|---|
| `kindi/mtp.py` passes `k_rmsfold_qT`'s `hp` argument | **we had the bug; fixed in `f934a55`.** `k_rmsfold_qT` took a trailing `hp` in 0.5.0; `spec.py` and `dspine.py` were updated, `mtp.py` was not. With `KINDI_TC=1` and `KINDI_MTP_HEAD` set, the head's graph failed (`cuGraphAddKernelNode failed (1)`) and the engine logged `speculation off: the MTP head ... did not load` — mode A did not run at all on the tensor-core layout for four releases, and a short argument tuple is not a Python error. Reproduced on the rig before the fix; after it: `MTP head attached: kindi-mtp-2b-v1.npz (61 MB, window 64)` |
| `kindi/spec.py` names node, grid, block and argument count on a graph-build failure | **ported in `f934a55`** — that diagnostic is what makes a missing argument visible instead of `failed (1)` |
| container packaging (`compose.yaml`, the Dockerfiles, `docs/CONTAINERS.md`, docs) | **already here** (from `dce89df`); the 0.6.1 delta is the version/image-tag strings and docs, while the official release publishes `0.6.1-cuda`, `0.6.1-rocm6` and `0.6.1-rocm7` images. A fork release would need to publish matching images; §4 is corrected accordingly. `compose.amd.yaml` is not part of the 0.6.1 commit; it came in with `dce89df` |
| version strings now `0.6.1` upstream | ours stay `0.6.0` plus this changelog: the installer pins upstream's published artifact, so a fork version bump needs a fork release |

The fix revives a documented feature but does not make it a win here. Measured on the i7-8086K +
RTX 4060 Ti (`KINDI_TC=1`, ctx 24576, one 400-token greedy request, temperature 0): 361.2/363.7 tok/s
without speculation against 236.9/239.6 with mode A (378 of 1,680 drafted tokens accepted, 22.5%),
and 230.9/247.2/246.6 with the guard forced on from position 0 — the first guard windows are already
spec windows, so a 400-token reply pays the loss before the guard can pause. Mode A therefore costs
**0.66x** on this card; the modes that draft a smaller model for a bigger one keep their measured
win. Details and the table: `docs/MTP.md`. The bug class is now pinned by
`tests/test_kernel_arity.py` (4 tests): it parses the kernel's 20 parameters out of `tcf.cu` and
every call site's argument tuple through `ast`, and dropping `hp` again fails exactly the `mtp.py`
case.

Rounds since §40 (`899a110` → `f934a55`):

| round | commit | what it changed |
|---|---|---|
| Lease rows out of the live handoff | `7e94156` | `AGENTS.md` 371 → 265 lines; the dated table moved to `docs/development/AGENTS-HISTORY.md` |
| Claims registry | `9d1a817` | every number the three claimed docs quote is re-derived from its tracked run (`docs/claims/*.json`, `tools/check_claims.py`, 12 gates / 45 numbers, 4 reasoned debt entries) |
| Adapter stop scan + real prompt ids | `1cac59f`, `e7ee6d3` | cross-piece stop tests for the remaining adapters and prompt-seeing filter tests; both defects were already fixed in `861d3eb` and the coverage is what was missing |
| `kindi use` symlink guard | `d2b124a` | the WinError 1314 guard also covers `NotImplementedError`, and its cleanup can no longer abort the selection |
| Coverage floors + untracked secret scan | `7eb6120` | measured floors for `engine.py` (24.3% → floor 19) and every `kindi/backends/*`; `fake.py` was below its recorded floor, so the gate was red; the file scan's untracked union is now pinned |
| `kindi plan` decode-slot arithmetic | `174204a` | the `slots:` line runs the engine's own fit twice (`dcache=0`) instead of claiming one displaced expert per slot; 10 slots cost 20 experts per layer, 162 vs 182 experts and 86.4% vs 91.6% of measured picks |
| External knowledge-base skill | `992d8f4` | read-only skill + scan ledger (`docs/development/AI-INVENTORY-SCAN.md`) for the local-AI base, with a stated gate before any idea becomes roadmap work |
| Secret scans: history and synced paths | `84aecd4`, `b710b1e` | `tools/check_secret_history.py` gates every commit's diff against a counted baseline; a synced-folder path now counts as personal data, which removed the knowledge-base path from the tree |
| MTP head graph (this section) | `f934a55` | the 0.6.1 `hp` port, its diagnostic, and the arity test |

Delta against official `v0.6.1` at the report base `e66471f`: **504 commits, 522 files changed,
+74,828 / −1,190 lines, 632 tracked files (172 upstream), 28 numbered requirements**, full suite
1164 passed / 28 skipped. At `f934a55` the same comparison was 440 commits, 468 files,
+67,156 / −1,125 lines, 579 tracked files and 22 requirements. `v0.6.1` is one commit past the
shared base `dce89df` and is not an ancestor of the fork, so the commit count is
`v0.6.1..HEAD`; the zero-deletion check is `git diff --diff-filter=D --name-only v0.6.1..HEAD`.
Everything in §4 that was open before this refresh is still open, with two exceptions: the
Windows installer has now been exercised for real (one step short of a complete
install, as recorded there), and the image-input loophole is closed.

The measurement that would most change this report is still the same one: a
second card (two-card TP and Lightning cache rows in `kindi autotune`), an
AVX-512 host, and a Metal/HIP rig for the calibration clause. None is
addressable from this machine.

## 42. Official v0.6.1 release published (2026-11-17)

The GitHub release was published **2026-11-16T21:49:20Z** (`published_at` in the
release API). This section uses the release page and its assets as the baseline,
not only the tag. The notes say the release fixes the 2B with its MTP head on
tensor-core cards: 0.5.0 and 0.6.0 failed `kindi verify` and stopped the
installer's "Ternary-Bonsai-2 2B" setup on NVIDIA compute capability 8.0 or
later with 9.5 GB or more (`--model 2b`). The recommended 27B setup did not use
the MTP head and was unaffected.

| release item | value |
|---|---|
| tag / commit | `v0.6.1` / `7fececbff13adc88bf21b8f07dea4e1c91996a9d` |
| published | 2026-11-16T21:49:20Z (not a draft or prerelease) |
| container images | `ghcr.io/ahtx/kindi:0.6.1-cuda`, `0.6.1-rocm6`, `0.6.1-rocm7`; the notes say each passed `kindi verify` on a real card |
| source tarball | `kindi-0.6.1.tar.gz`, 2,013,311 bytes, sha256 `63f20abfbba55ebe8cc341b56ea15758325b59d1c84a40ed71ec3b05fadebe41` |
| installers | `install.sh` sha256 `31c4a86025c4c25303b94ba7ff7436a34b2486de0621ac44a57f4b805f0a5a1c`; `install.ps1` sha256 `00d83e0dfbc54f46b99b8612d71044c97ba621e871b10fd47fdd7f913e3a7b23` |
| documentation asset | `How-To-Get-Work-Done-with-Kindi-0.6.0.pdf`, 4,163,915 bytes, sha256 `14d670b383bff4aa980511b367f700d7da0a1e7af6680ed2c19914a65a1ac523`; the filename still says 0.6.0 |

The release does not add engine behavior beyond the two v0.6.1 code changes already
recorded in §41: the MTP `hp` argument fix and the graph-build diagnostic. The
remaining release differences are version strings, `compose.yaml`, the three
Dockerfiles, Docker/install docs and installer scripts.

Against the official release, `main` at `e66471f` is **504 commits and 522 files
ahead** (`+74,828 / −1,190`, zero file deletions), with **632 tracked files** (172
in the v0.6.1 tree), **28 numbered requirements**, and the full suite at **1164
passed / 28 skipped**. The MTP fix is ported in `f934a55`; no newer official
kernel or engine path is missing. This fork does not publish a 0.6.1 image or
installer, still reports `0.6.0`, and its installer still pins the v0.6.0 source
archive (sha256 `613f7320…`). Those are release-owner differences, not engine
ones.

The commit count and line total above are measured at `e66471f`, the last code commit
of this refresh. The preview's chips are computed at render time from the worktree
`HEAD`, so a preview rendered after the report, claims and handoff commits names a
slightly higher commit count and insertion total; those later commits touch only this
report, `docs/` and `AGENTS.md`, so the code delta is unchanged. The two figures
measure different revisions, not a disagreement.

What changed in this update: the baseline is the official release, not only the
tag; §41 now separates the 16-file `v0.6.0..v0.6.1` range from the 15-file 0.6.1 commit and attributes `compose.amd.yaml` to `dce89df`; `0.6.1-rocm6`
is added; §4's missing/divergence rows and the installer pin now name the
published tarball and installer hashes; and the local HTML preview was
regenerated from this report.

Not independently verified here: no official 0.6.1 image was pulled or run, and
the official tarball was not installed. The release notes' image-verification
statement is recorded as the publisher's claim, not as a local run. No fork
release artifact exists.

## 43. Audit follow-up, full-suite rerun and report refresh (2026-11-18)

This section records the audit/release refresh that follows the v0.6.1 release
notes in §42. The live comparison is `CURRENT-VS-KINDI-0.7.md`; this is its dated
history.

- **Latest official release checked.** The GitHub releases API still reports
  **v0.6.1** as `latest` on 2026-11-18. The annotated tag `v0.6.1` (`116532a`)
  peels to commit `7fececb`, published 2026-11-16T21:49:20Z. No newer official
  release or tag exists in `origin`.
- **Comparison moved from `0184bb8` to the code/evidence snapshot at `296a03d`.**
  Against `v0.6.1` that is **622 commits and 721 files**, **+94,507/−1,248**, with
  **zero file deletions**. The current tree has **827 tracked files** (172 in
  the v0.6.1 tree) and **28 numbered requirements**. The extra commits after
  `0184bb8` are documentation, handoff and generated-reference updates, not
  engine changes. The later report/docs commit is documentation-only and changes
  no code, test or benchmark evidence; the preview is rendered from that final
  commit.
- **Audit follow-up (`0dbff91`).** DSV4.1 body errors now close the HTTP/1.1
  connection; current-facing docs and tracked benchmark evidence lost their
  local drive/profile roots; the forward-slash machine-root guard was added;
  the optional Bridge got a dependency-free smoke suite and CI job (with a
  clear missing-directory `ValueError`); and the action-pin guard now covers
  every workflow. `25bd587` recorded the handoff.
- **Main/worktree consolidation.** `main` was fast-forwarded through
  `0dc79e0` (handoff table), `a884397` (regenerated requirements history and
  REQ-0019's moved last-modified commit) and `77a3f84` (the stale main-worktree
  "bridge note" correction). Both worktrees are clean at the same commit; the
  `clean/kindi-0.5` and `wip/main-interrupted-*` branches remain archival and
  were not merged.
- **Full CPU gate after the Usage/plan edits.** `coverage run --branch
  --source=kindi -m pytest -q` first returned **1 failed, 1,230 passed, 28
  skipped** in 246.53 s: `test_config_reference` caught the generated
  `docs/CONFIGURATION-REFERENCE.md` stale. After regeneration the rerun returned
  **1,231 passed, 28 skipped, 0 failed** in 243.91 s, with **46.2186%**
  line+branch coverage (23,467 statements, 8,386 branches). `tools/check_coverage.py`
  passed all **46** per-module floors. The failed first pass is kept in the
  current docs; it is not replaced by the older clean 1,196/45.4% figure.
- **No GPU claim.** None of this round's changes ran CUDA, a model pack or a
  cold-tier pack; the live comparison's hardware and model sections are
  unchanged.

The comparison's old `0184bb8`/576-commit/704-file figures are retained only in
earlier history above this section; §43's live report and claims named the
code/evidence snapshot `296a03d`, 622 commits, 721 files and 827 tracked files.
The later report/docs commit is documentation-only.

## 44. Comparison refresh at `3002a40` (2026-11-19)

This section re-bases the live report on the tree as it stands on 2026-11-19. It is the
history behind `CURRENT-VS-KINDI-0.7.md`; the live report keeps only the current status.

- **Latest official release checked.** The GitHub releases API still reports **v0.6.1** as
  `latest` on 2026-11-19 (annotated tag `116532a` peeled to `7fececb`, published
  2026-11-16T21:49:20Z). Upstream `origin/main` now points at `7fececb`, the release commit
  itself, so unlike the 2026-11-18 check there is no post-tag upstream commit left to port.
  The shared base is still `dce89df`; `v0.6.1` is one commit past it and is not an ancestor
  of this fork.
- **The fork moved from `296a03d` to `3002a40`.** Against `v0.6.1` that is **649 commits,
  736 files changed, +97,703/−1,283**, with **zero file deletions**; **839 tracked files**
  against 172 in the v0.6.1 tree, and **28 numbered requirements**. The 27 commits after
  `296a03d` are the rounds below; none adds a new engine path.
- **Rounds since §43 (27 commits).**
  - Python quality ratchet extended from Ruff to mypy, Pyright and Pylint, with the measured
    legacy baseline and CI wiring (`7310590`); the Pyright baseline is the 1,035-error figure
    in `tools/python_quality_baseline.json`.
  - DSV4.1 CLI plus standalone-server/dspark/tool input validation moved before any model is
    mapped (`364814a`, `f069c1d`; `tests/test_dsv41_cli.py` 24 tests).
  - DSV4.1 413/400 body handling now half-closes and drains the refused body (`7325617`).
  - Nanbeige: the gateway path is explicitly non-drafting (`6e2753a`) while the MTP head loads
    on every engine path (`8163188`); the MTP smoke run record carries its provenance
    (`a1f75fd`) and REQ-0024's history is recorded at that commit (`063a2e8`).
  - Coverage and baseline provenance: per-module floors corrected (`2cd4d21`), the coverage
    run tracked (`f6e73ae`), three median base-cell transcripts linked with redactions
    (`38dd284`), `--save-baseline` now carries each cell's `logs` list and refuses to drop one
    (`a391e1d`), and the plan follow-up commands round-trip through real cmd.exe/Git Bash
    with a spaced pack path (`e6d1762`).
  - Correctness/doc rounds: ternary-MoE census and build wording (`c0ec5fa`, `da4edb5`,
    `a00f314`), the forward-slash drive-root rule in the shared scanner (`9dcb21b`), the
    Bridge smoke job pinned to its README command (`539a504`), the live handoff split into
    sections (`b16d8f5`), the archival-branch re-check (`f803b2d`), the 27B baseline summary
    pinned to its median pass duration (`3002a40`), and the report clarification (`6dc092a`).
- **Full CPU gate at `3002a40` (working tree, including the uncommitted `tests/test_mtp_smoke.py` test):** **1,274 passed, 28 skipped, 2 failed** in 222.01 s. Both
  failures are the known secret-scan findings, not engine or test-logic failures:
  `tests/test_mtp_smoke.py`'s home-path fixture and the handoff row in
  `docs/development/AGENTS-HISTORY.md` that quotes it. The forward-slash home rule added in
  `9dcb21b` is what turned them into scan hits; the fixture predates it. Both were replaced with non-home values on 2026-11-19. The last clean run
  was `539a504` (**1,241 passed / 28 skipped / 0 failed**, 227.62 s), before that handoff row
  landed. Generated-artifact checks (`project_memory --check`, `config_reference --check`,
  `support_matrix --check`) are clean, and `python tools/check_claims.py` reports **17 gates,
  95 numbers, 4 benchmark citations as debt**.
- **No GPU run.** None of the 27 commits ran CUDA, a model pack or a cold-tier pack; the
  live report's hardware and model sections are unchanged, and no new performance claim is
  made here.
- **Working tree.** Uncommitted: `docs/BENCHMARKS.md`, `docs/INTEL-ARC.md`,
  `docs/ROADMAP.md`, `docs/TANDEM-RUNTIME.md`, `docs/benchmarks/CROSS-MODE.md`,
  `docs/benchmarks/build_cross_mode.py`, `docs/claims/README.md`,
  `docs/claims/REGISTRY.json`, `docs/development/AI-INVENTORY-SCAN.md`,
  `tests/test_mtp_smoke.py`, plus untracked `tests/test_cross_mode_gain.py`. They carry the
  cross-mode tandem measurements, the tracked Intel-fork review and the 2026-11-19 roadmap
  follow-ups. No engine code is uncommitted, and no git figure in this section counts them.
- **Preview.** The local HTML render is regenerated from the live report. Its chips are
  computed from the committed HEAD and `git diff --shortstat v0.6.1..HEAD` at render time,
  so they name `3002a40` and the 736-file shortstat rather than this uncommitted section.

What changed in the live report: the date is 2026-11-19 and every figure is re-based at
`3002a40`; §1 records that upstream `main` now equals the release commit; §3 gains the
quality ratchet, pre-load CLI validation, DSV4.1 body cap and cold-expert slot rows; §4's
intro and zero-deletion check name the new HEAD; and §6 records the two known scan failures
instead of the earlier clean run.

## 45. Current worktree refresh: the 2026-11-19 round (2026-11-19)

This section records the dirty worktree behind the live report. It is not a commit: `3002a40`
remains HEAD, and the report keeps the committed 649-commit / 736-file figures separate from
the worktree.

- **Latest official release rechecked.** The GitHub release API and `git ls-remote --tags
  origin` still report **v0.6.1** (`7fececb`) as latest on 2026-11-19; there is no v0.6.2, and
  `origin/main` still points at the release commit. The baseline in `CURRENT-VS-KINDI-0.7.md`
  is unchanged.
- **Committed baseline.** `v0.6.1..3002a40` is still **649 commits, 736 files, +97,703/−1,283**,
  with zero file deletions, **839 tracked files** and 28 numbered requirements.
- **Worktree.** The tracked diff against `v0.6.1` now touches **766 files** (no deletions) and
  the index has **869 files**, including the staged benchmark pages; one untracked
  `tests/test_cross_mode_gain.py` and the new controlled-pair claim are outside the git
  figures. The worktree adds engine code (`kindi/moe_expert_lane.py`, `kindi/spec.py`'s MTP
  default, plan/DSV4.1 edits), tests, benchmark evidence and claims. No commit follows
  `3002a40`.
- **Mode A.** Explicit `KINDI_SPEC=mtp` now defaults to `KINDI_SPEC_K=1`; the live 2B check
  with no K passed 4/4 prompts, 160/160 tokens, 65/94 drafts accepted at 403.3 tok/s (ctx
  1024, RTX 4060 Ti). The 0.66x row remains only for explicit k=4.
- **Q2_0 CPU expert lane.** The 2026-11-19 gate is bit-identical to `lt_mv_q2` and
  `lt_mv_q2_down` (0 differing elements with device-resident or host-mapped weights), but the CPU
  expert costs 2.395 ms, against 0.783 ms for the copy lane (3.06x) and 0.662 ms for the engine's
  direct host-mapped GPU read (3.62x faster than the CPU lane; 1.18x faster than the copy lane),
  and the lane is not wired into `lightning.Engine`; the model row is blocked by the absent
  Lightning pack. This is a passed exactness gate, not a backend.
- **Controlled 9B CPU/tandem pair.** The same-session `cpu-first` six-layer run at ctx 24,576
  gives 2.9 tok/s CPU-only vs 16.8 tandem at 24K (5.8x over CPU-only), and 3.0x at 100; both
  placements verify 4/4, 133/133. This replaces the old cross-run shallow-context 4.3x and
  keeps the split below the full-GPU 9B row. Evidence: the new
  `docs/benchmarks/2026-11-19-i7-8086k-rtx4060ti-9b-controlled/` page and its claim.
- **Chunk sweep.** The 2B sweep at 4K/8K keeps `KINDI_CHUNK=256` as the measured optimum and
  default; 128 is a real loss (22.6% at 4K, 15.9% at 8K prefill), and `KINDI_CHUNK=256` still
  verifies 4/4. No default changed and no autotune candidate was added.
- **Tests and gates.** The current worktree's full CPU suite is **1,312 passed, 28 skipped, 0
  failed** in 217.78 s; the Python quality ratchet is 332/59/1,035/255 within baseline; claims
  are **20 gates / 109 numbers**. The earlier 1,274-pass / 2-failure note is superseded.
- **Preview.** `bench-logs/render_comparison.py` now shows the official v0.6.1 baseline, the
  committed `3002a40` shortstat and the current worktree shortstat as separate chips. The
  rendered page remains a local preview, not a tracked artifact.
- **Committed.** The round above was committed as `b893157` and `main` was fast-forwarded to
  it on 2026-11-19. The live report now names `b893157` as its baseline: **650 commits, 768
  files, +101,646/−1,289**, zero file deletions and **871 tracked files** against `v0.6.1`.
  The worktree counts above are kept as the pre-commit history of this section.

## 46. Consolidated tree and live comparison at `53198e5` (2026-11-19)

This is the first section in which the live comparison names a tree with no uncommitted
engine or evidence changes behind it. `CURRENT-VS-KINDI-0.7.md` and
`docs/claims/upstream-delta.json` now name **`53198e5`**; `main` and
`freebuff/tandem-cpu-recovery` are the same commit and both worktrees are clean.

- **Latest official release rechecked.** The GitHub release API and `git ls-remote --tags
  origin` still report **v0.6.1** (`7fececb`, published 2026-11-16T21:49:20Z) as latest on
  2026-11-19; there is no v0.6.2, and `origin/main` still points at the release commit. The
  shared base remains `dce89df`, and `v0.6.1` is one commit past it, so `v0.6.1..53198e5` is
  the comparison range.
- **Current delta.** Against `v0.6.1` the fork is **664 commits, 827 files changed,
  +106,882/−1,320**, with **zero file deletions**; the v0.6.1 tree has **172** files, the
  current tree **929** tracked files, and there are **28** numbered requirements.
- **The 14 commits after `b893157`.** They are the fresh MTP k=1 confirmation, the
  host-mapped Q2 lane, the mixed card/CPU combine contract and its gate, the ternary-MoE
  census correction, the 2026-11-18 audit follow-up, the comparison refresh at `f984962`, and
  the final consolidation: `f75253b` (DSV4.1 request-shape validation and batched Windows
  secret scanning), `84b6686` (controlled 2B/4B/9B pairs, 9B chunk sweep, context-growth and
  ternary-MoE pages, Q2 lane moved to `tools/`) and `53198e5` (handoff).
- **Full CPU gate.** On the consolidated content (`f75253b` + `84b6686`): **1,357 passed,
  28 skipped, 0 failed** in 229.53 s (one CuPy CUDA-path warning). The final handoff commit
  only changes `AGENTS.md` and its history; the handoff/docs tests are **7 passed**.
- **Current checks.** Python quality **316 Ruff / 59 mypy / 1,035 Pyright / 255 Pylint**
  within the reviewed baseline; `check_claims.py` **21 gates / 124 numbers**;
  `config_reference --check`, `support_matrix --check`, `project_memory --check`,
  `check_secret_baseline.py` (104 findings in 31 tracked files) and `git diff --check` clean.
- **No GPU or model run.** This round changed tests, documentation, evidence and the request
  validation/secret-scanning paths only; no CUDA, model pack or cold-tier run is claimed. No
  push or deployment was made.
- **Preview.** `bench-logs/render_comparison.py` renders the live report to the ignored
  `current-vs-kindi-0.6-preview.html`; its chips compute HEAD and the committed
  `v0.6.1..HEAD` shortstat at render time, so they name `53198e5` and the 827-file delta.

## 47. Round-3 audit disposition (2026-11-19)

The round-3 review was written against `f984962`; the fixes landed in `3fed704` (based on
`8c3b083`, 14 commits after the review), so every finding was re-checked by function name as
well as line number.

- **N1/N5 (cold reader teardown) — closed.** `lightning.Engine` created a `ColdExpertReader`
  and a daemon lookahead thread but never closed either. `Engine.close()` now joins the thread and
  closes the reader once; the registry adapter, `lightning_cli` verify/bench and the HTTP server's
  `Kindi.close()`/`serve()` teardown call it. `Engine.close()` suppresses a prefetch error that a
  consumer already surfaced and is idempotent.
- **N2 (quality baseline lowering) — closed.** `--update` requires `--reason`, refuses any new or
  increased file/rule count, and records the reason, measured commit and per-tool delta in
  `tools/python_quality_baseline.json`. A baseline edit can no longer be used to launder a new
  finding.
- **R1 (implicit join before queue access) — closed.** `ColdExpertReader.prefetch`, `read_many`,
  `stats` and `close` now share an internal `RLock`, so a future consumer that forgets
  `_join_prefetch()` cannot race the daemon or move a shared Windows file pointer mid-read.
- **N3 (forward-slash drive root in the history gate) — accepted by design.** The rule remains
  current-tree-only because the reviewed history snapshot predates it; a path added and deleted
  in one commit is the remaining gap, and the current-tree test is the primary guard.
- **N4 (bridge is a separate package) — not a finding.** The bridge smoke job and its own suite
  remain separate by design; `tests/test_ci_config.py` pins the README command to the job.
- **R2 (quality debt) — accepted, count corrected.** The current baseline is **321 Ruff / 59 mypy
  / 1,035 Pyright / 255 Pylint**, not the audit's 332 Ruff; the debt is still tolerated by the
  ratchet, and `docs/CI.md` now lists the actual 163 bugbear findings and the burn-down rule.
- **R3/M4 (history exposure) — decision recorded.** `docs/SECURITY.md` records the 160-row /
  ~3,318-hit history baseline and the decision not to rewrite again; a public push must disclose
  it, and another rewrite needs a new signed review and owner authorization.
- **R4/M5 (worktree behind main) — resolved.** `main` and `freebuff/tandem-cpu-recovery` were
  the same commit (`8c3b083`) before this round; the audit's `f984962` observation was stale.
  The fixes are now in `3fed704`.
- **M1/M2 (cold-tier hardware gate, GPU CI) — open and unmeasured.** No larger-than-RAM pack,
  `kindi verify` 4/4, tok/s or flash MiB/token is claimed, and the CUDA-graph/kernel-arity gates
  remain local-only facts.
- **M3 (quality clean-up) — incremental.** This round fixes the ratchet's review hole and
  documents the true debt; it does not claim a full lint/type burn-down.

No full suite, GPU, model pack or cold-tier run is claimed by this round. The focused checks are
recorded in `docs/development/AGENTS-HISTORY.md`.
