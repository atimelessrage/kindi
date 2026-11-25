# Audit triage: round 5 (MiniMax-M3.1-Flash + GLM-5.3-Flash, 2026-11-25)

Sources, both untracked review files kept by the maintainer: `MiniMax-M3.1-Flash_ReviewAudit_202611251826.txt`
(the main checkout's root) and `GLM-5.3-Flash_ReviewAudit_202611251755.txt` (this worktree's root). They were
written independently within an hour of each other and overlap on two subjects (the "2.5 7B" premise and the
MTP head), so they are triaged together here. Re-check findings by symbol, not by the reports' line numbers:
both read `bf6ede3`, the triage ran on that commit plus the uncommitted CPU-MTP work.

Status: **closed** = code or docs plus a test/gate on this tree; **open** = still a real limitation, with what
would close it; **owner** = needs an action this environment cannot take; **declined** = looked at and not
taken, with the reason; **stale / not reproducible** = checked and not true on this tree or this host.

## MiniMax round 5

| id | finding | status | evidence / next step |
|---|---|---|---|
| F1 | "Bonsai 2.5 7B" is not a size that exists | **closed (docs)** | the family is 2B/4B/9B/27B and the 2026-11-24 release is a 27B-only upgrade; `docs/BONSAI-2-SUPPORT.md` now says that once, above the scope paragraph, so the next review does not re-derive it. Both round-5 reports spent time on it. |
| F2 | `kindi intel-pack` traceback instead of a fail report when `bcodes.bin`/`dpm.bin` are missing | **closed** | real: two `os.path.getsize` calls in the `report["pack"].update(...)` block and the two `np.fromfile` reads ran outside every guard. The `getsize` calls were also redundant (both sizes are overwritten from the arrays below), so they are gone; the reads sit in the same `except (OSError, ValueError)` -> `_fail` pattern as `pack_tensors`. `tests/test_intel_pack.py::test_metadata_without_the_byte_files_is_a_fail_report`. |
| F3 | `median_ms` holds the arithmetic mean | **closed** | real: the loop enqueues `repeats` launches and divides the total elapsed time by `repeats`. A true per-launch median would change what the projection consumes (per-launch closed queues instead of amortized ones), so the honest minimal fix is the name: the key is `mean_ms` in the code, the summary line and a new run's JSON. The tracked 2026-11-25 artifacts keep `median_ms` (they are recorded evidence, not rewritten by hand); the benchmark page states which key is which and that the claims were not re-measured. |
| F4 | the `-mtp`-as-base conversion is sound | **confirmed ok** | no change; the head block stays recorded in the manifest instead of converted. |
| F5 | CPU MTP drafting was mid-flight and uncommitted | **open (WIP)** | the head, `kc_q4gemv` and the scoring harness are working code with 6 tests, but the measured acceptance is 7/108 (6%), far too low for the head to be wired as a drafting win, so nothing here claims it. Roadmap CP1/CP2 carry the batched verify that would make it a speedup. |
| F6 | `git status` says "behind 3" origin/main | **cosmetic, documented** | the three are upstream's v0.7.2 release plus two cherry-picks, ported by content at `d58dbf0`; `CURRENT-VS-KINDI-0.7.md` states "3 commits behind by history and 0 by content". No action that does not rewrite upstream history. |
| R1 | history still holds legacy paths | **owner** | unchanged: the count gate in `tools/secret_history_baseline.json` stops growth; removal needs a rewrite or a published disclosure (`docs/MANUAL-HELP-NEEDED.md`). |
| R2 | Python quality debt behind the ratchet | **open** | unchanged this round (the ratchet passes at 189/53/829/245); burning it down is a separate task with a GPU host for the mypy/Pyright debt. |
| R3 | the MTP files are uncommitted, so any finding about them is provisional | **accepted** | this triage does not treat them as reviewable code; see F5. |

## GLM-5.3-Flash

| id | finding | status | evidence / next step |
|---|---|---|---|
| 1 | `bonsai-2.5-support` is superseded (806 behind, 4 unique commits all ported) | **owner** | confirmed by its content; deleting a branch and closing the four already-merged PRs (#48, #49, #72, #76) is a remote action. `CURRENT-VS-KINDI-0.7.md` records that nothing on it is unique. |
| 2 | "2.5 7B" support is missing | **closed (docs)** | same as F1: there is no such release, so there is nothing to support; the header-probe method stays the way to settle any future size. |
| 3 | `low` rendering the `xhigh` instruction is silent | **partly closed** | the mapping is a decision, not a bug (DEC-0013), and the suggested replacement ("do not overthink") is out-of-distribution text the parity harness would reject. What changed: `high` is now accepted as the same alias (the real asymmetry was that OpenAI's legal `high` was a 400 while the vendor's unsupported `low` was accepted), `kindi serve` logs one startup line naming the mapping, and `docs/API.md` / `docs/CONFIGURATION*.md` / `docs/BONSAI-2-SUPPORT.md` say it. **Open:** the response still does not echo the requested-vs-effective effort; that needs a response-metadata field (and a compatibility decision) rather than a log line. |
| 4 | Arc: probing is mature, the engine path is the gap | **hardware/owner** | M2 needs a Linux A770 host (`docs/MANUAL-HELP-NEEDED.md` row 4a); the 20.0 tok/s figure stays a decode-memory-floor projection from one GEMV. No code change can move it. |
| 5.1 | unknown `reasoning_effort` may be a 500 | **closed (partly stale)** | it was already a 400: `check_prompt()` raises the renderer's `ValueError` and the chat route converts it before the first byte. What was real: the 400 named no field and no vocabulary was documented. `_validate_request(..., variant=)` now checks the 2.5 vocabulary and answers `param: reasoning_effort` on the chat, Ollama and Anthropic routes; a qwen35 pack has no such vocabulary and is not checked. `tests/test_server.py::test_unknown_reasoning_effort_is_a_400_naming_the_field`. |
| 5.2 | the 46.36% coverage figure is old (measured at `539a504`, suite now 1,658) | **open** | the handoff already says "the last coverage-measured run remains `539a504`"; re-measuring is a full `coverage run --branch` pass, which this task did not spend. Next step: one measured run, then either raise the floors or record that new modules sit below them. |
| 5.3 | branch and PR accumulation | **owner** | same as 1: closing PRs is a remote action. |
| 5.4 | README install one-liners 404 (the `v0.7.1-fork` release is unpublished) | **owner** | unchanged and already in the handoff; `docs/RELEASE-CHECKLIST.md` is the action. |
| 5.5 | `docs/BONSAI-2-SUPPORT.md` says "research and planning" while sections 1a/4.2 describe shipped behavior | **closed (docs)** | the status paragraph now separates the shipped 2.5 sections from the projection, and the reasoning-effort bullet carries the `high` alias and the 400 rule. |
| 5.6 | `.pytest_cache/` permission warnings in `git status` | **not reproducible** | both worktrees' `.pytest_cache` directories are owned by the user here and `.gitignore` never listed one as tracked, so the warning came from the auditor's sandbox user, not from the tree. |
| 5.7 | the matrix says "not converted" for packs that were converted | **closed (docs + generated file)** | the column is `PQ2_0/Q2_0`, and no q2 pack exists for 2.5 — but the cell read as "no pack at all". The 2.5 rows now say `PTQ1_0 only`, which is true in that column; `docs/SUPPORT-MATRIX.md` is regenerated and `tests/test_support_matrix.py` still passes. |
| 5.8 | the `low` entry in `REASONING_EFFORTS` is unreachable for rendering | **closed** | confirmed: `_render27` remaps `low` before the lookup. The entry stays (it is the template's own text and the parity fixture compares against it) with a comment saying it is never rendered, and the accepted vocabulary moved to `chat.ACCEPTED_EFFORTS` + `chat.check_effort()` so validation no longer depends on the render map. |
| W1 | wire the recorded 65th block into the gated MTP path | **open (the paused task)** | this is exactly the uncommitted CPU-MTP work (F5): the head drafts and scores, at 6% acceptance, and CPU decode cannot speed up until verify is batched (roadmap CP1/CP2). Not committed as a win. |

## Declined

- **Replacing `low`'s render with a custom "do not overthink" instruction** (GLM §3): out-of-distribution
  prompt text the model was never trained on, measured nowhere, and it would break the byte-parity contract
  with the published template. The vendor's own `low` text is already known not to work.
- **Grouping the six `kindi intel-*` verbs** (carried from round 4): renames working commands for no
  capability change.
- **Rewriting the tracked Arc artifacts to rename `median_ms`** (F3): evidence files are append-only here;
  the code and the page describe the old key instead.
