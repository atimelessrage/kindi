# Audit triage: MiniMax-M3.1-Flash round 4 (2026-11-25)

Source: `MiniMax-M3.1-Flash_ReviewAudit_202611251554.txt` in the main checkout's root (untracked,
kept by the maintainer). This page records what was re-checked against the current tree, what a later commit had
already closed, and what still needs work. Re-check findings by symbol, not by the report's line
numbers: the tree moved while that review was written (it read `a350741`; the triage was done at
`22898e4`, two commits later).

Status: **closed** = code or docs plus a test/gate on this tree; **open** = still a real limitation,
with what would close it; **owner** = needs an action this environment cannot take.

| id | finding | status | evidence / next step |
|---|---|---|---|
| N1 | the `-mtp` variant is offered but Kindi does not draft with its head | **closed (docs)** | it loads and converts as the 64-layer base model, which is a real capability, so the catalog entry stays; `docs/BONSAI-2-SUPPORT.md` now says plainly to download the base file unless the head is being kept for future drafting, because both files build the same pack and read the same weight stream (claims gate `arc-a770-pack-20261125`). Drafting itself is `REQ-0025`'s dspine row (waits on weights). |
| N2 | six `kindi intel-*` commands may imply more maturity than "probe only" | **closed (help text)** | `intel-info`, `intel-smoke` and `intel-cl` now carry the same "(no backend)" / "(no model, no backend)" phrase the other three had, so `kindi --help` says it on every row. Grouping them under one `kindi intel` verb was declined: it renames six working commands for no capability change. |
| N3 | the llama.cpp proxy's limits are not on the user-facing surface | **closed** | `kindi backends` prints a `limits=` column (`no-grammar,no-cache,no-spec,no-thinking,no-tools`) derived from each registration's capabilities, so the wildcard proxy names its gaps before a request hits them; `tests/test_backends_cli.py` pins the column. |
| N4 | the forward-slash drive-root rule skipped paths with spaces; three benchmark scripts hardcoded an LLVM install path | **closed** | `tools/secret_patterns.py::FORWARD_DRIVE_ROOT` now admits spaces and parentheses and `forward_roots` allow-lists Windows system roots (`SYSTEM_ROOT`), the same reasoning as the backslash rule; `docs/benchmarks/tandem-cache/{measure_prefix,run_check,run_server_check}.py` take `--clang`/`--llvm` or `KINDI_CC`/`KINDI_LLVM` and refuse to run without them, so no machine path is in the tree. `tests/test_no_secrets.py` pins both the spaced match and the system-root exemption. |
| N5 | `ColdExpertReader.close()` is never called | **stale** | `Engine.close()` (`kindi/lightning.py`) joins the prefetch thread and closes the reader, is idempotent, and is called by the Lightning adapter's `close()`/reload and by `kindi/server.py`'s teardown; `tests/test_cold_experts.py` asserts `reader.closed == 1` and `cold_reader is None` after a double close. |
| N6 | the secret-history baseline still carries personal data | **owner** | 164 rows / 3,328 recorded legacy hits, unchanged by this round; the count gate stops growth. Removing it means rewriting history or publishing with it and saying so (`docs/MANUAL-HELP-NEEDED.md`). |
| R1 | the cold-tier prefetch queue relies on every consumer calling `_join_prefetch()` | **open, documented** | the invariant holds on every path today (`_cold_prefetch` joins before queueing, `_cold_prepare` joins before its demand read, `close()` joins before the reader goes). Gate: make the queue reachable only through a helper that joins, so a new consumer cannot read it unjoined; a test would enumerate the consumers. |
| R2 | the quality ratchet tolerates existing debt | **partly closed** | Ruff findings 207 -> 189 this round (unused imports/names, import order in `tools/`, one dead local), baseline lowered with a reason so the gain is locked. Remaining: `B904` (54, needs `raise ... from`), `B902`-style false positives kept with a one-line reason (`# noqa: B007` where the loop variable is read after the loop or through a closure), `B905` in benchmark tools, plus the mypy/Pyright/Pylint debt that needs a GPU host to verify. |
| R3 | the history still contains personal data | **owner** | same decision as N6. |
| R4 | the extra PR worktrees confuse which code is canonical | **closed (docs)** | this session's worktree holds 26 `pr-*`/`wip/*` linked worktrees under `.worktrees/` (`git worktree list`: 28 entries); the protocol paragraph in `AGENTS.md` now says they are archival PR branches and never a merge source. |
| M1 | Bonsai 2.5 GPU cells still "not run" | **hardware gate** | 2.5's CPU run is 4/4 token-exact; GPU exactness needs a card this host does not have for that model. Unchanged by this round. |
| M2 | Intel Arc A770 Linux target | **hardware gate** | the probes run on Windows only; M2 needs the Linux host (`docs/MANUAL-HELP-NEEDED.md`). |
| M3 | the llama.cpp proxy has no live gate | **owner** | needs a pinned `llama-server`; the 12 CPU tests stay the evidence until then. |
| M4 | Python quality debt | **partly closed** | see R2. |
| M6 | forward-slash rule widened | **closed** | see N4. |

## Changed in this triage (2026-11-25)

- `tools/secret_patterns.py`: `FORWARD_DRIVE_ROOT` admits spaces and parentheses; `forward_roots`
  exempts Windows system roots. `tests/test_no_secrets.py` gains the spaced-root case and the
  system-root exemption, so the rule cannot go vacuous again.
- `docs/benchmarks/tandem-cache/*.py` (three scripts): a `toolchain()` resolver replaces the
  hardcoded clang/LLVM defaults; `--clang`/`--llvm` or `KINDI_CC`/`KINDI_LLVM` are required and the
  error names both. The recorded results are unaffected - those runs passed their own paths.
- `kindi/__main__.py`: `kindi backends` gains the `limits=` column; three Intel help strings name
  "no backend".
- `docs/BONSAI-2-SUPPORT.md`: says which 2.5 file to download and why.
- Lint: 18 Ruff findings fixed in `tools/` and `kindi/` (unused imports and names, import order, one
  dead local), three `# noqa: B007` with the reason (the loop variable is read after the loop or
  through a closure) instead of a rename that would have changed behaviour. Baseline lowered to
  189 Ruff / 53 mypy / 829 Pyright / 245 Pylint.
- Two renames were caught by tests before they landed (`kindi/caps.py`, `kindi/cascade.py`), which
  is the reason the ratchet's rule is "fix the finding, or say why it is not one".
