# Audit triage: MiniMax-M3.1-Flash round 2 (2026-11-18)

Source: `MiniMax-M3.1-Flash_ReviewAudit_202611172128.txt`, kept outside the repository by the
maintainer. This page records what was checked against the current tree, what a later commit had
already closed, and what still needs work. Re-check findings by symbol, not by the report's line
numbers: the tree moved while that review was written.

Status: **closed** = code or docs plus a test/gate on this tree; **open** = still a real
limitation, with what would close it; **owner** = needs an action this environment cannot take.

| id | finding | status | evidence / next step |
|---|---|---|---|
| C1 | four documented gates not in CI | **closed** | `.github/workflows/ci.yml` runs `check_claims.py`, `project_memory --check`, `config_reference --check` and `support_matrix --check`. All four pass on this tree. The gate also caught a stale `docs/CONFIGURATION-REFERENCE.md` left by the `--range` line shift at `a299651`; regenerated 2026-11-18. |
| C2 | two backend dispatch points | **closed** | `kindi/server.py::served_backend` is the one family table; `tests/test_server.py::test_the_server_names_only_families_it_can_serve` and `tests/backends/test_registry.py` pin it. The three special-family serve paths and the registry gateway build the same base engine with the same arguments (`make_engine` forwards `kv_cache` to the Nanbeige adapter). The gateway is deliberately non-drafting: `lookup` and `mtp_head` are the serve wrapper's draft sources, the adapter advertises `supports_speculation=False` and its `generate()` reports `drafted=0`/`accepted=0`, so neither the adapter nor `nanbeige_cli._engine` (the actual verify/bench builder, which does not go through `make_engine`) loads the head. `tests/test_served_engines.py` records the constructor from both paths for `kindi_nanbeige`, `kindi_lightning` and `kindi_gemma4` and pins that only serve attaches the head; `tests/backends/test_registry.py` pins the adapter's zero-draft done event and capability. |
| C3 | `kindi serve` accepts a dsv41 pack | **closed** | the same table refuses `kindi_dsv41` and names `python -m kindi.dsv41_server`; `tests/test_server.py::test_the_server_refuses_dsv41_with_the_standalone_server_command`. |
| C4 | coverage floors skip the new modules | **closed** | `tests/coverage_floors.json` has 46 measured floors, including the 2026-11 additions; `engine.py` moved 19% -> 20%. |
| C5 | installer one-liner and checksum disagree | **docs closed, asset owner** | `docs/INSTALL.md` now leads with download + checksum + a reviewed Windows download; the stale `v0.5.0-installer` asset remains an owner action in `docs/MANUAL-HELP-NEEDED.md` §5. |
| C6 | FA2 in the silicon-facts table | **closed** | `kindi/caps.py::card_features()` reports silicon only (FA2 dropped); `tests/test_caps.py::test_card_features_are_silicon_facts_not_library_algorithms`. |
| C7 | autotune cache sized by different arithmetic | **closed** | `kindi/autotune.py::_expert_cache_mib` uses `ltfit.rate_slots(room=..., n_moe=...)` and the host cap; the `expert-cache-<MiB>` rows in `tests/test_autotune.py` pin the sizing. |
| C8 | usage-count file shape assumed | **closed** | `kindi/moe.py::Usage` validates count/order shape and `split_host_tiers` reports when it falls back to rank order; `tests/test_moe_policy.py` and `tests/test_cold_experts.py`. |
| C9 | cold decode serializes on disk I/O and a stream sync | **partly closed, hardware gate open** | decode now keeps a pinned slot cache (`cold_slot_plan`), so a repeated cold expert takes no read; `_cold_prepare` still does one full `stream.synchronize()` and one blocking read for the current layer's misses. A gate run must measure reads/token and the compute/I/O split. |
| R1 | cold pointer table can carry stale entries | **closed in code, GPU gate open** | a routed expert's slot is never evicted (`cold_experts.cold_slot_plan` + tests), and the prompt path uses separate `tcp`/`tsp` staging tables, so decode and prompt pointers cannot overwrite each other. Still needs the cold/warm/forced-eviction token check on a card. |
| R2 | two levels of proof in one tree | **open, documented** | `docs/ROADMAP.md`, `CURRENT-VS-KINDI-0.6.md` and the open gates in `AGENTS.md` say which paths have measured evidence; the claims gate now runs in CI. |
| R3 | AGENTS.md "Current state" is one very long bullet | **open** | generated/sectioned handoff is a follow-up; `tests/test_agents_handoff.py` only checks the mechanical HEAD/table parts. |
| R4 | a second HTTP stack (`dsv41_server`) | **partly closed** | body size cap, 413/400 shapes and parse-error handling now match `kindi/server.py` (2026-11-18); its body-error close is now orderly, and a real-socket test pins the 413 before a pipelined valid request. Unifying the two servers remains a design decision. |
| M1-M3 | CI gates, one dispatch point, coverage floors | **closed** | see C1, C2/C3, C4. |
| M4 | cold-tier hardware gate | **owner/hardware** | larger-than-RAM Lightning pack, `kindi verify` 4/4, decode tok/s against the mmap-fault path and flash MiB/token; R1/C9 checks belong in the same run. |
| M5 | history-exposure decision | **owner** | rewrite the 128 baseline rows or publish with them and say so; the old -> new hash mapping of the 33 re-hashed commits is not recorded in the tree. |
| M6 | lint, type check, dependency scan | **lint/type closed, dependency scan open** | `tools/check_python_quality.py` ratchets Ruff 0.16.0 (`F`, `E9`, `B`, `I`) over the Python tree and mypy 2.4.0 over first-party production code. Baseline: **334 Ruff findings / 59 mypy errors**, recorded per file and rule/code in `tools/python_quality_baseline.json`; CI runs the gate and `tests/test_python_quality.py` pins the comparison. Existing debt is intentionally tolerated, not fixed. Dependency scanning is still open. |
| M7 | GPU CI runner | **owner** | steps are in `docs/CI.md` §5.1. |
| M8 | DSV4.1 server parity, lint/type/deps | **body cap and lint/type closed, dependency scan open** | `KINDI_MAX_BODY_MB` (16 MiB default), 413 `request_too_large`, 400 on malformed JSON and the socket-level close are pinned by `tests/test_dsv41_server.py`; lint/type use the M6 ratchet. Dependency scanning/pins remain open. |

## Changed in this triage (2026-11-18)

- `kindi/cold_experts.py::cold_slot_plan` + `kindi/lightning.py::_cold_prepare`: the hit test no
  longer uses `downer` (which `lt_cache_dec` fills only for host-staged experts). A per-layer
  owner/age table now survives decode steps, evicts only a slot whose owner is not routed in the
  step, and reports `slot_hits` in `cold_stats()`.
- `kindi/dsv41_server.py`: `read_json_body` with a hard cap, `KINDI_MAX_BODY_MB`, HTTP 413 for an
  over-cap body, and 400 for a malformed length or JSON (the warm endpoint previously returned 500).
- `docs/CONFIGURATION-REFERENCE.md`: regenerated after the line shift; the `--check` gate passes.
- Lesson L-0010 records the cache-ownership mistake; `docs/MOE-TANDEM.md` and
  `docs/EXTERNAL-ENGINES.md` describe the slot cache and the still-open C9 bound.

- `kindi/dsv41_server.py` now half-closes the response direction and drains a bounded prefix after
  an over-cap or unparseable body before the final close; the refused bytes must not be parsed as
  the next request, and unread bytes must not turn the response into a TCP reset. A real-socket test
  sends an oversized body followed by a valid request and pins one 413, `Connection: close` and no
  second response.
- The optional Bridge has a dependency-free smoke suite and CI job; its missing-directory path now
  raises `ValueError` rather than leaking `FileNotFoundError` from `Path.resolve(strict=True)`.
- The forward-slash drive-root gap is closed for current-facing files (tracked docs/evidence plus
  benchmark logs), with a current-tree guard; the historical baseline is unchanged.
- The action-pin guard covers every workflow, and AGENTS.md labels 1204/28 as the last recorded full
  gate rather than a run at the current HEAD.

No GPU, larger-than-RAM pack or tok/s measurement is claimed by any of these changes.
