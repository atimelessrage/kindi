---
name: ai-inventory
description: Mine the external local-AI knowledge base as read-only inspiration, and turn only verified ideas into Kindi roadmap steps.
license: AGPL-3.0-only
metadata:
  category: development
---

# AI inventory: the external knowledge base

The base named by `$KINDI_AI_INVENTORY` (a synced folder outside this repository; ask the user for
the path when the variable is unset, and never commit it) is a second, **external** base about local
AI on the same two machines Kindi targets (Machine 1 is the i7-8086K + RTX 4060 Ti this project is
developed on). A cron agent ("Hermes") rewrites it weekly; it is git-versioned and every number in it
carries a provenance tag. It is useful as **inspiration and a source of hypotheses**, never as
evidence for a claim about Kindi.

The local record of what has been read and what was decided is
[`docs/development/AI-INVENTORY-SCAN.md`](../../../docs/development/AI-INVENTORY-SCAN.md). Read it
before scanning; update it after every scan.

## Hard boundary: read only

- Read text with `read_files`/`grep`/`git log`. **Never write, commit, `git pull`, `ollama pull` or
  run its scripts** (`scripts/weekly_check.py`, `benchmark_models.py`, ...): they write snapshots,
  fetch provider catalogs and pull multi-GB models onto the user's machine.
- Never copy its files into this repository. Cite them: file path + line/topic + the tag the source
  gives the number.
- Its working tree is often dirty (Hermes' mid-week edits), so a git-only view is stale. Compare
  both the recorded commit and the file mtimes.

## When to run the check

- At the start of a work session whose task is roadmap, performance or "what should we do next"
  planning, and whenever the user asks for it.
- Otherwise it is cheap but not free: skip it when nothing roadmap-shaped is being planned, and skip
  it entirely if the ledger's `Last checked` is under 7 days old and the user did not ask.
- The knowledge base moves weekly. A scan that finds nothing new is a normal, one-line result.

## The check (incremental, cheapest first)

1. **What changed since last time**
   ```sh
   git -C "$KINDI_AI_INVENTORY" rev-parse HEAD
   git -C "$KINDI_AI_INVENTORY" log --oneline <last_commit>..HEAD
   git -C "$KINDI_AI_INVENTORY" status --short
   ```
   Then files newer than the ledger's date (`ls -lt`, or `find -newer <marker>`), which catch
   uncommitted work.
2. **Read only what changed**, plus the map entry for the topic you are actually working on. Do not
   re-read the whole base: it is ~4,000 files.
3. **Check "do we already do this?"** before proposing anything. Kindi is far past the point where
   most engine ideas are novel; `grep` the package and docs first.
4. **Record the result** in the ledger: last seen commit, date, files reviewed, and one row per
   candidate with a verdict.

## Where to look (the map)

| question | file in the knowledge base |
|---|---|
| how Kindi is seen from outside; upstream releases, cards, floors | `docs/KINDI.md`, `docs/KINDI_CHANGELOG.md` |
| engine knobs, KV cache types, Flash Attention, offload | `docs/LOCAL_ENGINE_TUNING.md` |
| prompt-read (prefill) cost vs decode; batching | `docs/PREFILL_TUNING.md`, `docs/COMMIT_REVIEW_MODEL.md` |
| llama.cpp / Ollama / router releases, MoE offload tricks | `docs/LOCAL_RUNTIME_DEVELOPMENTS.md`, `docs/*_CHANGELOG.md` |
| speculative decoding (drafter on GPU, target on CPU) | `SPECULATIVE_DECODING_GT1030.md` |
| the actual hardware, per machine | `docs/MACHINE_PROFILES.md`, `docs/MACHINE2_BIG_MODEL_PLAYBOOK.md` |
| token-cost techniques (compression, caching, schemas) | `docs/TOKEN_SAVING.md` |
| measured model comparisons, per-model notes | `models/*.md`, `benchmarks/*` |
| free cloud quotas, providers | `INVENTORY_PROVIDERS.md`, `providers/*` (not engine work) |

## How to judge what you find

1. **Provenance first.** `[M]` means measured — on *their* rig, with Ollama or llama.cpp, never on
   Kindi. `[E]` is their arithmetic, `[P]` a vendor claim, `[?]` untested. Every one of them is a
   hypothesis for us; a `[M]` tag does not transfer.
2. **Applicability.** A win for llama.cpp can be a non-issue for a different arithmetic path. State
   what in Kindi would have to change, and which file owns it.
3. **Exactness.** `kindi verify` must stay token-for-token. An approximate idea (quantised KV, a
   changed rounding) is only a roadmap step if it names its tolerance and is labelled non-exact, or
   is gated to a mode that already accepts approximation.
4. **Measurability.** A candidate with no measurement on hardware we have is a parked idea, not a
   roadmap step. Name the gate that would close it.

One good candidate per scan beats eight vague ones. "Already have it, here is the code" and
"rejected, here is why" are both first-class results — they stop the next agent re-litigating it.

## From candidate to roadmap step

- Add or amend the item in `docs/ROADMAP.md`, and record the requirement the way the
  [project-memory skill](../project-memory/SKILL.md) says (one `REQ-NNNN.md`, acceptance + evidence).
- The requirement names the **gate**: the benchmark, exactness check or test that proves the gain,
  and the host it must run on. If it needs hardware we do not have, it goes to the owner-action list
  in `docs/MANUAL-HELP-NEEDED.md` instead of pretending to be plannable.
- Do not implement an idea in the same pass that discovered it unless the user asked: the normal
  outcome of a scan is a ledger row and, at most, a roadmap/requirement step.

## Verifying a win (non-negotiable)

If a candidate is implemented, prove it moved:

```sh
python -m kindi bench ...            # before and after, same host, same pack, same ctx
python -m kindi verify               # exactness, when the change can touch arithmetic
python -m pytest -q                  # the CPU suite
python tools/check_claims.py         # every number the docs quote stays re-derivable
```

Record it as a new `docs/benchmarks/<date>-<host>-<topic>/RESULTS.md` and register the numbers in
`docs/claims/*.json`; a speedup without a measurement is not a win, and a claim quoting the
knowledge base's number as if it were ours is a fabrication.

## Rules

- Read only, always. The knowledge base is the user's, maintained by another agent.
- Never present a knowledge-base number as a Kindi measurement, and never let one skip a gate.
- The ledger is the memory: a scan that is not in `docs/development/AI-INVENTORY-SCAN.md` did not
  happen, and the next agent will redo it.
- Keep the ledger short: one row per scan, one line per verdict, links instead of quotation.
