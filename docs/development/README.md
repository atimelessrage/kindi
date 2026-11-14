# Development memory

This directory is the internal, durable record of **why Kindi is built the way it is**.
It complements the user-facing documentation in `docs/` and the handoff notes in
`AGENTS.md`. Start here after a context reset.

| file | use it for |
|---|---|
| [`requirements/`](requirements/) | One stable, implementable requirement per `REQ-NNNN.md` file. |
| [`REQUIREMENTS-HISTORY.md`](REQUIREMENTS-HISTORY.md) | Generated index with added, last-modified and deleted commit hashes. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Current components, data flow, boundaries and invariants. |
| [`DECISIONS.md`](DECISIONS.md) | Append-only design decisions with context and consequences. |
| [`LESSONS.md`](LESSONS.md) | Verified surprises and mistakes worth not repeating. |
| [`AGENTS-HISTORY.md`](AGENTS-HISTORY.md) | The dated history moved out of `AGENTS.md` on 2026-11-14: the full task lease table, the 2026-11-08 snapshot, the handoff notes, the old plan and the resume protocol. |
| [`KNOWLEDGE.md`](KNOWLEDGE.md) | Operational facts, commands and support caveats. |
| [`../../.agents/skills/project-memory/SKILL.md`](../../.agents/skills/project-memory/SKILL.md) | The project-local workflow skill. |

## Authoritative sources

When two pages disagree, use the source for that kind of claim:

| claim | authoritative source |
|---|---|
| supported capabilities and unverified targets | `docs/SUPPORT-MATRIX.md` and `docs/CARD-SUPPORT.md` |
| exactness requirements | `docs/development/requirements/REQ-*.md` with `REQUIREMENTS-HISTORY.md` |
| configuration keys and environment variables | generated `docs/CONFIGURATION-REFERENCE.md` |
| tandem runtime status and measured numbers | `docs/TANDEM-STATUS.md` and `docs/TANDEM-RUNTIME.md` |
| benchmark tables | generated `docs/benchmarks/CROSS-MODE.md` / `.csv` |
| release pin and installer artifact | `docs/RELEASE-CHECKLIST.md` and `docs/INSTALL.md` |
| secret baseline and scan policy | `.secrets.baseline`, `tools/check_secret_baseline.py` and `tests/test_no_secrets.py` |

## The rule

A change is not complete until its requirement, design, decision, lesson or external
documentation change is recorded here. Do not leave a durable fact only in a chat.

### Requirements

Each requirement has a permanent filename:

```text
docs/development/requirements/REQ-0001.md
```

The filename is the ID. Keep the ID stable when the text changes. Use the template:

```markdown
# REQ-0001: A short requirement

- Status: active

## Requirement

What must be true, in observable terms.

## Acceptance

How a reviewer can tell it is true.

## Evidence

Tests, commands, docs or verification records that prove it.
```

Valid statuses are `active`, `experimental`, `proposed` and `retired`. Deleting a
requirement file is a real removal; the generator keeps it in the deleted section.
Prefer `retired` for a requirement that is still historically useful.

### Commit history

Run this after **committing** requirement changes:

```sh
python tools/project_memory.py
python tools/project_memory.py --check
```

The generated index records the first commit that added a file and the latest commit
that changed it. New, uncommitted files are shown as `uncommitted`; commit them and run
the generator again. To inspect an old state:

```sh
git show <added-or-modified-hash>:docs/development/requirements/REQ-0001.md
git log --follow -p -- docs/development/requirements/REQ-0001.md
```

The generated file is not edited by hand and is checked by
`tests/test_project_memory.py`.

### Decisions and lessons

- Add a **decision** to `DECISIONS.md` when a choice is hard to reverse, affects an
  invariant or support claim, or is likely to be revisited. Record the alternatives.
- Add a **lesson** to `LESSONS.md` when a test or hardware run contradicts an assumption,
  or when a bug required a non-obvious guard.
- Put stable operational facts in `KNOWLEDGE.md`, not in a commit message or chat.

Records are append-only in practice: do not rewrite an old decision to make it look as
if it was always correct. Mark it `superseded` and link the replacement.

## Roadmap reconciliation

When `docs/ROADMAP.md` or a gate in a requirement changes:

1. Link the roadmap item to the requirement ID whose acceptance gate it uses.
2. Update the requirement status, acceptance criteria and evidence; prefer `retired` or
   `superseded` over deleting still-useful history.
3. Add a decision or lesson when the outcome changes a hard-to-reverse choice or reveals
   an assumption that was wrong.
4. Regenerate `REQUIREMENTS-HISTORY.md` after the requirement files are committed.
5. Update user-facing docs and examples for any resulting behavior change.

A roadmap item is not done merely because code exists. Its requirement must point at the
test, benchmark log or commit that satisfies the stated gate, or the roadmap must keep it
open, blocked, experimental or retired.

## External documentation

User-visible behavior belongs in `README.md`, `docs/*.md` or `examples/`, not only in
this directory. When a requirement changes:

1. Update the matching external page and any examples.
2. Regenerate generated references with
   `python -m kindi.config_reference` and `python -m kindi.support_matrix`.
3. Run `python -m pytest tests/test_docs_links.py tests/test_config_reference.py tests/test_support_matrix.py -q`.

The existing gates already fail on broken local links, stale generated configuration or
support matrices, and version inconsistencies.

## Definition of done

```sh
python tools/project_memory.py --check
python -m pytest tests/ -q
python -m compileall -q kindi
bash -n install.sh
git diff --check
```

State explicitly which hardware-dependent checks were not run.
