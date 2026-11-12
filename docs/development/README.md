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
| [`KNOWLEDGE.md`](KNOWLEDGE.md) | Operational facts, commands and support caveats. |
| [`../../.agents/skills/project-memory/SKILL.md`](../../.agents/skills/project-memory/SKILL.md) | The project-local workflow skill. |

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
