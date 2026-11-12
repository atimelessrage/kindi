---
name: project-memory
description: Keep Kindi requirements, architecture, decisions, lessons, knowledge, and user documentation synchronized with git.
license: AGPL-3.0-only
metadata:
  category: development
---

# Project memory

Use this skill whenever you change Kindi's behavior, supported hardware, configuration,
public API, generated documentation, or development process. The goal is that a future
agent can recover **what was required, what was decided, what was learned, and which git
state implemented it** without relying on chat history.

Start with [`docs/development/README.md`](../../../docs/development/README.md). Read the
relevant requirement files and [`docs/development/ARCHITECTURE.md`](../../../docs/development/ARCHITECTURE.md)
before editing.

## Workflow

1. **Classify the change.**
   - New requirement: add `docs/development/requirements/REQ-NNNN.md` using the next free
     number. Keep IDs stable forever; do not renumber on edits.
   - Changed requirement: edit the same file and update **Acceptance** and **Evidence**.
   - Removed requirement: delete the file. The generator preserves it in the deleted
     section; use `retired` instead when it remains useful history.
2. **Record durable reasoning.**
   - `docs/development/DECISIONS.md`: hard-to-reverse choices, alternatives and
     consequences. Mark superseded decisions; do not rewrite them.
   - `docs/development/LESSONS.md`: verified surprises, failures and non-obvious guards,
     with evidence.
   - `docs/development/KNOWLEDGE.md`: stable commands, paths, support caveats and
     operational facts.
3. **Update the architecture** when a component, boundary, flow or invariant changes.
4. **Update external documentation and examples** for every user-visible change:
   `README.md`, the relevant `docs/*.md` page, `examples/`, and generated references.
   Internal documentation is not a substitute for a user-facing instruction.
5. **Commit the requirement/implementation change.** Then run:

   ```sh
   python tools/project_memory.py
   python tools/project_memory.py --check
   ```

   The generated [`REQUIREMENTS-HISTORY.md`](../../../docs/development/REQUIREMENTS-HISTORY.md)
   must be committed after the requirement files. New files show `uncommitted` until
   their first commit; rerun the generator after that commit.

6. **Run the relevant checks.** The full definition of done is in
   `docs/development/README.md`; at minimum run the project-memory test, docs links,
   generated-reference checks, the CPU suite, `compileall`, `bash -n install.sh` and
   `git diff --check`.

## Rules

- Never edit `REQUIREMENTS-HISTORY.md` by hand. Never put a fabricated commit hash in a
  decision, lesson or requirement.
- Do not leave a durable fact only in a commit message, review comment or chat.
- Do not claim hardware support from an unmeasured path. Mark it proposed, experimental
  or explicitly unsupported, and record the missing verification.
- Keep requirement files small and observable; link to architecture, tests and external
  docs instead of duplicating long explanations.
- Preserve deleted requirement files in git history; the deleted section is intentional.
- The project-memory test is a gate, not a suggestion.
