# Contributing to Kindi

Bug reports, questions, test results and ideas are welcome. You do not need to sign a contributor agreement to
open an issue or use Kindi. For security reports, use the private process in [SECURITY.md](SECURITY.md).

## Before you open a pull request

Run the CPU-only gate from the repository root; it needs no GPU, model pack or Docker:

```sh
python -m compileall -q kindi
python -m pytest tests/ -q
bash -n install.sh
```

For installer changes, also run the GRUB fixture test and ShellCheck:

```sh
python -m pytest tests/test_installer_grub.py -q
shellcheck --severity=warning install.sh
```

`tests/README.md` describes the optional Turing/CUDA reference suite and static analysis. Report which checks
you ran, and say explicitly when a hardware-dependent path was not tested.

## Project memory

Internal requirements, architecture, decisions, lessons and operational knowledge live in
`docs/development/`. Read the relevant record before changing behavior, and update it when
the change lands. Requirements are one `REQ-NNNN.md` file each; after committing a
requirement change, run `python tools/project_memory.py` and commit the regenerated
`REQUIREMENTS-HISTORY.md`. The project-local workflow is
`.agents/skills/project-memory/SKILL.md`.

User-visible changes also need the relevant `README.md`, `docs/*.md` page and examples
updated. Run `python tools/project_memory.py --check`,
`python -m pytest tests/test_project_memory.py tests/test_docs_links.py -q` and the normal
gate above.

## Code and other submitted material

Before we include submitted material under the contributor agreement, we ask for explicit agreement to
[CLA.md](CLA.md). You keep your copyright. The agreement lets Amir Husain include the work in Kindi under
AGPL-3.0 and under a separate commercial license. Kindi's AGPL license also permits commercial use under its
terms; the separate license offers other terms.

AI-assisted work is welcome on the same terms. Disclose the tools and any third-party sources, review the work,
and confirm that you have the rights to grant the licenses. Do not submit copied code whose terms prevent that
grant. If rights are uncertain, say so before you make the promise.

## Reports and credit

We can discuss separate permission to quote a report or publish benchmark data. You do not need to sign the code agreement to report a bug or receive credit for an authorized report. Please provide the version,
model, hardware, commands and logs needed to check a result. Distinguish a stock build from a local patch.

## Policy history

We first published the agreement on the project website on 2026-10-30, after some users had sent their work. We
added it to this repository on 2026-10-31. We acknowledge that earlier submitters did not have this policy in the
repository when they submitted their work. We do not treat those submissions as agreement. We ask for explicit
permission before we include them under these terms and respect a decision not to contribute.

The original text is in [docs/cla/1.0.md](docs/cla/1.0.md). [CLA.md](CLA.md) is version 1.1. Acceptance of a previous
version remains governed by that version. A signed agreement does not guarantee that we will merge
a submission; we also review scope, correctness and maintenance needs.
