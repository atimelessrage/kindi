# Security and push policy

This page is the authoritative policy for secrets, the 2026-11-17 history rewrite and any push
from this checkout. It exists because a push publishes every object reachable from the pushed refs,
not just the current tree.

## Secrets

- Do not commit credentials, API keys, tokens, private URLs or personal paths. Test fixtures use
  reserved domains (`example.invalid`) and fake values.
- The present tree is covered by `tests/test_no_secrets.py` (tracked and untracked text), the
  detect-secrets baseline, and `tools/check_secret_history.py` for commit subjects, bodies and
  deleted lines. The history baseline records legacy hits; it is a counted snapshot, not an
  approval of those values. The forward-slash drive-root rule is current-tree-only: the reviewed
  history snapshot predates it, so a path added and deleted in one commit is the remaining gap.
  Keep reviewing diffs for machine paths; the current-tree guard is the primary check. The
  `local drive path` rule allows the default Windows system roots (`%WINDIR%` and
  `%ProgramFiles%`) because a recorded driver registration path is evidence, not a person's
  layout; test fixtures are exempt from that one rule, while a user home directory is still
  caught by `local home path`.
- A real key that reaches a commit must be rotated first. Removing it from the working tree or
  rewriting local history does not revoke it.
- Report a suspected leak privately to the repository owner rather than opening a public issue
  with the value.

## History redaction (2026-11-17)

The local history was rewritten on 2026-11-17 to remove legacy personal-data hits: one redacted
commit and its 32 descendants were re-hashed. The current `main` contains the rewritten history.
The old objects still exist in the local object store, in archival branches and in the stale
remote object stores. The 2026-11-18 audit observed `origin/main` at `dce89df` and `myfork/main` at
`fd54148`; neither is current, and the redaction commit is not an ancestor of the fork's published
tip. Treat both remotes as holding pre-redaction objects until the owner verifies otherwise.

## Current history baseline and disclosure decision

The 2026-11-19 audit re-counted the history gate at **160 rows / ~3,318 legacy hits**. The
current tree is clean; the hits are old benchmark dumps, prior revisions and commit messages. The
project decision is to **keep the reviewed count baseline and not rewrite history again**: the
2026-11-17 rewrite already re-hashed 32 descendants and left old objects in the local store and
in stale remote object stores, so another rewrite would add risk without removing those copies.
This is disclosure, not approval: a first public push must name this exposure in the release
notes, and the history baseline stays gated so no new row or count can be added. A future rewrite
needs a new signed review, a fresh remote audit and the owner's explicit authorization; it is not
an agent's routine cleanup.

## Push policy

- Agents do not push. A push needs the repository owner's explicit authorization for that exact
  ref and commit range.
- Before any push, fetch and inspect the range (`git log --oneline <remote>/<branch>..<branch>`),
  run `python tools/check_secret_history.py`, and confirm the destination's history. Do not infer a
  clean remote from a clean working tree.
- Never force-push a published branch or move a published tag. If a history rewrite must be
  published, publish it under a new ref or coordinate the rewrite with every clone owner; do not
  use `--force` as a shortcut.
- A fast-forward of a stale remote is not automatically safe: it can make pre-redaction objects
  reachable again. When in doubt, push to a new branch and let the owner compare it.
- Release artifacts, installer hashes and `SOURCE_COMMIT` follow
  [RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md). A tag move is a new release, never an edit of an
  existing one.

## Backup

Pushing is not a backup, and this checkout has no successful offsite push. Keep a verified
`git bundle` on a second medium as described in [BACKUP.md](BACKUP.md). A bundle contains the
archival refs and the legacy objects, so store it encrypted.
