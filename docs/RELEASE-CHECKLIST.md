# Release checklist: installer GRUB safety fix

This checklist is for the release that contains `cwsr_edit_grub()` in `install.sh`. The guard backs up
`/etc/default/grub`, rewrites the kernel command line, re-reads it, and restores the backup and stops when
the edit fails, adds nothing, or drops an existing parameter. The guard is committed in `2b84658`; the
fixture test is `tests/test_installer_grub.py`.

Do not publish from a worktree with an empty, stale, or locally modified `install.sh`.

## 1. Reconcile the worktrees first

```sh
git worktree list --porcelain
git status --short
git -C T:/kindi status --short -- install.sh docs/INSTALL.md
```

- The main worktree must not have an empty or uncommitted `install.sh` when publishing.
- If the main worktree has uncommitted `docs/INSTALL.md` changes, reconcile or hand them off before
  publishing; do not copy a working file between worktrees.
- Work on the branch that contains the guard, not on the empty main copy.

## 2. Run the no-GPU gate

Run from the repository root on Linux or a compatible shell:

```sh
bash -n install.sh
python -m pytest tests/test_installer_grub.py -q
python -m pytest tests/ -q
python -m compileall -q kindi
```

Expected for this release:

- `bash -n install.sh` prints nothing and exits 0.
- `tests/test_installer_grub.py` passes all 9 tests.
- The full CPU suite passes (58 tests at the last handoff).
- No test edits `/etc/default/grub`; the GRUB test uses the fixture and a temporary copy.

If you change the GRUB edit, reintroduce the old literal `0x01` bug on a copy and confirm the failure-path
tests fail. Do not run the installer against the real `/etc/default/grub` for this check.

On Linux, also run the installer without changing anything:

```sh
bash install.sh --dry-run
```

It needs Linux, Docker access and a model pack for the later checks; a dry run alone is not a GPU or
model verification.

## 3. Build and hash the exact release artifact

The published file is the **release artifact**, not necessarily the Git blob. Two hashes are relevant:

| source | sha256 |
|---|---|
| the current worktree release file (CRLF, `38fb…`) | `38fb341797825160f620045dd37ece2034e70e7b2a3deab76d5d2edea701fe4e` |
| the committed Git blob (LF, `9b25…`) | `9b25e78ba293ffdf0da9cfccdb5ea23ee787dce4ad7c8c6157507a2744a0da23` |

The published 0.4.0 file was an artifact hash (`592a…`), not the Git blob. This release records the
worktree artifact hash `38fb…` in `docs/INSTALL.md`; publish that exact file, not a re-saved copy.

```sh
sha256sum install.sh
# must print 38fb341797825160f620045dd37ece2034e70e7b2a3deab76d5d2edea701fe4e
```

If the release pipeline normalizes line endings, or you build the artifact from the Git blob instead,
the hash becomes `9b25e78…`. Stop, choose the exact artifact, and update **both** checksum lines in
`docs/INSTALL.md` plus this table before publishing. Never publish a file whose hash differs from the
documented value.

Confirm the guard is present and no control byte remains:

```sh
grep -n 'cwsr_edit_grub\|cwsr_line_ok' install.sh
grep -nP '\x01' install.sh && echo 'STOP: control byte in installer'
```

## 4. Update and verify the documentation

`docs/INSTALL.md` has the checksum in two places: the `sha256sum -c` command and the file table. Both must
match the release artifact.

```sh
grep -n '38fb341797825160f620045dd37ece2034e70e7b2a3deab76d5d2edea701fe4e' docs/INSTALL.md
# two lines expected
```

Download the candidate artifact and check it exactly as a user would:

```sh
curl -fsSLO https://amirhusain.com/kindi/install.sh
echo "38fb341797825160f620045dd37ece2034e70e7b2a3deab76d5d2edea701fe4e  install.sh" | sha256sum -c -
bash -n install.sh
```

If the URL still serves the old `592a…` file, the release has not been published; do not change the
documented hash to the old file.

## 5. Commit and tag

- Commit the guard (already in `2b84658`), the checksum update, and this checklist together in the release
  history. Do not amend or reuse a published tag.
- Decide the release version before tagging. `install.sh` and `kindi/__init__.py` currently say `0.4.0`; if
  this is a new patch release, bump both and add the changelog entry in the same release commit.
- Tag only from the release commit, and only after the artifact hash is fixed.
- Do not force-push or move an existing release tag.

## 6. Publish atomically

1. Upload the exact artifact named in `docs/INSTALL.md`.
2. Verify the public URL with `curl | sha256sum`.
3. Re-run the user-facing `sha256sum -c` command from `docs/INSTALL.md`.
4. Confirm `bash install.sh --dry-run` works from the downloaded release.
5. Keep the previous installer available for rollback.

Publishing, tagging and pushing require the owner's explicit authorization; preparing this commit does not
publish anything.

## 7. Post-release

- Record the release commit, tag, artifact hash and public URL in the release notes.
- Confirm `docs/INSTALL.md` at the public documentation URL shows the same hash.
- Watch the first installer reports for the GRUB step; a restore message means the guard worked and the
  installer stopped instead of corrupting the kernel command line.

## Rollback

If the public artifact hash is wrong, restore the previous installer and its documented hash first, then
fix the release artifact. Do not leave a published file whose checksum does not match `docs/INSTALL.md`.
