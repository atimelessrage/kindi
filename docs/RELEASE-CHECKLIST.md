# Release checklist: installer safety fixes

This checklist is for releases that contain the installer safety fixes: the GRUB guard
(`cwsr_edit_grub()`) and the source-archive checksum. The GRUB guard backs up `/etc/default/grub`,
rewrites the kernel command line, re-reads it, and restores the backup and stops when the edit fails,
adds nothing, or drops an existing parameter. The guard is committed in `2b84658`; the fixture test is
`tests/test_installer_grub.py`.

## Source archive integrity (this change)

`install.sh` now downloads the release source archive to a `.part` file and checks its sha256 before
extracting it. It no longer pipes `curl` into `tar`. The default v0.5.0 archive is pinned to
`26424ec08caf07a06a4f57547aa50d5b62a85ec8f7e8c231dcadc5c81519b1e2`; a custom `--source` URL can be
pinned with `KINDI_SOURCE_SHA256`. The regression test is `tests/test_installer_source.py`.

This change is published as the fork-hosted 0.5.0 installer at
`https://github.com/atimelessrage/kindi/releases/download/v0.5.0-installer/install.sh`.
The LF artifact hash is `5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6`; `docs/INSTALL.md` records it.

Artifact hash for this worktree:

| candidate | sha256 |
|---|---|
| LF release artifact | `5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6` |

Choose one exact candidate, publish that file, then replace both installer checksums in `docs/INSTALL.md`
with its hash. Do not update the documented published hash before the upload: users would then check a
file that is not yet served.

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
python -m pytest tests/test_installer_source.py -q
python -m pytest tests/ -q
python -m compileall -q kindi
```

Expected for this release:

- `bash -n install.sh` prints nothing and exits 0.
- `tests/test_installer_grub.py` passes all 9 tests.
- `tests/test_installer_source.py` passes all 4 tests, including the mismatch path that must not extract.
- The full CPU suite passes (242 passed, 3 skipped at the last handoff).
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
| the published LF artifact | `5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6` |

The published 0.5.0 installer is the LF artifact above. This release records that hash in
`docs/INSTALL.md`; publish that exact file, not a re-saved copy.

```sh
sha256sum install.sh
# must print 5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6
```

The published artifact is LF. Do not re-save it with different line endings. Update **both** checksum lines in
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
grep -n '5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6' docs/INSTALL.md
# two lines expected
```

Download the candidate artifact and check it exactly as a user would:

```sh
curl -fsSLO https://github.com/atimelessrage/kindi/releases/download/v0.5.0-installer/install.sh
echo "5f48beca4018d993f3250e8dcbc48b43f04bd678667c5e9222a35193df8590f6  install.sh" | sha256sum -c -
bash -n install.sh
```

If the URL still serves the old `592a…` file, the release has not been published; do not change the
documented hash to the old file.

## 5. Commit and tag

- Commit the guard (already in `2b84658`), the checksum update, and this checklist together in the release
  history. Do not amend or reuse a published tag.
- Decide the release version before tagging. `install.sh` and `kindi/__init__.py` currently say `0.5.0`; if
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
