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

This change is a candidate, not a published installer. The live artifact documented in
`docs/INSTALL.md` remains the published `b84df4fe...` file until the release owner uploads the new one.
Candidate hashes for this worktree:

| candidate | sha256 |
|---|---|
| LF build | `c3d1cfde7d1191b79541975a9a54c62935b5f58e3759178b4528910d6404d354` |
| CRLF worktree | `fc93fe956a881deb3932360ec579de96444991066490c9d3f6e718b2a57f55b9` |

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
- The full CPU suite passes (run the current suite; do not rely on a historical count).
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

The published 0.5.0 file is `b84df4fe0b4ae258433d043a34c67d0f9daaeecc0f9131eea8a2e3990f87d872`,
as recorded in `docs/INSTALL.md`. The installer in this worktree is a candidate because it adds the
CPU preflight and the source-archive pin. Its candidate hashes are:

| source | sha256 |
|---|---|
| current worktree file (CRLF) | `fc93fe956a881deb3932360ec579de96444991066490c9d3f6e718b2a57f55b9` |
| LF-normalized copy | `c3d1cfde7d1191b79541975a9a54c62935b5f58e3759178b4528910d6404d354` |

```sh
sha256sum install.sh
# compare with the CRLF candidate above
sed 's/\r$//' install.sh | sha256sum
# compare with the LF candidate above
```

Choose one exact candidate, publish that file, then update **both** checksum lines in
`docs/INSTALL.md` and this table. Do not update the published hash before the upload: users would
then check a file that is not yet served. Never publish a file whose hash differs from the documented
value.

Confirm the guard is present and no control byte remains:

```sh
grep -n 'cwsr_edit_grub\|cwsr_line_ok' install.sh
grep -nP '\x01' install.sh && echo 'STOP: control byte in installer'
```

## 4. Update and verify the documentation

`docs/INSTALL.md` has the checksum in two places: the `sha256sum -c` command and the file table. Both must
match the release artifact.

```sh
grep -n 'b84df4fe0b4ae258433d043a34c67d0f9daaeecc0f9131eea8a2e3990f87d872' docs/INSTALL.md
# two lines expected for the published file
```

Download the candidate artifact and check it exactly as a user would:

```sh
curl -fsSLO https://amirhusain.com/kindi/install.sh
echo "b84df4fe0b4ae258433d043a34c67d0f9daaeecc0f9131eea8a2e3990f87d872  install.sh" | sha256sum -c -
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
