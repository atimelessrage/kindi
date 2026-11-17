# Release checklist: installer safety fixes

This checklist is for releases that contain the installer safety fixes: the GRUB guard
(`cwsr_edit_grub()`), the source-archive checksum and the AVX CPU preflight. The GRUB guard backs up `/etc/default/grub`,
rewrites the kernel command line, re-reads it, and restores the backup and stops when the edit fails,
adds nothing, or drops an existing parameter. The guard is committed in `2b84658`; the fixture test is
`tests/test_installer_grub.py`.

## AVX CPU preflight (this change)

`install.sh` reads `/proc/cpuinfo` (or `KINDI_CPUINFO` in tests) for `avx` before it offers a CPU
setup. A CPU without AVX hides the CPU setups; `--gpu cpu` and a machine with no usable GPU stop
with an actionable message before downloading or building anything. GPU installs remain available
on a CPU without AVX. The fixture test is `tests/test_installer_cpu.py`: AVX is accepted,
AVX2/FMA/F16C without AVX is rejected, a missing CPU table is rejected, and the CPU setup guard is
present.

## Source archive integrity (this change)

`install.sh` now downloads the release source archive to a `.part` file and checks its sha256 before
extracting it. It no longer pipes `curl` into `tar`. The default v0.5.0 archive is pinned to the fork's main source archive at commit
`fd5414889e4acfd72b46bba3138a908b8e97c5be`, sha256
`3ca1e79d371c87bea5471841db10b0509b90d3bd557512cf59fc83854b929a94`; a custom `--source` URL can be
pinned with `KINDI_SOURCE_SHA256`. The regression test is `tests/test_installer_source.py`.

Current-head note: the published installer still installs commit `fd54148`, not the current best-of
`main` tree. Refreshing `SOURCE_COMMIT`/`SOURCE_SHA256` (and `KINDI_VERSION` if bumped) requires the
release owner to publish and verify the exact source archive for the desired release commit, then update
`docs/INSTALL.md`, this checklist and `tests/test_installer_source.py` in the same commit. Until that
publication happens, the installer payload is the `fd54148` release, not current HEAD.

This change is published as the fork-hosted 0.5.0 installer at
`https://github.com/atimelessrage/kindi/releases/download/v0.5.0-installer/install.sh`.
The LF artifact hash is `43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99`; `docs/INSTALL.md` records it. The matching
`install.sh.sha256` asset is published at the same release. The previous pair is retained as
`install.sh.pre-avx-preflight-1794385057` and
`install.sh.sha256.pre-avx-preflight-1794385057` for rollback.

Artifact hash for this worktree:

| candidate | sha256 |
|---|---|
| LF release artifact | `43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99` |

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
python -m pytest tests/test_installer_cpu.py -q
python -m pytest tests/ -q
python -m compileall -q kindi
```

Expected for this release:

- `bash -n install.sh` prints nothing and exits 0.
- `tests/test_installer_grub.py` passes all 9 tests.
- `tests/test_installer_source.py` passes all 4 tests, including the mismatch path that must not extract.
- `tests/test_installer_cpu.py` passes all 4 tests for the AVX preflight.
- The full CPU suite passes (254 passed, 7 skipped at the last handoff).
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
| the published LF artifact | `43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99` |

The published 0.5.0 installer is the LF artifact above. This release records that hash in
`docs/INSTALL.md`; publish that exact file, not a re-saved copy.

```sh
sha256sum install.sh
# must print 43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99
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
grep -n '43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99' docs/INSTALL.md
# two lines expected; the published install.sh.sha256 asset uses the binary-mode '*' marker
```

Download the candidate artifact and check it exactly as a user would:

```sh
curl -fsSLO https://github.com/atimelessrage/kindi/releases/download/v0.5.0-installer/install.sh
echo "43766d8bbed09fa0ccebe6c39c8076001da99cb15814ff065380f7a46c93db99  install.sh" | sha256sum -c -
bash -n install.sh
```

If the URL still serves the previous artifact (`5f48beca…`), the release has not been published; do not
change the documented hash to the old file.

## 5. Commit and tag

- Run the secret scans before tagging, not only the tracked-file one:
  `python tools/check_secret_history.py` fails on any new credential or personal-data row in history
  (it compares against `tools/secret_history_baseline.json`), and `python tools/check_secret_history.py
  --all-hits` prints every recorded row so the release owner can decide whether a row still belongs
  there. Also run `detect-secrets scan --all-files` on a clean export and confirm `.secrets.baseline`
  still matches. The tracked-file test scans neither commit metadata nor deleted lines.
- Commit the guard (already in `2b84658`), the checksum update, and this checklist together in the release
  history. Do not amend or reuse a published tag.
- Decide the release version before tagging. `install.sh` and `kindi/__init__.py` currently say `0.5.0`; if
  this is a new patch release, bump both and add the changelog entry in the same release commit.
- Tag only from the release commit, and only after the artifact hash is fixed.
- Do not force-push or move an existing release tag.

## 6. Publish atomically

1. Upload the exact `install.sh` artifact named in `docs/INSTALL.md` and the matching `install.sh.sha256`.
2. Verify both public URLs with `curl | sha256sum`.
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
