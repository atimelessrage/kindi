# Release checklist: Kindi 0.7.0-fork

This is the checklist for the fork's `0.7.0-fork` release. It is a **fork release**, not official
Kindi `0.6.1`. The `-fork` suffix, the `atimelessrage/kindi` release URLs and the fork container
registry keep it distinct from upstream.

## Identity decision

- Official baseline: `v0.6.1` (`7fececb`), published 2026-11-16.
- Shared base: `dce89df`; `v0.6.1` is one commit past that base and is **not** an ancestor of this
  fork. The release-code baseline is `20e2ea0` on `freebuff/tandem-cpu-recovery`; the final
  checksum-pin commit follows it.
- Current range at that baseline: `git rev-list --count v0.6.1..20e2ea0` = **674 commits**;
  `git diff --shortstat v0.6.1..20e2ea0` = **891 files, +125,461/−1,406**.
- The **649**-commit figure in `detailed-fork-changelog.md` was the `3002a40` snapshot. The
  `CURRENT-VS-KINDI-0.7.md` audit recorded 666 at `3fed704`; the integration adds six commits.
  Do not repeat 649 as the current count.
- Why `0.7.0-fork`: the fork adds substantial runtime, CLI, API and test surface and must not call
  itself `0.6.1`. The suffix prevents a fork build from being read as an official upstream patch,
  and the version does not encode a commit count that changes as the branch grows.

## Version surfaces (all must agree)

| Surface | Value |
|---|---|
| `kindi/__init__.py` | `VERSION = "0.7.0-fork"` |
| `install.sh` | `KINDI_VERSION="0.7.0-fork"` |
| `install.ps1` | `$KindiWinVersion = '0.7.0-fork'` |
| Docker build tags | `kindi:0.7.0-fork`, `kindi-rocm:0.7.0-fork`, `kindi-rocm7:0.7.0-fork` |
| Published tags after verification | `ghcr.io/atimelessrage/kindi:0.7.0-fork-cuda`, `-rocm6`, `-rocm7`; moving `cuda`, `latest`, `rocm6`, `rocm7` |
| Docs | README, `WHY.md`, `docs/AMD.md`, `docs/DOCKER.md`, `docs/INSTALL.md`, `docs/CONTAINERS.md`, `docs/CARDS.md`, `SECURITY.md`, `NOTICE` |

## Local artifacts (prepared, not published)

| Artifact | sha256 |
|---|---|
| `dist/kindi-0.7.0-fork.tar.gz` | `cb03cc938cfcda6bf2ff7afa4467153e0bf6ef770e2c704951295c2e114ae61b` |
| `dist/kindi-0.7.0-fork.tar.gz.sha256` | sidecar containing the archive hash |
| `install.sh` | `b38de76f9f9beef216d4ac3c832c3c0ce881e08f2db5b11ee04e067e5f28fd40` |
| `install.ps1` | `0c25abe5f6806ee57070b52b8dd04c1509f4a1cda583fef729b407a9a7ef2888` |

The archive is built from the working tree with:

```sh
python scripts/build_release_archive.py
(cd dist && sha256sum -c kindi-0.7.0-fork.tar.gz.sha256)
```

It includes tracked files plus non-ignored untracked files (including the two new
`token_classes` files), and excludes `.git`, `dist/`, `AGENTS.md`, both installers, `docs/INSTALL.md`, this
checklist and `.secrets.baseline`. The exclusions are deliberate: the installers and checksum pages
are separate assets, and the secret-scanner baseline follows the installer hash, so including them
would make the archive hash depend on itself. The archive writes the base HEAD
commit into `kindi/COMMIT`; the working tree itself keeps the `$Format:%H$` placeholder.

`install.sh` pins the archive hash in `SOURCE_SHA256` and downloads the archive from
`https://github.com/atimelessrage/kindi/releases/download/v0.7.0-fork/kindi-0.7.0-fork.tar.gz`.
**The URL is not live until the release is published.** The local archive is a release candidate,
not a claim that the asset already exists.

## Container smoke status (2026-11-20)

The Python 3.6 blocker is fixed. Removed the 3.7-only `from __future__ import annotations` import
from `kindi/sessions.py` and the three other modules that carried it (`kindi/cascade.py`,
`kindi/draft_policy.py`, `kindi/moe_compat.py`); none uses postponed annotations. The rebuilt CUDA
image `kindi:0.7.0-fork` (`sha256:b08916d7d25a19c87831196067a4407d0dac5d1143237b22c97dc59aa3b45af1`,
2.56 GB, Python 3.6.9) passed the **unmodified-image** smoke on the RTX 4060 Ti: `/health` returned
200, and one non-streaming chat returned 200 in **0.2337 s** with `container smoke ok`. GPU
passthrough was checked with `nvidia-smi` and `kindi info`; the latter exits 0 and reports the
RTX 4060 Ti plus the `/data/pack` 2B pack. Build and run outputs, including the full `kindi info`
output: [container smoke evidence](benchmarks/2026-11-20-i7-8086k-rtx4060ti-container-smoke/RESULTS.md).
The image label records revision `1627b96312509e6e2bdaad805116f7f6459b88f3-dirty`; the
`kindi/COMMIT` placeholder still makes `/health` report `commit: null`. No image was published.

The archive above was rebuilt from `20e2ea0` after the Python 3.6 fix, so it contains the
repaired `kindi/sessions.py`; the image label still records its earlier dirty build.

## 1. Verify the tree before building

```sh
git status --short
python -m pytest tests/test_version_consistency.py tests/test_installer_source.py -q
bash -n install.sh
python -m compileall -q kindi
git diff --check
```

Expected:

- `test_version_consistency.py` passes; every image tag in README/CARDS/DOCKER/AMD/INSTALL and the
  three Dockerfiles equals `0.7.0-fork`.
- `test_installer_source.py` passes; the source block still downloads to `.part`, checks sha256,
  and only then extracts.
- `bash -n install.sh` and `compileall` exit 0.
- No whitespace errors.

Then run the full gate on a machine with the test dependencies:

```sh
python -m pytest tests/ -q
python tools/check_claims.py
python tools/check_python_quality.py
python tools/check_secret_history.py
python tools/check_secret_baseline.py
python -m pytest tests/test_docs_links.py tests/test_project_memory.py -q
python tools/project_memory.py --check
```

Record the exact counts in the release notes. Do not claim a GPU/model benchmark or a Docker image
verification that was not run on this machine.

## 2. Build, hash and pin

```sh
python scripts/build_release_archive.py
sha256sum install.sh install.ps1 dist/kindi-0.7.0-fork.tar.gz
```

1. Copy the archive hash into `install.sh` (`SOURCE_SHA256`), `docs/INSTALL.md`, this checklist and
   the final release notes.
2. Copy the final `install.sh` and `install.ps1` hashes into `docs/INSTALL.md` and this checklist.
3. Re-run `python -m pytest tests/test_version_consistency.py tests/test_installer_source.py -q`
   and `bash -n install.sh`.
4. Confirm `(cd dist && sha256sum -c kindi-0.7.0-fork.tar.gz.sha256)` prints `OK`.
5. Do **not** rebuild the archive after this point unless source files change. If they do, rebuild,
   update all three hashes and rerun the checks; the archive excludes the checksum pages and
   installers, so changing those hashes alone does not change the archive.

## 3. Docker tags and images

The tree defines the tags, but no fork image has been built or published for this release:

```sh
docker build -t kindi:0.7.0-fork -f docker/Dockerfile \
  --build-arg KINDI_VERSION=0.7.0-fork --build-arg KINDI_REVISION=$(git rev-parse HEAD) .
docker build -t kindi-rocm:0.7.0-fork -f docker/Dockerfile.rocm \
  --build-arg KINDI_VERSION=0.7.0-fork --build-arg KINDI_REVISION=$(git rev-parse HEAD) .
docker build -t kindi-rocm7:0.7.0-fork -f docker/Dockerfile.rocm7 \
  --build-arg KINDI_VERSION=0.7.0-fork --build-arg KINDI_REVISION=$(git rev-parse HEAD) .
```

Before publishing any image tag:

- Run `kindi verify` on a real card for each backend. The 0.6.0 tags are the last verified images;
  the 0.7.0-fork tags are not verified by this bundle.
- Push `0.7.0-fork-cuda`, `0.7.0-fork-rocm6` and `0.7.0-fork-rocm7` to
  `ghcr.io/atimelessrage/kindi` only after each check passes.
- Move `cuda`, `latest`, `rocm6` and `rocm7` only after the versioned tags are in place.
- If a backend is not verified, do not publish its tag and correct `docs/CONTAINERS.md` before
  release.

## 4. Commit and tag

- Commit the release edits and the release notes together. Do not amend or reuse a published tag.
- Rebuild the archive from that release commit and update the hashes (the current local archive is
  based on `60a039b` plus uncommitted release edits; its `kindi/COMMIT` says `60a039b`).
- Tag `v0.7.0-fork` only from the release commit.
- Do not force-push or move an existing tag.
- Read `SECURITY.md` before any push. Pushing/tagging requires the owner's explicit authorization.

## 5. Publish atomically (owner action; not done here)

1. Create the GitHub release `v0.7.0-fork` with the exact `install.sh`, `install.ps1`,
   `kindi-0.7.0-fork.tar.gz` and the two `.sha256` sidecars.
2. Verify every public URL with `curl -fsSLO ... && sha256sum -c ...`.
3. Confirm the `sha256sum -c` commands in `docs/INSTALL.md` work from the downloaded files.
4. Run `bash install.sh --dry-run` from the downloaded release on Linux.
5. Keep the previous installer available for rollback.

Publishing, tagging, pushing and image uploads are **not** part of this prepared bundle. Until
they happen, the release is a local candidate and the `v0.7.0-fork` URLs are intentionally not
claimed as live.

## 6. Post-release

- Record the release commit, tag, archive hash, installer hashes and public URLs in the release
  notes.
- Confirm `docs/INSTALL.md` at the public documentation URL shows the same values.
- Watch the first installer reports for the GRUB and source-checksum paths; a checksum mismatch
  must stop before extraction.

## Rollback

Restore the previous installer and its documented hash first, then fix the release artifact. Do not
leave a published file whose checksum disagrees with `docs/INSTALL.md` or this checklist.
