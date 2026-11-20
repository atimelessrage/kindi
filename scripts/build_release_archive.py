#!/usr/bin/env python3
"""Build the local source archive and its SHA-256 sidecar.

The archive is built from the working tree (tracked files plus untracked files that are not
ignored), not from HEAD, so release edits that are not committed yet are included. The
installers and the two pages that contain release checksums are excluded: they are separate
release assets, and including them would make the archive hash depend on itself.

The archive contains a ``kindi/COMMIT`` file with the base HEAD commit, so ``kindi --version``
and ``/health`` report the commit without a ``.git`` directory. Rebuild after the release commit
is made to record the final commit. The secret-scanner baseline is development metadata that
follows the installer hash, so it is excluded as well; otherwise updating it would change the
archive hash and make the installer pin stale again.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXCLUDED = {
    "AGENTS.md",
    "install.sh",
    "install.ps1",
    "docs/INSTALL.md",
    "docs/RELEASE-CHECKLIST.md",
    ".secrets.baseline",
}


def _version() -> str:
    text = (ROOT / "kindi" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^VERSION = "([^"]+)"$', text, re.MULTILINE)
    if not match:
        raise SystemExit("kindi/__init__.py has no VERSION")
    return match.group(1)


def _files() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    names = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    return sorted(
        name for name in names
        if name not in EXCLUDED and not name.startswith("dist/") and not name.startswith(".git/")
    )


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _add_file(archive: tarfile.TarFile, source: Path, name: str, mtime: int) -> None:
    info = archive.gettarinfo(str(source), name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = mtime
    info.mode = 0o755 if info.mode & 0o111 else 0o644
    if info.issym():
        archive.addfile(info)
        return
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> None:
    version = _version()
    prefix = "kindi-%s" % version
    archive_path = DIST / ("%s.tar.gz" % prefix)
    sidecar_path = DIST / ("%s.tar.gz.sha256" % prefix)
    head = _head()
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0")) or int(
        subprocess.check_output(["git", "log", "-1", "--format=%ct"], cwd=ROOT, text=True).strip()
    )

    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kindi-release-") as temporary:
        staging = Path(temporary)
        for name in _files():
            source = ROOT / name
            target = staging / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            else:
                target.write_bytes(source.read_bytes())
        commit_file = staging / prefix / "kindi" / "COMMIT"
        commit_file.parent.mkdir(parents=True, exist_ok=True)
        commit_file.write_bytes((head[:12] + "\n").encode("ascii"))

        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                    for path in sorted((staging / prefix).rglob("*")):
                        relative = path.relative_to(staging).as_posix()
                        if path.is_dir():
                            continue
                        _add_file(archive, path, relative, mtime)

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar_path.write_bytes(("%s  %s\n" % (digest, archive_path.name)).encode("ascii"))
    print(archive_path)
    print(sidecar_path)
    print(digest)


if __name__ == "__main__":
    main()
