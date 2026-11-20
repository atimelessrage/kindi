#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create and verify a git bundle backup of every ref.

The repository and its benchmark evidence live on one volume, and pushing is not part of
this checkout's workflow. This tool writes one timestamped bundle with `--all`, verifies
it with `git bundle verify`, and records its sha256 beside it. Copy the result to a second
physical medium or encrypted cloud folder; the tool warns when the output is on the same
volume as the repository.

    python tools/backup_bundle.py --out /media/backup/kindi

A bundle contains every ref, including archival branches and the pre-redaction objects
reachable from them. Treat it as sensitive material, not as a public artifact.
"""
import argparse
import datetime
import hashlib
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BackupError(Exception):
    pass


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo] + list(args), cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=3600, check=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_path(out_dir, timestamp=None):
    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(out_dir, "kindi-%s.bundle" % stamp)


def create(repo, out_dir, timestamp=None, dry_run=False):
    repo = os.path.abspath(os.path.expanduser(repo))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    path = bundle_path(out_dir, timestamp)
    if os.path.exists(path):
        raise BackupError("refusing to overwrite %s" % path)
    command = ["git", "-C", repo, "bundle", "create", path, "--all"]
    print(" ".join(shlex.quote(part) for part in command))
    if dry_run:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    created = _git(repo, "bundle", "create", path, "--all")
    if created.returncode != 0:
        raise BackupError("git bundle create failed: %s" % (created.stdout or "").strip())
    verified = _git(repo, "bundle", "verify", path)
    if verified.returncode != 0:
        raise BackupError("git bundle verify failed: %s" % (verified.stdout or "").strip())
    digest = sha256(path)
    with open(path + ".sha256", "w", encoding="utf-8") as handle:
        handle.write("%s  %s\n" % (digest, os.path.basename(path)))
    size = os.path.getsize(path) / (1024.0 * 1024.0)
    print("bundle: %s (%.1f MiB)" % (path, size))
    print("sha256: %s" % digest)
    if os.path.splitdrive(repo)[0].lower() == os.path.splitdrive(out_dir)[0].lower():
        print("warning: the bundle is on the same volume as the repository; copy it offsite",
              file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create and verify a git bundle backup of every ref.")
    parser.add_argument("--repo", default=ROOT, help="repository to bundle (default: this checkout)")
    parser.add_argument("--out", required=True, help="directory for the bundle and its .sha256")
    parser.add_argument("--timestamp", default=None,
                        help="deterministic name suffix, YYYYMMDD-HHMMSS (default: now)")
    parser.add_argument("--dry-run", action="store_true", help="print the command, write nothing")
    args = parser.parse_args(argv)
    try:
        return create(args.repo, args.out, args.timestamp, args.dry_run)
    except (BackupError, OSError) as exc:
        print("backup_bundle: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
