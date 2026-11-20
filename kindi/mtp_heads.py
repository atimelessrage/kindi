# SPDX-License-Identifier: AGPL-3.0-only
"""The MTP head registry: `kindi/mtp_heads.json`, its digest check and the `kindi info` line.

The head is a downloaded weight file, and the registry next to this module is what makes it
checkable: `sha256` and `bytes` per file name. `verify()` is called before `MTPDraft` reads the
file, so a truncated, patched or wrong download never reaches the card. A head that is not in the
registry is refused too, unless `KINDI_MTP_HEAD_ALLOW_UNREGISTERED=1` names a local-development
escape hatch (REQ-0024 A3).

Kept free of CuPy and NumPy so `kindi info` and the CPU test suite can use it without a GPU.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "mtp_heads.json")
ALLOW = "KINDI_MTP_HEAD_ALLOW_UNREGISTERED"


class HeadError(ValueError):
    """A head file the registry does not accept. The message names the file, the expected and the
    found digest, so the log line alone is enough to re-download or delete the right file."""


def registry(path=None):
    """The shipped registry: file name -> {"sha256", "bytes", "model", ...}."""
    with open(path or REGISTRY, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, table=None, allow_unregistered=None):
    """Check a head file against the registry before its weights are loaded.

    Returns {"file", "sha256", "bytes", "registered"} and raises `HeadError` when the size or the
    sha256 disagrees, or when the file is not registered and the escape hatch is off.
    `table` and `allow_unregistered` are for tests; both default to the shipped registry and env.
    """
    if allow_unregistered is None:
        # the name is spelled out here, not through ALLOW, because the configuration-reference scanner
        # reads literal names and would otherwise leave this variable out of the documented table
        allow_unregistered = os.environ.get("KINDI_MTP_HEAD_ALLOW_UNREGISTERED", "0") == "1"
    name = os.path.basename(path)
    size = os.path.getsize(path)
    found = sha256_file(path)
    row = (registry() if table is None else table).get(name)
    if row is None:
        if allow_unregistered:
            return {"file": name, "sha256": found, "bytes": size, "registered": False}
        raise HeadError("the MTP head %s is not registered in kindi/mtp_heads.json (found sha256 %s, %d bytes); "
                        "download a registered head, or set %s=1 for a head you made yourself"
                        % (path, found[:12], size, ALLOW))
    want_sha, want_bytes = str(row["sha256"]), int(row["bytes"])
    if size != want_bytes or found != want_sha:
        raise HeadError("the MTP head %s does not match kindi/mtp_heads.json: expected sha256 %s, %d bytes; "
                        "found sha256 %s, %d bytes (a truncated, patched or wrong download)"
                        % (path, want_sha[:12], want_bytes, found[:12], size))
    return {"file": name, "sha256": found, "bytes": size, "registered": True}


def report(head=None, spec=None):
    """The `kindi info` line: the requested mode and the head's registry verdict.

    `kindi info` does not build an engine, so it cannot know the mode the engine will settle on;
    it reports the check that happens before the load and points at the engine's own line for the
    effective mode rather than repeating the requested one as if it were running (REQ-0024).
    """
    spec = (os.environ.get("KINDI_SPEC", "0") or "0") if spec is None else spec
    head = os.environ.get("KINDI_MTP_HEAD") if head is None else head
    if spec in ("0", "", "off", "ngram"):
        return "off: KINDI_SPEC=%s, no model drafter" % spec
    if not head:
        return ("%s requested, no head: KINDI_MTP_HEAD is not set, so nothing will draft; "
                "`kindi serve` logs the effective mode" % spec)
    try:
        h = verify(head)
    except (HeadError, OSError) as e:
        return "%s requested, head REFUSED - serving will not draft: %s" % (spec, e)
    return ("%s requested, head %s verified against mtp_heads.json (sha256 %s, %d bytes)%s; "
            "`kindi serve` logs the mode it actually runs"
            % (spec, h["file"], h["sha256"][:12], h["bytes"],
               "" if h["registered"] else " as an unregistered local head"))
