# SPDX-License-Identifier: AGPL-3.0-only
"""The MTP head's registry check, before its weights are loaded (REQ-0024 A3).

`kindi/mtp_heads.json` records `sha256` and `bytes` for each downloadable head, and the loader used
to open the file without looking at either: a truncated or patched 60 MB download would load and
draft. These tests pin the three outcomes the loader must tell apart - a registered file that
matches (loaded), a registered file that does not (refused), and a file nobody registered
(unregistered, refused unless `KINDI_MTP_HEAD_ALLOW_UNREGISTERED=1`).

They also pin the order: the check runs before `np.load`, so a refused file never becomes weights.
No GPU and no 60 MB head: the files here are bytes with a registry row built beside them.
"""
import hashlib
import os
import sys
import types

import pytest

from kindi import mtp_heads


def _row(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _stub_cupy_if_needed():
    """`kindi/mtp.py` imports CuPy at module level; CI has none."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        cp = types.ModuleType("cupy")
        cuda = types.ModuleType("cupy.cuda")
        cuda.runtime = types.ModuleType("cupy.cuda.runtime")
        cuda.runtime.is_hip = False
        cuda.compiler = types.ModuleType("cupy.cuda.compiler")
        cuda.function = types.ModuleType("cupy.cuda.function")
        cp.cuda = cuda
        cp.ndarray = type("ndarray", (), {})
        sys.modules.update({"cupy": cp, "cupy.cuda": cuda, "cupy.cuda.runtime": cuda.runtime,
                            "cupy.cuda.compiler": cuda.compiler, "cupy.cuda.function": cuda.function})


def test_a_registered_head_whose_digest_matches_is_accepted(tmp_path):
    """The loaded case: the file's size and sha256 are what the registry records."""
    data = b"kindi mtp head stand-in" * 32
    head = tmp_path / "kindi-mtp-2b-v1.npz"
    head.write_bytes(data)
    got = mtp_heads.verify(str(head), table={"kindi-mtp-2b-v1.npz": _row(data)}, allow_unregistered=False)
    assert got == {"file": "kindi-mtp-2b-v1.npz", "sha256": _row(data)["sha256"],
                   "bytes": len(data), "registered": True}


def test_a_truncated_head_is_refused_naming_expected_and_found(tmp_path):
    """The refused case: same name, right hash on record, wrong bytes on disk."""
    data = b"kindi mtp head stand-in" * 32
    head = tmp_path / "kindi-mtp-2b-v1.npz"
    head.write_bytes(data[:-7])
    with pytest.raises(mtp_heads.HeadError) as e:
        mtp_heads.verify(str(head), table={"kindi-mtp-2b-v1.npz": _row(data)}, allow_unregistered=False)
    msg = str(e.value)
    assert str(head) in msg                                   # names the file
    assert _row(data)["sha256"][:12] in msg                   # the expected digest
    assert _row(data[:-7])["sha256"][:12] in msg              # the found digest
    assert str(len(data)) in msg and str(len(data) - 7) in msg


def test_an_unregistered_head_is_refused_unless_the_escape_hatch_is_set(tmp_path, monkeypatch):
    """The unregistered case, both sides of KINDI_MTP_HEAD_ALLOW_UNREGISTERED."""
    head = tmp_path / "home-made.npz"
    head.write_bytes(b"not from the registry")
    monkeypatch.delenv(mtp_heads.ALLOW, raising=False)
    with pytest.raises(mtp_heads.HeadError) as e:
        mtp_heads.verify(str(head), table={})
    assert mtp_heads.ALLOW in str(e.value)                    # the message names the escape hatch
    assert hashlib.sha256(head.read_bytes()).hexdigest()[:12] in str(e.value)

    monkeypatch.setenv(mtp_heads.ALLOW, "1")
    assert mtp_heads.verify(str(head), table={})["registered"] is False


def test_the_loader_checks_the_registry_before_it_reads_any_weight(tmp_path, monkeypatch):
    """`MTPDraft` refuses an unregistered file before `np.load`, which is what keeps a wrong download
    off the card; with the escape hatch set it proceeds to the loader (the sentinel), proving the
    gate is before the read and not after it."""
    _stub_cupy_if_needed()
    from kindi import mtp

    head = tmp_path / "home-made.npz"                        # not in the registry
    head.write_bytes(b"x" * 128)
    monkeypatch.delenv(mtp_heads.ALLOW, raising=False)

    def weights_read(path, allow_pickle=None):
        # the loader must ask for the no-pickle form explicitly (a tampered head cannot run code)
        assert allow_pickle is False, "the head loader must pass allow_pickle=False"
        raise AssertionError("np.load ran before the registry check")

    monkeypatch.setattr(mtp.np, "load", weights_read)
    with pytest.raises(mtp_heads.HeadError):
        mtp.MTPDraft(types.SimpleNamespace(cards=[None]), str(head))

    monkeypatch.setenv(mtp_heads.ALLOW, "1")
    with pytest.raises(AssertionError, match="np.load ran"):
        mtp.MTPDraft(types.SimpleNamespace(cards=[None]), str(head))   # past the gate, at the weights


def test_the_startup_line_separates_the_request_from_the_head(tmp_path, monkeypatch):
    """`kindi info` cannot build an engine, so it reports the check it can make and says the mode is
    the engine's to report - never the requested mode as if it were running (REQ-0024)."""
    data = b"kindi mtp head stand-in" * 32
    head = tmp_path / "kindi-mtp-2b-v1.npz"

    assert mtp_heads.report(head=None, spec="0").startswith("off")

    line = mtp_heads.report(head=str(head), spec="mtp")       # missing file: refused
    assert "REFUSED" in line and "does not match" not in line  # OSError names the path

    head.write_bytes(data[:-4])
    monkeypatch.setattr(mtp_heads, "registry", lambda path=None: {"kindi-mtp-2b-v1.npz": _row(data)})
    refused = mtp_heads.report(head=str(head), spec="mtp")
    assert "REFUSED" in refused and "serving will not draft" in refused
    assert _row(data)["sha256"][:12] in refused

    head.write_bytes(data)
    loaded = mtp_heads.report(head=str(head), spec="mtp")
    assert "verified" in loaded and _row(data)["sha256"][:12] in loaded
    assert "logs the mode it actually runs" in loaded


def test_the_shipped_registry_row_is_well_formed():
    """The documented 2B head: the row the loader will compare against, with a full digest."""
    row = mtp_heads.registry()["kindi-mtp-2b-v1.npz"]
    assert row["sha256"] == "1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520"
    assert row["bytes"] == 60960552
    assert len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"])
