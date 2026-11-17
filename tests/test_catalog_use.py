# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi use` must verify an existing catalog GGUF before converting it."""

import hashlib
from pathlib import Path

import pytest

from kindi import catalog


def _entry(data=b"good"):
    return {
        "file": "Model.gguf",
        "repo": "owner/repo",
        "sha": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "one": 1,
        "two": 0,
        "what": "test model",
    }


def _cfg(tmp_path):
    return {"pack": str(tmp_path / "x" / "pack")}


def _gguf(tmp_path, data):
    path = tmp_path / "x" / "models" / "Model.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, tmp_path / "x" / "packs" / "lightning-30b"


def _manifest(pack):
    pack = pack if hasattr(pack, "mkdir") else Path(pack)
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "kindi_pack.json").write_text("{}")


def _unused_download(*args, **kwargs):  # pragma: no cover - replaced below
    raise AssertionError("download was expected but not called")


def test_a_verified_file_is_not_hashed_twice(tmp_path, monkeypatch):
    """Re-reading a 9.5-13 GB GGUF on every `kindi use` cost 40-90 s; a size+mtime
    marker skips the second read, and the first run still writes it."""
    entry = _entry(b"good")
    path = tmp_path / "Model.gguf"
    path.write_bytes(b"good")
    calls = []
    real = catalog._sha256_file
    monkeypatch.setattr(catalog, "_sha256_file", lambda p: calls.append(p) or real(p))
    assert catalog._matches_catalog(str(path), entry) is True
    assert catalog._matches_catalog(str(path), entry) is True
    assert len(calls) == 1


def test_a_changed_file_is_hashed_and_rejected_again(tmp_path, monkeypatch):
    entry = _entry(b"good")
    path = tmp_path / "Model.gguf"
    path.write_bytes(b"good")
    assert catalog._matches_catalog(str(path), entry) is True
    path.write_bytes(b"evil")             # same size, different bytes and mtime
    assert catalog._matches_catalog(str(path), entry) is False


def test_select_survives_a_symlink_it_cannot_create(tmp_path, monkeypatch):
    """Windows without Developer Mode/admin raises WinError 1314 from os.symlink.
    The config file is already updated at that point, so the command must not fail."""
    conf = tmp_path / "kindi.conf"
    conf.write_text("")
    monkeypatch.setenv("KINDI_PACK", str(tmp_path / "pack-link"))

    def boom(*args, **kwargs):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(catalog.os, "symlink", boom)
    logged = []
    catalog._select({"source": str(conf)}, str(tmp_path / "packs" / "m"), log=logged.append)
    assert any("cannot update the KINDI_PACK symlink" in line for line in logged)
    assert "pack = " in conf.read_text()
    # the failed link leaves nothing behind next to it
    assert not (tmp_path / "pack-link.new").exists()


def test_select_survives_a_platform_without_symlinks(tmp_path, monkeypatch):
    """os.symlink raises NotImplementedError where the platform has no symlinks at all; the
    selection must still succeed on the config file, the same as the WinError 1314 case."""
    conf = tmp_path / "kindi.conf"
    conf.write_text("")
    monkeypatch.setenv("KINDI_PACK", str(tmp_path / "pack-link"))

    def boom(*args, **kwargs):
        raise NotImplementedError("symlinks are not supported on this platform")

    monkeypatch.setattr(catalog.os, "symlink", boom)
    logged = []
    catalog._select({"source": str(conf)}, str(tmp_path / "packs" / "m"), log=logged.append)
    assert any("cannot update the KINDI_PACK symlink" in line for line in logged)
    assert "pack = " in conf.read_text()


def test_existing_bad_catalog_file_is_redownloaded_and_checked(tmp_path, monkeypatch):
    good = b"good"
    monkeypatch.setitem(catalog.MODELS, "lightning-30b", _entry(good))
    gguf, pack = _gguf(tmp_path, b"bad")
    downloaded = []

    def fake_download(url, path, size, sha, log):
        downloaded.append(path)
        with open(path, "wb") as f:
            f.write(good)

    def fake_convert(path, out, log=None):
        _manifest(out)
        return {"backend": "kindi_lightning"}

    monkeypatch.setattr(catalog, "_download", fake_download)
    monkeypatch.setattr("kindi.convert.convert", fake_convert)
    monkeypatch.setattr(catalog, "_select", lambda cfg, pack, log: None)
    assert catalog.use(_cfg(tmp_path), "lightning-30b", log=lambda line: None) == 0
    assert downloaded == [str(gguf)]
    assert gguf.read_bytes() == good
    assert (pack / "kindi_pack.json").is_file()


def test_existing_good_catalog_file_is_trusted_without_download(tmp_path, monkeypatch):
    good = b"good"
    monkeypatch.setitem(catalog.MODELS, "lightning-30b", _entry(good))
    gguf, pack = _gguf(tmp_path, good)
    _manifest(pack)
    monkeypatch.setattr(catalog, "_download", _unused_download)
    monkeypatch.setattr("kindi.convert.convert", lambda *a, **k: pytest.fail("convert unexpected"))
    monkeypatch.setattr(catalog, "_select", lambda cfg, pack, log: None)
    assert catalog.use(_cfg(tmp_path), "lightning-30b", log=lambda line: None) == 0
    assert gguf.read_bytes() == good


def test_catalog_matches_empty_and_missing_files():
    entry = _entry(b"good")
    assert catalog._matches_catalog("does-not-exist", entry) is False
    assert catalog._matches_catalog("", entry) is False
