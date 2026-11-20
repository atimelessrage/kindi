"""Startup checks for a missing or incomplete configured model pack."""
import pytest

from kindi import __main__ as cli


def _cfg(pack, source="test.conf"):
    return {"pack": str(pack), "source": source}


def test_missing_pack_directory_is_a_clear_startup_error(tmp_path):
    pack = tmp_path / "missing-pack"
    with pytest.raises(SystemExit) as exc:
        cli._require_pack(_cfg(pack))
    message = str(exc.value)
    assert "the configured pack directory does not exist" in message
    assert str(pack) in message
    assert "kindi use <model>" in message


def test_empty_pack_directory_reports_the_missing_manifest(tmp_path):
    pack = tmp_path / "empty-pack"
    pack.mkdir()
    with pytest.raises(SystemExit, match="the configured pack is incomplete"):
        cli._require_pack(_cfg(pack))


def test_complete_pack_is_accepted(tmp_path):
    pack = tmp_path / "bonsai-2b"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    assert cli._require_pack(_cfg(pack)) == str(pack)


def test_serve_checks_the_pack_before_setting_up_the_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_require_pack", lambda cfg: calls.append("pack"))
    monkeypatch.setattr(cli, "_setup_gpu_env", lambda cfg: calls.append("gpu"))
    import kindi.server
    monkeypatch.setattr(kindi.server, "serve", lambda cfg, log: calls.append("serve"))
    assert cli.cmd_serve({}, None) == 0
    assert calls == ["pack", "gpu", "serve"]
