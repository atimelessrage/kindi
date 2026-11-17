# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi io`: the measured memory-tier rates, and the honesty of each number."""
import json

from kindi import io_probe


def test_ram_copy_measures_a_real_rate():
    r = io_probe.ram_copy(mib=4, repeats=2)
    assert r["available"] is True
    assert r["mib"] == 4
    assert r["mib_s"] > 0 and r["seconds"] > 0
    assert "memcpy" in r["method"]


def test_disk_read_is_unbuffered_or_labelled_and_leaves_no_file(tmp_path):
    r = io_probe.disk_read(mib=8, directory=str(tmp_path))
    assert r["available"] is True
    assert r["mib_s"] > 0
    assert isinstance(r["unbuffered"], bool)
    assert ("buffered" in r["method"]) == (not r["unbuffered"])
    assert list(tmp_path.iterdir()) == []


def test_buffered_fallback_is_labelled(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise OSError("no direct I/O here")

    monkeypatch.setattr(io_probe, "_windows_nobuffer_read", refuse)
    monkeypatch.setattr(io_probe, "_posix_direct_read", refuse)
    r = io_probe.disk_read(mib=4, directory=str(tmp_path))
    assert r["unbuffered"] is False
    assert "buffered" in r["method"]
    assert r["mib_s"] > 0


def test_pcie_is_unavailable_without_cuda(monkeypatch):
    monkeypatch.setattr(io_probe, "_cuda", lambda: None)
    r = io_probe.pcie_copy()
    assert r["available"] is False
    assert "CuPy" in r["reason"]
    assert io_probe.probe(mib=4, skip=("ram", "disk"))["pcie"]["available"] is False


def test_probe_skips_what_is_asked():
    report = io_probe.probe(mib=4, skip=("ram", "disk", "pcie"))
    assert all(report[k]["available"] is False and report[k]["reason"] == "skipped"
               for k in ("ram", "disk", "pcie"))


def test_render_names_the_method_and_the_boundary():
    report = io_probe.probe(mib=4, skip=("disk", "pcie"))
    text = "\n".join(io_probe.render(report))
    assert "RAM copy:" in text and "MiB/s" in text
    assert "no other engine's rate is ours" in text


def test_cli_json_is_machine_readable(capsys):
    assert io_probe.main(["--mib", "4", "--skip", "disk,pcie", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"ram", "disk", "pcie"}
    assert report["ram"]["available"] is True
    assert report["disk"]["reason"] == "skipped"


def test_the_cli_command_is_wired(capsys):
    from kindi import __main__ as cli
    assert cli.main(["io", "--mib", "4", "--skip", "disk,pcie", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pcie"]["reason"] == "skipped"
