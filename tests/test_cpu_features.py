"""The CPU feature report and the preflight before building kcpu.c."""
import pytest

from kindi import cpu

INFO = """processor\t: 0
model name\t: Example i5
flags\t\t: sse avx avx2 fma f16c
"""


def _report(monkeypatch, text=INFO, physical=6, logical=12):
    monkeypatch.setattr(cpu, "_read", lambda path: text if path == "/proc/cpuinfo" else None)
    monkeypatch.setattr(cpu, "_physical_cores", lambda: physical)
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: logical)
    return cpu.feature_report()


def test_feature_report_reads_the_linux_cpu(monkeypatch):
    report = _report(monkeypatch)
    assert report == {
        "model": "Example i5",
        "physical": 6,
        "logical": 12,
        "features": {"avx2": True, "fma": True, "f16c": True},
    }


def test_missing_required_features_are_named(monkeypatch):
    report = _report(monkeypatch, INFO.replace("avx2 fma f16c", "avx"))
    assert cpu.missing_features(report) == ("AVX2", "FMA3", "F16C")


def test_unknown_flags_do_not_block_the_compile_time_guard(monkeypatch):
    report = _report(monkeypatch, text=None)
    assert report["features"] == {"avx2": None, "fma": None, "f16c": None}
    assert cpu.missing_features(report) == ()


def test_check_features_explains_the_missing_isa(monkeypatch):
    report = _report(monkeypatch, INFO.replace("avx2 fma f16c", "avx"))
    monkeypatch.setattr(cpu, "feature_report", lambda: report)
    with pytest.raises(SystemExit, match="needs AVX2, FMA3 and F16C; this CPU lacks AVX2, FMA3, F16C"):
        cpu.check_features()
