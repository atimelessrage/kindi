"""Requirement-based tests for `kindi verify` reference routing and the
`kindi bench` backend guards: what the user sees, not implementation internals.
"""
import json
import os

from kindi import bench, verify


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def test_reference_for_routes_by_model_size(tmp_path):
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-2B-PTQ1_0.gguf"})).endswith("reference.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-4B-PTQ1_0.gguf"})).endswith("reference-4b.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-9B-PTQ1_0.gguf"})).endswith("reference-9b.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf"})).endswith("reference-27b.json")


def test_reference_for_defaults_to_2b_and_missing_is_none(tmp_path):
    assert verify.reference_for(_pack(tmp_path, {})).endswith("reference.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-99B-x.gguf"})) is None


def test_bench_cupy_is_none_on_cpu_device(monkeypatch):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    assert bench._cupy() is None


def test_bench_refuses_nanbeige_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    pack = _pack(tmp_path, {"backend": "kindi_nanbeige"})
    logged = []
    assert bench.run({"pack": pack, "ctx": 1024}, log=logged.append) == 2
    assert any("Nanbeige" in line for line in logged)
