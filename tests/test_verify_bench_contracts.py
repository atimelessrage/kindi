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


def test_bench_routes_nanbeige_to_its_own_bench_when_a_gpu_is_present(tmp_path, monkeypatch):
    """The CPU guard must not hide the real route: with a GPU, bench calls nanbeige_cli.bench."""
    pack = _pack(tmp_path, {"backend": "kindi_nanbeige"})
    seen = []
    monkeypatch.setattr(bench, "_cupy", lambda: object())
    monkeypatch.setattr("kindi.nanbeige_cli.bench",
                        lambda cfg, man, log=print: seen.append((cfg, man)) or 0)
    assert bench.run({"pack": pack, "ctx": 1024}, log=lambda *a: None) == 0
    assert seen and seen[0][1]["backend"] == "kindi_nanbeige"


def test_verify_routes_nanbeige_to_its_own_gate(tmp_path, monkeypatch):
    pack = _pack(tmp_path, {"backend": "kindi_nanbeige"})
    seen = []
    monkeypatch.setattr("kindi.nanbeige_cli.verify",
                        lambda cfg, man, log=print: seen.append((cfg, man)) or 0)
    assert verify.run({"pack": pack}, log=lambda *a: None) == 0
    assert seen and seen[0][1]["backend"] == "kindi_nanbeige"


def test_bench_names_the_families_it_cannot_bench(tmp_path, monkeypatch):
    """`kindi bench` must not fall through to the dense loop for a Gemma 4 or DeepSeek pack."""
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    for backend, word in (("kindi_gemma4", "Gemma 4"), ("kindi_dsv41", "DeepSeek")):
        logged = []
        pack = _pack(tmp_path, {"backend": backend})
        assert bench.run({"pack": pack, "ctx": 1024}, log=logged.append) == 2
        assert any(word in line for line in logged), logged


def test_verify_refuses_the_deepseek_reference_matrix(tmp_path, capsys):
    """dsv41 has its own release gate (dsv41.verify); verify must say so, not run the llama.cpp matrix."""
    pack = _pack(tmp_path, {"backend": "kindi_dsv41"})
    logged = []
    assert verify.run({"pack": pack, "ctx": 1024}, log=logged.append) == 2
    assert any("dsv41.verify" in line for line in logged), logged
