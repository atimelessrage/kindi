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
    # the release is part of the key: 2.5 must not be verified against Bonsai 2 weights
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"})) is None
    assert verify.reference_for(_pack(tmp_path, {"model": "Ternary-Bonsai-2.5-27B"})) is None


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


class _BenchEngine(object):
    """The small engine surface bench needs; it never loads a model or a kernel."""

    def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
        self.ctx = ctx
        self.hsl = None
        self.shortlist = 0

    def reset(self):
        pass

    def feed(self, tok, pos):
        pass

    def step(self, tok, pos):
        pass


def test_bench_passes_configured_kv_cache_to_make_engine(tmp_path, monkeypatch):
    """`kindi bench` must build the same K/V-cache variant as serve, not drop the config key."""
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    pack = _pack(tmp_path, {})
    seen = {}

    def fake_make_engine(*args, **kwargs):
        seen.update(kwargs)
        return _BenchEngine(*args, **kwargs)

    monkeypatch.setattr("kindi.engine.make_engine", fake_make_engine)
    assert bench.run({"pack": pack, "ctx": 1024, "fast_clusters": 256,
                      "split_at": "none", "kv_cache": "q8_0"},
                     log=lambda *a: None) == 0
    assert seen["kv_cache"] == "q8_0"


def test_verify_passes_configured_kv_cache_to_make_engine(tmp_path, monkeypatch):
    """Generic `kindi verify` must forward the K/V-cache choice into the same gateway."""
    pack = _pack(tmp_path, {"model": "bonsai2-2b"})
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "prompts": [{"prompt": "p", "ids": [1, 2], "expected": [3, 4], "gap": []}]}),
        encoding="utf-8")
    monkeypatch.setattr(verify, "reference_for", lambda _: str(reference))
    monkeypatch.delenv("KINDI_SPEC", raising=False)
    seen = {}

    class Engine(object):
        def generate(self, ids, n):
            return [3, 4]

    def fake_make_engine(*args, **kwargs):
        seen.update(kwargs)
        return Engine()

    monkeypatch.setattr("kindi.engine.make_engine", fake_make_engine)
    assert verify.run({"pack": pack, "kv_cache": "q8_0"}, log=lambda *a: None) == 0
    assert seen["kv_cache"] == "q8_0"
