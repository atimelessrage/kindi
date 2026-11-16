"""The T4.2 gateway must not build the wrong engine silently.

`kindi.engine.make_engine` is the one place a pack turns into an engine, so every pack either
resolves through its backend adapter or says why it did not. The "Hf" 27B bug was the quiet
version: a placeholder `model` field made the Bonsai adapter miss its own pack and the gateway
fell back to the legacy builder without a word.
"""
import json

import pytest

import kindi.backends as backends_package
from kindi import engine


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def _recording_resolve(monkeypatch):
    """Replace the registry lookup: records (model id, config) and returns an engine-shaped backend."""
    seen = []

    class _Backend(object):
        class capabilities(object):
            name = "recording"

        engine = "the-engine"

    def fake_resolve(model_id, config=None):
        seen.append((model_id, dict(config or {})))
        return _Backend()

    # engine.make_engine imports it as `from .backends import resolve`, i.e. the package attribute
    monkeypatch.setattr(backends_package, "resolve", fake_resolve)
    return seen


def test_the_hf_27b_manifest_resolves_through_the_bonsai_adapter(tmp_path, monkeypatch):
    """The 27B pack's GGUF name is the placeholder "Hf"; the size in `source_file` still names it."""
    seen = _recording_resolve(monkeypatch)
    logged = []
    pack = _pack(tmp_path, {"model": "Hf", "source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                            "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "the-engine"
    assert [model_id for model_id, _ in seen] == ["bonsai2-27b"]
    assert seen[0][1]["pack"] == pack
    # the log names the adapter and the model id, so a mis-resolved pack is visible in any run
    assert any("bonsai2-27b" in line and "recording" in line for line in logged)


def test_a_dense_pack_no_adapter_claims_says_so_and_uses_the_legacy_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy", lambda *a, **k: "legacy-engine")
    logged = []
    pack = _pack(tmp_path, {"model": "Some-Other-Bonsai", "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "legacy-engine"
    assert any("no adapter claims" in line and "legacy dense builder" in line for line in logged)


def test_a_pack_whose_family_has_no_adapter_is_refused_not_built_dense(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the gateway built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_cascade", "model_path": "x.gguf"})
    with pytest.raises(SystemExit, match="kindi_cascade"):
        engine.make_engine(pack)


def test_the_gemma4_and_deepseek_families_route_through_their_adapters(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    for manifest, model_id in (({"backend": "kindi_gemma4", "model_path": "g.gguf"}, "gemma4-e4b"),
                               ({"backend": "kindi_dsv41", "model_path": "d.gguf"},
                                "deepseek-v41-flash")):
        del seen[:]
        assert engine.make_engine(_pack(tmp_path, manifest), log=lambda line: None) == "the-engine"
        assert [mid for mid, _ in seen] == [model_id]


def test_a_contradictory_manifest_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the gateway built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_lightning", "model": "Ternary-Bonsai-2-4B"})
    with pytest.raises(SystemExit, match="more than one backend"):
        engine.make_engine(pack)


def test_a_manifest_with_neither_family_nor_dense_weights_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the gateway built the dense engine"))
    pack = _pack(tmp_path, {"model": "Hf"})
    with pytest.raises(SystemExit, match="not a pack this version can build"):
        engine.make_engine(pack)


def test_the_legacy_valve_refuses_a_family_it_cannot_build(tmp_path, monkeypatch):
    """`KINDI_LEGACY_ENGINE_PATH=1` means the pre-port dense builder; a Gemma 4 or DeepSeek pack
    has no legacy builder there, so it must not quietly get a dense engine."""
    monkeypatch.setenv("KINDI_LEGACY_ENGINE_PATH", "1")
    monkeypatch.setattr(engine, "make_engine_legacy",
                        lambda *a, **k: pytest.fail("the valve built the dense engine"))
    pack = _pack(tmp_path, {"backend": "kindi_gemma4", "model_path": "g.gguf"})
    with pytest.raises(SystemExit, match="KINDI_LEGACY_ENGINE_PATH=1"):
        engine.make_engine(pack)


def test_the_legacy_valve_names_itself_in_the_log(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_LEGACY_ENGINE_PATH", "1")
    monkeypatch.setattr(engine, "make_engine_legacy", lambda *a, **k: "legacy-engine")
    logged = []
    pack = _pack(tmp_path, {"model": "Hf", "bcodes_bytes": 1024})
    assert engine.make_engine(pack, log=logged.append) == "legacy-engine"
    assert any("KINDI_LEGACY_ENGINE_PATH=1" in line for line in logged)


def test_local_packs_resolve_to_the_adapter_they_declare():
    """Harness over the packs on this host: each one resolves to the adapter its manifest names.

    Skipped where `~/.kindi/packs` holds none (CI images, a fresh container).
    """
    import os

    root = os.path.expanduser(os.path.join("~", ".kindi", "packs"))
    if not os.path.isdir(root):
        pytest.skip("no local packs")
    packs = [os.path.join(root, d) for d in sorted(os.listdir(root))
             if os.path.isfile(os.path.join(root, d, "kindi_pack.json"))]
    if not packs:
        pytest.skip("no local packs")
    expected = {"kindi_lightning": "nemotron-lightning-30b",
                "kindi_nanbeige": "nanbeige-33b"}
    for pack in packs:
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            man = json.load(f)
        model_id = engine.adapter_model_id(man)
        family = man.get("backend")
        if family in expected:
            assert model_id == expected[family], pack
        elif family in engine.DENSE_BACKENDS:
            # a dense pack on the host that no adapter claims is exactly the Hf class of bug
            assert model_id is not None, (
                "%s is a dense pack that no adapter claims (model %r): its manifest is missing the "
                "model name/source_file the Bonsai adapter reads" % (pack, man.get("model")))
        else:
            assert model_id is None, (pack, family)


def test_tandem_on_a_moe_pack_is_explained_and_left_off(tmp_path, monkeypatch):
    """A `--tandem` run on a MoE pack must not crash in, or silently build, the dense layer split:
    the family's own split is its expert tiering, and the log says so."""
    seen = _recording_resolve(monkeypatch)
    monkeypatch.setattr(engine, "adapter_model_id", lambda man: "lightning-30b")
    logged = []
    pack = _pack(tmp_path, {"backend": "kindi_lightning", "model": "lightning-30b"})
    assert engine.make_engine(pack, tandem="cpu-first", log=logged.append) == "the-engine"
    assert seen[0][1]["tandem"] == "off"
    assert any("mixture-of-experts" in line and "expert-tier" in line and "kindi plan" in line
               for line in logged)


def test_tandem_off_is_not_explained_on_a_moe_pack(tmp_path, monkeypatch):
    seen = _recording_resolve(monkeypatch)
    monkeypatch.setattr(engine, "adapter_model_id", lambda man: "lightning-30b")
    logged = []
    pack = _pack(tmp_path, {"backend": "kindi_lightning", "model": "lightning-30b"})
    engine.make_engine(pack, log=logged.append)
    assert seen[0][1]["tandem"] == "off"
    assert not any("mixture-of-experts" in line for line in logged)
