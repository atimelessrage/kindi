# SPDX-License-Identifier: AGPL-3.0-only
"""Registry tests for the backend plugin interface (T4.1)."""

import sys

import pytest

from kindi.backends import Backend, BackendCapabilities, FakeBackend
from kindi.backends import registry


class ToyBackend(Backend):
    capabilities = BackendCapabilities(
        name="toy",
        model_ids=("toy-1b",),
        architectures=("toyarch",),
        devices=("cpu",),
        max_context=1024,
        deterministic_greedy=True,
    )

    def load(self, config):
        self.config = dict(config or {})
        return self

    def close(self):
        pass

    def info(self):
        return {"engine": "toy", "loaded": True}

    def tokenize(self, text, add_special_tokens=True):
        return [ord(c) + 1 for c in (text or "")]

    def detokenize(self, token_ids):
        return "".join(chr(int(t) - 1) for t in token_ids if int(t) >= 1)

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        return "\n".join("%s: %s" % (m.get("role", "user"), m.get("content", ""))
                         for m in messages or [])

    def generate(self, prompt_ids, params, session=None, cancel=None):
        yield "done"

    def cache_stats(self):
        return {"cache_kind": "none"}


@pytest.fixture(autouse=True)
def isolate_registry():
    saved = dict(registry._REGISTRY)
    order = list(registry._ORDER)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    registry._ORDER[:] = order


def test_register_and_list():
    registry.register_backend(ToyBackend)
    rows = registry.list_backends()
    names = [row["name"] for row in rows]
    assert "toy" in names
    assert "fake" in names
    toy = [row for row in rows if row["name"] == "toy"][0]
    assert toy["model_ids"] == ["toy-1b"]
    assert toy["architectures"] == ["toyarch"]
    assert toy["devices"] == ["cpu"]
    assert toy["factory"] == "ToyBackend"


def test_duplicate_registration_is_refused():
    registry.register_backend(ToyBackend)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend(ToyBackend)


def test_resolve_by_model_id():
    registry.register_backend(ToyBackend)
    backend = registry.resolve("toy-1b")
    assert isinstance(backend, ToyBackend)


def test_resolve_by_architecture_via_metadata(monkeypatch):
    registry.register_backend(ToyBackend)
    monkeypatch.setattr(registry, "gguf_architecture", lambda path: "toyarch")
    import os
    real_file = os.path.join(os.path.dirname(__file__), "test_registry.py")
    backend = registry.resolve(real_file)
    assert isinstance(backend, ToyBackend)


def test_resolve_unknown_names_what_is_registered():
    with pytest.raises(registry.BackendNotFound) as excinfo:
        registry.resolve("no-such-model")
    message = str(excinfo.value)
    assert "no-such-model" in message
    assert "fake" in message
    assert "bonsai" in message


def test_resolve_respects_device_filter():
    registry.register_backend(ToyBackend)
    with pytest.raises(registry.BackendNotFound):
        registry.resolve("toy-1b", device="cuda")


def test_external_module_loading_registers_on_import(tmp_path, monkeypatch):
    module = tmp_path / "toy_ext_backend.py"
    module.write_text(
        "from kindi.backends.base import Backend, BackendCapabilities\n"
        "from kindi.backends import registry\n"
        "class ExtBackend(Backend):\n"
        "    capabilities = BackendCapabilities(name='ext', model_ids=('ext-1b',), devices=('cpu',))\n"
        "    def load(self, config):\n"
        "        return self\n"
        "    def close(self): pass\n"
        "    def info(self): return {'engine': 'ext'}\n"
        "    def tokenize(self, text, add_special_tokens=True): return []\n"
        "    def detokenize(self, token_ids): return ''\n"
        "    def apply_chat_template(self, messages, tools=None, enable_thinking=False): return ''\n"
        "    def generate(self, prompt_ids, params, session=None, cancel=None): yield 'done'\n"
        "    def cache_stats(self): return {}\n"
        "registry.register_backend(ExtBackend)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("KINDI_BACKEND_MODULES", "toy_ext_backend")
    loaded = registry.load_external_backends()
    assert loaded == ["toy_ext_backend"]
    assert "ext" in [row["name"] for row in registry.list_backends()]


def test_external_module_failure_is_named(tmp_path, monkeypatch):
    (tmp_path / "broken_ext_backend.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(registry.BackendUnavailable, match="broken_ext_backend"):
        registry.load_external_backends("broken_ext_backend")


def test_fake_backend_is_resolvable():
    backend = registry.resolve("fake")
    assert isinstance(backend, FakeBackend)


def test_bonsai_resolves_by_model_id_unloaded():
    from kindi.backends.bonsai import BonsaiBackend

    for model_id in ("bonsai2-2b", "bonsai2-4b", "bonsai2-9b", "bonsai2-27b"):
        backend = registry.resolve(model_id)
        assert isinstance(backend, BonsaiBackend)
        assert backend.loaded is False


def test_nanbeige_resolves_by_model_id_unloaded():
    from kindi.backends.nanbeige import NanbeigeBackend

    backend = registry.resolve("nanbeige-33b")
    assert isinstance(backend, NanbeigeBackend)
    assert backend.loaded is False


def test_nanbeige_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("nanbeige-33b", config={"pack": "no-such-pack-dir"})


def test_nanbeige_model_id_for_manifest():
    from kindi.backends.nanbeige import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_nanbeige"}) == "nanbeige-33b"
    assert model_id_for_manifest({"backend": "kindi_gemma4"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


class _FakeNanbeigeEngine(object):
    """Records the (tok, pos) that generate() feeds, like Engine.generate does."""

    def __init__(self):
        self.calls = []

    def prefill_batch(self, ids, pos0=0):
        self.calls.append(("prefill", list(ids), pos0))
        return 100                     # prediction for position len(ids)

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return 101


def test_nanbeige_generate_feeds_each_token_at_its_own_position():
    """Regression: generate() must step the first generated token at len(prompt)
    (like Engine.generate does) not at len(prompt) + 1 - every subsequent
    token would then occupy a position one too high."""
    from kindi.backends.base import SamplingParams
    from kindi.backends.nanbeige import NanbeigeBackend

    class _FakeTok(object):
        bos = 1
        tokens = ["", "a"]
        types = [0, 0]
        eos = None

        def decode(self, ids):
            return "?"

    backend = NanbeigeBackend()
    backend.loaded = True
    backend.engine = _FakeNanbeigeEngine()
    backend.tok = _FakeTok()
    backend.stop_ids = set()

    params = SamplingParams(max_tokens=3)
    events = list(backend.generate([10, 11, 12], params))
    tokens = [e.token_id for e in events if e.kind == "token"]
    assert tokens == [100, 101, 101]
    assert events[-1].kind == "done"
    # prefill occupies 0..2; generated tokens are fed at 3, 4, 5
    assert backend.engine.calls == [
        ("prefill", [10, 11, 12], 0),
        ("step", 100, 3),
        ("step", 101, 4),
    ]


def test_lightning_resolves_by_model_id_unloaded():
    from kindi.backends.lightning import LightningBackend

    backend = registry.resolve("nemotron-lightning-30b")
    assert isinstance(backend, LightningBackend)
    assert backend.loaded is False


def test_lightning_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("nemotron-lightning-30b", config={"pack": "no-such-pack-dir"})


def test_lightning_model_id_for_manifest():
    from kindi.backends.lightning import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_lightning"}) == "nemotron-lightning-30b"
    assert model_id_for_manifest({"backend": "kindi_nanbeige"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


def test_bonsai_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("bonsai2-2b", config={"pack": "no-such-pack-dir"})


def test_bonsai_model_id_for_manifest():
    from kindi.backends.bonsai import model_id_for_manifest

    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-2B"}) == "bonsai2-2b"
    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-27B"}) == "bonsai2-27b"
    # The 27B GGUF names itself only "Hf"; the size must come from source_file,
    # the same fallback config.py uses to name the served model.
    assert model_id_for_manifest({
        "model": "Hf",
        "source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    }) == "bonsai2-27b"
    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-XL"}) is None
    assert model_id_for_manifest({"model": "Other-Model"}) is None
    assert model_id_for_manifest({}) is None
