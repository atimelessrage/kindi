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
    # an unknown model id now resolves to the llama.cpp proxy wildcard (REQ-0019 T4.4);
    # the refusal text only fires when no wildcard is registered
    from kindi.backends.llamacpp import LlamaCppBackend

    backend = registry.resolve("no-such-model")
    assert isinstance(backend, LlamaCppBackend)
    registry.unregister_backend("llamacpp")
    with pytest.raises(registry.BackendNotFound) as excinfo:
        registry.resolve("no-such-model")
    message = str(excinfo.value)
    assert "no-such-model" in message
    assert "fake" in message
    assert "bonsai" in message


def test_resolve_respects_device_filter():
    registry.register_backend(ToyBackend)
    # the toy is cpu-only and is excluded by a cuda request; the proxy wildcard has no
    # device filter of its own, so it is what serves instead (device choice is the
    # upstream server's business)
    backend = registry.resolve("toy-1b", device="cuda")
    assert backend.capabilities.name == "llamacpp"
    assert isinstance(registry.resolve("toy-1b", device="cpu"), ToyBackend)


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
    # the adapter's contract is non-drafting: its capabilities and done event must agree
    assert NanbeigeBackend.capabilities.supports_speculation is False
    assert events[-1].drafted == 0 and events[-1].accepted == 0
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


class _FakeLightningTok(object):
    bos, eos, vocab, types = 1, 2, ["", "a", "b", "c"], [0, 0, 0, 0]

    def decode(self, ids):
        return "" if not ids else "b" if ids[0] == 101 else "?"


class _FakeLightningEngine(object):
    """Prefill/step plus the candidate window the sampler reads."""

    def __init__(self, cand=(), sc=()):
        import numpy as np
        from types import SimpleNamespace

        self.calls = []
        self.closed = False
        self.cand, self.sc = list(cand), list(sc)
        logits = np.zeros(128, dtype=np.float64)
        for token_id, score in zip(self.cand, self.sc):
            if 0 <= token_id < logits.size:
                logits[token_id] = score
        self.logits = logits
        self.cp = SimpleNamespace(asnumpy=np.asarray, flatnonzero=np.flatnonzero)

    def close(self):
        self.closed = True

    def reset(self):
        self.calls.append(("reset",))

    def prefill(self, ids, pos=None):
        self.calls.append(("prefill", list(ids), pos))
        return 100

    def step(self, tok):
        self.calls.append(("step", int(tok)))
        return 101

    def candidates(self, w):
        # what lightning.Engine.candidates returns: int64 ids, float64 logits
        import numpy as np

        return np.array(self.cand, dtype=np.int64), np.array(self.sc, dtype=np.float64)


def _lightning(greedy_engine=None):
    from kindi.backends.lightning import LightningBackend

    backend = LightningBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakeLightningTok()
    backend.engine = greedy_engine or _FakeLightningEngine()
    backend.stop_ids = {backend.tok.eos}
    return backend


def test_lightning_close_releases_its_engine():
    backend = _lightning()
    engine = backend.engine
    backend.close()
    assert engine.closed is True
    assert backend.engine is None
    assert backend.loaded is False


def test_lightning_generate_prefills_resets_and_stops_on_a_control_token():
    from kindi.backends.base import SamplingParams

    backend = _lightning()
    backend.stop_ids = {101}              # the second generated token is a control token
    events = list(backend.generate([10, 11], SamplingParams(max_tokens=4)))
    assert [e.token_id for e in events if e.kind == "token"] == [100, 101]
    assert events[-1].finish_reason == "stop"
    # no sampling is asked for at temperature 0, and the last token is not stepped again
    assert backend.engine.calls == [("reset",), ("prefill", [10, 11], 0), ("step", 100)]


def test_lightning_generate_reports_length_and_honours_stop_text():
    from kindi.backends.base import SamplingParams

    plain = list(_lightning().generate([10], SamplingParams(max_tokens=2)))
    assert plain[-1].finish_reason == "length"
    assert [e.token_id for e in plain if e.kind == "token"] == [100, 101]

    by_text = list(_lightning().generate([10], SamplingParams(max_tokens=3, stop=("?",))))
    assert by_text[-1].finish_reason == "stop"


def test_lightning_token_filter_argmaxes_over_the_allowed_candidates():
    """The mask runs before the 256 cap and before the argmax (the contract's
    constrained decoding), on the one sampler the server path also uses."""
    from kindi.backends.base import SamplingParams

    engine = _FakeLightningEngine(cand=[7, 8, 9], sc=[3.0, 9.0, 1.0])
    backend = _lightning(engine)
    params = SamplingParams(max_tokens=1, token_filter=lambda state: {7, 9})
    events = list(backend.generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [7]


def test_stop_scanner_finds_a_match_across_two_pieces():
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("END",))
    assert scanner.hit("before EN") is False
    assert scanner.hit("before END") is True
    assert StopScanner(()).hit("anything") is False


def test_stop_scanner_keeps_only_the_overlap_instead_of_rescanning():
    """The cursor is the O(n^2) fix: a long reply costs O(piece) per token, and a
    stop that straddles two pieces is still found."""
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("<eos>",))
    text = "x" * 5000
    assert scanner.hit(text) is False
    assert scanner.hit(text + "x") is False
    assert scanner.hit(text + "x<eo") is False
    assert scanner.hit(text + "x<eos>") is True


def test_lightning_token_filter_receives_the_prompt():
    """A prompt-dependent filter must see the prompt ids, not an empty list."""
    from kindi.backends.base import SamplingParams

    engine = _FakeLightningEngine(cand=[7, 8, 9], sc=[3.0, 9.0, 1.0])
    seen = []

    def filt(state):
        seen.append((list(state["prompt_ids"]), list(state["generated"])))
        return {7, 9}

    list(_lightning(engine).generate([10, 11], SamplingParams(max_tokens=1, token_filter=filt)))
    assert seen[0][0] == [10, 11]


def test_lightning_refuses_a_prompt_longer_than_the_context():
    """An over-long prompt used to become max_tokens=0: an empty reply reported as
    `length`, which hides the real problem from the caller."""
    from kindi.backends.base import SamplingParams

    backend = _lightning()
    backend.ctx = 3
    with pytest.raises(registry.BackendUnavailable, match="context is 3"):
        list(backend.generate([1, 2, 3, 4], SamplingParams(max_tokens=1)))


def test_a_shipped_backend_refuses_a_second_registration():
    """Every adapter used to register with replace=True, so a later registration could
    silently shadow a shipped family; the guard in register_backend is real again."""
    from kindi.backends.bonsai import BonsaiBackend

    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend(BonsaiBackend)


def test_lightning_apply_chat_template_needs_a_loaded_pack():
    from kindi.backends.lightning import LightningBackend

    with pytest.raises(registry.BackendUnavailable, match="no model loaded"):
        LightningBackend().apply_chat_template([{"role": "user", "content": "hi"}])


def test_ambiguous_resolution_is_refused_not_first_wins():
    """Two backends match equally well: registration order must not silently pick one."""
    from kindi.backends.base import BackendCapabilities

    class _Twin(FakeBackend):
        capabilities = BackendCapabilities(name="twin", model_ids=("kindi-fake",),
                                           architectures=("fake",))

    registry.register_backend(_Twin, replace=True)
    try:
        with pytest.raises(registry.BackendNotFound, match="ambiguous: fake and twin"):
            registry.resolve("kindi-fake")
        # a model id only one of them declares still resolves
        assert isinstance(registry.resolve("fake"), FakeBackend)
    finally:
        registry.unregister_backend("twin")


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


class _FakeBonsaiTok(object):
    """The tokenizer surface BonsaiBackend.generate uses: per-token decode and the stop ids."""

    def __init__(self, pieces):
        self.pieces = dict(pieces)
        self.sid = {"<|endoftext|>": 1, "<|im_end|>": 2}

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeBonsaiEngine(object):
    """feed/step in the adapter's order: prefill the prompt, then one prediction per step."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def feed(self, tok, pos):
        self.calls.append(("feed", int(tok), int(pos)))

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return self.script.pop(0) if self.script else 2      # 2 is <|im_end|>: a short script ends


def _bonsai(script):
    from kindi.backends.bonsai import BonsaiBackend

    backend = BonsaiBackend()
    backend.loaded = True
    backend.ctx = 4096
    # "END" is split as "EN" + "D" by the tokenizer, which is what the scanner must survive
    backend.tok = _FakeBonsaiTok({10: "The sky is ", 11: "EN", 12: "D", 13: "blue."})
    backend.engine = _FakeBonsaiEngine(script)
    return backend


def test_bonsai_generate_stops_on_a_stop_string_split_across_two_pieces():
    """The StopScanner swap has to hold on the adapter's own decoded pieces.

    "END" arrives as "EN" then "D": neither piece contains the stop, only the
    accumulated text does. A scanner that looked at the new piece alone, or that
    dropped its overlap between hits, would miss it and finish the turn on 13.
    """
    from kindi.backends.base import SamplingParams

    backend = _bonsai([10, 11, 12, 13])
    events = list(backend.generate([5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].kind == "done"
    assert events[-1].finish_reason == "stop"
    # three steps after prefill: the split stop is found on the third piece, and
    # the engine is not stepped again once it is, so 13 is never decoded
    assert backend.engine.calls == [
        ("reset",),
        ("feed", 5, 0),
        ("step", 6, 1),
        ("step", 10, 2),
        ("step", 11, 3),
    ]

    # contrast: the same script with a stop that fits inside one piece ends a piece earlier
    early = list(_bonsai([10, 11, 12, 13]).generate(
        [5, 6], SamplingParams(max_tokens=4, stop=("EN",))))
    assert [e.token_id for e in early if e.kind == "token"] == [10, 11]
    assert early[-1].finish_reason == "stop"


# --- T4.2 Gemma 4 and DeepSeek V4.1 adapters (CPU only: no pack, no engine) -------------


def test_gemma4_and_dsv41_resolve_by_model_id_unloaded():
    from kindi.backends.dsv41 import DeepseekV41Backend
    from kindi.backends.gemma4 import Gemma4Backend

    gemma = registry.resolve("gemma4-e4b")
    assert isinstance(gemma, Gemma4Backend)
    assert gemma.loaded is False
    grand = registry.resolve("deepseek-v41-flash")
    assert isinstance(grand, DeepseekV41Backend)
    assert grand.loaded is False


def test_gemma4_and_dsv41_load_reject_a_missing_pack_directory():
    for model_id in ("gemma4-e4b", "deepseek-v41-flash"):
        with pytest.raises(registry.BackendUnavailable, match="pack directory"):
            registry.resolve(model_id, config={"pack": "no-such-pack-dir"})


def test_gemma4_and_dsv41_model_id_for_manifest():
    from kindi.backends.dsv41 import model_id_for_manifest as dsv
    from kindi.backends.gemma4 import model_id_for_manifest as gemma

    assert gemma({"backend": "kindi_gemma4"}) == "gemma4-e4b"
    assert gemma({"backend": "kindi_dsv41"}) is None
    assert gemma(None) is None
    assert dsv({"backend": "kindi_dsv41"}) == "deepseek-v41-flash"
    assert dsv({"backend": "kindi_gemma4"}) is None
    assert dsv({}) is None


def test_dsv41_load_needs_the_model_path_and_says_what_to_add(tmp_path):
    (tmp_path / "kindi_pack.json").write_text('{"backend": "kindi_dsv41"}', encoding="utf-8")
    with pytest.raises(registry.BackendUnavailable, match="model_path"):
        registry.resolve("deepseek-v41-flash", config={"pack": str(tmp_path)})


def test_gemma4_load_wires_pack_ctx_two_card_choice_tokenizer_and_stop_ids(tmp_path, monkeypatch):
    """`load()` is executable CPU-side even though no Gemma 4 pack exists here.

    The engine/tokenizer modules are recording fakes, so this proves the wiring the first GPU
    run would otherwise be the first to discover: the manifest gains the pack path, the
    requested ctx reaches the chosen engine class, a two-card request picks `TPEngine`, the
    tokenizer is built from the manifest's `model_path`, and the adapter's stop ids come from
    that tokenizer.
    """
    import json
    import sys
    import types

    from kindi.backends.gemma4 import Gemma4Backend

    calls = {}

    class _Tok(object):
        bos, eos = 2, 1
        tokens = ["<eos>", "a", "b", "c"]
        stop_ids = {1, 9}

        def __init__(self, model_path):
            calls["tok"] = model_path

    class _FakeEngine(object):
        def __init__(self, man, ctx=None, log=None):
            calls["engine"] = (man, ctx, log)
            self.ctx = 4096                  # the loader clamps 16384 down to what fits
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeTPEngine(_FakeEngine):
        pass

    gemma4_mod = types.ModuleType("kindi.gemma4")
    gemma4_mod.Engine = _FakeEngine
    tp_mod = types.ModuleType("kindi.gemma4_tp")
    tp_mod.TPEngine = _FakeTPEngine
    tok_mod = types.ModuleType("kindi.gemma4_tokenizer")
    tok_mod.Tok = _Tok
    monkeypatch.setitem(sys.modules, "kindi.gemma4", gemma4_mod)
    monkeypatch.setitem(sys.modules, "kindi.gemma4_tp", tp_mod)
    monkeypatch.setitem(sys.modules, "kindi.gemma4_tokenizer", tok_mod)
    (tmp_path / "kindi_pack.json").write_text(json.dumps({
        "backend": "kindi_gemma4", "model": "gemma4-e4b", "model_path": "models/gemma4-e4b.gguf",
    }), encoding="utf-8")

    for tp_mode, engine_class in (("0", _FakeEngine), ("1", _FakeTPEngine)):
        monkeypatch.setenv("KINDI_G4_TP", tp_mode)
        calls.clear()
        logged = []
        log_sink = logged.append
        backend = Gemma4Backend()
        loaded = backend.load({"pack": str(tmp_path), "ctx": 16384, "log": log_sink})
        assert loaded is backend
        assert type(backend.engine) is engine_class
        man, ctx, log = calls["engine"]
        assert ctx == 16384
        assert log == log_sink
        assert man["pack_path"] == str(tmp_path)
        assert man["model_path"] == "models/gemma4-e4b.gguf"
        assert calls["tok"] == "models/gemma4-e4b.gguf"
        assert backend.ctx == 4096
        assert backend.stop_ids == {1, 9}
        assert backend.loaded is True
        engine = backend.engine
        backend.close()
        assert engine.closed is True
        assert backend.loaded is False


def test_dsv41_load_wires_model_path_ctx_tmax_tokenizer_and_stop_id(tmp_path, monkeypatch):
    """`load()` is executable CPU-side even though no DeepSeek V4.1 weights exist here.

    `kindi.dsv41.Model` is a recording fake, so this proves the manifest's first GGUF shard,
    the requested ctx/tmax, the engine's clamped ctx and tokenizer, and the one-id stop set
    before the first GPU run.
    """
    import json
    import sys
    import types

    from kindi.backends.dsv41 import DeepseekV41Backend

    calls = {}

    class _Tok(object):
        vocab = ["<eos>", "a", "b"]

    class _FakeModel(object):
        def __init__(self, path, ctx=None, tmax=None):
            calls["model"] = (path, ctx, tmax)
            self.ctx = 2048                  # the engine clamps 4096 down to what fits
            self.tok = _Tok()
            self.eos = 760

    dsv41_mod = types.ModuleType("kindi.dsv41")
    dsv41_mod.Model = _FakeModel
    monkeypatch.setitem(sys.modules, "kindi.dsv41", dsv41_mod)
    (tmp_path / "kindi_pack.json").write_text(json.dumps({
        "backend": "kindi_dsv41", "model": "DeepSeek-V4.1-Flash", "model_path": "shards/deepseek-00001.gguf",
    }), encoding="utf-8")

    backend = DeepseekV41Backend()
    loaded = backend.load({"pack": str(tmp_path), "ctx": 4096, "tmax": 16,
                           "log": lambda line: None})
    assert loaded is backend
    assert calls["model"] == ("shards/deepseek-00001.gguf", 4096, 16)
    assert isinstance(backend.engine, _FakeModel)
    assert backend.ctx == 2048
    assert backend.tok is backend.engine.tok
    assert backend.stop_ids == {760}
    assert backend.loaded is True
    assert backend.info()["ctx"] == 2048


def _gemma4(engine):
    from kindi.backends.gemma4 import Gemma4Backend

    class _Tok(object):
        bos, eos = 2, 1
        tokens = ["", "a", "b", "c"]
        stop_ids = {1}

        def decode_bytes(self, i):
            return ("b" if i == 5 else "x" if i == 6 else "").encode()

        def decode(self, ids):
            return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8")

    backend = Gemma4Backend()
    backend.loaded = True
    backend.ctx = 64
    backend.tok = _Tok()
    backend.engine = engine
    backend.stop_ids = set(_Tok.stop_ids)
    return backend


class _FakeG4Cp(object):
    """The two cupy calls the adapter makes (argmax/asnumpy), on numpy."""

    @staticmethod
    def argmax(z):
        import numpy as np

        class _Item(object):
            def __init__(self, v):
                self.v = v

            def item(self):
                return self.v

        return _Item(int(np.argmax(z)))

    @staticmethod
    def asnumpy(z):
        import numpy as np

        return np.asarray(z, dtype=np.float32)


class _FakeG4Engine(object):
    """Logits: token `peak` after the prompt, `peak + 1` on every later step."""

    def __init__(self, peak=5):
        self.cp = _FakeG4Cp()
        self.calls = []
        self.peak = peak
        self.steps = 0

    def reset(self):
        self.calls.append(("reset",))
        self.steps = 0

    def forward_batch(self, tokens, pos):
        self.calls.append(("forward_batch", list(tokens), pos))

    def forward(self, token, pos, logits=True):
        self.calls.append(("forward", int(token), pos, logits))
        self.steps += 1
        z = [0.0] * 8
        z[self.peak if self.steps == 1 else self.peak + 1] = 9.0
        return z


def test_gemma4_generate_prefills_in_one_batch_and_steps_its_own_position():
    from kindi.backends.base import SamplingParams

    eng = _FakeG4Engine(peak=5)
    events = list(_gemma4(eng).generate([10, 11, 12], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6, 6]
    assert events[-1].finish_reason == "length"
    assert eng.calls == [
        ("reset",),
        ("forward_batch", [10, 11], 0),
        ("forward", 12, 2, True),        # the last prompt token returns the first logits
        ("forward", 5, 3, True),         # generated token n sits at len(prompt) + n
        ("forward", 6, 4, True),
    ]


def test_gemma4_generate_stops_on_a_stop_id_and_on_stop_text():
    from kindi.backends.base import SamplingParams

    class _Breaks(_FakeG4Engine):
        def __init__(self):
            _FakeG4Engine.__init__(self, peak=1)      # 1 is eos

    events = list(_gemma4(_Breaks()).generate([10], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == []
    assert events[-1].finish_reason == "stop"

    class _Text(_FakeG4Engine):
        def __init__(self):
            _FakeG4Engine.__init__(self, peak=5)      # decodes as "b", then "x" -> "bx"

    backend = _gemma4(_Text())
    events = list(backend.generate([10], SamplingParams(max_tokens=4, stop=("bx",))))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6]
    assert events[-1].finish_reason == "stop"


def test_gemma4_token_filter_argmaxes_over_the_allowed_ids_at_temperature_0():
    from kindi.backends.base import SamplingParams

    eng = _FakeG4Engine(peak=5)                       # logits favour 5
    params = SamplingParams(max_tokens=1, token_filter=lambda st: {6})
    events = list(_gemma4(eng).generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_gemma4_and_dsv41_apply_chat_template_need_a_loaded_pack():
    from kindi.backends.dsv41 import DeepseekV41Backend
    from kindi.backends.gemma4 import Gemma4Backend

    for backend in (Gemma4Backend(), DeepseekV41Backend()):
        with pytest.raises(registry.BackendUnavailable, match="no model loaded"):
            backend.apply_chat_template([{"role": "user", "content": "hi"}])


class _FakeDsTok(object):
    vocab = ["", "a", "b", "c"]

    def encode(self, text, special=True):
        return [9]

    def decode_bytes(self, i):
        return ("b" if i == 5 else "x" if i == 6 else "").encode()

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8")


class _FakeDsEngine(object):
    """Model.forward: the whole prompt once, then one token per step (numpy logits)."""

    def __init__(self, peak=5):
        import numpy as np

        self._np = np
        self.peak = peak
        self.calls = []
        self.steps = 0
        self.tok = _FakeDsTok()
        self.eos = 1
        self.ctx = 64

    def reset(self):
        self.calls.append(("reset",))
        self.steps = 0

    def forward(self, ids):
        import numpy as np

        self.calls.append(("forward", list(ids)))
        self.steps += 1
        z = np.zeros(8, dtype=np.float32)
        z[self.peak if self.steps == 1 else self.peak + 1] = 9.0
        return z


def _dsv41(engine):
    from kindi.backends.dsv41 import DeepseekV41Backend

    backend = DeepseekV41Backend()
    backend.loaded = True
    backend.ctx = 64
    backend.engine = engine
    backend.tok = engine.tok
    backend.stop_ids = {engine.eos}
    return backend


def test_dsv41_generate_feeds_the_prompt_at_once_then_one_token_per_step():
    from kindi.backends.base import SamplingParams

    eng = _FakeDsEngine(peak=5)
    events = list(_dsv41(eng).generate([10, 11, 12], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6, 6]
    assert events[-1].finish_reason == "length"
    assert eng.calls == [("reset",), ("forward", [10, 11, 12]), ("forward", [5]), ("forward", [6])]


def test_dsv41_generate_stops_on_eos_and_token_filter():
    from kindi.backends.base import SamplingParams

    eng = _FakeDsEngine(peak=1)                       # 1 is eos
    events = list(_dsv41(eng).generate([10], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == []
    assert events[-1].finish_reason == "stop"

    eng = _FakeDsEngine(peak=5)
    params = SamplingParams(max_tokens=1, token_filter=lambda st: {6})
    events = list(_dsv41(eng).generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_gemma4_load_needs_the_model_path(tmp_path):
    (tmp_path / "kindi_pack.json").write_text('{"backend": "kindi_gemma4"}', encoding="utf-8")
    with pytest.raises(registry.BackendUnavailable, match="model_path"):
        registry.resolve("gemma4-e4b", config={"pack": str(tmp_path)})


# --- the shared stop scan and the token-filter state, on the adapters that lacked them ---------

# "END" is never one decoded piece here: the fakes split it as "EN" + "D", so only the
# accumulated text contains the stop. Token 13 must never be emitted once it is found.
_PIECES = {10: "The sky is ", 11: "EN", 12: "D", 13: "blue."}
_SCRIPT = [10, 11, 12, 13]


class _FakePieceTok(object):
    """decode([id]) with scripted text, for the adapters that decode one id at a time."""

    eos = None

    def __init__(self, pieces):
        self.pieces = dict(pieces)
        self.sid = {"<|im_end|>": 2, "<|endoftext|>": 1}

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeLightningScriptEngine(object):
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def prefill(self, ids, pos=None):
        self.calls.append(("prefill", list(ids), pos))
        return self.script.pop(0) if self.script else 2

    def step(self, tok):
        self.calls.append(("step", int(tok)))
        return self.script.pop(0) if self.script else 2


def test_lightning_generate_stops_on_a_stop_string_split_across_two_pieces():
    from kindi.backends.base import SamplingParams
    from kindi.backends.lightning import LightningBackend

    backend = LightningBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakePieceTok(_PIECES)
    backend.engine = _FakeLightningScriptEngine(_SCRIPT)
    backend.stop_ids = set()
    events = list(backend.generate([5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].finish_reason == "stop"


class _FakeNanbeigeTok(object):
    bos = 1
    eos = None
    tokens = [""] * 32
    types = [0] * 32

    def __init__(self, pieces):
        self.pieces = dict(pieces)

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeNanbeigeScriptEngine(object):
    """prefill_batch/step plus the logits window _sample_next reads."""

    def __init__(self, script, vocab=32):
        import numpy as np
        from types import SimpleNamespace

        self.script, self.vocab, self.n, self.calls = list(script), vocab, 0, []
        self.card = SimpleNamespace(dev=0, logits=np.full(vocab, -10.0))
        self.cards = [self.card]

        class _Device(object):
            def __init__(self, dev):
                self.dev = dev

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Cuda(object):
            Device = _Device

        self.cp = SimpleNamespace(cuda=_Cuda(), asnumpy=np.asarray,
                                  flatnonzero=np.flatnonzero)

    def _logits(self):
        import numpy as np

        z = np.full(self.vocab, -10.0)
        if self.n < len(self.script):
            z[self.script[self.n]] = 10.0
        self.n += 1
        self.card.logits = z
        return z

    def prefill_batch(self, ids, pos0=0):
        self.calls.append(("prefill", list(ids), pos0))
        return int(self._logits().argmax())

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return int(self._logits().argmax())


def _nanbeige(script, pieces=None):
    from kindi.backends.nanbeige import NanbeigeBackend

    backend = NanbeigeBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakeNanbeigeTok(pieces or _PIECES)
    backend.engine = _FakeNanbeigeScriptEngine(script)
    backend.stop_ids = set()
    return backend


def test_nanbeige_generate_stops_on_a_stop_string_split_across_two_pieces():
    from kindi.backends.base import SamplingParams

    events = list(_nanbeige(_SCRIPT).generate(
        [5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].finish_reason == "stop"


def test_dsv41_generate_stops_on_a_stop_string_split_across_two_pieces():
    """dsv41's own pieces are "b" then "x" through _FakeDsTok, so "bx" only exists in the
    accumulated text - the case a per-piece check would miss."""
    from kindi.backends.base import SamplingParams

    events = list(_dsv41(_FakeDsEngine(peak=5)).generate(
        [10], SamplingParams(max_tokens=4, stop=("bx",))))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6]
    assert events[-1].finish_reason == "stop"


def test_gemma4_and_dsv41_token_filters_receive_the_prompt():
    """A prompt-dependent filter (a grammar seeded from the prompt, a "do not stop inside a
    fenced block" rule) is useless if the adapter hands it an empty list: it must see the real
    prompt ids, and the masked id must be the token that comes out."""
    from kindi.backends.base import SamplingParams

    for backend in (_gemma4(_FakeG4Engine(peak=5)), _dsv41(_FakeDsEngine(peak=5))):
        seen = []

        def filt(state, seen=seen):
            seen.append((list(state["prompt_ids"]), list(state["generated"])))
            return {6}

        events = list(backend.generate([10, 11, 12], SamplingParams(max_tokens=1,
                                                                   token_filter=filt)))
        assert seen == [([10, 11, 12], [])], type(backend).__name__
        assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_nanbeige_token_filter_receives_the_prompt():
    from kindi.backends.base import SamplingParams

    seen = []

    def filt(state):
        seen.append(list(state["prompt_ids"]))
        return {10}

    events = list(_nanbeige([10]).generate([5, 6, 7], SamplingParams(max_tokens=1,
                                                                    token_filter=filt)))
    assert seen == [[5, 6, 7]]
    assert [e.token_id for e in events if e.kind == "token"] == [10]


class _CountingText(str):
    """A str that counts the characters slicing hands out, so a rescan is measurable."""

    read = 0

    def __getitem__(self, item):
        piece = str.__getitem__(self, item)
        _CountingText.read += len(piece)
        return piece


def test_stop_scanner_reads_only_the_tail_of_a_long_reply():
    """The O(n^2) defect re-scanned the whole reply per token; the cursor must read only the
    new characters plus the longest-stop overlap, however long the reply gets."""
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("<eos>",))
    _CountingText.read = 0
    text = _CountingText("x" * 4000)
    assert scanner.hit(text) is False
    first = _CountingText.read
    for _ in range(200):
        text = _CountingText(text + "xxxx")
        assert scanner.hit(text) is False
    tail = _CountingText.read - first
    # 200 pieces of 4 chars each, read with the 4-char overlap window: ~8 chars per call.
    assert tail <= 200 * 12, "the scanner re-read %d chars for 800 new ones" % tail
    assert scanner.hit(_CountingText(text + "<eos>")) is True
