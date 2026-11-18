# SPDX-License-Identifier: AGPL-3.0-only
"""C2: `kindi serve` and the registry gateway must build the same engine for a family.

Serve uses its generation wrappers (`nanbeige_gen`, `lightning_gen`) and the inline Gemma 4
branch; verify/bench use `make_engine` -> the backend registry. These stubs record the engine
constructor both paths call, so a wrapper that drifts to another class or loses a config value
fails here instead of only on the card.
"""
import json

import pytest

from kindi import engine as engine_mod
from kindi import server

LOG = lambda *a: None


class _FakeGGUF(object):
    def __init__(self, path):
        self.kv = {}


class _FakeSPM(object):
    types = (0, 3)
    eos = 3

    def __init__(self, *a, **k):
        pass


class _FakeG4Tok(object):
    stop_ids = ()

    def __init__(self, *a, **k):
        pass


class _FakeLT(object):
    vocab = ["a", "b", "<|im_start|>"]
    types = (0, 0, 3)
    eos = 2
    sid = {"<|im_start|>": 2}

    def __init__(self, *a, **k):
        pass


def _record_engine(calls):
    class FakeEngine(object):
        def __init__(self, man_or_path, **kw):
            first = dict(man_or_path) if isinstance(man_or_path, dict) else man_or_path
            calls.append((first, dict(kw)))
            self.ctx = kw.get("ctx", 4096)
            self.ring_slots = 1024

    return FakeEngine


@pytest.mark.parametrize("family", ("kindi_nanbeige", "kindi_lightning", "kindi_gemma4"))
def test_served_family_engine_matches_the_registry_gateway(tmp_path, monkeypatch, family):
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = {"backend": family, "model_path": str(tmp_path / "model.gguf")}
    (pack / "kindi_pack.json").write_text(json.dumps(manifest))
    calls = []
    fake_engine = _record_engine(calls)
    monkeypatch.setattr("kindi.gguf.GGUF", _FakeGGUF)
    if family == "kindi_nanbeige":
        monkeypatch.setenv("KINDI_NB_SELFCHECK", "0")
        monkeypatch.setattr("kindi.spm.SPM", _FakeSPM)
        monkeypatch.setattr("kindi.nanbeige.Engine", fake_engine)
    elif family == "kindi_lightning":
        monkeypatch.setenv("KINDI_LT_SELFCHECK", "0")
        monkeypatch.setattr("kindi.lightning_tokenizer.LightningTokenizer", _FakeLT)
        monkeypatch.setattr("kindi.lightning.Engine", fake_engine)
    else:
        monkeypatch.setattr("kindi.backends.gemma4.two_cards", lambda: False)
        monkeypatch.setattr("kindi.gemma4.Engine", fake_engine)
        monkeypatch.setattr("kindi.gemma4_tokenizer.Tok", _FakeG4Tok)
        monkeypatch.setattr("kindi.gemma4_cache.G4Cache", lambda eng, log=print: None)

    cfg = {"pack": str(pack), "ctx": 2048, "kv_cache": "q8_0", "head": "exact"}
    server.Kindi(cfg, log=LOG)
    engine_mod.make_engine(str(pack), ctx=2048, kv_cache=cfg["kv_cache"], log=LOG)

    assert len(calls) == 2
    assert calls[0] == calls[1], "serve and the registry gateway built different engines"
