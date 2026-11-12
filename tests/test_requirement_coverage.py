"""Requirement coverage for behavior that is easy to verify on the CPU.

These tests are written against the requirements in ``docs/development/requirements/``
and the user-visible contracts in the docs, not against private implementation details.
They cover the open review findings for config path semantics, saved-opening
invalidation, and cache-failure metrics.
"""
import json
import os
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kindi import cache, config, server


@pytest.fixture(autouse=True)
def clean_kind_env(monkeypatch, tmp_path):
    """Keep config loading hermetic: no host KINDI_* values and no real user config."""
    for key in list(os.environ):
        if key.startswith("KINDI_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_conf(path, body):
    path.write_text("[kindi]\n" + body, encoding="utf-8")
    return str(path)


def test_relative_pack_resolves_against_process_cwd_not_config_directory(tmp_path, monkeypatch):
    """REQ-0009 / review B9: relative path keys follow the server process cwd."""
    cfg_dir = tmp_path / "config-dir"
    work = tmp_path / "work"
    cfg_dir.mkdir()
    work.mkdir()
    conf = _write_conf(cfg_dir / "kindi.conf", "pack = relative/pack\n")
    monkeypatch.chdir(work)

    cfg = config.load(explicit=conf)

    assert cfg["pack"] == os.path.join(str(work), "relative", "pack")


def test_model_name_is_detected_from_the_pack_manifest(tmp_path):
    """REQ-0009: a pack manifest, not the config default, names the served model."""
    gemma = tmp_path / "gemma-pack"
    gemma.mkdir()
    (gemma / "kindi_pack.json").write_text(
        json.dumps({"backend": "kindi_gemma4", "model_id": "gemma4-e4b"}), encoding="utf-8")
    conf = _write_conf(tmp_path / "kindi.conf", "pack = %s\n" % gemma)
    assert config.load(explicit=conf)["model_name"] == "gemma4-e4b"

    bonsai = tmp_path / "bonsai-pack"
    bonsai.mkdir()
    (bonsai / "kindi_pack.json").write_text(
        json.dumps({"model": "Ternary-Bonsai-2-9B"}), encoding="utf-8")
    conf = _write_conf(tmp_path / "bonsai.conf", "pack = %s\n" % bonsai)
    assert config.load(explicit=conf)["model_name"] == "bonsai2-9b"


def test_invalid_head_mode_is_rejected_before_a_backend_starts(tmp_path):
    """REQ-0009: unsupported head modes fail in config, not in an engine kernel."""
    conf = _write_conf(tmp_path / "kindi.conf", "head = approximate\n")
    with pytest.raises(SystemExit, match="head must be exact or fast"):
        config.load(explicit=conf)


class _Kv(object):
    """A minimal K/V owner: the observable contract is in-place restore of a prefix."""

    def __init__(self, shape, fill):
        self.data = np.full(shape, fill, dtype=np.float16)
        self.set_calls = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        parent = self

        class _Slice(object):
            def get(self):
                return parent.data[key].copy()

            def set(self, value):
                parent.data[key] = value
                parent.set_calls += 1

        return _Slice()

    def set(self, value):
        self.data[...] = value
        self.set_calls += 1


class _Engine(object):
    ctx = 8

    def __init__(self):
        self.M = [np.array([1], dtype=np.float32)]
        self.conv = [np.array([2], dtype=np.float32)]
        self.Kc = [_Kv((8, 2, 2), 3)]
        self.Vc = [_Kv((8, 2, 2), 4)]


def test_saved_opening_with_inconsistent_metadata_is_not_restored(tmp_path):
    """REQ-0006 / review B7: metadata that disagrees with the entry must not restore state."""
    eng = _Engine()
    openings = cache.Openings(eng, str(tmp_path / "openings"), 4, lambda *a: None)
    tokens = [11, 12, 13]
    openings.save(tokens, [np.array([7], dtype=np.float32)], [np.array([8], dtype=np.float32)])
    key = openings.key_of(tokens)
    meta_path = os.path.join(openings.root, key, "meta.json")
    meta = json.loads(open(meta_path).read())
    meta["p"] = 2
    json.dump(meta, open(meta_path, "w"))

    with pytest.raises(ValueError, match="metadata prefix length"):
        openings.load(key)

    assert eng.M[0][0] == 1 and eng.conv[0][0] == 2
    assert eng.Kc[0].set_calls == 0 and eng.Vc[0].set_calls == 0


class _MetricsEngine(object):
    ctx = 4096

    def reset(self):
        pass


class _MetricsRunner(object):
    def __init__(self):
        self.eng = _MetricsEngine()
        self.tokens = []
        self.ckpt = {}
        self.cache_errors = 0
        self.openings = type("Openings", (), {"save_failures": 3})()


class _MetricsKindi(object):
    cp = None
    supports_logprobs = False

    def __init__(self):
        self.cfg = {"model_name": "bonsai2-2b", "api_key": ""}
        self.pack_id = "fake"
        self.eng = _MetricsEngine()
        self.runner = _MetricsRunner()
        self.lock = threading.Lock()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0}
        self.t0 = 0.0

    def head_active(self):
        return "exact"


def test_metrics_exposes_saved_opening_save_failures(monkeypatch):
    """REQ-0005 / review B8: a swallowed cache save failure must be visible in /metrics."""
    K = _MetricsKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": ""}, log=lambda *a: None)
    with TestClient(app) as client:
        body = client.get("/metrics").text
    assert "kindi_opening_save_failures_total 3" in body
