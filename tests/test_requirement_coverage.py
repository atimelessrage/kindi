"""Requirement coverage for behavior that is easy to verify on the CPU.

These tests are written against the requirements in ``docs/development/requirements/``
and the user-visible contracts in the docs, not against private implementation details.
They cover the open review findings for config path semantics, saved-opening
invalidation, and cache-failure metrics.
"""
import configparser
import json
import os
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kindi import cache, catalog, config, hostram, sched, server


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


# --- Host-RAM preflight (REQ-0005: the serving boundary is bounded) -----------

def test_host_ram_refusal_names_the_load_and_the_remedy(monkeypatch):
    """A load that cannot fit stops at once, saying what to do (no minutes-long stall)."""
    monkeypatch.delenv("KINDI_HOST_RAM_CHECK", raising=False)
    logged = []
    with pytest.raises(SystemExit, match="about .* GB of free host RAM") as exc:
        hostram.check(16 * 2 ** 30, "the 27B weights", log=logged.append, avail=2 * 2 ** 30)
    message = str(exc.value)
    assert "stop the other model servers" in message
    assert "KINDI_HOST_RAM_CHECK=0" in message
    assert logged and "needed" in logged[0]


def test_host_ram_check_passes_when_the_load_fits(monkeypatch):
    monkeypatch.delenv("KINDI_HOST_RAM_CHECK", raising=False)
    logged = []
    hostram.check(2 * 2 ** 30, "the 2B weights", log=logged.append, avail=64 * 2 ** 30)
    assert logged  # the preflight line is still printed


def test_host_ram_check_opt_out_loads_anyway(monkeypatch):
    """KINDI_HOST_RAM_CHECK=0 is the documented escape hatch, even with ~no RAM free."""
    monkeypatch.setenv("KINDI_HOST_RAM_CHECK", "0")
    logged = []
    hostram.check(64 * 2 ** 30, "the 27B weights", log=logged.append, avail=0)  # must not raise
    assert logged == []


def test_host_ram_check_unknown_free_ram_does_not_block(monkeypatch):
    """When free RAM cannot be determined, the preflight must not invent a refusal."""
    monkeypatch.delenv("KINDI_HOST_RAM_CHECK", raising=False)
    monkeypatch.setattr(hostram, "available", lambda: None)
    hostram.check(64 * 2 ** 30, "the 27B weights")  # must not raise


# --- Model catalog fit and selection (REQ-0008: installer classifies safely) ---

def test_uneven_pair_fits_when_the_second_card_holds_its_half():
    """REQ-0017 / REQ-0008: the 27B split needs 3.8 GiB on *each* card, so a
    6 GB + 8 GB pair (or any pair where neither card alone fits) runs as a split."""
    m = catalog.MODELS["bonsai-27b"]  # one=7600, two=3800
    assert catalog.fits(m, [("6gb", 6144), ("8gb", 8192)]) == "one card"  # the 8 GB card alone fits
    assert catalog.fits(m, [("6gb", 6144), ("7gb", 7168)]) == "two cards"  # neither alone; both hold a half
    assert catalog.fits(m, [("7gb", 7168), ("6gb", 6144)]) == "two cards"  # order independent
    assert catalog.fits(m, [("6gb", 6144), ("tiny", 3000)]) == ""


def test_select_writes_the_pack_and_keeps_the_other_keys(tmp_path):
    """`kindi use` edits the config in place and never drops unrelated settings."""
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\nctx = 4096\ntemperature = 0.25\n", encoding="utf-8")
    logged = []
    catalog._select({"source": str(conf)}, "/x/packs/bonsai-2b", log=logged.append)
    c = configparser.ConfigParser()
    c.read(str(conf))
    assert c["kindi"]["pack"] == "/x/packs/bonsai-2b"
    assert c["kindi"]["ctx"] == "4096" and c["kindi"]["temperature"] == "0.25"
    assert any("selected:" in line for line in logged)


# --- Blocking sync default (REQ-0005': bounded host behaviour) ----------------

def test_blocking_sync_keeps_the_driver_default_when_the_driver_fails(monkeypatch):
    """With no usable CUDA driver the default-on sync must log and continue, not raise."""
    import kindi.gpudrv as gpudrv
    sched._done[0] = False
    monkeypatch.delenv("KINDI_BLOCKING_SYNC", raising=False)
    monkeypatch.setattr(gpudrv, "HIP", False)
    def _boom():
        raise OSError("no driver")
    monkeypatch.setattr(gpudrv, "driver", _boom)
    logged = []
    sched.blocking_sync(log=logged.append)  # must not raise
    assert any("driver default kept" in line for line in logged)
    assert sched._done[0] is True  # the once-per-process slot is consumed
