"""CPU-only tests for the one effective runtime-configuration view (roadmap 2026-11-20 #2)."""
from kindi import runtime

CFG = {"pack": "C:/packs/bonsai-9b", "ctx": "16384", "head": "fast",
       "tandem_vram_mb": "6144", "tandem_reserve_mb": "512"}


class _Engine(object):
    tc = True
    kv = "q8_0"
    chunk = {"T": 256}
    threads = None

    def __init__(self, ctx=8192, fast=False):
        self.ctx = ctx
        if fast:
            self.fast_centroids = object()

    def spec_begin(self, *args, **kwargs):
        return None


class _Kindi(object):
    variant = "qwen35"
    pack_id = "abc123def456"

    def __init__(self, spec, head):
        self._spec, self._head = spec, head

    def drafting(self):
        return self._spec

    def head_active(self):
        return self._head


def _spec(mode="mtp"):
    return {"enabled": mode is not None, "method": mode, "requested": "mtp", "reason": None,
            "head": "kindi-mtp-2b-v1.npz", "head_sha256": "1b5c0c790a59", "drafter": None,
            "last": None}


def test_unloaded_runtime_reports_requests_not_effective_values(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    state = runtime.describe(CFG)
    assert state["loaded"] is False
    assert state["engine"] is None
    assert state["ctx_requested"] == 16384 and state["ctx_active"] is None
    assert state["ctx_active"] is None
    assert state["tensor_cores"] is None and state["kv_cache"] is None
    assert state["head"] is None and state["head_requested"] == "fast"
    assert state["spec"]["requested"] == "mtp"
    assert state["spec"]["method"] is None and state["spec"]["reason"] == "engine not loaded"
    assert state["chunk_requested"] is None
    assert state["vram_budget_mb"] == 6144 and state["vram_reserve_mb"] == 512
    line = runtime.render(state)
    assert line.startswith("runtime: not loaded")
    assert "head requested=fast" in line and "spec requested=mtp" in line


def test_loaded_engine_reports_the_values_it_settled_on(monkeypatch):
    monkeypatch.setenv("KINDI_TC_HP", "1")
    engine = _Engine(ctx=8192)
    engine.threads = 6
    spec = _spec("mtp")
    state = runtime.describe(CFG, engine, _Kindi(spec, "fast"))
    assert state["loaded"] is True and state["engine"] == "_Engine"
    assert state["variant"] == "qwen35" and state["pack_id"] == "abc123def456"
    assert state["ctx_active"] == 8192 and state["ctx_requested"] == 16384
    assert state["tensor_cores"] is True and state["hp"] is True
    assert state["kv_cache"] == "q8_0" and state["chunk"] == 256
    assert state["threads"] == 6
    assert state["head"] == "fast"
    assert state["spec"] is spec
    line = runtime.render(state)
    assert "engine=_Engine (qwen35)" in line
    assert "tc=yes hp=yes kv=q8_0 ctx=8192/16384 spec=mtp head=fast chunk=256 threads=6" in line


def test_fast_head_request_falls_back_to_exact_without_a_fast_head():
    state = runtime.describe(CFG, _Engine())
    assert state["head_requested"] == "fast"
    assert state["hp"] is None                  # TC, but the plan recorded no HP value

    class NoTc(object):
        ctx = 4096

    assert runtime.describe(CFG, NoTc())["hp"] is False
    assert runtime.describe(CFG)["hp"] is None
    assert state["head"] == "exact"                 # the requested head is not the effective one
    assert runtime.render(state).find("head=exact") > 0


def test_engine_without_a_speculative_path_says_so():
    class Plain(object):
        ctx = 4096

    state = runtime.describe(CFG, Plain())
    assert state["spec"]["enabled"] is False
    assert state["spec"]["method"] is None
    assert "no speculative path" in state["spec"]["reason"]


def test_engine_speculation_state_is_used_without_a_kindi_wrapper(monkeypatch):
    def fake_state(engine):
        return {"requested": "mtp", "mode": "mtp", "reason": None,
                "head": {"file": "head.npz", "state": "loaded", "sha256": "a" * 64}}

    monkeypatch.setattr("kindi.spec.effective_state", fake_state)
    state = runtime.describe(CFG, _Engine())
    assert state["spec"]["enabled"] is True and state["spec"]["method"] == "mtp"
    assert state["spec"]["head"] == "head.npz"
    assert state["spec"]["head_sha256"] == "a" * 12
    assert state["spec"]["drafter"] is None
