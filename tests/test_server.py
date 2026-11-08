"""HTTP boundary tests for malformed requests, API-key enforcement, body limits and prompt-cache preservation.

These tests use a stub Kindi, so they need no GPU, model pack or tokenizer. The stub deliberately raises
TypeError for a malformed message list, which is what the old generic exception path turned into a 500 after
clearing the real Runner. The validator must stop the request before that path is reached.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from kindi import server


class _FakeEngine(object):
    ctx = 4096

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _FakeRunner(object):
    def __init__(self):
        self.eng = _FakeEngine()
        self.tokens = [1, 2, 3]
        self.ckpt = {0: object()}


class _FakeKindi(object):
    cp = None

    def __init__(self):
        self.cfg = {"model_name": "bonsai2-2b", "api_key": ""}
        self.eng = _FakeEngine()
        self.runner = _FakeRunner()
        self.lock = threading.Lock()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0 = 0.0

    def head_active(self):
        return "exact"

    def generate(self, req, cancel=None):
        # This is the old failure path: a malformed request reached the engine and raised a TypeError.
        if not isinstance(req.get("messages"), list):
            raise TypeError("messages must be a list")
        for m in req["messages"]:
            if not isinstance(m, dict) or "role" not in m:
                raise TypeError("missing role")
        cached = len(self.runner.tokens)
        yield "done", {
            "content": "ok", "tool_calls": [], "reasoning": "", "finish": "stop", "logprobs": None,
            "usage": {"prompt_tokens": cached, "completion_tokens": 1, "total_tokens": cached + 1,
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0},
        }


@pytest.fixture
def app_client(monkeypatch):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": ""}, log=lambda *a: None)
    with TestClient(app) as client:
        yield client, K


@pytest.fixture
def auth_client(monkeypatch):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "secret-token"}, log=lambda *a: None)
    with TestClient(app) as client:
        yield client, K


MALFORMED = [
    {"messages": "not a list"},
    {"messages": [{"content": "no role"}]},
    {"messages": [{"role": "user", "content": "hello"}], "stop": 3},
    {"messages": [{"role": "user", "content": "hello"}], "top_k": "20"},
    {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 0},
    {"messages": [{"role": "user", "content": "hello"}], "stream": "yes"},
]


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_request_is_400_and_keeps_the_prompt_cache(app_client, payload):
    client, K = app_client
    bad = client.post("/v1/chat/completions", json=payload)
    assert bad.status_code == 400, bad.text
    assert K.runner.eng.resets == 0           # _after_engine_error was not called
    assert K.runner.tokens == [1, 2, 3]
    assert K.runner.ckpt

    good = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello"}]})
    assert good.status_code == 200, good.text
    assert good.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert K.runner.eng.resets == 0


def test_nan_sampling_value_is_400_and_keeps_the_prompt_cache(app_client):
    client, K = app_client
    raw = b'{"messages":[{"role":"user","content":"hello"}],"temperature":NaN}'
    bad = client.post("/v1/chat/completions", content=raw,
                      headers={"content-type": "application/json"})
    assert bad.status_code == 400, bad.text
    assert "finite" in bad.json()["error"]["message"]
    assert K.runner.eng.resets == 0
    assert K.runner.tokens == [1, 2, 3]

    good = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello"}]})
    assert good.status_code == 200, good.text
    assert good.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


def test_invalid_streaming_request_is_400_before_sse(app_client):
    client, K = app_client
    bad = client.post("/v1/chat/completions", json={"messages": "not a list", "stream": True})
    assert bad.status_code == 400, bad.text
    assert K.runner.eng.resets == 0
    assert K.runner.tokens == [1, 2, 3]


def test_api_key_is_required_and_matches_exactly(auth_client):
    client, _ = auth_client
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_oversized_body_is_413_and_keeps_the_prompt_cache(app_client, monkeypatch):
    client, K = app_client
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 512)
    raw = b'{"messages":[{"role":"user","content":"' + b"x" * 1024 + b'"}]}'
    bad = client.post("/v1/chat/completions", content=raw,
                      headers={"content-type": "application/json"})
    assert bad.status_code == 413, bad.text
    assert bad.json()["error"]["type"] == "request_too_large"
    assert K.runner.eng.resets == 0
    assert K.runner.tokens == [1, 2, 3]

    good = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello"}]})
    assert good.status_code == 200, good.text
    assert good.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
