"""HTTP boundary tests for malformed requests, API-key enforcement, body limits and prompt-cache preservation.

These tests use a stub Kindi, so they need no GPU, model pack or tokenizer. The stub deliberately raises
TypeError for a malformed message list, which is what the old generic exception path turned into a 500 after
clearing the real Runner. The validator must stop the request before that path is reached.
"""
import json
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
        self.pack_id = "fake"
        self.last_req = None
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
        self.last_req = req
        yield "text", "ok"
        yield "done", {
            "content": "ok", "tool_calls": [], "reasoning": "", "finish": "stop", "logprobs": None,
            "usage": {"prompt_tokens": cached, "completion_tokens": 1, "total_tokens": cached + 1,
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0},
        }


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    pack = tmp_path / "text-pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b"}', encoding="utf-8")
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(pack)}, log=lambda *a: None)
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


def test_queue_gate_bounds_the_wait():
    gate = server.Gate(1)
    gate.enter()
    gate.enter()
    assert gate.waiting == 1
    with pytest.raises(server.QueueFull):
        gate.enter()
    assert gate.rejected == 1
    gate.release()
    gate.enter()                       # a freed slot is reusable
    assert gate.admitted == 2
    gate.release()
    gate.release()
    assert gate.admitted == 0          # release never underflows


def test_full_queue_is_429_with_retry_after(monkeypatch):
    class _FullGate(object):
        max_queue = 8
        rejected = 0
        waiting = 0

        def enter(self):
            raise server.QueueFull("the server is busy: 8 request(s) queued (KINDI_QUEUE_MAX=8)")

        def release(self):
            pass

    monkeypatch.setattr(server, "Gate", lambda n: _FullGate())
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": ""}, log=lambda *a: None)
    with TestClient(app) as client:
        for path, payload in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                              ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                       "stream": True}),
                              ("/api/generate", {"prompt": "hi"})):
            r = client.post(path, json=payload)
            assert r.status_code == 429, r.text
            assert r.headers["retry-after"] == "1"
            assert r.json()["error"]["type"] == "server_busy"
    assert K.last_req is None               # rejected before the engine ran


def test_metrics_endpoint_reports_the_model_and_queue(app_client):
    client, _ = app_client
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "kindi_up 1" in body
    assert 'kindi_model_info{model="bonsai2-2b",ctx="4096",head="exact"} 1' in body
    assert "kindi_queue_max %d" % server.MAX_QUEUE in body
    assert "kindi_queue_depth 0" in body
    assert "kindi_requests_rejected_total 0" in body


def test_metrics_and_ollama_endpoints_need_the_api_key(auth_client):
    client, _ = auth_client
    for path in ("/metrics", "/api/tags"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_ollama_tags_lists_the_model(app_client):
    client, _ = app_client
    r = client.get("/api/tags")
    assert r.status_code == 200, r.text
    model = r.json()["models"][0]
    assert model["name"] == "bonsai2-2b" and model["model"] == "bonsai2-2b"
    assert model["details"]["format"] == "kindi"


def test_ollama_chat_streams_ndjson(app_client):
    client, _ = app_client
    r = client.post("/api/chat", json={"model": "bonsai2-2b",
                                       "messages": [{"role": "user", "content": "hi"}],
                                       "stream": True})
    assert r.status_code == 200, r.text
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events[0] == {"model": "bonsai2-2b", "created_at": events[0]["created_at"], "done": False,
                         "message": {"role": "assistant", "content": "ok"}}
    assert events[-1]["done"] is True and events[-1]["eval_count"] == 1
    assert events[-1]["message"]["content"] == ""


def test_ollama_generate_maps_prompt_and_options(app_client):
    client, K = app_client
    r = client.post("/api/generate", json={"prompt": "hello", "stream": False,
                                           "options": {"temperature": 0, "num_predict": 7}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"] == "ok" and body["done"] is True
    assert body["prompt_eval_count"] == 3 and body["eval_count"] == 1
    assert K.last_req["messages"] == [{"role": "user", "content": "hello"}]
    assert K.last_req["temperature"] == 0 and K.last_req["max_tokens"] == 7


def test_ollama_malformed_messages_is_400_before_the_engine(app_client):
    client, K = app_client
    r = client.post("/api/chat", json={"messages": "nope"})
    assert r.status_code == 400, r.text
    assert K.last_req is None


@pytest.mark.parametrize("stream", (False, True))
def test_image_part_is_400_without_vision_weights(app_client, stream):
    client, K = app_client
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    }
    if stream:
        payload["stream"] = True
    bad = client.post("/v1/chat/completions", json=payload)
    assert bad.status_code == 400, bad.text
    assert "no vision weights" in bad.json()["error"]["message"]
    assert K.last_req is None
    assert K.runner.eng.resets == 0
    assert K.runner.tokens == [1, 2, 3]

    good = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello"}]})
    assert good.status_code == 200, good.text
    assert good.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


def test_ollama_images_are_400_without_vision_weights(app_client):
    client, K = app_client
    cases = (
        ("/api/chat", {"messages": [{"role": "user", "content": "describe",
                                      "images": ["AAAA"]}], "stream": True}),
        ("/api/generate", {"prompt": "describe", "images": ["AAAA"], "stream": False}),
    )
    for path, payload in cases:
        r = client.post(path, json=payload)
        assert r.status_code == 400, (path, r.text)
        assert "no vision weights" in r.json()["error"]["message"]
        assert K.last_req is None
        assert K.runner.eng.resets == 0
        assert K.runner.tokens == [1, 2, 3]


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
