# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the llama.cpp proxy backend (REQ-0019 T4.4).

No llama-server is needed: the backend takes an injectable opener (the same
pattern as kindi.intel's runner), so the request shape, the event stream and
every failure mode are pinned CPU-only. The live proxy gate — a real
llama-server behind `kindi verify` — is recorded as open in REQ-0019.
"""
import json

import pytest

from kindi.backends import registry
from kindi.backends.base import CancelToken, SamplingParams
from kindi.backends.llamacpp import LlamaCppBackend


class _Recorder(object):
    """An opener that records requests and replays canned responses."""

    def __init__(self, replies):
        self.replies = dict(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies.get((request["method"], request["url"]))
        if isinstance(reply, Exception):
            raise reply
        return reply


def _backend(replies, **kwargs):
    opener = _Recorder(replies)
    backend = LlamaCppBackend(opener=opener, **kwargs)
    return backend, opener


def _loaded(replies=None, **kwargs):
    replies = replies or {}
    backend, opener = _backend(replies, **kwargs)
    backend.load({"base_url": "http://127.0.0.1:8080"})
    return backend, opener


def test_load_resolves_base_url_from_config_env_and_default(monkeypatch):
    monkeypatch.delenv("KINDI_LLAMACPP_BASE_URL", raising=False)
    monkeypatch.delenv("KINDI_LLAMACPP_API_KEY", raising=False)
    backend, _ = _backend({})
    backend.load({})
    assert backend.base_url == "http://127.0.0.1:8080"
    assert backend.api_key == ""

    monkeypatch.setenv("KINDI_LLAMACPP_BASE_URL", "http://lan:9090/")
    monkeypatch.setenv("KINDI_LLAMACPP_API_KEY", "secret")
    backend, _ = _backend({})
    backend.load({})
    assert backend.base_url == "http://lan:9090"  # trailing slash stripped
    assert backend.api_key == "secret"

    backend, _ = _backend({})
    backend.load({"base_url": "http://other:1", "api_key": "k", "model": "m"})
    assert (backend.base_url, backend.api_key, backend.model) == ("http://other:1", "k", "m")


def test_load_records_an_unreachable_server_without_raising():
    """load() must not fail when llama-server is down — generate() is the gate,
    and the contract harness calls load({}) on every registered backend."""
    down = _Recorder({("GET", "http://127.0.0.1:8080/health"):
                      registry.BackendUnavailable("cannot reach llama-server at ...")})
    backend = LlamaCppBackend(opener=down)
    backend.load({})
    assert backend.loaded is True
    assert backend.reachable is False
    assert "cannot reach" in backend.health_error
    assert backend.info()["reachable"] is False


def test_generate_posts_the_prompt_and_streams_the_reply_as_events():
    replies = {("POST", "http://127.0.0.1:8080/completion"):
               {"content": "Hello!", "truncated": False, "tokens_predicted": 6}}
    backend, opener = _loaded(replies)
    ids = backend.tokenize("Say hello", add_special_tokens=True)
    events = list(backend.generate(ids, SamplingParams(max_tokens=16, temperature=0.0)))
    assert [e.kind for e in events][-1] == "done"
    assert events[-1].finish_reason == "stop"
    assert "".join(e.text for e in events if e.kind == "token") == "Hello!"
    request = [r for r in opener.requests if r["method"] == "POST"][0]
    assert request["body"]["prompt"] == "Say hello"
    assert request["body"]["n_predict"] == 16
    assert request["body"]["temperature"] == 0.0
    assert request["body"]["cache_prompt"] is True
    assert request["body"]["stream"] is False


def test_generate_maps_truncated_to_length_and_applies_stop_strings_locally():
    replies = {("POST", "http://127.0.0.1:8080/completion"):
               {"content": "answer END tail", "truncated": True, "tokens_predicted": 3}}
    backend, _ = _loaded(replies)
    events = list(backend.generate([1], SamplingParams(max_tokens=3, stop=("END",))))
    assert "".join(e.text for e in events if e.kind == "token") == "answer "
    assert events[-1].finish_reason == "stop"  # the stop wins over truncated

    replies = {("POST", "http://127.0.0.1:8080/completion"):
               {"content": "abc", "truncated": True, "tokens_predicted": 3}}
    backend, _ = _loaded(replies)
    events = list(backend.generate([1], SamplingParams(max_tokens=3)))
    assert events[-1].finish_reason == "length"


def test_generate_refuses_a_token_filter_loudly():
    backend, _ = _loaded()
    with pytest.raises(registry.BackendUnavailable, match="token_filter"):
        list(backend.generate([1], SamplingParams(max_tokens=1,
                                                  token_filter=lambda state: {1})))


def test_generate_before_load_and_a_dead_server_are_named_errors():
    backend, _ = _backend({})
    with pytest.raises(registry.BackendUnavailable, match="not loaded"):
        list(backend.generate([1], SamplingParams(max_tokens=1)))

    replies = {("POST", "http://127.0.0.1:8080/completion"):
               registry.BackendUnavailable(
                   "cannot reach llama-server at http://127.0.0.1:8080/completion (refused); "
                   "start it with `llama-server -m <model.gguf>`")}
    backend, _ = _loaded(replies)
    with pytest.raises(registry.BackendUnavailable, match="llama-server -m"):
        list(backend.generate([1], SamplingParams(max_tokens=1)))


def test_generate_rejects_an_unexpected_reply_shape():
    replies = {("POST", "http://127.0.0.1:8080/completion"): {"error": "boom"}}
    backend, _ = _loaded(replies)
    with pytest.raises(registry.BackendUnavailable, match="unexpected reply"):
        list(backend.generate([1], SamplingParams(max_tokens=1)))


def test_generate_honours_a_cancelled_token_before_the_request():
    backend, _ = _loaded({("POST", "http://127.0.0.1:8080/completion"):
                          {"content": "x", "truncated": False}})
    cancel = CancelToken()
    cancel.set()
    events = list(backend.generate([1], SamplingParams(max_tokens=1), cancel=cancel))
    assert events[-1].finish_reason == "cancelled"
    assert not [e for e in events if e.kind == "token"]


def test_text_primitives_are_a_byte_round_trip_and_the_template_is_chatml():
    backend = LlamaCppBackend(opener=lambda request: {})
    prompt = backend.apply_chat_template([{"role": "user", "content": "hello"}])
    assert prompt == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    ids = backend.tokenize(prompt)
    assert backend.detokenize(ids) == prompt
    assert backend.tokenize("é", add_special_tokens=False) == [0xC3 + 1, 0xA9 + 1]


def test_the_proxy_is_the_wildcard_and_specific_backends_still_win():
    proxy = registry.resolve("some-arbitrary-gguf-model")
    assert isinstance(proxy, LlamaCppBackend)
    from kindi.backends.fake import FakeBackend
    assert isinstance(registry.resolve("fake"), FakeBackend)
    assert registry.resolve("bonsai2-2b").capabilities.name == "bonsai"
    assert "llamacpp" in [row["name"] for row in registry.list_backends()]


def test_capabilities_refuse_what_the_proxy_cannot_do():
    caps = LlamaCppBackend.capabilities
    assert caps.supports_json_grammar is False
    assert caps.supports_token_filter is False
    assert caps.supports_tools is False
    assert caps.cache_kind == "none"
    assert caps.model_ids == () and caps.architectures == ()
    assert caps.to_dict()["name"] == "llamacpp"
    assert LlamaCppBackend().verify_prompts() == []


def test_the_default_opener_sends_a_bearer_token_and_json(monkeypatch):
    """The urllib opener is exercised against a stubbed urlopen: shape only."""
    import urllib.request

    seen = {}

    class _Response(object):
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Response({"content": "ok", "truncated": False})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = LlamaCppBackend()
    backend.load({"base_url": "http://127.0.0.1:8080", "api_key": "k"})
    events = list(backend.generate(backend.tokenize("hi"), SamplingParams(max_tokens=4)))
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["prompt"] == "hi"
    assert "".join(e.text for e in events if e.kind == "token") == "ok"
