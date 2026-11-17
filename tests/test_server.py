"""HTTP boundary tests for malformed requests, API-key enforcement, body limits and prompt-cache preservation.

These tests use a stub Kindi, so they need no GPU, model pack or tokenizer. The stub deliberately raises
TypeError for a malformed message list, which is what the old generic exception path turned into a 500 after
clearing the real Runner. The validator must stop the request before that path is reached.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gbnf_fixtures import CANONICAL_JSON
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
    supports_logprobs = False

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
        self.calls = 0
        self.t0 = 0.0

    def head_active(self):
        return "exact"

    def encode_text(self, text, special=True):
        return [ord(c) % 256 for c in str(text)] or [1]   # one id per character, like _StopTok

    def decode_ids(self, ids):
        return "x" * len(ids)

    def after_turn(self, locked=False):
        return None

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def generate(self, req, cancel=None):
        # This is the old failure path: a malformed request reached the engine and raised a TypeError.
        if not isinstance(req.get("messages"), list):
            raise TypeError("messages must be a list")
        for m in req["messages"]:
            if not isinstance(m, dict) or "role" not in m:
                raise TypeError("missing role")
        cached = len(self.runner.tokens)
        self.calls += 1
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
    {"messages": [{"role": "user", "content": "hello"}], "presence_penalty": 2.5},
    {"messages": [{"role": "user", "content": "hello"}], "presence_penalty": -2.5},
    {"messages": [{"role": "user", "content": "hello"}], "presence_penalty": "1"},
    {"messages": [{"role": "user", "content": "hello"}], "n": 2},
    {"messages": [{"role": "user", "content": "hello"}], "n": "2"},
    {"messages": [{"role": "user", "content": "hello"}], "n": 0},
    {"messages": [{"role": "user", "content": "hello"}], "response_format": {"type": "grammar"}},
    {"messages": [{"role": "user", "content": "hello"}], "response_format": {"type": "json_schema"}},
    {"messages": [{"role": "user", "content": "hello"}], "response_format": {"type": "json_schema", "json_schema": {"schema": "x"}}},
    {"messages": [{"role": "user", "content": "hello"}], "tools": [{"type": "function", "function": {"name": "f"}}], "response_format": {"type": "json_object"}},
    {"messages": [{"role": "user", "content": "hello"}], "response_format": "json_object"},
]


def test_unsupported_fields_name_their_param(app_client):
    client, _ = app_client
    base = {"messages": [{"role": "user", "content": "hello"}]}
    for payload, param in (
            (dict(base, n=2), "n"),
            (dict(base, response_format={"type": "grammar"}), "response_format"),
            (dict(base, response_format={"type": "json_schema"}), "response_format"),
            (dict(base, response_format={"type": "json_schema",
                                        "json_schema": {"schema": {"type": "frob"}}}),
             "response_format"),
            (dict(base, response_format={"type": "grammar",
                                        "grammar": 'root ::= "a"{1,100000}'}),
             "response_format"),                     # bounded compilation, not a memory request
            (dict(base, response_format={"type": "json_object"}, tools=[{"type": "function", "function": {"name": "f"}}]),
             "response_format")):
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == param, r.text


class _StopTok(object):
    """A character codec for driving the real generate() loop without a model."""

    vocab = ["v%d" % i for i in range(256)]

    def encode(self, s):
        return [ord(c) % 256 for c in str(s)] or [1]

    def decode_bytes(self, t):
        return bytes([t % 256])


class _StopEngine(object):
    ctx = 4096
    hsl = None
    shortlist = 0


class _StopRunner(object):
    def prepare(self, *a, **k):
        return None, 0

    def step(self, tok):
        return None

    def idle(self):
        return None


def _drive_stop(script, req_extra=None, think=False):
    """Run the real `Kindi.generate` loop with scripted tokens (characters as ords).

    The loop's per-token work (stop detection, think routing, truncation) is the code under
    test; only token generation itself is scripted.
    """
    K = object.__new__(server.Kindi)
    K.cfg = {"model_name": "fake", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
             "max_tokens": 512, "repeat_stop": False, "fast_clusters": 0}
    K.native = False
    K.nanbeige = None
    K.variant = None
    K.tok = _StopTok()
    K.eng = _StopEngine()
    K.runner = _StopRunner()
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
              "read": 0, "generated": 0, "started": 0.0}
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
               "gen_seconds": 0.0, "spec_passes": 0, "spec_drafted": 0, "spec_accepted": 0}
    K.log = lambda *a: None
    K.head_mode = lambda: "exact"
    K.sample = lambda *a, **k: script.pop(0)
    K._pick = lambda *a, **k: script.pop(0)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30,
           "temperature": 0.0, "top_p": 1.0, "top_k": 0}
    if req_extra:
        req.update(req_extra)
    if think:
        req["enable_thinking"] = True
    items = list(K.generate(req))
    done = next(v for kind, v in items if kind == "done")
    return done, items


def test_stop_spans_token_pieces_and_truncates():
    pieces = ["ab", "cdef", "STOP", "ghij"]
    script = [ord(c) for piece in pieces for c in piece]
    done, items = _drive_stop(script, req_extra={"stop": ["cdefSTOP"]})
    assert done["finish"] == "stop"
    assert done["content"] == "ab"


def test_stop_seen_while_thinking_still_stops_after_think_closes():
    # the stop check is skipped while thinking; a match inside the thinking text must
    # still be found by the first check after </think> closes (frozen stop_from)
    pieces = ["STOP"] + ["<"] + ["/"] + ["t", "h", "i", "n", "k", ">"] + ["abc", "def"]
    script = [ord(c) for piece in pieces for c in piece]
    done, items = _drive_stop(script, req_extra={"stop": ["STOP"]}, think=True)
    assert done["finish"] == "stop"


def test_json_schema_and_object_response_formats_are_accepted(app_client):
    client, _ = app_client
    for rf in ({"type": "json_object"},
               {"type": "json_schema", "json_schema": {"name": "x", "schema": {
                   "type": "object", "properties": {"a": {"type": "string"}},
                   "required": ["a"]}}},
               {"type": "grammar", "grammar": 'root ::= "ok" | "no"'}):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": rf,
        })
        assert r.status_code == 200, r.text


RECURSIVE_GRAMMARS = [
    'root ::= "[" root "]" | "x"',        # the smallest self-recursive grammar
    CANONICAL_JSON,                        # llama.cpp's own json.gbnf, the one people send
]


def _grammar_request(grammar):
    return {"messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "grammar", "grammar": grammar}}


def test_the_recursion_opt_in_is_a_400_by_default_and_a_200_when_enabled(app_client, monkeypatch):
    """The env switch end to end, through HTTP: the same recursive grammar is refused by the
    validator (400, `response_format`, no engine call) until KINDI_GBNF_MAX_DEPTH is set, and
    then the same request is accepted. The compiler tests prove the language; this proves the
    operator's switch reaches it."""
    client, K = app_client
    monkeypatch.delenv("KINDI_GBNF_MAX_DEPTH", raising=False)
    for grammar in RECURSIVE_GRAMMARS:
        r = client.post("/v1/chat/completions", json=_grammar_request(grammar))
        assert r.status_code == 400, r.text
        error = r.json()["error"]
        assert error["param"] == "response_format"
        assert "recursive" in error["message"]
    assert K.calls == 0                     # refused in validation, before the engine ran

    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", "3")
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "60000")   # depth 3 costs 21,252 states
    for grammar in RECURSIVE_GRAMMARS:
        r = client.post("/v1/chat/completions", json=_grammar_request(grammar))
        assert r.status_code == 200, r.text
    assert K.calls == len(RECURSIVE_GRAMMARS)


def test_the_opt_in_reply_is_constrained_through_http(app_client, monkeypatch):
    """The opt-in end to end: the same recursive grammar is an enveloped 400 by default, and
    with KINDI_GBNF_MAX_DEPTH set the request is accepted *and the reply stays in the unrolled
    language* - the script offers a byte the grammar forbids and it is never emitted."""
    client, K = app_client
    doc = '{"a": 1}'
    # the script offers \x01 in the middle of a valid document. That byte is in no part of
    # the canonical grammar (not ws, not a key, not a number), so it can never become legal
    # later the way a skipped `}` could; if the constraint works the reply is exactly `doc`.
    script = [ord(c) for c in doc[:6]] + [1] + [ord(c) for c in doc[6:]] + [248046]

    def post(grammar=None):
        K.generate = _constrained_kindi(list(script)).generate   # a fresh script per request
        body = _grammar_request(grammar) if grammar else {
            "messages": [{"role": "user", "content": "hello"}]}
        return client.post("/v1/chat/completions", json=body)

    monkeypatch.delenv("KINDI_GBNF_MAX_DEPTH", raising=False)
    r = post(CANONICAL_JSON)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"

    # the control: the same scripted tokens without a grammar do emit \x01, so the
    # equality below is the constraint's doing and not the script's
    content = post().json()["choices"][0]["message"]["content"]
    assert "\x01" in content, repr(content)

    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", "3")
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "60000")
    r = post(CANONICAL_JSON)
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    assert content == doc, content          # the forbidden byte never reached the reply

    from kindi.gbnf import Grammar
    g = Grammar.from_string(CANONICAL_JSON)
    assert g.done(g.advance(g.start(), content))     # and the reply is in the language


def _constrained_kindi(script):
    """A real `Kindi` whose tokens come from `script`: every pick must be inside the
    sampler's allowed set, so a token the constraint forbids is never emitted."""
    K = object.__new__(server.Kindi)
    K.cfg = {"model_name": "fake", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
             "max_tokens": 512, "repeat_stop": False, "fast_clusters": 0}
    K.native = False
    K.nanbeige = None
    K.variant = None
    K.tok = _StopTok()
    K.eng = _StopEngine()
    K.runner = _StopRunner()
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
              "read": 0, "generated": 0, "started": 0.0}
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
               "gen_seconds": 0.0, "spec_passes": 0, "spec_drafted": 0, "spec_accepted": 0}
    K.log = lambda *a: None
    K.head_mode = lambda: "exact"

    def pick(*a, **k):
        allowed = set(k.get("allowed").tolist()) if k.get("allowed") is not None else None
        for i, t in enumerate(script):
            if allowed is None or t in allowed:
                script.pop(i)
                return t
        raise AssertionError("no scripted token fits the grammar")

    K.sample = pick
    K._pick = pick
    return K


def _drive_constrained(script, schema=None, response_format=None):
    """Drive the real `Kindi.generate` loop with a grammar: every scripted pick must be
    inside the pick's allowed set, so a stray token would raise here."""
    K = _constrained_kindi(script)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 512,
           "temperature": 0.0, "top_p": 1.0, "top_k": 0,
           "response_format": response_format or {
               "type": "json_schema", "json_schema": {"schema": schema}}}
    items = list(K.generate(req))
    done = next(v for kind, v in items if kind == "done")
    return done


def test_constrained_generate_produces_schema_valid_json():
    schema = {"type": "object",
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
              "required": ["name", "age"]}
    doc = '{"name": "Ada", "age": 36}'
    script = [ord(c) for c in doc] + [248046]  # 248046 = <|im_end|>
    done = _drive_constrained(list(script), schema)
    assert done["finish"] == "stop"
    assert json.loads(done["content"]) == {"name": "Ada", "age": 36}


def test_grammar_response_format_generates_inside_the_language():
    """A GBNF response_format is decode-side constraint on the dense path too, not a
    post-hoc check: `no` is in the language, `nope` cannot be produced."""
    rf = {"type": "grammar", "grammar": 'root ::= "yes" | "no"'}
    done = _drive_constrained([ord(c) for c in "no"] + [248046], response_format=rf)
    assert done["finish"] == "stop"
    assert done["content"] == "no"

    # the out-of-language tail is never emitted: the script asks for 'p' next, the
    # allowed set does not contain it, so the loop falls through to EOS and stops
    done = _drive_constrained([ord(c) for c in "nope"] + [248046], response_format=rf)
    assert done["finish"] == "stop"
    assert done["content"] == "no"


def test_nesting_past_the_unrolled_depth_is_a_legal_shallower_reply(monkeypatch):
    """The audit predicted a 500 when the GBNF depth budget runs out. The budget is spent when
    the grammar is compiled (recursion is unrolled and the deeper copies are pruned), so at run
    time an over-deep token is simply not in the allowed set: the reply stays inside the
    language, comes out shallower, and finishes normally. `json_schema` remains the exact
    recursive path; this only pins that the bounded one degrades instead of erroring."""
    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", "2")
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "200000")
    rf = {"type": "grammar",
          "grammar": 'root ::= value\nvalue ::= "{" "\\"a\\"" ":" value "}" | "0"'}
    script = [ord(c) for c in '{"a":{"a":{"a":0}}}'] + [248046]
    done = _drive_constrained(script, response_format=rf)
    assert done["finish"] == "stop"
    doc, depth = json.loads(done["content"]), 0
    while isinstance(doc, dict):
        depth, doc = depth + 1, doc["a"]
    assert depth <= 2


def test_constrained_generate_rejects_a_grammar_escape():
    # without the constraint these would pass the sampler; under it, 'x' for a string
    # start can never be picked (the pick helper would find nothing fitting)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    script = [ord('{'), ord('x')]  # a value-start 'x' is not grammar-legal inside an object
    with pytest.raises(AssertionError):
        _drive_constrained(script, schema)


class _G4Tok(object):
    """A Gemma-style tokenizer stand-in: decoding is token text, stop set has one EOS."""

    def __init__(self, texts, eos):
        self.tokens = list(texts)
        self.stop_ids = {eos}
        self.ids = {}
        self.eos = eos

    def encode(self, s):
        return [1]

    def decode_bytes(self, i):
        return self.tokens[int(i)].encode("utf-8")


class _G4Engine(object):
    ctx = 4096
    shortlist = 0
    fast_centroids = None

    def __init__(self, script):
        self.script = list(script)

    def reset(self):
        pass

    def forward_batch(self, ids, pos):
        pass

    def forward(self, tok, pos, logits=False):
        return self.script.pop(0)


class _G4Cache(object):
    def prepare(self, ids):
        return 0

    def advanced(self, ids, checkpoint=False):
        pass

    def append(self, tok):
        pass


class _Np(object):
    """Small cupy stand-in: _generate_gemma only touches these numpy-backed calls."""

    asnumpy = staticmethod(np.asarray)
    float32 = np.float32

    def __getattr__(self, name):
        return getattr(np, name)


def _drive_gemma(monkeypatch, logits_script, texts, eos, req_extra=None):
    """Drive the real `_generate_gemma` loop without a GPU or a Gemma pack."""
    from kindi import gemma4_chat

    monkeypatch.setattr(gemma4_chat, "render", lambda *a, **k: "prompt")
    K = object.__new__(server.Kindi)
    K.cfg = {"model_name": "fake-gemma", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
             "max_tokens": 64, "fast_clusters": 0, "head": "exact"}
    K.native = True
    K.variant = "gemma4"
    K.tok = _G4Tok(texts, eos)
    K.eng = _G4Engine(logits_script)
    K.g4cache = _G4Cache()
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
              "read": 0, "generated": 0, "started": 0.0}
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
               "gen_seconds": 0.0, "last_tok_s": 0.0}
    K.log = lambda *a: None
    K.cp = _Np()
    K.head_mode = lambda: "exact"
    K.model_path = ""
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
           "temperature": 0.0, "top_p": 1.0, "top_k": 0}
    if req_extra:
        req.update(req_extra)
    items = list(K._generate_gemma(req))
    done = next(v for kind, v in items if kind == "done")
    return done


def _logits(n, peak, second=None):
    z = np.full(n, -10.0)
    z[peak] = 100.0
    if second is not None:
        z[second] = 50.0
    return z


def test_gemma_json_schema_is_constrained(monkeypatch):
    # the top logit is always on '<' (never grammar-legal); the pick must skip it
    texts = ["{", '"ok"', ":", "true", "}", "<", "!", "<eos>"]
    script = [_logits(8, 5, 0), _logits(8, 5, 1), _logits(8, 5, 2),
              _logits(8, 5, 3), _logits(8, 5, 4), _logits(8, 7)]
    done = _drive_gemma(monkeypatch, script, texts, eos=7, req_extra={
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}},
    })
    assert done["finish"] == "stop"
    assert json.loads(done["content"]) == {"ok": True}


def test_gemma_without_response_format_is_unconstrained(monkeypatch):
    # the same logits with no response_format: the illegal top pick goes through verbatim
    texts = ["{", '"ok"', ":", "true", "}", "<", "!", "<eos>"]
    done = _drive_gemma(monkeypatch, [_logits(8, 5), _logits(8, 7)], texts, eos=7)
    assert done["content"] == "<"


def test_structured_output_rejected_on_unsupported_backend(app_client):
    client, K = app_client
    K.supports_json_grammar = False
    K.variant = "dsv41"
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
    })
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"


def test_default_n_and_text_response_format_are_accepted(app_client):
    client, _ = app_client
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "n": 1,
        "response_format": {"type": "text"},
    })
    assert r.status_code == 200, r.text


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


def test_logprobs_are_rejected_on_a_backend_that_does_not_support_them(app_client):
    client, K = app_client
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}], "logprobs": True})
    assert r.status_code == 400, r.text
    assert "logprobs are not supported" in r.json()["error"]["message"]
    assert K.last_req is None


def test_non_loopback_without_api_key_warns(monkeypatch):
    logs = []
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: _FakeKindi())
    server.create_app({"model_name": "bonsai2-2b", "api_key": "", "host": "0.0.0.0"},
                      log=logs.append)
    assert any("without api_key" in message and "0.0.0.0" in message for message in logs)


def _access_app(monkeypatch, tmp_path, **cfg):
    pack = tmp_path / "text-pack"
    pack.mkdir(exist_ok=True)
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b"}', encoding="utf-8")
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda c, log: K)
    lines = []
    app = server.create_app(dict({"model_name": "bonsai2-2b", "api_key": "", "pack": str(pack)}, **cfg),
                            log=lines.append)
    return app, K, lines


def _last_access_line(lines):
    parsed = [json.loads(line) for line in lines if line.startswith("{")]
    assert parsed, lines
    return parsed[-1]


def test_json_access_log_records_a_completion(monkeypatch, tmp_path):
    app, _K, lines = _access_app(monkeypatch, tmp_path, log_json=1)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hello"}]},
                        headers={"X-Request-ID": "test-request-1"})
    assert r.status_code == 200, r.text
    assert r.headers["x-request-id"] == "test-request-1"
    line = _last_access_line(lines)
    assert line["request_id"] == "test-request-1"
    assert line["method"] == "POST" and line["path"] == "/v1/chat/completions"
    assert line["status"] == 200 and line["stream"] is False
    assert line["prompt_tokens"] == 3 and line["completion_tokens"] == 1
    assert line["cached_tokens"] == 3
    assert isinstance(line["duration_ms"], float) and isinstance(line["queue_wait_ms"], float)


def test_json_access_log_records_a_rejected_request(monkeypatch, tmp_path):
    app, K, lines = _access_app(monkeypatch, tmp_path, log_json=1)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"messages": "not a list"})
    assert r.status_code == 400, r.text
    line = _last_access_line(lines)
    assert line["status"] == 400
    assert "messages must be a non-empty list" in line["error"]
    assert K.last_req is None


def test_json_access_log_records_a_stream(monkeypatch, tmp_path):
    app, _K, lines = _access_app(monkeypatch, tmp_path, log_json=1)
    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hello"}],
                                 "stream": True}) as r:
            assert r.status_code == 200, r.text
            body = "".join(r.iter_text())
    assert "[DONE]" in body
    line = _last_access_line(lines)
    assert line["stream"] is True and line["status"] == 200
    assert line["completion_tokens"] == 1


def test_json_access_log_is_off_by_default(monkeypatch, tmp_path):
    app, _K, lines = _access_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200, r.text
    assert [line for line in lines if line.startswith("{")] == []


@pytest.mark.parametrize(("name", "value"), (
    ("KINDI_MAX_BODY_MB", "not-a-number"),
    ("KINDI_MAX_BODY_MB", "0"),
    ("KINDI_QUEUE_MAX", "-1"),
))
def test_invalid_server_limit_environment_is_a_clear_startup_error(name, value):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), **{name: value})
    proc = subprocess.run([sys.executable, "-c", "import kindi.server"], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert proc.returncode != 0
    assert name in proc.stderr.decode("utf-8", "replace")


def test_api_key_is_required_and_matches_exactly(auth_client):
    client, _ = auth_client
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_health_hides_details_without_the_api_key(auth_client):
    client, _ = auth_client
    public = client.get("/health")
    assert public.status_code == 200, public.text
    assert public.json() == {"ok": True, "kindi": server.__version__, "commit": server.COMMIT,
                             "model": "bonsai2-2b", "ctx": 4096}
    full = client.get("/health", headers={"Authorization": "Bearer secret-token"})
    assert full.status_code == 200, full.text
    assert full.json()["queue"]["waiting"] == 0
    assert "gpu_free_mib" in full.json() and "stats" in full.json()


def test_health_without_a_key_keeps_the_detailed_view(app_client):
    client, _ = app_client
    body = client.get("/health").json()
    assert body["ok"] is True and body["model"] == "bonsai2-2b"
    assert body["queue"]["max"] == server.MAX_QUEUE
    assert "gpu_free_mib" in body and "uptime_s" in body


class _SpecEngine(_FakeEngine):
    """The two states REQ-0024 is about: an MTP head that attached, and one the loader refused."""

    def __init__(self, loaded):
        super(_SpecEngine, self).__init__()
        self.spec_begin = lambda *a, **k: None
        self.drafter = None
        self.mtp = object() if loaded else None
        self._mtp_heads = {"mtp": ({"file": "kindi-mtp-2b-v1.npz", "state": "loaded", "registered": True,
                                   "sha256": "1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520",
                                   "bytes": 60960552} if loaded else {
                                       "file": "kindi-mtp-2b-v1.npz", "state": "refused",
                                       "reason": "the MTP head /heads/kindi-mtp-2b-v1.npz does not match "
                                                 "kindi/mtp_heads.json: expected sha256 1b5c0c790a59, "
                                                 "60960552 bytes; found sha256 000000000000, 12 bytes"})}


class _SpecKindi(_FakeKindi):
    """The real `Kindi.drafting` on a stub engine: /health must report what attached, not KINDI_SPEC."""
    drafting = server.Kindi.drafting
    native = False
    nanbeige = None

    def __init__(self, loaded):
        super(_SpecKindi, self).__init__()
        self.eng = _SpecEngine(loaded)


@pytest.mark.parametrize("loaded", [True, False], ids=["head-loaded", "head-refused"])
def test_health_reports_the_effective_speculation_state(monkeypatch, loaded):
    """REQ-0024 A1: KINDI_SPEC=mtp is the request; /health answers what is running. With the head
    attached it names the head and its digest, and with the head refused it says disabled and why,
    instead of echoing the requested mode as `enabled: true` for four releases."""
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    K = _SpecKindi(loaded)
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": ""}, log=lambda *a: None)
    with TestClient(app) as client:
        d = client.get("/health").json()["drafting"]
    assert d["requested"] == "mtp"
    if loaded:
        assert d["enabled"] is True and d["method"] == "mtp" and d["reason"] is None
        assert d["head"] == "kindi-mtp-2b-v1.npz" and d["head_sha256"] == "1b5c0c790a59"
    else:
        assert d["enabled"] is False and d["method"] is None
        assert "does not match kindi/mtp_heads.json" in d["reason"]
        assert d["head"] is None and d["head_sha256"] is None


@pytest.mark.parametrize("loaded, requested, expected", [
    (True, "mtp", "mtp"),
    (False, "mtp", "none"),
    (False, "draft2b", "ngram"),
], ids=["head-loaded", "head-refused", "drafter-fallback"])
def test_metrics_report_the_effective_speculation_mode(monkeypatch, loaded, requested, expected):
    """REQ-0024 A2: the scrape names the mode `drafting()` resolved, not KINDI_SPEC. A refused
    head or an absent drafter must not leave an acceptance ratio attributed to the requested mode."""
    monkeypatch.setenv("KINDI_SPEC", requested)
    K = _SpecKindi(loaded)
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": ""}, log=lambda *a: None)
    with TestClient(app) as client:
        body = client.get("/metrics").text
    assert 'kindi_spec_mode_effective{mode="%s"} 1' % expected in body
    if expected != requested:
        assert 'kindi_spec_mode_effective{mode="%s"} 1' % requested not in body


def test_keyless_non_loopback_health_gets_only_liveness_fields(monkeypatch):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "host": "0.0.0.0"},
                            log=lambda *a: None)
    with TestClient(app, client=("192.0.2.1", 12345)) as client:
        body = client.get("/health")
    assert body.status_code == 200, body.text
    assert body.json() == {"ok": True, "kindi": server.__version__, "commit": server.COMMIT,
                           "model": "bonsai2-2b", "ctx": 4096}
    assert "queue" not in body.json() and "gpu_free_mib" not in body.json()


def test_config_limits_are_applied(monkeypatch):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "",
                             "max_body_mb": 1, "queue_max": 2, "rate_limit_per_min": 30},
                            log=lambda *a: None)
    assert app.state.max_body_bytes == 1024 * 1024
    assert K.gate.max_queue == 2


def test_rate_limiter_uses_a_fixed_window_per_client():
    now = [0.0]
    limiter = server.RateLimiter(2, clock=lambda: now[0])
    assert limiter.retry_after("a") == 0
    assert limiter.retry_after("a") == 0
    assert limiter.retry_after("a") == 60
    assert limiter.retry_after("b") == 0          # another source has its own budget
    assert limiter.rejected == 1
    now[0] = 60.0
    assert limiter.retry_after("a") == 0          # the next minute starts a fresh window


def test_rate_limited_generation_is_429_with_retry_after(monkeypatch):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "rate_limit_per_min": 2},
                            log=lambda *a: None)
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
        blocked = client.post("/v1/chat/completions", json=payload)
        assert blocked.status_code == 429, blocked.text
        assert blocked.json()["error"]["type"] == "rate_limit_exceeded"
        assert 1 <= int(blocked.headers["retry-after"]) <= 60
        assert client.get("/health").status_code == 200      # a monitor is never limited
        assert client.get("/v1/models").status_code == 200
        metrics = client.get("/metrics").text
        assert "kindi_rate_limit_per_min 2" in metrics
        assert "kindi_rate_limited_total 1" in metrics
    assert K.calls == 2                                   # the rejected request never reached the engine


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


def test_queue_gate_rejects_concurrent_admissions_without_overcommit():
    """Concurrent contenders cannot over-admit or lose the active slot."""
    gate = server.Gate(0)
    gate.enter()
    results = []
    lock = threading.Lock()

    def contender():
        try:
            gate.enter()
        except server.QueueFull:
            with lock:
                results.append("rejected")
        else:
            with lock:
                results.append("entered")
            gate.release()

    threads = [threading.Thread(target=contender) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["rejected"] * 32
    assert gate.rejected == 32
    assert gate.admitted == 1
    gate.release()
    assert gate.admitted == 0
    assert gate.waiting == 0


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
    assert "kindi_rate_limit_per_min 0" in body
    assert "kindi_rate_limited_total 0" in body
    assert "kindi_cache_errors_total 0" in body


def test_metrics_reports_the_tandem_probe_when_it_is_on(app_client):
    _, K = app_client
    K.eng.timing_summary = lambda: {"tokens": 128, "cpu_ms_per_token": 74.7, "gpu_ms_per_token": 8.1,
                                    "transfer_ms_per_token": 3.09, "head_ms_per_token": 17.6,
                                    "total_ms_per_token": 103.4, "transfers_per_token": 7.0}
    body = app_client[0].get("/metrics").text
    assert "kindi_tandem_tokens_total 128" in body
    assert "kindi_tandem_cpu_ms_per_token 74.700" in body
    assert "kindi_tandem_gpu_ms_per_token 8.100" in body
    assert "kindi_tandem_transfer_ms_per_token 3.090" in body
    assert "kindi_tandem_head_ms_per_token 17.600" in body
    assert "kindi_tandem_total_ms_per_token 103.400" in body
    assert "kindi_tandem_transfers_per_token 7.000" in body


def test_metrics_omits_the_tandem_probe_when_it_is_off(app_client):
    client, _ = app_client
    body = client.get("/metrics").text
    assert "kindi_tandem_tokens_total" not in body
    assert "kindi_tandem_total_ms_per_token" not in body


def test_metrics_reports_the_prompt_cache_telemetry(app_client):
    # the main prompt cache exposes the same telemetry shape as the Gemma 4 cache (D3)
    _, K = app_client
    K.runner.metrics = lambda: {"budget_bytes": 1000, "used_bytes": 400, "peak_bytes": 700,
                                "checkpoints": 2, "parked": 1, "evictions": 3,
                                "evicted_bytes": 300, "checkpoints_dropped": 2, "parked_dropped": 1}
    body = app_client[0].get("/metrics").text
    assert "kindi_cache_budget_bytes 1000" in body
    assert "kindi_cache_used_bytes 400" in body
    assert "kindi_cache_peak_bytes 700" in body
    assert "kindi_cache_checkpoints 2" in body
    assert "kindi_cache_parked 1" in body
    assert "kindi_cache_evictions_total 3" in body
    assert "kindi_cache_evicted_bytes_total 300" in body
    assert "kindi_cache_checkpoints_dropped_total 2" in body
    assert "kindi_cache_parked_dropped_total 1" in body


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


def test_presence_penalty_is_accepted_and_forwarded(app_client):
    client, K = app_client
    good = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello"}], "presence_penalty": 1.5})
    assert good.status_code == 200, good.text
    assert K.last_req["presence_penalty"] == 1.5
    low = client.post("/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "hello"}], "presence_penalty": -2})
    assert low.status_code == 200, low.text
    assert K.last_req["presence_penalty"] == -2


def test_ollama_presence_penalty_maps_to_the_request(app_client):
    client, K = app_client
    r = client.post("/api/generate", json={"prompt": "hello", "stream": False,
                                           "options": {"presence_penalty": 1.2}})
    assert r.status_code == 200, r.text
    assert K.last_req["presence_penalty"] == 1.2


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


def test_an_image_part_is_400_even_when_the_pack_declares_vision(monkeypatch, tmp_path):
    """A manifest flag must not unlock acceptance: no build has a projector to serve it."""
    pack = tmp_path / "vision-pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b", "vision": true}',
                                           encoding="utf-8")
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(pack)},
                            log=lambda *a: None)
    with TestClient(app) as client:
        bad = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}],
        })
    assert bad.status_code == 400, bad.text
    message = bad.json()["error"]["message"]
    assert "projector" in message, message
    assert "mmproj" in message, message
    assert K.last_req is None


def test_a_validated_grammar_is_reused_instead_of_compiled_twice(monkeypatch):
    """Validation compiles the automaton to answer a bad grammar early; generation reuses it."""
    from kindi import gbnf, jsongrammar

    compiled = []
    real_from_string = gbnf.Grammar.from_string.__func__

    def counting_from_string(cls, text):
        compiled.append(text)
        return real_from_string(cls, text)

    monkeypatch.setattr(gbnf.Grammar, "from_string", classmethod(counting_from_string))
    req = {"messages": [{"role": "user", "content": "hello"}],
           "response_format": {"type": "grammar", "grammar": 'root ::= "a"'}}
    server._validate_request(req)
    assert compiled == ['root ::= "a"']
    grammar = server._response_format_grammar(req["response_format"], req)
    assert compiled == ['root ::= "a"'], "the decode path compiled the same grammar again"
    assert grammar is req["_grammar"]

    schemas = []
    real_schema = jsongrammar.grammar_from_schema

    def counting_schema(schema):
        schemas.append(schema)
        return real_schema(schema)

    monkeypatch.setattr(jsongrammar, "grammar_from_schema", counting_schema)
    req = {"messages": [{"role": "user", "content": "hello"}],
           "response_format": {"type": "json_schema", "json_schema": {
               "schema": {"type": "object",
                          "properties": {"a": {"type": "integer"}},
                          "required": ["a"]}}}}
    server._validate_request(req)
    assert len(schemas) == 1
    schema_grammar = server._response_format_grammar(req["response_format"], req)
    assert len(schemas) == 1, "the decode path built the schema automaton again"
    assert schema_grammar is req["_grammar"]


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


def test_the_tokenize_routes_enforce_the_same_body_cap(app_client):
    """`_read_json_body` is the documented 16 MiB cap; the tokenize/detokenize routes used
    `await request.json()`, so a LAN client could make those two allocate an unbounded body."""
    client, K = app_client
    client.app.state.max_body_bytes = 512
    raw = b'{"content":"' + b"x" * 1024 + b'"}'
    for path in ("/v1/tokenize", "/detokenize"):
        bad = client.post(path, content=raw, headers={"content-type": "application/json"})
        assert bad.status_code == 413, (path, bad.text)
        assert bad.json()["error"]["type"] == "request_too_large"

    good = client.post("/v1/tokenize", json={"content": "hello"})
    assert good.status_code == 200, good.text          # the capped reader did not break the route
    assert good.json()["count"] == 5


def test_a_dry_run_that_cannot_count_is_an_error_not_a_fit(app_client):
    """A preflight is acted on, so a counting failure must not answer `fits: true` with 0 tokens:
    the swallowed exception used to make every such run look like it fit."""
    client, K = app_client

    def boom(req):
        raise RuntimeError("cuda fault")

    K.check_prompt = boom
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}], "dry_run": True})
    assert r.status_code == 500, r.text
    assert "could not be counted" in r.json()["error"]["message"]
    assert "fits" not in r.json()


def _shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _message_tokens(req):
    return [ord(ch) for m in req.get("messages") or [] for ch in str(m.get("content") or "")]


class _PrefixKindi(_FakeKindi):
    """A stub whose reply is derived from the shared prefix, not the new suffix."""

    def __init__(self):
        super().__init__()
        self.runner.tokens = []
        self.last_cached = None

    def generate(self, req, cancel=None):
        ids = _message_tokens(req)
        cached = _shared_prefix(self.runner.tokens, ids)
        shared = ids[:cached] if cached else ids
        reply = "reply: " + "".join(chr(c) for c in shared)
        self.runner.tokens = ids
        self.last_req = req
        self.last_cached = cached
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(ids)
        self.stats["cached_tokens"] += cached
        self.stats["completion_tokens"] += 1
        yield "text", reply
        yield "done", {
            "content": reply, "tool_calls": [], "reasoning": "", "finish": "stop", "logprobs": None,
            "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1,
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0},
        }


def test_shared_prefix_reports_cached_tokens_and_identical_reply(monkeypatch, tmp_path):
    pack = tmp_path / "text-pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b"}', encoding="utf-8")
    K = _PrefixKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(pack)},
                            log=lambda *a: None)

    with TestClient(app) as client:
        first = client.post("/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "shared prefix"}]})
        second = client.post("/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "shared prefix plus suffix"}]})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body, second_body = first.json(), second.json()
    prefix = len("shared prefix")
    assert first_body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second_body["usage"]["prompt_tokens_details"]["cached_tokens"] == prefix
    assert K.last_cached == prefix
    assert K.stats["cached_tokens"] == prefix
    assert first_body["choices"][0]["message"]["content"] == second_body["choices"][0]["message"]["content"]
    assert first_body["choices"][0]["message"]["content"] == "reply: shared prefix"


def test_oversized_body_is_413_and_keeps_the_prompt_cache(app_client):
    client, K = app_client
    client.app.state.max_body_bytes = 512
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
