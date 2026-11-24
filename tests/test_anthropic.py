# SPDX-License-Identifier: AGPL-3.0-only
"""The Anthropic Messages endpoint (REQ-0028 A1/A2/A3).

Two layers, so a failure names the half that broke:

* the pure translation (`kindi/api/anthropic.py`): request -> chat request,
  finished answer -> envelope, engine items -> SSE event order;
* the app-level gate A3 asks for: a Claude-Code-shaped request through
  `create_app` with a stub engine, asserting the envelope, the SSE order and the
  tool round-trip, all without a GPU, a pack or a tokenizer.

The one thing these tests cannot cover is A3's real harness run (Claude Code
against `kindi serve`); that stays open in REQ-0028.
"""
import json

import pytest
from fastapi.testclient import TestClient

from kindi import server
from kindi.api import anthropic

# --------------------------------------------------------------------------- translation: request


def test_a_claude_code_shaped_request_maps_onto_the_chat_shape():
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "You are Kindi."}],
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "stop_sequences": ["</answer>"],
        "tools": [{"name": "read_file", "description": "read a file",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "tool_choice": {"type": "tool", "name": "read_file"},
        "messages": [
            {"role": "user", "content": "read README.md"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "README.md"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "# Kindi"}]},
                {"type": "text", "text": "summarize it"}]},
        ],
    }
    req = anthropic.to_chat(body)
    assert req["stream"] is False
    assert req["max_tokens"] == 1024 and req["stop"] == ["</answer>"]
    assert req["temperature"] == 0.2 and req["top_p"] == 0.9 and req["top_k"] == 20
    assert req["tools"][0]["function"]["name"] == "read_file"
    assert req["tools"][0]["function"]["parameters"]["properties"]["path"]["type"] == "string"
    assert req["tool_choice"] == {"type": "function", "function": {"name": "read_file"}}
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    call = req["messages"][2]["tool_calls"][0]
    assert call["id"] == "toolu_1" and call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}
    assert req["messages"][3]["tool_call_id"] == "toolu_1"
    assert req["messages"][3]["content"] == "# Kindi"
    assert req["messages"][4]["content"] == "summarize it"


def test_a_plain_string_message_stays_a_plain_string():
    req = anthropic.to_chat({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_the_system_prompt_may_be_one_string_or_text_blocks():
    assert anthropic.to_chat({"max_tokens": 8, "system": "sys",
                              "messages": [{"role": "user", "content": "hi"}]})["messages"][0] == \
        {"role": "system", "content": "sys"}
    joined = anthropic.to_chat({"max_tokens": 8,
                                "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                                "messages": [{"role": "user", "content": "hi"}]})
    assert joined["messages"][0]["content"] == "ab"


@pytest.mark.parametrize("body,param", [
    ({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}, "messages"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}, "max_tokens"),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": "8"}, "max_tokens"),
    ({"messages": [{"role": "user", "content": "hi"}], "stream": "yes"}, "stream"),
    ({"messages": [{"role": "system", "content": "x"}]}, "messages"),
    ({"messages": [{"role": "user", "content": "hi"}], "stop_sequences": "</x>"}, "stop_sequences"),
    ({"messages": [{"role": "assistant", "content": []}]}, "messages"),
    ({"messages": [{"role": "user", "content": [{"type": "blob"}]}]}, "messages"),
    ({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]}, "messages"),
    ({"messages": "hi"}, "messages"),
    ({"messages": [{"role": "user", "content": "hi"}],
      "tool_choice": {"type": "sideways"}}, "tool_choice"),
])
def test_a_malformed_request_names_its_field(body, param):
    with pytest.raises(ValueError) as excinfo:
        anthropic.to_chat(body)
    assert str(excinfo.value.param).startswith(param), excinfo.value.param


def test_an_unknown_block_type_is_refused_rather_than_dropped():
    with pytest.raises(ValueError) as excinfo:
        anthropic.to_chat({"messages": [{"role": "assistant", "content": [{"type": "audio"}]}]})
    assert "unsupported block type" in str(excinfo.value)


# --------------------------------------------------------------------------- translation: response


def test_the_envelope_carries_text_blocks_stop_reason_and_usage():
    info = {"content": "hello", "tool_calls": [], "finish": "stop",
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
    out = anthropic.message(info, "bonsai2-2b", "msg_1")
    assert out["type"] == "message" and out["role"] == "assistant"
    assert out["model"] == "bonsai2-2b" and out["id"] == "msg_1"
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn" and out["stop_sequence"] is None
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_a_tool_call_becomes_a_tool_use_block_with_a_parsed_input():
    info = {"content": "", "finish": "tool_calls",
            "tool_calls": [{"id": "toolu_9", "type": "function",
                            "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
    out = anthropic.message(info, "bonsai2-2b", "msg_2")
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [{"type": "tool_use", "id": "toolu_9", "name": "read_file",
                               "input": {"path": "a.txt"}}]


def test_an_unparsable_argument_string_keeps_the_raw_text():
    info = {"content": "", "finish": "tool_calls",
            "tool_calls": [{"id": "toolu_8", "function": {"name": "f", "arguments": "{oops"}}]}
    block = anthropic.message(info, "m", "msg_3")["content"][0]
    assert block["input"] == {"_raw": "{oops"}


def test_stop_reasons_map_onto_the_anthropic_values():
    assert anthropic.stop_reason("length") == "max_tokens"
    assert anthropic.stop_reason("tool_calls") == "tool_use"
    assert anthropic.stop_reason("stop") == "end_turn"
    assert anthropic.stop_reason("something-new") == "end_turn"


# --------------------------------------------------------------------------- translation: stream


def events(text):
    out = []
    for frame in text.strip().split("\n\n"):
        name = frame.split("\n")[0].split(": ", 1)[1]
        payload = json.loads(frame.split("\n")[1].split(": ", 1)[1])
        out.append((name, payload))
    assert all(payload["type"] == name for name, payload in out), out
    return out


def test_the_stream_event_order_is_the_anthropic_one():
    chunks = [("text", "he"), ("text", "llo"), ("done", {"content": "hello", "tool_calls": [],
                                                         "finish": "stop",
                                                         "usage": {"prompt_tokens": 5,
                                                                   "completion_tokens": 2}})]
    out = events("".join(anthropic.stream(chunks, "bonsai2-2b", "msg_4", input_tokens=5)))
    assert [name for name, _ in out] == ["message_start", "content_block_start", "content_block_delta",
                                         "content_block_delta", "content_block_stop",
                                         "message_delta", "message_stop"]
    assert out[0][1]["message"]["usage"]["input_tokens"] == 5
    assert out[0][1]["message"]["id"] == "msg_4"
    assert "".join(p["delta"]["text"] for n, p in out if n == "content_block_delta") == "hello"
    assert out[-2][1]["delta"]["stop_reason"] == "end_turn"
    assert out[-2][1]["usage"]["output_tokens"] == 2


def test_a_streamed_tool_call_is_not_leaked_as_text():
    call = {"id": "toolu_1", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}
    chunks = [("text", "let me "), ("text", "<tool_call>{\"name\": "), ("text", "read_file}</tool_call>"),
              ("done", {"content": "", "tool_calls": [call], "finish": "tool_calls",
                        "usage": {"prompt_tokens": 3, "completion_tokens": 9}})]
    out = events("".join(anthropic.stream(chunks, "m", "msg_5")))
    text = "".join(p["delta"]["text"] for n, p in out if n == "content_block_delta"
                   and p["delta"]["type"] == "text_delta")
    assert text == "let me "                       # the call text never reaches a text_delta
    assert "<tool_call>" not in text
    starts = [p for n, p in out if n == "content_block_start"]
    assert starts[-1]["content_block"] == {"type": "tool_use", "id": "toolu_1",
                                          "name": "read_file", "input": {}}
    partial = "".join(p["delta"]["partial_json"] for n, p in out if n == "content_block_delta"
                      and p["delta"]["type"] == "input_json_delta")
    assert json.loads(partial) == {"path": "a"}
    assert out[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_text_held_before_an_incomplete_call_is_flushed_not_dropped():
    chunks = [("text", "answer <tool_call>{half"), ("done", {"content": "", "tool_calls": [],
                                                             "finish": "length", "usage": {}})]
    out = events("".join(anthropic.stream(chunks, "m", "msg_6")))
    text = "".join(p["delta"]["text"] for n, p in out if n == "content_block_delta")
    assert text == "answer <tool_call>{half"
    assert out[-2][1]["delta"]["stop_reason"] == "max_tokens"


# --------------------------------------------------------------------------- app-level gate (A3)


class _FakeEngine(object):
    ctx = 8192


class _FakeKindi(object):
    cp = None
    supports_logprobs = False
    variant = "qwen35"

    def __init__(self, tool_call=False):
        self.cfg = {"model_name": "bonsai2-2b", "api_key": ""}
        self.eng = _FakeEngine()
        self.lock = __import__("threading").Lock()
        self.t0 = 0.0
        self.last_req = None
        self.tool_call = tool_call

    def check_prompt(self, req):
        return [1] * sum(len(str(m.get("content") or "")) for m in req["messages"])

    def after_turn(self, locked=False):
        return None

    def progress(self):
        return None

    def generate(self, req, cancel=None):
        self.last_req = req
        if self.tool_call:
            yield "text", "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}</tool_call>"
            yield "done", {"content": "", "reasoning": "", "finish": "tool_calls", "logprobs": None,
                           "tool_calls": [{"id": "toolu_1", "type": "function",
                                           "function": {"name": "read_file",
                                                        "arguments": "{\"path\": \"a\"}"}}],
                           "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
                           "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0}}
        else:
            yield "text", "hello"
            yield "done", {"content": "hello", "reasoning": "", "finish": "stop", "logprobs": None,
                           "tool_calls": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1,
                                                       "total_tokens": 8},
                           "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0}}


def make_client(monkeypatch, tool_call=False, api_key=""):
    K = _FakeKindi(tool_call=tool_call)
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": api_key}, log=lambda *a: None)
    return TestClient(app), K


CLAUDE_CODE_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 256,
    "system": "You are a coding agent.",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "say hello"}]}],
    "temperature": 0.0,
}


def test_the_endpoint_answers_in_the_anthropic_envelope(monkeypatch):
    client, _ = make_client(monkeypatch)
    with client:
        response = client.post("/v1/messages", json=CLAUDE_CODE_BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 1}
    assert body["id"].startswith("msg_")


def test_the_streamed_endpoint_emits_the_event_sequence(monkeypatch):
    client, _ = make_client(monkeypatch)
    with client:
        with client.stream("POST", "/v1/messages", json=dict(CLAUDE_CODE_BODY, stream=True)) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
    out = events(raw)
    assert [name for name, _ in out] == ["message_start", "content_block_start", "content_block_delta",
                                         "content_block_stop", "message_delta", "message_stop"]
    # message_start carries the counted prompt (system + text), not the stub's reported usage
    assert out[0][1]["message"]["usage"]["input_tokens"] == len("You are a coding agent.") + len("say hello")


def test_a_tool_call_round_trips_through_the_endpoint(monkeypatch):
    tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object",
                                                                           "properties": {
                                                                               "path": {"type": "string"}}}}]
    body = {"max_tokens": 64,
            "messages": [{"role": "user", "content": "read a"}], "tools": tools}
    client, K = make_client(monkeypatch, tool_call=True)
    with client:
        response = client.post("/v1/messages", json=body)
        assert response.status_code == 200, response.text
        out = response.json()
        assert out["stop_reason"] == "tool_use"
        assert out["content"][0]["type"] == "tool_use" and out["content"][0]["name"] == "read_file"
        assert out["content"][0]["input"] == {"path": "a"}
        # the next turn carries the result back in as the engine's tool message
        follow_up = {"max_tokens": 64, "tools": tools, "messages": [
            {"role": "user", "content": "read a"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                              "input": {"path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "A"}]}]}
        assert client.post("/v1/messages", json=follow_up).status_code == 200
    assert K.last_req["messages"][-1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "A"}


def test_a_malformed_request_is_an_anthropic_error_not_a_500(monkeypatch):
    client, _ = make_client(monkeypatch)
    for body in ({"messages": [{"role": "user", "content": [{"type": "image"}]}], "max_tokens": 8},
                 {"messages": [], "max_tokens": 8},
                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}):
        with client:
            response = client.post("/v1/messages", json=body)
        assert response.status_code == 400, response.text
        payload = response.json()
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["message"]


def test_authentication_accepts_x_api_key_and_bearer(monkeypatch):
    client, _ = make_client(monkeypatch, api_key="secret-token")
    with client:
        assert client.post("/v1/messages", json=CLAUDE_CODE_BODY).status_code == 401
        assert client.post("/v1/messages", json=CLAUDE_CODE_BODY,
                           headers={"x-api-key": "wrong"}).status_code == 401
        assert client.post("/v1/messages", json=CLAUDE_CODE_BODY,
                           headers={"x-api-key": "secret-token"}).status_code == 200
        assert client.post("/v1/messages", json=CLAUDE_CODE_BODY,
                           headers={"authorization": "Bearer secret-token"}).status_code == 200
