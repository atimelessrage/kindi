# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI-client conformance tests (T5.1).

The official openai Python client talks to the live server started in
conftest.py. These tests pin the wire shape agent frameworks depend on:
non-streaming and streaming completions, usage (including
stream_options.include_usage and prompt_tokens_details.cached_tokens), the
models listing and the finish_reason vocabulary. They run against the stub
engine, so they are GPU-free; the maintainer gate (docs/API.md) repeats them
against a real pack.
"""
import pytest
from openai import OpenAI


@pytest.fixture()
def client(api_server):
    base_url, _stub = api_server
    return OpenAI(base_url=base_url + "/v1", api_key="test-key", timeout=10)


def _chat(client, **kwargs):
    kwargs.setdefault("model", "kindi-fake")
    kwargs.setdefault("messages", [{"role": "user", "content": "hello"}])
    kwargs.setdefault("max_tokens", 8)
    return client.chat.completions.create(**kwargs)


def test_non_streaming_completion_shape(client):
    response = _chat(client)
    assert response.object == "chat.completion"
    assert response.model == "kindi-fake"
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello from the fake backend."
    assert choice.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 7
    details = response.usage.prompt_tokens_details
    assert details.cached_tokens == 2


def test_streaming_completion_terminates_with_done(client):
    stream = _chat(client, stream=True, stream_options={"include_usage": True})
    chunks = list(stream)
    assert chunks, "the stream must yield at least one chunk"
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk.model == "kindi-fake" for chunk in chunks)
    joined = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert joined == "Hello from the fake backend."
    # the final chunk carries usage and a terminal finish reason
    last = chunks[-1]
    assert last.usage is not None
    assert last.usage.total_tokens == 7


def test_models_listing_shape(client):
    models = client.models.list()
    assert models.object == "list"
    assert len(models.data) == 1
    model = models.data[0]
    assert model.id == "kindi-fake"
    assert model.object == "model"
    assert model.owned_by == "kindi"
    # the real context under every name agent harnesses read
    assert model.context_length == 8192
    assert model.max_model_len == 8192
    assert model.meta["n_ctx"] == 8192


def test_structured_output_is_accepted_and_records_the_request(client, api_server):
    _base_url, stub = api_server
    response = _chat(client, response_format={"type": "json_schema", "json_schema": {
        "name": "x", "schema": {"type": "object",
                            "properties": {"a": {"type": "string"}}, "required": ["a"]}}})
    assert response.choices[0].message.content == "Hello from the fake backend."
    assert stub is None or stub.last_req.get("response_format", {}).get("type") == "json_schema"


def test_structured_output_rejected_on_a_backend_without_the_grammar(api_server):
    import httpx

    _base_url, stub = api_server
    if stub is None:
        pytest.skip("KINDI_API_URL runs against the real server; backend flags belong to the stub harness")
    stub.supports_json_grammar = False
    try:
        response = httpx.post(_base_url + "/v1/chat/completions", json={
            "model": "kindi-fake",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {
                "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
        }, timeout=10)
    finally:
        stub.supports_json_grammar = True
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_error_envelope_for_a_bad_request(client, api_server):
    import httpx

    base_url, _stub = api_server
    response = httpx.post(base_url + "/v1/chat/completions",
                          json={"messages": "not a list"}, timeout=10)
    assert response.status_code == 400
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["type"] == "invalid_request_error"
    assert isinstance(error["message"], str) and error["message"]
