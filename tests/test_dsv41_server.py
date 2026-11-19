# SPDX-License-Identifier: AGPL-3.0-only
"""The standalone DSV4.1 server caps a request body before parsing, like `kindi.server`."""
import io
import json
import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from kindi import dsv41_server


def test_a_body_under_the_cap_is_parsed():
    body = json.dumps({"messages": []}).encode()
    assert dsv41_server.read_json_body(
        {"Content-Length": str(len(body))}, io.BytesIO(body)) == {"messages": []}
    # no header: the old empty-body behavior, and nothing is read
    assert dsv41_server.read_json_body({}, io.BytesIO(b"ignored")) == {}


def test_a_declared_body_over_the_cap_is_refused_before_reading():
    class Unread:
        def read(self, size=-1):
            raise AssertionError("read() called for an over-cap body")

    with pytest.raises(dsv41_server.BodyTooLarge, match="larger than 8 bytes"):
        dsv41_server.read_json_body({"Content-Length": "9"}, Unread(), limit=8)


def test_a_body_exactly_at_the_cap_is_accepted():
    body = b'{"a": 1}'
    assert dsv41_server.read_json_body(
        {"Content-Length": str(len(body))}, io.BytesIO(body), limit=len(body)) == {"a": 1}


def test_a_malformed_length_or_body_is_a_request_error():
    for headers, body in (({"Content-Length": "fast"}, b"{}"),
                          ({"Content-Length": "-1"}, b"{}"),
                          ({"Content-Length": "2"}, b"no")):
        with pytest.raises(ValueError):
            dsv41_server.read_json_body(headers, io.BytesIO(body))


class _Engine:
    class model:
        ctx = 1024


def _handler(headers, body):
    """A handler instance with the socket parts replaced: only the body/error path runs."""
    h = object.__new__(dsv41_server.make_handler(_Engine))
    h.headers = headers
    h.rfile = io.BytesIO(body)
    h.close_connection = False
    sent = []
    h._err = lambda code, msg, typ="invalid_request_error": sent.append((code, msg, typ))
    return h, sent


def test_the_tokenize_endpoint_returns_413_for_an_over_cap_body():
    h, sent = _handler({"Content-Length": str(dsv41_server.MAX_BODY_BYTES + 1)}, b"")
    h._tokenize()
    assert sent and sent[0][0] == 413 and sent[0][2] == "request_too_large"


def test_the_tokenize_endpoint_returns_400_for_invalid_json():
    h, sent = _handler({"Content-Length": "2"}, b"no")
    h._tokenize()
    assert sent and sent[0][0] == 400 and sent[0][2] == "invalid_request_error"


def test_a_body_error_closes_the_keepalive_connection():
    """An oversized body is refused before it is read, so the unread bytes must not become the
    next request on the same HTTP/1.1 connection. A malformed body is closed too because its
    declared length cannot be trusted."""
    for headers, body in (({"Content-Length": str(dsv41_server.MAX_BODY_BYTES + 1)}, b""),
                          ({"Content-Length": "fast"}, b"{}"),
                          ({"Content-Length": "2"}, b"no")):
        h, _sent = _handler(headers, body)
        h._tokenize()
        assert h.close_connection is True, (headers, body)


def test_the_error_response_announces_the_close():
    h = object.__new__(dsv41_server.make_handler(_Engine))
    sent = []
    h.send_response = lambda code: sent.append(("status", code))
    h.send_header = lambda key, value: sent.append((key, value))
    h.end_headers = lambda: None
    h.wfile = io.BytesIO()
    h.close_connection = True
    h._json(413, {"error": {"message": "too large"}})
    assert ("status", 413) in sent
    assert ("Connection", "close") in sent


class _SocketEngine:
    class model:
        ctx = 1024

        class tok:
            @staticmethod
            def encode(text):
                return list(text.encode())

    class a:
        api_key = ""


def _post(body):
    return (b"POST /v1/tokenize HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
            b"Connection: keep-alive\r\nContent-Length: " + str(len(body)).encode() +
            b"\r\n\r\n" + body)


def test_an_oversized_body_does_not_hide_the_413_or_leak_into_the_next_request(monkeypatch):
    """A real socket: 1 MiB declared against a small test cap, then a valid pipelined request.

    Closing with that unread body used to reset the connection before the client read the 413
    (zero bytes on Windows). The response must arrive, the connection must close, and the valid
    request must not be served on the same socket.
    """
    cap = 64
    real = dsv41_server.read_json_body
    monkeypatch.setattr(dsv41_server, "read_json_body",
                        lambda headers, rfile: real(headers, rfile, limit=cap))
    server = ThreadingHTTPServer(("127.0.0.1", 0), dsv41_server.make_handler(_SocketEngine()))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sock = None
    try:
        host, port = server.server_address
        sock = socket.create_connection((host, port), timeout=5)
        sock.sendall(_post(b"x" * (1 << 20)) + _post(b'{"content":"ok"}'))
        raw = b""
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                raw += chunk
        except ConnectionError:
            pass
        assert b"HTTP/1.1 413" in raw, raw[:200]
        assert b'"type": "request_too_large"' in raw
        assert b"Connection: close" in raw
        assert raw.count(b"HTTP/1.1 ") == 1, raw[:400]
        assert b'"count"' not in raw
    finally:
        if sock is not None:
            sock.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_valid_sampling_fields_are_accepted():
    req = {"messages": [{"role": "user", "content": "hello"}],
           "max_tokens": 4, "max_completion_tokens": 8, "temperature": 0.7, "top_p": 0.9,
           "top_k": 20, "min_p": 0.05, "seed": 2 ** 32 - 1}
    assert dsv41_server.validate_request(req) is req
    # the default/off values are valid too
    assert dsv41_server.validate_request({"messages": req["messages"], "temperature": 0,
                                          "top_p": 1, "top_k": 0, "min_p": 0})


def test_malformed_or_non_finite_sampling_fields_are_400_before_generation():
    """The HTTP boundary rejects the field with 400 before count() or run() is reached."""
    class NoEngine:
        class a:
            api_key = ""

        class model:
            ctx = 1024

        @staticmethod
        def count(req):
            raise AssertionError("count() ran before the request was validated")

        @staticmethod
        def run(*args, **kwargs):
            raise AssertionError("run() ran before the request was validated")

    cases = (
        ({"max_tokens": "4"}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": -1}, "max_tokens"),
        ({"max_completion_tokens": 1.5}, "max_completion_tokens"),
        ({"max_completion_tokens": 0}, "max_completion_tokens"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": "0.7"}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_p": -0.1}, "top_p"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": float("nan")}, "top_p"),
        ({"top_k": 2.0}, "top_k"),
        ({"top_k": True}, "top_k"),
        ({"top_k": -1}, "top_k"),
        ({"min_p": float("nan")}, "min_p"),
        ({"min_p": 1.1}, "min_p"),
        ({"min_p": -0.1}, "min_p"),
        ({"seed": 1.5}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2 ** 32}, "seed"),
    )
    for extra, field in cases:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        payload.update(extra)
        body = json.dumps(payload).encode()
        h = object.__new__(dsv41_server.make_handler(NoEngine))
        h.path = "/v1/chat/completions"
        h.headers = {"Content-Length": str(len(body))}
        h.rfile = io.BytesIO(body)
        h.close_connection = False
        sent = []
        h._err = lambda code, msg, typ="invalid_request_error", box=sent: box.append((code, msg, typ))
        h.do_POST()
        assert sent and sent[0][0] == 400, (extra, sent)
        assert field in sent[0][1], (extra, sent)


@pytest.mark.parametrize(("field", "extra"), (
    ("messages", {"messages": None}),
    ("messages", {"messages": {}}),
    ("messages", {"messages": []}),
    ("messages[0]", {"messages": [1]}),
    ("messages[0].role", {"messages": [{"role": "robot", "content": "hi"}]}),
    ("messages[0].content", {"messages": [{"role": "user", "content": 1}]}),
    ("messages[0].content[0]", {"messages": [{"role": "user", "content": [1]}]}),
    ("messages[0].content[0].text",
     {"messages": [{"role": "user", "content": [{"type": "text", "text": 1}]}]}),
    ("messages[0].tool_calls", {"messages": [{"role": "assistant", "tool_calls": {}}]}),
    ("messages[0].tool_calls[0]", {"messages": [{"role": "assistant", "tool_calls": [{}]}]}),
    ("messages[0].tool_calls[0].arguments",
     {"messages": [{"role": "assistant", "tool_calls":
                    [{"function": {"name": "f", "arguments": 1}}]}]}),
    ("tools", {"tools": {}}),
    ("tools[0]", {"tools": [1]}),
    ("tools[0].function.parameters",
     {"tools": [{"function": {"name": "f", "parameters": []}}]}),
    ("stream", {"stream": "true"}),
    ("stream", {"stream": 1}),
    ("stream_options", {"stream_options": []}),
    ("stream_options.include_usage", {"stream_options": {"include_usage": "true"}}),
    ("stop", {"stop": 1}),
    ("stop", {"stop": [1]}),
    ("stop", {"stop": ["ok", 1]}),
))
def test_malformed_shapes_are_400_before_count(field, extra):
    """A bad message/tool/stream/stop shape is a 400 at the boundary, not a late render failure."""
    class NoEngine:
        class a:
            api_key = ""

        class model:
            ctx = 1024

        @staticmethod
        def count(req):
            raise AssertionError("count() ran before the request was validated")

        @staticmethod
        def run(*args, **kwargs):
            raise AssertionError("run() ran before the request was validated")

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    payload.update(extra)
    body = json.dumps(payload).encode()
    h = object.__new__(dsv41_server.make_handler(NoEngine))
    h.path = "/v1/chat/completions"
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h.close_connection = False
    sent = []
    h._err = lambda code, msg, typ="invalid_request_error", box=sent: box.append((code, msg, typ))
    h.do_POST()
    assert sent and sent[0][0] == 400, (extra, sent)
    assert field in sent[0][1], (extra, sent)


def test_valid_message_tool_stream_and_stop_shapes_are_accepted():
    req = {
        "messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "stop": ["END"],
    }
    assert dsv41_server.validate_request(req) is req
