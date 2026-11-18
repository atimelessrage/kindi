# SPDX-License-Identifier: AGPL-3.0-only
"""The standalone DSV4.1 server caps a request body before parsing, like `kindi.server`."""
import io
import json

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
