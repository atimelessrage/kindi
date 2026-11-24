# SPDX-License-Identifier: AGPL-3.0-only
"""kindi.route: the front-door router must authenticate in constant time and
refuse malformed or oversized requests before touching a backend.

No socket is opened: a fake reader/writer drives Router._handle directly, and
every case here returns before any backend connection is attempted (wrong key,
bad Content-Length, oversized body, oversized header block).
"""
import asyncio
import json

from kindi import route


class FakeReader(object):
    def __init__(self, head=b"", body=b"", overrun=False):
        self._head, self._body, self._overrun = head, body, overrun

    async def readuntil(self, sep):
        if self._overrun:
            raise asyncio.LimitOverrunError("header too large", len(self._head) + 1)
        return self._head

    async def readexactly(self, n):
        assert len(self._body) == n
        return self._body


class FakeWriter(object):
    def __init__(self):
        self.data = b""

    def write(self, b):
        self.data += b

    async def drain(self):
        pass


def run_handle(router, reader):
    writer = FakeWriter()
    asyncio.get_event_loop().run_until_complete(router._handle(reader, writer))
    head, _, body = writer.data.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body.decode())


def make_router(key="s3cret"):
    return route.Router({"b1": ("http://127.0.0.1:1", ["m1"])}, api_key=key, log=lambda m: None)


def test_wrong_key_is_401():
    head = b"GET /v1/models HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer wrong\r\n\r\n"
    code, _ = run_handle(make_router(), FakeReader(head=head))
    assert code == 401


def test_missing_key_is_401_when_a_key_is_set():
    head = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"
    code, _ = run_handle(make_router(), FakeReader(head=head))
    assert code == 401


def test_no_key_configured_means_no_auth():
    # with no key the request proceeds past auth; an unknown model then 404s
    # without any backend connection being attempted.
    body = b'{"model": "nope"}'
    head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n"
            % len(body))
    code, _ = run_handle(make_router(key=None), FakeReader(head=head, body=body))
    assert code == 404  # not 401: auth was not the failure


def test_non_numeric_content_length_is_400():
    head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n"
    code, _ = run_handle(make_router(), FakeReader(head=head))
    assert code == 400


def test_negative_content_length_is_400():
    head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n"
    code, _ = run_handle(make_router(), FakeReader(head=head))
    assert code == 400


def test_oversized_body_is_413():
    head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n"
            % (route.MAX_BODY_BYTES + 1))
    code, _ = run_handle(make_router(), FakeReader(head=head))
    assert code == 413


def test_oversized_header_block_is_431():
    code, _ = run_handle(make_router(), FakeReader(head=b"x" * 70000, overrun=True))
    assert code == 431


def test_auth_uses_the_shared_constant_time_check():
    import inspect

    from kindi.api import auth

    assert auth.bearer_matches("Bearer s3cret", "s3cret") is True
    assert auth.bearer_matches("Bearer wrong", "s3cret") is False
    assert "bearer_matches" in inspect.getsource(route.Router._handle)
