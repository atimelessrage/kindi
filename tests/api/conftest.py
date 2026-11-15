# SPDX-License-Identifier: AGPL-3.0-only
"""Conformance harness for the agent-grade API (T5.1).

Starts the real FastAPI application in a uvicorn thread on a random local port,
with the engine replaced by a scripted stub. The stub implements the server's
current engine interface (generate/check_prompt/encode_text/decode_ids), which
is the interface the HTTP layer actually calls. FakeBackend implements the
new backend contract (T4.1); wiring it into the server is the T4.2 adapter
work, which is deliberately not part of this slice, so this harness drives the
real HTTP stack with the current interface stubbed at the engine boundary.
"""
import threading

import pytest
import uvicorn


class _StubEngine(object):
    ctx = 8192

    def reset(self):
        pass


class _StubKindi(object):
    """The server.Kindi surface the HTTP layer uses, with scripted output."""

    cp = None
    supports_logprobs = True

    def __init__(self, script="Hello from the fake backend."):
        self.cfg = {"model_name": "kindi-fake", "api_key": ""}
        self.pack_id = "stub"
        self.eng = _StubEngine()
        self.lock = threading.Lock()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0}
        self.script = script
        self.t0 = 0.0
        self.last_req = None

    def head_active(self):
        return "exact"

    def after_turn(self, locked=False):
        return None

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def encode_text(self, text, special=True):
        return [ord(c) % 997 + 1 for c in (text or "")]

    def decode_ids(self, ids):
        return "".join(chr((int(i) - 1) % 997) for i in ids)

    def check_prompt(self, req):
        return self.encode_text("prompt")

    def generate(self, req, cancel=None):
        self.last_req = req
        yield "text", self.script
        yield "done", {
            "content": self.script,
            "tool_calls": [],
            "reasoning": "",
            "finish": "stop",
            "logprobs": None,
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 2}},
            "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0},
        }


@pytest.fixture(scope="module")
def api_server():
    """A live HTTP server (uvicorn thread) serving the real app with a stub engine.

    When KINDI_API_URL is set, no server is started: the suite runs against
    that (real, GPU-backed) server instead, which is the maintainer gate.
    """
    import os
    external = os.environ.get("KINDI_API_URL")
    if external:
        yield external.rstrip("/"), None
        return
    from kindi import server

    stub = _StubKindi()
    original = server._make_engine
    server._make_engine = lambda cfg, log: stub
    app = server.create_app({"model_name": "kindi-fake", "api_key": ""},
                            log=lambda *args: None)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    # uvicorn binds asynchronously; wait for the socket.
    import time
    for _ in range(100):
        if uvicorn_server.started:
            break
        time.sleep(0.05)
    host, port = uvicorn_server.servers[0].sockets[0].getsockname()[:2]
    yield "http://%s:%d" % (host, port), stub
    uvicorn_server.should_exit = True
    thread.join(timeout=5)
    server._make_engine = original
