"""`kindi selftest` proves an installed pack end to end through the real app, without a GPU.

The fake Kindi implements only what the health and chat routes touch; the point is that the
selftest drives `server.create_app` and both HTTP routes, so a broken route or a silent empty
generation fails the command rather than printing a misleading PASS.
"""
import http.server
import json
import pathlib
import threading
import time

import pytest

from kindi import __main__ as cli
from kindi import selftest, server


class _FakeEngine(object):
    ctx = 4096


class _FakeKindi(object):
    cp = None
    supports_logprobs = False
    supports_json_grammar = True
    variant = "qwen35"

    def __init__(self, fail=False):
        self.cfg = {"model_name": "bonsai2-2b", "api_key": "", "ctx": 4096}
        self.pack_id = "fake"
        self.eng = _FakeEngine()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0,
                     "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0 = time.time()
        self.lock = threading.RLock()
        self.fail = fail
        self.closed = False

    def head_active(self):
        return "exact"

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def check_prompt(self, req):
        return [1, 2, 3]

    def encode_text(self, text, special=True):
        return [1, 2, 3]

    def decode_ids(self, ids):
        return "kindi"

    def after_turn(self, locked=False):
        return None

    def progress(self):
        return None

    def close(self):
        self.closed = True

    def generate(self, req, cancel=None):
        if self.fail:
            raise RuntimeError("decode failed")
        yield "text", "ok"
        yield "done", {"content": "ok", "tool_calls": [], "reasoning": "", "finish": "stop",
                       "logprobs": None,
                       "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0}}


def _pack(tmp_path, sha="a" * 64):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps({"model": "bonsai2-2b", "source_sha256": sha}),
                                          encoding="utf-8")
    return pack


def _cfg(pack):
    return {"model_name": "bonsai2-2b", "pack": str(pack), "ctx": 4096, "api_key": ""}


def test_selftest_proves_health_generation_and_the_effective_runtime(monkeypatch, tmp_path):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    pack = _pack(tmp_path)
    out = tmp_path / "selftest.json"
    lines = []

    rc = selftest.run(_cfg(pack), log=lines.append, json_path=str(out))

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["pack"]["source_sha256"] == "a" * 64
    assert report["pack"]["pack_id"] == "a" * 12
    assert report["health"] == {"status": 200, "model": "bonsai2-2b", "ctx": 4096,
                                "head": "exact", "drafting": report["runtime"]["spec"]}
    assert report["runtime"]["loaded"] is True
    assert report["runtime"]["ctx_active"] == 4096
    assert report["generation"]["text"] == "ok"
    assert report["generation"]["completion_tokens"] == 1
    assert report["generation"]["finish"] == "stop"
    # `/v1/messages` is driven in process too, so a broken translation fails the install check
    assert report["anthropic"]["text"] == "ok"
    assert report["anthropic"]["output_tokens"] == 1
    assert any("/v1/messages" in line for line in lines)
    assert report["seconds"] is not None
    assert any("runtime:" in line for line in lines)
    assert any("PASS" in line for line in lines)
    # REQ-0028 A4: the same greedy request through both dialects must decode the same text
    assert report["dialects"] == {"same_text": True, "same_tokens": True,
                                  "chat_text": "ok", "anthropic_text": "ok"}
    assert any("dialects agree" in line for line in lines)
    assert K.closed is True


def test_selftest_fails_when_the_dialects_decode_different_text(monkeypatch, tmp_path):
    """A4's gate: translation only. The engine returns the same text for the same request; if the
    Anthropic route produced something else, something reached decoding and the check must fail."""
    class _Drifting(_FakeKindi):
        def generate(self, req, cancel=None):
            self.calls = getattr(self, "calls", 0) + 1
            second = self.calls > 1
            text = "kindi" if second else "ok"
            yield "text", text
            yield "done", {"content": text, "tool_calls": [], "reasoning": "", "finish": "stop",
                           "logprobs": None,
                           "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                                     "prompt_tokens_details": {"cached_tokens": 0}},
                           "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0}}

    K = _Drifting()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    out = tmp_path / "selftest.json"
    lines = []
    rc = selftest.run(_cfg(_pack(tmp_path)), log=lines.append, json_path=str(out))
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is False and "changed decoding" in report["error"]
    assert report["dialects"]["same_text"] is False
    assert report["dialects"]["chat_text"] == "ok"
    assert report["dialects"]["anthropic_text"] == "kindi"
    assert not any("PASS" in line for line in lines)


def test_the_in_process_client_stays_python36_compatible():
    """The CUDA image runs Python 3.6.9, where asyncio.run (3.7) does not exist: the container
    smoke would fail before it proved anything."""
    src = pathlib.Path(selftest.__file__).read_text(encoding="utf-8")
    assert "asyncio.run(" not in src
    assert "new_event_loop" in src


def test_a_keyed_container_still_proves_itself_in_process(monkeypatch, tmp_path):
    """KINDI_API_KEY is set in the image's environment; the in-process app must not demand it."""
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    seen = {}
    real = server.create_app

    def spy(cfg, log=print):
        seen.update(cfg)
        return real(cfg, log)

    monkeypatch.setattr(server, "create_app", spy)
    cfg = dict(_cfg(_pack(tmp_path)), api_key="container-key", host="0.0.0.0")
    lines = []
    assert selftest.run(cfg, log=lines.append) == 0
    assert seen["api_key"] == "" and seen["host"] == "127.0.0.1"


def test_selftest_missing_pack_is_a_clear_failure(tmp_path):
    lines = []
    rc = selftest.run({"model_name": "bonsai2-2b", "pack": str(tmp_path / "missing")}, log=lines.append)
    assert rc == 1
    assert any("no model pack" in line and "FAIL" in line for line in lines)


def test_selftest_bad_tokenizer_or_engine_is_reported(monkeypatch, tmp_path):
    def broken(cfg, log):
        raise ValueError("merges.txt is missing")

    monkeypatch.setattr(server, "_make_engine", broken)
    lines = []
    rc = selftest.run(_cfg(_pack(tmp_path)), log=lines.append)
    assert rc == 1
    assert any("engine/tokenizer load failed" in line and "merges.txt" in line for line in lines)


def test_selftest_a_failed_generation_is_not_a_pass(monkeypatch, tmp_path):
    K = _FakeKindi(fail=True)
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    lines = []
    rc = selftest.run(_cfg(_pack(tmp_path)), log=lines.append)
    assert rc == 1
    assert any("generation returned HTTP 500" in line for line in lines)
    assert not any("PASS" in line for line in lines)
    assert K.closed is True


def test_selftest_is_wired_into_the_cli():
    a = cli._build_parser().parse_args(["selftest", "--max-tokens", "4"])
    assert a.cmd == "selftest"
    assert a.url is None
    assert a.max_tokens == 4
    a = cli._build_parser().parse_args(["selftest", "--url", "http://127.0.0.1:8093"])
    assert a.url == "http://127.0.0.1:8093"
    assert cli._COMMANDS["selftest"] is cli.cmd_selftest


ANTHROPIC_REPLY = {"id": "msg_1", "type": "message", "role": "assistant", "model": "fake-2b",
                   "content": [{"type": "text", "text": "kindi"}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 1}}


class _Server(http.server.BaseHTTPRequestHandler):
    """A running server stand-in for --url: three routes, no engine."""

    def _reply(self, status, body):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        cfg = self.server.cfg
        cfg["health_calls"] = cfg.get("health_calls", 0) + 1
        if self.path != "/health":
            return self._reply(404, {})
        self._reply(cfg.get("health_status", 200),
                    cfg.get("health", {"ok": True, "model": "fake-2b", "ctx": 4096,
                                       "runtime": {"head": "exact", "spec": {"enabled": False}}}))

    def do_POST(self):
        cfg = self.server.cfg
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        cfg["authorization"] = self.headers.get("Authorization")
        if self.path == "/v1/messages":
            cfg["anthropic_calls"] = cfg.get("anthropic_calls", 0) + 1
            cfg["anthropic_body"] = body
            return self._reply(cfg.get("anthropic_status", 200),
                               cfg.get("anthropic", ANTHROPIC_REPLY))
        cfg["chat_calls"] = cfg.get("chat_calls", 0) + 1
        cfg["chat_body"] = body
        if self.path != "/v1/chat/completions":
            return self._reply(404, {})
        self._reply(cfg.get("chat_status", 200),
                    cfg.get("chat", {"choices": [{"message": {"content": "kindi"}, "finish_reason": "stop"}],
                                     "usage": {"completion_tokens": 1}}))

    def log_message(self, *args):
        pass


@pytest.fixture
def running_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    srv.cfg = {}
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _url(srv):
    return "http://127.0.0.1:%d" % srv.server_address[1]


def test_url_mode_proves_a_server_that_is_already_running(running_server, tmp_path):
    """--url is what the installer calls: no pack load, one health check and one generation."""
    out = tmp_path / "report.json"
    lines = []
    rc = selftest.run_url(_url(running_server), log=lines.append, json_path=str(out))
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "url" and report["model"] == "fake-2b"
    assert report["health"]["status"] == 200 and report["runtime"]["head"] == "exact"
    assert report["generation"]["completion_tokens"] == 1
    assert running_server.cfg["chat_body"]["stream"] is False
    assert any("PASS" in line for line in lines)
    assert report["pack"] is None                    # nothing was loaded here
    # the Anthropic dialect is proven against the same server, in its own shape
    assert running_server.cfg["anthropic_calls"] == 1
    assert running_server.cfg["anthropic_body"]["messages"][0]["content"] == [
        {"type": "text", "text": selftest.PROMPT}]
    assert report["anthropic"]["text"] == "kindi"
    assert report["anthropic"]["stop_reason"] == "end_turn"
    assert report["anthropic"]["blocks"] == ["text"]
    assert report["dialects"]["same_text"] is True               # both routes answered "kindi"


def test_url_mode_fails_when_the_dialects_disagree(running_server):
    """The installer's check must see a served server that answers the two dialects differently."""
    running_server.cfg["anthropic"] = dict(ANTHROPIC_REPLY,
                                          content=[{"type": "text", "text": "something else"}])
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 1
    assert any("changed decoding" in line for line in lines)


def test_url_mode_authenticates_when_the_server_requires_a_key(running_server, monkeypatch):
    """--lan installs require a key; docker exec carries KINDI_API_KEY from the container."""
    monkeypatch.setenv("KINDI_API_KEY", "kindi-test-key")
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 0
    assert running_server.cfg["authorization"] == "Bearer kindi-test-key"


def test_url_mode_reports_a_non_200_health(running_server):
    running_server.cfg["health_status"] = 503
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 1
    assert any("/health returned HTTP 503" in line for line in lines)
    assert running_server.cfg.get("chat_calls", 0) == 0


def test_url_mode_reports_a_failed_generation(running_server):
    running_server.cfg["chat_status"] = 500
    running_server.cfg["chat"] = {"error": {"message": "decode failed"}}
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 1
    assert any("generation returned HTTP 500: decode failed" in line for line in lines)


def test_url_mode_reports_an_anthropic_answer_without_a_message(running_server):
    """A 200 with the wrong shape must fail: the check is that the dialect answers, not that the route exists."""
    running_server.cfg["anthropic"] = {"type": "chat.completion", "content": []}
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 1
    assert any("not a message" in line for line in lines)


def test_url_mode_reports_an_anthropic_route_error(running_server):
    running_server.cfg["anthropic_status"] = 400
    running_server.cfg["anthropic"] = {"type": "error",
                                       "error": {"type": "invalid_request_error", "message": "bad block"}}
    lines = []
    assert selftest.run_url(_url(running_server), log=lines.append) == 1
    assert any("/v1/messages returned HTTP 400: bad block" in line for line in lines)


def test_url_mode_without_a_server_is_a_clear_failure():
    lines = []
    assert selftest.run_url("http://127.0.0.1:9", log=lines.append, timeout=2) == 1
    assert any("FAIL" in line and "/health" in line for line in lines)
