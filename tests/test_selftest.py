"""`kindi selftest` proves an installed pack end to end through the real app, without a GPU.

The fake Kindi implements only what the health and chat routes touch; the point is that the
selftest drives `server.create_app` and both HTTP routes, so a broken route or a silent empty
generation fails the command rather than printing a misleading PASS.
"""
import json
import threading
import time

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
    assert report["seconds"] is not None
    assert any("runtime:" in line for line in lines)
    assert any("PASS" in line for line in lines)
    assert K.closed is True


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
    assert a.max_tokens == 4
    assert cli._COMMANDS["selftest"] is cli.cmd_selftest
