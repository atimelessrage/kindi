"""Scratch: how far does create_app get with a registry-backend proxy instead of Kindi?

Run: python docs/benchmarks/2026-11-25-llamacpp-proxy-serving/probe.py
"""
import os
import sys
import threading
import time
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "kindi")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from kindi import server
from kindi.backends.base import SamplingParams
from kindi.backends.llamacpp import LlamaCppBackend


def make_backend(opener=None):
    def canned(request):
        if request["url"].endswith("/health"):
            return {"status": "ok"}
        body = request.get("body") or {}
        prompt = body.get("prompt") or ""
        return {"content": "hello from llama-server", "truncated": False,
                "tokens_predicted": 4,
                "timings": {"prompt_n": len(prompt.split()), "predicted_n": 4}}

    backend = LlamaCppBackend(opener=opener or canned)
    backend.load({})
    return backend


class Proxy(object):
    def __init__(self, backend):
        self.backend = backend
        self.cfg = {"model_name": "proxy-gguf", "api_key": "", "temperature": 0.0, "top_p": 1.0,
                    "top_k": 0, "max_tokens": 64, "fast_clusters": 0, "repeat_stop": False,
                    "reasoning_effort": None, "thinking_budget": 0, "state_dir": "bench-logs/.probe",
                    "host": "127.0.0.1", "pack": None}
        self.log = lambda *a: None
        self.cp = None
        self.variant = "llamacpp"
        self.native = False
        self.nanbeige = None
        self.supports_logprobs = False
        self.supports_json_grammar = False
        self.supports_tools = False
        self.runner = None
        self.tok = None
        self.eng = types.SimpleNamespace(ctx=8192, hsl=None, shortlist=0)
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0,
                     "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                      "errors": 0, "gen_seconds": 0.0, "last_tok_s": 0.0}
        self.lock = threading.RLock()
        self.t0 = time.time()

    def head_mode(self):
        return "exact"

    def head_active(self):
        return "exact"

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def check_prompt(self, req):
        raise NotImplementedError("the proxy cannot count upstream tokens")

    def encode_text(self, text, special=True):
        return self.backend.tokenize(text, special)

    def decode_ids(self, ids):
        return self.backend.detokenize(ids)

    def after_turn(self, locked=False):
        return None

    def generate(self, req, cancel=None):
        messages = req.get("messages") or []
        text = self.backend.apply_chat_template(messages)
        ids = self.backend.tokenize(text)
        params = SamplingParams(max_tokens=int(req.get("max_tokens") or 64),
                                temperature=float(req.get("temperature") or 0.0),
                                top_p=float(req.get("top_p") or 1.0),
                                top_k=int(req.get("top_k") or 0),
                                stop=tuple(req.get("stop") or ()),
                                seed=req.get("seed"))
        collected, timing = "", {}
        for event in self.backend.generate(ids, params, cancel=cancel):
            if event.kind == "token":
                collected += event.text
                yield "text", event.text
            elif event.kind == "done":
                timing = dict(event.timing or {})
        self.stats["requests"] = self.stats.get("requests", 0) + 1
        self.stats["completion_tokens"] = self.stats.get("completion_tokens", 0) + len(collected)
        self.stats["gen_seconds"] = self.stats.get("gen_seconds", 0.0) + float(timing.get("gen_s") or 0.0)
        yield "done", {"content": collected, "tool_calls": [], "reasoning": "",
                       "finish": "stop", "logprobs": None,
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(collected),
                                 "total_tokens": len(ids) + len(collected),
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "timing": {"prompt_s": 0.0, "gen_s": timing.get("gen_s", 0.0),
                                  "gen_tok_s": timing.get("gen_tok_s", 0.0)}}

    def close(self):
        self.backend.close()


def main():
    proxy = Proxy(make_backend())
    server._make_engine = lambda cfg, log: proxy
    from fastapi.testclient import TestClient
    app = server.create_app(dict(proxy.cfg), log=lambda *a: None)
    with TestClient(app) as client:
        for path in ("/health", "/v1/models", "/metrics"):
            r = client.get(path)
            print(path, r.status_code, (r.text or "")[:160].replace("\n", " "))
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 8})
        print("chat", r.status_code, r.text[:300])
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 8, "stream": True})
        print("stream", r.status_code, r.text[:300].replace("\n", " "))
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                     "response_format": {"type": "json_object"}})
        print("structured", r.status_code, r.text[:200])


if __name__ == "__main__":
    main()
