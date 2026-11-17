#!/usr/bin/env python3
"""End-to-end half of tools/grammar_cost.py: one server, three response_formats, one prompt.

  KINDI_GBNF_MAX_DEPTH=1 python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --port 8102
  python tools/grammar_cost_serve.py http://127.0.0.1:8102                  # 20-token reply
  python tools/grammar_cost_serve.py http://127.0.0.1:8102 --doc long       # 200-token reply

Sends the same 49-token prompt with no constraint, with a `json_schema` and with llama.cpp's
`json.gbnf` as `{"type": "grammar"}`, `max_tokens 20`, temperature 0, and prints wall time, the
server's generation timing, tok/s and the completion so the three answers can be compared. The server
needs `KINDI_GBNF_MAX_DEPTH` set or the recursive grammar is refused with a 400 -- that refusal is
printed, not swallowed, and makes the run exit 1.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("grammar_cost", os.path.join(HERE, "grammar_cost.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
JSON_GBNF, SCHEMA = _mod.JSON_GBNF, _mod.SCHEMA
LONG_SHP = (_mod.long_grammar(), _mod.LONG_SCHEMA)

PROMPT = ("Reply with only a JSON object with keys name (a string), tags (an array of strings) and n "
          "(an integer), describing a river. No prose, no code fence.")
# the long prompt asks for the ordered record both constraints describe, so neither has to refuse a
# key or reorder the reply; 20 river names is what makes the document ~200 tokens
PROMPT_LONG = ("Reply with only a JSON object with exactly these keys in this order: id, type, name, "
               "count, items, value, tags, n. 'items' holds 20 river names as strings, 'tags' at least "
               "5 strings, count and n are integers, value is one sentence. No other keys, no prose, "
               "no code fence.")
PROMPT_TIGHT = ("Reply with only a compact JSON object, no spaces and no newlines, with exactly these "
                "keys in this order: id, type, name, count, items, value, tags, n. items holds 20 "
                "river names, tags at least 5 strings, count and n are integers.")
MAX_TOKENS = {"short": 20, "long": 200}


def post(url, payload, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return time.perf_counter() - t0, json.loads(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8102")
    ap.add_argument("--doc", choices=("short", "long"), default="short")
    ap.add_argument("--tokens", type=int, default=None,
                    help="max_tokens (default: 20 short, 200 long)")
    ap.add_argument("--tight", action="store_true",
                    help="long only: measure the whitespace-free grammar instead of the pair. A "
                         "grammar that permits whitespace lets the greedy 2B spend the whole budget "
                         "on it after the first colon, so only this variant makes a long reply "
                         "progress (measured 2026-11-17)")
    args = ap.parse_args()
    url = args.url
    grammar, schema = (JSON_GBNF, SCHEMA) if args.doc == "short" else LONG_SHP
    max_tokens = args.tokens or MAX_TOKENS[args.doc]
    prompt = PROMPT if args.doc == "short" else PROMPT_LONG
    if args.tight:
        grammar = _mod.tight_grammar()
        prompt = PROMPT_TIGHT
    base = {"model": "bonsai2-2b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    cases = [(label, body) for label, body in (
        ("unconstrained", {}),
        ("json_schema", {"response_format": {"type": "json_schema",
                                            "json_schema": {"name": "record", "schema": schema}}}),
        ("gbnf", {"response_format": {"type": "grammar", "grammar": grammar}}),
    ) if not (args.tight and label == "json_schema")]
    failed = False
    for label, extra in cases:
        payload = dict(base)
        payload.update(extra)
        try:
            wall, d = post(url, payload)
        except urllib.error.HTTPError as e:
            failed = True
            print("%-14s FAILED %s: %s" % (label, e.code, e.read()[:300].decode("utf-8", "replace")))
            continue
        except Exception as e:
            failed = True
            print("%-14s FAILED: %s" % (label, e))
            continue
        t = d.get("timing") or {}
        usage = d.get("usage") or {}
        text = d["choices"][0]["message"].get("content") or ""
        print("%-14s wall=%7.2f s  gen_s=%7.2f  gen_tok_s=%7.1f  tokens=%s  chars=%d  %s"
              % (label, wall, t.get("gen_s", 0), t.get("gen_tok_s", 0),
                 usage.get("completion_tokens", "?"), len(text), text[:96].replace("\n", " ")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
