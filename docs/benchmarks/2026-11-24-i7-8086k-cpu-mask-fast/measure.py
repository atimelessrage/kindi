#!/usr/bin/env python3
"""Structured-vs-unconstrained CPU decode at one reply length, and the schema filter cost.

The end-to-end half posts the same prompt and the same `max_tokens` to a running `kindi serve`
twice - once with a `json_schema` `response_format`, once without - and reports the median
milliseconds per *content* token of each arm, so the constrained overhead is a like-for-like
difference rather than a comparison of two different reply lengths.

  KINDI_DEVICE=cpu python -m kindi --pack ~/.kindi/packs/bonsai-2b serve --host 127.0.0.1 --port 8099
  python measure.py --url http://127.0.0.1:8099/v1 --label after

The filter half walks the same 22-token document as `tools/grammar_cost.py` through the schema's
token-class table on this pack's real vocabulary, and reports the per-state mask cost. Reporting
only: it prints JSON and exits 0 whether or not a grammar refused the document (the `ok` field says
which), so a broken row is visible instead of a stack trace.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import grammar_cost  # noqa: E402

PROMPT = ("Reply with only a JSON object with keys name (a string), tags (an array of strings) and n "
          "(an integer), describing a river. No prose, no code fence.")


def stream(base, payload):
    """(seconds between the first and last content chunk, completion tokens) for one request."""
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    first = last = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usage"):
                tokens = obj["usage"].get("completion_tokens") or tokens
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    return (last - first), tokens


def server_arm(base, prompt_tokens, runs):
    body = {"model": "bonsai2-2b", "messages": [{"role": "system", "content": "You are a terse assistant."},
                                                 {"role": "user", "content": PROMPT}],
            "max_tokens": prompt_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    structured = dict(body, response_format={"type": "json_schema",
                                             "json_schema": {"name": "r", "schema": grammar_cost.SCHEMA,
                                                             "strict": True}})
    for _ in range(3):                      # warm both shapes: engine state and grammar states
        stream(base, structured)
        stream(base, body)
    rows = {"structured": [], "unconstrained": []}
    for _ in range(runs):
        for label, payload in (("structured", structured), ("unconstrained", body)):
            secs, tokens = stream(base, payload)
            rows[label].append(secs / max(1, tokens) * 1e3)
    out = {"runs": runs, "max_tokens": prompt_tokens}
    for label, vals in rows.items():
        out[label] = {"ms_per_token": round(statistics.median(vals), 2),
                      "min": round(min(vals), 2), "max": round(max(vals), 2),
                      "all": [round(v, 2) for v in vals]}
    out["gap_ms_per_token"] = round(out["structured"]["ms_per_token"] - out["unconstrained"]["ms_per_token"], 2)
    return out


def filter_row(pack):
    """The 22-token schema document through the token-class table, plus the split of one miss."""
    tok, texts = grammar_cost.vocab_texts(pack)
    grammar = grammar_cost.grammar_from_schema(grammar_cost.SCHEMA)
    t0 = time.perf_counter()
    grammar.vocab_allowed(grammar.start(), texts)
    first_mask = (time.perf_counter() - t0) * 1e3
    ids = tok.encode(grammar_cost.DOC)
    states = [grammar.start()]
    for token in ids:
        nxt = grammar.advance(states[-1], texts[token])
        if nxt is None:
            break
        states.append(nxt)
    costs, misses, ok = [], 0, len(states) - 1 == len(ids)
    cache = getattr(grammar, "_vocab_cache", None)
    for state in states:
        before = cache.misses if cache is not None else 0
        t0 = time.perf_counter()
        grammar.vocab_allowed(state, texts)
        dt = (time.perf_counter() - t0) * 1e3
        hit = cache is None or cache.misses == before
        if not hit:
            misses += 1
            costs.append(dt)
    return {"tokens": len(ids), "states": len(states), "misses": misses,
            "filter_ms": round(sum(costs), 1), "miss_median_ms": round(statistics.median(costs), 1) if costs else None,
            "miss_max_ms": round(max(costs), 1) if costs else None,
            "first_mask_ms": round(first_mask, 1), "doc_ok": ok}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"))
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    args = ap.parse_args(argv)
    out = {"label": args.label, "pack": os.path.basename(args.pack)}
    if not args.skip_server:
        out["server"] = server_arm(args.url, args.max_tokens, args.runs)
    if not args.skip_filter:
        out["filter"] = filter_row(args.pack)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
