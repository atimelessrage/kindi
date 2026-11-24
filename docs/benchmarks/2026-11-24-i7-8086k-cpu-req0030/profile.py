#!/usr/bin/env python3
"""REQ-0030 A1, native CPU path: profile on the 2B with the CPU engine.

Components (no server, real pack): BPE encode of real rendered prompts, the
token-class table/trie build and steady masks via tools/grammar_cost.py's
walk (imported, not reimplemented), and decode_bytes cost.

Server (separate, the serve process already running on :8099): streaming
TTFT / content / wall for the same 49-token request shape as the 2026-11-22
GPU profile, unconstrained and json_schema, cold-structured (first) vs warm.

Prints one JSON object; the caller records it as the benchmark artifact.
Exits non-zero if any component fails, so a partial run is never a record.
"""
import contextlib
import io
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for import_path in (ROOT, os.path.join(ROOT, "tools")):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import grammar_cost  # noqa: E402

from kindi import chat  # noqa: E402

PACK = os.path.expanduser("~/.kindi/packs/bonsai-2b")

out = {"host": "i7-8086K CPU engine (KINDI_DEVICE=cpu)", "pack": "bonsai-2b"}

tok, texts = grammar_cost.vocab_texts(PACK)
texts = [t for t in texts] if texts else None
VOCAB = len(texts) if texts else len(tok.tokens)
out["vocab"] = VOCAB

# tokenizer: the API prompt shape (49 tokens on the GPU profile) + long contexts
prompts = {}
api_messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
    {"role": "user", "content": "What is the capital of France?"},
]
api_text = chat.render(api_messages)[0]
prompts["api"] = api_text
prompts["repo-2k"] = "The quick brown fox jumps over the lazy dog. " * 45
prompts["repo-6k"] = "The quick brown fox jumps over the lazy dog. " * 135
prompts["repo-13k"] = "The quick brown fox jumps over the lazy dog. " * 300
tok_rows = {}
for name, text in prompts.items():
    cold_ids = tok.encode(text)
    rows = []
    for _ in range(5):
        t0 = time.perf_counter()
        tok.encode(text)
        rows.append((time.perf_counter() - t0) * 1e3)
    tok_rows[name] = {"tokens": len(cold_ids),
                      "cold_ms": None,
                      "warm_ms": statistics.median(rows),
                      "warm_min_ms": min(rows)}
out["tokenizer"] = tok_rows

# decode_bytes over 2000 ids
ids = (tok.encode(prompts["repo-2k"]) * 2)[:2000]
t0 = time.perf_counter()
tok.decode(ids)
out["decode_2000_ms"] = round((time.perf_counter() - t0) * 1e3, 2)

# mask: first build (shared index) + warm walk, short and long docs.
# grammar_cost.build/walk time compile+first mask and the per-token walk.
mask_rows = {}
for label, doc, schema in (("short", grammar_cost.DOC, grammar_cost.SCHEMA),
                           ("long", grammar_cost.LONG_DOC, grammar_cost.LONG_SCHEMA)):
    grammar = grammar_cost.grammar_from_schema(schema)
    t0 = time.perf_counter()
    allowed = grammar.vocab_allowed(grammar.start(), texts)
    first_ms = (time.perf_counter() - t0) * 1e3
    with contextlib.redirect_stdout(io.StringIO()):  # walk() prints its own rows
        mean, ok, per_token, misses = grammar_cost.walk(grammar, tok, texts, doc, label)
    per_miss = sorted(c for c in per_token if c > 0)
    mask_rows[label] = {
        "first_mask_ms": round(first_ms, 1),
        "allowed_at_start": len(allowed),
        "ok": bool(ok),
        "misses": misses,
        "median_miss_ms": round(per_miss[len(per_miss) // 2], 1) if per_miss else None,
        "max_miss_ms": round(max(per_miss), 1) if per_miss else None,
        "mean_ms_per_token": round(mean, 1),
    }
out["mask"] = mask_rows

# framing: replay the server's chunk() cost is sub-microsecond per token
# (2026-11-22: 0.0046 ms/token); re-measured here as json.dumps of one chunk.
chunk = {"id": "chatcmpl-x", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"},
                      "finish_reason": None}]}
rows = []
for _ in range(1000):
    t0 = time.perf_counter()
    json.dumps(chunk)
    rows.append((time.perf_counter() - t0) * 1e3)
out["framing_ms_per_chunk"] = round(statistics.median(rows), 4)

# server: streaming, same shape as the GPU profile
BASE = "http://127.0.0.1:8099/v1"
MODEL = json.loads(urllib.request.urlopen(BASE + "/../health.json", timeout=5).read() or b"{}") \
    if False else "bonsai2-2b"


def stream(payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + "/chat/completions", data=body)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    first_content = None
    last_content = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                now = time.perf_counter()
                if first_content is None:
                    first_content = now
                last_content = now
                chunks += 1
    wall = time.perf_counter() - t0
    return {"ttft_s": round((first_content - t0), 3) if first_content else None,
            "content_s": round((last_content - first_content), 3) if first_content else None,
            "wall_s": round(wall, 3), "content_chunks": chunks, "usage": usage}


def med(runs):
    keys = ("ttft_s", "content_s", "wall_s")
    return {k: (statistics.median([r[k] for r in runs if r[k] is not None])
                if any(r[k] is not None for r in runs) else None) for k in keys}


base_payload = {"model": MODEL, "messages": api_messages, "max_tokens": 24,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
server = {}
# first structured request (cold mask build on this server would be the NVIDIA
# shape; here the prewarm already ran, so the first json_schema request is the
# warm-shape measurement — the build cost itself is the in-process row above)
server["structured_first"] = stream(dict(base_payload, response_format={
    "type": "json_schema",
    "json_schema": {"name": "r", "schema": grammar_cost.SCHEMA, "strict": True}}))
server["structured_warm"] = [stream(dict(base_payload, response_format={
    "type": "json_schema",
    "json_schema": {"name": "r", "schema": grammar_cost.SCHEMA, "strict": True}}))
    for _ in range(3)]
server["unconstrained_warm"] = [stream(base_payload) for _ in range(3)]
out["server"] = {
    "structured_first": server["structured_first"],
    "structured_warm_median": med(server["structured_warm"]),
    "unconstrained_warm_median": med(server["unconstrained_warm"]),
    "unconstrained_warm_runs": server["unconstrained_warm"],
}

print(json.dumps(out, indent=2, sort_keys=True))
