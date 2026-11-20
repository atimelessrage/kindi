#!/usr/bin/env python3
"""Generate kindi/reference-nemotron-cascade-2-30b-a3b.json from llama-server.

The reference is the token-exactness target for the future GPU K-quant
Nemotron-H kernels: greedy (temperature 0) completions for fixed prompts,
recorded with the GGUF's sha256 and the pinned llama.cpp commit.

Example:
    llama-server -m Nemotron-Cascade-2-30B-A3B.Q4_K_M.gguf --port 8080 -ngl 99
    python tools/cascade_reference.py --gguf Nemotron-Cascade-2-30B-A3B.Q4_K_M.gguf \
        --llama-cpp-commit $(git -C llama.cpp rev-parse HEAD)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from kindi.cascade import inspect_cascade  # noqa: E402

PROMPTS = [
    "Write a Python function that reads a JSON file and prints the keys sorted.",
    "Explain in two sentences why a ternary weight matrix uses -1, 0, and 1.",
    "Fill in the missing line: def add(a, b):\n    return ",
    "Summarize the Python GIL in one paragraph.",
]
N_TOKENS = 40


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _tokenize(server, text):
    try:
        out = _post(server.rstrip("/") + "/tokenize", {"content": text})
        return list(out.get("tokens") or out.get("ids") or [])
    except Exception:
        return []


def build_reference(gguf_path, server, llama_cpp_commit, log=print):
    info = inspect_cascade(gguf_path)
    prompts = []
    for text in PROMPTS:
        ids = _tokenize(server, text)
        comp = _post(server.rstrip("/") + "/completion", {
            "prompt": text, "n_predict": N_TOKENS, "temperature": 0.0,
            "cache_prompt": False, "stream": False,
        })
        expected = []
        # llama-server returns token ids in `tokens` for /completion logprobs off;
        # fall back to re-tokenizing the generated text when ids are unavailable.
        if isinstance(comp.get("tokens"), list) and comp["tokens"]:
            expected = [int(t) for t in comp["tokens"][:N_TOKENS]]
        elif comp.get("content"):
            expected = _tokenize(server, comp["content"])[:N_TOKENS]
        if not ids or not expected:
            raise SystemExit("cascade reference: llama-server at %s did not return token ids "
                             "for %r; enable the ids endpoint or re-run with tokenize support"
                             % (server, text[:40]))
        prompts.append({"prompt": text, "ids": ids, "expected": expected,
                        "gap": [None] * len(expected)})
        log("reference prompt: %d ids, %d expected" % (len(ids), len(expected)))
    return {
        "model": info["name"],
        "model_id": "nemotron-cascade-2-30b-a3b",
        "source_file": os.path.basename(gguf_path),
        "source_sha256": sha256(gguf_path),
        "llama_cpp_commit": llama_cpp_commit,
        "source": "llama-server greedy (temperature 0, n_predict=%d) on the exact GGUF above" % N_TOKENS,
        "prompts": prompts,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gguf", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--llama-cpp-commit", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "kindi", "reference-nemotron-cascade-2-30b-a3b.json"))
    a = p.parse_args(argv)
    ref = build_reference(a.gguf, a.server, a.llama_cpp_commit)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
