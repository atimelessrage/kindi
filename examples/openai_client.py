#!/usr/bin/env python3
"""Minimal Kindi client (standard library only, no `openai` package needed).

Usage:
    python3 examples/openai_client.py "Say hello."
    KINDI_API_KEY=... python3 examples/openai_client.py "Say hello."

Environment:
    KINDI_BASE_URL   default http://127.0.0.1:8093/v1
    KINDI_MODEL      default bonsai2-2b
    KINDI_API_KEY    sent as `Authorization: Bearer <key>` when set
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("KINDI_BASE_URL", "http://127.0.0.1:8093/v1").rstrip("/")
MODEL = os.environ.get("KINDI_MODEL", "bonsai2-2b")
KEY = os.environ.get("KINDI_API_KEY", "")


def chat(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + KEY} if KEY else {})})
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello."
    print(chat(prompt))
