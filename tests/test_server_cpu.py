# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end server startup on the CPU engine, with no GPU.

`tests/test_server.py` stubs the engine, so a startup failure in the real engine, tokenizer or
spec patching (the paths `kindi serve` takes before it can answer) is invisible there. This test
boots the real `create_app` with `KINDI_DEVICE=cpu` on a real Bonsai 2B pack and answers real
HTTP requests: `/health`, `/v1/models` and a greedy chat completion.

It needs a converted pack, so it resolves one from `KINDI_CPU_PACK` (the same convention as
`tests/test_cpu_layers.py`), falling back to the default `~/.kindi/packs/bonsai-2b`, and skips
when neither exists (CI has no pack). The first run also compiles the CPU library if its cache
is cold, so expect it to be slow once, then it is a normal test.
"""
import json
import os

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from kindi import server  # noqa: E402
from kindi.config import DEFAULTS  # noqa: E402


def _pack():
    candidates = [os.environ.get("KINDI_CPU_PACK"),
                  os.path.expanduser("~/.kindi/packs/bonsai-2b")]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "kindi_pack.json")):
            return cand
    return None


PACK = _pack()
pytestmark = pytest.mark.skipif(PACK is None,
                                reason="no Bonsai 2B pack (set KINDI_CPU_PACK to one)")

# environment that could pull a GPU/drafter path into a CPU boot
_CLEAR = ["KINDI_TC", "KINDI_TP", "KINDI_SPEC", "KINDI_MTP_HEAD", "KINDI_SPEC_PACK",
          "KINDI_SPEC_DRAFT_HEAD", "KINDI_DSPINE", "KINDI_TANDEM", "KINDI_HEAD"]


@pytest.fixture(scope="module")
def cpu_app(tmp_path_factory):
    saved = {name: os.environ.get(name) for name in ["KINDI_DEVICE"] + _CLEAR}
    os.environ["KINDI_DEVICE"] = "cpu"
    for name in _CLEAR:
        os.environ.pop(name, None)
    try:
        cfg = dict(DEFAULTS)
        cfg.update(pack=PACK, model_name="bonsai2-2b", ctx=2048, host="127.0.0.1", port=8093,
                   api_key="", state_dir=str(tmp_path_factory.mktemp("state")), head="exact",
                   max_tokens=8, temperature=0.0, top_p=1.0, top_k=0)
        app = server.create_app(cfg, log=lambda *a: None)
        with TestClient(app) as client:
            yield client
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_cpu_server_boots_and_answers(cpu_app):
    health = cpu_app.get("/health")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["model"] == "bonsai2-2b" and body["ctx"] == 2048 and body["ok"] is True

    models = cpu_app.get("/v1/models")
    assert models.status_code == 200, models.text
    assert models.json()["data"][0]["id"] == "bonsai2-2b"

    reply = cpu_app.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say OK."}],
        "temperature": 0, "max_tokens": 8})
    assert reply.status_code == 200, reply.text
    data = reply.json()
    message = data["choices"][0]["message"]
    assert isinstance(message["content"], str)
    assert data["usage"]["prompt_tokens"] >= 1
    assert data["usage"]["completion_tokens"] >= 1


def test_cpu_server_reports_the_pack_in_health(cpu_app):
    man = json.load(open(os.path.join(PACK, "kindi_pack.json"), encoding="utf-8"))
    assert man.get("backend") not in ("kindi_gemma4", "kindi_nanbeige", "kindi_lightning")
    body = cpu_app.get("/health").json()
    assert body["head"] == "exact"  # the CPU engine has no fast head
