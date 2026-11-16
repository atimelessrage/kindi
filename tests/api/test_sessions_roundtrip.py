# SPDX-License-Identifier: AGPL-3.0-only
"""Sessions HTTP round trip through the real app over the OpenAI-client harness (REQ-0016).

The /sessions routes are not part of the OpenAI wire, and the stub harness
runs the real FastAPI app with a stub engine: these tests drive the host-only
session routes over the same live server, and cover the stateful editing loop
(create → tag → status → export → read) and the malformed-id 400. They are
skip-gated under KINDI_API_URL, like the rest of the suite, because a real
server's state_dir is not the fixture's temp dir.
"""
import glob
import os

import pytest

from kindi.sessions import SessionStore


@pytest.fixture()
def api_server_with_sessions(api_server, tmp_path, monkeypatch):
    """A harness server whose app reads sessions from a fixture state_dir.

    conftest starts its app with state_dir="" which falls back to the CWD;
    this starts its own app on the same stub with a temp state_dir, so the
    sessions routes read the fixture store.
    """
    import threading

    import pytest as _pytest
    import uvicorn

    from kindi import server

    base_url, stub = api_server
    if stub is None:
        _pytest.skip("KINDI_API_URL runs against a real server; sessions need a fixture state_dir")
    state_dir = str(tmp_path / "state")
    os.makedirs(state_dir, exist_ok=True)
    app = server.create_app({"model_name": "kindi-fake", "api_key": "", "state_dir": state_dir},
                            log=lambda *args: None)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    import time
    for _ in range(100):
        if uvicorn_server.started:
            break
        time.sleep(0.05)
    host, port = uvicorn_server.servers[0].sockets[0].getsockname()[:2]
    yield "http://%s:%d" % (host, port), state_dir
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def _store(state_dir):
    return SessionStore(os.path.join(state_dir, "sessions"))


def test_sessions_http_round_trip(api_server_with_sessions):
    import httpx

    base_url, state_dir = api_server_with_sessions
    store = _store(state_dir)
    rec = store.create(title="plan a trip", messages=[("user", "book a flight")])
    sid = rec["id"]
    assert os.path.isfile(store.path(sid))

    response = httpx.get(base_url + "/sessions/%s" % sid, timeout=10)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == sid
    assert response.json()["messages"][0]["content"] == "book a flight"

    # malformed ids never reach a file read: enveloped 400 naming the param
    bad = httpx.get(base_url + "/sessions/not-an-id", timeout=10)
    assert bad.status_code == 400, bad.text
    assert bad.json()["error"]["param"] == "id"

    missing = httpx.get(base_url + "/sessions/0123456789ab", timeout=10)
    assert missing.status_code == 404, missing.text
    assert missing.json()["error"]["param"] == "id"

    bad_status = httpx.get(base_url + "/sessions?status=bogus", timeout=10)
    assert bad_status.status_code == 400, bad_status.text
    assert bad_status.json()["error"]["param"] == "status"


def test_sessions_list_reflects_the_store(api_server_with_sessions):
    import httpx

    base_url, state_dir = api_server_with_sessions
    store = _store(state_dir)
    rec = store.create(title="plan a trip", messages=[("user", "hi")])
    store.add_tag(rec["id"], "travel")
    response = httpx.get(base_url + "/sessions", params={"tag": "travel"}, timeout=10)
    assert response.status_code == 200, response.text
    rows = response.json()["sessions"]
    assert [r["id"] for r in rows] == [rec["id"]]
    filtered = httpx.get(base_url + "/sessions", params={"tag": "other"}, timeout=10)
    assert filtered.status_code == 200
    assert filtered.json()["sessions"] == []


def test_sessions_status_tags_and_read_over_http(api_server_with_sessions):
    import httpx

    base_url, state_dir = api_server_with_sessions
    store = _store(state_dir)
    rec = store.create(title="plan a trip", messages=[("user", "hi")])
    sid = rec["id"]
    store.mark_read(sid)
    # status + tags mutation round trip over HTTP
    r = httpx.post(base_url + "/sessions/%s/status" % sid,
                   json={"status": "needs-review"}, timeout=10)
    assert r.status_code == 200, r.text
    assert store.get(sid)["status"] == "needs-review"
    tagged = httpx.post(base_url + "/sessions/%s/tags" % sid,
                        json={"tag": "flight"}, timeout=10)
    assert tagged.status_code == 200, tagged.text
    assert store.get(sid)["tags"] == ["flight"]
    # export is the verbatim raw markdown
    exported = httpx.get(base_url + "/sessions/%s/export" % sid, timeout=10)
    assert exported.status_code == 200, exported.text
    assert exported.headers["content-type"].startswith("text/markdown")
    assert "hi" in exported.text
    # the read route clears unread
    store.create(title="unread one", messages=[("user", "yo")])["unread"]
    others = store.list()
    # find an unread record and clear it over HTTP
    unread = [o for o in others if o.get("unread")][0]
    read = httpx.post(base_url + "/sessions/%s/read" % unread["id"], timeout=10)
    assert read.status_code == 200, read.text
    assert store.get(unread["id"])["unread"] is False
