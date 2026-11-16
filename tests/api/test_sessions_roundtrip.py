# SPDX-License-Identifier: AGPL-3.0-only
"""Sessions HTTP round trip through the real app over the OpenAI-client harness (REQ-0016).

The /sessions routes are not part of the OpenAI wire, and the stub harness
runs the real FastAPI app with a stub engine: these tests drive the host-only
session routes over the same live server. Only the transport glue lives here
(show, tag-filtered list, the list-status 400, status/tags/read mutations,
verbatim export); route semantics are pinned in-process by
tests/test_server_sessions.py, so malformed/missing-id cases are not repeated.
Skip-gated under KINDI_API_URL, like the rest of the suite, because a real
server's state_dir is not the fixture's temp dir.
"""
import os

import httpx

from kindi.sessions import SessionStore


def _store(state_dir):
    return SessionStore(os.path.join(state_dir, "sessions"))


def test_sessions_show_and_list_over_http(api_server_with_sessions):
    base_url, state_dir = api_server_with_sessions
    store = _store(state_dir)
    rec = store.create(title="plan a trip", messages=[("user", "book a flight")])
    sid = rec["id"]
    store.add_tag(sid, "travel")

    response = httpx.get(base_url + "/sessions/%s" % sid, timeout=10)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == sid
    assert response.json()["messages"][0]["content"] == "book a flight"

    listed = httpx.get(base_url + "/sessions", params={"tag": "travel"}, timeout=10)
    assert listed.status_code == 200, listed.text
    assert [r["id"] for r in listed.json()["sessions"]] == [sid]
    filtered = httpx.get(base_url + "/sessions", params={"tag": "other"}, timeout=10)
    assert filtered.status_code == 200
    assert filtered.json()["sessions"] == []

    bad_status = httpx.get(base_url + "/sessions?status=bogus", timeout=10)
    assert bad_status.status_code == 400, bad_status.text
    assert bad_status.json()["error"]["param"] == "status"


def test_sessions_status_tags_and_read_over_http(api_server_with_sessions):
    base_url, state_dir = api_server_with_sessions
    store = _store(state_dir)
    sid = store.create(title="plan a trip", messages=[("user", "hi")])["id"]

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

    # the read route clears the unread flag raised on creation
    unread_id = store.create(title="unread one", messages=[("user", "yo")])["id"]
    read = httpx.post(base_url + "/sessions/%s/read" % unread_id, timeout=10)
    assert read.status_code == 200, read.text
    assert store.get(unread_id)["unread"] is False
