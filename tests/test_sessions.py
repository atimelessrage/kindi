# SPDX-License-Identifier: AGPL-3.0-only
"""Chat session store: record, triage, tags, raw-markdown export/import."""
import pytest

from kindi.sessions import SessionStore, session_id_message_hash, slug


def test_record_list_and_get(tmp_path):
    st = SessionStore(str(tmp_path / "sessions"))
    rec = st.create(title="Hello world", messages=[("user", "hi"), ("assistant", "hello")])
    assert st.get(rec["id"])["messages"][0]["content"] == "hi"
    assert [r["id"] for r in st.list()] == [rec["id"]]


def test_triage_status_transitions(tmp_path):
    st = SessionStore(str(tmp_path))
    a = st.create(title="a")
    b = st.create(title="b")
    st.set_status(b, "needs-review")
    assert {r["id"] for r in st.triage()} == {a["id"], b["id"]}
    st.set_status(a, "resolved")
    assert [r["id"] for r in st.triage()] == [b["id"]]
    with pytest.raises(ValueError, match="status must be"):
        st.set_status(a, "archived")


def test_tags_slugified_deduped_and_filterable(tmp_path):
    st = SessionStore(str(tmp_path))
    rec = st.create(title="My Chat!", tags=["Work Notes", "work-notes", ""])
    assert rec["tags"] == ["work-notes"]
    st.add_tag(rec, "Bug Fix")
    assert st.get(rec["id"])["tags"] == ["bug-fix", "work-notes"]
    st.remove_tag(rec, "work notes")
    assert st.get(rec["id"])["tags"] == ["bug-fix"]
    assert [r["id"] for r in st.list(tag="Bug Fix")] == [rec["id"]]


def test_export_is_raw_markdown_and_import_round_trips(tmp_path):
    st = SessionStore(str(tmp_path))
    raw_user = "line one\n  indented line\n\nparagraph two"
    rec = st.create(title="My chat", messages=[("user", raw_user), ("assistant", "answer")])
    st.add_tag(rec, "docs")
    text = st.export_markdown(rec)
    assert "# My chat" in text and raw_user in text  # verbatim, never reflowed
    path = st.export_file(rec, str(tmp_path / "out"))
    assert path.endswith("my-chat.md") or "my-chat" in path
    back = st.import_markdown(text)
    assert back["title"] == "My chat"
    assert [m["content"] for m in back["messages"]] == [raw_user, "answer"]
    assert back["tags"] == ["docs"]


def test_sessions_cli_list_export_round_trip(tmp_path, capsys):
    from kindi.__main__ import main
    root = str(tmp_path)
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert "no sessions" in capsys.readouterr().out
    import os
    st = SessionStore(os.path.join(root, "sessions"))
    rec = st.create(title="CLI demo", messages=[("user", "q")])
    assert main(["--state-dir", root, "sessions", "list"]) == 0
    assert rec["id"] in capsys.readouterr().out
    assert main(["--state-dir", root, "sessions", "export", "--out", str(tmp_path / "out"), rec["id"]]) == 0
    assert list((tmp_path / "out").glob("*.md"))


def test_append_messages_reuses_the_session(tmp_path):
    st = SessionStore(str(tmp_path))
    rec = st.create(title="Chat", messages=[("user", "first")])
    before = st.get(rec["id"])                                                # noqa: E501
    st.append_messages(rec["id"], [("assistant", "answer"), ("user", "follow-up")])
    after = st.get(rec["id"])                                                # noqa: E501
    assert [m["content"] for m in after["messages"]] == ["first", "answer", "follow-up"]
    assert after["updated"] >= before["updated"]
    with pytest.raises(ValueError, match="unknown role"):
        st.append_messages(rec["id"], [("narrator", "boo")])


def test_chat_resume_seed_and_append_round_trip(tmp_path, monkeypatch):
    from kindi import __main__ as cli
    cfg = {"state_dir": str(tmp_path / "state")}
    st = SessionStore(str(tmp_path / "state" / "sessions"))
    rec = st.create(title="Chat", messages=[("user", "first question"), ("assistant", "old answer")])
    seeded_rec, msgs = cli._resume_chat(cfg, rec["id"])
    assert seeded_rec["id"] == rec["id"]
    assert msgs == [{"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "old answer"}]
    msgs += [{"role": "user", "content": "next question"}, {"role": "assistant", "content": "new answer"}]
    cli._append_chat(cfg, rec["id"], msgs[2:])
    contents = [m["content"] for m in st.get(rec["id"])["messages"]]          # noqa: E501
    assert contents == ["first question", "old answer", "next question", "new answer"]
    # a malformed or unknown id refuses loudly instead of seeding an empty chat
    with pytest.raises(SystemExit, match="12 hex characters"):
        cli._resume_chat(cfg, "../etc/passwd")
    with pytest.raises(SystemExit, match="no session"):
        cli._resume_chat(cfg, "0" * 12)


def test_slug_and_hash(tmp_path):
    assert slug("  Hello, World!  ") == "hello-world"
    assert slug("") == "session"
    st = SessionStore(str(tmp_path))
    rec = st.create(messages=[("user", "x")])
    assert session_id_message_hash(rec) == session_id_message_hash(st.get(rec["id"]))
