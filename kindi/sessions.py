# SPDX-License-Identifier: AGPL-3.0-only
"""Chat session history: a host-only store for `kindi chat` turns.

Sessions live as one JSON file each under `<state_dir>/sessions/`. A session
has a title, an ordered message list, tags, and a triage status
(`open` -> `needs-review` -> `resolved`). Export is raw markdown: the exact
message text, verbatim, with a small header — no rewriting, no reflowing —
and `import_markdown` reads that same format back.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import uuid

STATUSES = ("open", "needs-review", "resolved")


def slug(title, fallback="session"):
    """A safe lowercase dash slug for filenames and titles."""
    s = re.sub(r"[^a-z0-9]+", "-", str(title or "").lower()).strip("-")
    return (s or fallback)[:60]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class SessionStore(object):
    """One JSON file per session in a directory; no GPU, no pack needed."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def path(self, sid):
        return os.path.join(self.root, str(sid) + ".json")

    # ---- creation and messages ----
    def create(self, title="", messages=None, tags=None, status="open"):
        if status not in STATUSES:
            raise ValueError("status must be one of %s (got %r)" % (STATUSES, status))
        sid = uuid.uuid4().hex[:12]
        rec = {"id": sid, "title": str(title or "untitled"), "created": _now(),
               "updated": _now(), "messages": [], "tags": sorted({slug(t) for t in (tags or []) if slug(t) != "session"}),
               "status": status}
        for role, content in (messages or []):
            self.add_message(rec, role, content)
        self._save(rec)
        return rec

    def add_message(self, rec_or_id, role, content):
        rec = self._coerce(rec_or_id)
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unknown role %r" % role)
        rec["messages"].append({"role": role, "content": str(content), "ts": _now()})
        rec["updated"] = _now()
        self._save(rec)
        return rec

    def append_messages(self, rec_or_id, messages):
        """Append new turns to a stored session and save once (`kindi chat --resume` reuses the id)."""
        rec = self._coerce(rec_or_id)
        for role, content in (messages or []):
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError("unknown role %r" % role)
        for role, content in (messages or []):
            rec["messages"].append({"role": role, "content": str(content), "ts": _now()})
        rec["updated"] = _now()
        self._save(rec)
        return rec

    # ---- triage and tags ----
    def set_status(self, rec_or_id, status):
        if status not in STATUSES:
            raise ValueError("status must be one of %s (got %r)" % (STATUSES, status))
        rec = self._coerce(rec_or_id)
        rec["status"] = status
        rec["updated"] = _now()
        self._save(rec)
        return rec

    def add_tag(self, rec_or_id, tag):
        rec = self._coerce(rec_or_id)
        t = slug(tag) if str(tag or "").strip() else ""
        if t and t not in rec["tags"]:
            rec["tags"].append(t)
            rec["tags"].sort()
            rec["updated"] = _now()
            self._save(rec)
        return rec

    def remove_tag(self, rec_or_id, tag):
        rec = self._coerce(rec_or_id)
        t = slug(tag) if str(tag or "").strip() else ""
        if t in rec["tags"]:
            rec["tags"].remove(t)
            rec["updated"] = _now()
            self._save(rec)
        return rec

    # ---- lookup ----
    def get(self, sid):
        p = self.path(sid)
        if not os.path.isfile(p):
            raise KeyError("no session %r in %s" % (sid, self.root))
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def list(self, status=None, tag=None):
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError):
                continue
            if status is not None and rec.get("status") != status:
                continue
            if tag is not None and slug(tag) not in (rec.get("tags") or []):
                continue
            out.append(rec)
        out.sort(key=lambda r: (r.get("updated") or "", r.get("id") or ""), reverse=True)
        return out

    def triage(self):
        """Sessions still needing attention: open or needs-review, newest first."""
        return [r for r in self.list() if r.get("status") in ("open", "needs-review")]

    # ---- raw-markdown export / import ----
    def export_markdown(self, rec_or_id):
        rec = self._coerce(rec_or_id)
        lines = ["# %s" % rec["title"], "",
                 "- id: %s" % rec["id"],
                 "- created: %s" % rec["created"],
                 "- status: %s" % rec["status"],
                 "- tags: %s" % (", ".join(rec.get("tags") or []) or "-"), ""]
        for m in rec.get("messages") or []:
            lines.append("## %s" % m["role"].capitalize())
            lines.append("")
            lines.append(m["content"])
            lines.append("")
        return "\n".join(lines)

    def export_file(self, rec_or_id, directory):
        rec = self._coerce(rec_or_id)
        os.makedirs(directory, exist_ok=True)
        name = "%s-%s.md" % (rec["created"][:10], slug(rec["title"]) or rec["id"])
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.export_markdown(rec))
        return path

    def import_markdown(self, text):
        """Parse the format `export_markdown` writes."""
        title, tags, status, messages, cur = "imported", [], "open", [], None
        for line in str(text).splitlines():
            if line.startswith("# "):
                title = line[2:].strip() or title
            elif line.startswith("- tags:"):
                tags = [t for t in re.split(r"[,\s]+", line.split(":", 1)[1].strip()) if t and t != "-"]
            elif line.startswith("- status:"):
                status = line.split(":", 1)[1].strip() or status
            elif line.startswith("## "):
                cur = {"role": line[3:].strip().lower(), "content": ""}
                messages.append(cur)
            elif cur is not None:
                cur["content"] += line + "\n"
        if status not in STATUSES:
            status = "open"
        msgs = [(m["role"], m["content"].strip("\n")) for m in messages]
        return self.create(title=title, messages=msgs, tags=tags, status=status)

    # ---- internals ----
    def _save(self, rec):
        # a unique tmp per write: two writers must never share one path, or one
        # truncating it mid-write leaves a torn file behind that reads as nothing
        tmp = "%s.%s.tmp" % (self.path(rec["id"]), uuid.uuid4().hex[:8])
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path(rec["id"]))

    def _coerce(self, rec_or_id):
        if isinstance(rec_or_id, dict):
            return rec_or_id
        return self.get(str(rec_or_id))


def session_id_message_hash(rec):
    """A stable short hash of a session's transcript (for dedupe/logging)."""
    blob = "\n".join("%s:%s" % (m["role"], m["content"]) for m in rec.get("messages") or [])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
