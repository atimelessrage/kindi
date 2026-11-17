"""Evidence-claims harness: a number in a doc may not outlive the run that produced it.

Every measured gate owns one `docs/claims/<gate>.json`. A claim records the numbers the docs quote,
the artifact the run left behind, and how to recompute each number from that artifact. This checker
re-derives every number, then checks that the exact text the docs quote is still there. A doc edit that
changes a number without a new run, or a run whose artifact no longer supports the quoted figure, fails
here instead of passing review.

    python tools/check_claims.py            # exit 1 with every problem it found
    python tools/check_claims.py --list     # gate -> numbers, for coverage review

Kinds of check (one per number, `check` field):
    {"kind": "json",   "file": ..., "pointer": "/3/kld_nats", "expect": 2.07e-07}
    {"kind": "csv",    "file": ..., "where": {"mode": "tc", "model": "2b"}, "field": "read",
                       "expect": 471.7}
    {"kind": "ratio",  "a": {"file": ..., "pointer": ...}, "b": {...}, "expect": 470}
    {"kind": "compare","pairs": [[file_a, pointer_a, file_b, pointer_b], ...],
                       "expect_percent": 1.0}          # max relative gap, in percent
    {"kind": "text",   "file": ..., "needle": "486-488 MiB"}   # verbatim in the artifact
    {"kind": "git",    "op": "diffstat", "revs": ["v0.6.0", "v0.6.1"], "field": "files",
                       "expect": 16}                   # numbers about the repository itself

A `git` check re-derives a figure from a revision range rather than from a file: `commits`
(`rev-list --count a..b`), `diffstat` (`files`/`insertions`/`deletions` of `git diff --numstat`),
`deleted` (files removed between two revisions) and `treefiles` (tracked files at one revision,
optionally narrowed by `path` and a basename `match` glob). Its evidence is not a file but a ref:
an evidence entry `ref:<tag-or-commit>` must resolve to a commit, so a claim on a tag cannot survive
the tag moving or disappearing.

Numbers are compared with a relative tolerance (`tolerance`, default 0.2% = the rounding the docs use).
Artifacts must be tracked: a claim cannot rest on gitignored local output, because no future reader or
CI run can reproduce it. Gitignored runs are named in the doc text as such and stay out of the registry.
"""
from __future__ import print_function

import csv
import fnmatch
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
CLAIMS_DIR = "docs/claims"
REGISTRY = "docs/claims/REGISTRY.json"
DEFAULT_TOLERANCE = 0.002
REQUIRED_CLAIM_KEYS = ("gate", "title", "measured", "commit", "command", "evidence", "numbers")
REQUIRED_NUMBER_KEYS = ("id", "display", "quoted_in", "check")


class ClaimError(Exception):
    pass


def _root(root):
    return DEFAULT_ROOT if root is None else root


def _join(root, rel):
    return os.path.join(_root(root), rel.replace("/", os.sep))


def _read(root, rel):
    path = _join(root, rel)
    if not os.path.isfile(path):
        raise ClaimError("missing artifact: %s" % rel)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _json(root, rel):
    try:
        return json.loads(_read(root, rel))
    except ValueError as exc:
        raise ClaimError("%s is not JSON: %s" % (rel, exc))


def norm(text):
    """Collapse whitespace so a display string survives the doc's line wrapping."""
    return re.sub(r"\s+", " ", text).strip()


def _pointer(obj, pointer, rel):
    cur = obj
    if pointer in ("", "/"):
        return cur
    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, ValueError, TypeError):
            raise ClaimError("%s has no %s" % (rel, pointer))
    return cur


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClaimError("%s is not a number: %r" % (what, value))
    return float(value)


def _close(got, expect, tolerance):
    expect = _number(expect, "expect")
    if expect == 0:
        return abs(got) <= tolerance
    return abs(got - expect) / abs(expect) <= tolerance


def _row_value(root, spec):
    """Pick the row of a CSV by exact field match, then return one numeric field of it."""
    text = _read(root, spec["file"])
    where = spec.get("where") or {}
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ClaimError("%s has no rows" % spec["file"])
    unknown = sorted(set(where) - set(rows[0]))
    if unknown:
        raise ClaimError("%s has no column %s" % (spec["file"], ", ".join(unknown)))
    for row in rows:
        if all(str(row.get(k, "")).strip() == str(v) for k, v in where.items()):
            return float(row[spec["field"]])
    raise ClaimError("%s has no row matching %r" % (spec["file"], where))


def _git(root, args):
    """Run one read-only git query against the repository, rewriting a failure as a claim error."""
    command = ["git", "-C", _root(root)] + list(args)
    try:
        out = subprocess.check_output(command, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = (getattr(exc, "output", b"") or b"").decode("utf-8", "replace").strip()
        raise ClaimError("git %s failed: %s" % (" ".join(args), detail or exc))
    return out.decode("utf-8", "replace")


def _git_value(root, spec):
    """One repository figure, recomputed from the revision range the claim names."""
    op = spec.get("op")
    if op == "commits":
        first, second = spec["revs"]
        return float(_git(root, ["rev-list", "--count", "%s..%s" % (first, second)]).strip())
    if op == "deleted":
        first, second = spec["revs"]
        names = _git(root, ["diff", "--diff-filter=D", "--name-only", first, second])
        return float(len([line for line in names.splitlines() if line.strip()]))
    if op == "diffstat":
        first, second = spec["revs"]
        field = spec.get("field", "files")
        if field not in ("files", "insertions", "deletions"):
            raise ClaimError("unknown diffstat field %r" % field)
        counts = {"files": 0, "insertions": 0, "deletions": 0}
        for line in _git(root, ["diff", "--numstat", first, second]).splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            counts["files"] += 1
            # a binary file shows "-" in both columns; it counts as a changed file, no lines
            if len(parts) > 1 and parts[0].isdigit():
                counts["insertions"] += int(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                counts["deletions"] += int(parts[1])
        return float(counts[field])
    if op == "treefiles":
        args = ["ls-tree", "-r", "--name-only", spec["rev"]]
        if spec.get("path"):
            args += ["--", spec["path"]]
        names = [line for line in _git(root, args).splitlines() if line.strip()]
        if spec.get("match"):
            names = [name for name in names if fnmatch.fnmatch(os.path.basename(name), spec["match"])]
        return float(len(names))
    raise ClaimError("unknown git op %r" % op)


def run_check(root, spec):
    """Recompute one number; returns the value the artifact gives (or None for a text check)."""
    kind = spec.get("kind")
    if kind == "git":
        return _git_value(root, spec)
    if kind == "json":
        value = _pointer(_json(root, spec["file"]), spec["pointer"], spec["file"])
        return _number(value, "%s%s" % (spec["file"], spec["pointer"]))
    if kind == "csv":
        return _row_value(root, spec)
    if kind == "text":
        needle = spec["needle"]
        if norm(needle) not in norm(_read(root, spec["file"])):
            raise ClaimError("%s does not contain %r" % (spec["file"], needle))
        return None
    if kind == "ratio":
        a = _number(_pointer(_json(root, spec["a"]["file"]), spec["a"]["pointer"],
                             spec["a"]["file"]), "numerator")
        b = _number(_pointer(_json(root, spec["b"]["file"]), spec["b"]["pointer"],
                             spec["b"]["file"]), "denominator")
        if b == 0:
            raise ClaimError("ratio denominator is zero")
        return a / b
    if kind == "compare":
        worst = 0.0
        for file_a, ptr_a, file_b, ptr_b in spec["pairs"]:
            a = _number(_pointer(_json(root, file_a), ptr_a, file_a), "compare a")
            b = _number(_pointer(_json(root, file_b), ptr_b, file_b), "compare b")
            if b == 0:
                raise ClaimError("compare denominator is zero")
            worst = max(worst, abs(a - b) / abs(b) * 100.0)
        return worst
    raise ClaimError("unknown check kind %r" % kind)


def _tracked(root, rel):
    """True when git tracks the artifact; a claim may not rest on local-only output."""
    try:
        subprocess.check_output(["git", "-C", _root(root), "ls-files", "--error-unmatch", rel],
                                stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def load_claims(root=None):
    directory = _join(root, CLAIMS_DIR)
    if not os.path.isdir(directory):
        raise ClaimError("no %s directory" % CLAIMS_DIR)
    claims = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == os.path.basename(REGISTRY):
            continue
        path = "%s/%s" % (CLAIMS_DIR, name)
        try:
            claim = _json(root, path)
        except ClaimError as exc:
            raise ClaimError("%s: %s" % (name, exc))
        if not isinstance(claim, dict):
            raise ClaimError("%s: a claim must be a JSON object" % name)
        missing = [k for k in REQUIRED_CLAIM_KEYS if k not in claim]
        if missing:
            raise ClaimError("%s: missing %s" % (name, ", ".join(repr(m) for m in missing)))
        claim["_file"] = path
        claims.append(claim)
    return claims


def load_registry(root=None):
    try:
        registry = _json(root, REGISTRY)
    except ClaimError as exc:
        raise ClaimError("%s: %s" % (REGISTRY, exc))
    if not isinstance(registry, dict) or not isinstance(registry.get("docs"), list):
        raise ClaimError("%s must be an object with a docs list" % REGISTRY)
    return registry


def check_claim(claim, root=None, tracked=True):
    """Every problem with one claim: schema, tracked evidence, reproduced numbers, surviving quotes."""
    problems = []
    name = claim.get("gate") or claim.get("_file")

    for key in REQUIRED_CLAIM_KEYS:
        if key not in claim:
            problems.append("%s: missing %r" % (name, key))
    if problems:
        return problems
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", claim["gate"]):
        problems.append("%s: gate must be lowercase words with dashes" % name)
    if not claim["evidence"]:
        problems.append("%s: evidence is empty" % name)

    for rel in claim["evidence"]:
        if rel.startswith("ref:"):
            try:
                _git(root, ["rev-parse", "--verify", "--quiet", rel[4:] + "^{commit}"])
            except ClaimError:
                problems.append("%s: evidence %s does not resolve to a commit" % (name, rel))
            continue
        if not os.path.exists(_join(root, rel)):
            problems.append("%s: evidence %s does not exist" % (name, rel))
        elif tracked and not _tracked(root, rel):
            problems.append("%s: evidence %s is not tracked by git" % (name, rel))

    docs = {}
    for doc in claim.get("docs", []):
        docs[doc] = True

    seen = set()
    for number in claim["numbers"]:
        ident = number.get("id") if isinstance(number, dict) else None
        if not isinstance(number, dict):
            problems.append("%s: a number must be an object" % name)
            continue
        missing = [k for k in REQUIRED_NUMBER_KEYS if k not in number]
        if missing:
            problems.append("%s/%s: missing %s" % (name, ident or "?", ", ".join(missing)))
            continue
        if ident in seen:
            problems.append("%s: duplicate number id %r" % (name, ident))
        seen.add(ident)
        for doc in number["quoted_in"]:
            docs[doc] = True
            try:
                text = norm(_read(root, doc))
            except ClaimError as exc:
                problems.append("%s/%s: %s" % (name, ident, exc))
                continue
            if norm(number["display"]) not in text:
                problems.append("%s/%s: %s no longer quotes %r"
                                % (name, ident, doc, number["display"]))
        tolerance = number.get("tolerance", DEFAULT_TOLERANCE)
        try:
            got = run_check(root, number["check"])
        except ClaimError as exc:
            problems.append("%s/%s: %s" % (name, ident, exc))
            continue
        if got is None:
            continue
        if "expect" not in number["check"]:
            problems.append("%s/%s: check has no expect" % (name, ident))
            continue
        # "compare" is a bound (the gap must stay under it), every other kind is an equality.
        if number["check"].get("kind") == "compare":
            limit = _number(number["check"]["expect"], "expect") + tolerance * 100.0
            if got > limit:
                problems.append("%s/%s: artifacts differ by %.2f%%, over the quoted %.2f%%"
                                % (name, ident, got, limit))
            continue
        if not _close(got, number["check"]["expect"], tolerance):
            problems.append("%s/%s: artifact says %r, the claim quotes %r (tolerance %.3g%%)"
                            % (name, ident, round(got, 6), number["check"]["expect"],
                               tolerance * 100.0))
    claim["_docs"] = sorted(docs)
    return problems


def cited_dirs(registry, root=None):
    """Every docs/benchmarks/<dir> the claimed docs cite, with the docs that cite it."""
    pattern = re.compile(r"(?:docs/)?benchmarks/([A-Za-z0-9][A-Za-z0-9._-]*)")
    cites = {}
    for doc in registry["docs"]:
        try:
            text = _read(root, doc)
        except ClaimError:
            continue
        for match in pattern.finditer(text):
            cites.setdefault(match.group(1), set()).add(doc)
    return dict((k, sorted(v)) for k, v in cites.items())


def claim_dirs(claims):
    dirs = set()
    for claim in claims:
        for rel in claim.get("evidence", []):
            parts = rel.split("/")
            if "benchmarks" in parts:
                after = parts[parts.index("benchmarks") + 1:]
                if after:
                    dirs.add(after[0])
    return dirs


def check_registry(claims, root=None):
    """A benchmark citation must be claimed, or listed as debt with a reason."""
    registry = load_registry(root)
    problems = []
    entries = registry.get("unclaimed") or []
    listed = {}
    for entry in entries:
        if not isinstance(entry, dict) or "dir" not in entry or "reason" not in entry:
            problems.append("%s: every unclaimed entry needs dir and reason" % REGISTRY)
            continue
        listed[entry["dir"]] = entry
    claimed = claim_dirs(claims)
    cites = cited_dirs(registry, root)
    for directory in sorted(cites):
        if directory in claimed or directory in listed:
            continue
        problems.append("unclaimed benchmark citation: docs/benchmarks/%s cited by %s"
                        % (directory, ", ".join(cites[directory])))
    for directory in sorted(claimed & set(listed)):
        problems.append("%s is both claimed and listed as unclaimed" % directory)
    return problems, registry, cited_count(cites, claimed), cites


def cited_count(cites, claimed):
    return len(set(cites) - set(claimed))


def main(argv):
    root = None
    args = [a for a in argv[1:] if not a.startswith("-")]
    if args:
        root = args[0]
    try:
        claims = load_claims(root)
        problems = []
        for claim in claims:
            problems.extend(check_claim(claim, root=root))
        registry_problems, registry, outstanding, cites = check_registry(claims, root)
    except ClaimError as exc:
        print("FAIL %s" % exc)
        return 1
    problems.extend(registry_problems)

    if "--list" in argv[1:]:
        for claim in claims:
            print("%-26s %-10s %s" % (claim.get("gate"), claim.get("measured"),
                                       ", ".join(n["id"] for n in claim["numbers"])))
        print("%d gates, %d numbers, %d cited benchmark dirs, %d listed as debt"
              % (len(claims), sum(len(c["numbers"]) for c in claims), len(cites), outstanding))
        return 0 if not problems else 1

    for problem in problems:
        print("FAIL %s" % problem)
    if problems:
        print("\n%d problem(s) across %d claims; a quoted number must match its run"
              % (len(problems), len(claims)))
        return 1
    print("OK %d gates, %d numbers re-derived from tracked evidence; %d docs keep their quotes; "
          "%d benchmark citation(s) listed as debt"
          % (len(claims), sum(len(c["numbers"]) for c in claims), len(registry["docs"]), outstanding))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
