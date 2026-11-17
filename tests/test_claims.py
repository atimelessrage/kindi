"""The evidence-claims harness: a quoted number must still match the run behind it.

Two layers of tests:

* the repository's own claims must reproduce from their tracked artifacts and the three docs must
  still quote them (`tools/check_claims.py` run for real), and
* the checker itself must fail when a doc number drifts, when an artifact moves, when evidence is
  not tracked, or when a doc starts citing a benchmark run no claim accounts for. A checker that
  cannot fail is worse than none, so the negative cases are built on a small fixture tree.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_claims  # noqa: E402


def _write(root, rel, text):
    path = os.path.join(str(root), rel.replace("/", os.sep))
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _fixture(tmp_path, value=471.7, doc_number="471.7", extra_doc="", unclaimed=None):
    """One gate, one CSV artifact, one doc that quotes it."""
    root = str(tmp_path)
    _write(root, "docs/benchmarks/run-a/data.csv", "mode,model,read\ntc,2b,%s\n" % value)
    _write(root, "docs/DOC.md", "The plan engine reads %s tok/s.\n%s" % (doc_number, extra_doc))
    claim = {
        "gate": "run-a",
        "title": "fixture gate",
        "measured": "2026-01-01",
        "commit": "0000000",
        "command": "python -m kindi bench",
        "evidence": ["docs/benchmarks/run-a/data.csv"],
        "numbers": [{
            "id": "read-2b",
            "display": doc_number,
            "quoted_in": ["docs/DOC.md"],
            "check": {"kind": "csv", "file": "docs/benchmarks/run-a/data.csv",
                      "where": {"mode": "tc", "model": "2b"}, "field": "read", "expect": 471.7},
        }],
    }
    _write(root, "docs/claims/run-a.json", json.dumps(claim, indent=1))
    _write(root, "docs/claims/REGISTRY.json", json.dumps(
        {"docs": ["docs/DOC.md"], "unclaimed": unclaimed or []}, indent=1))
    return root


def _problems(root):
    claims = check_claims.load_claims(root)
    problems = []
    for claim in claims:
        problems.extend(check_claims.check_claim(claim, root=root, tracked=False))
    registry_problems, _registry, _outstanding, _cites = check_claims.check_registry(claims, root)
    problems.extend(registry_problems)
    return problems


def test_repository_claims_reproduce():
    """Every number the three docs quote is re-derived here, from tracked evidence."""
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_claims.py"),
                             ROOT], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK " in result.stdout


def test_repository_claim_files_are_one_per_gate():
    claims = check_claims.load_claims(ROOT)
    assert claims, "no claims found"
    for claim in claims:
        assert os.path.basename(claim["_file"]) == claim["gate"] + ".json", claim["_file"]


def test_repository_evidence_is_tracked():
    claims = check_claims.load_claims(ROOT)
    for claim in claims:
        for rel in claim["evidence"]:
            assert os.path.isfile(os.path.join(ROOT, rel.replace("/", os.sep))), rel
            assert check_claims._tracked(ROOT, rel), "%s rests on untracked %s" % (claim["gate"], rel)


def test_repository_citations_are_accounted_for():
    claims = check_claims.load_claims(ROOT)
    problems, _registry, outstanding, cites = check_claims.check_registry(claims, ROOT)
    assert not problems, problems
    assert cites, "the registry's docs cite no benchmark runs; the coverage rule would be vacuous"
    assert outstanding <= 4, "benchmark citations listed as debt grew to %d" % outstanding


def test_a_fixture_claim_passes_and_then_a_doc_drift_fails(tmp_path):
    root = _fixture(tmp_path)
    assert _problems(root) == []

    _write(root, "docs/DOC.md", "The plan engine reads 480.0 tok/s.\n")
    problems = _problems(root)
    assert any("no longer quotes '471.7'" in p for p in problems), problems


def test_an_artifact_that_moved_fails(tmp_path):
    root = _fixture(tmp_path, value=500.0)
    problems = _problems(root)
    assert any("artifact says" in p for p in problems), problems


def test_a_number_within_tolerance_passes(tmp_path):
    root = _fixture(tmp_path, value=471.9)
    assert _problems(root) == []


def test_untracked_evidence_is_refused(tmp_path):
    root = _fixture(tmp_path)
    claims = check_claims.load_claims(root)
    problems = check_claims.check_claim(claims[0], root=root, tracked=True)
    assert any("not tracked by git" in p for p in problems), problems


def test_a_new_unregistered_citation_fails(tmp_path):
    root = _fixture(tmp_path, extra_doc="See `docs/benchmarks/run-b/RESULTS.md` too.\n")
    problems = _problems(root)
    assert any("unclaimed benchmark citation" in p and "run-b" in p for p in problems), problems


def test_a_listed_debt_entry_is_accepted(tmp_path):
    root = _fixture(tmp_path, extra_doc="See `docs/benchmarks/run-b/RESULTS.md` too.\n",
                    unclaimed=[{"dir": "run-b", "reason": "numbers not requoted yet"}])
    assert _problems(root) == []


def test_a_claim_missing_a_number_fails(tmp_path):
    root = _fixture(tmp_path)
    claim_path = os.path.join(root, "docs", "claims", "run-a.json")
    claim = json.load(open(claim_path, encoding="utf-8"))
    del claim["numbers"]
    _write(root, "docs/claims/run-a.json", json.dumps(claim, indent=1))
    with pytest.raises(check_claims.ClaimError) as exc:
        check_claims.load_claims(root)
    assert "missing 'numbers'" in str(exc.value)


def test_ratio_and_compare_kinds(tmp_path):
    root = _fixture(tmp_path)
    _write(root, "docs/benchmarks/run-a/pair.json", json.dumps(
        {"a": 109.2, "b": 110.7, "big": 471.7, "small": 0.001}))
    spec_ratio = {"kind": "ratio",
                  "a": {"file": "docs/benchmarks/run-a/pair.json", "pointer": "/big"},
                  "b": {"file": "docs/benchmarks/run-a/pair.json", "pointer": "/small"},
                  "expect": 471700}
    assert check_claims.run_check(root, spec_ratio) == pytest.approx(471700.0, rel=0.01)
    spec_pair = {"kind": "compare", "expect_percent": 2.0,
                 "pairs": [["docs/benchmarks/run-a/pair.json", "/a",
                            "docs/benchmarks/run-a/pair.json", "/b"]]}
    assert check_claims.run_check(root, spec_pair) == pytest.approx(1.355, abs=0.05)
    spec_tight = dict(spec_pair, expect=1.0)
    claim_path = os.path.join(root, "docs", "claims", "run-a.json")
    claim = json.load(open(claim_path, encoding="utf-8"))
    claim["numbers"].append({"id": "parity", "display": "471.7", "quoted_in": ["docs/DOC.md"],
                             "check": spec_tight})
    _write(root, "docs/claims/run-a.json", json.dumps(claim, indent=1))
    problems = _problems(root)
    assert any("over the quoted" in p for p in problems), problems


def test_text_kind_needs_the_needle_in_the_artifact(tmp_path):
    root = _fixture(tmp_path)
    spec = {"kind": "text", "file": "docs/benchmarks/run-a/data.csv", "needle": "tc,2b"}
    assert check_claims.run_check(root, spec) is None
    spec["needle"] = "tc,9b,133.9"
    with pytest.raises(check_claims.ClaimError):
        check_claims.run_check(root, spec)
