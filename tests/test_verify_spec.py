# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi verify` under KINDI_SPEC: the mode it names, and the refusal when it cannot run (REQ-0025).

`kindi verify` used to build an engine with speculation off whatever `KINDI_SPEC` said, so a PASS
was always the plain engine's and mode A's exactness was never gated. These tests pin the two halves
that can be checked without a card: the first line names the effective mode (and the loaded head),
and a requested mode whose drafter did not attach exits non-zero instead of quietly verifying the
plain path.

The engines here are stand-ins: `spec_state` reads exactly the attributes the real engine build
sets (`mtp`, `drafter`, `dspine`, `spec_begin`, `_mtp_heads`), and no kernel runs.
"""
import importlib
import sys
import types

import pytest

from kindi import verify


def _stub_cupy_if_needed():
    try:
        import cupy  # noqa: F401
    except ImportError:
        cp = types.ModuleType("cupy")
        cuda = types.ModuleType("cupy.cuda")
        cuda.runtime = types.ModuleType("cupy.cuda.runtime")
        cuda.runtime.is_hip = False
        cuda.compiler = types.ModuleType("cupy.cuda.compiler")
        cuda.function = types.ModuleType("cupy.cuda.function")
        cp.cuda = cuda
        cp.ndarray = type("ndarray", (), {})
        sys.modules.update({"cupy": cp, "cupy.cuda": cuda, "cupy.cuda.runtime": cuda.runtime,
                            "cupy.cuda.compiler": cuda.compiler, "cupy.cuda.function": cuda.function})


HEAD = {"file": "kindi-mtp-2b-v1.npz", "state": "loaded", "registered": True,
        "sha256": "1b5c0c790a594af368c3646370c527c319cdb3bb0592dd3e523e752f721a5520",
        "bytes": 60960552}
REFUSED = {"file": "kindi-mtp-2b-v1.npz", "state": "refused",
           "reason": "the MTP head /heads/kindi-mtp-2b-v1.npz does not match kindi/mtp_heads.json: "
                     "expected sha256 1b5c0c790a59, 60960552 bytes; found sha256 000000000000, 12 bytes"}


def _engine(**kw):
    """The attributes a built engine has: `spec_begin` marks the plan engine; the rest stay None."""
    eng = types.SimpleNamespace(spec_begin=lambda *a, **k: None, drafter=None, dspine=None, mtp=None,
                                _mtp_heads={})
    eng.__dict__.update(kw)
    return eng


@pytest.fixture(scope="module", autouse=True)
def _spec_importable():
    _stub_cupy_if_needed()
    importlib.import_module("kindi.spec")


def test_no_spec_request_reports_off_and_verifies_the_plain_path(monkeypatch):
    monkeypatch.delenv("KINDI_SPEC", raising=False)
    mode, line, refusal = verify.spec_state(_engine())
    assert mode is None and refusal is None
    assert line == "speculation: off (KINDI_SPEC=0)"


def test_the_head_that_attached_is_named_with_its_digest(monkeypatch):
    """The loaded case: requested mtp, effective mtp, the head file and its sha256 prefix."""
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    mode, line, refusal = verify.spec_state(_engine(mtp=object(), _mtp_heads={"mtp": HEAD}))
    assert mode == "mtp" and refusal is None
    assert "requested mtp, effective mtp" in line
    assert "kindi-mtp-2b-v1.npz sha256 1b5c0c790a59" in line


def test_a_requested_mode_whose_drafter_did_not_attach_is_a_refusal(monkeypatch):
    """REQ-0025 A4: no quiet verification of the plain engine under a mode that did not attach."""
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    mode, line, refusal = verify.spec_state(_engine(_mtp_heads={"mtp": REFUSED}))
    assert mode is None
    assert "effective 0" in line and "does not match kindi/mtp_heads.json" in line
    assert refusal and "mtp was requested but the effective mode is 0" in refusal
    assert "does not match kindi/mtp_heads.json" in refusal


def test_draft2b_without_its_pack_is_a_refusal_not_an_ngram_pass(monkeypatch):
    """`draft2b` falls back to `ngram` when the 2B pack is missing; that would test the wrong mode."""
    monkeypatch.setenv("KINDI_SPEC", "draft2b")
    mode, line, refusal = verify.spec_state(_engine())
    assert mode == "ngram"
    assert refusal and "draft2b was requested but the effective mode is ngram" in refusal
    assert "the 2B draft pack is not attached" in refusal


def test_auto_may_settle_on_another_mode(monkeypatch):
    """`auto` exists to choose, so a different effective mode is not a refusal."""
    monkeypatch.setenv("KINDI_SPEC", "auto")
    mode, line, refusal = verify.spec_state(_engine(mtp=object(), _mtp_heads={"mtp": HEAD}))
    assert (mode, refusal) == ("mtp", None)


def test_an_engine_without_speculation_says_so_instead_of_implying_a_mode(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "0")
    eng = types.SimpleNamespace()
    mode, line, refusal = verify.spec_state(eng)
    assert mode is None and refusal is None and "unavailable" in line
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    mode, line, refusal = verify.spec_state(eng)
    assert refusal and "this engine cannot speculate" in refusal


def test_the_cli_flag_selects_the_mode(monkeypatch, tmp_path):
    """`kindi verify --spec mtp` is `KINDI_SPEC=mtp` for the run, and nothing else changes."""
    import kindi
    main = importlib.import_module("kindi.__main__")
    seen = {}
    monkeypatch.setattr(verify, "run", lambda cfg, log=print: seen.update(called=True) or 0)
    monkeypatch.delenv("KINDI_SPEC", raising=False)
    assert main.cmd_verify({"state_dir": str(tmp_path)}, types.SimpleNamespace(spec="mtp")) == 0
    assert seen.get("called") is True and main.os.environ["KINDI_SPEC"] == "mtp"
    assert kindi  # the package imported, not a stub


def _reference(tmp_path):
    """One prompt, two expected tokens: enough for `run` to take its whole path."""
    import json
    rp = tmp_path / "reference.json"
    rp.write_text(json.dumps({"prompts": [{"prompt": "p", "ids": [1, 2], "expected": [3, 4], "gap": []}]}),
                  encoding="utf-8")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b"}', encoding="utf-8")
    return str(pack), str(rp)


class _Gen(object):
    """A built engine's generation surface: the reference's tokens, and a record of the spec calls."""

    def __init__(self, **kw):
        self.spec_begin_calls, self.spec_end_calls, self.generate_calls = [], 0, 0
        self.__dict__.update(kw)

    def spec_begin(self, greedy=True, *a, **k):
        self.spec_begin_calls.append(greedy)

    def spec_end(self):
        self.spec_end_calls += 1
        return {"passes": 1, "drafted": 1, "accepted": 1}

    def generate(self, ids, n):
        self.generate_calls += 1
        return [3, 4]


def test_verify_runs_the_selected_mode_and_prints_it(monkeypatch, tmp_path):
    """REQ-0025 A1/A5: with a mode requested, `run` wraps its generation in spec_begin/spec_end,
    prints the mode first, and still token-compares against the reference; the plain path calls
    neither."""
    pack, rp = _reference(tmp_path)
    monkeypatch.setattr(verify, "reference_for", lambda p: rp)
    eng = _Gen(mtp=object(), _mtp_heads={"mtp": HEAD})
    monkeypatch.setattr("kindi.engine.make_engine", lambda *a, **k: eng)
    said = []
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    assert verify.run({"pack": pack}, log=said.append) == 0
    assert "speculation: requested mtp, effective mtp" in said[0]
    assert eng.spec_begin_calls == [True] and eng.spec_end_calls == 1 and eng.generate_calls == 1
    assert any("2/2 tokens match" in s and "1/1 drafts accepted" in s for s in said)

    plain = _Gen()
    monkeypatch.setattr("kindi.engine.make_engine", lambda *a, **k: plain)
    monkeypatch.delenv("KINDI_SPEC", raising=False)
    said = []
    assert verify.run({"pack": pack}, log=said.append) == 0
    assert said[0] == "speculation: off (KINDI_SPEC=0)"
    assert plain.spec_begin_calls == [] and plain.spec_end_calls == 0 and plain.generate_calls == 1


def test_a_refused_mode_stops_before_any_generation(monkeypatch, tmp_path):
    """REQ-0025 A4 through `run`: exit 2 and a FAIL line naming the reason, no prompt generated."""
    pack, rp = _reference(tmp_path)
    monkeypatch.setattr(verify, "reference_for", lambda p: rp)
    eng = _Gen(_mtp_heads={"mtp": REFUSED})
    monkeypatch.setattr("kindi.engine.make_engine", lambda *a, **k: eng)
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    said = []
    assert verify.run({"pack": pack}, log=said.append) == 2
    assert any(s.startswith("FAIL: mtp was requested") for s in said)
    assert eng.generate_calls == 0
