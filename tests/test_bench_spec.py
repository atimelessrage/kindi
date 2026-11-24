# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi bench --spec`: the mode it names, the acceptance it counts, the refusal it keeps (REQ-0026).

`kindi bench` used to build the plan engine with speculation off whatever `KINDI_SPEC` said, so no
speculation mode could be benchmarked at all. These tests pin the CPU-checkable halves: the first
line names the effective mode (same resolver as verify's `spec_state` and `/health`), the generation
curve runs inside `spec_begin`/`spec_end` so the JSON's `spec` block carries real drafted/accepted
counts and k, and a requested mode whose drafter did not attach exits 2 with no JSON row — never a
plain-engine measurement with a spec label.

The engine here is a stand-in: `spec_state` reads exactly the attributes a built engine sets
(`spec_begin`, `mtp`, `drafter`, `dspine`, `_mtp_heads`, `spec_k`), and no kernel runs.
"""
import importlib
import json
import sys
import types

import pytest

from kindi import bench


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


@pytest.fixture(scope="module", autouse=True)
def _spec_importable():
    _stub_cupy_if_needed()
    importlib.import_module("kindi.spec")


class FakeSpecEngine(object):
    """The `test_bench_json` engine plus the speculation surface a plan engine has."""

    def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
        self.ctx = ctx
        self.hsl = None
        self.shortlist = 0
        self.calls = {"feed": 0, "step": 0}
        self.drafter = None
        self.dspine = None
        self.mtp = object()
        self._mtp_heads = {"mtp": HEAD}
        self.spec_k = 0
        self.spec_begin_calls = []
        self.spec_end_calls = 0

    def spec_begin(self, greedy=True, *a, **k):
        self.spec_begin_calls.append(greedy)
        self.spec_k = 1

    def spec_end(self):
        self.spec_end_calls += 1
        return {"passes": 5, "drafted": 20, "accepted": 7}

    def reset(self):
        pass

    def feed(self, tok, pos):
        self.calls["feed"] += 1

    def feed_batch(self, toks, pos):
        self.calls["feed"] += len(toks)

    def step(self, tok, pos):
        self.calls["step"] += 1


def _run(tmp_path, monkeypatch, engine, spec=None, json_name="b.json", mutate=None):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / json_name)
    # bench calls make_engine as a factory; hand back a fresh engine built with the run's ctx
    made = []

    def _factory(*a, **k):
        eng = type(engine)(a[0] if a else k.get("pack"), ctx=k.get("ctx"),
                           clusters=k.get("clusters"), tandem=k.get("tandem"),
                           cpu_layers=k.get("cpu_layers"))
        if mutate is not None:
            mutate(eng)
        made.append(eng)
        return eng

    monkeypatch.setattr("kindi.engine.make_engine", _factory)
    if spec is None:
        monkeypatch.delenv("KINDI_SPEC", raising=False)
    else:
        monkeypatch.setenv("KINDI_SPEC", spec)
    logged = []
    rc = bench.run({"pack": str(tmp_path), "ctx": 4096, "fast_clusters": 256,
                    "split_at": "none", "json": dest}, log=logged.append)
    return rc, logged, dest, made[0] if made else engine


def test_bench_without_spec_names_off_and_calls_nothing(tmp_path, monkeypatch):
    engine = FakeSpecEngine(str(tmp_path))
    rc, logged, dest, engine = _run(tmp_path, monkeypatch, engine)
    assert rc == 0
    assert logged[0] == "speculation: off (KINDI_SPEC=0)"
    assert engine.spec_begin_calls == [] and engine.spec_end_calls == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["spec"]["requested"] == "0" and got["spec"]["mode"] is None
    assert got["spec"]["drafted"] is None and got["spec"]["k"] is None


def test_bench_under_a_mode_counts_its_drafts_in_the_json(tmp_path, monkeypatch):
    """REQ-0026 A1/A3: the curve runs under spec_begin/spec_end and the JSON carries the mode,
    the head, drafted/accepted, the ratio and the k the engine used."""
    engine = FakeSpecEngine(str(tmp_path))
    rc, logged, dest, engine = _run(tmp_path, monkeypatch, engine, spec="mtp")
    assert rc == 0
    assert "speculation: requested mtp, effective mtp" in logged[0]
    assert "kindi-mtp-2b-v1.npz sha256 1b5c0c790a59" in logged[0]
    assert engine.spec_begin_calls == [True] and engine.spec_end_calls == 1
    assert any("speculation mtp: 7/20 drafts accepted (35.0%, k=1)" in line for line in logged)
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["spec"]["mode"] == "mtp" and got["spec"]["requested"] == "mtp"
    assert got["spec"]["head"] == "kindi-mtp-2b-v1.npz"
    assert got["spec"]["head_sha256"] == "1b5c0c790a59"
    assert got["spec"]["drafted"] == 20 and got["spec"]["accepted"] == 7
    assert got["spec"]["acceptance_ratio"] == 0.35
    assert got["spec"]["k"] == 1


def test_a_refused_mode_exits_2_with_no_json_and_no_timing(tmp_path, monkeypatch):
    """REQ-0026 A2: a mode whose drafter did not attach is a named failure, not a plain row."""
    engine = FakeSpecEngine(str(tmp_path))

    def _refuse(eng):
        eng._mtp_heads = {"mtp": REFUSED}
        eng.mtp = None

    rc, logged, dest, engine = _run(tmp_path, monkeypatch, engine, spec="mtp", mutate=_refuse)
    assert rc == 2
    assert any(line.startswith("FAIL: mtp was requested") for line in logged)
    import os
    assert not os.path.exists(dest)          # no row a reader could mistake for a spec measurement
    assert engine.calls["feed"] == 0 and engine.calls["step"] == 0
    assert engine.spec_begin_calls == [] and engine.spec_end_calls == 0


class PlainEngine(object):
    """The bench surface bench.py drives, with no speculation surface at all."""

    def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
        self.ctx = ctx
        self.hsl = None
        self.shortlist = 0

    def reset(self):
        pass

    def feed(self, tok, pos):
        pass

    def feed_batch(self, toks, pos):
        pass

    def step(self, tok, pos):
        pass


def test_a_spec_request_on_an_engine_without_speculation_is_a_refusal(tmp_path, monkeypatch):
    assert not hasattr(PlainEngine(str(tmp_path)), "spec_begin")
    engine = PlainEngine(str(tmp_path))
    rc, logged, dest, engine = _run(tmp_path, monkeypatch, engine, spec="mtp")
    assert rc == 2
    assert any("FAIL:" in line and "cannot speculate" in line for line in logged)
    import os
    assert not os.path.exists(dest)


def test_the_cli_flag_selects_the_mode(monkeypatch, tmp_path):
    """`kindi bench --spec mtp` is `KINDI_SPEC=mtp` for the run, and nothing else changes."""
    import kindi
    main = importlib.import_module("kindi.__main__")
    seen = {}
    monkeypatch.setattr("kindi.bench.run", lambda cfg, log=print: seen.update(called=True) or 0)
    monkeypatch.setenv("KINDI_SPEC", "0")   # cmd_bench writes the process env; keep it from leaking
    assert main.cmd_bench({"state_dir": str(tmp_path)}, types.SimpleNamespace(spec="mtp")) == 0
    assert seen.get("called") is True and main.os.environ["KINDI_SPEC"] == "mtp"
    assert kindi  # the package imported, not a stub


def test_bench_help_advertises_the_spec_flag():
    import subprocess
    out = subprocess.run([sys.executable, "-m", "kindi", "bench", "--help"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0
    assert "--spec" in out.stdout
