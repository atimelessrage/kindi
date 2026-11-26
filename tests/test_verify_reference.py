"""kindi verify --reference: a reference file from anywhere is read and shape-checked before any engine
loads, and a release with no bundled reference now names that route instead of only refusing."""
import json
import os

from kindi import verify


def _ref(**body):
    return body


def test_the_bundled_reference_is_a_reference():
    path = os.path.join(os.path.dirname(os.path.abspath(verify.__file__)), "reference.json")
    ref = verify.load_reference(path, log=lambda *_: None)
    assert ref and len(ref["prompts"]) == 4
    assert all(r["ids"] and r["expected"] for r in ref["prompts"])


def test_a_user_reference_is_accepted(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"source": "llama.cpp", "prompts": [
        {"prompt": "hi", "ids": [1, 2], "expected": [3, 4]}]}), encoding="utf-8")
    ref = verify.load_reference(str(p), log=lambda *_: None)
    assert ref is not None and ref["prompts"][0]["expected"] == [3, 4]


def test_a_missing_file_names_itself(tmp_path):
    said = []
    assert verify.load_reference(str(tmp_path / "nope.json"), log=said.append) is None
    assert said and "nope.json" in said[0]


def test_wrong_shapes_are_refused_and_say_which_file(tmp_path):
    cases = {
        "empty": {},
        "no-prompts": {"prompts": []},
        "no-ids": {"prompts": [{"prompt": "x", "expected": [1]}]},
        "no-expected": {"prompts": [{"prompt": "x", "ids": [1]}]},
        "empty-ids": {"prompts": [{"prompt": "x", "ids": [], "expected": [1]}]},
        "not-a-dict-row": {"prompts": ["x"]},
    }
    for name, body in cases.items():
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(body), encoding="utf-8")
        said = []
        assert verify.load_reference(str(p), log=said.append) is None, name
        assert said and p.name in said[0], name


def test_run_refuses_a_missing_reference_without_loading_a_pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    said = []
    rc = verify.run({"pack": str(pack)}, log=said.append, reference=str(tmp_path / "absent.json"))
    assert rc == 2
    assert any("no such reference file" in line for line in said)


def test_run_refuses_a_malformed_reference_without_loading_a_pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"prompts": [{"prompt": "x"}]}), encoding="utf-8")
    said = []
    assert verify.run({"pack": str(pack)}, log=said.append, reference=str(bad)) == 2
    assert any("bad.json" in line for line in said)


class _FakeEngine(object):
    """Only what the reference loop touches: run() is exercised without a model."""

    def __init__(self, reply):
        self.reply = reply

    def generate(self, ids, n):
        return self.reply


def _run_with(tmp_path, monkeypatch, n_ids, expected, reply=None):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"prompts": [{"ids": [1] * n_ids, "expected": expected}]}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr("kindi.packcheck.check", lambda pack, mode, man: 0)

    def fake_engine(pack, ctx=None, **kw):
        seen["ctx"] = ctx
        return _FakeEngine(expected if reply is None else reply)

    monkeypatch.setattr("kindi.engine.make_engine", fake_engine)
    monkeypatch.delenv("KINDI_SPEC", raising=False)
    said = []
    rc = verify.run({"pack": str(pack)}, log=said.append, reference=str(ref))
    return rc, seen, said


def test_the_context_is_sized_from_the_longest_reference_row(tmp_path, monkeypatch):
    """The bundled rows fit in 1024 tokens; a user's may not, and the engine needs the whole row."""
    rc, seen, said = _run_with(tmp_path, monkeypatch, 1100, [2] * 8)
    assert rc == 0, said
    assert seen["ctx"] == 1116                 # 1100 prompt + 8 expected, plus the 8-token headroom
    assert any("user-supplied" in line for line in said)


def test_a_short_reference_keeps_the_default_context(tmp_path, monkeypatch):
    rc, seen, said = _run_with(tmp_path, monkeypatch, 30, [2] * 8)
    assert rc == 0, said
    assert seen["ctx"] == 1024


def test_the_verdict_names_the_file_it_scored_against(tmp_path, monkeypatch):
    """A PASS on a user's file must not read as a PASS against llama.cpp."""
    rc, _seen, said = _run_with(tmp_path, monkeypatch, 30, [2] * 8)
    assert rc == 0, said
    assert any(line.startswith("ALL PASS: token-exact with ref.json") for line in said), said
    assert not any("token-exact with llama.cpp" in line for line in said)


def test_a_mismatch_fails_and_names_the_token(tmp_path, monkeypatch):
    rc, _seen, said = _run_with(tmp_path, monkeypatch, 30, [2] * 8, reply=[2, 2, 9, 2, 2, 2, 2, 2])
    assert rc == 1
    assert any("first difference at token 2" in line for line in said), said


def test_the_cli_takes_the_flag_and_passes_it_on(monkeypatch):
    import types

    from kindi import __main__ as m

    parser = m._build_parser()
    assert parser.parse_args(["verify", "--reference", "ref.json"]).reference == "ref.json"
    assert parser.parse_args(["verify"]).reference is None

    seen = {}
    monkeypatch.setattr("kindi.verify.run",
                        lambda cfg, log=print, reference=None: seen.update(reference=reference) or 0)
    assert m.cmd_verify({"state_dir": "x"}, types.SimpleNamespace(reference="ref.json")) == 0
    assert seen == {"reference": "ref.json"}
