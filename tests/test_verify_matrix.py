"""The tc/plain matrix must construct deterministic cells and parse verify output without a GPU."""
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verify_matrix.py"


def load():
    spec = importlib.util.spec_from_file_location("verify_matrix", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mode_env_is_explicit_and_complete():
    mod = load()
    plain = mod.mode_env("plain", cards=1)
    tc = mod.mode_env("tc", cards=1)
    for env in (plain, tc):
        assert env["KINDI_DEVICE"] == "gpu"
        assert env["KINDI_HEAD"] == "exact"
        assert env["KINDI_KV"] == "q8_0"
        assert env["KINDI_SPEC"] == "0"
        assert env["KINDI_CARDS"] == "1"
    assert plain["KINDI_TP"] == "0"
    assert plain["KINDI_TC"] == "0"
    assert tc["KINDI_TP"] == "1"
    assert tc["KINDI_TC"] == "1"
    assert tc["KINDI_FD"] == "1"
    assert tc["KINDI_TC_TMAX"] == "4"
    assert "KINDI_TC_TMAX" not in plain


def test_27b_pack_override_and_default():
    mod = load()
    assert "27b" in mod.MODELS
    assert mod.PACK_NAMES["27b"] == "bonsai-27b"
    base = Path("/packs")
    assert mod.pack_for("27b", base) == base / "bonsai-27b"
    custom = Path("/data/27b/bonsai-27b")
    assert mod.pack_for("27b", base, custom) == custom
    assert mod.pack_for("4b", base, custom) == base / "bonsai-4b"


def test_child_env_drops_stale_kindi_settings():
    mod = load()
    env = mod.child_env("tc", cards=1, base={"PATH": "/bin", "KINDI_PACK": "/old", "KINDI_TC": "9"})
    assert env["PATH"] == "/bin"
    assert "KINDI_PACK" not in env
    assert env["KINDI_TC"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_parse_pass_with_tie():
    mod = load()
    text = """[kindi 10:00:00] prompt 1 'A' PASS  40/40 tokens match  (0.1s)
[kindi 10:00:00] prompt 2 'B' TIE  39/40 tokens match  (0.1s)
[kindi 10:00:00]    first difference at token 24 (llama.cpp top-2 gap there 0.0157)
[kindi 10:00:00] PASS: 1 of 2 prompts token-exact, 1 end at a llama.cpp tie, 0 fail - 123.4 tokens/s over the run
C:/.../cupy/_environment.py:286: UserWarning: CUDA path could not be detected.
  warnings.warn(
"""
    out = mod.parse_verify_output(text, 0)
    assert out["status"] == "pass"
    assert [p["state"] for p in out["prompts"]] == ["PASS", "TIE"]
    assert (out["prompts_pass"], out["prompts_tie"], out["prompts_fail"]) == (1, 1, 0)
    assert out["tokens_per_second"] == 123.4
    assert out["first_diff_token"] == 24
    assert out["first_diff_gap"] == 0.0157
    assert "PASS:" in out["summary"]


def test_parse_hard_failure():
    mod = load()
    text = """[kindi 10:00:00] prompt 1 'A' PASS  40/40 tokens match  (0.1s)
[kindi 10:00:00] prompt 2 'B' FAIL  6/40 tokens match  (0.2s)
[kindi 10:00:00]    first difference at token 6 (llama.cpp top-2 gap there 0.0447)
[kindi 10:00:00] FAIL - 189.6 tokens/s over the run
"""
    out = mod.parse_verify_output(text, 1)
    assert out["status"] == "fail"
    assert (out["prompts_pass"], out["prompts_fail"]) == (1, 1)
    assert out["first_diff_token"] == 6
    assert out["first_diff_gap"] == 0.0447


def test_parse_refusal_is_not_an_exception():
    mod = load()
    text = ("kindi: the tensor-core plan engine supports Hadamard blocks of 512 and 1024 "
            "(this pack has 4096); use the plain engine (KINDI_TP=0)\n")
    out = mod.parse_verify_output(text, 1)
    assert out["status"] == "refused"
    assert out["tokens_per_second"] is None
    assert out["prompts"] == []
    assert "supports Hadamard blocks" in out["summary"]


def test_parse_timeout_and_missing_reference():
    mod = load()
    assert mod.parse_verify_output("TIMEOUT after 1.0 s\n", 124)["status"] == "timeout"
    assert mod.parse_verify_output("no llama.cpp reference for this model\n", 2)["status"] == "no-reference"


def test_dry_run_does_not_create_output(tmp_path, capsys):
    mod = load()
    out = tmp_path / "results"
    rc = mod.main(["--dry-run", "--models", "4b", "--modes", "tc",
                   "--pack-dir", str(tmp_path / "packs"), "--out", str(out)])
    assert rc == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "4b tc" in text
    assert "KINDI_TP=1" in text
    assert "verify" in text


def _row(*, status="pass", tps=38.0, passes=3, ties=1, fails=0):
    return {
        "model": "27b", "mode": "tc", "status": status,
        "prompts_pass": passes, "prompts_tie": ties, "prompts_fail": fails,
        "tokens_per_second": tps, "exit_code": 0 if status == "pass" else 1,
        "duration_s": 13.0, "first_diff_token": 14,
        "first_diff_gap": 0.0064, "summary": status, "log": "logs/27b-tc.log",
        "config": {}, "command": [],
    }


def test_compare_reports_status_change_and_throughput_delta():
    mod = load()
    baseline = {"results": [_row(status="pass", tps=40.0)]}
    current = [_row(status="fail", tps=30.0, passes=2, ties=0, fails=2)]
    comparisons, baseline_only = mod.compare_results(current, baseline)
    row = comparisons[0]
    assert baseline_only == []
    assert row["status_change"] == "pass->fail"
    assert row["baseline_exactness"] == "3/1/0"
    assert row["exactness"] == "2/0/2"
    assert row["exactness_change"] == "3/1/0->2/0/2"
    assert row["tokens_per_second_delta"] == -10.0
    assert row["tokens_per_second_delta_pct"] == -25.0
    assert mod._format_delta(row["tokens_per_second_delta"], row["tokens_per_second_delta_pct"]) == "-10.0 (-25.0%)"


def test_compare_marks_a_new_cell():
    mod = load()
    comparisons, baseline_only = mod.compare_results([_row()], {"results": []})
    row = comparisons[0]
    assert baseline_only == []
    assert row["status_change"] == "new"
    assert row["baseline_exactness"] is None
    assert row["tokens_per_second_delta"] is None


def test_save_baseline_and_compare_cli(tmp_path, capsys, monkeypatch):
    mod = load()
    packs = tmp_path / "packs"
    pack = packs / "bonsai-27b"
    pack.mkdir(parents=True)
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    baseline = tmp_path / "baseline.json"
    saved = tmp_path / "saved.json"
    baseline.write_text(json.dumps({
        "started_utc": "old", "git_revision": "old",
        "results": [_row(status="pass", tps=40.5)],
    }), encoding="utf-8")

    current = _row(status="pass", tps=38.1)
    monkeypatch.setattr(mod, "run_cell", lambda *args, **kwargs: current)
    rc = mod.main([
        "--models", "27b", "--modes", "tc", "--pack-dir", str(packs),
        "--out", str(out), "--baseline", str(baseline), "--save-baseline", str(saved),
    ])

    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    row = comparison["results"][0]
    assert row["status_change"] == "same"
    assert row["tokens_per_second_delta"] == -2.4
    assert (out / "comparison.csv").exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["results"][0]["tokens_per_second"] == 38.1
    output = capsys.readouterr().out
    assert "comparison vs" in output
    assert "-2.4" in output
    assert "saved baseline" in output


def test_dry_run_uses_27b_override(tmp_path, capsys):
    mod = load()
    out = tmp_path / "results"
    pack27 = tmp_path / "packs27" / "bonsai-27b"
    rc = mod.main(["--dry-run", "--models", "27b", "--modes", "tc",
                   "--pack-dir", str(tmp_path / "packs"), "--pack-27b", str(pack27),
                   "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "27b tc" in text
    assert str(pack27) in text
    assert "KINDI_TC_TMAX=4" in text