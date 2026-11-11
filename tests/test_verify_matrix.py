"""The tc/plain matrix must construct deterministic cells and parse verify output without a GPU."""
import importlib.util
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