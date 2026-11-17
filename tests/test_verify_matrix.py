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


def test_tc_hp_mode_adds_only_the_high_precision_flag():
    mod = load()
    tc = mod.mode_env("tc", cards=1)
    tc_hp = mod.mode_env("tc-hp", cards=1)
    assert "KINDI_TC_HP" not in tc
    assert tc_hp["KINDI_TC_HP"] == "1"
    assert {k: v for k, v in tc_hp.items() if k != "KINDI_TC_HP"} == tc
    assert "tc-hp" in mod.MODES


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


def test_regression_gate_rejects_status_exactness_and_throughput_losses():
    mod = load()
    baseline = {"results": [_row(status="pass", tps=40.0, passes=3, ties=1, fails=0)]}
    current = [_row(status="fail", tps=30.0, passes=2, ties=0, fails=2)]
    comparisons, baseline_only = mod.compare_results(current, baseline)
    failures = mod.regression_failures(comparisons, baseline_only, max_tps_drop_pct=10)
    assert any("status pass -> fail" in failure for failure in failures)
    assert any("exactness 3/1/0 -> 2/0/2" in failure for failure in failures)
    assert any("25.0% drop" in failure for failure in failures)


def test_regression_gate_allows_improvements_and_a_new_cell():
    mod = load()
    baseline = {"results": [_row(status="pass", tps=30.0, passes=2, ties=2, fails=0)]}
    current = [
        _row(status="pass", tps=35.0, passes=3, ties=1, fails=0),
        _row(status="pass", tps=20.0, passes=4, ties=0, fails=0),
    ]
    current[1]["model"] = "4b"
    comparisons, baseline_only = mod.compare_results(current, baseline)
    assert mod.regression_failures(comparisons, baseline_only, max_tps_drop_pct=10) == []


def test_regression_gate_rejects_missing_cells_and_throughput():
    mod = load()
    baseline = {"results": [
        _row(status="pass", tps=40.0, passes=4, ties=0, fails=0),
        {**_row(status="pass", tps=40.0), "mode": "plain"},
    ]}
    current = [_row(status="pass", tps=None, passes=4, ties=0, fails=0)]
    comparisons, baseline_only = mod.compare_results(current, baseline)
    failures = mod.regression_failures(comparisons, baseline_only, max_tps_drop_pct=10)
    assert any("throughput missing" in failure for failure in failures)
    assert any("27b/plain: baseline cell missing" in failure for failure in failures)


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


def test_fail_on_regression_cli_exits_nonzero(tmp_path, capsys, monkeypatch):
    mod = load()
    packs = tmp_path / "packs"
    pack = packs / "bonsai-27b"
    pack.mkdir(parents=True)
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"results": [_row(status="pass", tps=40.0)]}), encoding="utf-8")

    monkeypatch.setattr(mod, "run_cell", lambda *args, **kwargs: _row(status="pass", tps=30.0))
    rc = mod.main([
        "--models", "27b", "--modes", "tc", "--pack-dir", str(packs),
        "--out", str(out), "--baseline", str(baseline),
        "--fail-on-regression", "--max-tps-drop-pct", "10",
    ])

    assert rc == 1
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert len(comparison["regressions"]) == 1
    output = capsys.readouterr().out
    assert "regression gate: FAIL" in output
    assert "25.0% drop" in output


def test_fail_on_regression_needs_a_baseline(tmp_path):
    mod = load()
    try:
        mod.main(["--models", "27b", "--modes", "tc", "--pack-dir", str(tmp_path),
                  "--out", str(tmp_path / "out"), "--fail-on-regression"])
    except SystemExit as exc:
        assert "needs --baseline" in str(exc)
    else:
        raise AssertionError("expected a SystemExit")


def test_environment_records_the_card_driver_and_pack_hash(tmp_path, monkeypatch):
    # a rig result without the card and the pack hash cannot be entered into the support matrix
    mod = load()
    pack = tmp_path / "bonsai-9b"
    pack.mkdir()
    manifest = pack / "kindi_pack.json"
    manifest.write_text(json.dumps({"source_sha256": "abc123", "hadamard_block": 4096}), encoding="utf-8")
    (pack / "w.bin").write_bytes(b"x" * 10)
    import subprocess
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "0, Quadro P4000, 535.104.05, 8192, 6.1\n", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    record = mod.environment([pack])
    assert record["gpus"] == [{"index": "0", "name": "Quadro P4000", "driver": "535.104.05",
                               "memory_mib": 8192, "compute_capability": "6.1"}]
    entry = record["packs"][0]
    assert entry["source_sha256"] == "abc123"
    assert entry["hadamard_block"] == 4096
    assert entry["bytes"] == manifest.stat().st_size + 10
    described = mod._describe_environment(record)
    assert "Quadro P4000 sm 6.1, driver 535.104.05, 8192 MiB" in described


def test_environment_without_nvidia_smi_is_recorded_not_fatal(tmp_path, monkeypatch):
    # a CPU-only host (or a container without the tool) still writes a comparable results.json
    mod = load()

    def fake_run(command, **kwargs):
        if command and command[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise AssertionError(command)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    record = mod.environment([])
    assert record["gpus"] == []
    assert record["packs"] == []
    assert mod._describe_environment(record).startswith("no card detected")


def test_results_json_records_the_environment(tmp_path, monkeypatch, capsys):
    mod = load()
    packs = tmp_path / "packs"
    pack = packs / "bonsai-2b"
    pack.mkdir(parents=True)
    (pack / "kindi_pack.json").write_text(json.dumps({"source_sha256": "def456"}), encoding="utf-8")
    monkeypatch.setattr(mod, "run_cell", lambda *args, **kwargs: _row(status="pass", tps=30.0))
    monkeypatch.setattr(mod, "_gpus", lambda: [{"index": "0", "name": "Quadro M4000", "driver": "535.104.05",
                                                "memory_mib": 8192, "compute_capability": "5.2"}])
    rc = mod.main(["--models", "2b", "--modes", "plain", "--pack-dir", str(packs),
                   "--out", str(tmp_path / "out")])
    assert rc == 0
    payload = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert payload["environment"]["gpus"][0]["name"] == "Quadro M4000"
    assert payload["environment"]["packs"][0]["source_sha256"] == "def456"
    assert "recorded: Quadro M4000 sm 5.2" in capsys.readouterr().out


def test_offered_uses_the_catalog_fit():
    mod = load()
    assert mod.offered("2b", 900) is True
    assert mod.offered("2b", 899) is False
    assert mod.offered("27b", 8000) is True
    assert mod.offered("27b", 7000) is False
    assert mod.offered("2b", None) is None
    assert mod.offered("nope", 10**9) is None


def test_cached_tokens_reads_the_usage_details():
    mod = load()
    resp = {"usage": {"prompt_tokens_details": {"cached_tokens": 867}}}
    assert mod._cached_tokens(resp) == 867
    assert mod._cached_tokens({}) == 0
    assert mod._cached_tokens({"usage": {}}) == 0


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_serve_cycle_passes_on_repeats_invalid_and_restart(tmp_path, monkeypatch):
    mod = load()
    contents = ["reply"] * 4
    procs = []
    cached = [0, 400, 400, 400]

    class FakePopen:
        def __new__(cls, *a, **k):
            p = _FakeProc()
            procs.append(p)
            return p

    state = {"i": 0}

    def fake_http(port, path, body, timeout):
        if path == "/health":
            return 200, {}
        if body is not None and "messages" not in body:
            return 400, {}
        n = cached[state["i"]]
        text = contents[state["i"]]
        state["i"] += 1
        return 200, {"choices": [{"message": {"content": text}}],
                     "usage": {"prompt_tokens_details": {"cached_tokens": n}}}

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod, "_http", fake_http)
    row = mod.run_serve_cycle("2b", "plain", tmp_path / "bonsai-2b", tmp_path, 1, 5.0)
    assert row["status"] == "pass", row
    assert row["cached_tokens"] == [0, 400, 400, 400]
    assert row["invalid_status"] == 400
    assert row["contents_identical"] is True
    assert len(procs) == 2 and all(p.terminated for p in procs)


def test_serve_cycle_fails_without_a_cache_hit(tmp_path, monkeypatch):
    mod = load()
    cached = [0, 0, 0, 0]

    class FakePopen:
        def __new__(cls, *a, **k):
            return _FakeProc()

    state = {"i": 0}

    def fake_http(port, path, body, timeout):
        if path == "/health":
            return 200, {}
        if body is not None and "messages" not in body:
            return 400, {}
        n = cached[state["i"]]
        state["i"] += 1
        return 200, {"choices": [{"message": {"content": "same"}}],
                     "usage": {"prompt_tokens_details": {"cached_tokens": n}}}

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod, "_http", fake_http)
    row = mod.run_serve_cycle("2b", "plain", tmp_path / "bonsai-2b", tmp_path, 1, 5.0)
    assert row["status"] == "error"
    assert "not a cache hit" in row["summary"]


def test_serve_cycle_fails_on_divergent_replies(tmp_path, monkeypatch):
    mod = load()
    cached = [0, 400, 400, 400]
    texts = ["one", "two", "one", "one"]

    class FakePopen:
        def __new__(cls, *a, **k):
            return _FakeProc()

    state = {"i": 0}

    def fake_http(port, path, body, timeout):
        if path == "/health":
            return 200, {}
        if body is not None and "messages" not in body:
            return 400, {}
        n = cached[state["i"]]
        text = texts[state["i"]]
        state["i"] += 1
        return 200, {"choices": [{"message": {"content": text}}],
                     "usage": {"prompt_tokens_details": {"cached_tokens": n}}}

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod, "_http", fake_http)
    row = mod.run_serve_cycle("2b", "plain", tmp_path / "bonsai-2b", tmp_path, 1, 5.0)
    assert row["status"] == "error"
    assert "not bit-identical" in row["summary"]


def test_bench_row_records_the_json(tmp_path, monkeypatch):
    mod = load()
    data = {"read_tok_s": 42.5, "gen_tok_s": {"exact:8000": 37.6}, "ctx": 24576, "ctx_reduced": False}

    class Done:
        returncode = 0
        stdout = "bench ok"
        stderr = ""

    def fake_run(command, **kwargs):
        dest = command[command.index("--json") + 1]
        import json as _json
        Path(dest).write_text(_json.dumps(data), encoding="utf-8")
        return Done()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    row = mod.run_bench_row("2b", "plain", tmp_path / "bonsai-2b", tmp_path, 1, 5.0)
    assert row["status"] == "pass"
    assert row["bench"]["gen_tok_s"]["exact:8000"] == 37.6


def test_bench_row_fails_without_json(tmp_path, monkeypatch):
    mod = load()

    class Done:
        returncode = 1
        stdout = "boom"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: Done())
    row = mod.run_bench_row("2b", "plain", tmp_path / "bonsai-2b", tmp_path, 1, 5.0)
    assert row["status"] == "error"


def test_dry_run_lists_serve_and_bench(tmp_path, capsys):
    mod = load()
    rc = mod.main(["--dry-run", "--models", "2b", "--modes", "plain",
                   "--pack-dir", str(tmp_path / "packs"), "--out", str(tmp_path / "results")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "would serve" in text and "would bench" in text
    rc = mod.main(["--dry-run", "--models", "2b", "--modes", "plain", "--no-serve", "--no-bench",
                   "--pack-dir", str(tmp_path / "packs"), "--out", str(tmp_path / "results")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "would serve" not in text and "would bench" not in text


def test_spec_modes_chain_from_each_engine_mode_and_change_only_kindi_spec():
    mod = load()
    for base in mod.SPEC_BASES:
        base_env = mod.mode_env(base, cards=1)
        for spec in mod.SPEC_MODES:
            env = mod.mode_env(base + "-" + spec, cards=1)
            assert env["KINDI_SPEC"] == spec
            assert {k: v for k, v in env.items() if k != "KINDI_SPEC"} == \
                {k: v for k, v in base_env.items() if k != "KINDI_SPEC"}


def test_spec_env_names_only_the_files_the_mode_uses(tmp_path):
    mod = load()
    head = tmp_path / "kindi-mtp-2b-v1.npz"
    head.write_bytes(b"x")
    pack = tmp_path / "bonsai-2b"
    pack.mkdir()
    assert mod.spec_env("ngram", head, pack, head) == {}
    assert mod.spec_env("mtp", head, pack, head) == {"KINDI_MTP_HEAD": str(head)}
    assert mod.spec_env("draft2b", head, pack, head) == {"KINDI_SPEC_PACK": str(pack)}
    assert mod.spec_env("both", head, pack, head) == {
        "KINDI_SPEC_PACK": str(pack),
        "KINDI_SPEC_DRAFT_HEAD": str(head),
    }


def test_spec_missing_is_empty_when_the_files_exist_and_names_what_is_absent(tmp_path):
    mod = load()
    head = tmp_path / "head.npz"
    head.write_bytes(b"x")
    pack = tmp_path / "bonsai-2b"
    pack.mkdir()
    assert mod.spec_missing("ngram", None, None, None) == []
    assert mod.spec_missing("mtp", head, pack, head) == []
    assert mod.spec_missing("draft2b", None, pack, None) == []
    assert mod.spec_missing("both", None, pack, head) == []
    assert "no MTP head" in mod.spec_missing("mtp", None, pack, None)[0]
    assert "no draft head" in mod.spec_missing("both", head, pack, None)[0]
    assert "no 2B drafter pack" in mod.spec_missing("draft2b", head, None, None)[0]


def test_parse_spec_mode_scores_the_effective_mode_and_drafts():
    mod = load()
    text = """[kindi] speculation: requested mtp, effective mtp, head kindi-mtp-2b-v1.npz sha256 1b5c0c790a59
[kindi] prompt 1 'A' PASS  40/40 tokens match  (0.1s, 5/12 drafts accepted)
[kindi] prompt 2 'B' PASS  40/40 tokens match  (0.1s, 3/8 drafts accepted)
[kindi] ALL PASS: token-exact with llama.cpp with KINDI_SPEC=mtp - 100.0 tokens/s over the run
"""
    out = mod.parse_verify_output(text, 0)
    assert out["status"] == "pass"
    assert out["spec_requested"] == "mtp" and out["spec_effective"] == "mtp"
    assert (out["drafts_accepted"], out["drafts_drafted"]) == (8, 20)


def test_parse_a_spec_fallback_is_refused_not_a_missing_reference():
    mod = load()
    text = """[kindi] speculation: requested draft2b, effective ngram, the 2B draft pack is not attached
[kindi] FAIL: draft2b was requested but the effective mode is ngram: the 2B draft pack is not attached
"""
    out = mod.parse_verify_output(text, 2)
    assert out["status"] == "refused"
    assert out["spec_requested"] == "draft2b" and out["spec_effective"] == "ngram"


def test_run_cell_passes_the_spec_flag_and_records_the_scores(tmp_path, monkeypatch):
    mod = load()
    pack = tmp_path / "bonsai-2b"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    head = tmp_path / "head.npz"
    seen = {}

    class Done:
        returncode = 0
        stdout = (
            "[kindi] speculation: requested mtp, effective mtp, head head.npz sha256 abc\n"
            "[kindi] prompt 1 'A' PASS  40/40 tokens match  (0.1s, 2/3 drafts accepted)\n"
            "[kindi] ALL PASS: token-exact with llama.cpp with KINDI_SPEC=mtp - 10.0 tokens/s over the run\n"
        )
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return Done()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    row = mod.run_cell("2b", "tc-mtp", pack, tmp_path, 1, 5.0,
                       extra_env={"KINDI_MTP_HEAD": str(head)})
    assert seen["command"][-2:] == ["--spec", "mtp"]
    assert seen["env"]["KINDI_SPEC"] == "mtp" and seen["env"]["KINDI_MTP_HEAD"] == str(head)
    assert row["status"] == "pass"
    assert row["spec_requested"] == "mtp" and row["spec_effective"] == "mtp"
    assert (row["drafts_accepted"], row["drafts_drafted"]) == (2, 3)


def test_run_cell_refuses_a_pass_that_never_named_the_mode(tmp_path, monkeypatch):
    mod = load()
    pack = tmp_path / "bonsai-2b"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")

    class Done:
        returncode = 0
        stdout = "[kindi] ALL PASS: token-exact with llama.cpp - 10.0 tokens/s over the run\n"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: Done())
    row = mod.run_cell("2b", "tc-ngram", pack, tmp_path, 1, 5.0)
    assert row["status"] == "refused"
    assert row["spec_requested"] == "ngram" and row["spec_effective"] is None


def test_dry_run_lists_the_spec_cells_commands_and_skips(tmp_path, capsys):
    mod = load()
    packs = tmp_path / "packs"
    pack = packs / "bonsai-2b"
    pack.mkdir(parents=True)
    (pack / "kindi_pack.json").write_text("{}", encoding="utf-8")
    head = tmp_path / "head.npz"
    head.write_bytes(b"x")
    rc = mod.main(["--dry-run", "--models", "2b", "--modes", "plain",
                   "--spec-modes", "ngram,mtp", "--spec-models", "2b",
                   "--pack-dir", str(packs), "--mtp-head", str(head),
                   "--out", str(tmp_path / "out")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "2b tc-ngram" in text and "2b tc-mtp" in text
    assert "--spec ngram" in text and "KINDI_MTP_HEAD=%s" % head in text
    rc = mod.main(["--dry-run", "--models", "2b", "--modes", "plain",
                   "--spec-modes", "mtp", "--spec-models", "2b",
                   "--pack-dir", str(packs), "--mtp-head", str(tmp_path / "absent.npz"),
                   "--out", str(tmp_path / "out")])
    assert rc == 0
    assert "would skip: 2b tc-mtp" in capsys.readouterr().out
