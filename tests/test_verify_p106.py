# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for tools/verify_p106.py.

The fake-server test exercises the complete cache record (cold chat, saved opening,
in-process hit, restart, invalid request, metrics) without a GPU or model pack.
The hardware and verify-evidence tests pin the rules that keep a one-card or
non-P106 run from being recorded as a P106 result.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verify_p106.py"


def load():
    spec = importlib.util.spec_from_file_location("verify_p106", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CARDS_TEXT = """0, NVIDIA P106-100, 6077, 5900, 6.1, 3.0, 16, 580.88
1, NVIDIA P106-100, 6077, 6000, 6.1, 3.0, 16, 580.88
2, NVIDIA GeForce RTX 4060 Ti, 8188, 7000, 8.9, 4.0, 8, 580.88
"""


def test_parse_card_rows_keeps_the_p106_fields_and_no_identifiers():
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    assert [card["index"] for card in cards] == [0, 1, 2]
    assert cards[0]["name"] == "NVIDIA P106-100"
    assert cards[0]["memory_total_mib"] == 6077
    assert cards[0]["compute_cap"] == "6.1"
    assert cards[0]["pcie_width"] == "16"
    assert cards[0]["driver_version"] == "580.88"
    assert "uuid" not in cards[0] and "serial" not in cards[0]


def test_select_cards_requires_two_for_a_full_run():
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    selected, error = mod.select_cards(cards, None, False)
    assert not error
    assert [card["index"] for card in selected] == [0, 1]

    selected, error = mod.select_cards(cards, None, True)
    assert not error
    assert [card["index"] for card in selected] == [0]

    selected, error = mod.select_cards(cards[:1], None, False)
    assert selected == []
    assert "exactly two" in error

    selected, error = mod.select_cards(cards, "0", False)
    assert selected == []
    assert "exactly two" in error

    selected, error = mod.select_cards(cards, "0,2", False)
    assert selected == []
    assert "not a P106-100" in error

    selected, error = mod.select_cards(cards, "0,0", False)
    assert selected == []
    assert "twice" in error


def test_single_env_drops_stale_kindi_settings_and_pins_the_card():
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    env = mod.single_env([cards[0]], 24576, base={
        "PATH": "/bin", "KINDI_TC": "9", "KINDI_PACK": "/old", "CUDA_VISIBLE_DEVICES": "9",
    })
    assert env["PATH"] == "/bin"
    assert "KINDI_PACK" not in env
    assert env["KINDI_TC"] == "0"
    assert env["KINDI_TP"] == "0"
    assert env["KINDI_CARDS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_dual_env_is_the_legacy_two_card_profile():
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    env = mod.dual_env(cards[:2], 24576, base={"KINDI_TC": "1", "KINDI_TP": "1"})
    assert env["KINDI_CARDS"] == "2"
    assert env["KINDI_TC"] == "0"
    assert "KINDI_TP" not in env          # leave the 27B plan-engine choice to make_engine
    assert env["KINDI_CHECKPOINT_BUDGET_MB"] == "4096"
    assert env["KINDI_REPEAT_STOP"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_pack_for_supports_the_legacy_container_names(tmp_path):
    mod = load()
    legacy = tmp_path / "pack9b"
    legacy.mkdir()
    (legacy / "kindi_pack.json").write_text("{}", encoding="utf-8")
    assert mod.pack_for(tmp_path, "9b") == legacy


def test_verify_status_needs_a_named_p106_and_two_plan_cards():
    mod = load()
    status, note = mod.verify_status("pass", {"p106_seen": False, "plan_cards": None}, False)
    assert status == "fail" and "P106" in note

    status, note = mod.verify_status("pass", {"p106_seen": True, "plan_cards": 1}, True)
    assert status == "fail" and "two cards" in note

    assert mod.verify_status("pass", {"p106_seen": True, "plan_cards": None}, False) == ("pass", "")
    assert mod.verify_status("pass", {"p106_seen": True, "plan_cards": 2}, True) == ("pass", "")


class _FakeServer:
    def __init__(self, **kwargs):
        self.tag = kwargs["tag"]
        self.calls = 0
        self.saved = False
        self.log_path = kwargs["log_path"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_json(self, path):
        assert path == "/health"
        return {"ok": True, "model": "bonsai2-2b", "ctx": 24576, "head": "exact"}

    def chat(self, messages, n=16):
        self.calls += 1
        if self.tag == "server-save":
            if self.calls == 2:
                self.saved = True
            cached = 512 if self.calls == 3 else 0
        else:
            cached = 512
        return "same reply", {"prompt_tokens": 700, "prompt_tokens_details": {"cached_tokens": cached}}

    def post_json(self, path, body):
        if body.get("messages") == []:
            return 400, {"error": "messages must be a non-empty list"}
        return 200, {}

    def get_text(self, path):
        assert path == "/metrics"
        return "kindi_cached_prompt_tokens_total 1536\n"

    def lines(self):
        if self.tag == "server-save":
            return ["saved an opening of 700 tokens"] if self.saved else []
        return ["saved openings: 1 in /tmp/p106", "loaded a saved opening of 700 tokens"]


def test_server_cache_records_a_full_save_and_restart(tmp_path):
    mod = load()
    result = mod.run_server_cache(
        tmp_path / "bonsai-2b", tmp_path, {"PATH": "/bin"}, 24576, 30,
        server_cls=_FakeServer,
    )
    assert result["status"] == "pass", result["checks"]
    names = {check["name"]: check["ok"] for check in result["checks"]}
    for name in ("health", "cold chat", "parting after the opening saves it",
                 "in-process cache hit", "in-process reply is token-identical",
                 "restart loads the opening", "restart resumes token-identically",
                 "invalid request is 400", "invalid request keeps the cache",
                 "metrics report cached prompt tokens"):
        assert names[name], (name, result["checks"])
    assert result["checks"] and all(check["ok"] for check in result["checks"])


def test_run_verify_carries_the_generated_config_and_requires_evidence(tmp_path, monkeypatch):
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    env = mod.single_env(cards[:1], 24576, base={"PATH": "/bin"})
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="GPU: NVIDIA P106-100, sm_61\nALL PASS: token-exact with llama.cpp\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    config = tmp_path / "kindi.conf"
    row = mod.run_verify("2b", tmp_path / "bonsai-2b", tmp_path, env, 5, config_path=config)
    assert row["status"] == "pass"
    assert "--config" in calls[0] and str(config) in calls[0]
    assert row["config"]["KINDI_CARDS"] == "1"
    assert row["engine_evidence"]["p106_seen"] is True
    assert (tmp_path / "logs" / "single-2b.log").is_file()

    def fake_dual(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="plan engine ready: 2 cards, ctx 1024\nALL PASS: token-exact with llama.cpp\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_dual)
    row = mod.run_verify("27b", tmp_path / "bonsai-27b", tmp_path, env, 5, dual=True, config_path=config)
    assert row["status"] == "pass"
    assert row["engine_evidence"]["plan_cards"] == 2


def test_dry_run_writes_nothing_and_prints_both_card_phases(tmp_path, capsys, monkeypatch):
    mod = load()
    cards = mod.parse_card_rows(CARDS_TEXT)
    monkeypatch.setattr(mod, "query_cards", lambda: (cards, ""))
    out = tmp_path / "out"
    rc = mod.main(["--dry-run", "--pack-dir", str(tmp_path / "packs"), "--out", str(out)])
    assert rc == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "would run: single 2b" in text
    assert "KINDI_CARDS=1" in text
    assert "two-card 27b" in text
    assert "KINDI_CARDS=2" in text
    assert "KINDI_CHECKPOINT_BUDGET_MB=4096" in text


def test_results_are_written_as_json_and_markdown(tmp_path):
    mod = load()
    payload = {
        "started_utc": "2026-11-11T00:00:00Z",
        "duration_s": 1.5,
        "git": {"commit": "abc1234", "branch": "test", "dirty": False},
        "host": {"python": "3.13", "platform": "test", "driver_version": "580.88", "cuda_version": "12.8"},
        "preflight": {"selected_cards": mod.parse_card_rows(CARDS_TEXT)[:2], "visible_devices": "0,1"},
        "single": [{
            "phase": "single", "model": "2b", "status": "pass", "prompts_pass": 4,
            "prompts_tie": 0, "prompts_fail": 0, "tokens_per_second": 20.0,
            "exit_code": 0, "log": "logs/single-2b.log",
        }],
        "server_cache": {
            "status": "pass", "checks": [{"name": "health", "ok": True, "detail": "ctx=24576"}],
            "log": "logs/server-save.log", "restart_log": "logs/server-restart.log",
        },
        "dual": {
            "phase": "dual", "model": "27b", "status": "pass", "prompts_pass": 4,
            "prompts_tie": 0, "prompts_fail": 0, "tokens_per_second": 8.0,
            "engine_evidence": {"plan_cards": 2}, "log": "logs/dual-27b.log",
        },
        "verdict": "pass",
    }
    json_path, md_path = mod.write_results(tmp_path, payload)
    assert json.loads(json_path.read_text(encoding="utf-8"))["verdict"] == "pass"
    text = md_path.read_text(encoding="utf-8")
    assert "P106-100 verification record" in text
    assert "Single-card verify" in text
    assert "Server and prompt cache" in text
    assert "Two-card 27B" in text
    assert "| 2b | pass | 4/0/0 | 20.0 | 0 | logs/single-2b.log |" in text
