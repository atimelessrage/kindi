"""`kindi autotune`: candidates come from the hardware, scores come from real bench runs only."""
import argparse
import json
import os

from kindi import autotune, io_probe

ONE_CARD = {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4060 Ti", "compute_cap": "8.9"}],
            "gpu_source": "nvidia-smi", "cpu": {"model": "Intel i7", "physical": 6, "logical": 12}}
TWO_CARDS = {"gpus": [{"index": 0, "name": "RTX 3080", "compute_cap": "8.6"},
                      {"index": 1, "name": "RTX 3080", "compute_cap": "8.6"}],
             "gpu_source": "nvidia-smi", "cpu": {"model": "Intel i7", "physical": 6, "logical": 12}}
NO_CARD = {"gpus": [], "gpu_source": "none", "cpu": {"model": "Intel i7", "physical": 6,
                                                     "logical": 12}}
DENSE = {"model": "bonsai2-2b"}
LIGHTNING = {"model": "nemotron-lightning", "backend": "kindi_lightning"}


def _names(options):
    return [item["name"] for item in options]


def test_one_card_gets_no_two_card_split_and_no_cpu_run():
    options = autotune.candidates(ONE_CARD, DENSE, 8192, max_runs=10)
    names = _names(options)
    assert names[0] == "gpu0"
    assert not [n for n in names if n.startswith("tp-")]
    assert "cpu" not in names
    assert "tandem-auto" in names


def test_one_card_offers_both_tandem_head_directions():
    """cpu-first and gpu-first own different heads, so the measured search must include both.

    The candidate is real bench flags, not a label: it runs gpu-first with the automatic slice.
    """
    options = autotune.candidates(ONE_CARD, DENSE, 8192, max_runs=10)
    option = next(item for item in options if item["name"] == "tandem-gpu-first")
    assert option["args"] == ["--tandem", "gpu-first", "--tandem-cpu-layers", "auto"]
    assert "output head" in option["why"]


def test_two_cards_get_both_tensor_parallel_splits():
    names = _names(autotune.candidates(TWO_CARDS, DENSE, 8192, max_runs=10))
    assert names[:3] == ["gpu0", "tp-2-auto", "tp-2-equal"]


def test_max_runs_truncates_and_keeps_the_most_likely_useful_first():
    options = autotune.candidates(TWO_CARDS, DENSE, 8192, max_runs=2)
    assert _names(options) == ["gpu0", "tp-2-auto"]


def test_a_moe_pack_gets_its_decode_cache_knob_and_no_dense_tandem():
    names = _names(autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10))
    assert "dcache-16" in names and "dcache-6" in names
    assert not [n for n in names if n.startswith("tandem")]


RATES = {"source": "kindi io", "ram_mib_s": 13456.7, "disk_mib_s": 2509.8,
         "ram_bytes_s": 13456.7 * 2 ** 20, "disk_bytes_s": 2509.8 * 2 ** 20,
         "pcie_h2d_gib_s": 6.0, "pcie_bytes_s": 6.0 * 2 ** 30}


def test_moe_slot_candidates_follow_the_rate_sized_count():
    options = autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32})
    names = _names(options)
    assert "dcache-16" in names and "dcache-8" in names and "dcache-32" in names
    assert "dcache-6" not in names
    why = next(item["why"] for item in options if item["name"] == "dcache-16")
    assert "sized from the measured I/O rates" in why
    assert "PCIe H2D 6.0 GiB/s" in why


def test_moe_cache_candidate_exposes_the_cold_tier_budget_and_rate_size():
    options = autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4,
                                       "n_card": 16, "expert_bytes": 1 << 20, "fit": True})
    option = next(item for item in options if item["name"].startswith("expert-cache-"))
    # 4 selected experts + ceil((RAM + disk) / PCIe) lookahead = 16 slots per layer
    # 16 hosts/layer x 4 MoE layers x 1 MiB/expert = 64 MiB
    assert option["name"] == "expert-cache-64"
    assert option["env"] == {"KINDI_LT_COLD": "1", "KINDI_LT_HOST_MB": "64"}
    assert "rate-sized slot working set" in option["why"]
    assert "PCIe H2D 6.0 GiB/s" in option["why"]


def test_autotune_reuses_a_saved_io_report_without_probing(tmp_path, monkeypatch):
    saved = tmp_path / "io.json"
    io_probe.save_report({
        "ram": {"available": True, "mib_s": 13456.7},
        "disk": {"available": True, "mib_s": 2509.8, "unbuffered": True},
        "pcie": {"available": True, "h2d_gib_s": 6.0, "d2h_gib_s": 6.1,
                 "device": "test"},
    }, str(saved))
    monkeypatch.setattr(io_probe, "probe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the probe ran")))
    lines = []
    rates = autotune.io_rates(argparse.Namespace(io_report=str(saved)), mib=8, log=lines.append)
    assert rates["pcie_bytes_s"] == 6.0 * 2 ** 30
    assert any("using saved I/O report" in line for line in lines)


def test_autotune_falls_back_when_the_saved_io_report_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(io_probe, "probe", lambda **kw: {
        "ram": {"available": True, "mib_s": 100.0},
        "disk": {"available": True, "mib_s": 50.0, "unbuffered": True},
        "pcie": {"available": True, "h2d_gib_s": 2.0, "d2h_gib_s": 2.1},
    })
    lines = []
    rates = autotune.io_rates(argparse.Namespace(io_report=str(tmp_path / "missing.json")),
                              mib=8, log=lines.append)
    assert rates["ram_bytes_s"] == 100.0 * 2 ** 20
    assert any("no saved I/O report at" in line and "measuring" in line for line in lines)


def test_moe_cache_candidate_survives_the_default_run_budget_on_two_cards():
    options = autotune.candidates(TWO_CARDS, LIGHTNING, 8192, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4,
                                       "n_card": 16, "expert_bytes": 1 << 20, "fit": True})
    names = _names(options)
    assert "expert-cache-64" in names
    assert len(options) == 6


def test_moe_cache_candidate_is_capped_by_the_host_tier():
    options = autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4,
                                       "n_card": 24, "expert_bytes": 1 << 20, "fit": True})
    option = next(item for item in options if item["name"].startswith("expert-cache-"))
    assert option["name"] == "expert-cache-32"       # 8 host experts/layer, not 16


def test_moe_cache_candidate_is_capped_by_available_host_ram(monkeypatch):
    # A 2 GiB host leaves 1 GiB after the default headroom (max(1 GiB, 10%)); the rate-sized
    # working set below would otherwise ask for 2 GiB.
    monkeypatch.setattr("kindi.hostram.available", lambda: 2 << 30)
    options = autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4,
                                       "n_card": 16, "expert_bytes": 32 << 20, "fit": True})
    option = next(item for item in options if item["name"].startswith("expert-cache-"))
    assert option["name"] == "expert-cache-1024"
    assert option["env"]["KINDI_LT_HOST_MB"] == "1024"


def test_moe_cache_candidate_is_omitted_when_host_ram_is_unknown(monkeypatch):
    """No free-RAM figure means no safe cap, so the candidate is not proposed at all."""
    monkeypatch.setattr("kindi.hostram.available", lambda: None)
    options = autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10, rates=RATES,
                                  moe={"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4,
                                       "n_card": 16, "expert_bytes": 1 << 20, "fit": True})
    assert not [item for item in options if item["name"].startswith("expert-cache-")]
    assert not [item for item in options
                if (item.get("env") or {}).get("KINDI_LT_HOST_MB")]
    assert "gpu0" in _names(options)           # the rest of the candidate list remains


def test_moe_cache_candidate_uses_the_plans_room_and_charges_its_slots():
    """The cache is sized from the same fit the engine uses, so a tight room lowers the proposal.

    With 100 MiB for experts, four MoE layers and 1 MiB experts, the staging buffers and seven
    slots per layer leave room for the four selected experts; an eighth slot does not. The host
    cache therefore covers seven host experts per layer (28 MiB), not the rate-only 16 (64 MiB).
    """
    moe = {"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4, "n_card": 16,
           "expert_bytes": 1 << 20, "room": 100 << 20, "fit": True}
    option = next(item for item in autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10,
                                                      rates=RATES, moe=moe)
                  if item["name"].startswith("expert-cache-"))
    assert option["name"] == "expert-cache-28"
    assert option["env"]["KINDI_LT_HOST_MB"] == "28"
    assert not [item for item in autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10,
                                                     rates=RATES, moe=dict(moe, room=24 << 20))
                if item["name"].startswith("expert-cache-")]


def test_moe_cache_candidate_needs_measured_rates_and_a_host_tier():
    moe = {"dcache": 16, "n_used": 4, "n_exp": 32, "n_moe": 4, "n_card": 16,
           "expert_bytes": 1 << 20, "fit": True}
    assert not [item for item in autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10,
                                                    moe=moe)
                if item["name"].startswith("expert-cache-")]
    assert not [item for item in autotune.candidates(ONE_CARD, LIGHTNING, 8192, max_runs=10,
                                                    rates=RATES, moe=dict(moe, n_card=32))
                if item["name"].startswith("expert-cache-")]


def test_a_host_without_a_card_only_offers_the_cpu_engine():
    assert _names(autotune.candidates(NO_CARD, DENSE, 8192)) == ["cpu"]


def test_include_cpu_adds_the_cpu_run_next_to_the_card():
    names = _names(autotune.candidates(ONE_CARD, DENSE, 8192, max_runs=10, include_cpu=True))
    assert names[-1] == "cpu"


def test_a_small_cpu_fleet_does_not_offer_a_tandem_it_cannot_feed():
    tiny = dict(ONE_CARD, cpu={"model": "Atom", "physical": 2, "logical": 4})
    names = _names(autotune.candidates(tiny, DENSE, 8192, max_runs=10))
    assert not [n for n in names if n.startswith("tandem")]


def test_bench_argv_puts_the_global_context_before_the_subcommand():
    argv = autotune.bench_argv(8192, "out.json", {"args": ["--tandem", "cpu-first"]},
                               python="py")
    assert argv[:6] == ["py", "-m", "kindi", "--ctx", "8192", "bench"]
    assert "--sample-vram" in argv                 # the fit side of a placement is measured too
    assert argv[-2:] == ["--tandem", "cpu-first"]


def test_bench_argv_carries_the_pack_autotune_was_given():
    argv = autotune.bench_argv(8192, "out.json", {"args": []}, python="py",
                               pack="T:/kindi-packs/bonsai-27b")
    assert argv[:7] == ["py", "-m", "kindi", "--pack", "T:/kindi-packs/bonsai-27b",
                        "--ctx", "8192"]
    assert argv[7] == "bench"


class _Done(object):
    def __init__(self, returncode=0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


def test_run_candidate_records_the_measured_numbers(monkeypatch, tmp_path):
    payload = {"read_tok_s": 790.2, "gen_tok_s": {"g100": 110.0, "g2k": 118.4},
               "ctx_reduced": False, "ctx_requested": 8192,
               "vram_peak": {"peak_used_mib": 1204, "peak_delta_mib": 900, "available": True}}

    def fake_run(argv, env=None, stdout=None, stderr=None, timeout=None):
        target = argv[argv.index("--json") + 1]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return _Done()

    monkeypatch.setattr(autotune.subprocess, "run", fake_run)
    result = autotune.run_candidate({"name": "gpu0", "args": [], "env": {}}, 8192, workdir=str(tmp_path))
    assert result["ok"] is True
    assert result["gen_tok_s"] == 118.4          # the fastest decode rate, not the first
    assert result["read_tok_s"] == 790.2
    assert result["ctx_reduced"] is False
    assert autotune._peak_mib(result["vram_peak"]) == 1204
    assert not os.path.exists(result["argv"][result["argv"].index("--json") + 1])


def test_run_candidate_keeps_the_env_it_was_asked_for(monkeypatch, tmp_path):
    seen = {}

    def fake_run(argv, env=None, stdout=None, stderr=None, timeout=None):
        seen.update(env or {})
        return _Done(returncode=1, stdout=b"kindi: no such pack\n")

    monkeypatch.setattr(autotune.subprocess, "run", fake_run)
    result = autotune.run_candidate({"name": "tp-2-auto", "args": [], "env": {"KINDI_TP": "1"}},
                                    8192, workdir=str(tmp_path))
    assert seen["KINDI_TP"] == "1"
    assert result["ok"] is False
    assert result["error"] == "kindi: no such pack"      # the child's own last line
    assert result["gen_tok_s"] is None                   # nothing is invented for a failed run


def test_a_failed_candidate_is_reported_without_a_number():
    results = [{"name": "cpu", "ok": False, "error": "kindi: the CPU engine does not have this pack"}]
    text = "\n".join(autotune.render(results, 8192, [], {"model": "x", "physical": 6}))
    assert "failed: kindi: the CPU engine does not have this pack" in text
    assert "nothing is recommended" in text


def test_ranking_puts_a_full_context_ahead_and_failures_last():
    results = [
        {"name": "reduced", "ok": True, "gen_tok_s": 200.0, "read_tok_s": 900.0, "ctx_reduced": True},
        {"name": "failed", "ok": False, "error": "boom"},
        {"name": "slow", "ok": True, "gen_tok_s": 40.0, "read_tok_s": 300.0, "ctx_reduced": False},
        {"name": "fast", "ok": True, "gen_tok_s": 90.0, "read_tok_s": 700.0, "ctx_reduced": False},
    ]
    assert [r["name"] for r in autotune.rank(results)] == ["fast", "slow", "reduced", "failed"]


def test_render_names_the_winner_and_the_command_that_reproduces_it():
    results = [
        {"name": "gpu0", "ok": True, "gen_tok_s": 80.0, "read_tok_s": 600.0, "ctx_reduced": False,
         "argv": [], "args": [], "env": {}},
        {"name": "tandem-gdn:8", "ok": True, "gen_tok_s": 90.0, "read_tok_s": 500.0,
         "ctx_reduced": False, "vram_peak": {"peak_used_mib": 3000}, "env": {},
         "args": ["--tandem", "cpu-first", "--tandem-cpu-layers", "gdn:8"]},
    ]
    text = "\n".join(autotune.render(results, 8192, ONE_CARD["gpus"], ONE_CARD["cpu"]))
    assert "best: tandem-gdn:8 - 90.0 tok/s decode at ctx 8192" in text
    assert "kindi --ctx 8192 serve --tandem cpu-first --tandem-cpu-layers gdn:8" in text
    assert "run `kindi verify`" in text


def test_render_names_the_pack_the_winner_was_measured_with():
    results = [{"name": "gpu0", "ok": True, "gen_tok_s": 90.0, "read_tok_s": 700.0,
                "ctx_reduced": False, "vram_peak": {"peak_used_mib": 3000}, "env": {},
                "args": []}]
    text = "\n".join(autotune.render(results, 8192, ONE_CARD["gpus"], ONE_CARD["cpu"],
                                      pack="T:/kindi-packs/bonsai-27b"))
    assert "serve it: kindi --pack T:/kindi-packs/bonsai-27b --ctx 8192 serve" in text


def test_render_shows_the_environment_of_an_env_only_winner():
    results = [{"name": "tp-2-auto", "ok": True, "gen_tok_s": 120.0, "read_tok_s": 900.0,
                "ctx_reduced": False, "args": [], "env": {"KINDI_TP": "1", "KINDI_TP_SPLIT": "auto"}}]
    text = "\n".join(autotune.render(results, 8192, TWO_CARDS["gpus"], TWO_CARDS["cpu"]))
    assert "serve it: KINDI_TP=1 KINDI_TP_SPLIT=auto kindi --ctx 8192 serve" in text
    assert "no flag for them" in text


def test_dry_run_plans_without_loading_anything(monkeypatch, tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text(json.dumps(DENSE), encoding="utf-8")
    monkeypatch.setattr(autotune, "candidates",
                        lambda *a, **k: [{"name": "gpu0", "why": "baseline", "args": [], "env": {}}])
    monkeypatch.setattr(autotune, "run_candidate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a dry run loaded a model")))
    lines = []
    args = argparse.Namespace(ctx=None, dry_run=True, max_runs=3, include_cpu=False,
                              timeout_s=60.0, json=None)
    rc = autotune.run({"pack": str(pack), "ctx": 8192}, args, log=lines.append)
    assert rc == 0
    assert any("dry run: nothing is loaded" in ln for ln in lines)
    assert any("gpu0" in ln and "baseline" in ln for ln in lines)


def test_a_missing_pack_is_a_loud_error_not_a_guess(tmp_path):
    lines = []
    args = argparse.Namespace(ctx=None, dry_run=True, max_runs=3, include_cpu=False,
                              timeout_s=60.0, json=None)
    rc = autotune.run({"pack": str(tmp_path / "nope"), "ctx": 8192}, args, log=lines.append)
    assert rc == 2
    assert any("no pack at" in ln for ln in lines)
