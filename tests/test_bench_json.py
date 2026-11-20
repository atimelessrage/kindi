"""Requirement-based tests for `kindi bench --json`: the numbers recorded in the JSON
are the numbers printed on stdout (read rate, generation at each position, placement
and context), so benchmark runs do not have to be scraped out of the human log.
"""
import json
import os
import time

import pytest

from kindi import bench


class FakeEngine(object):
    """An engine whose per-token time differs by branch, so the printed and recorded
    numbers can be compared cell by cell while the costs stay tiny."""

    def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
        self.ctx = ctx
        self.hsl = None
        self.shortlist = 0
        self.calls = {"feed": 0, "step": 0}

    def reset(self):
        pass

    def feed(self, tok, pos):
        self.calls["feed"] += 1

    def feed_batch(self, toks, pos):
        self.calls["feed"] += len(toks)
        time.sleep(0.0001)          # a nonzero prefill clock too, so the share is not 100% one way

    def step(self, tok, pos):
        self.calls["step"] += 1
        time.sleep(0.0002)          # a nonzero clock, so the split's shares are meaningful


def test_bench_json_records_the_printed_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "bench.json")
    logged = []
    monkeypatch.setattr("kindi.engine.make_engine", FakeEngine)
    rc = bench.run({"pack": str(tmp_path), "ctx": 24576, "fast_clusters": 256,
                    "json": dest}, log=logged.append)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["ctx"] == 24576
    assert got["ctx_requested"] == 24576
    assert got["ctx_reduced"] is False
    assert got["tandem"] == "off"
    assert got["device"] == "cpu"
    # every printed rate appears in the JSON, cell by cell
    text = "\n".join(logged)
    assert ("%.1f" % got["read_tok_s"]) in text
    for key, rate in got["gen_tok_s"].items():
        assert ("%.1f" % rate) in text
    assert "wrote %s" % dest in text
    assert sorted(k.split(":")[0] for k in got["gen_tok_s"]) == ["exact"] * len(got["gen_tok_s"])
    assert sorted(int(k.split(":")[1]) for k in got["gen_tok_s"]) == [100, 2000, 8000, 16000, 24000]


def test_bench_json_creates_a_missing_directory(tmp_path, monkeypatch):
    # a typo'd or new directory used to end the run with a traceback *after* the measurement, and a
    # 27B run costs two minutes; the row must land instead
    from kindi import bench
    dest = str(tmp_path / "new" / "nested" / "bench.json")
    bench.bench_json({"ok": True}, dest)
    assert json.loads(open(dest, encoding="utf-8").read()) == {"ok": True}


def test_bench_json_records_the_actual_ctx_when_reduced(tmp_path, monkeypatch):
    """When the loader reduces the context to fit, the JSON carries the actual ctx (from
    eng.ctx) and flags the reduction, so a 17,664-token run is never recorded as 24,576."""
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "b.json")

    class SmallEngine(FakeEngine):
        def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
            super().__init__(pack, ctx=ctx, clusters=clusters, tandem=tandem, cpu_layers=cpu_layers)
            self.ctx = 17664                # the loader reduced the requested 24,576 to fit

    monkeypatch.setattr("kindi.engine.make_engine", SmallEngine)
    rc = bench.run({"pack": str(tmp_path), "ctx": 24576, "fast_clusters": 256, "json": dest},
                   log=lambda *a, **k: None)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["ctx"] == 17664
    assert got["ctx_requested"] == 24576
    assert got["ctx_reduced"] is True


def test_bench_json_flags_even_a_small_reduction(tmp_path, monkeypatch):
    """A small shortfall is still a reduction (no tolerance): the loader only lowers the context
    when it does not fit, so any eng.ctx below the request is a real reduction."""
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "b.json")

    class SlightlyReduced(FakeEngine):
        def __init__(self, pack, ctx=None, clusters=None, tandem=None, cpu_layers=None, **_):
            super().__init__(pack, ctx=ctx, clusters=clusters, tandem=tandem, cpu_layers=cpu_layers)
            self.ctx = 24500

    monkeypatch.setattr("kindi.engine.make_engine", SlightlyReduced)
    rc = bench.run({"pack": str(tmp_path), "ctx": 24576, "fast_clusters": 256, "json": dest},
                   log=lambda *a, **k: None)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["ctx_reduced"] is True


def test_bench_json_records_tandem_placement(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "b.json")
    monkeypatch.setattr("kindi.engine.make_engine", FakeEngine)
    rc = bench.run({"pack": str(tmp_path), "ctx": 4096, "fast_clusters": 256,
                    "tandem": "cpu-first", "tandem_cpu_layers": 6, "json": dest}, log=lambda *a, **k: None)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert got["tandem"] == "cpu-first"
    assert got["cpu_layers"] == 6
    # ctx 4096 cannot hold positions 8000/16000/24000, and it skips them
    assert sorted(int(k.split(":")[1]) for k in got["gen_tok_s"]) == [100, 2000]


def test_bench_json_helper_writes_parseable_file(tmp_path):
    result = {"pack": "x", "model": "m", "ctx": 1024, "tandem": "off",
              "device": "gpu", "read_tok_s": 12.3, "gen_tok_s": {"exact:100": 4.5},
              "free_mib_after_load": 123, "total_mib": 8192}
    out = str(tmp_path / "r.json")
    bench.bench_json(result, out)
    assert json.loads(open(out, encoding="utf-8").read()) == result
    assert os.path.getsize(out) > 0


def test_bench_help_accepts_tandem_flags():
    """The module docstring promises verify/bench/serve/chat all accept --tandem; bench used not to."""
    from kindi import __main__ as m
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "kindi", "bench", "--help"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0
    assert "--tandem" in out.stdout and "--tandem-cpu-layers" in out.stdout
    assert "--dcache" in out.stdout


def test_bench_dcache_flag_sets_and_restores_the_engine_budget(tmp_path, monkeypatch):
    """The flag is the portable form of KINDI_LT_DCACHE: it reaches the engine for one run and the
    caller's environment is restored afterwards."""
    seen = {}

    class DcacheEngine(FakeEngine):
        def __init__(self, *args, **kwargs):
            seen["dcache"] = os.environ.get("KINDI_LT_DCACHE")
            super().__init__(*args, **kwargs)

    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    monkeypatch.setenv("KINDI_LT_DCACHE", "3")
    monkeypatch.setattr("kindi.engine.make_engine", DcacheEngine)
    rc = bench.run({"pack": str(tmp_path), "ctx": 1024, "fast_clusters": 256,
                    "split_at": "none", "dcache": 7}, log=lambda *_: None)
    assert rc == 0
    assert seen["dcache"] == "7"
    assert os.environ["KINDI_LT_DCACHE"] == "3"


def test_bench_reports_the_prefill_decode_split_of_one_request(tmp_path, monkeypatch):
    """`kindi bench` reports the share of wall clock, from one timed request, not two rates."""
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "split.json")
    monkeypatch.setattr("kindi.engine.make_engine", FakeEngine)
    logged = []
    rc = bench.run({"pack": str(tmp_path), "ctx": 8448, "fast_clusters": 256, "json": dest,
                    "split_at": "512,4096", "split_generated": 32}, log=logged.append)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert sorted(got["split"]) == ["4096", "512"]
    row = got["split"]["4096"]
    assert row["prompt_tokens"] == 4096 and row["generated_tokens"] == 32
    assert row["prefill_ms"] > 0 and row["decode_ms"] > 0
    assert row["prefill_share"] + row["decode_share"] == pytest.approx(100.0, abs=0.2)
    assert row["prefill_tok_s"] == pytest.approx(4096 / (row["prefill_ms"] / 1000.0), rel=0.01)
    assert row["decode_tok_s"] == pytest.approx(32 / (row["decode_ms"] / 1000.0), rel=0.01)
    assert "request split: 4096 prompt + 32 generated" in "\n".join(logged)


def test_a_split_that_does_not_fit_the_context_is_skipped_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_DEVICE", "cpu")
    dest = str(tmp_path / "split.json")
    monkeypatch.setattr("kindi.engine.make_engine", FakeEngine)
    logged = []
    rc = bench.run({"pack": str(tmp_path), "ctx": 1024, "fast_clusters": 256, "json": dest,
                    "split_at": "512,4096", "split_generated": 128}, log=logged.append)
    assert rc == 0
    got = json.loads(open(dest, encoding="utf-8").read())
    assert sorted(got["split"]) == ["512"]
    assert any("4096 prompt tokens skipped" in line for line in logged), logged


def test_split_prompts_parsing():
    assert bench.split_prompts({"split_at": "none"}) == []
    assert bench.split_prompts({"split_at": "off"}) == []
    assert bench.split_prompts({"split_at": ""}) == []
    assert bench.split_prompts({"split_at": "512, 4096"}) == [512, 4096]
    assert bench.split_prompts({"split_at": [1024]}) == [1024]
    assert bench.split_prompts({}) == list(bench.SPLIT_PROMPTS)
    for bad in ("4096,fast", "0", "-1", "512,0"):
        with pytest.raises(ValueError, match="positive integers.*none/off"):
            bench.split_prompts({"split_at": bad})


def test_bench_cli_rejects_unknown_comma_values_before_loading():
    import subprocess, sys
    for args, expected in ((["--split-at", "4096,fast"], "positive integers"),
                           (["--tandem-cpu-layers", "0,1,fast"], "layer spec"),
                           (["--vram-budget-mib", "-1"], "must not be negative")):
        out = subprocess.run([sys.executable, "-m", "kindi", "bench"] + args,
                             capture_output=True, text=True, timeout=30)
        assert out.returncode == 2, out.stderr
        assert expected in out.stderr, out.stderr
