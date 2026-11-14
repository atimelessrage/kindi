"""Requirement-based tests for `kindi bench --json`: the numbers recorded in the JSON
are the numbers printed on stdout (read rate, generation at each position, placement
and context), so benchmark runs do not have to be scraped out of the human log.
"""
import json
import os

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

    def step(self, tok, pos):
        self.calls["step"] += 1


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
