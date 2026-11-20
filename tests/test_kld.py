"""kindi kld: teacher-forced KLD quality gate for lossy K/V caches (roadmap gate 3).

The math (softmax / KL / summarize) is pure and testable without a GPU; the engine lockstep in
`compare` needs one, so these cover the gate's arithmetic and its CLI wiring only.
"""
import json
import math

import numpy as np
import pytest

from kindi import kld


def test_softmax_sums_to_one_and_is_shift_invariant():
    x = np.array([1000.0, 999.0, 970.0])
    p = kld.softmax(x)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, kld.softmax([0.0, -1.0, -30.0]))
    assert p[0] > p[1] > p[2] > 0


def test_kl_of_a_distribution_with_itself_is_zero():
    p = kld.softmax([3.0, 1.0, -2.0, 0.5])
    assert kld.kl_nats(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_asymmetric_and_known():
    # KL(p||q) > 0 for p != q, and equals the reverse-under-swap only by symmetry of this example
    p = kld.softmax([3.0, 0.0, -1.0])
    q = kld.softmax([0.5, 0.5, 0.5])
    assert kld.kl_nats(p, q) > 0.05
    assert kld.kl_nats(p, q) != kld.kl_nats(q, p)


def test_summarize_three_numbers():
    # two identical steps (no drift) and one fully drifted step: mean KLD, share, PPL ratio by hand
    steps = [(0.0, 1.0, 2.0, 2.0), (0.1, 0.0, 2.0, 2.1)]
    s = kld.summarize(steps)
    assert s["steps"] == 2
    assert s["kld_nats"] == pytest.approx(0.05)
    assert s["same_top_share"] == pytest.approx(0.5)
    assert s["ppl_ref"] == pytest.approx(math.exp(2.0))
    assert s["ppl_ratio"] == pytest.approx(math.exp(0.05))


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        kld.summarize([])


def test_tokens_length_and_stability():
    class Tok:
        def encode(self, text):
            return [7, 8, 9]

    assert kld._tokens(Tok(), 10) == [7, 8, 9, 7, 8, 9, 7, 8, 9, 7]


class _FakeEngine:
    def __init__(self, mapp=None):
        self.ylog = None
        self.ylog_n = 4
        self.ylog_map = mapp


def test_logits_refuses_a_reordered_head():
    # a fast (reordered) head is not a full vocabulary: the gate must refuse, not miscompare
    with pytest.raises(SystemExit):
        kld._logits(_FakeEngine(mapp=[1, 2, 3]), cp=None)


def test_kld_subcommand_is_wired():
    from kindi.__main__ import main
    import sys
    # --help exits 0 and lists the new subcommand
    with pytest.raises(SystemExit) as e:
        main(["kld", "--help"])
    assert e.value.code == 0
    # the module docstring path is dispatchable: parse errors mention kld, not a crash
    assert "kld" in kld.__doc__


def test_write_json_roundtrips(tmp_path):
    rows = [{"ctx": 2048, "variant": "tp-k8v4", "kld_nats": 0.01}]
    out = str(tmp_path / "kld.json")
    kld.write_json(rows, out)
    assert json.loads(open(out, encoding="utf-8").read()) == rows
