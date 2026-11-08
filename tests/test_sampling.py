"""Sampler order: top-k, then top-p, then temperature (llama.cpp), and the hf order."""
import math

import numpy as np

from kindi import sampling
from kindi.sampling import Sampler, penalize


class RecordingRng(object):
    def __init__(self):
        self.idx = None
        self.p = None

    def choice(self, idx, p=None):
        self.idx = [int(i) for i in np.asarray(idx)]
        self.p = None if p is None else [float(x) for x in np.asarray(p)]
        return self.idx[0]


def pick(z, cand=None, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0):
    cand = list(range(len(z))) if cand is None else cand
    rng = RecordingRng()
    out = Sampler._pick(np.asarray(cand, dtype=np.int64), np.asarray(z, dtype=np.float64),
                        temperature, top_p, top_k, rng, min_p)
    return out, rng


def test_top_k_keeps_the_k_best_candidates():
    out, rng = pick([3.0, 2.0, 1.0, 0.0], top_k=2)
    assert set(rng.idx) == {0, 1}
    assert out in (0, 1)


def test_top_p_keeps_the_nucleus():
    _, rng = pick([math.log(0.6), math.log(0.3), math.log(0.1)], top_p=0.8)
    assert set(rng.idx) == {0, 1}


def test_hf_order_applies_temperature_before_top_p(monkeypatch):
    z = [math.log(0.6), math.log(0.3), math.log(0.1)]
    _, llama = pick(z, temperature=0.4, top_p=0.8)
    monkeypatch.setattr(sampling, "SAMPLER_ORDER", "hf")
    _, hf = pick(z, temperature=0.4, top_p=0.8)
    assert set(llama.idx) == {0, 1}
    assert set(hf.idx) == {0}


def test_min_p_drops_low_probability_tokens():
    _, rng = pick([math.log(0.8), math.log(0.15), math.log(0.05)], min_p=0.5)
    assert set(rng.idx) == {0}


def test_repeat_penalty_only_moves_recent_tokens():
    cand = np.array([1, 2, 3])
    z = np.array([2.0, -2.0, 1.0])
    assert penalize(cand, z, 2.0, [1, 2]).tolist() == [1.0, -4.0, 1.0]


def test_greedy_bypasses_sampling():
    # a minimal engine object: Sampler takes the host path, and temperature 0 returns the argmax
    assert Sampler(object())(7, 0.0, 1.0, 0, None) == 7
