"""Sampler order: top-k, then top-p, then temperature (llama.cpp), and the hf order."""
import math

import numpy as np

from kindi import sampling
from kindi.sampling import Sampler, apply_presence, penalize


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


def test_presence_penalty_is_additive_once_per_token():
    # OpenAI semantics: a seen token loses pp once, no matter how often it appeared
    cand = np.array([1, 2, 3])
    z = np.array([2.0, 1.0, 0.0])
    seen = np.zeros(8, dtype=bool)
    seen[[1, 1, 2]] = True
    assert apply_presence(cand, z, 1.5, seen).tolist() == [0.5, -0.5, 0.0]


def test_negative_presence_penalty_rewards_seen_tokens():
    cand = np.array([1, 2])
    z = np.array([0.0, 0.0])
    seen = np.array([False, False, True], dtype=bool)      # token 2 is in the prompt
    assert apply_presence(cand, z, -1.0, seen).tolist() == [0.0, 1.0]


def test_presence_penalty_zero_is_a_no_op():
    z = np.array([1.0, 2.0])
    assert apply_presence(np.array([1, 2]), z, 0.0, np.ones(3, dtype=bool)) is z


def test_presence_penalty_changes_a_greedy_choice():
    class HostEngine(object):
        ylog = np.array([2.0, 1.0, 0.0])
        ylog_n = 3
        ylog_map = None

    seen = np.array([True, False, False], dtype=bool)
    assert Sampler(HostEngine())(0, 0.0, 1.0, 0, None, presence=(1.5, seen)) == 1


def test_greedy_bypasses_sampling():
    # a minimal engine object: Sampler takes the host path, and temperature 0 returns the argmax
    assert Sampler(object())(7, 0.0, 1.0, 0, None) == 7


class _ConstrainedHostEngine(object):
    def __init__(self):
        self.ylog = np.array([2.0, 1.0, 0.0])
        self.ylog_n = 3
        self.ylog_map = None


def test_allowed_set_overrides_greedy():
    # a grammar that accepts only token 1: greedy picks 1, not the argmax 0
    assert Sampler(_ConstrainedHostEngine())(0, 0.0, 1.0, 0, None, allowed=[1]) == 1


def test_allowed_set_with_temperature_stays_inside():
    rng = RecordingRng()
    out = Sampler(_ConstrainedHostEngine())(0, 1.0, 1.0, 0, rng, allowed=[1, 2])
    assert set(rng.idx) == {1, 2}
    assert out in (1, 2)


def test_allowed_none_leaves_uncostrained_behaviour():
    assert Sampler(_ConstrainedHostEngine())(0, 0.0, 1.0, 0, None, allowed=None) == 0


def test_allowed_empty_intersection_is_a_loud_error():
    try:
        Sampler(_ConstrainedHostEngine())(0, 0.0, 1.0, 0, None, allowed=[])
    except ValueError as e:
        assert "no allowed token" in str(e)
    else:
        raise AssertionError("an empty allowed set must raise, not silently free-pick")


class _CpuEngine(object):
    """The CPU engine's logit shape: the whole vocabulary in ``ylog``, and no ``ylog_map``."""

    def __init__(self, n):
        self.ylog = np.round(np.random.default_rng(7).standard_normal(n) * 1e3).astype(np.float32) / 1e3
        self.ylog_n = n
        self.ylog_map = None


def _old_constrained(allowed):
    """The generic path ``Sampler._constrained`` used before ``_masked_logits``."""
    eng = _CpuEngine(97)
    s = Sampler(eng)
    cand, z = s._logits()
    keep = np.isin(cand, np.asarray(allowed, dtype=np.int64))
    return cand[keep], z[keep]


def test_masked_logits_is_the_generic_selection():
    """The CPU fast path must be the same (ids, logits) pair as arange + np.isin, not a re-pick."""
    eng = _CpuEngine(97)
    s = Sampler(eng)
    for allowed in ([0, 5, 6, 96], list(range(0, 97, 7)), [3], [-4, 2, 200]):
        cand, z = s._masked_logits(np.asarray(allowed, dtype=np.int64))
        old_cand, old_z = _old_constrained(allowed)
        assert cand.tolist() == old_cand.tolist(), allowed
        assert z.tolist() == old_z.tolist(), allowed


def test_constrained_pick_matches_the_generic_path():
    """Same token from the fast path and the pre-change path, greedy and sampled."""
    eng = _CpuEngine(97)
    s = Sampler(eng)
    for allowed in ([0, 5, 6, 96], list(range(0, 97, 5)), [40, 41]):
        fast = s._constrained(np.asarray(allowed, dtype=np.int64), 0.0, 1.0, 0, RecordingRng(), 0.0,
                              None, None)
        cand, z = _old_constrained(allowed)
        assert fast == int(cand[int(np.argmax(z))])
    ids = np.asarray([0, 5, 6, 96], dtype=np.int64)
    rng_fast, rng_old = RecordingRng(), RecordingRng()
    fast = s._constrained(ids, 0.7, 1.0, 0, rng_fast, 0.0, None, None)
    cand, z = _old_constrained([0, 5, 6, 96])
    band = z >= z.max() - sampling.BAND_T * 0.7
    old = Sampler._pick(cand[band], z[band], 0.7, 1.0, 0, rng_old, 0.0)
    assert fast == old
    assert rng_fast.idx == rng_old.idx
