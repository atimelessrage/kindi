"""Prompt-cache decisions that need no engine: prefix matching, spacing, parking, eviction."""
from kindi.cache import Runner, _lcp


def bare_runner():
    """A Runner without __init__: only the attributes the decision under test reads."""
    return Runner.__new__(Runner)


def test_longest_common_prefix():
    assert _lcp([], []) == 0
    assert _lcp([1, 2, 3], [1, 2, 9]) == 2
    assert _lcp([1, 2], [1, 2, 3]) == 2
    assert _lcp([9], [1]) == 0


def test_dense_checkpoint_spacing():
    r = bare_runner()
    r.dense, r.dense_limit = 256, 1024
    assert r._dense(0)
    assert r._dense(256)
    assert not r._dense(300)
    assert not r._dense(1024)          # at the limit and above: only the evenly spaced checkpoints
    r.dense = 0
    assert not r._dense(256)


def test_short_conversations_are_not_parked():
    r = bare_runner()
    r.tokens, r.prefix_min, r.park_max = [0] * 511, 512, 3
    r._park()
    assert not hasattr(r, "parked")    # returned before touching any other cache state


def test_parking_disabled_keeps_the_active_conversation():
    r = bare_runner()
    r.tokens, r.prefix_min, r.park_max = [0] * 2048, 512, 0
    r._park()
    assert not hasattr(r, "parked")


def test_enforce_drops_the_oldest_checkpoints_first():
    r = bare_runner()
    r.ckpt = {100: object(), 200: object(), 300: object(), 400: object()}
    r.parked, r.keep_pos = [], []
    r.state_bytes, r.budget = 10, 25
    r.log = lambda *a, **k: None
    r._enforce()
    assert sorted(r.ckpt) == [300, 400]


def test_enforce_never_drops_an_opening_checkpoint():
    r = bare_runner()
    r.ckpt = {100: object(), 200: object(), 300: object(), 400: object()}
    r.parked, r.keep_pos = [], [100]
    r.state_bytes, r.budget = 10, 25
    r.log = lambda *a, **k: None
    r._enforce()
    assert 100 in r.ckpt
    assert len(r.ckpt) == 3            # the 3 newest are kept when nothing else can go
