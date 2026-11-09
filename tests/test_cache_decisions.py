"""Prompt-cache decisions that need no engine: prefix matching, spacing, parking, eviction."""
import os
from types import SimpleNamespace

from kindi import cache
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


def test_arithmetic_fingerprint_includes_new_compiled_kernels(tmp_path):
    (tmp_path / "new_kernel.cu").write_text("// kernel\n", encoding="utf-8")
    (tmp_path / "new_kernel.c").write_text("/* kernel */\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not arithmetic\n", encoding="utf-8")
    files = cache._arith_files(str(tmp_path))
    assert "new_kernel.cu" in files and "new_kernel.c" in files
    assert "notes.txt" not in files
    assert "engine.py" in files and "spec.py" in files


def test_engine_fingerprint_hashes_every_compiled_kernel():
    here = os.path.dirname(os.path.abspath(cache.__file__))
    _, description = cache.engine_fingerprint(SimpleNamespace(), {"head": "exact"})
    compiled = {name for name in os.listdir(here) if name.endswith((".cu", ".c"))}
    assert compiled <= set(description["code"])
    assert "engine.py" in description["code"]
