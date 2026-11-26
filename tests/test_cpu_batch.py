# SPDX-License-Identifier: AGPL-3.0-only
"""The batched T-row pass (roadmap CP1) must be plain decode, row for row.

The measured page is `docs/benchmarks/2026-11-26-i7-8086k-27b-cpu-batch-verify/`; what a fast test
can pin is the invariant that makes it usable at all: every row's argmax and every row's logits equal
what one-token-at-a-time decode produced, the GDN state after a rollback lands exactly where plain
decode would have it, and the state block a drafter rewinds with round-trips.

The engine tests are opt-in with KINDI_CPU_PACK (they load a real Bonsai pack and need a C compiler);
the guards are checked without either.
"""
import os

import numpy as np
import pytest

from kindi import cpu


def _pack():
    for cand in (os.environ.get("KINDI_CPU_PACK"), os.path.expanduser("~/.kindi/packs/bonsai-2b")):
        if cand and os.path.isfile(os.path.join(cand, "kindi_pack.json")):
            return cand
    return None


class _FakeLib(object):
    def __init__(self, rc=-1):
        self.rc, self.calls = rc, []

    def kc_step(self, m, tok, pos, head):
        self.calls.append(("step", tok, pos, head))
        return 5

    def kc_step_batch(self, m, toks, n, pos, out):
        self.calls.append(("batch", [toks[i] for i in range(n)], n, pos))
        return self.rc


def _fake_engine(rc=-1):
    eng = object.__new__(cpu.Engine)
    eng.m = cpu.Model()
    eng.ctx = 16
    eng.head_rows = None
    eng.head_off = 0
    eng.lib = _FakeLib(rc)
    return eng


def test_one_row_goes_through_the_plain_step():
    """A single row is not a batch; it must not need the batched entry point at all."""
    eng = _fake_engine()
    assert eng.step_batch([7], 3) == [5]
    assert [c[0] for c in eng.lib.calls] == ["step"]


def test_a_refused_batch_names_the_rows_and_the_code():
    eng = _fake_engine(rc=-3)
    with pytest.raises(RuntimeError, match=r"refused 3 rows at 4 \(code -3\)"):
        eng.step_batch([1, 2, 3], 4)


def test_a_batch_past_the_context_is_rejected_before_c():
    eng = _fake_engine()
    with pytest.raises(OverflowError):
        eng.step_batch([1, 2, 3], 15)


@pytest.fixture(scope="module")
def engine():
    pack = _pack()
    if pack is None:
        pytest.skip("no Bonsai pack (set KINDI_CPU_PACK to one)")
    eng = cpu.Engine(pack, ctx=64, threads=2, log=None)
    if not hasattr(eng.lib, "kc_step_batch"):
        pytest.skip("this CPU library predates the batched pass")
    return eng


def test_batch_rows_are_plain_decode(engine):
    """Every row's tokens and logits equal one-at-a-time decode; ids are arbitrary valid tokens."""
    ids = [1, 2, 3, 4, 5, 6]
    engine.reset()
    for i, t in enumerate(ids[:-1]):
        engine.feed(t, i)
    tok, pos, ref = ids[-1], len(ids) - 1, []
    for _ in range(4):
        tok = engine.step(tok, pos)
        ref.append((tok, engine.logits.copy()))
        pos += 1
    seq = ids + [r[0] for r in ref]

    engine.reset()
    for i, t in enumerate(ids[:-1]):
        engine.feed(t, i)
    out = engine.step_batch(seq[len(ids) - 1:len(ids) + 3], len(ids) - 1)
    assert out == [r[0] for r in ref]
    for r in range(4):
        np.testing.assert_array_equal(engine.batch_logits(r), ref[r][1])


def test_rollback_lands_where_plain_decode_would(engine):
    """A rejected row's GDN state is undone by the replay: the next plain rows then match decode."""
    ids = [1, 2, 3, 4, 5, 6]
    start = len(ids) - 1

    def prefix():
        engine.reset()
        for i, t in enumerate(ids[:start]):
            engine.feed(t, i)

    prefix()
    tok, pos, ref = ids[-1], start, []
    for _ in range(5):
        tok = engine.step(tok, pos)
        ref.append(tok)
        pos += 1
    seq = ids + ref

    prefix()
    engine.step_batch(seq[start:start + 4], start)
    engine.batch_rollback(2)
    tail = []
    for i in range(2, 4):
        tail.append(engine.step(seq[start + i], start + i))
    assert tail == ref[2:4]


def test_state_block_round_trips(engine):
    """save_state/load_state is what an incremental drafter rewinds with; it must be exact."""
    engine.reset()
    for i, t in enumerate([1, 2, 3, 4]):
        engine.feed(t, i)
    snap = engine.save_state()
    assert snap.size == engine.state_len()
    after = [engine.step(5, 4), engine.step(6, 5)]
    moved = engine.save_state()
    assert np.any(moved != snap)
    engine.load_state(snap)
    np.testing.assert_array_equal(engine.save_state(), snap)
    engine.load_state(snap)
    assert [engine.step(5, 4), engine.step(6, 5)] == after


def test_speculative_decode_returns_generate(engine):
    """Drafts are only ever confirmed by the model's own argmax, so the output is plain decode.

    The oracle proposes exactly the tokens plain decode emits, which pins the round's alignment: a
    drafter fed the prefix *without* the pending row is one position out and accepts nothing.
    """
    from kindi import mtp_cpu

    ids = [1, 2, 3, 4, 5]
    n = 6
    plain = engine.generate(ids, n)

    class _Oracle(object):
        def reset(self):
            pass

        def draft(self, prefix, hin, k=None):
            i = len(prefix) - len(ids)
            return plain[i:i + int(k or 2)]

    tokens, stats = mtp_cpu.speculative_decode(engine, ids, n, _Oracle(), k=2)
    assert tokens == plain
    assert stats["rate"] == 1.0, stats
    assert stats["rounds"] >= 1 and stats["rows"] == 3 * stats["rounds"]
