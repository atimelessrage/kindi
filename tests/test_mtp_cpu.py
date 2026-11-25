"""The CPU MTP head's arithmetic and its draft scoring, without a 6.8 GB pack.

The head's real weights and acceptance numbers live in an untracked `bench-logs/` run made with
`tools/mtp_cpu_bench.py` (no evidence page, no speed claim); these tests pin the pieces that can go
silently wrong: the Q4_0 layout and the C GEMV built on it, the
folded-input transform, the Hadamard block transform, and the scoring loop's bookkeeping.
"""
import numpy as np

from kindi import cpu, mtp_cpu


def _q4_bytes(rows, cols, seed=3):
    rng = np.random.RandomState(seed)
    nb = cols // 32
    d = rng.uniform(0.005, 0.02, size=(rows, nb)).astype(np.float16)
    qs = rng.randint(0, 256, size=(rows, nb, 16)).astype(np.uint8)
    raw = np.empty((rows, nb, 18), dtype=np.uint8)
    raw[:, :, 0:2] = np.frombuffer(d.tobytes(), dtype=np.uint8).reshape(rows, nb, 2)
    raw[:, :, 2:] = qs
    return raw.reshape(rows * nb, 18)


def test_trits_decodes_every_column_of_a_ptq1_block():
    """A PTQ1_0 block is 128 trits in three byte groups (16 + 8 + 2); the head's embedding needs all
    of them.  The local copy this pins used to fill only the first 80 and leave 48 uninitialised."""
    from kindi.convert import trits_from_raw
    raw = np.random.RandomState(7).randint(0, 256, size=(6, 28)).astype(np.uint8)
    got = mtp_cpu.trits(raw)
    assert got.shape == (6, 128)
    assert set(np.unique(got).tolist()) <= {0, 1, 2}
    assert np.array_equal(got, trits_from_raw(raw))
    assert np.any(got[:, 80:] != got[:, :48])          # the tail is decoded, not a copy of the head


def test_q4_dequant_layout():
    """element j < 16 is the low nibble of qs[j], element j + 16 the high nibble, times the fp16 scale."""
    raw = np.zeros((1, 18), dtype=np.uint8)
    raw[0, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    raw[0, 2:] = 9 | (7 << 4)                       # 9 - 8 = +1 and 7 - 8 = -1
    v = mtp_cpu.q4_dequant(raw, 1, 32)
    assert np.array_equal(v[0, :16], np.ones(16, dtype=np.float32))
    assert np.array_equal(v[0, 16:], -np.ones(16, dtype=np.float32))


def test_q4_gemv_matches_the_dequantised_matrix():
    raw = _q4_bytes(16, 64)
    W = mtp_cpu.q4_dequant(raw, 16, 64)
    x = np.random.RandomState(11).randn(64).astype(np.float32)
    assert np.allclose(cpu.q4_gemv(raw, x, 16, 64), W @ x, atol=1e-5)


def test_q4_gemv_refuses_a_wrong_shape():
    raw = _q4_bytes(4, 32)
    try:
        cpu.q4_gemv(raw, np.zeros(32, dtype=np.float32), 4, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("a mis-sized Q4_0 matrix was accepted")


def test_fwht_is_the_hadamard_matrix():
    n = 8
    H = np.array([[(-1.0) ** bin(i & j).count("1") for j in range(n)] for i in range(n)]) / np.sqrt(n)
    x = np.random.RandomState(2).randn(n).astype(np.float32)
    assert np.allclose(mtp_cpu.fwht(x, n), H @ x, atol=1e-6)


def test_fold_preserves_dot_products():
    """folding the input is only sound if the transform is orthogonal: dot(fold(w), fold(x)) == dot(w, x)."""
    rng = np.random.RandomState(5)
    x, w = rng.randn(64).astype(np.float32), rng.randn(64).astype(np.float32)
    signs = np.where(rng.rand(64) < 0.5, -1.0, 1.0).astype(np.float32)
    assert abs(float(mtp_cpu.fold(x, signs, 16) @ mtp_cpu.fold(w, signs, 16)) - float(x @ w)) < 1e-4


class _Engine(object):
    """The scoring loop only needs greedy steps and the residual row; the head supplies the drafts."""
    head_name = "output.weight"
    ylog_n = 300

    def __init__(self, seq):
        self.seq, self.i = list(seq), 0

    def reset(self):
        self.i = 0

    def feed(self, tok, pos):
        pass

    def step(self, tok, pos):
        t = self.seq[self.i]
        self.i += 1
        return t

    def get_hidden(self):
        return np.zeros(4, dtype=np.float32)


class _Drafter(object):
    """Any drafter: the loop only calls reset() and draft(prefix, hin, k)."""

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def draft(self, prefix, hin, k):
        self.calls.append((len(prefix), k))
        return [11, 12]


def test_score_drafts_keeps_the_model_tokens_and_counts_matches():
    seq = [10, 11, 12, 13, 14, 15, 16, 17]
    eng, drafter = _Engine(seq), _Drafter()
    out, stats = mtp_cpu.score_drafts(eng, [1, 2], 6, drafter, k=2)
    assert out == seq[:6]                     # exactly the plain greedy tokens
    # a draft is scored only when its position was actually decoded: with a prefix of L known tokens
    # the drafts sit at positions L and L+1, and a draft past the end of the run is not offered
    offered = want = 0
    for base, _ in drafter.calls:
        for i, cand in enumerate([11, 12]):
            at = base + i - len([1, 2])
            if 0 <= at < len(out):
                offered += 1
                want += out[at] == cand
    assert stats["offered"] == offered
    assert stats["matched"] == want
    assert len(stats["depth"]) == 2            # every depth is recorded, including the last


def test_score_drafts_hands_the_head_the_row_it_reads():
    """The head's row q reads the residual at q and the token at q + 1 (blk.64 is trained that way).

    This pins the plumbing without a pack: a stub engine tags every residual with its position.
    """
    class Stub(object):
        def __init__(self):
            self.pos = -1
            self.seen = []
            self.head_name, self.ylog_n, self.head_off, self.HB = "output.weight", 4, 0, 512

        def reset(self):
            self.pos = -1

        def feed(self, tok, pos):
            self.pos = int(pos)

        def step(self, tok, pos):
            self.pos = int(pos)
            return (int(tok) + 1) % 8

        def get_hidden(self):
            return [float(self.pos)]                 # the residual at the position just reached

        def gemv_rows(self, name, rows, x):
            import numpy as np
            return np.zeros(4, dtype="float32")

    class Drafter(object):
        def __init__(self):
            self.rows = []

        def reset(self):
            pass

        def draft(self, prefix, hin, k=None):
            self.rows.append((len(prefix), hin[0]))
            return [0] * int(k or 1)

    eng, dr = Stub(), Drafter()
    ids = [3, 5, 7]
    mtp_cpu.score_drafts(eng, ids, 6, dr, k=2)
    # every round drafts rows from len(prefix) - 2, and receives the residual at that position
    assert [(p - 2) for p, _pos in dr.rows] == [pos for _, pos in dr.rows]
    assert [p for p, _ in dr.rows] == [4, 5, 6, 7, 8]      # one row per token after the first
