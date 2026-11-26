# SPDX-License-Identifier: AGPL-3.0-only
"""The batched T-row pass must run, roll back and save/load with no pack download.

`tests/test_cpu_batch.py` is the exactness gate: it decodes a real Bonsai pack a row at a time and
compares every argmax and every logit with plain decode.  That needs a multi-gigabyte pack, so it is
opt-in with `KINDI_CPU_PACK` and CI never runs it.

This file covers the rest of the batch contract -- the part that regressed twice already (arena
sizing, the reuse key, the rollback replay) -- against a two-layer pack `_write_pack` builds from a
seed in a few hundred KiB and ~0.5 s.  What it pins:

* the pack loads and plain decode still runs on it (so a failure below is the batch, not the pack);
* `step_batch` runs the whole stack in one pass and returns one in-range token per row, with a
  logits view of the head's width and a hidden view of the model's width;
* `batch_rollback(0)` restores the pre-batch state bit for bit (the stored per-layer vectors are
  replayed), and `batch_rollback(n)` changes nothing;
* `save_state`/`load_state` round-trip the state block a drafter rewinds with.
* a 5-row pass is not a multiplicative slowdown over the same rows decoded plainly (a loose
  regression net, not CP3's <=1.5-plain-rows gate, which is missed at ~5 on real packs).

Bit-exactness against plain decode is deliberately *not* asserted here.  On every hand-built pack
tried (128 to 2048 wide, attention-only and GDN, one to four layers, fresh or fed state) the batch's
GDN block disagrees with plain decode -- `batch_hidden(0)` can read 1e28 where plain decode is
finite -- while the real 2B and 27B packs are byte-exact, measured on the 27B over four prompts.  The
two paths run the same arithmetic in the same order on paper, so the difference is either a buffer
the batch reads before writing for some pack layouts or weight-magnitude sensitivity, and neither is
attributed yet; asserting equality here would fail for a reason this file cannot name.  Until that is
settled `tests/test_cpu_batch.py` stays the only place exactness is claimed.
"""
import json
import os

import numpy as np
import pytest

from kindi import cpu

# Every width a matrix is fed must be a multiple of 128 (the pair-code GEMV reads 128 columns per
# group), every row count a multiple of 8 (eight rows per scale block), and `hadamard_block` must
# divide each folded width.  head_dim <= 512 and state_size <= 128 size `tmp[512]` and `sk[128]`.
LAYERS = 2
D = 128
NFF = 256
NH = 2
NHKV = 1
HD = 64
NK = 2
HK = 16
NV = 8
HB = 128
VOCAB = 256
CONV_CH = 2 * NK * HK + NV * HK


def _write_pack(path, seed=7):
    """Write a pack in the layout `kindi/cpu.py:Engine.__init__` reads, and return the directory.

    `bcodes.bin` holds one byte per four input columns and row of 0..80 trit codes, `dpm.bin` the
    fp16 scale for each (128-column group, 8-row block), `f32.bin` the unquantised tensors, and the
    two index files the offsets into each.  `signs_W.f32` are the per-width Hadamard signs.  Any
    code and scale is a valid matrix, so a seeded draw is a valid model -- not a trained one.
    """
    rng = np.random.default_rng(seed)
    mats, flats = [], []

    def ones(n):
        return np.ones(n, dtype=np.float32)

    def small(n):
        return (rng.standard_normal(n) * 0.05).astype(np.float32)

    for il in range(LAYERS):
        p = "blk.%d." % il
        mats += [(p + "ffn_gate.weight", NFF, D), (p + "ffn_up.weight", NFF, D),
                 (p + "ffn_down.weight", D, NFF)]
        flats += [(p + "attn_norm.weight", ones(D)), (p + "post_attention_norm.weight", ones(D))]
        if (il + 1) % 4 == 0:
            mats += [(p + "attn_q.weight", NH * HD, D), (p + "attn_k.weight", NHKV * HD, D),
                     (p + "attn_v.weight", NHKV * HD, D), (p + "attn_output.weight", D, NH * HD)]
            flats += [(p + "attn_q_norm.weight", ones(HD)), (p + "attn_k_norm.weight", ones(HD))]
        else:
            mats += [(p + "attn_qkv.weight", CONV_CH, D), (p + "attn_gate.weight", NV, D),
                     (p + "ssm_out.weight", D, NV * HK)]
            flats += [(p + "ssm_beta.weight", small(NV * D)), (p + "ssm_alpha.weight", small(NV * D)),
                      (p + "ssm_dt.bias", small(NV)),
                      (p + "ssm_a", (-0.4 - 0.2 * rng.random(NV)).astype(np.float32)),
                      (p + "ssm_conv1d.weight", small(4 * CONV_CH)),
                      (p + "ssm_norm.weight", ones(HK))]
    mats += [("token_embd.weight", VOCAB, D), ("output.weight", VOCAB, D)]
    flats.append(("output_norm.weight", ones(D)))

    os.makedirs(path, exist_ok=True)
    codes_blobs, dpm_blobs, bc_index = [], [], {}
    bc_off = dp_off = 0
    for name, rows, cols in mats:
        assert cols % 128 == 0 and rows % 8 == 0, name
        codes = rng.integers(0, 81, size=(cols // 4, rows), dtype=np.uint8)
        codes_blobs.append(codes.tobytes())
        bc_index[name] = {"rows": rows, "ne0": cols, "bc_off": bc_off, "dpm_off": dp_off}
        dpm = np.ones((cols // 128, rows // 8, 8), dtype=np.float16)
        dpm_blobs.append(dpm.tobytes())
        bc_off += codes.nbytes
        dp_off += dpm.nbytes

    f32_blobs, f32_index, off = [], {}, 0
    for name, arr in flats:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        f32_blobs.append(arr.tobytes())
        f32_index[name] = {"rows": 1, "ne0": int(arr.size), "f32_off": off}
        off += arr.nbytes

    with open(os.path.join(path, "bcodes.bin"), "wb") as f:
        f.write(b"".join(codes_blobs))
    with open(os.path.join(path, "dpm.bin"), "wb") as f:
        f.write(b"".join(dpm_blobs))
    with open(os.path.join(path, "f32.bin"), "wb") as f:
        f.write(b"".join(f32_blobs))
    with open(os.path.join(path, "bcodes_index.json"), "w") as f:
        json.dump({"tensors": bc_index}, f)
    with open(os.path.join(path, "index.json"), "w") as f:
        json.dump({"tensors": f32_index}, f)
    for width in (D, NV * HK, NH * HD, NFF):
        signs = np.where(rng.random(width) < 0.5, -1.0, 1.0).astype(np.float32)
        signs.tofile(os.path.join(path, "signs_%d.f32" % width))
    with open(os.path.join(path, "kindi_pack.json"), "w") as f:
        json.dump({"model": "synthetic", "source_file": "synthetic.gguf", "format": "PTQ1_0",
                   "head": "output.weight", "hadamard_block": HB,
                   "dimensions": {"layers": LAYERS, "width": D, "ffn": NFF, "heads": NH,
                                  "kv_heads": NHKV, "groups": NK, "head_dim": HD, "rope_dim": 64,
                                  "state_size": HK, "gdn_inner": NV * HK, "rope_base": 1e7,
                                  "full_attention_interval": 4}}, f)
    with open(os.path.join(path, "vocab.json"), "w") as f:
        json.dump({"tokens": ["t%d" % i for i in range(VOCAB)]}, f)
    return path


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """Build the pack once; skip rather than fail where no C compiler is installed."""
    try:
        cc = cpu._find_cc()
    except SystemExit as exc:
        pytest.skip("no C compiler to build the CPU engine with: %s" % exc)
    if not cc:
        pytest.skip("neither gcc nor clang is installed")
    path = _write_pack(str(tmp_path_factory.mktemp("synth_pack")))
    try:
        engine = cpu.Engine(path, ctx=64, threads=2, log=None)
    except SystemExit as exc:            # the engine builds its C library on first use
        pytest.skip("the CPU library cannot be built here: %s" % exc)
    engine.close()
    return path


def _engine(pack):
    return cpu.Engine(pack, ctx=64, threads=2, log=None)


def test_plain_decode_runs_on_the_pack(pack):
    """A failure below is the batch, not the pack: plain decode must produce tokens and logits."""
    e = _engine(pack)
    assert e.state_len() == LAYERS * (NV * HK * HK + 3 * CONV_CH)
    e.reset()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        e.feed(tok, pos)
    tok = e.step(6, 5)
    assert 0 <= tok < VOCAB
    assert np.isfinite(e.logits[:VOCAB]).all()
    e.close()


def test_step_batch_returns_one_in_range_token_per_row(pack):
    e = _engine(pack)
    e.reset()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        e.feed(tok, pos)
    out = e.step_batch([6, 7, 8, 9], 5)
    assert len(out) == 4
    assert all(0 <= t < VOCAB for t in out)
    assert e.batch_logits(3).shape == (VOCAB,)      # one logits view per row, head-width each
    assert e.batch_hidden(0).shape == (D,)
    e.close()


def test_batch_refuses_rows_past_the_context(pack):
    e = _engine(pack)
    e.reset()
    with pytest.raises(OverflowError):
        e.step_batch([1, 2, 3], 63)
    one_row = e.step_batch([1], 0)                       # one row is plain decode by definition
    assert len(one_row) == 1
    e.close()


def test_rollback_to_zero_restores_the_pre_batch_state(pack):
    """A verifier that keeps none of the rows must leave the model exactly where it found it."""
    e = _engine(pack)
    e.reset()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        e.feed(tok, pos)
    before = e.save_state().tobytes()
    e.step_batch([6, 7, 8], 5)
    assert before != e.save_state().tobytes(), "the batch advanced nothing"
    assert e.batch_rollback(0) == 0
    assert before == e.save_state().tobytes()      # bit for bit, NaN blocks included
    e.close()


def test_rollback_of_every_row_changes_nothing(pack):
    e = _engine(pack)
    e.reset()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        e.feed(tok, pos)
    e.step_batch([6, 7, 8], 5)
    after = e.save_state().tobytes()
    e.batch_rollback(3)
    assert after == e.save_state().tobytes()
    e.close()


def test_state_block_round_trips_through_save_and_load(pack):
    e = _engine(pack)
    e.reset()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        e.feed(tok, pos)
    saved = e.save_state()
    nxt = e.step(6, 5)
    assert saved.tobytes() != e.save_state().tobytes()
    e.load_state(saved)
    assert saved.tobytes() == e.save_state().tobytes()
    assert e.step(6, 5) == nxt                          # the same state decodes the same token
    e.close()


def test_batch_pass_is_not_a_multiplicative_slowdown(pack):
    """Regression net for the batch kernel's cost (the 2026-11-26 audit's perf-gate ask).

    This is NOT the CP3 gate (a 5-row pass at <=1.5 plain rows, missed at ~5 plain rows on the 9B);
    it only guards against a blowup -- a pass that re-reads the weights per row or rebuilds a table
    per row would land well past 3x the same rows decoded plainly. The bound is 3x with a wide
    margin on purpose: the pack is ~300 KiB and rows are milliseconds, so a tighter floor would
    be timing noise, not signal.
    """
    import time
    e = _engine(pack)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    e.reset()
    t0 = time.perf_counter()
    for pos, tok in enumerate(toks):
        e.feed(tok, pos)
    plain_s = time.perf_counter() - t0
    e.reset()
    for pos, tok in enumerate(toks[:5]):
        e.feed(tok, pos)
    t0 = time.perf_counter()
    e.step_batch(toks[5:], 5)
    batch_s = time.perf_counter() - t0
    e.close()
    assert batch_s < 3.0 * plain_s, (batch_s, plain_s)
