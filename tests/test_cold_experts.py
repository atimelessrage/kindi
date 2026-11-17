# SPDX-License-Identifier: AGPL-3.0-only
"""The cold-expert file reader: exact bytes, read-ahead, and the accounting behind the gate."""
import numpy as np
import pytest

from kindi import cold_experts


def _file(tmp_path, size=5 * 4096):
    data = bytes((i * 37 + 11) % 251 for i in range(size))
    path = tmp_path / "pack.bin"
    path.write_bytes(data)
    return path, data


def test_the_flag_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("KINDI_LT_COLD", raising=False)
    assert not cold_experts.cold_enabled()
    monkeypatch.setenv("KINDI_LT_COLD", "0")
    assert not cold_experts.cold_enabled()
    monkeypatch.setenv("KINDI_LT_COLD", "1")
    assert cold_experts.cold_enabled()
    assert cold_experts.cold_enabled({"KINDI_LT_COLD": "yes"})
    assert not cold_experts.cold_enabled({"KINDI_LT_COLD": "off"})


def test_read_many_returns_exact_bytes_and_accounts_for_them(tmp_path):
    path, data = _file(tmp_path)
    spans = [(123, 17), (4100, 33), (9000, 11)]
    reader = cold_experts.ColdExpertReader(path, read_ahead=0)
    try:
        got = reader.read_many(spans)
        assert got == [data[o:o + n] for o, n in spans]
        stats = reader.stats()
        assert stats["requested_bytes"] == sum(n for _, n in spans)
        assert stats["read_bytes"] >= stats["requested_bytes"]
        assert stats["readahead_bytes"] + stats["alignment_bytes"] == \
            stats["read_bytes"] - stats["requested_bytes"]
        assert stats["unbuffered"] in (True, False)
        assert stats["method"] in ("O_DIRECT", "FILE_FLAG_NO_BUFFERING") or \
            stats["method"].startswith("buffered fallback")
        assert stats["reads"] >= 1
        assert stats["alignment"] == cold_experts.ALIGN
        # A single exact read is charged once, not once per requested byte or span.
        before = reader.stats()["requested_bytes"]
        assert reader.read(2048, 64) == data[2048:2112]
        assert reader.stats()["requested_bytes"] == before + 64
    finally:
        reader.close()


def test_read_ahead_merges_neighbouring_ranges_into_one_read(tmp_path):
    path, data = _file(tmp_path)
    spans = [(0, 16), (4096, 16)]
    reader = cold_experts.ColdExpertReader(path, read_ahead=1)
    try:
        got = reader.read_many(spans)
        assert got == [data[0:16], data[4096:4112]]
        stats = reader.stats()
        assert stats["reads"] == 1
        assert stats["requested_bytes"] == 32
        assert stats["read_bytes"] == 2 * 4096      # first block + the read-ahead block
        assert stats["readahead_bytes"] == 2 * 4096 - 32 - stats["alignment_bytes"]
    finally:
        reader.close()


def test_a_gap_larger_than_the_ahead_window_is_not_read(tmp_path):
    path, data = _file(tmp_path)
    spans = [(0, 16), (2 * 4096, 16)]
    reader = cold_experts.ColdExpertReader(path, read_ahead=0)
    try:
        assert reader.read_many(spans) == [data[0:16], data[8192:8208]]
        assert reader.stats()["reads"] == 2
        assert reader.stats()["read_bytes"] == 2 * 4096      # no gap bytes between the ranges
    finally:
        reader.close()


def test_spans_are_returned_in_input_order_after_offset_sorting(tmp_path):
    path, data = _file(tmp_path)
    spans = [(8192, 5), (0, 7), (4096, 9)]
    reader = cold_experts.ColdExpertReader(path, read_ahead=0)
    try:
        assert reader.read_many(spans) == [data[o:o + n] for o, n in spans]
    finally:
        reader.close()


def test_warm_slots_are_not_reread_and_duplicates_read_once():
    cold = {1, 2, 3}
    # 2 is already owned by a warm device slot; 9 is not a cold expert at all.
    assert cold_experts.cold_misses([1, 2, 9, 1, 3], [2, -1, -1], cold) == [1, 3]
    assert cold_experts.cold_misses([2], [2, -1], cold) == []
    assert cold_experts.cold_misses([1], [], cold) == [1]


def test_q2_split_keeps_codes_and_scales_separate():
    rows, blocks = 2, 3
    raw = np.arange(rows * blocks * 18, dtype=np.uint8)
    codes, scales = cold_experts.split_q2(raw.tobytes(), rows, blocks)
    blk = raw.reshape(rows * blocks, 18)
    assert codes.tolist() == blk[:, 2:].reshape(-1).tolist()
    assert scales.tolist() == blk[:, :2].reshape(-1).tolist()
    assert len(codes) == rows * blocks * 16
    assert len(scales) == rows * blocks * 2
    with pytest.raises(ValueError):
        cold_experts.split_q2(b"short", rows, blocks)


def test_short_reads_are_an_error_not_a_silent_truncation(tmp_path):
    path, _ = _file(tmp_path, size=8192)
    reader = cold_experts.ColdExpertReader(path)
    try:
        with pytest.raises(IOError):
            reader.read(8190, 16)
    finally:
        reader.close()
