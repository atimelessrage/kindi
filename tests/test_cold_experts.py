# SPDX-License-Identifier: AGPL-3.0-only
"""The cold-expert file reader: exact bytes, read-ahead, and the accounting behind the gate."""
import numpy as np
import pytest
import types

from kindi import cold_experts, hostram, lightning


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


def test_the_host_budget_defaults_to_free_ram_with_headroom_and_can_be_set(monkeypatch):
    assert cold_experts.host_budget_bytes({"KINDI_LT_HOST_MB": "128"}, available=1 << 40) == 128 << 20
    assert cold_experts.host_budget_bytes({"KINDI_LT_HOST_MB": "0"}, available=1 << 40) == 0
    assert cold_experts.host_budget_bytes({}, available=10 << 30) == 9 << 30
    monkeypatch.setattr(hostram, "available", lambda: None)
    assert cold_experts.host_budget_bytes({}, available=None) == 0
    with pytest.raises(ValueError, match="negative"):
        cold_experts.host_budget_bytes({"KINDI_LT_HOST_MB": "-1"})
    with pytest.raises(ValueError, match="integer"):
        cold_experts.host_budget_bytes({"KINDI_LT_HOST_MB": "lots"})


def test_the_hottest_host_experts_stay_pinned_and_the_overflow_goes_cold():
    """The global measured shares decide the split, not the per-layer count."""
    host = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    order = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    counts = {0: [10, 5, 1, 0], 1: [8, 4, 2, 0]}
    warm, cold, stats = cold_experts.split_host_tiers(host, order, 350, 100, counts)
    assert warm == {0: (0, 1), 1: (0,)}
    assert cold == {0: (2, 3), 1: (1, 2, 3)}
    assert stats == {"budget_bytes": 350, "host_bytes": 300, "disk_bytes": 500,
                     "host_experts": 3, "disk_experts": 5}
    # The pinned set is exact, under budget, and the tier bytes add up to the whole host tier.
    assert stats["host_bytes"] <= stats["budget_bytes"]
    assert stats["host_bytes"] + stats["disk_bytes"] == 8 * 100


def test_a_zero_host_budget_makes_every_host_expert_file_backed():
    host = {0: (0, 1, 2), 1: (3, 4, 5)}
    order = {0: [0, 1, 2], 1: [3, 4, 5]}
    warm, cold, stats = cold_experts.split_host_tiers(host, order, 0, 100)
    assert warm == {0: (), 1: ()}
    assert cold == {0: (0, 1, 2), 1: (3, 4, 5)}
    assert stats["host_bytes"] == 0 and stats["disk_bytes"] == 600


def test_host_budget_without_counts_keeps_the_usage_order_hot():
    host = {0: (3, 4), 1: (5, 6)}
    order = {0: [4, 3], 1: [6, 5]}
    warm, cold, _stats = cold_experts.split_host_tiers(host, order, 100, 100)
    assert warm == {0: (4,), 1: ()}       # rank 0 of the first layer wins the tie
    assert cold == {0: (3,), 1: (5, 6)}


def test_the_engine_reports_pinned_and_disk_bytes_without_a_gpu():
    """`cold_stats` counts each expert's codes and scales on both sides of the split."""
    eng = lightning.Engine.__new__(lightning.Engine)
    eng.cold_reader = type("Reader", (), {"stats": lambda self: {"method": "test"}})()
    eng.cold_active = True
    eng.cold_experts = 4
    eng.cold_read_experts = 2
    eng.host_budget = 250
    eng.layers = [{"up": type("W", (), {"cw": 100, "sw": 20, "hwarm": (0, 1),
                                        "hcold": (2, 3)})(),
                   "down": type("W", (), {"cw": 100, "sw": 20, "hwarm": (0,),
                                           "hcold": (1, 2)})()}]
    stats = eng.cold_stats()
    assert stats["host_experts"] == 3
    assert stats["cold_experts"] == 4
    assert stats["host_bytes"] == 3 * 120
    assert stats["cold_bytes"] == 4 * 120
    assert stats["host_budget_bytes"] == 250


def test_the_engine_splits_each_layer_host_tier_before_loading():
    eng = lightning.Engine.__new__(lightning.Engine)
    eng.s = types.SimpleNamespace(kind=["moe"], n_exp=4)
    eng.n_card = 1
    eng.placement = None
    eng.cold_reader = object()
    eng.host_budget = 200
    eng._expert_bytes = lambda: 100
    eng._usage = lambda: {0: [0, 1, 2, 3]}
    eng._usage_counts = lambda: {0: [10, 5, 1, 0]}
    eng._split_host_tiers()
    assert eng.host_warm == {0: (1, 2)}       # expert 0 is on the card, so it is not host
    assert eng.host_cold == {0: (3,)}
    assert eng.host_tier_stats["host_bytes"] == 200
    assert eng.host_tier_stats["disk_bytes"] == 100
    assert eng.cold_experts == 1 and eng.cold_active


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
