# SPDX-License-Identifier: AGPL-3.0-only
"""The cold-expert file reader: exact bytes, coalescing, prefetch and the accounting behind the gate."""
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
                     "host_experts": 3, "disk_experts": 5,
                     "counted_layers": 2, "counts_used": True}
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


def test_the_split_accounts_for_different_expert_sizes_per_layer():
    """Byte accounting is per expert, not per layer: two layers of different size fill the budget
    exactly and the two tiers still add up to every host expert."""
    host = {0: (0, 1, 2), 1: (0, 1)}
    order = {0: [0, 1, 2], 1: [0, 1]}
    counts = {0: [9, 5, 1], 1: [8, 2]}
    warm, cold, stats = cold_experts.split_host_tiers(host, order, 250, {0: 100, 1: 150}, counts)
    assert warm == {0: (0,), 1: (0,)}          # the hottest expert of each layer, 100 + 150
    assert cold == {0: (1, 2), 1: (1,)}
    assert stats == {"budget_bytes": 250, "host_bytes": 250, "disk_bytes": 350,
                     "host_experts": 2, "disk_experts": 3,
                     "counted_layers": 2, "counts_used": True}
    assert stats["host_bytes"] + stats["disk_bytes"] == 3 * 100 + 2 * 150 == 600


def test_host_budget_without_counts_keeps_the_usage_order_hot():
    host = {0: (3, 4), 1: (5, 6)}
    order = {0: [4, 3], 1: [6, 5]}
    warm, cold, stats = cold_experts.split_host_tiers(host, order, 100, 100)
    assert warm == {0: (4,), 1: ()}       # rank 0 of the first layer wins the tie
    assert cold == {0: (3,), 1: (5, 6)}
    assert stats["counts_used"] is False and stats["counted_layers"] == 0


def test_a_partial_or_malformed_count_map_is_reported_and_falls_back_to_order():
    host = {0: (3, 4), 1: (5, 6)}
    order = {0: [4, 3], 1: [6, 5]}
    for counts in ({0: [9, 1]},                       # layer 1 missing
                   {0: [9, 1], 1: [1]},               # layer 1 too short for its ids
                   {0: [9, 1], 1: [1, "bad"]}):       # non-numeric row
        warm, cold, stats = cold_experts.split_host_tiers(host, order, 100, 100, counts)
        assert stats["counts_used"] is False and stats["counted_layers"] < 2, counts
        assert warm == {0: (4,), 1: ()} and cold == {0: (3,), 1: (5, 6)}


def test_the_engine_reports_pinned_and_disk_bytes_without_a_gpu():
    """`cold_stats` counts a pinned host expert once, while the byte totals stay per matrix."""
    eng = lightning.Engine.__new__(lightning.Engine)
    eng.cold_reader = type("Reader", (), {"stats": lambda self: {"method": "test"}})()
    eng.cold_active = True
    eng.cold_experts = 4
    eng.cold_read_experts = 2
    eng.cold_slot_hits = 5
    eng.host_budget = 250
    eng.layers = [{"up": type("W", (), {"cw": 100, "sw": 20, "hwarm": (0, 1),
                                        "hcold": (2, 3)})(),
                   "down": type("W", (), {"cw": 100, "sw": 20, "hwarm": (0,),
                                           "hcold": (1, 2)})()}]
    stats = eng.cold_stats()
    assert stats["host_experts"] == 2          # {0, 1}: unique pinned experts, not up 2 + down 1
    assert stats["cold_experts"] == 4
    assert stats["slot_hits"] == 5              # a routed expert served from its unchanged slot
    assert stats["host_bytes"] == 3 * 120      # up 2 + down 1, each matrix counted once
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
    reader = cold_experts.ColdExpertReader(path, coalesce=0)
    try:
        got = reader.read_many(spans)
        assert got == [data[o:o + n] for o, n in spans]
        stats = reader.stats()
        assert stats["requested_bytes"] == sum(n for _, n in spans)
        assert stats["read_bytes"] >= stats["requested_bytes"]
        assert stats["coalesced_bytes"] + stats["alignment_bytes"] == \
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


def test_coalescing_merges_neighbouring_ranges_into_one_read(tmp_path):
    path, data = _file(tmp_path)
    spans = [(0, 16), (4096, 16)]
    reader = cold_experts.ColdExpertReader(path, coalesce=1)
    try:
        got = reader.read_many(spans)
        assert got == [data[0:16], data[4096:4112]]
        stats = reader.stats()
        assert stats["reads"] == 1
        assert stats["requested_bytes"] == 32
        assert stats["read_bytes"] == 2 * 4096      # first block + the coalesced block
        assert stats["coalesced_bytes"] == 2 * 4096 - 32 - stats["alignment_bytes"]
    finally:
        reader.close()


def test_prefetch_queue_serves_later_demand_without_another_read(tmp_path):
    """A queued range was fetched before demand; the later read consumes those bytes, not the file."""
    path, data = _file(tmp_path)
    spans = [(4096, 16), (8192, 16)]
    reader = cold_experts.ColdExpertReader(path, coalesce=0, prefetch_bytes=4096)
    try:
        assert reader.prefetch(spans) == 2
        before = reader.stats()
        assert before["prefetched_bytes"] == 32
        assert before["prefetch_cached_bytes"] == 32
        assert before["reads"] == 2
        assert reader.read_many(spans) == [data[o:o + n] for o, n in spans]
        after = reader.stats()
        assert after["reads"] == before["reads"]       # no second file read for the demand
        assert after["prefetch_hits"] == 2
        assert after["prefetch_used_bytes"] == 32
        assert after["prefetch_cached_bytes"] == 0
        assert after["read_bytes"] == before["read_bytes"]  # the bytes were already in the queue
        assert after["requested_bytes"] == 32
        assert after["read_bytes"] == (after["requested_bytes"] + after["coalesced_bytes"]
                                       + after["alignment_bytes"] + after["prefetched_bytes"]
                                       - after["prefetch_used_bytes"])
    finally:
        reader.close()


def test_prefetch_queue_evicts_oldest_and_duplicate_prefetches_do_not_reread(tmp_path):
    """A duplicate prefetch is free; overflowing the byte limit drops the oldest span."""
    path, data = _file(tmp_path)
    spans = [(0, 16), (4096, 16), (8192, 16)]
    reader = cold_experts.ColdExpertReader(path, coalesce=0, prefetch_bytes=32)
    try:
        assert reader.prefetch([spans[0]]) == 1
        assert reader.prefetch([spans[1]]) == 1
        before = reader.stats()
        assert before["prefetch_cached_bytes"] == 32
        assert before["reads"] == 2

        # Re-requesting both queued spans, including a duplicate in one call, reads nothing.
        assert reader.prefetch([spans[0], spans[1], spans[0]]) == 0
        after = reader.stats()
        assert after["reads"] == before["reads"]
        assert after["prefetch_requests"] == before["prefetch_requests"]
        assert after["prefetched_bytes"] == before["prefetched_bytes"]
        assert after["prefetch_cached_bytes"] == 32

        # A third span fits only by evicting the oldest one (spans[0]).
        assert reader.prefetch([spans[2]]) == 1
        assert reader.stats()["prefetch_cached_bytes"] == 32
        assert reader.stats()["reads"] == 3

        # The evicted span goes back to the file; the two newer spans do not.
        reads = reader.stats()["reads"]
        assert reader.read_many([spans[0]]) == [data[0:16]]
        assert reader.stats()["reads"] == reads + 1
        reads = reader.stats()["reads"]
        assert reader.read_many([spans[1], spans[2]]) == [data[4096:4112], data[8192:8208]]
        final = reader.stats()
        assert final["reads"] == reads
        assert final["prefetch_hits"] == 2
        assert final["prefetch_cached_bytes"] == 0
    finally:
        reader.close()


def test_a_gap_larger_than_the_coalesce_window_is_not_read(tmp_path):
    path, data = _file(tmp_path)
    spans = [(0, 16), (2 * 4096, 16)]
    reader = cold_experts.ColdExpertReader(path, coalesce=0)
    try:
        assert reader.read_many(spans) == [data[0:16], data[8192:8208]]
        assert reader.stats()["reads"] == 2
        assert reader.stats()["read_bytes"] == 2 * 4096      # no gap bytes between the ranges
    finally:
        reader.close()


def test_spans_are_returned_in_input_order_after_offset_sorting(tmp_path):
    path, data = _file(tmp_path)
    spans = [(8192, 5), (0, 7), (4096, 9)]
    reader = cold_experts.ColdExpertReader(path, coalesce=0)
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


def test_a_cold_slot_plan_keeps_hits_and_only_evicts_unrouted_owners():
    plan, owners, ages, tick = cold_experts.cold_slot_plan([1, 2, 3], {1, 2, 3}, [1, -1, 3])
    assert plan == [(1, 0, False), (2, 1, True), (3, 2, False)]
    assert owners == [1, 2, 3]
    assert ages == [1, 1, 1] and tick == 1


def test_a_cold_slot_plan_never_steals_a_routed_experts_slot():
    # 1 is routed and stays in slot 0; 4 takes the slot of 2, which is not routed this step.
    plan, owners, _ages, _tick = cold_experts.cold_slot_plan([1, 4], {1, 4}, [1, 2, 3])
    assert plan == [(1, 0, False), (4, 1, True)]
    assert owners == [1, 4, 3]


def test_a_cold_slot_plan_evicts_the_least_recently_used_unrouted_slot():
    plan, owners, ages, tick = cold_experts.cold_slot_plan([8], {5, 6, 7, 8}, [5, 6, 7], [1, 3, 2], tick=9)
    assert plan == [(8, 0, True)]          # slot 0 has age 1, the oldest of the three
    assert owners == [8, 6, 7] and ages == [10, 3, 2] and tick == 10


def test_a_cold_slot_plan_dedupes_and_ignores_non_cold_experts():
    plan, owners, _ages, _tick = cold_experts.cold_slot_plan([1, 2, 1, 9], {1, 2}, [1, -1])
    assert plan == [(1, 0, False), (2, 1, True)]
    assert owners == [1, 2]


def test_a_cold_slot_plan_refuses_a_table_too_small_for_the_route():
    with pytest.raises(ValueError, match="no slot"):
        cold_experts.cold_slot_plan([1, 2], {1, 2}, [])
    with pytest.raises(ValueError, match="ages"):
        cold_experts.cold_slot_plan([1], {1}, [1], [])


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


class _Pinned(bytearray):
    """A pinned host allocation the engine reads back as a numpy buffer."""

    def __init__(self, size, ptr):
        bytearray.__init__(self, size)
        self.ptr = ptr


class _Stream(object):
    ptr = 0x9000

    def wait_event(self, ev):
        return None

    def synchronize(self):
        return None


class _Event(object):
    def record(self, stream):
        return None


class _SpanReader(object):
    """Serves a byte blob and records the spans asked for, so a warm read would show up."""

    def __init__(self, blob):
        self.blob = blob
        self.spans = []

    def read_many(self, spans, **kwargs):
        self.spans.extend(spans)
        return [self.blob[o:o + n] for o, n in spans]


class _PrefetchReader(object):
    """Just records queued spans; the real queue's consumption is tested above."""

    def __init__(self):
        self.spans = []

    def prefetch(self, spans):
        self.spans.extend(spans)
        return len(spans)


def test_the_engine_queues_the_next_cold_layers_likely_experts():
    def tensor(raw_off):
        return types.SimpleNamespace(hcold=(1, 2, 3), raw_off=raw_off, raw_stride=10, raw_bytes=10)

    eng = lightning.Engine.__new__(lightning.Engine)
    eng.cold_reader = _PrefetchReader()
    eng.layers = [{"cold": True, "up": tensor(0), "down": tensor(100)},
                  {"cold": True, "up": tensor(200), "down": tensor(300)}]
    eng.s = types.SimpleNamespace(n_used=2)
    eng.cold_order = {1: [3, 1, 2]}              # the next layer's measured order
    eng._cold_prefetch(0)
    assert eng.cold_reader.spans == [(230, 10), (330, 10), (210, 10), (310, 10)]


def _stage_tensor(hoff, warm, cold, codes, scales, raw_off):
    """One Q2_0 expert stack: 16 code bytes and 2 scale bytes per expert, as `_q2` lays it out."""
    mem = _Pinned(len(warm) * 18, 0x1000 + raw_off)
    for j, e in enumerate(warm):
        mem[j * 16:(j + 1) * 16] = codes[e]
        mem[len(warm) * 16 + j * 2:len(warm) * 16 + (j + 1) * 2] = scales[e]
    return types.SimpleNamespace(
        cw=16, sw=2, rows=1, cols=64, hoff=tuple(hoff), hwarm=tuple(warm), hcold=tuple(cold),
        hpos={e: j for j, e in enumerate(warm)}, hbytes=len(hoff) * 18, hmem=mem,
        raw_off=raw_off, raw_stride=18, raw_bytes=18)


def test_a_staged_host_block_puts_each_pinned_expert_at_its_host_slot():
    """Warm and cold experts interleave in expert-id order and the prompt staging block is laid out
    in `hoff` order, so a pinned expert's bytes must go to its hoff slot, not its warm-list slot."""
    hoff, warm, cold = (0, 1, 2, 3), (0, 2), (1, 3)
    codes = {e: bytes([0xA0 + e]) * 16 for e in hoff}
    scales = {e: bytes([0x10 + e, 0x00]) for e in hoff}
    up = _stage_tensor(hoff, warm, cold, codes, scales, 1000)
    down = _stage_tensor(hoff, warm, cold, codes, scales, 2000)
    blob = bytearray(4000)
    for base, w in ((1000, up), (2000, down)):
        for e in cold:                              # GGUF block: 2 scale bytes, then 16 code bytes
            blob[base + e * 18:base + (e + 1) * 18] = scales[e] + codes[e]
    total = up.hbytes + down.hbytes
    stage, host = _Pinned(total, 0x3000), _Pinned(total, 0x2000)
    buffers = {stage.ptr: stage, host.ptr: host}

    def memcpy_async(dst, src, n, kind, stream):
        buffers[dst][:n] = buffers[src][:n]

    eng = lightning.Engine.__new__(lightning.Engine)
    eng.layers = [{"up": up, "down": down, "stage": 0}]
    eng.stage_host = [host]
    eng.stage = [types.SimpleNamespace(data=types.SimpleNamespace(ptr=stage.ptr))]
    eng.stage_n = total
    eng.copy_stream, eng.ev_used, eng.ev_staged = _Stream(), [_Event()], [_Event()]
    eng.cold_reader = _SpanReader(blob)
    eng.cp = types.SimpleNamespace(
        cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(memcpyAsync=memcpy_async)))
    eng._stage_cold(0)

    expect = bytearray(total)
    for base, w in ((0, up), (up.hbytes, down)):
        for i, e in enumerate(hoff):
            expect[base + i * w.cw:base + (i + 1) * w.cw] = codes[e]
            expect[base + len(hoff) * w.cw + i * w.sw:
                   base + len(hoff) * w.cw + (i + 1) * w.sw] = scales[e]
    assert bytes(stage) == bytes(expect)
    # Only the overflow is file-backed: two cold experts on each tensor, and no warm id is read.
    assert eng.cold_reader.spans == [(1018, 18), (1054, 18), (2018, 18), (2054, 18)]
    assert sum(n for _, n in eng.cold_reader.spans) == len(cold) * 2 * 18
