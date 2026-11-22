"""CPU-only tests for the Level Zero memory-path smoke (REQ-0020 M2 partial)."""
import ctypes
import json

from kindi import intel_l0


class _FakeFn(object):
    def __init__(self, fn):
        self.fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.fn(*args)


class FakeL0(object):
    def __init__(self, group_flags=(0x2,)):
        self.group_flags = group_flags
        self.queue_ordinal = None
        self.list_ordinal = None
        self.buffer = None
        self.host_buffers = []
        self.device_addr = None
        self.freed = False
        self.destroyed = False
        self.queue_destroyed = False
        self.list_destroyed = False
        self.context_driver = None
        self.alloc_result = 0
        self.zeInit = _FakeFn(self._init)
        self.zeDriverGet = _FakeFn(self._driver_get)
        self.zeDeviceGet = _FakeFn(self._device_get)
        self.zeDeviceGetCommandQueueGroupProperties = _FakeFn(self._queue_groups)
        self.zeContextCreate = _FakeFn(self._context_create)
        self.zeContextDestroy = _FakeFn(self._context_destroy)
        self.zeMemAllocDevice = _FakeFn(self._alloc_device)
        self.zeMemAllocShared = _FakeFn(self._alloc_shared)
        self.zeMemFree = _FakeFn(self._free)
        self.zeCommandQueueCreate = _FakeFn(self._queue_create)
        self.zeCommandQueueDestroy = _FakeFn(self._queue_destroy)
        self.zeCommandQueueExecuteCommandLists = _FakeFn(self._queue_execute)
        self.zeCommandQueueSynchronize = _FakeFn(self._queue_sync)
        self.zeCommandListCreate = _FakeFn(self._list_create)
        self.zeCommandListDestroy = _FakeFn(self._list_destroy)
        self.zeCommandListClose = _FakeFn(self._list_close)
        self.zeCommandListAppendMemoryCopy = _FakeFn(self._append_copy)

    def _init(self, flags):
        return 0

    def _driver_get(self, count, handles):
        count._obj.value = 1
        if handles is not None:
            handles[0] = 0x100
        return 0

    def _device_get(self, driver, count, handles):
        count._obj.value = 1
        if handles is not None:
            handles[0] = 0x200
        return 0

    def _queue_groups(self, device, count, properties):
        count._obj.value = len(self.group_flags)
        if properties is not None:
            for index, flags in enumerate(self.group_flags):
                properties[index].stype = 0x6
                properties[index].flags = flags
                properties[index].max_fill_pattern_size = 16
                properties[index].num_queues = 1
        return 0

    def _context_create(self, driver, desc, context):
        self.context_driver = driver
        context._obj.value = 0x300
        return 0

    def _context_destroy(self, context):
        self.destroyed = True
        return 0

    def _alloc_device(self, context, desc, size, alignment, device, pointer):
        if self.alloc_result:
            return self.alloc_result
        self.buffer = (ctypes.c_ubyte * size)()
        self.device_addr = ctypes.addressof(self.buffer)
        pointer._obj.value = self.device_addr
        return 0

    def _alloc_shared(self, context, device_desc, host_desc, size, alignment, device, pointer):
        buffer = (ctypes.c_ubyte * size)()
        self.host_buffers.append(buffer)
        pointer._obj.value = ctypes.addressof(buffer)
        return 0

    def _queue_create(self, context, device, desc, queue):
        self.queue_ordinal = ctypes.cast(
            desc, ctypes.POINTER(intel_l0._CommandQueueDesc)).contents.ordinal
        queue._obj.value = 0x400
        return 0

    def _queue_destroy(self, queue):
        self.queue_destroyed = True
        return 0

    def _queue_execute(self, queue, count, lists, fence):
        return 0

    def _queue_sync(self, queue, timeout):
        return 0

    def _list_create(self, context, device, desc, command_list):
        self.list_ordinal = ctypes.cast(
            desc, ctypes.POINTER(intel_l0._CommandListDesc)).contents.command_queue_group_ordinal
        command_list._obj.value = 0x500
        return 0

    def _list_destroy(self, command_list):
        self.list_destroyed = True
        return 0

    def _list_close(self, command_list):
        return 0

    def _append_copy(self, command_list, dst, src, size, event, wait_count, waits):
        dst_addr = dst.value if hasattr(dst, "value") else int(dst)
        src_addr = src.value if hasattr(src, "value") else int(src)
        if dst_addr == self.device_addr:
            ctypes.memmove(self.device_addr, src_addr, size)
        elif src_addr == self.device_addr:
            ctypes.memmove(dst_addr, self.device_addr, size)
        else:
            raise AssertionError("copy does not touch the device allocation")
        return 0

    def _free(self, context, pointer):
        self.freed = True
        return 0


def _run(fake, mib=1):
    return intel_l0.run_smoke(mib, loader="fake-ze", opener=lambda _name: fake)


def test_memory_smoke_round_trips_device_usm():
    fake = FakeL0()
    report = _run(fake)
    assert report["result"] == "pass"
    assert report["bytes"] == 1024 * 1024
    assert report["device_count"] == 1
    assert report["copy_queue_group"] == 0
    assert report["queue_groups"] == [{"ordinal": 0, "flags": "0x2", "num_queues": 1}]
    assert report["context"] is True
    assert report["shared_usm"] is True
    assert report["device_usm"] is True
    assert report["copy_round_trip"] is True
    assert report["mismatches"] == 0
    assert report["returncodes"]["zeInit"] == "0x00000000"
    assert fake.context_driver == 0x100       # the driver handle, not its index
    assert fake.freed is True
    assert fake.destroyed is True
    assert fake.queue_destroyed is True
    assert fake.list_destroyed is True
    assert "no kernel" in intel_l0.render(report)[0]


def test_memory_smoke_prefers_the_copy_only_queue_group():
    fake = FakeL0(group_flags=(0x7, 0x2))
    report = _run(fake)
    assert report["result"] == "pass"
    assert report["copy_queue_group"] == 1
    assert fake.queue_ordinal == 1
    assert fake.list_ordinal == 1
    assert [group["ordinal"] for group in report["queue_groups"]] == [0, 1]


def test_memory_smoke_fails_without_a_copy_capable_group():
    fake = FakeL0(group_flags=(0x1,))
    report = _run(fake)
    assert report["result"] == "fail"
    assert "no copy-capable queue group" in report["error"]
    assert report["copy_queue_group"] is None
    assert fake.queue_ordinal is None


def test_memory_smoke_reports_allocation_failure_and_cleans_up():
    fake = FakeL0()
    fake.alloc_result = 0x78000006
    report = _run(fake)
    assert report["result"] == "fail"
    assert "zeMemAllocDevice" in report["error"]
    assert "0x78000006" in report["error"]
    assert report["device_usm"] is False
    assert fake.freed is False
    assert fake.destroyed is True


def test_memory_smoke_rejects_bad_arguments_before_loading():
    assert _run(FakeL0(), 0)["error"].startswith("--mib")
    assert _run(FakeL0(), 257)["error"].startswith("--mib")
    report = intel_l0.run_smoke(1, loader="missing-ze", opener=lambda _name: (_ for _ in ()).throw(OSError("no loader")))
    assert report["result"] == "fail"
    assert "could not load missing-ze" in report["error"]
    assert report["loader"]["available"] is False


def test_memory_smoke_reports_out_of_range_device():
    report = intel_l0.run_smoke(1, loader="fake-ze", opener=lambda _name: FakeL0(), device=2)
    assert report["result"] == "fail"
    assert "outside the 1 visible device" in report["error"]


def test_intel_smoke_cli_prints_json_and_returns_failure(monkeypatch, capsys, tmp_path):
    from kindi import __main__ as cli

    report = _run(FakeL0())
    monkeypatch.setattr(intel_l0, "run_smoke", lambda *args, **kwargs: report)
    conf = tmp_path / "kindi.conf"
    conf.write_text("[kindi]\n", encoding="utf-8")
    monkeypatch.setenv("KINDI_CONFIG", str(conf))

    assert cli.main(["intel-smoke", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "pass"

    failed = dict(report, result="fail", error="synthetic failure")
    monkeypatch.setattr(intel_l0, "run_smoke", lambda *args, **kwargs: failed)
    assert cli.main(["intel-smoke"]) == 1
    assert "synthetic failure" in capsys.readouterr().out
