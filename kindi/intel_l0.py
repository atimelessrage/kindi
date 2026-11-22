"""Level Zero device-memory smoke test for Intel Arc.

This is the first executable slice of M2 in REQ-0020 / ``docs/INTEL-ARC.md``:
it proves that the installed Level Zero runtime can create a context, select
an Intel device, allocate device USM and copy a known buffer host -> device ->
host without a CPU fallback.  It launches no kernel, loads no model and makes
no performance or support claim.

The SYCL/oneAPI kernel half is still required for ``kindi convert``/``verify``
and a real Arc backend.  Keeping the memory path separate means this check can
run on the Windows A770 host now, while the kernel gate remains explicit.
"""
import ctypes
import json
import os
import time

ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xE
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0xF
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
ZE_RESULT_SUCCESS = 0
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
ZE_COMMAND_LIST_FLAG_DEFAULT = 0
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 0x1
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = 0x2
ZE_UINT64_MAX = 0xffffffffffffffff
MAX_MIB = 256
DEFAULT_MIB = 1
DEFAULT_LOADER = "ze_loader.dll" if os.name == "nt" else "libze_loader.so.1"


class _ContextDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32)]


class _DeviceMemAllocDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("ordinal", ctypes.c_uint32)]


class _HostMemAllocDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32)]


class _QueueGroupProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("max_fill_pattern_size", ctypes.c_size_t),
                ("num_queues", ctypes.c_uint32)]


class _CommandQueueDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("ordinal", ctypes.c_uint32),
                ("index", ctypes.c_uint32),
                ("flags", ctypes.c_uint32),
                ("mode", ctypes.c_uint32),
                ("priority", ctypes.c_uint32)]


class _CommandListDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("command_queue_group_ordinal", ctypes.c_uint32),
                ("flags", ctypes.c_uint32)]


def _hex(code):
    return "0x%08x" % (int(code) & 0xffffffff)


def _base_report(loader, mib):
    return {
        "probe": "intel-l0-smoke",
        "schema": 1,
        "read_only": False,
        "support_claim": False,
        "backend": "level-zero memory path only; no kernel or engine",
        "scope": "device USM allocation and H2D/D2H copy; no kernel launch or model",
        "loader": {"name": loader, "available": False, "error": None},
        "mib": mib,
        "bytes": int(mib) * 1024 * 1024 if isinstance(mib, int) else None,
        "device": None,
        "device_index": None,
        "device_count": None,
        "queue_groups": [],
        "copy_queue_group": None,
        "context": False,
        "shared_usm": False,
        "device_usm": False,
        "copy_round_trip": False,
        "mismatches": None,
        "first_mismatch": None,
        "elapsed_ms": None,
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def _open(loader, opener):
    open_library = opener or (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)
    try:
        return open_library(loader), None
    except (OSError, TypeError, AttributeError) as exc:
        return None, "could not load %s: %s" % (loader, exc)


def _bind(lib):
    lib.zeInit.restype = ctypes.c_int
    lib.zeInit.argtypes = [ctypes.c_uint32]
    lib.zeDriverGet.restype = ctypes.c_int
    lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                                ctypes.POINTER(ctypes.c_void_p)]
    lib.zeDeviceGet.restype = ctypes.c_int
    lib.zeDeviceGet.argtypes = [ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint32),
                                ctypes.POINTER(ctypes.c_void_p)]
    lib.zeContextCreate.restype = ctypes.c_int
    lib.zeContextCreate.argtypes = [ctypes.c_void_p,
                                    ctypes.POINTER(_ContextDesc),
                                    ctypes.POINTER(ctypes.c_void_p)]
    lib.zeContextDestroy.restype = ctypes.c_int
    lib.zeContextDestroy.argtypes = [ctypes.c_void_p]
    lib.zeMemAllocDevice.restype = ctypes.c_int
    lib.zeMemAllocDevice.argtypes = [ctypes.c_void_p,
                                     ctypes.POINTER(_DeviceMemAllocDesc),
                                     ctypes.c_size_t, ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p)]
    lib.zeMemAllocShared.restype = ctypes.c_int
    lib.zeMemAllocShared.argtypes = [ctypes.c_void_p,
                                     ctypes.POINTER(_DeviceMemAllocDesc),
                                     ctypes.POINTER(_HostMemAllocDesc),
                                     ctypes.c_size_t, ctypes.c_size_t,
                                     ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p)]
    lib.zeDeviceGetCommandQueueGroupProperties.restype = ctypes.c_int
    lib.zeDeviceGetCommandQueueGroupProperties.argtypes = [ctypes.c_void_p,
                                                           ctypes.POINTER(ctypes.c_uint32),
                                                           ctypes.POINTER(_QueueGroupProperties)]
    lib.zeCommandQueueCreate.restype = ctypes.c_int
    lib.zeCommandQueueCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.POINTER(_CommandQueueDesc),
                                         ctypes.POINTER(ctypes.c_void_p)]
    lib.zeCommandQueueDestroy.restype = ctypes.c_int
    lib.zeCommandQueueDestroy.argtypes = [ctypes.c_void_p]
    lib.zeCommandQueueExecuteCommandLists.restype = ctypes.c_int
    lib.zeCommandQueueExecuteCommandLists.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                      ctypes.POINTER(ctypes.c_void_p),
                                                      ctypes.c_void_p]
    lib.zeCommandQueueSynchronize.restype = ctypes.c_int
    lib.zeCommandQueueSynchronize.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.zeCommandListCreate.restype = ctypes.c_int
    lib.zeCommandListCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.POINTER(_CommandListDesc),
                                        ctypes.POINTER(ctypes.c_void_p)]
    lib.zeCommandListDestroy.restype = ctypes.c_int
    lib.zeCommandListDestroy.argtypes = [ctypes.c_void_p]
    lib.zeCommandListClose.restype = ctypes.c_int
    lib.zeCommandListClose.argtypes = [ctypes.c_void_p]
    lib.zeCommandListAppendMemoryCopy.restype = ctypes.c_int
    lib.zeCommandListAppendMemoryCopy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_size_t,
                                                  ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.c_void_p]
    lib.zeMemFree.restype = ctypes.c_int
    lib.zeMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def _devices(lib, report):
    count = ctypes.c_uint32(0)
    code = int(lib.zeDriverGet(ctypes.byref(count), None))
    report["returncodes"]["zeDriverGet"] = _hex(code)
    if code != ZE_RESULT_SUCCESS:
        return None, "zeDriverGet returned %s" % _hex(code)
    if count.value == 0:
        return [], None
    drivers = (ctypes.c_void_p * count.value)()
    code = int(lib.zeDriverGet(ctypes.byref(count), drivers))
    report["returncodes"]["zeDriverGetHandles"] = _hex(code)
    if code != ZE_RESULT_SUCCESS:
        return None, "zeDriverGet(handles) returned %s" % _hex(code)

    found = []
    for driver_index, driver in enumerate(drivers):
        n = ctypes.c_uint32(0)
        code = int(lib.zeDeviceGet(driver, ctypes.byref(n), None))
        if code != ZE_RESULT_SUCCESS:
            return None, "zeDeviceGet returned %s" % _hex(code)
        if not n.value:
            continue
        handles = (ctypes.c_void_p * n.value)()
        code = int(lib.zeDeviceGet(driver, ctypes.byref(n), handles))
        if code != ZE_RESULT_SUCCESS:
            return None, "zeDeviceGet(handles) returned %s" % _hex(code)
        found.extend((driver, driver_index, handle) for handle in handles)
    return found, None


def _copy_group(lib, device_handle, report):
    """A copy-capable queue group, preferring a copy-only engine."""
    count = ctypes.c_uint32(0)
    code = int(lib.zeDeviceGetCommandQueueGroupProperties(device_handle,
                                                          ctypes.byref(count), None))
    report["returncodes"]["zeDeviceGetCommandQueueGroupProperties"] = _hex(code)
    if code != ZE_RESULT_SUCCESS:
        return None, "zeDeviceGetCommandQueueGroupProperties returned %s" % _hex(code)
    if not count.value:
        return None, "the device advertises no command queue groups"
    groups = (_QueueGroupProperties * count.value)()
    for group in groups:
        group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES
        group.pNext = None
    code = int(lib.zeDeviceGetCommandQueueGroupProperties(device_handle,
                                                          ctypes.byref(count), groups))
    report["returncodes"]["zeDeviceGetCommandQueueGroupPropertiesQuery"] = _hex(code)
    if code != ZE_RESULT_SUCCESS:
        return None, "queue-group query returned %s" % _hex(code)
    candidates = []
    for ordinal, group in enumerate(groups):
        entry = {"ordinal": ordinal, "flags": "0x%x" % group.flags,
                 "num_queues": group.num_queues}
        report["queue_groups"].append(entry)
        if group.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY and group.num_queues:
            candidates.append((ordinal, group.flags))
    if not candidates:
        return None, "the device advertises no copy-capable queue group"
    # Prefer a copy-only group.  On the measured A770 driver, group 0 advertises
    # COMPUTE|COPY|COOPERATIVE but silently ignores copy commands; group 1 is
    # the copy engine and executes them.
    copy_only = [item for item in candidates
                 if not item[1] & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE]
    ordinal = (copy_only or candidates)[0][0]
    report["copy_queue_group"] = ordinal
    return ordinal, None


def _pattern(size):
    seed = bytes((i * 31 + 7) & 0xff for i in range(256))
    return (seed * (size // len(seed) + 1))[:size]


def run_smoke(mib=DEFAULT_MIB, loader=None, opener=None, device=0):
    """Allocate device USM and verify a host->device->host byte round trip.

    ``opener`` is injectable so the ABI and failure paths are testable without
    Level Zero.  The real path uses the host's loader and the selected device;
    it never falls back to host memory or a CPU engine.
    """
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, mib)
    if not isinstance(mib, int) or not DEFAULT_MIB <= mib <= MAX_MIB:
        return _fail(report, "--mib must be an integer in [%d, %d]" % (DEFAULT_MIB, MAX_MIB))
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    lib, error = _open(loader, opener)
    if lib is None:
        return _fail(report, error or "Level Zero loader unavailable")
    report["loader"]["available"] = True
    try:
        _bind(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "Level Zero loader is missing an API: %s" % exc)

    context = ctypes.c_void_p()
    host_src = ctypes.c_void_p()
    host_out = ctypes.c_void_p()
    allocation = ctypes.c_void_p()
    queue = ctypes.c_void_p()
    command_list = ctypes.c_void_p()
    try:
        code = int(lib.zeInit(0))
        report["returncodes"]["zeInit"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeInit returned %s" % _hex(code))

        devices, error = _devices(lib, report)
        if devices is None:
            return _fail(report, error or "Level Zero device enumeration failed")
        report["device_count"] = len(devices)
        if not devices:
            return _fail(report, "Level Zero reports no devices")
        if device >= len(devices):
            return _fail(report, "--device %d is outside the %d visible device(s)" % (device, len(devices)))
        driver_handle, driver_index, device_handle = devices[device]
        report["device_index"] = device
        report["device"] = {"driver_index": driver_index, "ordinal": device}
        copy_group, error = _copy_group(lib, device_handle, report)
        if copy_group is None:
            return _fail(report, error or "no copy queue group")

        context_desc = _ContextDesc(ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
        code = int(lib.zeContextCreate(driver_handle, ctypes.byref(context_desc),
                                       ctypes.byref(context)))
        report["returncodes"]["zeContextCreate"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeContextCreate returned %s" % _hex(code))
        report["context"] = True

        size = mib * 1024 * 1024
        alloc_desc = _DeviceMemAllocDesc(ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, None, 0, 0)
        code = int(lib.zeMemAllocDevice(context, ctypes.byref(alloc_desc), size, 64,
                                        device_handle, ctypes.byref(allocation)))
        report["returncodes"]["zeMemAllocDevice"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeMemAllocDevice returned %s" % _hex(code))
        report["device_usm"] = True

        host_desc = _HostMemAllocDesc(ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, None, 0)
        code = int(lib.zeMemAllocShared(context, ctypes.byref(alloc_desc), ctypes.byref(host_desc),
                                        size, 64, device_handle, ctypes.byref(host_src)))
        report["returncodes"]["zeMemAllocSharedSource"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeMemAllocShared(source) returned %s" % _hex(code))
        code = int(lib.zeMemAllocShared(context, ctypes.byref(alloc_desc), ctypes.byref(host_desc),
                                        size, 64, device_handle, ctypes.byref(host_out)))
        report["returncodes"]["zeMemAllocSharedOutput"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeMemAllocShared(output) returned %s" % _hex(code))
        report["shared_usm"] = True

        queue_desc = _CommandQueueDesc(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, None, copy_group, 0,
                                       0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                       ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
        code = int(lib.zeCommandQueueCreate(context, device_handle, ctypes.byref(queue_desc),
                                           ctypes.byref(queue)))
        report["returncodes"]["zeCommandQueueCreate"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeCommandQueueCreate returned %s" % _hex(code))

        list_desc = _CommandListDesc(ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, None, copy_group,
                                     ZE_COMMAND_LIST_FLAG_DEFAULT)
        code = int(lib.zeCommandListCreate(context, device_handle, ctypes.byref(list_desc),
                                          ctypes.byref(command_list)))
        report["returncodes"]["zeCommandListCreate"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeCommandListCreate returned %s" % _hex(code))

        source = _pattern(size)
        ctypes.memmove(host_src, source, size)
        started = time.perf_counter()
        code = int(lib.zeCommandListAppendMemoryCopy(command_list, allocation, host_src, size,
                                                     None, 0, None))
        report["returncodes"]["zeCommandListAppendMemoryCopyH2D"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "host-to-device append copy returned %s" % _hex(code))
        code = int(lib.zeCommandListAppendMemoryCopy(command_list, host_out,
                                                     allocation, size, None, 0, None))
        report["returncodes"]["zeCommandListAppendMemoryCopyD2H"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "device-to-host append copy returned %s" % _hex(code))
        code = int(lib.zeCommandListClose(command_list))
        report["returncodes"]["zeCommandListClose"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeCommandListClose returned %s" % _hex(code))
        lists = (ctypes.c_void_p * 1)(command_list)
        code = int(lib.zeCommandQueueExecuteCommandLists(queue, 1, lists, None))
        report["returncodes"]["zeCommandQueueExecuteCommandLists"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeCommandQueueExecuteCommandLists returned %s" % _hex(code))
        code = int(lib.zeCommandQueueSynchronize(queue, ZE_UINT64_MAX))
        report["returncodes"]["zeCommandQueueSynchronize"] = _hex(code)
        if code != ZE_RESULT_SUCCESS:
            return _fail(report, "zeCommandQueueSynchronize returned %s" % _hex(code))
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

        actual = ctypes.string_at(host_out, size)
        report["copy_round_trip"] = actual == source
        if actual == source:
            report["mismatches"] = 0
            report["result"] = "pass"
        else:
            first = next((i for i, (a, b) in enumerate(zip(actual, source, strict=True))
                          if a != b), None)
            report["mismatches"] = sum(1 for a, b in zip(actual, source, strict=True) if a != b)
            report["first_mismatch"] = first
            report["result"] = "fail"
            report["error"] = "copy mismatch at byte %s" % first
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "Level Zero memory path failed: %s" % exc)
    finally:
        if command_list:
            code = int(lib.zeCommandListDestroy(command_list))
            report["returncodes"]["zeCommandListDestroy"] = _hex(code)
        if queue:
            code = int(lib.zeCommandQueueDestroy(queue))
            report["returncodes"]["zeCommandQueueDestroy"] = _hex(code)
        if host_out:
            code = int(lib.zeMemFree(context, host_out))
            report["returncodes"]["zeMemFreeSharedOutput"] = _hex(code)
        if host_src:
            code = int(lib.zeMemFree(context, host_src))
            report["returncodes"]["zeMemFreeSharedSource"] = _hex(code)
        if allocation:
            code = int(lib.zeMemFree(context, allocation))
            report["returncodes"]["zeMemFree"] = _hex(code)
        if context:
            code = int(lib.zeContextDestroy(context))
            report["returncodes"]["zeContextDestroy"] = _hex(code)


def render(report):
    """Human-readable form of the memory-path smoke report."""
    loader = report["loader"]
    lines = ["Level Zero memory smoke (no kernel, no model, no support claim)",
             "  loader: %s%s" % (loader["name"], "" if loader["available"] else " (unavailable)")]
    if report["device_count"] is not None:
        lines.append("  devices: %s; selected: %s" % (report["device_count"], report["device_index"]))
    if report["bytes"]:
        lines.append("  allocation: %d MiB device USM + shared USM%s" % (
            report["mib"], " (allocated)" if report["device_usm"] and report["shared_usm"] else ""))
    if report["copy_round_trip"] is not None and report["device_usm"] and report["shared_usm"]:
        lines.append("  host -> device -> host: %s; mismatches: %s" % (
            "pass" if report["copy_round_trip"] else "fail", report["mismatches"]))
    if report["elapsed_ms"] is not None:
        lines.append("  copy elapsed: %.3f ms" % report["elapsed_ms"])
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
