# SPDX-License-Identifier: AGPL-3.0-only
"""Cold routed-expert reads from a pack/GGUF file, behind `KINDI_LT_COLD`.

The host expert tier in `kindi/lightning.py` is pinned RAM. Without this module the whole expert
set is a RAM requirement. With the opt-in cold tier the hottest host experts stay pinned up to a
host byte budget and only the overflow is file-backed. This module is the file side of that tier:
it reads the Q2_0 bytes of a selected expert range with an unbuffered read where the OS allows one
(``O_DIRECT`` on Linux, ``FILE_FLAG_NO_BUFFERING`` on Windows), and coalesces neighbouring ranges
into one read-ahead window. It also owns the pure host/disk split so the engine, tests and reports
use the same accounting. It does not allocate VRAM, run a kernel, or decide which expert is needed;
the engine owns those parts.

Every read is accounted for separately:

* ``requested_bytes``: the exact expert bytes the caller asked for;
* ``read_bytes``: bytes actually transferred, including alignment and read-ahead gaps;
* ``readahead_bytes``: the gap bytes fetched by merging neighbouring ranges;
* ``alignment_bytes``: block padding added by aligned reads.

``read_bytes == requested_bytes + readahead_bytes + alignment_bytes`` for every completed read.

A buffered fallback is reported as such. It is still correct, but its number is a page-cache rate,
not the device's, so a throughput claim must never use it silently.
"""
import ctypes
import mmap
import os

import numpy as np

ALIGN = 4096
COLD_ENV = "KINDI_LT_COLD"
HOST_ENV = "KINDI_LT_HOST_MB"
HOST_RESERVE = 1 << 30


def cold_enabled(environ=None):
    """True only for an explicit opt-in; an unset or false value keeps the pinned-RAM path."""
    if environ is None:
        value = os.environ.get("KINDI_LT_COLD", "")
    else:
        value = environ.get("KINDI_LT_COLD", "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def host_budget_bytes(environ=None, available=None):
    """Bytes available to pin host experts under the cold tier.

    ``KINDI_LT_HOST_MB`` is an explicit budget in MiB (0 pins none). Without it, use the host's
    currently available RAM less a safety margin: 10% or 1 GiB, whichever is larger. That keeps
    the hot tier automatic for a pack larger than RAM while leaving the OS and the rest of the
    process room to run. When neither an explicit value nor an available-RAM figure exists, the
    safe answer is 0 (nothing extra is pinned).
    """
    if environ is None:
        raw = os.environ.get("KINDI_LT_HOST_MB", "")
    else:
        raw = environ.get("KINDI_LT_HOST_MB", "")
    raw = str(raw).strip()
    if raw:
        try:
            mib = int(raw)
        except (TypeError, ValueError):
            raise ValueError("KINDI_LT_HOST_MB must be an integer number of MiB (got %r)" % raw)
        if mib < 0:
            raise ValueError("KINDI_LT_HOST_MB must not be negative (got %r)" % raw)
        return mib << 20
    if available is None:
        try:
            from .hostram import available as _available
            available = _available()
        except Exception:
            available = None
    if available is None:
        return 0
    reserve = max(HOST_RESERVE, int(available) // 10)
    return max(0, int(available) - reserve)


def split_host_tiers(host, order, budget, expert_bytes, counts=None):
    """Split non-card experts into pinned host and file-backed disk tiers.

    `host` maps a layer to its non-card expert ids; `order` maps a layer to the same ids in
    most-used-first order. When `counts` has a measured count for every host layer, the global
    ordering uses each expert's share of its layer's picks (every token passes every MoE layer
    once, so layers are comparable); otherwise it uses the order rank. The hottest experts that
    fit `budget` stay host-pinned and the rest become cold. Expert bytes are counted from the
    actual ids selected, not from the budget, so `host_bytes` is always the pinned set's size and
    is never larger than the budget.

    Returns `(warm, cold, stats)`: warm/cold are per-layer id tuples and stats contains the exact
    host/disk expert counts and bytes. Layer ordering is stable, so the split is reproducible.
    """
    if isinstance(expert_bytes, dict):
        eb = {il: int(expert_bytes[il]) for il in host}
    else:
        eb = {il: int(expert_bytes) for il in host}
    budget = max(0, int(budget))
    counts = counts or {}
    use_counts = bool(host) and all(il in counts and counts[il] for il in host)
    candidates = []
    for il in host:
        hs = [int(e) for e in host[il]]
        ordr = [int(e) for e in order.get(il, ())]
        rank = {e: i for i, e in enumerate(ordr)}
        for e in hs:
            rank.setdefault(e, len(rank))
        total = float(sum(counts[il])) if use_counts else 0.0
        for e in hs:
            if use_counts and total:
                key = (-(counts[il][e] / total), rank[e], il, e)
            else:
                key = (rank[e], il, e)
            candidates.append((key, il, e))
    candidates.sort(key=lambda row: row[0])
    warm = {il: [] for il in host}
    cold = {il: [] for il in host}
    room = budget
    for _key, il, e in candidates:
        if eb[il] <= room:
            warm[il].append(e)
            room -= eb[il]
        else:
            cold[il].append(e)
    warm = {il: tuple(sorted(v)) for il, v in warm.items()}
    cold = {il: tuple(sorted(v)) for il, v in cold.items()}
    host_bytes = sum(len(warm[il]) * eb[il] for il in host)
    cold_bytes = sum(len(cold[il]) * eb[il] for il in host)
    stats = {"budget_bytes": budget, "host_bytes": host_bytes, "disk_bytes": cold_bytes,
             "host_experts": sum(len(v) for v in warm.values()),
             "disk_experts": sum(len(v) for v in cold.values())}
    return warm, cold, stats


def cold_misses(ids, owners, cold):
    """Cold expert ids that need a read: selected, in the cold set, and not already in a warm slot.

    `owners` is the decode slot table's previous-step contents (an expert id per slot, -1 for a
    free slot). A warm owner is a hit even when the expert is cold, so it takes no file read."""
    warm = {int(e) for e in owners if int(e) >= 0}
    out = []
    for e in ids:
        e = int(e)
        if e in cold and e not in warm and e not in out:
            out.append(e)
    return out


def split_q2(raw, rows, cols_blocks):
    """Split the interleaved Q2_0 expert bytes into packed codes and fp16 scales.

    GGUF stores each 64-column block as 16 code bytes followed by a 2-byte scale. The kernels take
    two separate arrays, so a cold expert pays this de-interleave once per read. `cols_blocks` is
    `expert_cols // 64`; `rows * cols_blocks * 18` bytes are required.
    """
    raw = np.frombuffer(raw, np.uint8)
    need = int(rows) * int(cols_blocks) * 18
    if raw.size != need:
        raise ValueError("Q2_0 expert bytes: got %d, need %d" % (raw.size, need))
    blk = raw.reshape(int(rows) * int(cols_blocks), 18)
    codes = np.ascontiguousarray(blk[:, 2:]).reshape(-1)
    scales = np.ascontiguousarray(blk[:, :2]).reshape(-1)
    return codes, scales


class ColdExpertReader(object):
    """Read byte ranges from one file, unbuffered and with a bounded read-ahead merge.

    The file is opened lazily on the first read so a pack with no cold experts (for example, one
    that fits the card) never opens it. `read_many` sorts ranges by offset and merges ranges whose
    gap is at most ``read_ahead * alignment``; that is the read-ahead. If unbuffered I/O cannot be
    opened or used, the reader falls back to an ordinary unbuffered Python file and labels it.
    """

    def __init__(self, path, alignment=ALIGN, read_ahead=1, log=None):
        if int(alignment) < 512:
            raise ValueError("cold read alignment must be at least 512 bytes")
        if int(read_ahead) < 0:
            raise ValueError("cold read-ahead must not be negative")
        self.path = os.fspath(path)
        self.alignment = int(alignment)
        self.read_ahead = int(read_ahead)
        self.log = log
        self.closed = False
        self.unbuffered = None
        self.method = "not opened"
        self.reads = 0
        self.requested_bytes = 0
        self.read_bytes = 0
        self.readahead_bytes = 0
        self.alignment_bytes = 0
        self._fd = None
        self._handle = None
        self._k32 = None
        self._file = None

    # ------------------------------------------------------------------ opening
    def _open_posix(self):
        direct = getattr(os, "O_DIRECT", None)
        if direct is None or not hasattr(os, "preadv"):
            raise OSError("O_DIRECT/preadv unavailable")
        self._fd = os.open(self.path, os.O_RDONLY | direct)
        self.method = "O_DIRECT"

    def _open_windows(self):
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                         ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
        generic_read, share_read, open_existing = 0x80000000, 1, 3
        no_buffering, sequential = 0x20000000, 0x08000000
        handle = k32.CreateFileW(self.path, generic_read, share_read, None, open_existing,
                                 no_buffering | sequential, None)
        if not handle or handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self._k32, self._handle = k32, handle
        self.method = "FILE_FLAG_NO_BUFFERING"

    def _open(self):
        if self.closed:
            raise ValueError("cold reader is closed")
        if self._fd is not None or self._handle is not None or self._file is not None:
            return
        try:
            if os.name == "nt":
                self._open_windows()
            else:
                self._open_posix()
            self.unbuffered = True
        except OSError as exc:
            self._close_os()
            self._file = open(self.path, "rb", buffering=0)
            self.unbuffered = False
            self.method = "buffered fallback (%s)" % exc

    def _close_os(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._handle is not None and self._k32 is not None:
            try:
                self._k32.CloseHandle(self._handle)
            except Exception:
                pass
        self._handle, self._k32 = None, None

    def _switch_to_buffered(self, exc):
        self._close_os()
        self._file = open(self.path, "rb", buffering=0)
        self.unbuffered = False
        self.method = "buffered fallback (%s)" % exc
        if self.log:
            self.log("cold experts: unbuffered read unavailable (%s); using a buffered fallback" % exc)

    # ------------------------------------------------------------------ reads
    def _read_unbuffered_posix(self, start, size):
        buf = mmap.mmap(-1, size)
        view = memoryview(buf)
        try:
            got = 0
            while got < size:
                n = os.preadv(self._fd, [view[got:]], start + got)
                if n <= 0:
                    break
                got += n
            return bytes(view[:got])
        finally:
            view.release()
            buf.close()

    def _read_unbuffered_windows(self, start, size):
        from ctypes import wintypes

        buf = mmap.mmap(-1, size)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        try:
            if not self._k32.SetFilePointerEx(self._handle, ctypes.c_longlong(start), None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            got = 0
            while got < size:
                n = wintypes.DWORD(0)
                if not self._k32.ReadFile(self._handle, ctypes.c_void_p(addr + got),
                                         size - got, ctypes.byref(n), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if n.value == 0:
                    break
                got += n.value
            return bytes(memoryview(buf)[:got])
        finally:
            buf.close()

    def _read_buffered(self, start, size):
        self._file.seek(start)
        chunks = []
        got = 0
        while got < size:
            data = self._file.read(size - got)
            if not data:
                break
            chunks.append(data)
            got += len(data)
        return b"".join(chunks)

    def _read_range(self, start, size):
        """Read one aligned range and return exactly `size` bytes from `start`."""
        start = int(start)
        size = int(size)
        if size <= 0:
            return b""
        aligned_start = start - start % self.alignment
        aligned_end = (start + size + self.alignment - 1) // self.alignment * self.alignment
        span = aligned_end - aligned_start
        self._open()
        self.reads += 1
        if self.unbuffered:
            try:
                if os.name == "nt":
                    data = self._read_unbuffered_windows(aligned_start, span)
                else:
                    data = self._read_unbuffered_posix(aligned_start, span)
            except OSError as exc:
                self._switch_to_buffered(exc)
                data = self._read_buffered(aligned_start, span)
        else:
            data = self._read_buffered(aligned_start, span)
        self.read_bytes += len(data)
        self.alignment_bytes += span - size
        first = start - aligned_start
        if len(data) < first + size:
            raise IOError("cold read %s: short read at %d (%d of %d bytes)"
                          % (self.path, start, len(data), first + size))
        return data[first:first + size]

    def read(self, offset, size):
        """Read one exact range (the file read itself is aligned)."""
        spans = self.read_many([(offset, size)])
        return spans[0]

    def read_many(self, spans, read_ahead=None):
        """Read ranges in file order, merging gaps up to ``read_ahead * alignment``.

        Returns one ``bytes`` object per input span, in input order. Merged reads are the
        read-ahead: a neighbouring expert or the alignment tail is fetched in the same syscall,
        while ``requested_bytes`` still counts only what the caller asked for.
        """
        spans = [(int(o), int(n)) for o, n in spans]
        if not spans:
            return []
        ahead = self.read_ahead if read_ahead is None else max(0, int(read_ahead))
        order = sorted(range(len(spans)), key=lambda i: (spans[i][0], i))
        out = [None] * len(spans)
        i = 0
        while i < len(order):
            first = order[i]
            start, size = spans[first]
            end = start + size
            j = i + 1
            while j < len(order):
                nxt, nsize = spans[order[j]]
                if nxt <= end or nxt - end <= ahead * self.alignment:
                    end = max(end, nxt + nsize)
                    j += 1
                else:
                    break
            asked = sum(spans[q][1] for q in order[i:j])
            self.readahead_bytes += max(0, (end - start) - asked)
            data = self._read_range(start, end - start)
            for q in order[i:j]:
                off, size = spans[q]
                out[q] = data[off - start:off - start + size]
                self.requested_bytes += size
            i = j
        return out

    def stats(self):
        """The byte accounting a benchmark or a test may quote."""
        return {"path": self.path, "method": self.method, "unbuffered": self.unbuffered,
                "alignment": self.alignment, "read_ahead_blocks": self.read_ahead,
                "reads": self.reads, "requested_bytes": self.requested_bytes,
                "read_bytes": self.read_bytes, "readahead_bytes": self.readahead_bytes,
                "alignment_bytes": self.alignment_bytes}

    def close(self):
        if self.closed:
            return
        self._close_os()
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self.closed = True
