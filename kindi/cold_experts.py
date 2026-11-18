# SPDX-License-Identifier: AGPL-3.0-only
"""Cold routed-expert reads from a pack/GGUF file, behind `KINDI_LT_COLD`.

The host expert tier in `kindi/lightning.py` is pinned RAM. Without this module the whole expert
set is a RAM requirement. With the opt-in cold tier the hottest host experts stay pinned up to a
host byte budget and only the overflow is file-backed. This module is the file side of that tier:
it reads the Q2_0 bytes of a selected expert range with an unbuffered read where the OS allows one
(``O_DIRECT`` on Linux, ``FILE_FLAG_NO_BUFFERING`` on Windows), coalesces neighbouring ranges into
one aligned read, and keeps a bounded lookahead queue for ranges requested ahead of demand. It
also owns the pure host/disk split so the engine, tests and reports use the same accounting. It
does not allocate VRAM, run a kernel, or decide which expert is needed; the engine owns those
parts.

Every read is accounted for separately:

* ``requested_bytes``: the exact expert bytes the caller asked for;
* ``read_bytes``: bytes actually transferred, including alignment, coalesced gaps and prefetches;
* ``coalesced_bytes``: the gap bytes fetched by coalescing neighbouring ranges;
* ``prefetched_bytes``: the exact bytes fetched by the lookahead queue;
* ``prefetch_used_bytes``: prefetched bytes later returned to a demand read;
* ``alignment_bytes``: block padding added by aligned reads.

``read_bytes == requested_bytes + coalesced_bytes + alignment_bytes + prefetched_bytes - prefetch_used_bytes``
for every completed read: a cached prefetch is charged once when fetched, not again on demand.

A buffered fallback is reported as such. It is still correct, but its number is a page-cache rate,
not the device's, so a throughput claim must never use it silently.
"""
import ctypes
import mmap
import os
from collections import OrderedDict

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
    parsed = {}
    for il in host:
        row = counts.get(il)
        if not isinstance(row, (list, tuple)) or not row:
            continue
        if any(int(e) < 0 or int(e) >= len(row) for e in host[il]):
            continue
        try:
            values = [float(v) for v in row]
        except (TypeError, ValueError):
            continue
        if any(v < 0 or v != v or v in (float("inf"), float("-inf")) for v in values):
            continue
        if sum(values) > 0:
            parsed[il] = values
    counted = len(parsed)
    use_counts = bool(host) and counted == len(host)
    if use_counts:
        counts = parsed
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
             "disk_experts": sum(len(v) for v in cold.values()),
             "counted_layers": counted, "counts_used": use_counts}
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
    """Read byte ranges from one file, unbuffered, with bounded coalescing and prefetch.

    The file is opened lazily on the first read so a pack with no cold experts (for example, one
    that fits the card) never opens it. `read_many` sorts ranges by offset and coalesces ranges
    whose gap is at most ``coalesce * alignment`` into one aligned read. `prefetch` reads exact
    ranges into a bounded lookahead queue; a later `read_many` consumes those bytes before it
    touches the file. If unbuffered I/O cannot be opened or used, the reader falls back to an
    ordinary unbuffered Python file and labels it.
    """

    def __init__(self, path, alignment=ALIGN, coalesce=1, log=None,
                 prefetch_bytes=64 << 20, read_ahead=None):
        if int(alignment) < 512:
            raise ValueError("cold read alignment must be at least 512 bytes")
        if read_ahead is not None:                 # temporary compatibility with the old name
            if coalesce != 1:
                raise ValueError("pass coalesce, not both coalesce and read_ahead")
            coalesce = read_ahead
        if int(coalesce) < 0:
            raise ValueError("cold coalesce blocks must not be negative")
        if int(prefetch_bytes) < 0:
            raise ValueError("cold prefetch bytes must not be negative")
        self.path = os.fspath(path)
        self.alignment = int(alignment)
        self.coalesce = int(coalesce)
        self.prefetch_bytes = int(prefetch_bytes)
        self.log = log
        self.closed = False
        self.unbuffered = None
        self.method = "not opened"
        self.reads = 0
        self.requested_bytes = 0
        self.read_bytes = 0
        self.coalesced_bytes = 0
        self.alignment_bytes = 0
        self.prefetch_requests = 0
        self.prefetched_bytes = 0
        self.prefetch_used_bytes = 0
        self.prefetch_hits = 0
        self._prefetch = OrderedDict()
        self._prefetch_cached = 0
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

    def _read_spans(self, spans, ahead):
        """Read exact spans, coalescing gaps up to ``ahead * alignment``; no demand accounting."""
        if not spans:
            return []
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
            self.coalesced_bytes += max(0, (end - start) - asked)
            data = self._read_range(start, end - start)
            for q in order[i:j]:
                off, size = spans[q]
                out[q] = data[off - start:off - start + size]
            i = j
        return out

    def _cache_put(self, span, data):
        """Put one exact span in the bounded lookahead queue, evicting oldest first."""
        if not data or len(data) > self.prefetch_bytes or span in self._prefetch:
            return False
        while self._prefetch and self._prefetch_cached + len(data) > self.prefetch_bytes:
            _, old = self._prefetch.popitem(last=False)
            self._prefetch_cached -= len(old)
        self._prefetch[span] = data
        self._prefetch_cached += len(data)
        return True

    def prefetch(self, spans):
        """Fetch exact ranges into the lookahead queue for a later ``read_many``.

        Returns the number of newly queued spans. Already queued spans are not read again, and
        the queue drops its oldest entries when ``prefetch_bytes`` is full. This is real
        lookahead: the bytes are read now and a demand read later consumes them without another
        file read.
        """
        spans = [(int(o), int(n)) for o, n in spans if int(n) > 0]
        if not spans or self.prefetch_bytes <= 0:
            return 0
        seen, missing = set(), []
        for span in spans:
            if span in seen or span in self._prefetch:
                continue
            seen.add(span)
            missing.append(span)
        if not missing:
            return 0
        got = self._read_spans(missing, self.coalesce)
        self.prefetch_requests += len(missing)
        self.prefetched_bytes += sum(n for _, n in missing)
        return sum(1 for span, raw in zip(missing, got) if self._cache_put(span, raw))

    def read_many(self, spans, coalesce=None, read_ahead=None):
        """Read ranges in file order, consuming queued prefetches and coalescing adjacent gaps.

        Returns one ``bytes`` object per input span, in input order. ``coalesce`` is the gap
        window in alignment blocks (the old ``read_ahead`` keyword is accepted for callers that
        have not migrated). Queued prefetch bytes are used first, so a demand read that follows
        ``prefetch`` does not touch the file again.
        """
        spans = [(int(o), int(n)) for o, n in spans]
        if not spans:
            return []
        if coalesce is None:
            ahead = self.coalesce if read_ahead is None else max(0, int(read_ahead))
        else:
            ahead = max(0, int(coalesce))
        out = [None] * len(spans)
        missing = []
        for i, span in enumerate(spans):
            if span in self._prefetch:
                out[i] = self._prefetch.pop(span)
                self._prefetch_cached -= len(out[i])
                self.prefetch_hits += 1
                self.prefetch_used_bytes += len(out[i])
            else:
                missing.append(i)
        if missing:
            got = self._read_spans([spans[i] for i in missing], ahead)
            for i, raw in zip(missing, got):
                out[i] = raw
        self.requested_bytes += sum(n for _, n in spans)
        return out

    def stats(self):
        """The byte accounting a benchmark or a test may quote."""
        return {"path": self.path, "method": self.method, "unbuffered": self.unbuffered,
                "alignment": self.alignment, "coalesce_blocks": self.coalesce,
                "prefetch_limit_bytes": self.prefetch_bytes,
                "prefetch_cached_bytes": self._prefetch_cached,
                "prefetch_requests": self.prefetch_requests,
                "prefetched_bytes": self.prefetched_bytes,
                "prefetch_used_bytes": self.prefetch_used_bytes,
                "prefetch_hits": self.prefetch_hits,
                "reads": self.reads, "requested_bytes": self.requested_bytes,
                "read_bytes": self.read_bytes, "coalesced_bytes": self.coalesced_bytes,
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
