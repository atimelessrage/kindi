# SPDX-License-Identifier: AGPL-3.0-only
"""Cold routed-expert reads from a pack/GGUF file, behind `KINDI_LT_COLD`.

The host expert tier in `kindi/lightning.py` is pinned RAM: every expert the card does not hold is
copied there at load time. That makes the whole expert set a RAM requirement. This module is the
file side of the alternative tier: it reads the Q2_0 bytes of a selected expert range with an
unbuffered read where the OS allows one (``O_DIRECT`` on Linux, ``FILE_FLAG_NO_BUFFERING`` on
Windows), and coalesces neighbouring ranges into one read-ahead window. It does not allocate VRAM,
run a kernel, or decide which expert is needed; the engine owns those parts.

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


def cold_enabled(environ=None):
    """True only for an explicit opt-in; an unset or false value keeps the pinned-RAM path."""
    if environ is None:
        value = os.environ.get("KINDI_LT_COLD", "")
    else:
        value = environ.get("KINDI_LT_COLD", "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


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
        flags = os.O_RDONLY
        direct = getattr(os, "O_DIRECT", 0)
        self._fd = os.open(self.path, flags | direct)
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
