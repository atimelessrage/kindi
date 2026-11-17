"""Measure stock Kindi v0.6.1 decode at position 512, separately from `kindi bench`.

Stock `kindi bench` prints generation only for positions that satisfy
`p + 20 < ctx` (100, 2000, 8000, ...), so its ctx-512 rows stop at position 100.
This harness calls the same stock engine, with the same `step(token, pos)` call
and the same 10-step timing the bench uses, for position 100 (a fidelity anchor)
and position 512. The context must hold position 512 plus the 10 steps, so the
default here is 8192; at ctx 512 a literal position 512 is outside the K/V cache.

It is not part of `kindi bench` and its rows are reported separately. No stock
module, kernel, plan or engine decision is patched: the only Windows change is
the same `libcuda.so.1` -> `nvcuda.dll` name shim `run_stock.py` uses.

Usage: python pos512_stock.py PACK [CTX]
"""
import ctypes
import os
import sys
import time

if os.name == "nt":
    _cdll = ctypes.CDLL

    def _cdll_windows(name, *args, **kwargs):
        if name == "libcuda.so.1":
            name = "nvcuda.dll"
        return _cdll(name, *args, **kwargs)

    ctypes.CDLL = _cdll_windows

import numpy as np

from kindi.engine import make_engine

POSITIONS = (100, 512)


def main(argv):
    pack = argv[1]
    ctx = int(argv[2]) if len(argv) > 2 else 8192
    import cupy as cp

    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=ctx, clusters=dict(np.load(cl)) if os.path.exists(cl) else None,
                      log=print)
    free, total = cp.cuda.runtime.memGetInfo()
    print("GPU memory: %.0f MiB free of %.0f with a %d-token context"
          % (free / 2 ** 20, total / 2 ** 20, eng.ctx))
    sync = cp.cuda.Device().synchronize
    eng.reset()
    eng.shortlist = 0                       # the bench's exact-head mode

    def timed(fn, n):
        fn(0)                               # warm-up, like the bench
        sync()
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        sync()
        return (time.perf_counter() - t0) / n * 1000

    # The bench measures prompt reading (20 x 512 tokens through feed) before its
    # generation sweep; run the same block untimed so clocks and kernels warm the
    # same way and the position-100 anchor can be checked against the sweep.
    timed(lambda i: eng.feed(1000 + i, 10 + i), 20)

    print("stock v0.6.1 position harness (separate from the bench sweep)")
    for p in POSITIONS:
        if p + 20 >= eng.ctx:
            print("generation (exact head) at position %5d: skipped (context %d)" % (p, eng.ctx))
            continue
        ms = timed(lambda i, p=p: eng.step(2000 + i, p + i), 10)
        print("generation (exact head) at position %5d: %5.1f tokens/s" % (p, 1000.0 / ms))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
