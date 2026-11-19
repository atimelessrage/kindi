# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-compute lane for the Q2_0 routed experts of the Lightning big-MoE backend.

The Lightning path (Nemotron 3.5 Lightning 30B-A3B: 52 layers, 23 of them MoE,
128 routed experts chosen top-6 plus one shared expert, hidden 2688, expert
width 1856) keeps every expert in host RAM and lets the card read the selected
ones over PCIe. This module computes one of those experts on the host CPUs
instead, so a miss no longer has to cross PCIe as weights.

Exactness is the whole point of the lane, so it reproduces ``lightning.cu``'s
arithmetic rather than a mathematically equivalent one:

* a Q2_0 run is ``sum(c * q) - sum(q)`` over 32 ternary weights (codes 0/1/2,
  the int8 activation runs of ``lt_quant``), an exact int32;
* each run's value enters lane ``h % 32`` with one fused multiply-add, in run
  order (the kernel contracts ``f0 * dd * x`` into ``fma`` because NVRTC
  compiles the engine with ``--use_fast_math``; the probe in
  ``tools/q2_expert_lane.py`` measures this, it is not assumed);
* the 32 lane sums are reduced by the same five-step XOR tree.

``tools/q2_expert_lane.py`` gates this against the real ``lt_mv_q2`` and
``lt_mv_q2_down`` kernels on synthetic Q2_0 experts of the real geometry, bit
for bit, and measures it against the copy lane.

The lane does **not** quantize. The engine compiles ``lightning.cu`` with
``--use_fast_math``, whose ``div.approx.f32`` is not reproducible in host
floating point, so activations must come from the card's own ``lt_quant`` (as
``xq``/``dx``/``sx``). Everything this module computes from those runs is
bit-identical to the kernel's expert output.

It is not wired into ``kindi.lightning.Engine``: the first exact consumer is the
MoE tandem CPU lane (``docs/MOE-TANDEM.md`` M2), and the engine still needs the
combine contract for a layer that mixes card and CPU experts. No model tok/s is
claimed.
"""
import numpy as np

__all__ = ["LaneUnavailable", "available", "expert_bytes", "Q2Matrix", "matvec"]


class LaneUnavailable(RuntimeError):
    """numba (or its FMA intrinsic path) is not installed on this host."""


try:                                                    # optional dependency
    import llvmlite.ir as _ir
    from numba import njit, prange
    from numba.core import types
    from numba.extending import intrinsic
    _HAVE = True
except ImportError:                                     # pragma: no cover - host without numba
    _HAVE = False


def available():
    """True when this host can run the lane (numba with a working ``fmaf``)."""
    if not _HAVE:
        return False
    try:
        _probe_fma()
        return True
    except Exception:                                   # pragma: no cover - broken toolchain
        return False


def expert_bytes(rows, cols):
    """Bytes of one Q2_0 matrix: 16 code bytes and one fp16 scale per 64 weights."""
    if cols % 64:
        raise ValueError("cols must be a multiple of 64, not %r" % (cols,))
    return rows * (cols // 64) * 18


if _HAVE:
    @intrinsic
    def _fma32(typingctx, a, b, c):
        """One IEEE float32 fused multiply-add, the operation the kernel contracts to."""
        sig = types.float32(types.float32, types.float32, types.float32)

        def codegen(context, builder, signature, args):
            f32 = _ir.FloatType()
            fnty = _ir.FunctionType(f32, [f32, f32, f32])
            module = builder.module
            fn = next((f for f in module.functions if f.name == "fmaf"), None)
            if fn is None:
                fn = _ir.Function(module, fnty, "fmaf")
            return builder.call(fn, args)
        return sig, codegen

    @njit(fastmath=False)
    def _probe_math():
        # the intrinsic is a 3-argument callable at run time; the static tools see its
        # four-parameter definition (typing context included)
        return _fma32(np.float32(1.0000001), np.float32(2.0000002),  # type: ignore  # pylint: disable=no-value-for-parameter
                      np.float32(-2.0000005))

    @njit(parallel=True, fastmath=False)
    def _matvec_impl(codes, scales, xq, dx, sx, rows, runs, nb, expert):
        out = np.empty(rows, np.float32)
        cbase = expert * rows * runs * 2
        sbase = expert * rows * nb
        for r in prange(rows):  # pylint: disable=not-an-iterable
            acc = np.zeros(32, np.float32)
            row = cbase + r * runs * 2
            srow = sbase + r * nb
            for h in range(runs):
                idot = 0
                for grp in range(8):
                    word = np.uint32(codes[row + 2 * h + (0 if grp < 4 else 1)])
                    shift = 2 * (grp % 4)
                    qv = np.int32(xq[h * 8 + grp])
                    for j in range(4):
                        code = np.int32((word >> np.uint32(shift + 8 * j)) & np.uint32(3))
                        b = (qv >> (8 * j)) & 255
                        if b >= 128:
                            b -= 256
                        idot += code * b
                value = np.float32(idot - sx[h])
                scale = scales[srow + h // 2] * dx[h]
                lane = h % 32
                acc[lane] = _fma32(scale, value, acc[lane])  # type: ignore  # pylint: disable=no-value-for-parameter
            step = 16
            while step:
                old = acc.copy()
                for lane in range(32):
                    acc[lane] = old[lane] + old[lane ^ step]
                step >>= 1
            out[r] = acc[0]
        return out

def _probe_fma():
    impl = globals().get("_probe_math")
    return None if impl is None else impl()


def _matvec(*args):
    impl = globals().get("_matvec_impl")
    if impl is None:
        raise LaneUnavailable("the CPU expert lane needs numba with its fmaf intrinsic")
    return impl(*args)


class Q2Matrix(object):
    """Host-resident Q2_0 weights of one matrix stack: ``experts`` matrices of ``rows x cols``.

    ``codes`` is uint32 ``[experts, rows, cols / 16]`` (two words per 32-weight
    run, 16 weights per word); ``scales`` is the fp16 block scales renamed to
    float32 (exact) as ``[experts, rows, cols / 64]``. The layouts are the ones
    ``lightning.py`` repacks the GGUF file into.
    """

    def __init__(self, codes, scales, rows, cols, experts=None):
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0 or cols % 64:
            raise ValueError("rows must be positive and cols a multiple of 64, not %r x %r" % (rows, cols))
        runs, nb = cols // 32, cols // 64
        c = np.ascontiguousarray(codes, dtype=np.uint32)
        s = np.ascontiguousarray(scales, dtype=np.float32)
        if c.ndim == 1:
            if c.size % (rows * runs * 2):
                raise ValueError("codes has %d words, expected a multiple of %d" % (c.size, rows * runs * 2))
            n = c.size // (rows * runs * 2)
        elif c.ndim == 2:
            n = c.shape[0]
            if c.shape[1] != rows * runs * 2:
                raise ValueError("codes row has %d words, expected %d" % (c.shape[1], rows * runs * 2))
        elif c.ndim == 3:
            n = c.shape[0]
            if c.shape[1:] != (rows, runs * 2):
                raise ValueError("codes has shape %r, expected (%d, %d, %d)"
                                 % (c.shape, n, rows, runs * 2))
        else:
            raise ValueError("codes must be 1-D, 2-D or 3-D, not %d-D" % c.ndim)
        if s.ndim == 1:
            if s.size % (rows * nb):
                raise ValueError("scales has %d values, expected a multiple of %d" % (s.size, rows * nb))
            if s.size // (rows * nb) != n:
                raise ValueError("scales names %d experts, codes %d" % (s.size // (rows * nb), n))
        elif s.ndim in (2, 3):
            want = (n, rows * nb) if s.ndim == 2 else (n, rows, nb)
            if s.shape != want:
                raise ValueError("scales has shape %r, expected %r" % (s.shape, want))
        else:
            raise ValueError("scales must be 1-D, 2-D or 3-D, not %d-D" % s.ndim)
        if experts is not None and int(experts) != n:
            raise ValueError("experts=%r, but the arrays hold %d" % (experts, n))
        self.codes, self.scales = c.reshape(-1), s.reshape(-1)
        self.rows, self.cols, self.experts = rows, cols, n
        self.bytes = n * expert_bytes(rows, cols)

    def check_input(self, xq, dx, sx):
        runs = self.cols // 32
        q = np.ascontiguousarray(xq, dtype=np.int32).reshape(-1)
        d = np.ascontiguousarray(dx, dtype=np.float32).reshape(-1)
        s = np.ascontiguousarray(sx, dtype=np.int32).reshape(-1)
        if q.size != runs * 8:
            raise ValueError("xq has %d words, expected %d (%d runs of 32 values)" % (q.size, runs * 8, runs))
        if d.size != runs or s.size != runs:
            raise ValueError("dx/sx have %d/%d runs, expected %d" % (d.size, s.size, runs))
        return q, d, s

    def matvec(self, expert, xq, dx, sx):
        """``W_expert x`` for one expert, as ``lt_mv_q2`` computes it, bit for bit."""
        if not available():
            raise LaneUnavailable("the CPU expert lane needs numba with its fmaf intrinsic")
        expert = int(expert)
        if expert < 0 or expert >= self.experts:
            raise ValueError("expert %r is outside 0..%d" % (expert, self.experts - 1))
        q, d, s = self.check_input(xq, dx, sx)
        return _matvec(self.codes, self.scales, q, d, s, self.rows, self.cols // 32,
                       self.cols // 64, expert)


def matvec(codes, scales, rows, cols, expert, xq, dx, sx):
    """Convenience wrapper: build a one-off ``Q2Matrix`` and run one expert."""
    return Q2Matrix(codes, scales, rows, cols).matvec(expert, xq, dx, sx)
