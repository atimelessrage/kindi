#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the native Gemma 4 FP8 kernels on Blackwell sm_100/sm_120.

This is a hardware lane, not a model benchmark.  It refuses to report a pass unless the active
CUDA device is exactly compute capability 10.0 or 12.0, then exercises the E4M3/E5M2 quantizer,
the tensor-core GEMM at full-tile and tail shapes, and the ``Ops.linear_batch`` dispatch.  The
reference is the same dequantized f32 product used by ``tests/test_gemma4_fp8_mma.py``.

Exit codes:
  0  all hardware checks passed
  1  a check failed (including a disabled native path or a kernel mismatch)
  2  unsupported hardware, no CUDA device, or an arch mismatch (never a pass)

The result is written to ``results.json`` and ``RESULTS.md`` even when the run is unsupported, so
the CI artifact records why no Blackwell result exists.  ``--dry-run`` needs no CUDA and writes
nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCH_BY_CC = {100: "sm_100", 120: "sm_120"}
ARCH_CHOICES = ("auto", "sm_100", "sm_120")
QUANT_SHAPE = (5, 64)
GEMM_SHAPES = ((64, 64, 64), (37, 96, 5), (130, 128, 70))
OPS_SHAPE = (64, 64, 8)
TOLERANCE = 2e-3


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cc_int(compute_capability):
    """Normalize an int or a ``(major, minor)`` pair to CuPy's integer form."""
    if isinstance(compute_capability, (tuple, list)) and len(compute_capability) == 2:
        return int(compute_capability[0]) * 10 + int(compute_capability[1])
    return int(compute_capability)


def arch_label(compute_capability):
    """Return ``sm_100``/``sm_120`` for a supported capability, else None."""
    try:
        return ARCH_BY_CC.get(_cc_int(compute_capability))
    except (TypeError, ValueError):
        return None


def arch_matches(compute_capability, requested="auto"):
    """Whether a device capability is covered by the requested lane."""
    if requested not in ARCH_CHOICES:
        raise ValueError("unsupported arch %r" % (requested,))
    label = arch_label(compute_capability)
    return label is not None and (requested == "auto" or requested == label)


def _add(checks, name, ok, detail, elapsed_s=0.0):
    check = {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "elapsed_ms": round(elapsed_s * 1000.0, 3),
    }
    checks.append(check)
    print(("PASS " if ok else "FAIL ") + name + ": " + detail)
    return check


def _finite_bytes(kind):
    from kindi.gemma4 import fp8_to_f32

    values = fp8_to_f32(np.arange(256, dtype=np.uint8), kind)
    return np.flatnonzero(np.isfinite(values)).astype(np.uint8)


def _weights(rows, cols, kind, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(_finite_bytes(kind), size=rows * cols).astype(np.uint8)


def _quantize(cp, fp8, x, kind):
    nt, cols = x.shape
    q = cp.empty((nt, cols), dtype=cp.uint8)
    s = cp.empty(nt, dtype=cp.float32)
    fp8["quant"][kind](
        (nt,),
        (256,),
        (cp.asarray(x, dtype=cp.float32), q, s, np.int32(cols), np.int32(nt)),
    )
    return q, s


def _dequant_rows(q, s, kind, cols):
    from kindi.gemma4 import fp8_to_f32

    return fp8_to_f32(q.get(), kind).reshape(q.shape[0], cols) * s.get()[:, None]


def _quant_check(cp, fp8, kind, checks):
    start = time.perf_counter()
    try:
        nt, cols = QUANT_SHAPE
        x = np.linspace(-1.0, 1.0, nt * cols, dtype=np.float32).reshape(nt, cols)
        q, s = _quantize(cp, fp8, x, kind)
        cp.cuda.Stream.null.synchronize()
        got = _dequant_rows(q, s, kind, cols)
        maxv = 448.0 if kind == 43 else 57344.0
        finite = bool(np.isfinite(got).all())
        peak = bool(
            np.allclose(
                np.abs(got).max(axis=1), np.abs(x).max(axis=1), rtol=0.02, atol=0.0
            )
        )
        close = bool(np.allclose(got, x, rtol=0.13, atol=1e-5 * maxv))
        _add(
            checks,
            "quantize-%s" % ("e4m3" if kind == 43 else "e5m2"),
            finite and peak and close,
            "finite=%s peak=%s roundtrip=%s" % (finite, peak, close),
            time.perf_counter() - start,
        )
    except Exception as exc:
        _add(
            checks,
            "quantize-%s" % ("e4m3" if kind == 43 else "e5m2"),
            False,
            "exception: %s" % exc,
            time.perf_counter() - start,
        )


def _gemm_check(cp, fp8, kind, rows, cols, nt, checks):
    name = "gemm-%s-%dx%dx%d" % (
        "e4m3" if kind == 43 else "e5m2",
        rows,
        cols,
        nt,
    )
    start = time.perf_counter()
    try:
        rs = np.random.RandomState(rows * 131 + cols * 7 + nt)
        w = _weights(rows, cols, kind, seed=rows + cols + nt)
        x = (rs.standard_normal((nt, cols)) * 4.0).astype(np.float32)
        q, s = _quantize(cp, fp8, x, kind)
        out = cp.empty((nt, rows), dtype=cp.float32)
        fp8["mm"][kind](
            ((rows + 63) // 64, (nt + 63) // 64),
            (256,),
            (
                cp.asarray(w),
                q,
                s,
                out,
                np.int32(rows),
                np.int32(cols),
                np.int32(nt),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        got = cp.asnumpy(out)
        from kindi.gemma4 import fp8_to_f32

        wf = fp8_to_f32(w, kind).reshape(rows, cols).astype(np.float32)
        ref = _dequant_rows(q, s, kind, cols) @ wf.T
        err = float(np.abs(got - ref).max())
        limit = float(TOLERANCE * np.abs(ref).max())
        ok = bool(np.isfinite(got).all() and err <= limit)
        _add(
            checks,
            name,
            ok,
            "max_abs_error=%.6g limit=%.6g finite=%s" % (err, limit, np.isfinite(got).all()),
            time.perf_counter() - start,
        )
    except Exception as exc:
        _add(checks, name, False, "exception: %s" % exc, time.perf_counter() - start)


def _ops_check(cp, fp8, kind, checks):
    name = "ops-linear-%s" % ("e4m3" if kind == 43 else "e5m2")
    start = time.perf_counter()
    try:
        from kindi.gemma4 import Ops, fp8_to_f32

        rows, cols, nt = OPS_SHAPE
        rs = np.random.RandomState(23)
        w = _weights(rows, cols, kind, seed=5)
        x = (rs.standard_normal((nt, cols)) * 4.0).astype(np.float32)
        ops = Ops.__new__(Ops)
        # The native FP8 branch of linear_batch uses only cp and fp8; do not compile the unrelated
        # general Gemma 4 module just to construct an Ops object.
        ops.cp, ops.k, ops.fp8 = cp, None, fp8
        got = cp.asnumpy(
            ops.linear_batch(cp.asarray(w), cp.asarray(x), rows, cols, kind)
        )
        q, s = _quantize(cp, fp8, x, kind)
        ref = _dequant_rows(q, s, kind, cols) @ fp8_to_f32(w, kind).reshape(rows, cols).T
        err = float(np.abs(got - ref).max())
        limit = float(TOLERANCE * np.abs(ref).max())
        ok = bool(np.isfinite(got).all() and err <= limit)
        _add(
            checks,
            name,
            ok,
            "max_abs_error=%.6g limit=%.6g finite=%s" % (err, limit, np.isfinite(got).all()),
            time.perf_counter() - start,
        )
    except Exception as exc:
        _add(checks, name, False, "exception: %s" % exc, time.perf_counter() - start)


def _device_info(cp):
    dev = cp.cuda.Device()
    cc = _cc_int(dev.compute_capability)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    total = props.get("totalGlobalMem") or props.get("total_global_mem") or 0
    return {
        "index": int(dev.id),
        "name": str(name),
        "compute_capability": cc,
        "arch": arch_label(cc),
        "total_memory_mib": round(float(total) / (1024.0 * 1024.0), 1),
    }


def _runtime_info(cp):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        info["cupy"] = cp.__version__
    except Exception:
        pass
    for key, getter in (
        ("cuda_driver", cp.cuda.runtime.driverGetVersion),
        ("cuda_runtime", cp.cuda.runtime.runtimeGetVersion),
    ):
        try:
            info[key] = int(getter())
        except Exception:
            pass
    return info


def write_report(out, payload):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks = payload.get("checks") or []
    lines = [
        "# Blackwell FP8 verification",
        "",
        "Status: **%s**" % payload.get("status", "unknown"),
        "",
        "- requested arch: `%s`" % payload.get("requested_arch", "auto"),
        "- detected arch: `%s`" % (payload.get("arch") or "n/a"),
        "- started: `%s`" % payload.get("started_utc", "n/a"),
        "- finished: `%s`" % payload.get("finished_utc", "n/a"),
    ]
    device = payload.get("device")
    if device:
        lines.extend(
            [
                "- device: %s (device %s), compute capability %s"
                % (
                    device.get("name", "unknown"),
                    device.get("index", "?"),
                    device.get("compute_capability", "?"),
                ),
                "- memory: %s MiB" % device.get("total_memory_mib", "?"),
            ]
        )
    runtime = payload.get("runtime") or {}
    if runtime:
        lines.append(
            "- runtime: Python %s, NumPy %s, CuPy %s, CUDA driver %s, CUDA runtime %s"
            % (
                runtime.get("python", "?"),
                runtime.get("numpy", "?"),
                runtime.get("cupy", "?"),
                runtime.get("cuda_driver", "?"),
                runtime.get("cuda_runtime", "?"),
            )
        )
    if payload.get("error"):
        lines.extend(["", "Error: `%s`" % payload["error"]])
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | result | detail | elapsed ms |",
            "|---|---|---|---:|",
        ]
    )
    for check in checks:
        detail = str(check.get("detail", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| `%s` | %s | %s | %s |"
            % (
                check.get("name", "?"),
                "PASS" if check.get("ok") else "FAIL",
                detail,
                check.get("elapsed_ms", "?"),
            )
        )
    lines.extend(
        [
            "",
            "A pass requires a Blackwell device with compute capability exactly 100 or 120, "
            "loaded native FP8 kernels, and every check above green. Unsupported hardware exits 2.",
            "",
        ]
    )
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "run.log").write_text("\n".join(lines), encoding="utf-8")


def _finish(payload, checks, out, status, code, error=None):
    payload["status"] = status
    payload["checks"] = checks
    payload["finished_utc"] = _utc_now()
    if error:
        payload["error"] = str(error)
    try:
        write_report(out, payload)
    except Exception as exc:
        print("kindi: could not write the Blackwell result: %s" % exc, file=sys.stderr)
        if code == 0:
            code = 1
    print("Blackwell FP8 verification: %s (exit %d)" % (status, code))
    return code


def run(args):
    checks = []
    started = time.perf_counter()
    payload = {
        "status": "unsupported",
        "requested_arch": args.arch,
        "arch": None,
        "device": None,
        "runtime": {},
        "started_utc": _utc_now(),
        "command": " ".join(
            [
                sys.executable,
                str(Path(__file__).relative_to(ROOT)),
                "--arch",
                args.arch,
                "--out",
                str(args.out),
            ]
        ),
    }
    try:
        import cupy as cp
    except Exception as exc:
        _add(checks, "cuda-import", False, "CuPy/CUDA is unavailable: %s" % exc)
        return _finish(
            payload, checks, args.out, "unsupported", 2, error="CuPy/CUDA is unavailable"
        )

    try:
        if getattr(cp.cuda.runtime, "is_hip", False):
            raise RuntimeError("this lane requires CUDA, not HIP")
        device = _device_info(cp)
    except Exception as exc:
        _add(checks, "cuda-device", False, "no usable CUDA device: %s" % exc)
        return _finish(
            payload, checks, args.out, "unsupported", 2, error="no usable CUDA device"
        )

    payload["device"] = device
    payload["arch"] = device["arch"]
    payload["runtime"] = _runtime_info(cp)
    cc = device["compute_capability"]
    if not arch_matches(cc, args.arch):
        _add(
            checks,
            "device-arch",
            False,
            "compute capability %d (%s); this lane requires 100/sm_100 or 120/sm_120"
            % (cc, device["arch"] or "not Blackwell"),
        )
        return _finish(
            payload,
            checks,
            args.out,
            "unsupported",
            2,
            error="device does not match the requested Blackwell lane",
        )
    _add(
        checks,
        "device-arch",
        True,
        "compute capability %d (%s)" % (cc, device["arch"]),
    )

    if os.environ.get("KINDI_G4_FP8", "1") == "0":
        _add(
            checks,
            "native-kernels",
            False,
            "KINDI_G4_FP8=0 disables the native FP8 path",
        )
        return _finish(
            payload,
            checks,
            args.out,
            "fail",
            1,
            error="native FP8 path is disabled",
        )

    try:
        from kindi.gemma4 import fp8_kernels

        fp8 = fp8_kernels()
    except Exception as exc:
        _add(checks, "native-kernels", False, "exception: %s" % exc)
        return _finish(payload, checks, args.out, "fail", 1)
    if fp8 is None:
        _add(
            checks,
            "native-kernels",
            False,
            "fp8_kernels() returned None (NVRTC/PTX load failed or the path is unavailable)",
        )
        return _finish(payload, checks, args.out, "fail", 1)
    _add(checks, "native-kernels", True, "E4M3 and E5M2 kernels loaded")

    for kind in (43, 44):
        _quant_check(cp, fp8, kind, checks)
    for kind in (43, 44):
        for rows, cols, nt in GEMM_SHAPES:
            _gemm_check(cp, fp8, kind, rows, cols, nt, checks)
    for kind in (43, 44):
        _ops_check(cp, fp8, kind, checks)

    elapsed = time.perf_counter() - started
    payload["elapsed_s"] = round(elapsed, 3)
    status = "pass" if checks and all(c["ok"] for c in checks) else "fail"
    return _finish(payload, checks, args.out, status, 0 if status == "pass" else 1)


def dry_run(args):
    print("Blackwell FP8 verification lane (dry run, no CUDA imported)")
    print("Requires the active CUDA device to report compute capability 100 or 120.")
    print("Requested arch: %s" % args.arch)
    print("Checks that would run:")
    print("  - device and arch/index/name/memory record")
    print("  - E4M3 and E5M2 per-token quantization round trip")
    for rows, cols, nt in GEMM_SHAPES:
        print("  - native GEMM shapes %d rows x %d cols x %d tokens (both kinds)" % (rows, cols, nt))
    print("  - Ops.linear_batch native dispatch (both kinds)")
    print("Unsupported hardware would exit 2 and must not be reported as a pass.")
    print("No result file is written by --dry-run.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        choices=ARCH_CHOICES,
        default="auto",
        help="require sm_100 or sm_120, or auto for either Blackwell architecture",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "bench-logs" / "blackwell-fp8"),
        help="directory for results.json, RESULTS.md and run.log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the checks without importing CUDA or writing a result",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
