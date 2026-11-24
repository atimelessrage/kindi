# Intel Arc A770: the SYCL vector-add smoke on Level Zero (2026-11-24)

REQ-0020 M2 needs a SYCL runtime that selects the Arc **through Level Zero**, not through the
OpenCL driver that the earlier probes used. This page records the first run of that smoke on the
Windows test rig: one exact vector-add, three sizes, every acceptance check the plan named.

It is a host-local precursor, not M2 itself. M2's target is the Linux rig, and one exact vector-add
is not a model, engine or Kindi kernel result. Nothing here changes the support matrix or the
installer, and the report carries `support_claim: false`.

## 1. The host saw the card once oneAPI's own `bin` directories were on `PATH`

`sycl-ls` exists in the oneAPI install, but running it by absolute path with a bare `PATH` found
only OpenCL devices. The Unified Runtime Level Zero adapter is loaded from the oneAPI `bin`
directories, so without them there is no adapter to enumerate the card through
([sycl-ls.txt](sycl-ls.txt) is the run with them on `PATH`):

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.17.39894+6]
```

Two consequences are now in the code. `kindi/intel.py` locates the oneAPI install (`KINDI_ONEAPI_ROOT`,
then `ONEAPI_ROOT`, then the default root) and runs the installed `sycl-ls` with the install's `bin`
directories prepended to `PATH`, so `kindi intel-info` reports the Level Zero device instead of
"none". And the parser accepts the current `[backend:kind][backend:index]` layout that this
`sycl-ls` prints; the older `[backend:kind:index]` layout still parses. Before that fix the probe ran
the command and then dropped every line it returned.

## 2. The run

`python -m kindi intel-sycl --json` with `KINDI_ONEAPI_ROOT` naming the oneAPI install
([sycl-smoke.json](sycl-smoke.json), a second invocation in [sycl-smoke.txt](sycl-smoke.txt)):

| part | value |
|---|---|
| compiler | oneAPI 2026.2 `icpx` (DPC++), after `vcvars64.bat` from VS Community 2026 18 |
| build | ok, source sha256 `eff03a5f…`, binary sha256 `83b3b81e…` |
| device | `Intel(R) Arc(TM) A770 Graphics`, backend `ext_oneapi_level_zero`, driver `1.17.39894+6` |
| memory | 15,932 MiB global reported by SYCL, 64 KiB local |
| kernel | `nd_range<1>`, local size 64, `i < n` bounds guard |
| inputs | integer-valued float32 (`a[i] = (31*i) % 1000`, `b[i] = (17*i) % 1000`) |

| elements | repeats | mismatches | max_abs_error | 5 launches |
|---|---|---|---|---|
| 1,048,576 | 5 | 0 | 0 | 14.38 ms |
| 1 | 5 | 0 | 0 | 1.18 ms |
| 1,000,003 | 5 | 0 | 0 | 4.53 ms |

The comparison is exact, not a tolerance: the host reference is computed independently and compared
bit for bit over every element. The 1,000,003 case is not a multiple of the local size, so the guard
is what keeps it in range. The second copy of the run measured 14.77 / 1.55 / 4.52 ms with the same
zeros; these are observations of a launch-bound smoke, not a throughput measurement.

## 3. Acceptance checks

| # | check | result |
|---|---|---|
| A1 | the runtime sees the card | pass: one `[level_zero:gpu]` line naming the A770 ([sycl-ls.txt](sycl-ls.txt)) |
| A2 | selection is explicit | pass: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls` prints only that line, no host or OpenCL device ([sycl-ls-selector.txt](sycl-ls-selector.txt)) |
| A3 | build | pass: `icpx -fsycl -O2`, exit 0, both hashes recorded |
| A4 | the child asserts its device | pass: backend `ext_oneapi_level_zero`, `is_gpu()` true, name contains `A770`, 15,932 MiB global |
| A5 | exactness | pass: 0 mismatches and `max_abs_error = 0` for every size and repeat, over all 1,048,576 elements |
| A6 | bounds guard | pass: 1,000,003 elements, 0 mismatches |
| A7 | no CPU/OpenCL fallback | pass: `--selector opencl:gpu` exits 1 and reports the child's refusal, `no level_zero:gpu device (runtime offered: … (opencl))` ([selector-refusal.txt](selector-refusal.txt)); the driver also rejects a child report whose backend is not Level Zero |
| A8 | failure is fail | pass: the non-Level-Zero child above is a `result: fail` with exit 1, and the unit tests cover build failure and launch failure |
| A9 | report | pass: `probe: "intel-sycl-smoke"`, `schema: 1`, `support_claim: false`, `scope`, `toolchain`, `device`, `build`, `vector_add`, `result`, `error` |
| A10 | host tests | pass: `tests/test_intel_sycl.py` and `tests/test_intel.py` need no oneAPI, compiler or GPU |

## 4. What this does not show

- **Not M2.** The plan's M2 is the Linux rig; this is Windows, where the vendor runtime and driver
  pair differ. The same smoke still has to run there.
- **Not a Kindi kernel.** A vector-add at 1M elements measures launch and copy, so the times say
  nothing about the q8 GEMM or a model. The GEMM pages stay the throughput record
  ([`../2026-11-24-i7-8086k-arc-a770-gemm-tiled/`](../2026-11-24-i7-8086k-arc-a770-gemm-tiled/)).
- **Not a support claim.** `intel-info` and `intel-sycl` are probes; the engine still has no SYCL
  backend and `docs/SUPPORT-MATRIX.md` is unchanged.
- **Device selection is still by name.** The `level_zero:0` index in the `sycl-ls` line is not
  stable across reboots and driver updates, so the smoke selects by backend and GPU kind, never by
  index.

The machine roots are replaced with `<oneapi-root>`, `<vs-root>` and `<workdir>` in the saved
copies; the live commands print them.
