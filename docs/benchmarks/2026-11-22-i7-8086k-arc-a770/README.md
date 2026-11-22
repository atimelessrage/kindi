# Intel Arc A770 16 GB: M1 hardware and Level Zero qualification (2026-11-22)

The A770 is now installed in the i7-8086K rig. This page records what the host can prove **without a
Kindi Intel backend**: the card's identity, its memory, the Level Zero runtime, and the CPU baseline.
No model ran on the Arc, no SYCL kernel exists, and nothing here is a support claim
([REQ-0020](../../development/requirements/REQ-0020.md), [INTEL-ARC.md](../../INTEL-ARC.md)).

## Host

| item | measured value |
|---|---|
| CPU | Intel Core i7-8086K, 6 physical / 12 logical |
| OS | Windows 10 Pro, build 19045 (native) |
| Python | 3.13.6 |
| GPU | Intel(R) Arc(TM) A770 Graphics, PCI `8086:56A0`, subsystem `60101849`, rev 08 |
| dedicated memory | 16,384 MiB (17,179,869,184 bytes) |
| display driver | 32.0.101.9175, service `igfxnd`, PnP status OK |
| Level Zero loader | `ze_loader.dll` 1.33.1 |
| Level Zero driver | `ze_intel_gpu64.dll` 32.0.101.9175, registered by the display class (DriverStore) |
| Level Zero enumeration | `zeInit` 0, 1 driver, 1 device |
| oneAPI / SYCL | not installed; no `sycl-ls`, no `clinfo`, no `icpx` |
| OpenVINO | not installed |
| Resizable BAR | **not read** |
| PCIe link width / generation | **not read** |

`Win32_VideoController.AdapterRAM` reports 4,293,918,720 bytes for this 16 GiB card (the field is
32-bit), so the probe reads the driver's `HardwareInformation.qwMemorySize` instead. That is the
16,384 MiB row above.

## What ran

`python -m kindi intel-info --json` — a read-only probe; the full report is
[intel-info.json](intel-info.json). It reads the adapters from CIM, joins the display class registry
for VRAM and the Level Zero registration, and asks the loader for driver/device counts. It fills no
versioned device struct, so it reports counts rather than a device name read at a guessed offset.

```
Intel display devices:
  ? [32902:56a0] Intel(R) Arc(TM) A770 Graphics arch=Xe-HPG driver=n/a render=none integrated=no source=win32
    vram=16384 MiB, driver_version=32.0.101.9175, status=OK
Level Zero runtime: loader 1.33.1
  Intel(R) Arc(TM) A770 Graphics: driver=...\ze_intel_gpu64.dll version=32.0.101.9175
  enumeration: 1 driver(s), 1 device(s)
```

The CPU baseline used the clang 18.1.8 install on this host (`KINDI_CC` / `KINDI_LLVM`), 6 threads,
context 1,024, `KINDI_DEVICE=cpu`:

| pack | result | rate | library build |
|---|---|---|---|
| Bonsai 2B | 4/4 PASS (40/40, 40/40, 40/40, 40/40 tokens) | 22.2 tok/s | 31.7 s |
| Bonsai 4B | 4/4 PASS (40/40, 40/40, 40/40, 40/40 tokens) | 11.1 tok/s | 65.3 s |
| Bonsai 9B | 4/4 PASS (40/40, 40/40, 13/13, 40/40 tokens) | 6.5 tok/s | 138.8 s |

Build time is the first-run clang compile of `kcpu.c` into `~/.cache/kindi`; later runs reuse it.
These are CPU-engine numbers on the A770 host, not Arc numbers.

## M1 verdict

The plan's M1 is worded for Linux ([REQ-0020](../../development/requirements/REQ-0020.md)):
oneAPI/SYCL, `sycl-ls`, ReBAR and PCIe state, plus CPU and llama.cpp SYCL baselines. On this Windows
host:

| M1 item | state |
|---|---|
| A770 visible at 16 GiB | **pass** — CIM + `qwMemorySize` + PnP device id |
| Level Zero device visible | **pass** — 1 driver, 1 device |
| pinned oneAPI/SYCL toolchain | **not met** — no oneAPI, no `sycl-ls`/`icpx` |
| ReBAR recorded | **not met** — not exposed by CIM/WMI; needs Intel Arc Control or GPU-Z here |
| PCIe link state recorded | **not met** — no `lspci` on Windows |
| CPU baseline | **pass for 2B/4B/9B** — token-exact 4/4 each |
| llama.cpp SYCL baseline | **not run** — no SYCL toolchain |

So: **hardware and Level Zero are qualified; M1 is not passed** (toolchain, ReBAR, PCIe and the
llama.cpp baseline are open), and Windows is not the plan's first support target. M2 still has
nothing to compile with: a SYCL kernel needs oneAPI (or a Linux oneAPI host), and an unverifiable
kernel would be a claim rather than a feature.

## Notes

- The RTX 4060 Ti and GT 1030 appear only as non-present PnP nodes (Status Unknown), and
  `nvidia-smi` could not query a device, so this run has no CUDA comparison.
- The display class registers `igdrcl64.dll` (OpenCL), but no Khronos `OpenCL\Vendors` key is present
  in either registry view, so the OpenCL ICD loader would not find it. Kindi's plan is Level
  Zero/SYCL anyway; this is recorded to avoid a later "OpenCL was there all along" claim.
- The probe records `driver=n/a` for a Windows adapter: the display class exposes a driver
  *version*, not the Linux `i915`/`xe` service name that `eligible()` checks.
