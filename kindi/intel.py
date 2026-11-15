"""Intel integrated graphics: discovery, eligibility and the shared-memory budget.

This is the G0 (hardware-qualification) slice of REQ-0021 / docs/INTEL-IGPU-TANDEM.md:
pure Python, no oneAPI dependency, so it runs and is testable on a CPU-only host.
It answers three questions without a SYCL kernel in sight:

- ``discover()``: which Intel GPUs are visible (Linux ``/dev/dri`` + sysfs)?
- ``eligible(dev)``: is the device at or above Intel's oneAPI floor
  (UHD for 11th generation or newer / Iris Xe or newer / built-in Arc)?
- ``shared_budget_mb()``: how much host RAM may the iGPU use (the fit input
  that replaces VRAM in the NVIDIA/AMD path)?

There is deliberately no engine here: the SYCL backend (REQ-0020) is the kernel
half, and it needs hardware to verify. ``KINDI_DEVICE=intel`` refuses clearly
from ``kindi/engine.py`` until that backend exists, instead of silently
falling back to the CPU engine.
"""
import glob
import os
import re

# PCI device-id -> (display name, architecture). Conservative by design: only
# ids whose architecture is known to sit at or above the oneAPI floor are
# listed; anything else is "unknown" and refused until measured (the same
# rule as docs/INTEL-IGPU-TANDEM.md: pre-11th-gen, e.g. UHD 630 / Gen9, is out).
# Extend this table only with a measured `sycl-ls` / Level Zero identity.
DEVICE_ARCH = {
    # Xe-LP (Gen12): 11th-13th gen Core UHD / Iris Xe
    0x9A49: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9A51: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9A71: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9AF0: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9BCA: ("Intel UHD Graphics 770", "Xe-LP"),
    0x9BC8: ("Intel UHD Graphics 730", "Xe-LP"),
    0x46A8: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x46A6: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x46B0: ("Intel UHD Graphics 750", "Xe-LP"),
    0x46A2: ("Intel UHD Graphics 710", "Xe-LP"),
    # Xe2-LPG: built-in Arc in Core Ultra (Meteor Lake / Lunar Lake / Panther Lake)
    0x7D55: ("Intel Arc 140V (built-in)", "Xe2-LPG"),
    0x7D67: ("Intel Arc 130V (built-in)", "Xe2-LPG"),
    0x7D51: ("Intel Arc 140T (built-in)", "Xe2-LPG"),
}

SUPPORTED_ARCH = frozenset(("Xe-LP", "Xe2-LPG", "Xe-HPG"))

# Hybrid kernel drivers that can drive a supported iGPU on Linux.
GPU_DRIVERS = frozenset(("i915", "xe"))

_SYS_DRM = "/sys/class/drm"
_DEV_DRI = "/dev/dri"


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return ""


def arch_of(device_id):
    """(display name, architecture) for a PCI device id, or (None, None)."""
    return DEVICE_ARCH.get(int(device_id), (None, None))


def discover(drm_root=_SYS_DRM, dri_root=_DEV_DRI, glob_fn=glob.glob):
    """List visible Intel GPUs from DRM sysfs. Returns [] where unsupported.

    Each entry is a dict: render_node, pci_id, device_id, name, arch, driver.
    On hosts without /dev/dri or sysfs (e.g. Windows), this honestly returns [].
    """
    found = []
    if not os.path.isdir(drm_root):
        return found
    for card in sorted(glob_fn(os.path.join(drm_root, "card[0-9]*"))):
        vendor = _read(os.path.join(card, "device", "vendor"))
        if vendor.lower() != "0x8086":
            continue
        device = _read(os.path.join(card, "device", "device"))
        try:
            device_id = int(device, 16)
        except ValueError:
            continue
        uevent = _read(os.path.join(card, "device", "uevent"))
        driver = ""
        for line in uevent.splitlines():
            if line.startswith("DRIVER="):
                driver = line.split("=", 1)[1].strip()
        name, arch = arch_of(device_id)
        render = ""
        nodes = sorted(glob_fn(os.path.join(dri_root, "renderD*")))
        # renderD128+ are compute-capable; match by card index order.
        card_index = os.path.basename(card)[4:]
        try:
            idx = int(card_index)
            if 0 <= idx < len(nodes):
                render = nodes[idx]
        except ValueError:
            pass
        found.append({
            "render_node": render,
            "pci_id": os.path.basename(card),
            "device_id": device_id,
            "name": name,
            "arch": arch,
            "driver": driver,
        })
    return found


def eligible(dev):
    """True when the device clears the oneAPI floor (known arch + GPU driver).

    Unknown architectures are refused, not assumed: the same conservatism as
    the support matrix, which renders unmeasured cards as `not measured`.
    """
    if dev.get("arch") not in SUPPORTED_ARCH:
        return False
    if dev.get("driver") and dev["driver"] not in GPU_DRIVERS:
        return False
    return True


def eligibility_reason(dev):
    """One human-readable line explaining the eligible() verdict."""
    if dev.get("arch") is None:
        return "unknown PCI device id 0x%04x: not in the measured table" % dev.get("device_id", 0)
    if dev.get("arch") not in SUPPORTED_ARCH:
        return "%s (%s) is below the oneAPI floor (11th gen / Iris Xe or newer)" % (
            dev.get("name") or "device", dev["arch"])
    if dev.get("driver") and dev["driver"] not in GPU_DRIVERS:
        return "%s is bound to %s, not a GPU driver (i915/xe)" % (dev.get("name"), dev["driver"])
    return "%s (%s, %s) is eligible" % (dev.get("name"), dev["arch"], dev.get("driver") or "driver unknown")


def shared_budget_mb(total_ram_mb, override_pct=None):
    """Usable iGPU memory in MiB.

    Windows caps the graphics share at one-half of system RAM (Intel graphics
    memory FAQ). Core Ultra Series 2+ adds a Shared GPU Memory Override
    (default 57%, up to 87-93% with enough RAM); pass ``override_pct`` to use
    it, e.g. ``shared_budget_mb(65536, 87)``. Linux i915/xe has no such fixed
    cap: pass ``override_pct=100`` minus what the OS keeps, or measure the
    device's reported limit instead of guessing.
    """
    if override_pct is None:
        return int(total_ram_mb) // 2
    pct = max(0, min(100, int(override_pct)))
    return int(total_ram_mb) * pct // 100


def fits(budget_mb, weights_mb, kv_mb, reserve_mb=512):
    """True when a ternary pack fits the shared budget with headroom.

    ``weights_mb``/``kv_mb`` come from the pack manifest (`catalog.py`
    figures); the iGPU shares the bus with the CPU, so a reserve for the OS
    and the CPU-side layers is part of the fit, not an afterthought.
    """
    return int(weights_mb) + int(kv_mb) + int(reserve_mb) <= int(budget_mb)


def tandem_gpu_contract(engine):
    """Check an engine exposes the TandemEngine GPU-side surface.

    The tandem boundary needs ``get_hidden()`` / ``set_hidden()`` plus the
    layer-range constructor surface. A future SYCL iGPU engine must pass
    this before ``TandemEngine`` may wrap it; returns a list of missing
    attributes (empty = contract satisfied).
    """
    missing = []
    for attr in ("get_hidden", "set_hidden", "D", "vocab"):
        if not hasattr(engine, attr):
            missing.append(attr)
    return missing
