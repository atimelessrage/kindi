"""G0 slice of REQ-0021: Intel iGPU discovery/eligibility/budget (CPU-only)."""
import os

import pytest

from kindi import intel


def _fake_sysfs(tmp_path, cards):
    """Build a fake /sys/class/drm tree: [(card, vendor, device, driver)]."""
    root = tmp_path / "drm"
    for card, vendor, device, driver in cards:
        d = root / card / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor)
        (d / "device").write_text(device)
        (d / "uevent").write_text("MAJOR=226\nMINOR=0\nDEVNAME=%s\nDRIVER=%s\n" % (card, driver))
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    (dri / "renderD129").touch()
    return str(root), str(dri)


def test_discover_finds_intel_card_and_skips_others(tmp_path):
    drm, dri = _fake_sysfs(tmp_path, [
        ("card0", "0x8086", "0x9a49", "i915"),       # Iris Xe
        ("card1", "0x10de", "0x2484", "nvidia"),      # not Intel: skipped
    ])
    devs = intel.discover(drm_root=drm, dri_root=dri)
    assert len(devs) == 1
    dev = devs[0]
    assert dev["device_id"] == 0x9A49
    assert dev["name"] == "Intel Iris Xe Graphics"
    assert dev["arch"] == "Xe-LP"
    assert dev["driver"] == "i915"
    assert dev["render_node"].endswith("renderD128")


def test_discover_without_sysfs_is_honest(tmp_path):
    assert intel.discover(drm_root=str(tmp_path / "nope"), dri_root=str(tmp_path)) == []


def test_eligibility_floor():
    assert intel.eligible({"arch": "Xe-LP", "driver": "i915"})
    assert intel.eligible({"arch": "Xe2-LPG", "driver": "xe"})
    assert intel.eligible({"arch": "Xe-HPG", "driver": ""})
    assert not intel.eligible({"arch": None, "driver": "i915"})          # unknown id
    assert not intel.eligible({"arch": "Gen9", "driver": "i915"})        # UHD 630 class
    assert not intel.eligible({"arch": "Xe-LP", "driver": "snd_hda"})    # not bound to GPU driver


def test_eligibility_reason_names_the_cause():
    assert "unknown PCI" in intel.eligibility_reason({"arch": None, "device_id": 0x1234})
    assert "below the oneAPI floor" in intel.eligibility_reason(
        {"arch": "Gen9", "name": "Intel UHD 630", "driver": "i915"})
    assert "eligible" in intel.eligibility_reason(
        {"arch": "Xe2-LPG", "name": "Intel Arc 140V (built-in)", "driver": "xe"})


def test_shared_budget_rules():
    assert intel.shared_budget_mb(65536) == 32768          # Windows half-RAM cap
    assert intel.shared_budget_mb(65536, 57) == 37355      # Core Ultra 2+ override default
    assert intel.shared_budget_mb(65536, 93) == 60948      # Panther Lake example
    assert intel.shared_budget_mb(65536, 150) == 65536     # clamped


def test_fits_includes_reserve():
    assert intel.fits(32768, 5200, 2000)                   # 2B-class pack
    assert not intel.fits(7300, 5200, 2000)                # reserve pushes it out
    assert intel.fits(7300, 5200, 2000, reserve_mb=0)


def test_tandem_gpu_contract():
    class Good(object):
        D = 2048
        vocab = 1000

        def get_hidden(self):
            return None

        def set_hidden(self, h):
            pass

    class Bad(object):
        pass

    assert intel.tandem_gpu_contract(Good()) == []
    assert set(intel.tandem_gpu_contract(Bad())) == {"get_hidden", "set_hidden", "D", "vocab"}


def test_make_engine_refuses_intel_loudly(monkeypatch):
    from kindi import engine
    monkeypatch.setenv("KINDI_DEVICE", "intel")
    with pytest.raises(SystemExit) as exc:
        engine.make_engine("nonexistent-pack")
    assert "not implemented yet" in str(exc.value)
    assert "REQ-0020/REQ-0021" in str(exc.value)


def test_no_hard_oneapi_dependency():
    # kindi/intel.py must import and work with no oneAPI/SYCL present (this host).
    assert not os.path.exists("/opt/intel/oneapi")
    import kindi.intel  # noqa: F401
