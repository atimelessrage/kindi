"""The DSpine drafter's selection gate, without a GPU.

0.6.0's 27B drafter is GPU-unverified in this repo: the risk is silent misrouting (speculation
on/off, weights absent) rather than kernel math. These tests pin the CPU-visible gate:
`wanted()` (env-driven routing to the drafter) and `fits()` (the shape/backend refusal) plus
the memory reserve. `fits()` reads `cp.cuda.runtime.is_hip` and `spec.HP`, both stubbed.
"""
import os
from types import SimpleNamespace

import pytest

from kindi import dspine


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("KINDI_SPEC", "KINDI_DSPINE", "KINDI_DSPINE_MB", "KINDI_DSPINE_K"):
        monkeypatch.delenv(k, raising=False)


def test_wanted_is_off_by_default():
    assert not dspine.wanted(None)


@pytest.mark.parametrize("spec", ["dspine", "auto"])
def test_wanted_needs_spec_and_weights(monkeypatch, spec):
    monkeypatch.setenv("KINDI_SPEC", spec)
    assert not dspine.wanted(None)                    # weights directory not named
    monkeypatch.setenv("KINDI_DSPINE", "/tmp/dspine")
    assert dspine.wanted(None)


def test_wanted_needs_a_spec_flag(monkeypatch):
    monkeypatch.setenv("KINDI_DSPINE", "/tmp/dspine")
    for spec in ("0", "2b", ""):
        monkeypatch.setenv("KINDI_SPEC", spec)
        assert not dspine.wanted(None), spec


def test_reserve_default_and_override(monkeypatch):
    assert dspine.reserve_mb() == 1400
    monkeypatch.setenv("KINDI_DSPINE_MB", "900")
    assert dspine.reserve_mb() == 900


class _HipRuntime(object):
    is_hip = True


def _spec_dim(D=5120, NL=64, W=1, lane=False, layout=True):
    return SimpleNamespace(D=D, NL=NL, tp=SimpleNamespace(W=W), tc_lane=lane, tc_layout=layout)


def _fits_with(monkeypatch, hip=False, hp="0"):
    """Run dspine.fits against a stubbed runtime (the real one needs a driver init)."""
    monkeypatch.setattr(dspine, "cp", SimpleNamespace(
        cuda=SimpleNamespace(runtime=_HipRuntime() if hip else SimpleNamespace(is_hip=False))))
    monkeypatch.setenv("KINDI_TC_HP", hp)
    import kindi.spec as spec_mod
    monkeypatch.setattr(spec_mod, "HP", hp == "1")
    return dspine.fits(_spec_dim())


def test_fits_accepts_the_27b_shape(monkeypatch):
    assert _fits_with(monkeypatch)


@pytest.mark.parametrize("kw", [
    {"D": 2560}, {"NL": 24}, {"W": 2}, {"lane": True}, {"layout": False},
])
def test_fits_refuses_other_shapes(monkeypatch, kw):
    monkeypatch.setattr(dspine, "cp", SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False))))
    monkeypatch.setenv("KINDI_TC_HP", "0")
    import kindi.spec as spec_mod
    monkeypatch.setattr(spec_mod, "HP", False)
    assert not dspine.fits(_spec_dim(**kw)), kw


def test_fits_refuses_hip_and_hp(monkeypatch):
    assert not _fits_with(monkeypatch, hip=True)
    assert not _fits_with(monkeypatch, hp="1")
