# SPDX-License-Identifier: AGPL-3.0-only
"""The split-ratio tool folds measurements into a predicted rate table; the pure parts must be
right before any bench cycle is spent on them."""
import importlib.util
from pathlib import Path

import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        "measure_split_ratio", Path(__file__).resolve().parents[1] / "tools" / "measure_split_ratio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_predicted_rate_falls_with_the_cpu_layer_count():
    mod = load()
    slow = mod.predicted_tok_s(0.0, 8.0, 0.5, 0.2, 64, 16, 24576)
    fast = mod.predicted_tok_s(0.0, 8.0, 0.5, 0.2, 64, 4, 24576)
    assert fast > slow > 0                                # fewer CPU layers, faster token
    full_gpu = mod.predicted_tok_s(0.0, 8.0, 0.5, 0.2, 64, 0, 24576)
    assert slow < full_gpu


def test_full_gpu_row_reproduces_the_recorded_token_time():
    mod = load()
    # the recorded 27B full-GPU token time at 24K is 33.1 ms (30.2 tok/s); the per-layer
    # cost in this model is token_time / layer_count, with embed+head lumped into it
    gpu_layer = 33.1 / 64
    rate = mod.predicted_tok_s(0.0, 8.0, gpu_layer, 0.0, 64, 0, 24576)
    assert rate == pytest.approx(30.2, abs=0.2)


def test_effective_cpu_cost_is_the_probe_slope_after_gpu_reduction():
    mod = load()
    probe = {11: 78.0, 13: 82.6, 16: 90.5, 20: 100.0}   # token ms, 4 GDN layers apart
    eff = mod.effective_cpu_ms(probe, 0.5, 64)
    assert eff == pytest.approx((100.0 - 78.0) / 9 - 0.5, abs=0.001)
    assert mod.effective_cpu_ms({}, 0.5, 64) is None
