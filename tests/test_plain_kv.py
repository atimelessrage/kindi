# SPDX-License-Identifier: AGPL-3.0-only
"""The plain engine's K/V format selection (kindi/engine.py:_kv_mode), CPU-only.

fp16 stays the default (the 2B/4B/9B verify references are fp16); q8_0 is an opt-in, and the tandem's
plain GPU owner passes kv="q8_0" so a split frees the offloaded layers' K/V too. k8v4 is refused here.
"""
import pytest

from kindi import engine


def test_default_is_fp16(monkeypatch):
    monkeypatch.delenv("KINDI_KV", raising=False)
    assert engine._kv_mode() == "f16"
    assert engine._kv_mode(None) == "f16"


def test_q8_0_opt_in(monkeypatch):
    for v in ("q8_0", "q8", "Q8_0", " q8_0 "):
        monkeypatch.setenv("KINDI_KV", v)
        assert engine._kv_mode() == "q8_0"
    monkeypatch.delenv("KINDI_KV")
    assert engine._kv_mode("q8_0") == "q8_0"
    assert engine._kv_mode("f16") == "f16"


def test_k8v4_is_refused_on_the_plain_engine(monkeypatch):
    monkeypatch.setenv("KINDI_KV", "k8v4")
    with pytest.raises(SystemExit, match="k8v4 is not supported by the plain engine"):
        engine._kv_mode()
    monkeypatch.delenv("KINDI_KV")
    with pytest.raises(SystemExit, match="kv must be f16 or q8_0"):
        engine._kv_mode("k8v4")


def test_unknown_format_is_refused(monkeypatch):
    monkeypatch.setenv("KINDI_KV", "fp32")
    with pytest.raises(SystemExit, match="must be f16 or q8_0"):
        engine._kv_mode()
    with pytest.raises(SystemExit, match="kv must be f16 or q8_0"):
        engine._kv_mode("fp32")
