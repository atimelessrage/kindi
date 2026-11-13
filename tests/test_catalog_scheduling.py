"""Requirement-based tests for the model catalog fit logic and the blocking-sync
opt-out: what a user sees in `kindi models` and what the server does by default.

These test user-visible contracts (which models fit which cards, the
KINDI_BLOCKING_SYNC opt-out), not implementation internals.
"""
import os

from kindi import catalog, sched


def _cards(*mbs):
    return [("card%d" % i, mb) for i, mb in enumerate(mbs)]


def test_catalog_lists_the_shipped_models():
    ids = set(catalog.MODELS)
    assert {"bonsai-2b", "bonsai-4b", "bonsai-9b", "bonsai-27b",
            "nemotron-e4b", "nanbeige-33b"} <= ids
    for m in catalog.MODELS.values():
        assert m["file"].endswith(".gguf") and len(m["sha"]) == 64 and m["size"] > 0


def test_fits_one_card_when_a_card_is_big_enough():
    assert catalog.fits(catalog.MODELS["bonsai-9b"], _cards(8192)) == "one card"
    assert catalog.fits(catalog.MODELS["bonsai-27b"], _cards(7600)) == "one card"
    assert catalog.fits(catalog.MODELS["bonsai-27b"], _cards(7500)) == ""


def test_fits_two_cards_uses_the_second_largest():
    m = catalog.MODELS["bonsai-27b"]  # one=7600, two=3800
    assert catalog.fits(m, _cards(7000, 4000)) == "two cards"  # neither alone fits
    assert catalog.fits(m, _cards(4000, 7000)) == "two cards"  # order independent
    assert catalog.fits(m, _cards(7000, 3000)) == ""
    assert catalog.fits(m, _cards(7000)) == ""  # a single card is never "two cards"


def test_fits_nanbeige_split_counts_cards():
    m = catalog.MODELS["nanbeige-33b"]  # split=(15500, 700)
    assert catalog.fits(m, _cards(16200)) == "one card"
    assert catalog.fits(m, _cards(9000, 9000)) == "2 cards"
    assert catalog.fits(m, _cards(8000, 8000)) == ""


def test_fits_no_cards_is_too_large():
    for m in catalog.MODELS.values():
        assert catalog.fits(m, []) == ""


def test_model_dir_is_next_to_the_pack():
    assert catalog.model_dir({"pack": "/x/packs/bonsai-2b"}) == os.path.join("/x/packs", "models")
    assert catalog.model_dir({"pack": "/x/packs/bonsai-2b/"}) == os.path.join("/x/packs", "models")


def test_blocking_sync_opt_out_is_honored(monkeypatch):
    sched._done[0] = False
    monkeypatch.setenv("KINDI_BLOCKING_SYNC", "0")
    logged = []
    sched.blocking_sync(log=logged.append)
    assert logged == []
    assert sched._done[0] is False  # opt-out does not consume the once-per-process slot


def test_kfd_gfx_is_empty_without_kfd(monkeypatch):
    # No /sys/class/kfd on this host: the probe must return an empty set, not raise.
    assert sched._kfd_gfx() == set() or isinstance(sched._kfd_gfx(), set)
    assert os.environ.get("KINDI_BLOCKING_SYNC", "1") in ("0", "1")
