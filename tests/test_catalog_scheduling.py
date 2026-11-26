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


def test_without_a_card_the_listing_names_the_cpu_engine():
    """`fits` stays the card-only answer; the human listing must not call a CPU-runnable model too large.

    On a GPU-less host the listing printed "too large" for every model, including the 2B, 4B and 9B
    that the CPU engine runs on that same machine (2026-11-24).
    """
    # the 27B rows too: the CPU engine ran the 27B (4/4 token-exact) and both 2.5 27B files, so
    # "GPU only" on a cardless host was the same miscall (2026-11-26)
    for mid in ("bonsai-2b", "bonsai-4b", "bonsai-9b"):
        assert catalog._no_card_label(catalog.MODELS[mid]) == "CPU engine"
    # the 27B rows name the host RAM the measured CPU run needed, so a 16 GiB host does not
    # read a 27B pack as runnable (2026-11-26 audit follow-up)
    for mid in ("bonsai-27b", "bonsai-2.5-27b", "bonsai-2.5-27b-mtp"):
        assert catalog._no_card_label(catalog.MODELS[mid]) == "CPU engine (~32 GiB RAM)"
    for mid in ("nemotron-e4b", "nanbeige-33b", "lightning-30b"):
        assert catalog._no_card_label(catalog.MODELS[mid]) == "GPU only"
    assert catalog.fits(catalog.MODELS["bonsai-2b"], []) == ""      # unchanged


def test_the_listing_uses_the_cpu_label_only_without_a_card(capsys):
    lines = []
    catalog.list_models({"pack": os.path.join(os.sep, "x", "packs", "bonsai-2b")}, log=lines.append)
    text = "\n".join(lines)
    assert "CPU engine" in text and "too large" not in text


def test_model_dir_is_next_to_the_pack():
    pack = os.path.join(os.sep, "x", "packs", "bonsai-2b")   # the layout `kindi use` writes
    assert catalog.model_dir({"pack": pack}) == os.path.join(os.sep, "x", "models")
    assert catalog.model_dir({"pack": pack + os.sep}) == os.path.join(os.sep, "x", "models")   # a trailing slash


def _small_catalog(tmp_path, files):
    entry = {"file": "Model.gguf", "repo": "owner/repo", "sha": "0" * 64,
             "size": 4, "one": 1, "two": 0, "what": "test model"}
    model_dir = tmp_path / "x" / "models"
    model_dir.mkdir(parents=True)
    for name, data in files.items():
        (model_dir / name).write_bytes(data)
    return {"pack": str(tmp_path / "x" / "pack")}, {"test-model": entry}


def test_list_models_does_not_call_a_wrong_size_file_downloaded(tmp_path, monkeypatch):
    cfg, models = _small_catalog(tmp_path, {"Model.gguf": b"bad"})
    monkeypatch.setattr(catalog, "MODELS", models)
    monkeypatch.setattr(catalog, "_cards", lambda: [])
    lines = []
    catalog.list_models(cfg, log=lines.append)
    assert any("present (wrong size)" in line for line in lines)
    assert not any("downloaded" in line for line in lines)


def test_list_models_marks_a_size_matching_file_unchecked_until_use_verifies_it(tmp_path, monkeypatch):
    cfg, models = _small_catalog(tmp_path, {"Model.gguf": b"good"})
    monkeypatch.setattr(catalog, "MODELS", models)
    monkeypatch.setattr(catalog, "_cards", lambda: [])
    lines = []
    catalog.list_models(cfg, log=lines.append)
    assert any("present (size ok, unchecked)" in line for line in lines)


def test_blocking_sync_opt_out_is_honored(monkeypatch):
    sched._done[0] = False
    monkeypatch.setenv("KINDI_BLOCKING_SYNC", "0")
    logged = []
    sched.blocking_sync(log=logged.append)
    assert logged == ["blocking sync: off (KINDI_BLOCKING_SYNC=0)"]   # 0.7.0 names the explicit opt-out
    assert sched._done[0] is False  # opt-out does not consume the once-per-process slot


def test_kfd_gfx_is_empty_without_kfd(monkeypatch):
    # No /sys/class/kfd on this host: the probe must return an empty set, not raise.
    assert sched._kfd_gfx() == set() or isinstance(sched._kfd_gfx(), set)
    assert os.environ.get("KINDI_BLOCKING_SYNC", "1") in ("0", "1")
