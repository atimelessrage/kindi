"""Per-machine calibration cache (roadmap gate 2): the store is CPU-only, so these run without a GPU.

The GPU half (timing in tp.g5_tuned, kindi calibrate) is measured on a card; here the keyed
file, its invalidation, and its tolerance of corrupt state are pinned.
"""
import json
import os

import pytest

from kindi import calib


@pytest.fixture
def cal(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDI_CALIB_DIR", str(tmp_path / "calib"))
    calib._MEMO.clear()
    calib._BUILD_TAG = None
    return tmp_path


def test_gpu_key_is_filename_safe():
    k = calib.gpu_key("NVIDIA GeForce RTX 4060 Ti", "sm89")
    assert "4060" in k and "sm89" in k and " " not in k and "/" not in k
    assert calib.gpu_key("gfx1151", "gfx1151") != k


def test_g5_store_and_lookup_roundtrip(cal):
    calib._MEMO.clear()
    assert calib.lookup_g5("gpu-sm89", "abc123def456", False, 1024, 40, False) is None
    calib.store_g5("gpu-sm89", "abc123def456", False, 1024, 40, False, (512, 8, 4, "r"))
    assert calib.lookup_g5("gpu-sm89", "abc123def456", False, 1024, 40, False) == (512, 8, 4, "r")
    # a different shape, lane, or pack hash is still a miss
    assert calib.lookup_g5("gpu-sm89", "abc123def456", False, 1024, 40, True) is None
    assert calib.lookup_g5("gpu-sm89", "abc123def456", True, 1024, 40, False) is None
    assert calib.lookup_g5("gpu-sm89", "zzz999", False, 1024, 40, False) is None


def test_stored_file_is_valid_json_with_the_key_fields(cal):
    calib.store_g5("gpu-sm89", "abc123def456", True, 512, 16, True, (256, 0, 8, "r"))
    d = json.load(open(calib.path_for("gpu-sm89", "abc123def456"), encoding="utf-8"))
    assert d["kind"] == "kindi-calib/1"
    assert d["build"] == calib.build_tag()
    assert d["gpu"] == "gpu-sm89"
    assert d["g5"]["lane"]["512,16,1"] == [256, 0, 8, "r"]


def test_record_sections_coexist_with_g5(cal):
    calib.store_g5("gpu-sm89", "abc123def456", False, 1024, 40, False, (512, 8, 4, "r"))
    calib.record("gpu-sm89", "abc123def456", "draft_k", {"default": 4, "auto_draft2b": 3})
    calib.record("gpu-sm89", "abc123def456", "placement", {"ctx": 24576, "auto": [0, 1]})
    d = calib.read("gpu-sm89", "abc123def456")
    assert d["draft_k"]["default"] == 4
    assert d["placement"]["auto"] == [0, 1]
    assert d["g5"]["row"]["1024,40,0"] == [512, 8, 4, "r"]   # recording never clobbers the configs


def test_wrong_build_tag_is_a_miss_not_stale_configs(cal):
    path = calib.path_for("gpu-sm89", "abc123def456")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kind": "kindi-calib/1", "build": "deadbeefdeadbeef",
                   "g5": {"lane": {}, "row": {"1024,40,0": [512, 8, 4, "r"]}}}, f)
    assert calib.lookup_g5("gpu-sm89", "abc123def456", False, 1024, 40, False) is None


def test_corrupt_file_is_a_miss_not_an_exception(cal):
    path = calib.path_for("gpu-sm89", "abc123def456")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert calib.lookup_g5("gpu-sm89", "abc123def456", False, 1024, 40, False) is None
    assert calib.read("gpu-sm89", "abc123def456") is None


def test_write_failure_is_best_effort(cal, monkeypatch):
    # a read-only cache dir must never break a start: store swallows the OSError
    d = str(cal / "calib")
    os.makedirs(d, exist_ok=True)
    if os.name != "posix":
        pytest.skip("chmod does not make a dir read-only on Windows")
    monkeypatch.chmod(d, 0o500)
    calib.store_g5("gpu-sm89", "abc123def456", False, 1024, 40, False, (512, 8, 4, "r"))


def test_build_tag_changes_with_the_kernels(cal):
    tag = calib.build_tag()
    assert len(tag) == 16
    calib._BUILD_TAG = None
    assert calib.build_tag() == tag          # stable while nothing changed


def test_calibrate_subcommand_is_wired():
    from kindi.__main__ import main
    with pytest.raises(SystemExit) as e:
        main(["calibrate", "--help"])
    assert e.value.code == 0
