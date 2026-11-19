# SPDX-License-Identifier: AGPL-3.0-only
"""The MTP smoke's post-request mode guard and run record, without a GPU or a real server.

`tools/mtp_smoke.py` checks `drafting.last.method` after its completion because the
pre-request `/health` block can say `mtp` while the request itself used another
drafter, and it records the run's provenance so a future claim does not need a
"not captured" caveat. These pin both halves.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "mtp_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("mtp_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_drafting_last_whose_method_is_not_mtp_is_rejected():
    smoke = _load_smoke()
    last = {"method": "ngram", "passes": 2, "drafted": 8, "accepted": 3}
    with pytest.raises(smoke.SmokeError) as caught:
        smoke._check_last_method(last)
    assert caught.value.code == 1
    message = str(caught.value)
    assert "drafting.last.method" in message
    assert "'ngram'" in message
    assert "not mtp" in message


def test_a_drafting_last_with_no_method_field_is_rejected():
    smoke = _load_smoke()
    with pytest.raises(smoke.SmokeError) as caught:
        smoke._check_last_method({})
    assert caught.value.code == 1
    message = str(caught.value)
    assert "drafting.last.method is None" in message
    assert "not mtp" in message


def test_a_null_drafting_last_is_rejected_without_an_attribute_error():
    smoke = _load_smoke()
    with pytest.raises(smoke.SmokeError) as caught:
        smoke._check_last_method(None)
    assert caught.value.code == 1
    message = str(caught.value)
    assert "drafting.last is missing or null" in message
    assert "expected an MTP result" in message


def test_a_drafting_last_whose_method_is_mtp_is_accepted():
    smoke = _load_smoke()
    last = {"method": "mtp", "passes": 2, "drafted": 8, "accepted": 3}
    assert smoke._check_last_method(last) is None


def _record(smoke):
    return smoke._run_record(
        "/packs/bonsai 2b", {"model": "Ternary-Bonsai-2-2B"},
        "/heads/kindi-mtp-2b-v1.npz",
        {"sha256": "a" * 64, "bytes": 61000000}, 4096, 4, "RTX 4060 Ti (sm_89)",
        ["python", "-m", "kindi", "--config", "/tmp/x y/kindi.conf", "serve",
         "--host", "127.0.0.1", "--port", "8093"],
        {"KINDI_PACK": "/packs/bonsai 2b", "KINDI_SPEC": "mtp", "KINDI_SPEC_K": "4",
         "HOME": "/srv/kindi"}, 1)


def test_the_run_record_names_the_context_spec_k_pack_head_command_and_requests():
    smoke = _load_smoke()
    text = "\n".join(smoke._record_lines(_record(smoke)))
    assert "ctx: 4096" in text
    assert "KINDI_SPEC_K: 4" in text
    assert "pack: /packs/bonsai 2b (Ternary-Bonsai-2-2B)" in text
    assert "head: /heads/kindi-mtp-2b-v1.npz sha256 %s" % ("a" * 64) in text
    assert "requests: 1" in text
    assert "serve command: KINDI_PACK='/packs/bonsai 2b'" in text
    assert "KINDI_SPEC=mtp KINDI_SPEC_K=4" in text      # the run's environment, not just argv
    assert "-m kindi --config '/tmp/x y/kindi.conf' serve" in text
    assert "--port 8093" in text
    assert "HOME=" not in text                          # only KINDI_* belongs in the record


def test_the_log_header_carries_the_same_run_record(tmp_path):
    smoke = _load_smoke()
    path = tmp_path / "server.log"
    path.write_text(smoke._record_header(_record(smoke)), encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("server line\n")                    # the child's output follows the header
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# mtp_smoke run record\n")
    assert text.rstrip().endswith("server line")
    for line in ("# ctx: 4096", "# KINDI_SPEC_K: 4", "# requests: 1", "# serve command: "):
        assert line in text
    assert "# head: /heads/kindi-mtp-2b-v1.npz" in text
