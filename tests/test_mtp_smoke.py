# SPDX-License-Identifier: AGPL-3.0-only
"""The MTP smoke's post-request mode guard, without a GPU or a real server.

`tools/mtp_smoke.py` checks `drafting.last.method` after its completion because the
pre-request `/health` block can say `mtp` while the request itself used another
drafter. This pins the rejection half of that check.
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
