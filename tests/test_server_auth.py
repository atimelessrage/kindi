# SPDX-License-Identifier: AGPL-3.0-only
"""One bearer-token check for every HTTP surface (2026-11-22 audit).

`kindi.server` compared the header in constant time from the start;
`kindi.dsv41_server` compared the same header with `==`. The shared helper is the
fix, and the source check below keeps a future server from copying the plain
comparison back in.
"""
from pathlib import Path

from kindi.api.auth import bearer_matches

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ("server.py", "dsv41_server.py")


def test_accepts_the_exact_bearer_value():
    assert bearer_matches("Bearer secret-token", "secret-token")


def test_rejects_a_missing_wrong_or_prefixed_value():
    assert not bearer_matches("", "secret-token")
    assert not bearer_matches(None, "secret-token")
    assert not bearer_matches("Bearer wrong", "secret-token")
    assert not bearer_matches("secret-token", "secret-token")
    assert not bearer_matches("Bearer secret-token-extra", "secret-token")


def test_an_empty_key_disables_authentication():
    assert bearer_matches("", "")
    assert bearer_matches(None, "")


def test_every_server_uses_the_shared_check_and_not_a_plain_comparison():
    for name in SERVERS:
        text = (ROOT / "kindi" / name).read_text(encoding="utf-8")
        assert "bearer_matches" in text, name
        assert '== "Bearer "' not in text, name
        assert "== 'Bearer '" not in text, name
