# SPDX-License-Identifier: AGPL-3.0-only
"""One bearer-token check for every Kindi HTTP surface.

The OpenAI server (``kindi.server``) and the DeepSeek server
(``kindi.dsv41_server``) must answer the same question the same way: does this
request carry the configured key, compared in constant time? Keeping the answer
here means a new server cannot quietly copy the check with ``==`` (the 2026-11-22
audit found exactly that in the DeepSeek handler).

An empty key means "no authentication", the loopback default. ``header`` is the
raw ``Authorization`` value; both server stacks look it up case-insensitively,
and ``None`` and ``""`` both mean "no header".

This module has no FastAPI or HTTP-server dependency so it can be imported and
unit-tested on a CPU-only host and on Python 3.6.
"""
import hmac


def bearer_matches(header, key):
    """True when ``header`` is ``Bearer <key>``, or when no key is configured."""
    if not key:
        return True
    supplied = (header or "").encode("utf-8")
    return hmac.compare_digest(supplied, ("Bearer " + key).encode("utf-8"))
