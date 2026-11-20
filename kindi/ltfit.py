# SPDX-License-Identifier: AGPL-3.0-only
"""The Lightning expert-fit arithmetic, in one place.

`kindi plan` used to carry its own copy of this (its `_lt_fit`) next to the engine's
`lightning._plan`, which is how the report's slots line drifted from what the engine did. Both call
`fit_experts` now, so the report cannot disagree with the run: the numbers it prints are the numbers
the engine plans with.

The arithmetic: an MoE layer's experts are all resident when they fit. Otherwise the two prompt
staging buffers (`n_exp` experts each) and the `dcache` decode slots of every MoE layer come off the
room first, and what is left divides over the `n_moe - 2` layers that are not staging - which is why
one slot per layer costs `n_moe / (n_moe - 2)` experts per layer, not one.
"""
import math
import os

DCACHE_DEFAULT = 10


def dcache_default(n_used):
    """The engine's decode-slot count per MoE layer: KINDI_LT_DCACHE, never below `n_used`."""
    return max(int(n_used), int(os.environ.get("KINDI_LT_DCACHE", str(DCACHE_DEFAULT))))


def expert_seconds(eb, bytes_s):
    """Time to move one expert at a measured byte rate, or None without the rate."""
    return float(eb) / float(bytes_s) if bytes_s else None


def _max_slots(room, n_moe, n_exp, n_used, eb):
    """Largest decode-slot count that still leaves `n_used` experts per layer resident."""
    if room is None or not n_moe:
        return n_exp
    if fit_experts(room, n_moe, n_exp, n_used, eb, 0) is None:
        return None
    lo, hi = 0, n_exp
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fit_experts(room, n_moe, n_exp, n_used, eb, mid) is None:
            hi = mid - 1
        else:
            lo = mid
    return lo


def rate_slots(n_used, n_exp, eb, rates, room=None, n_moe=None):
    """Predict decode slots per MoE layer from measured transfer rates and the room.

    One token needs `n_used` experts per layer. The lookahead is the number of experts whose
    transfer can be hidden behind one RAM/disk pass at the measured rates: a slot set holds the
    current working set plus `ceil((ram + disk) / pcie)` more. That is a prediction from rates,
    not a measured optimum; the Lightning-pack comparison remains the gate. `room` caps it at
    what still leaves `n_used` residents; None when even that does not fit.
    """
    rates = rates or {}
    pcie = expert_seconds(eb, rates.get("pcie_bytes_s"))
    if not pcie:
        return None
    stage = (expert_seconds(eb, rates.get("ram_bytes_s")) or 0.0) + \
            (expert_seconds(eb, rates.get("disk_bytes_s")) or 0.0)
    depth = 1 + int(math.ceil(stage / pcie)) if stage else 1
    wanted = max(1, int(n_used)) * depth
    wanted = min(int(n_exp), wanted)
    limit = _max_slots(room, n_moe, n_exp, n_used, eb)
    if limit is None:
        return None
    return max(int(n_used), min(wanted, limit))


def fit_experts(room, n_moe, n_exp, n_used, eb, dcache):
    """Experts per MoE layer that fit in `room` bytes; None when even `n_used` per layer do not.

    `room` is what is left for experts after the dense core, K/V, SSM state, buffers and reserve.
    `dcache` is the decode slots per MoE layer; pass 0 for the counterfactual without them. A model
    with two or fewer MoE layers has no staging/rest split to divide, so the all-resident formula is
    the only one applied - not a division by zero.
    """
    if n_moe <= 0 or eb <= 0:
        return None
    fit = max(0, int(room // (n_moe * eb)))
    if fit < n_exp and n_moe > 2:
        fit = max(0, int((room - 2 * n_exp * eb - n_moe * dcache * eb) // ((n_moe - 2) * eb)))
    if fit < n_used:
        return None
    return min(n_exp, fit)
