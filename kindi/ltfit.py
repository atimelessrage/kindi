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
import os

DCACHE_DEFAULT = 10


def dcache_default(n_used):
    """The engine's decode-slot count per MoE layer: KINDI_LT_DCACHE, never below `n_used`."""
    return max(int(n_used), int(os.environ.get("KINDI_LT_DCACHE", str(DCACHE_DEFAULT))))


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
