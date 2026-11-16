# SPDX-License-Identifier: AGPL-3.0-only
"""One resolver for "which registry model id does this pack manifest claim".

The ids live with their adapters (``MODEL_ID`` plus a ``model_id_for_manifest`` hook in
``kindi/backends/*.py``). The config's ``model_name: auto`` inference and the engine gateway both
resolve through here, so a family cannot be added to one and forgotten in the other — that split is
what made a Nanbeige pack report ``bonsai2-33b`` to clients, and a DeepSeek pack likewise
(2026-11-16 audit, F3).

The adapter modules are imported on first use, not at import time: ``config`` is loaded by every
entry point, including ones that never build an engine.
"""
import importlib

# A manifest two adapters claim is contradictory, so every hook is asked rather than stopping at the
# first claim: resolution must be able to refuse, not pick by import order.
_ADAPTERS = ("bonsai", "nanbeige", "lightning", "gemma4", "dsv41")


def adapter_model_id(man):
    """The registry model id a backend adapter claims for this pack manifest, or None.

    ``None`` means no adapter claims the pack (a foreign or hand-edited manifest); the caller falls
    back to its own heuristic rather than inventing a family.
    """
    if not isinstance(man, dict) or not man:
        return None
    claims = []
    for name in _ADAPTERS:
        module = importlib.import_module(".backends." + name, __package__)
        mid = module.model_id_for_manifest(man)
        if mid:
            claims.append(mid)
    if len(claims) > 1:
        raise SystemExit("kindi: the pack manifest is claimed by more than one backend (%s): its "
                         "`model` and `backend` fields contradict each other" % ", ".join(claims))
    return claims[0] if claims else None
