"""One effective runtime-configuration view for every diagnostic surface.

The 2026-11-20 review found three diagnostics bugs with the same shape: a surface re-derived
engine state and drifted from what was running (the default path named the wrong engine, the HP
line was printed before the tensor-core layout was confirmed, and ``/health`` echoed the requested
speculation mode).  This module is the single resolver those surfaces read.

``describe`` never probes the device, opens a pack or changes state: every effective field is read
from the object that already decided it (the engine, the server's ``Kindi`` wrapper, the config).
Before a model loads it reports the requested values and ``loaded: false`` instead of pretending
an effective state exists.  ``kindi plan`` keeps its own pre-load planner output because no engine
exists there to describe.
"""

import os


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flag(value):
    if value is None:
        return None
    return value.strip().lower() not in ("", "0", "off", "false", "no")


def _preload_spec():
    return {"enabled": False, "method": None,
            "requested": os.environ.get("KINDI_SPEC", "0") or "0",
            "reason": "engine not loaded", "head": None, "head_sha256": None,
            "drafter": None, "last": None}


def _spec_from_engine(engine):
    """The effective speculation state for a caller without a ``Kindi`` wrapper (bench)."""
    requested = os.environ.get("KINDI_SPEC", "0") or "0"
    out = {"enabled": False, "method": None, "requested": requested, "reason": None,
           "head": None, "head_sha256": None, "drafter": None, "last": None}
    if not hasattr(engine, "spec_begin"):
        out["reason"] = "this engine has no speculative path"
        return out
    try:
        from . import spec as spec_mod

        state = spec_mod.effective_state(engine)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not fail the surface that reports it
        out["reason"] = "speculation state unavailable: %s" % exc
        return out
    head = state.get("head") or {}
    out.update(enabled=state.get("mode") is not None, method=state.get("mode"),
               requested=state.get("requested", requested), reason=state.get("reason"),
               head=head.get("file") if head.get("state") == "loaded" else None,
               head_sha256=(head.get("sha256") or "")[:12] or None)
    return out


def _head(cfg, engine, kindi):
    """The head that runs: the same rule as ``Kindi.head_active`` when no wrapper is available."""
    if kindi is not None and hasattr(kindi, "head_active"):
        return kindi.head_active()
    if engine is None:                     # nothing has run yet: the request is reported separately
        return None
    wanted = str(cfg.get("head", "exact") or "exact")
    if (wanted == "fast" and engine is not None
            and getattr(engine, "hsl", None) is None
            and getattr(engine, "fast_centroids", None) is None):
        return "exact"
    return wanted


def describe(cfg, engine=None, kindi=None):
    """The effective runtime state; ``engine`` is the built engine, ``kindi`` the server wrapper.

    Fields are ``None`` when they cannot be known (before a load, or on an engine that does not
    expose them).  ``spec`` has the same shape as ``Kindi.drafting()`` so ``/health`` can publish
    it unchanged; ``head`` is the effective head, not the requested one.
    """
    loaded = engine is not None
    if kindi is not None and hasattr(kindi, "drafting"):
        spec = kindi.drafting()
    elif loaded:
        spec = _spec_from_engine(engine)
    else:
        spec = _preload_spec()
    chunk = getattr(engine, "chunk", None) if loaded else None
    tc = bool(getattr(engine, "tc", False)) if loaded else None
    ctx_active = _int(getattr(engine, "ctx", None)) if loaded else None
    ctx_requested = _int(cfg.get("ctx"))
    if ctx_requested is None:
        # a minimal config (tests, an embedder) may omit ctx; the engine's own value is then the
        # only context this process knows, and reporting it beats reporting a dash.
        ctx_requested = ctx_active
    return {
        "loaded": loaded,
        "engine": type(engine).__name__ if loaded else None,
        "variant": getattr(kindi, "variant", None) if kindi is not None else None,
        # no local pack path here: /health is a network surface, and pack_id is enough to identify it.
        "pack_id": getattr(kindi, "pack_id", None) if kindi is not None else None,
        "ctx_requested": ctx_requested,
        "ctx_active": ctx_active,
        "tensor_cores": tc,
        # make_engine writes the plan's HP decision into KINDI_TC_HP before kindi.spec is imported;
        # a non-tensor-core engine has no HP path at all, so that is a no, not an unknown.
        "hp": _flag(os.environ.get("KINDI_TC_HP")) if tc else (False if loaded else None),
        "kv_cache": getattr(engine, "kv", None) if loaded else None,
        "chunk": chunk.get("T") if isinstance(chunk, dict) and chunk else None,
        "chunk_requested": os.environ.get("KINDI_CHUNK") or None,
        "threads": _int(getattr(engine, "threads", None)) if loaded else None,
        "head_requested": str(cfg.get("head", "exact") or "exact"),
        "head": _head(cfg, engine, kindi),
        "spec": spec,
        "vram_budget_mb": _int(cfg.get("tandem_vram_mb")),
        "vram_reserve_mb": _int(cfg.get("tandem_reserve_mb")),
    }


def _word(value):
    return "unknown" if value is None else ("yes" if value else "no")


def render(state):
    """One log line for the resolver's state, used by the startup log and ``kindi info``."""
    if not state["loaded"]:
        chunk = state["chunk_requested"] or "auto"
        return ("runtime: not loaded; ctx requested=%s; head requested=%s; spec requested=%s; "
                "chunk=%s; vram ceiling=%s MiB, reserve=%s MiB"
                % (state["ctx_requested"], state["head_requested"], state["spec"]["requested"],
                   chunk, state["vram_budget_mb"], state["vram_reserve_mb"]))
    spec = state["spec"] or {}
    return ("runtime: engine=%s%s tc=%s hp=%s kv=%s ctx=%s/%s spec=%s head=%s chunk=%s "
            "threads=%s vram=%s/%s MiB"
            % (state["engine"], " (%s)" % state["variant"] if state["variant"] else "",
               _word(state["tensor_cores"]), _word(state["hp"]),
               state["kv_cache"] or "unknown", state["ctx_active"], state["ctx_requested"],
               spec.get("method") or "off", state["head"], state["chunk"] or "off",
               state["threads"] if state["threads"] is not None else "n/a",
               state["vram_budget_mb"], state["vram_reserve_mb"]))
