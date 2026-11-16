"""Compatibility labels for small MoE GGUFs that Kindi does not execute.

This module is deliberately not an engine: it only names an architecture and the
next project milestone that would be needed before the model can be converted,
catalogued, or installed. A match here means Kindi recognizes the file well enough
to refuse it with a specific reason, not that Kindi can serve it.
"""
from __future__ import annotations

_SMALL_MOE = {
    "qwen3moe": (
        "Qwen3-Coder-30B-A3B",
        "Kindi has no qwen3moe converter or engine. That is a different architecture from "
        "the dense qwen35 Bonsai backend; support needs a Qwen3-MoE loader plus a pinned "
        "llama.cpp reference before it can enter the catalog.",
    ),
    "qwen35moe": (
        "Qwen3.6-35B-A3B",
        "Kindi has no qwen35moe converter or engine. It is not the dense qwen35 Ternary-Bonsai "
        "backend: support needs the DeltaNet/hybrid attention loader, the MoE router and "
        "expert path, and a pinned llama.cpp reference.",
    ),
    "cohere2moe": (
        "North Mini Code 1.0",
        "Kindi has no cohere2moe converter or engine. Support needs a Cohere2-MoE loader, "
        "its chat template and tokenizer checks, and a pinned llama.cpp reference.",
    ),
    "glm4moelite": (
        "GLM-4.7-Flash",
        "Kindi has no glm4moelite converter or engine. Support needs a GLM4-MoE-Lite loader, "
        "the MLA/KV attention path, the routed-plus-shared expert path, and a pinned "
        "llama.cpp reference.",
    ),
    "flint": (
        "Recursive Flint 40B/2.5B",
        "Kindi has no flint converter or engine. Support needs that architecture's loader, "
        "router and expert path, and a pinned llama.cpp reference.",
    ),
}


def _name(g):
    kv = getattr(g, "kv", None) or {}
    return kv.get("general.name") or kv.get("general.basename") or ""


def _arch(g):
    kv = getattr(g, "kv", None) or {}
    return kv.get("general.architecture")


def _tensor_names(g):
    return {str(t[0]) for t in getattr(g, "tensors", ()) if t}


def _tensor_types(g):
    return {int(t[2]) for t in getattr(g, "tensors", ()) if len(t) > 2}


def unsupported_reason(g):
    """Return the compatibility refusal for a recognized small MoE file, else None."""
    arch = _arch(g)
    name = _name(g)

    if arch in _SMALL_MOE:
        model, why = _SMALL_MOE[arch]
        return "%s (%s) is a recognized small MoE, but Kindi does not have a native backend for it: %s" % (
            model,
            arch,
            why,
        )

    if arch == "gemma4":
        names = _tensor_names(g)
        kv = getattr(g, "kv", None) or {}
        is_moe = any("ffn_gate_up_exps" in n or "ffn_down_exps" in n for n in names)
        is_26b = int(kv.get("gemma4.block_count", 0) or 0) != 42 or int(
            kv.get("gemma4.embedding_length", 0) or 0
        ) != 2560
        if is_moe or is_26b:
            return (
                "Gemma 4 26B-A4B is a recognized gemma4 MoE, but the current Kindi Gemma 4 "
                "backend is E4B-only (42 layers, width 2560) and has no router or expert path. "
                "The design contract is docs/GEMMA4-26B-MOE.md; support needs a separate 26B "
                "backend and a pinned reference before it can enter the catalog."
            )

    return None
