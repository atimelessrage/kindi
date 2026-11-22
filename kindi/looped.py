# SPDX-License-Identifier: AGPL-3.0-only
"""Looped-transformer GGUFs: recognize them and map logical layers to stored blocks.

SenseNova-Loop-9B and Inkling 2 Mini store each looped weight once, but their GGUF
``block_count`` is the *logical* (unrolled) count and the tensor names cover only the
stored blocks. A loader that walks ``range(block_count)`` therefore asks for blocks that
do not exist: SenseNova has no ``blk.28`` (28-51 are the replay of 4-27) and Inkling has
no ``blk.18`` (18-41 replay 6-17). This module is the one definition of that map, shared
by the converter now and by a future loader.

It deliberately has no engine and reads no tensors: a match means Kindi names the
family and refuses it with the real reason, not that Kindi can serve it. The measured
header inventory and the gate ladder are in REQ-0031 and
``docs/benchmarks/2026-11-22-looped-model-headers/``.
"""
from collections import namedtuple

LoopSpec = namedtuple(
    "LoopSpec",
    "arch name stored logical core_start core_end passes post_base detail",
)

# Measured from the public Q4_K_M headers on 2026-11-22 (32 MiB range request, no weights).
# `stored` is the number of block indices the tensor names actually carry; `post_base` is the
# stored index of the first block after the loop (the two files use different conventions:
# SenseNova keeps the logical index 52, Inkling restarts at the physical 18).
_SENSENOVA = LoopSpec(
    "sensenova_loop", "SenseNova-Loop-9B",
    stored=33, logical=57, core_start=4, core_end=28, passes=2, post_base=52,
    detail="two-stream mHC; 3 linear-attention + 1 full-attention layer per four; 512K context",
)
_INKLING = LoopSpec(
    "inkling2", "Inkling 2 Mini",
    stored=24, logical=48, core_start=6, core_end=18, passes=3, post_base=18,
    detail="128-expert MoE (top-8); hybrid local/global attention; 1M context",
)
SPECS = dict((spec.arch, spec) for spec in (_SENSENOVA, _INKLING))


def spec_from_gguf(g):
    """The loop spec for this GGUF, or None.

    A spec is returned only when the file's declared ``block_count`` matches the measured
    logical count, so a different variant is not silently described with the wrong map.
    """
    kv = getattr(g, "kv", None) or {}
    spec = SPECS.get(kv.get("general.architecture"))
    if spec is None:
        return None
    try:
        declared = int(kv.get(spec.arch + ".block_count", spec.logical))
    except (TypeError, ValueError):
        return None
    return spec if declared == spec.logical else None


def core_length(spec):
    return spec.core_end - spec.core_start


def stored_index(spec, layer):
    """The stored block index for one logical decoder layer, or None outside the model."""
    if layer < 0 or layer >= spec.logical:
        return None
    if layer < spec.core_end:
        return layer
    post_start = spec.core_end + core_length(spec) * (spec.passes - 1)
    if layer < post_start:
        return spec.core_start + (layer - spec.core_start) % core_length(spec)
    return spec.post_base + (layer - post_start)


def stored_indices(spec):
    """Every stored block index the logical model visits, in order."""
    return sorted(set(index for index in (stored_index(spec, layer) for layer in range(spec.logical))
                      if index is not None))


def refusal(g):
    """A specific refusal for a recognized looped model, else None."""
    spec = spec_from_gguf(g)
    if spec is None:
        return None
    return (
        "%s (%s) is a recognized looped transformer, but Kindi has no looped-block loader "
        "or engine: %d stored blocks serve %d logical layers (layers %d-%d run %d times). "
        "%s. The measured map and the gate ladder are in REQ-0031; support needs the "
        "physical-to-logical remap, the family's block kernels and a pinned reference before "
        "the model can enter the catalog."
        % (spec.name, spec.arch, spec.stored, spec.logical, spec.core_start,
           spec.core_end - 1, spec.passes, spec.detail)
    )
