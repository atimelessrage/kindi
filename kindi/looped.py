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
import re
from collections import namedtuple

LoopSpec = namedtuple(
    "LoopSpec",
    "arch name stored logical core_start core_end passes post_base detail",
)
LayerStep = namedtuple("LayerStep", "logical stored phase pass_index")
LoopReport = namedtuple("LoopReport", "spec present missing unexpected plan")
_BLOCK_TENSOR = re.compile(r"^blk\.(\d+)\.")

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
# The GGUF's declared architecture is "inkling2_loop" (kindi/inkling2.py's Spec and
# inkling2_cli.is_inkling both require exactly that); "inkling2" is the name the
# 2026-11-22 header inventory recorded. Both keys map to the same measured map so
# `kindi loop-info` recognizes a real file instead of falling through to the generic error.
_INKLING_LOOP = LoopSpec(
    "inkling2_loop", "Inkling 2 Mini",
    stored=24, logical=48, core_start=6, core_end=18, passes=3, post_base=18,
    detail="128-expert MoE (top-8); hybrid local/global attention; 1M context",
)
SPECS = dict((spec.arch, spec) for spec in (_SENSENOVA, _INKLING, _INKLING_LOOP))


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


def _declared_block_count(kv, spec):
    try:
        return int(kv.get(spec.arch + ".block_count", spec.logical))
    except (TypeError, ValueError):
        return None


def mismatch(g):
    """A specific refusal when a recognized family declares a different block count.

    The measured map is only valid for the declared logical count.  Returning None here
    means the file is either unrecognized or matches its family's measured count.
    """
    kv = getattr(g, "kv", None) or {}
    spec = SPECS.get(kv.get("general.architecture"))
    if spec is None:
        return None
    declared = _declared_block_count(kv, spec)
    if declared == spec.logical:
        return None
    shown = "no valid block_count" if declared is None else "block_count %d" % declared
    return (
        "%s (%s) declares %s, but the measured loop map is for %d logical layers. "
        "Refusing rather than applying another variant's map; a different variant needs its "
        "own measured map (REQ-0031)."
        % (spec.name, spec.arch, shown, spec.logical)
    )


def execution_plan(spec):
    """The logical layer order a loader must execute, with the stored block for each."""
    core = core_length(spec)
    if core <= 0:
        raise ValueError("a loop spec needs a non-empty core")
    post_start = spec.core_end + core * (spec.passes - 1)
    steps = []
    for logical in range(spec.logical):
        stored = stored_index(spec, logical)
        if logical < spec.core_start:
            phase, pass_index = "pre", 0
        elif logical < post_start:
            phase, pass_index = "loop", (logical - spec.core_start) // core + 1
        else:
            phase, pass_index = "post", 0
        steps.append(LayerStep(logical, stored, phase, pass_index))
    return tuple(steps)


def stored_blocks(g):
    """The block indices present in the GGUF tensor names, regardless of tensor data."""
    blocks = set()
    for tensor in getattr(g, "tensors", ()) or ():
        name = tensor[0] if isinstance(tensor, (tuple, list)) and tensor else getattr(tensor, "name", None)
        if not isinstance(name, str):
            continue
        match = _BLOCK_TENSOR.match(name)
        if match:
            blocks.add(int(match.group(1)))
    return frozenset(blocks)


def layout(g):
    """The map plus the file's stored-block coverage, or None for an unrecognized file."""
    spec = spec_from_gguf(g)
    if spec is None:
        return None
    present = stored_blocks(g)
    expected = frozenset(stored_indices(spec))
    return LoopReport(
        spec=spec,
        present=tuple(sorted(present)),
        missing=tuple(sorted(expected - present)),
        unexpected=tuple(sorted(present - expected)),
        plan=execution_plan(spec),
    )


def format_ranges(values):
    """Compact `0-3,7,9-10` formatting for layer and block lists."""
    values = sorted(set(int(v) for v in values))
    if not values:
        return "none"
    parts = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else "%d-%d" % (start, previous))
        start = previous = value
    parts.append(str(start) if start == previous else "%d-%d" % (start, previous))
    return ",".join(parts)


def report_dict(report):
    """A JSON-ready view of a looped layout."""
    spec = report.spec
    return {
        "arch": spec.arch,
        "name": spec.name,
        "stored": spec.stored,
        "logical": spec.logical,
        "core_start": spec.core_start,
        "core_end": spec.core_end,
        "passes": spec.passes,
        "post_base": spec.post_base,
        "detail": spec.detail,
        "present_blocks": list(report.present),
        "missing_blocks": list(report.missing),
        "unexpected_blocks": list(report.unexpected),
        "plan": [
            {"logical": step.logical, "stored": step.stored, "phase": step.phase,
             "pass": step.pass_index}
            for step in report.plan
        ],
    }


def _plan_ranges(plan):
    groups = []
    for step in plan:
        key = (step.phase, step.pass_index)
        if groups and groups[-1][0] == key and step.logical == groups[-1][2] + 1:
            groups[-1] = (key, groups[-1][1], step.logical)
        else:
            groups.append((key, step.logical, step.logical))
    out = []
    for (phase, pass_index), first, last in groups:
        label = phase if phase != "loop" else "loop pass %d" % pass_index
        out.append("%s %d-%d" % (label, first, last))
    return "; ".join(out)


def render(report):
    """Human-readable lines for `kindi loop-info`."""
    spec = report.spec
    expected_blocks = set(stored_indices(spec))
    expected = len(expected_blocks)
    present = len(set(report.present) & expected_blocks)
    lines = [
        "%s (%s)" % (spec.name, spec.arch),
        "  stored blocks: %d; logical layers: %d; loop: %d-%d x%d"
        % (spec.stored, spec.logical, spec.core_start, spec.core_end - 1, spec.passes),
        "  post-loop stored at %d; tensor coverage: %d/%d stored blocks present"
        % (spec.post_base, present, expected),
        "  plan: " + _plan_ranges(report.plan),
    ]
    if report.missing:
        lines.append("  missing stored blocks: " + format_ranges(report.missing))
    if report.unexpected:
        lines.append("  unexpected block tensors: " + format_ranges(report.unexpected))
    lines.append("  no looped loader or engine is implemented (REQ-0031); this is a map check.")
    return lines


def refusal(g):
    """A specific refusal for a recognized looped model, else None."""
    changed = mismatch(g)
    if changed:
        return changed
    spec = spec_from_gguf(g)
    if spec is None:
        return None
    report = layout(g)
    extra = ""
    if report is not None and report.present and report.missing:
        extra += " Its tensor index is missing stored blocks %s." % format_ranges(report.missing)
    if report is not None and report.unexpected:
        extra += " Unexpected block tensors %s are outside the measured map." % format_ranges(report.unexpected)
    return (
        "%s (%s) is a recognized looped transformer, but Kindi has no looped-block loader "
        "or engine: %d stored blocks serve %d logical layers (layers %d-%d run %d times). "
        "%s. The measured map and the gate ladder are in REQ-0031; support needs the "
        "physical-to-logical remap, the family's block kernels and a pinned reference before "
        "the model can enter the catalog. Inspect the map with `kindi loop-info FILE.gguf`.%s"
        % (spec.name, spec.arch, spec.stored, spec.logical, spec.core_start,
           spec.core_end - 1, spec.passes, spec.detail, extra)
    )
