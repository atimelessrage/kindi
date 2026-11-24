#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""List every warp shuffle/barrier in ``kindi/*.cu`` and fail on an unreviewed mask.

A warp intrinsic with a mask that names lanes which cannot reach it is undefined behaviour, and it
shows up as a *hang*, not as a wrong number: a kernel that hung on RTX 20/30/40 cards is how the
Inkling 2 Mini engine failed before 0.7.1 (``__shfl_*`` naming lanes that never arrive). This gate
is the check that release documents, kept as a counted baseline so a new site cannot slip in.

Every intrinsic is classified:

``full-mask``
    ``__shfl*_sync(0xffffffffu, ...)``, ``__ballot_sync(0xffffffffu, ...)``, ``__syncwarp()``,
    ``__syncwarp(0xffffffffu)`` — every lane of the warp participates, which is the safe form when the
    call is outside divergent control flow.
``activemask``
    the mask is ``__activemask()`` — the lanes that actually converged; correct when the participating
    set is stable across the intrinsic (a reduction loop inside one predicate).
``compat-macro``
    a use of a file-local compatibility macro (``KSHFLX``, ``KSHFLD``, ...). The macro carries the
    split itself (``#if __CUDA_ARCH__ < 700`` or ``#if KHIP``), so the CUDA build always gets the
    ``_sync`` form and the HIP build the legacy one.
``legacy-hip``
    a pre-Volta intrinsic inside the HIP port block of a file that compiles for both HIP and CUDA
    (``#ifndef KHIP`` ... ``#if KHIP``) or in a HIP-only file: HIP has no ``_sync`` variants, so the
    legacy form is the correct one there.
``partial-mask``
    a literal that is not the full mask (``0x000000FFu``, ``0x0000FFFFu``): a deliberate
    sub-warp group. Each one must be listed in ``WARP_MASK_BASELINE`` with a reason.
``legacy-unguarded``
    a pre-Volta intrinsic in a CUDA-only file with no guard and no compat macro: on sm_70+ nvcc
    converts it to the full-mask ``_sync`` form, so a call under divergence is undefined behaviour
    (a hang, not a wrong number). Each one must be listed with a reason, and the list is expected to
    stay empty.

Run it as ``python tools/check_warp_masks.py``; ``--verbose`` prints the per-file split and
``--update --reason "..."`` records the current sites after review (it refuses to add a
``legacy-unguarded`` row with no reason, so it cannot launder the class of bug it exists for).
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "warp_mask_baseline.json"

INTRINSIC = re.compile(r"\b(__shfl\w*|__ballot\w*|__any\w*|__all\w*|__syncwarp|__activemask)\s*\(")
FULL_MASK = re.compile(r"^(?:0[xX][fF]{8}[uU]?|~0[uU]?)$")
# the file-local compatibility macros that carry the __CUDA_ARCH__ split themselves
COMPAT_MACROS = ("KSHFLX", "KSHFL", "SHFLX", "SHFL")
MACRO_DEF = re.compile(r"^#\s*define\s+(KSHFLX|KSHFL|SHFLX|SHFL)\s*\(")
ARCH_GUARD = re.compile(r"^\s*#\s*(?:if|elif)\b.*__CUDA_ARCH__\s*<\s*700")
HIP_GUARD = re.compile(r"^\s*#\s*(?:if|elif)\b.*\bKHIP\b")
HIP_MARK = re.compile(r"^\s*#\s*(?:ifndef\s+KHIP|if\s+defined\(__HIP|if\s+KHIP\b)")


def _strip_comment(line):
    """Drop a trailing // comment and a leading block-comment line, without touching strings."""
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
    cut = line.find("//")
    return line if cut < 0 else line[:cut]


def _guarded_at(lines, index):
    """True when this line sits in a ``#if __CUDA_ARCH__ < 700`` branch.

    Walks backwards to the nearest conditional that has not been closed, which is enough for the
    flat ``#if/#else/#endif`` form these kernels use. A macro definition line is not a use site.
    """
    depth = 0
    for back in range(index - 1, -1, -1):
        text = lines[back].strip()
        if text.startswith("#endif"):
            depth += 1
        elif text.startswith(("#if", "#ifdef", "#ifndef")):
            if depth == 0:
                return bool(ARCH_GUARD.match(lines[back]))
            depth -= 1
    return False


def _has_port_split(lines):
    """True when this file compiles the same kernel for HIP and CUDA (a ``KHIP`` port block).

    Such a file keeps the legacy intrinsics in its HIP branch on purpose; the CUDA branch uses the
    compat macros. Treating the HIP branch as a defect would make the gate cry wolf on every
    ROCm-capable kernel.
    """
    if any(HIP_MARK.match(line) for line in lines):
        return True
    # a HIP shim: the file defines the _sync name as a macro over the legacy instruction, because
    # HIP has no _sync variants ("#define __shfl_xor_sync(m, v, o) __shfl_xor(v, o)")
    return any(re.match(r"^\s*#\s*define\s+__shfl\w*_sync\s*\(", line) for line in lines)


def _compat_macro_defined(lines, name):
    """A compat macro counts when the file defines it with a pre-Volta or HIP split."""
    for i, line in enumerate(lines):
        m = MACRO_DEF.match(line.strip())
        if m and m.group(1) == name:
            return _guarded_at(lines, i) or _in_hip_branch(lines, i)
    return False


def _in_hip_branch(lines, index):
    """True when this line sits in a ``#if KHIP`` branch or a HIP platform block."""
    depth = 0
    for back in range(index - 1, -1, -1):
        text = lines[back].strip()
        if text.startswith("#endif"):
            depth += 1
        elif text.startswith(("#if", "#ifdef", "#ifndef")):
            if depth == 0:
                return bool(HIP_GUARD.match(lines[back]) or "HIP_PLATFORM" in lines[back]
                            or "__HIPCC_RTC__" in lines[back])
            depth -= 1
    return False


def classify(path, lines, index, match):
    name = match.group(1)
    rest = lines[index][match.end():]
    if name == "__activemask":
        return "activemask", ""
    if name == "__syncwarp":
        arg = rest.split(")")[0].strip()
        return ("full-mask" if not arg or FULL_MASK.match(arg) else "partial-mask"), arg
    if "_sync" not in name:
        if _guarded_at(lines, index):
            return "legacy-guarded", name
        if _in_hip_branch(lines, index) or _has_port_split(lines):
            return "legacy-hip", name
        return "legacy-unguarded", name
    mask = rest.split(",")[0].strip()
    if FULL_MASK.match(mask):
        return "full-mask", mask
    if "activemask" in mask:
        return "activemask", mask
    return "partial-mask", mask


def scan(path):
    """Every warp intrinsic in one .cu file: (line, intrinsic, class, mask)."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    out = []
    for i, raw in enumerate(lines):
        line = _strip_comment(raw)
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        found = []
        for match in INTRINSIC.finditer(line):
            cls, mask = classify(path, lines, i, match)
            found.append((match.group(1), cls, mask))
        for name, cls, mask in found:
            if name == "__activemask" and any(c == "activemask" for _n, c, _m in found if _n != "__activemask"):
                continue                       # the mask argument of that _sync call: one site, not two
            out.append((i + 1, name, cls, mask))
        for macro in COMPAT_MACROS:
            for _m in re.finditer(r"\b%s\s*\(" % macro, line):
                # the macro carries the split itself (defined here or in the shared header)
                out.append((i + 1, macro, "compat-macro", ""))
    return out


def collect(root=None):
    """{relative file: [(line, intrinsic, class, mask), ...]} for every kindi/*.cu."""
    files = {}
    for path in sorted(((root or ROOT) / "kindi").glob("*.cu")):
        sites = scan(path)
        if sites:
            files[path.name] = sites
    return files


def reviewable(files):
    """The sites that need a reviewed reason: partial masks and unguarded legacy intrinsics."""
    return {name: [(line, intrinsic, mask) for line, intrinsic, cls, mask in sites
                   if cls in ("partial-mask", "legacy-unguarded")]
            for name, sites in files.items()}


def counts(files):
    out = {}
    for sites in files.values():
        for _line, _intrinsic, cls, _mask in sites:
            out[cls] = out.get(cls, 0) + 1
    return dict(sorted(out.items()))


def load_baseline(path=BASELINE):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("sites", {})


def compare(files, baseline):
    """(new, unexplained, stale) rows for a baseline check."""
    current = reviewable(files)
    new, unexplained, stale = [], [], []
    for name, sites in current.items():
        have = {(row["line"], row["intrinsic"], row["mask"]): row for row in baseline.get(name, [])}
        for line, intrinsic, mask in sites:
            row = have.get((line, intrinsic, mask))
            if row is None:
                new.append("%s:%d %s(%s)" % (name, line, intrinsic, mask))
            elif not (row.get("reason") or "").strip():
                unexplained.append("%s:%d %s(%s)" % (name, line, intrinsic, mask))
    for name, rows in baseline.items():
        now = {(line, intrinsic, mask) for line, intrinsic, mask in current.get(name, [])}
        for row in rows:
            if (row["line"], row["intrinsic"], row["mask"]) not in now:
                stale.append("%s:%d %s(%s)" % (name, row["line"], row["intrinsic"], row["mask"]))
    return new, unexplained, stale


def build_baseline(files, reasons):
    """The JSON body for --update: one reason per (file, line, intrinsic, mask)."""
    sites = {}
    for name, rows in sorted(reviewable(files).items()):
        if not rows:
            continue
        sites[name] = [{"line": line, "intrinsic": intrinsic, "mask": mask,
                        "reason": reasons.get("%s:%d" % (name, line), "")}
                       for line, intrinsic, mask in rows]
    return {"note": "Reviewed warp intrinsics whose mask is not the full warp. A partial mask names "
                    "a deliberate sub-warp group; a legacy intrinsic needs a __CUDA_ARCH__ guard. "
                    "New or unexplained rows fail tools/check_warp_masks.py.",
            "sites": sites}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_warp_masks.py")
    ap.add_argument("--update", action="store_true", help="record the current sites in the baseline")
    ap.add_argument("--reason", default="", help="why the current sites are safe (required to add one)")
    ap.add_argument("--verbose", action="store_true", help="print the per-file split")
    a = ap.parse_args(argv)
    files = collect()
    if not files:
        print("check_warp_masks: no kindi/*.cu found; run from the repository root", file=sys.stderr)
        return 2
    total = counts(files)
    if a.verbose:
        for name, sites in files.items():
            per = {}
            for _line, _intrinsic, cls, _mask in sites:
                per[cls] = per.get(cls, 0) + 1
            print("%-22s %s" % (name, ", ".join("%s %d" % kv for kv in sorted(per.items()))))
        print("total: %s" % ", ".join("%s %d" % kv for kv in total.items()))
    if a.update:
        if not a.reason.strip():
            print("check_warp_masks: --update needs --reason", file=sys.stderr)
            return 2
        body = build_baseline(files, {})
        for rows in body["sites"].values():
            for row in rows:
                row["reason"] = a.reason.strip()
        BASELINE.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        n = sum(len(rows) for rows in body["sites"].values())
        print("check_warp_masks: recorded %d reviewed sites in %s" % (n, BASELINE.name))
        return 0
    baseline = load_baseline()
    new, unexplained, stale = compare(files, baseline)
    print("check_warp_masks: %d warp intrinsics (%s); %d need a reviewed reason"
          % (sum(total.values()), ", ".join("%s %d" % kv for kv in total.items()),
             sum(len(r) for r in reviewable(files).values())))
    problems = []
    for label, rows in (("new (not in the baseline)", new),
                        ("in the baseline with no reason", unexplained),
                        ("stale (the site moved or went away)", stale)):
        if rows:
            problems.append("%s:\n  %s" % (label, "\n  ".join(rows)))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print("Review each site, then record it with --update --reason \"...\"", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
