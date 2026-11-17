"""Every graph call site passes exactly the arguments its CUDA kernel declares.

A kernel can gain a parameter that its Python callers never learn about: `k_rmsfold_qT` took a trailing
`hp` from 0.5.0, `spec.py` and `dspine.py` were updated and `mtp.py` was not, so the MTP head's CUDA
graph did not build (`cuGraphAddKernelNode failed (1)`) and the engine logged "speculation off" - a
documented feature silently dead for four releases (fixed in 0.6.1, ported here). A short argument
tuple is not a Python error, so nothing caught it.

This test is the launch-shape contract for the four modules that build graphs (`spec.py`,
`dspine.py`, `mtp.py`, `tp.py`): it reads each kernel's parameter count from its `.cu` declaration and
each call site's argument tuple from the Python source. Call sites it cannot resolve statically are
**listed and pinned** (`UNRESOLVED_SITES`) rather than skipped, so a new dynamic launch has to be
classified by a person instead of disappearing from the check.

`test_every_checked_site_is_detected_when_mutated` proves each check is real: for every measured call
site it writes a copy of the module with that site's argument tuple shortened and requires the checker
to report exactly that mismatch.
"""
import ast
import collections
import functools
import os
import pathlib
import re
import tempfile

import pytest

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the four modules that build CUDA graphs; a launch here is Plan.add(kernel, grid, block, args) or
# E(kernel, grid, block, args)
MODULES = ("kindi/spec.py", "kindi/dspine.py", "kindi/mtp.py", "kindi/tp.py")

# Call sites the parser cannot resolve statically, pinned by (module, expression, reason) so that a
# new one fails the test until somebody records why it is out of reach. Keep this list honest: if a
# site becomes resolvable, delete its row here and let the checker cover it.
UNRESOLVED_SITES = [
    # spec.py
    ('kindi/spec.py', 'kern', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'mv[nw]', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'mv3[nw]', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', '{512: SK.k_rmsfold2_512, 1024: SK.k_rmsfold2_1024}[HB] if...', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'mv[nw]', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'smf', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'getattr(SK, vver)', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'smf', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'SK.mv3[nw]', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'getattr(SK, "k_g5t%d_%d" % (U, T))', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'fdk', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'getattr(K, vdec)', 'kernel expression is not a static k_* attribute'),
    ('kindi/spec.py', 'SK.mv[nw]', 'kernel expression is not a static k_* attribute'),
    # dspine.py
    ('kindi/dspine.py', 'self.k', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'self.k8 if n <= 8 else self.k16', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'fn if hasattr(fn, "kernel") else _Fk(fn)', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'self.hmv', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_rms_rows"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_rpos_set"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_rms_rows"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_gather_tok"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_top16a"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_top16b"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_tail_base"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_gather_sid"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_gather_h"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_tail_sc"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_chain"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_headnorm_rope"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_ring_mod"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_ring_mod"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_headnorm_rope"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_headnorm_rope"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_ring_store"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_ring_store"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_att_part"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_att_comb"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_rms_rows"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_silu_mul"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_bridge_a"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_h16mm8"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_bridge_codes"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_h16mm8"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_bridge_msg"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_h16mm8"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_bridge_out"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_rms_rows"]', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'kf["k_h16mm16"]', 'kernel expression is not a static k_* attribute'),
    # mtp.py
    ('kindi/mtp.py', 'SK.mv[nw]', 'kernel expression is not a static k_* attribute'),
    ('kindi/mtp.py', 'SK.mv[nw]', 'kernel expression is not a static k_* attribute'),
    # tp.py
    ('kindi/tp.py', 'tuple(args) + (x, a[0][3])', 'argument count is not a literal tuple'),
    ('kindi/tp.py', 'tuple(args) + (x, a[0][3])', 'argument count is not a literal tuple'),
    ('kindi/tp.py', 'store_kv', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'attn_s', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'attn_v', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'tuple(args) + (x, a[0][3])', 'argument count is not a literal tuple'),
    ('kindi/tp.py', 'store_kv', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'K.k_attn_sq8d if s.dp4a else K.k_attn_sq8t', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'attn_v', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'kern', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'kern', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'tuple(args) + (x, a[0][3])', 'argument count is not a literal tuple'),
    ('kindi/tp.py', 'store_kv', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'fs', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'fs', 'kernel expression is not a static k_* attribute'),
    ('kindi/tp.py', 'f', 'kernel expression is not a static k_* attribute'),
]



@functools.lru_cache(maxsize=1)
def _kernel_arity():
    """name -> set of declared parameter counts, over every `__global__ void [__launch_bounds__(N)] k_*(...)`.

    A name can be declared more than once with different counts: `k_tc_q8` takes `hp` in `tc.cu` and
    not in the HIP/Pascal variants, and `spec.py` adds it behind a runtime condition, so the contract
    is the declared *set* rather than a single number.
    """
    decls = {}
    for source in sorted(list((ROOT / "kindi").glob("*.cu")) + list((ROOT / "kindi").glob("*.cuh"))):
        text = source.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'__global__\s+void\s+(?:__launch_bounds__\s*\(\s*\d+\s*\)\s+)?'
                             r'(k_[A-Za-z0-9_]+)\s*\(', text):
            i, depth, buf, params = m.end(), 0, [], []
            while True:
                ch = text[i]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                if ch == "," and depth == 0:
                    params.append("".join(buf))
                    buf = []
                else:
                    buf.append(ch)
                i += 1
            params.append("".join(buf))
            decls.setdefault(m.group(1), []).append([p.strip() for p in params if p.strip()])
    out = {name: {len(p) for p in plists} for name, plists in decls.items()}
    # `kindi/chunk.py::_wrappers` generates token-batched copies as `name_T(params, int T_, one stride
    # per pointer parameter)`, so those are declared too, just not in a .cu file.
    for name, plists in decls.items():
        wname = name + "_T"
        if wname not in out:
            out[wname] = {len(p) + 1 + sum("*" in q for q in p) for p in plists}
    return out


def _norm(node, src):
    """Source text of an expression, whitespace collapsed and shortened for the pinned inventory."""
    try:
        text = ast.get_source_segment(src, node) or ""
    except Exception:                                                    # pragma: no cover
        text = ""
    text = " ".join(text.split())
    return text if len(text) <= 60 else text[:57] + "..."


def _kernel_name(node):
    """The kernel a launch names, or None when the expression is not a static attribute."""
    if isinstance(node, ast.Attribute) and node.attr.startswith("k_"):
        return node.attr
    return None


def _count(node):
    """The argument count a literal tuple expression yields, or None when it cannot be counted."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _count(node.left), _count(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.IfExp):                 # `() if rf2 else (hp,)`: the larger branch
        body, other = _count(node.body), _count(node.orelse)
        if body is None or other is None:
            return None
        return max(body, other)
    return None


def _launches(path):
    """(call node, kernel expr node, args expr node) for every graph launch in a module."""
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 4:
            continue
        func = node.func
        is_add = isinstance(func, ast.Attribute) and func.attr == "add"
        is_e = isinstance(func, ast.Name) and func.id == "E"
        if not (is_add or is_e):
            continue
        if not isinstance(node.args[1], (ast.Tuple, ast.List, ast.Name, ast.BinOp, ast.Call,
                                         ast.Subscript, ast.IfExp)):
            continue
        yield node, node.args[0], node.args[3], src


def check_module(path):
    """(checked, unresolved, mismatches) for one module.

    checked:    [(lineno, kernel, count, declared)]
    unresolved: [(expression, reason)]
    mismatches: [(lineno, kernel, count, declared)]
    """
    arities = _kernel_arity()
    checked, unresolved, mismatches = [], [], []
    for _node, kernel_expr, args_expr, src in _launches(path):
        name = _kernel_name(kernel_expr)
        if name is None:
            unresolved.append((_norm(kernel_expr, src), "kernel expression is not a static k_* attribute"))
            continue
        if name not in arities:
            unresolved.append(("%s (args)" % _norm(kernel_expr, src),
                               "kernel %s has no .cu declaration" % name))
            continue
        count = _count(args_expr)
        if count is None:
            unresolved.append((_norm(args_expr, src), "argument count is not a literal tuple"))
            continue
        checked.append((_node.lineno, name, count, sorted(arities[name])))
        if count not in arities[name]:
            mismatches.append((_node.lineno, name, count, sorted(arities[name])))
    return checked, unresolved, mismatches


def test_the_contract_covers_every_module():
    """Every module contributes measurements, and the run prints how much is measured against how much
    is skipped, so a passing suite does not imply full coverage (REQ-0023 A1)."""
    rows, measured, skipped = [], 0, 0
    for path in MODULES:
        checked, unresolved, _ = check_module(path)
        assert checked, "%s has no measurable launch; the module list would be vacuous" % path
        measured += len(checked)
        skipped += len(unresolved)
        rows.append("%s %d/%d" % (path.split("/")[-1], len(checked), len(checked) + len(unresolved)))
    print("kernel arity: %d call sites checked, %d skipped (pinned with a reason in UNRESOLVED_SITES); %s"
          % (measured, skipped, ", ".join(rows)))


def test_every_call_site_passes_the_kernels_argument_count():
    problems = []
    for path in MODULES:
        checked, _, mismatches = check_module(path)
        assert checked, "%s has no measurable launch" % path
        for lineno, name, count, declared in mismatches:
            problems.append("%s:%d passes %d argument(s) to %s, which declares %s - a short tuple is "
                            "not a Python error, it is a graph that does not build"
                            % (path, lineno, count, name, "/".join(str(d) for d in declared)))
    assert not problems, "\n".join(problems)


def test_unresolved_call_sites_are_listed_not_skipped():
    seen = collections.Counter()
    for path in MODULES:
        _, unresolved, _ = check_module(path)
        for expr, reason in unresolved:
            seen[(path, expr, reason)] += 1
    pinned = collections.Counter(UNRESOLVED_SITES)
    missing = sorted(set(pinned) - set(seen))
    extra = sorted(set(seen) - set(pinned))
    assert not missing and not extra, (
        "the pinned list of unresolvable launch sites is stale.\n"
        "new (add to UNRESOLVED_SITES with a reason): %r\n"
        "gone (remove from UNRESOLVED_SITES): %r\n"
        "seen: %r" % (extra, missing, sorted(seen)))


def test_measured_sites_are_the_pinned_count():
    """A pin on the number checked, so a silent drop in coverage is a failure too."""
    counts = {path: len(check_module(path)[0]) for path in MODULES}
    total = sum(counts.values())
    assert total == 71, counts
    assert counts["kindi/spec.py"] >= 26, counts


def _mutate(path, kernel_expr, args_expr, mode="drop"):
    """Source bytes of the module with that call site's argument tuple shortened.

    `drop` removes the last argument, which is the shape of the real `hp` bug. A kernel declared with
    two arities (`k_tc_q8`: 7 on HIP/Pascal, 8 on CUDA) can land on the other declaration, so the
    caller falls back to `empty` - an empty argument tuple - when `drop` is not detected.
    """
    raw = (ROOT / path).read_bytes()
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def off(node):
        return starts[node.lineno - 1] + node.col_offset

    def end(node):
        return starts[node.end_lineno - 1] + node.end_col_offset

    if mode == "drop" and isinstance(args_expr, (ast.Tuple, ast.List)) and len(args_expr.elts) >= 2:
        prev, last = args_expr.elts[-2], args_expr.elts[-1]
        return raw[:end(prev)] + raw[end(last):]
    return raw[:off(args_expr)] + b"()" + raw[end(args_expr):]


def test_every_checked_site_is_detected_when_mutated():
    detected, missed = [], []
    for path in MODULES:
        src = (ROOT / path).read_text(encoding="utf-8")
        for node, kernel_expr, args_expr, _ in _launches(path):
            if _kernel_name(kernel_expr) is None or _count(args_expr) is None:
                continue
            name = _kernel_name(kernel_expr)
            want = node.lineno
            caught, last_reason = False, ""
            for mode in ("drop", "empty"):
                mutated = _mutate(path, kernel_expr, args_expr, mode)
                with tempfile.TemporaryDirectory() as tmp:
                    target = os.path.join(tmp, os.path.basename(path))
                    tree = ast.parse(mutated.decode("utf-8"), filename=target)
                    sites = [m for m in ast.walk(tree) if isinstance(m, ast.Call) and m.lineno == want and
                             len(m.args) == 4 and
                             ((isinstance(m.func, ast.Name) and m.func.id == "E") or
                              (isinstance(m.func, ast.Attribute) and m.func.attr == "add"))]
                    if not sites:
                        last_reason = "site moved after mutation"
                        continue
                    _c, _u, mism = check_module_from_source(mutated.decode("utf-8"), path)
                    if any(k == name and c not in declared for _, k, c, declared in mism):
                        caught = True
                        break
                    last_reason = "%s mutation not detected" % mode
            if caught:
                detected.append("%s:%d %s" % (path, want, name))
            else:
                missed.append("%s:%d %s (%s)" % (path, want, name, last_reason))
    assert not missed, "mutations that no check caught:\n" + "\n".join(missed)
    assert len(detected) == 71, len(detected)


def check_module_from_source(src, name):
    """check_module over an in-memory copy (the mutation test must not touch the worktree)."""
    arities = _kernel_arity()
    tree = ast.parse(src, filename=str(name))
    checked, unresolved, mismatches = [], [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 4:
            continue
        func = node.func
        if not ((isinstance(func, ast.Attribute) and func.attr == "add") or
                (isinstance(func, ast.Name) and func.id == "E")):
            continue
        kernel_expr, args_expr = node.args[0], node.args[3]
        kname = _kernel_name(kernel_expr)
        if kname is None or kname not in arities:
            unresolved.append((_norm(kernel_expr, src), "not static"))
            continue
        count = _count(args_expr)
        if count is None:
            unresolved.append((_norm(args_expr, src), "not a literal tuple"))
            continue
        checked.append((node.lineno, kname, count, sorted(arities[kname])))
        if count not in arities[kname]:
            mismatches.append((node.lineno, kname, count, sorted(arities[kname])))
    return checked, unresolved, mismatches
