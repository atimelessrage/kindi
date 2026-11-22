"""Every graph call site passes exactly the arguments its CUDA kernel declares.

A kernel can gain a parameter that its Python callers never learn about: `k_rmsfold_qT` took a trailing
`hp` from 0.5.0, `spec.py` and `dspine.py` were updated and `mtp.py` was not, so the MTP head's CUDA
graph did not build (`cuGraphAddKernelNode failed (1)`) and the engine logged "speculation off" - a
documented feature silently dead for four releases (fixed in 0.6.1, ported here). A short argument
tuple is not a Python error, so nothing caught it.

This test is the launch-shape contract for the four modules that build graphs (`spec.py`,
`dspine.py`, `mtp.py`, `tp.py`): it reads each kernel's parameter count from its `.cu` declaration and
each call site's argument tuple from the Python source. DSpine's `self.kf` dictionary and its
`self.k` / `self.k8` / `self.k16` assignments are resolved from the literals that build them, but only
when the table is rooted at `self`; `other.k = ...` is not this object's table. A conditional kernel
expression is checked branch by branch: matching the union of the branches' arities is not enough.
Call sites it cannot resolve statically are **listed and pinned** (`UNRESOLVED_SITES`) rather than
skipped, so a new dynamic launch has to be classified by a person instead of disappearing from the
check.

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
    # dspine.py: `kf[...]`, `self.k` and `self.k8 if ... else self.k16` are resolved from the
    # literals that build their tables; these two are still not kernel expressions.
    ('kindi/dspine.py', 'fn if hasattr(fn, "kernel") else _Fk(fn)', 'kernel expression is not a static k_* attribute'),
    ('kindi/dspine.py', 'self.hmv', 'kernel expression is not a static k_* attribute'),
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
    ('kindi/spec.py', '(FO[k * QL:], KN[k * L:], _tp.Dyn(pk, s.cos_t.data.ptr, r...', 'argument count is not a literal tuple'),
    ('kindi/spec.py', '(KRO[k * L:], VB[k * L:], _tp.Dyn(pk, S["Kq"][kvi].data.p...', 'argument count is not a literal tuple'),
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


def _string_values(node, consts):
    """Possible string values of a small literal expression (`"k_" + sfx`, `x if c else y`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(consts.get(node.id, ()))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return {a + b for a in _string_values(node.left, consts)
                for b in _string_values(node.right, consts)}
    if isinstance(node, ast.IfExp):
        return _string_values(node.body, consts) | _string_values(node.orelse, consts)
    return set()


def _assign_pairs(target, value):
    """`a, b = x, y` as pairs, otherwise one pair."""
    if (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)):
        return list(zip(target.elts, value.elts))
    return [(target, value)]


def _dict_keys(node):
    """The static keys of `{k: ... for k in (...)}` (or a literal dict)."""
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    if isinstance(node, ast.DictComp):
        keys = set()
        for gen in node.generators:
            if isinstance(gen.iter, (ast.Tuple, ast.List)):
                keys.update(v.value for v in gen.iter.elts
                            if isinstance(v, ast.Constant) and isinstance(v.value, str))
        return keys
    return set()


def _function_names(node, consts):
    """Kernel names behind `M.get_function("name" + suffix)`; empty when not such a call."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_function" and node.args):
        return _string_values(node.args[0], consts)
    return set()


def _is_self_attr(node):
    """True only for a direct `self.<name>` attribute, not `other.name` or a chained table."""
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self")


def _kernel_tables(src):
    """Resolve the local kernel dictionaries/attributes the checker can read statically.

    `dspine.py` builds `self.kf` from a literal list of names and keeps kernels on `self.k` /
    `self.k8`/`self.k16`; those are as static as `SK.k_*`, but the old checker only accepted the
    latter.  Only assignments rooted at `self` are followed: `other.k = ...` is not this object's
    table.  The table keeps the dict keys, the attributes' possible names, and the local aliases
    (`kf = self.kf`) so a launch through any of them is measured instead of pinned.
    """
    tree = ast.parse(src)
    consts = {}
    for _ in range(3):                       # resolve `sfx = ...` before `"prefix" + sfx`
        before = {k: set(v) for k, v in consts.items()}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                for target, value in _assign_pairs(node.targets[0], node.value):
                    if isinstance(target, ast.Name):
                        consts.setdefault(target.id, set()).update(_string_values(value, consts))
        if consts == before:
            break
    attrs, dicts, aliases = {}, {}, set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        for target, value in _assign_pairs(node.targets[0], node.value):
            if _is_self_attr(target) and target.attr == "kf":
                keys = _dict_keys(value)
                if keys:
                    dicts.setdefault("kf", set()).update(keys)
            elif _is_self_attr(target):
                names = _function_names(value, consts)
                if names:
                    attrs.setdefault(target.attr, set()).update(names)
            elif isinstance(target, ast.Name) and target.id == "kf":
                if (_is_self_attr(value) and value.attr == "kf") or (
                        isinstance(value, ast.Name) and value.id in aliases):
                    aliases.add(target.id)
    return {"attrs": attrs, "dicts": dicts, "aliases": aliases}


def _kernel_names(node, tables):
    """Possible kernel names a launch expression denotes, or None when it is not static."""
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("k_"):
            return {node.attr}
        if _is_self_attr(node):
            return set(tables["attrs"].get(node.attr, ())) or None
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        is_table = (_is_self_attr(node.value) and node.value.attr == "kf") or (
            isinstance(node.value, ast.Name) and node.value.id in tables["aliases"])
        if is_table and node.slice.value in tables["dicts"].get("kf", ()):
            return {node.slice.value}
        return None
    if isinstance(node, ast.IfExp):
        body, other = _kernel_names(node.body, tables), _kernel_names(node.orelse, tables)
        return body | other if body is not None and other is not None else None
    return None


def _counts(node):
    """Every argument count a literal expression can yield, or None when it cannot be counted.

    A conditional tuple is not one tuple with its larger branch's length: `() if rf2 else (hp,)`
    is either zero or one argument, and both are checked. Keeping the branches apart is what stops
    a short branch from riding on the largest branch's count.
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        return {len(node.elts)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _counts(node.left), _counts(node.right)
        if left is None or right is None:
            return None
        return {a + b for a in left for b in right}
    if isinstance(node, ast.IfExp):
        body, other = _counts(node.body), _counts(node.orelse)
        return None if body is None or other is None else body | other
    return None


def _count(node):
    """The largest possible count, for callers that only ask whether an expression is countable."""
    counts = _counts(node)
    return None if counts is None else max(counts)


def _launch_sites(src, path):
    """(call node, kernel expr node, args expr node) for every graph launch in `src`."""
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


def _launches(path):
    src = (ROOT / path).read_text(encoding="utf-8")
    yield from _launch_sites(src, path)


def _check_source(src, path, arities=None):
    """(checked, unresolved, mismatches) over one module source.

    checked:    [(lineno, kernel, counts, declared)] where counts is every argument count the
                expression can take
    unresolved: [(expression, reason)]
    mismatches: [(lineno, kernel_branch, count, declared)]; every kernel branch must accept every
                argument count, so neither a union of the branches' declared counts nor the largest
                tuple branch is enough.
    """
    arities = _kernel_arity() if arities is None else arities
    tables = _kernel_tables(src)
    checked, unresolved, mismatches = [], [], []
    for node, kernel_expr, args_expr, source in _launch_sites(src, path):
        names = _kernel_names(kernel_expr, tables)
        if names is None:
            unresolved.append((_norm(kernel_expr, source), "kernel expression is not a static k_* attribute"))
            continue
        missing = names - set(arities)
        if missing:
            unresolved.append(("%s (args)" % _norm(kernel_expr, source),
                               "kernel %s has no .cu declaration" % "+".join(sorted(missing))))
            continue
        counts = _counts(args_expr)
        if counts is None:
            unresolved.append((_norm(args_expr, source), "argument count is not a literal tuple"))
            continue
        name = "+".join(sorted(names))
        declared = sorted(set().union(*(arities[n] for n in names)))
        checked.append((node.lineno, name, tuple(sorted(counts)), declared))
        for count in sorted(counts):
            for branch in sorted(names):
                branch_declared = sorted(arities[branch])
                if count not in arities[branch]:
                    mismatches.append((node.lineno, branch, count, branch_declared))
    return checked, unresolved, mismatches


def check_module(path):
    return _check_source((ROOT / path).read_text(encoding="utf-8"), path)


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


def test_kernel_tables_are_only_followed_when_rooted_at_self():
    src = '''\
def launch(self, other):
    self.k = M.get_function("k_self")
    other.k = M.get_function("k_other")
    other.kf = {"k_other": M.get_function("k_other")}
    E(self.k, (1,), (1,), (a,))
    E(other.k, (1,), (1,), (a,))
    E(other.kf["k_other"], (1,), (1,), (a,))
'''
    checked, unresolved, mismatches = _check_source(
        src, "fixture.py", {"k_self": {1}, "k_other": {1}})
    assert [name for _, name, _, _ in checked] == ["k_self"]
    assert [expr for expr, _ in unresolved] == ["other.k", 'other.kf["k_other"]']
    assert not mismatches


def test_every_kernel_branch_must_accept_the_argument_count():
    """A conditional launch is not valid when only the union of its branches' arities matches."""
    src = '''\
def launch(self, cond):
    self.k = M.get_function("k_six")
    self.k8 = M.get_function("k_seven")
    E(self.k if cond else self.k8, (1,), (1,), (a, b, c, d, e, f))
'''
    checked, unresolved, mismatches = _check_source(
        src, "fixture.py", {"k_six": {6}, "k_seven": {7}})
    assert not unresolved
    assert [name for _, name, _, _ in checked] == ["k_seven+k_six"]
    assert [(name, count, declared) for _, name, count, declared in mismatches] == [
        ("k_seven", 6, [7])]


def test_every_argument_branch_must_match_the_kernel():
    """A shorter tuple branch cannot ride on the largest branch's count."""
    src = '''
def launch(self, cond):
    self.k = M.get_function("k_seven")
    E(self.k, (1,), (1,), (a, b, c, d, e, f) if cond else (a, b, c, d, e, f, g))
'''
    checked, unresolved, mismatches = _check_source(
        src, "fixture.py", {"k_seven": {7}})
    assert not unresolved
    assert [(name, counts) for _, name, counts, _ in checked] == [("k_seven", (6, 7))]
    assert [(name, count, declared) for _, name, count, declared in mismatches] == [
        ("k_seven", 6, [7])]


def test_measured_sites_are_the_pinned_count():
    """A pin on the number checked, so a silent drop in coverage is a failure too."""
    counts = {path: len(check_module(path)[0]) for path in MODULES}
    total = sum(counts.values())
    assert total == 118, counts
    assert counts["kindi/dspine.py"] == 37, counts
    assert counts["kindi/spec.py"] >= 26, counts
    assert len(UNRESOLVED_SITES) == 39


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
        tables = _kernel_tables(src)
        for node, kernel_expr, args_expr, _ in _launch_sites(src, path):
            names = _kernel_names(kernel_expr, tables)
            if names is None or _count(args_expr) is None:
                continue
            name = "+".join(sorted(names))
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
                    if any(k in names and c not in declared for _, k, c, declared in mism):
                        caught = True
                        break
                    last_reason = "%s mutation not detected" % mode
            if caught:
                detected.append("%s:%d %s" % (path, want, name))
            else:
                missed.append("%s:%d %s (%s)" % (path, want, name, last_reason))
    assert not missed, "mutations that no check caught:\n" + "\n".join(missed)
    assert len(detected) == 118, len(detected)


def check_module_from_source(src, name):
    """check_module over an in-memory copy (the mutation test must not touch the worktree)."""
    return _check_source(src, name)
