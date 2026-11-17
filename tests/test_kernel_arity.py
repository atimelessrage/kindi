"""Every Python call site passes exactly the arguments its CUDA kernel declares.

A kernel can gain a parameter that its Python callers never learn about: `k_rmsfold_qT` took a
trailing `hp` from 0.5.0, `spec.py` and `dspine.py` were updated and `mtp.py` was not, so the MTP
head's CUDA graph did not build (`cuGraphAddKernelNode failed (1)`) and the engine logged
"speculation off" - a documented feature silently dead for four releases (fixed in 0.6.1, ported
here). Nothing caught it because a too-short argument tuple is not a Python error.

This test reads the kernel's parameter count from the CUDA source and the argument tuple each
Python call site passes, so the next parameter is caught at test time rather than by a log line.
Only literal tuples and literal `+ conditional` tuples are measured; anything it cannot count is
skipped rather than guessed.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "k_rmsfold_qT"


def _kernel_arity(name=KERNEL):
    """Parameters declared by `extern "C" __global__ void <name>(...)` in kindi/*.cu."""
    for source in sorted((ROOT / "kindi").glob("*.cu")):
        text = source.read_text(encoding="utf-8", errors="replace")
        start = text.find("__global__ void %s(" % name)
        if start < 0:
            continue
        i = text.index("(", start) + 1
        depth, params, buf = 0, [], []
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
        return [p.strip() for p in params if p.strip()]
    return []


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


def _call_sites(path):
    """[(lineno, arg_count or None)] for every call whose kernel argument mentions KERNEL."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not node.args:
            continue
        kernel = node.args[0]                       # Plan.add(SK.<kernel>, grid, block, (args...))
        if not any(isinstance(n, ast.Attribute) and n.attr == KERNEL for n in ast.walk(kernel)):
            continue
        args = node.args[3:]                        # Plan.add / E(kernel, grid, block, (kernel args))
        if len(args) != 1:
            yield node.lineno, None
            continue
        yield node.lineno, _count(args[0])


def test_the_kernel_declares_the_parameters_this_test_counts():
    params = _kernel_arity()
    assert len(params) == 20, params
    assert params[-1] == "int hp"


@pytest.mark.parametrize("module", ["kindi/mtp.py", "kindi/dspine.py", "kindi/spec.py"])
def test_every_call_site_passes_the_kernels_argument_count(module):
    arity = len(_kernel_arity())
    seen = 0
    for lineno, count in _call_sites(ROOT / module):
        if count is None:
            continue                            # not a literal tuple: not measurable here
        seen += 1
        assert count == arity, (
            "%s:%d passes %d argument(s) to %s, which declares %d - a short tuple is not a Python "
            "error, it is a graph that does not build" % (module, lineno, count, KERNEL, arity))
    assert seen, "%s has no measurable %s call site; the test would be vacuous" % (module, KERNEL)
