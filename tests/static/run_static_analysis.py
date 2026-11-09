#!/usr/bin/env python3
"""Build the full CUDA translation units the way kindi.spec.SpecKernels does and run cppcheck and clang-tidy.

The Turing verification (docs/VERIFICATION.md) has no CUDA toolkit on the analysing machines, so the kernels are
checked as source only: tests/static/cuda_stub.h supplies the execution-space qualifiers, vector types and builtins,
cppcheck parses the concatenated TU as plain C++, and clang-tidy parses it in real CUDA device mode for the target
architecture (nvptx64, so the inline-PTX constraints are the ones that actually apply).

Usage (from the repository root, with Kindi's Python dependencies importable - cupy is needed to build the wrappers):

    python tests/static/run_static_analysis.py                # Turing and Ampere TUs
    python tests/static/run_static_analysis.py --only turing  # one of them
    python tests/static/run_static_analysis.py --keep out/    # keep the generated TUs

The tools are found on PATH or given with CPPCHECK / CLANG_TIDY.  A missing tool is reported and skipped, so the
script also serves machines that have only one of them.  The exit status is non-zero only when a tool fails to run
(not for warnings: the known cppcheck style notes and the two baseline clang-analyzer dead-store reports in
kindi.cu are expected, see docs/VERIFICATION.md).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# (tag, tc source, tca source, CUDA arch)
TARGETS = (
    ("turing", "tc_turing.cu", "tca_turing.cu", "sm_75"),
    ("ampere", "tc.cu", "tca.cu", "sm_86"),
)


def build_tu(tc_name, tca_name):
    """kindi.cu + the generated wrappers + spec.cu + tc*.cu + tcf.cu + tca*.cu, in SpecKernels' order"""
    from kindi import chunk
    from kindi.spec import HERE as KINDIR, WRAP
    base = open(os.path.join(KINDIR, "kindi.cu"), encoding="utf-8").read()
    wr, _ = chunk._wrappers(re.sub(r"__launch_bounds__\(\d+\) (k_\w+)\(", r"\1(", base), WRAP)
    own = open(os.path.join(KINDIR, "spec.cu"), encoding="utf-8").read()
    tcs = open(os.path.join(KINDIR, tc_name), encoding="utf-8").read()
    tcf = open(os.path.join(KINDIR, "tcf.cu"), encoding="utf-8").read()
    tca = open(os.path.join(KINDIR, tca_name), encoding="utf-8").read()
    return base + "\n" + wr + "\n" + own + "\n" + tcs + "\n" + tcf + "\n" + tca


def find_tool(env, name):
    p = os.environ.get(env) or shutil.which(name)
    if p:
        p = os.path.abspath(p)
    return p


def run(cmd, title):
    print("\n=== %s ===\n%s" % (title, " ".join(cmd)), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print("(exit %d)" % r.returncode, flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("turing", "ampere"), help="check one translation unit")
    ap.add_argument("--keep", metavar="DIR", help="write the generated TUs to DIR (kept)")
    ap.add_argument("--no-cppcheck", action="store_true")
    ap.add_argument("--no-clang-tidy", action="store_true")
    ap.add_argument("--bugprone", action="store_true",
                    help="also run bugprone-* (noisy: it flags the CUDA-style names and parameter lists)")
    args = ap.parse_args()

    cppcheck = None if args.no_cppcheck else find_tool("CPPCHECK", "cppcheck")
    clang_tidy = None if args.no_clang_tidy else find_tool("CLANG_TIDY", "clang-tidy")
    if not args.no_cppcheck and not cppcheck:
        print("cppcheck not found (set CPPCHECK); skipping", flush=True)
    if not args.no_clang_tidy and not clang_tidy:
        print("clang-tidy not found (set CLANG_TIDY); skipping", flush=True)

    outdir = args.keep or tempfile.mkdtemp(prefix="kindi-tu-")
    os.makedirs(outdir, exist_ok=True)
    bad = 0
    try:
        for tag, tc_name, tca_name, arch in TARGETS:
            if args.only and args.only != tag:
                continue
            src = build_tu(tc_name, tca_name)
            tu = os.path.join(outdir, "tu_%s.cu" % tag)
            with open(tu, "w", encoding="utf-8", newline="\n") as f:
                f.write(src)
            rel = tu
            stub = os.path.join(HERE, "cuda_stub.h")
            print("\n## %s translation unit: %s (%d bytes)" % (tag, rel, len(src)), flush=True)

            if cppcheck:
                rc = run([cppcheck, "--enable=warning,style,performance,portability", "--language=c++",
                          "--std=c++14", "--suppress=missingIncludeSystem", "--inline-suppr",
                          "--include=" + stub,
                          "--template={file}:{line}: {severity}: {message} [{id}]", rel],
                         "cppcheck %s" % tag)
                bad |= rc != 0
            if clang_tidy:
                ea = ["--extra-arg=" + a for a in
                      ("-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=" + arch, "-nocudainc", "-nocudalib",
                       "-std=c++14", "-include", stub)]
                checks = "-*,clang-analyzer-*" + (",bugprone-*" if args.bugprone else "")
                rc = run([clang_tidy, "--checks=" + checks, "--header-filter=^$"] + ea + [rel],
                         "clang-tidy %s (%s)" % (tag, arch))
                bad |= rc != 0
    finally:
        if not args.keep:
            shutil.rmtree(outdir, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
