"""Real-file conformance for the PQ2_0 / Q2_0 import: convert a GGUF and compare it to a PTQ1_0 pack.

PQ2_0 (ggml type 142) and Q2_0 (type 42) hold the same ternary weights as PTQ1_0 in a coarser
layout, so a correct import must produce the same pack streams byte for byte: bcodes.bin,
dpm.bin, f32.bin, both sign vectors, vocab.json and merges.txt. Only index.json may differ, and
only in the per-tensor `type` (142/42 instead of 143).

    python bench-logs/convert_pq2_q2.py --ref ~/.kindi/packs/bonsai-2b \
        --gguf ~/.kindi/models/Ternary-Bonsai-2-2B-PQ2_0.gguf

The converted pack goes to a temporary directory unless --out is given. Exit 0 only when every
stream matches.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kindi.convert import convert  # noqa: E402

STREAMS = ("bcodes.bin", "dpm.bin", "f32.bin", "signs_2048.f32", "signs_6144.f32",
           "vocab.json", "merges.txt", "bcodes_index.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the PTQ1_0-derived pack to compare against")
    ap.add_argument("--gguf", required=True, help="the PQ2_0 or Q2_0 GGUF")
    ap.add_argument("--out", help="keep the converted pack here (default: a temporary directory)")
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="kindi-format-")
    ok = True
    man = convert(a.gguf, out)
    print("source_format: %s" % man.get("source_format"))
    for name in STREAMS:
        p, q = os.path.join(a.ref, name), os.path.join(out, name)
        if not (os.path.exists(p) and os.path.exists(q)):
            print("%-20s MISSING (%s)" % (name, "ref" if not os.path.exists(p) else "converted"))
            ok = False
            continue
        same = sha(p) == sha(q)
        ok &= same
        print("%-20s %s" % (name, "identical" if same else "DIFFERENT"))
    # index.json: same tensor table, except that the source type is recorded per tensor
    idx = {n: dict(r) for n, r in json.load(open(os.path.join(out, "index.json")))["tensors"].items()}
    ref = {n: dict(r) for n, r in json.load(open(os.path.join(a.ref, "index.json")))["tensors"].items()}
    types = {r.pop("type") for r in idx.values()}
    for r in ref.values():
        r.pop("type")
    same = idx == ref
    ok &= same
    print("%-20s %s (source types %s)" % ("index.json", "same tensor table" if same else "DIFFERENT", sorted(types)))
    ref_man = json.load(open(os.path.join(a.ref, "kindi_pack.json")))
    keys = [k for k in man if k not in ("source_file", "source_sha256", "source_format", "model", "kindi_version")]
    diff = [k for k in keys if man.get(k) != ref_man.get(k)]
    ok &= not diff
    print("%-20s %s" % ("kindi_pack.json", "same fields" if not diff else "DIFFERENT %s" % diff))
    if not a.out:
        print("pack kept at %s" % out)
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
