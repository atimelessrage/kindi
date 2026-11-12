#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Dump a checksummed GGUF metadata and tensor inventory for the Dory D0 audit.

This reads only the GGUF header and tensor-info table; it never maps or loads tensor data.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kindi.gguf import GGUF

# GGML tensor types used by the released Dory Q5_K_M file and common fallbacks. Unknown ids are
# retained numerically so the inventory remains complete if the upstream enum grows.
TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4",
    32: "Q4_0_4_8", 33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
    37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8", 39: "MXFP4",
}
# Keys whose values are large tokenizer tables. They are summarized rather than copied into the
# inventory; the inventory's purpose is the model/layer contract, not a tokenizer dump.
LARGE = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_types"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def summarize_metadata(g):
    out = {}
    for key, value in g.kv.items():
        if key in LARGE:
            if isinstance(value, list):
                out[key] = {"count": len(value), "type": "array",
                            "sha256": hashlib.sha256("\n".join(map(str, value)).encode()).hexdigest()}
            else:
                out[key] = {"type": type(value).__name__, "value": value}
        else:
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="path to the released Dory GGUF")
    ap.add_argument("-o", "--output", help="write JSON here instead of stdout")
    args = ap.parse_args()
    if not os.path.isfile(args.gguf):
        raise SystemExit("not a file: %s" % args.gguf)
    g = GGUF(args.gguf)
    tensors = []
    for name, ne, ty, off in g.tensors:
        tensors.append({"name": name, "shape": list(ne), "type_id": int(ty),
                        "type": TYPES.get(int(ty), "UNKNOWN_%d" % int(ty)),
                        "offset": int(off)})
    doc = {
        "file": os.path.abspath(args.gguf),
        "file_size": os.path.getsize(args.gguf),
        "sha256": sha256(args.gguf),
        "gguf_version": g.version,
        "metadata_count": len(g.kv),
        "tensor_count": len(tensors),
        "data_start": g.data_start,
        "metadata": summarize_metadata(g),
        "tensor_types": {TYPES.get(int(t), "UNKNOWN_%d" % int(t)): sum(1 for x in g.tensors if int(x[2]) == int(t))
                         for t in sorted({int(x[2]) for x in g.tensors})},
        "tensors": tensors,
    }
    text = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s (%d tensors)" % (args.output, len(tensors)))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
