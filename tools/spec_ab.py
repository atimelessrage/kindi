#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Direct 27B A/B: plain greedy output against ngram, draft2b and both, byte for byte.

`kindi verify --spec <mode>` compares each mode with llama.cpp's reference, which can pass while
the plain and speculative paths disagree with each other in the same way. This harness builds one
27B engine with the 2B drafter and its head attached, then for every reference prompt generates the
plain output and each speculation mode's output from the same engine and prompt. It compares the
generated token ids and the decoded token bytes directly; the reference verdict is not involved.

The engine is the tensor-core plan path at the verifier's 1,024-token context. The harness sets the
27B verification switches itself (KINDI_TC/TP/FD/ATT_TC/TC_TMAX and KINDI_SPEC=both at build), so a
caller's shell cannot silently make the A/B run another engine. It needs a tensor-core GPU, the 27B
pack, the 2B drafter pack and the registered draft head.

    python tools/spec_ab.py --pack <kindi-model-root>/kindi-27b/packs/bonsai-27b \\
      --drafter ~/.kindi/packs/bonsai-2b \\
      --head ~/.kindi/models/kindi-mtp-2b-v1.npz \\
      --json docs/benchmarks/2026-11-18-i7-8086k-rtx4060ti-27b-spec-ab/results.json

Exit codes: 0 every prompt and mode is byte-identical, 2 a required file/mode is unavailable,
1 the direct comparison found a token or byte difference.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from the repo root

MODES = ("ngram", "draft2b", "both")


def output_bytes(tok, ids):
    """The decoded output bytes, token by token; this is what \"byte-identical\" compares."""
    return b"".join(tok.decode_bytes(int(i)) for i in ids)


def compare_outputs(plain_ids, spec_ids, plain_bytes, spec_bytes):
    """Compare the two generation paths directly, without consulting a reference verdict."""
    plain_ids = list(plain_ids)
    spec_ids = list(spec_ids)
    token_mismatch = next((i for i, (a, b) in enumerate(zip(plain_ids, spec_ids)) if a != b), None)
    if token_mismatch is None and len(plain_ids) != len(spec_ids):
        token_mismatch = min(len(plain_ids), len(spec_ids))
    byte_mismatch = next((i for i, (a, b) in enumerate(zip(plain_bytes, spec_bytes)) if a != b), None)
    if byte_mismatch is None and len(plain_bytes) != len(spec_bytes):
        byte_mismatch = min(len(plain_bytes), len(spec_bytes))
    return {
        "identical": token_mismatch is None and byte_mismatch is None,
        "tokens_identical": token_mismatch is None,
        "bytes_identical": byte_mismatch is None,
        "first_token_mismatch": token_mismatch,
        "first_byte_mismatch": byte_mismatch,
        "plain_tokens": len(plain_ids),
        "spec_tokens": len(spec_ids),
        "plain_bytes": len(plain_bytes),
        "spec_bytes": len(spec_bytes),
        "plain_token_sha256": hashlib.sha256(repr(plain_ids).encode("ascii")).hexdigest(),
        "spec_token_sha256": hashlib.sha256(repr(spec_ids).encode("ascii")).hexdigest(),
        "plain_byte_sha256": hashlib.sha256(plain_bytes).hexdigest(),
        "spec_byte_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    }


def _positive_int(text):
    """A positive integer, rejected at argparse time before the engine is built."""
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive integer (got %r)" % (text,))
    return value


def _modes(text):
    """Comma-separated modes from MODES, rejected at argparse time."""
    modes = list(dict.fromkeys(mode.strip().lower() for mode in str(text).split(",")
                               if mode.strip()))
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown mode(s): %s (choose from %s)" % (", ".join(unknown), ", ".join(MODES)))
    if not modes:
        raise argparse.ArgumentTypeError(
            "at least one mode is required (choose from %s)" % ", ".join(MODES))
    return modes


def _require(path, what):
    if not path:
        raise SystemExit("spec_ab: no %s" % what)
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(p):
        raise SystemExit("spec_ab: %s not found: %s" % (what, p))
    return p


def _generate(eng, ids, n, mode):
    """Generate one output on the same engine; reset happens inside eng.generate."""
    os.environ["KINDI_SPEC"] = mode if mode != "plain" else "0"
    if mode == "plain":
        return eng.generate(ids, n), None
    eng.spec_begin(True)
    try:
        out = eng.generate(ids, n)
    finally:
        st = eng.spec_end() or {}
    return out, st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, help="27B pack")
    ap.add_argument("--drafter", required=True, help="2B drafter pack")
    ap.add_argument("--head", required=True, help="registered 2B MTP head")
    ap.add_argument("--reference", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kindi", "reference-27b.json"),
        help="reference prompt file (default: kindi/reference-27b.json)")
    ap.add_argument("--ctx", type=_positive_int, default=1024,
                    help="context (a positive integer; default: 1024, the verifier's)")
    ap.add_argument("--modes", type=_modes, default=",".join(MODES),
                    help="comma-separated modes from %s (default: %s)" % (
                        ", ".join(MODES), ", ".join(MODES)))
    ap.add_argument("--json", default=None, help="write the result rows here")
    args = ap.parse_args(argv)

    modes = args.modes

    pack = _require(args.pack, "27B pack")
    drafter = _require(args.drafter, "2B drafter pack")
    head = _require(args.head, "draft head")
    reference = _require(args.reference, "reference file")
    if not os.path.isfile(os.path.join(pack, "kindi_pack.json")):
        raise SystemExit("spec_ab: no kindi_pack.json in %s" % pack)

    # Build one engine with both drafter modes attachable. The direct A/B switches only KINDI_SPEC
    # between generations; weights, card and kernels stay the same.
    os.environ.update({
        "KINDI_DEVICE": "gpu",
        "KINDI_HEAD": "exact",
        "KINDI_KV": "q8_0",
        "KINDI_TC": "1",
        "KINDI_TP": "1",
        "KINDI_FD": "1",
        "KINDI_ATT_TC": "1",
        "KINDI_TC_TMAX": "4",
        "KINDI_SPEC": "both",
        "KINDI_SPEC_PACK": drafter,
        "KINDI_SPEC_DRAFT_HEAD": head,
    })

    from kindi import spec as _sp
    from kindi.engine import make_engine
    from kindi.tokenizer import Tok

    eng = make_engine(pack, ctx=args.ctx, log=print)
    if getattr(eng, "drafter", None) is None:
        raise SystemExit("spec_ab: the 2B drafter did not attach; refusing to compare ngram with itself")
    if getattr(eng.drafter, "mtp", None) is None:
        raise SystemExit("spec_ab: the 2B draft head did not attach; refusing to run both")
    tok = Tok(eng.vocab, os.path.join(pack, "merges.txt"))
    ref = json.load(open(reference, encoding="utf-8"))
    prompts = ref.get("prompts") or []
    if not prompts:
        raise SystemExit("spec_ab: %s has no prompts" % reference)

    rows = []
    failures = []
    for index, item in enumerate(prompts, 1):
        ids = [int(t) for t in item["ids"]]
        n = len(item["expected"])
        plain_ids, _ = _generate(eng, ids, n, "plain")
        plain_bytes = output_bytes(tok, plain_ids)
        for mode in modes:
            state = _sp.effective_state(eng, mode)
            if state.get("mode") != mode:
                raise SystemExit("spec_ab: %s was requested but the effective mode is %s (%s)" % (
                    mode, state.get("mode") or "0", state.get("reason") or "no reason"))
            spec_ids, st = _generate(eng, ids, n, mode)
            if getattr(eng, "spec_mode", None) != mode:
                raise SystemExit("spec_ab: %s generated with effective mode %r" % (
                    mode, getattr(eng, "spec_mode", None)))
            spec_bytes = output_bytes(tok, spec_ids)
            row = {
                "prompt": index,
                "prompt_text": item.get("prompt"),
                "mode": mode,
                "tokens": n,
                "effective": getattr(eng, "spec_mode", None) if mode != "plain" else None,
                "drafted": int((st or {}).get("drafted", 0)),
                "accepted": int((st or {}).get("accepted", 0)),
            }
            row.update(compare_outputs(plain_ids, spec_ids, plain_bytes, spec_bytes))
            rows.append(row)
            if not row["identical"]:
                failures.append((index, mode, row))
            print("prompt %d %-60s %-7s %s  %d tokens, %d bytes, sha256 %s" % (
                index, repr(item.get("prompt", ""))[:60], mode,
                "IDENTICAL" if row["identical"] else "DIFFERENT",
                row["plain_tokens"], row["plain_bytes"], row["plain_byte_sha256"][:12]))
            if not row["identical"]:
                print("  first token difference: %s; first byte difference: %s" % (
                    row["first_token_mismatch"], row["first_byte_mismatch"]))
                print("  plain sha256 %s / spec sha256 %s" % (
                    row["plain_byte_sha256"], row["spec_byte_sha256"]))

    total = len(rows)
    identical = sum(row["identical"] for row in rows)
    print("summary: %d/%d prompt/mode pairs byte-identical, %d different" % (
        identical, total, total - identical))
    payload = {
        "pack": os.path.basename(os.path.normpath(pack)),
        "drafter": os.path.basename(os.path.normpath(drafter)),
        "head": os.path.basename(head),
        "ctx": int(eng.ctx),
        "modes": modes,
        "prompts": len(prompts),
        "pairs": total,
        "identical_pairs": identical,
        "rows": rows,
    }
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("wrote %s" % args.json)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
