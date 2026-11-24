#!/usr/bin/env python3
"""Regenerate a grammar-mask benchmark artifact from tools/grammar_cost.py.

The REQ-0027 rows are produced by `tools/grammar_cost.py`; this script runs the same walks in
process and writes their rows as JSON, so a benchmark page's numbers come from the harness rather
than from transcription. It measures, in the harness's own order: the short document on the
unrolled JSON GBNFs, the record GBNF and the schema, then the 200-token document on the ordered
record GBNF and its schema.

  python tools/grammar_mask_record.py --pack ~/.kindi/packs/bonsai-2b --out FILE.json

No GPU, no server: this is the CPU filter cost. Exits non-zero when a constraint refuses a
document, because a refused document makes the row meaningless.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

DESCRIPTION = (__doc__ or "Grammar-mask rows as JSON").splitlines()[0]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import grammar_cost as harness  # noqa: E402
from grammar_cost import long_grammar  # noqa: E402

from kindi.gbnf import Grammar  # noqa: E402
from kindi.jsongrammar import grammar_from_schema  # noqa: E402
from kindi.token_classes import prewarm  # noqa: E402


def _row(g, tok, texts, doc, label):
    """One walk row, capturing the harness's printed line instead of re-deriving its numbers."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ms_per_token, ok, costs, misses = harness.walk(g, tok, texts, doc, label)
        printed = buf.getvalue().splitlines()[1:]
    quarters = printed[0].split(": ")[-1] if printed else None
    return {"tokens": len(costs), "misses": misses,
            "filter_ms": round(ms_per_token * len(costs), 1),
            "ms_per_token": round(ms_per_token, 1),
            "states": getattr(g, "states", 0),
            "quarters_ms_per_token": quarters}, ok


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc", choices=("both", "short", "long"), default="both",
                    help="which documents to walk: the harness runs each in its own process, so a "
                         "long row recorded beside the short ones reuses a warm vocabulary and is "
                         "not comparable to `grammar_cost.py --doc long`")
    ap.add_argument("--prewarm", action="store_true",
                    help="call token_classes.prewarm first (a server does this at startup)")
    args = ap.parse_args()

    args.pack = os.path.expanduser(args.pack)
    tok, texts = harness.vocab_texts(args.pack)
    started = time.perf_counter()
    if args.prewarm:
        prewarm(texts)
    built_ms = (time.perf_counter() - started) * 1e3

    shown_pack = args.pack
    home = os.path.expanduser("~")
    for prefix in (home, home.replace(os.sep, "/")):
        shown_pack = shown_pack.replace(prefix, "~")
    out = {"date": time.strftime("%Y-%m-%d"), "pack": shown_pack, "vocab_entries": len(texts),
           "doc": args.doc, "prewarm": args.prewarm,
           "vocabulary_build_ms": round(built_ms, 1), "short": {}, "long": {}}
    bad = False
    if args.doc in ("both", "short"):
        for depth in (1, 2, 3):
            os.environ["KINDI_GBNF_MAX_DEPTH"] = str(depth)
            g = Grammar.from_string(harness.JSON_GBNF)
            row, ok = _row(g, tok, texts, harness.DOC, "gbnf json d%d" % depth)
            out["short"]["gbnf_json_d%d" % depth] = row
            bad = bad or not ok
        g = Grammar.from_string(harness.RECORD_GBNF)
        row, ok = _row(g, tok, texts, harness.DOC, "gbnf record")
        out["short"]["gbnf_record"] = row
        bad = bad or not ok
        row, ok = _row(grammar_from_schema(harness.SCHEMA), tok, texts, harness.DOC, "json_schema")
        out["short"]["json_schema"] = row
        bad = bad or not ok

    if args.doc in ("both", "long"):
        row, ok = _row(Grammar.from_string(long_grammar()), tok, texts, harness.LONG_DOC,
                       "gbnf long")
        out["long"]["gbnf_ordered_record"] = row
        bad = bad or not ok
        row, ok = _row(grammar_from_schema(harness.LONG_SCHEMA), tok, texts, harness.LONG_DOC,
                       "json_schema")
        out["long"]["json_schema"] = row
        bad = bad or not ok

    out["rows_ok"] = not bad
    out["wall_ms"] = round((time.perf_counter() - started) * 1e3, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out["short"], indent=1))
    print(json.dumps(out["long"], indent=1))
    print("wrote %s" % args.out)
    if bad:
        print("a constraint refused a document: those rows are not a measurement")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
