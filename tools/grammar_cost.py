#!/usr/bin/env python3
"""What a constrained vocab filter costs per token, on a pack's real vocabulary.

`Grammar.vocab_allowed` / `JsonGrammar.vocab_allowed` enumerate every vocabulary entry and advance
the automaton over that entry's text, caching the surviving ids **per automaton state**. So a
request pays one full vocabulary sweep per *new* state it reaches, and the per-token price has to
count real cache misses over a real document -- which is what this measures. No GPU, no server.

  python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b

Prints one row per grammar and exits non-zero if a grammar refused the document: a rejected document
makes the row meaningless, and this harness must not report a number it did not measure. The
2026-11-17 run on the 2B is recorded in
docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md.

`--long` adds the longer document. It was rejected by the schema and by the depth-1 grammar in the
recorded run (undeclared keys / deeper nesting), so those rows are opt-in and are reported as
refusals rather than swallowed.
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kindi.gbnf import Grammar                                         # noqa: E402
from kindi.jsongrammar import grammar_from_schema                      # noqa: E402
from kindi.tokenizer import Tok                                        # noqa: E402

# llama.cpp's own grammars/json.gbnf (value -> object | array | string | number | literal, with
# object/array recursing into value) -- the canonical recursive JSON grammar.
JSON_GBNF = r'''
root   ::= object
value  ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? "}" ws
array  ::= "[" ws ( value ( "," ws value )* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\"" ws
number ::= ( "-"? ( [0-9] | [1-9] [0-9]{0,15} ) ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9] [1-9]{0,15} )? ws
boolean ::= ( "true" | "false" ) ws
null   ::= "null" ws
ws     ::= [ \t\n\r]*
'''

# A record grammar with its keys enumerated: the same document language, more states per level,
# which is how a tool-call / record schema grows without recursion.
RECORD_GBNF = r'''
root   ::= "{" ws pair ( "," ws pair )* "}" ws
pair   ::= "\"" key "\"" ws ":" ws value
key    ::= "name" | "tags" | "n" | "id" | "type" | "value" | "items" | "count"
value  ::= string | number | "true" | "false" | "null" | list
list   ::= "[" ws ( value ( "," ws value )* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""
number ::= ( "-"? ( [0-9] | [1-9] [0-9]{0,15} ) ) ( "." [0-9]+ )?
ws     ::= [ \t\n\r]*
'''

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "n": {"type": "integer"},
    },
    "required": ["name", "tags", "n"],
    "additionalProperties": False,
}
DOC = '{"name": "alpha", "tags": ["x", "y"], "n": 42}'
LONG_DOC = ('{"id": "req-0001", "type": "object", "count": 7, "items": ["alpha", "beta", "gamma"], '
            '"value": {"name": "nested", "n": 3, "tags": ["p", "q"]}}')


def vocab_texts(pack):
    vocab = json.load(open(os.path.join(pack, "vocab.json"), encoding="utf-8"))
    tok = Tok(vocab, os.path.join(pack, "merges.txt"))
    return tok, [tok.decode_bytes(i).decode("utf-8", "replace") for i in range(len(vocab))]


def walk(g, tok, texts, doc, label):
    """Time every cold vocabulary sweep the document forces; returns (ms_per_token, accepted)."""
    cache = getattr(g, "_vocab_cache", None)
    if cache is None:
        g._vocab_cache = cache = {}
    ids = tok.encode(doc)[:96]
    t0 = time.perf_counter()
    g.vocab_allowed(g.start(), texts)
    t_start = time.perf_counter() - t0
    state, per_miss, ok = g.start(), [], True
    t_walk = time.perf_counter()
    for i in ids:
        state = g.advance(state, texts[i])
        if state is None:
            ok = False
            break
        hit = state in cache
        t1 = time.perf_counter()
        g.vocab_allowed(state, texts)
        if not hit:
            per_miss.append((time.perf_counter() - t1) * 1e3)
    t_walk = time.perf_counter() - t_walk
    total = sum(per_miss)
    print("%-22s tokens=%2d states=%5d misses=%2d  filter=%8.0f ms  %7.0f ms/token  "
          "miss med=%6.0f max=%6.0f ms  walk=%7.0f ms  start=%5.0f ms  doc_ok=%s"
          % (label, len(ids), getattr(g, "states", 0), len(per_miss), total,
             total / max(1, len(ids)), statistics.median(per_miss) if per_miss else 0,
             max(per_miss) if per_miss else 0, t_walk * 1e3, t_start * 1e3, ok))
    return total / max(1, len(ids)), ok


def build(text, depth, max_states="200000"):
    os.environ["KINDI_GBNF_MAX_DEPTH"] = str(depth)
    os.environ["KINDI_GBNF_MAX_STATES"] = max_states
    t0 = time.perf_counter()
    g = Grammar.from_string(text)
    return g, (time.perf_counter() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"),
                    help="pack directory with vocab.json and merges.txt (default: %(default)s)")
    ap.add_argument("--long", action="store_true",
                    help="also walk the longer document (refused by the depth-1 grammar and by the "
                         "schema; the recorded run reports those as refusals)")
    args = ap.parse_args()

    tok, texts = vocab_texts(args.pack)
    print("pack: %s\nvocab entries: %d\n" % (args.pack, len(texts)))

    rows, bad = [], False
    for depth in (1, 2, 3):
        try:
            g, ms = build(JSON_GBNF, depth)
        except ValueError as e:
            print("gbnf json depth=%d refused: %s" % (depth, e))
            continue
        print("gbnf json depth=%d: states=%d compile=%.1f ms" % (depth, g.states, ms))
        rows.append((g, "gbnf json d%d" % depth))

    g, ms = build(RECORD_GBNF, 1)
    print("gbnf record: states=%d compile=%.1f ms\n" % (g.states, ms))
    rows.append((g, "gbnf record"))

    for g, label in rows:
        _, ok = walk(g, tok, texts, DOC, label)
        bad = bad or not ok
    _, ok = walk(grammar_from_schema(SCHEMA), tok, texts, DOC, "json_schema")
    bad = bad or not ok

    if args.long:
        print("")
        for g, label in rows:
            _, ok = walk(g, tok, texts, LONG_DOC, label + " long")
            bad = bad or not ok
        _, ok = walk(grammar_from_schema(SCHEMA), tok, texts, LONG_DOC, "json_schema long")
        bad = bad or not ok

    if bad:
        print("\na document was refused: those rows are not a measurement")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
