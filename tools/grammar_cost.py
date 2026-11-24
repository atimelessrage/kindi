#!/usr/bin/env python3
"""What a constrained vocab filter costs per token, on a pack's real vocabulary.

`Grammar.vocab_allowed` / `JsonGrammar.vocab_allowed` use the token-class table, whose
vocabulary-only index is shared across grammars and whose surviving ids are cached **per automaton
state**. The per-token price therefore counts real state-mask misses over a real document; this
harness also keeps the pre-token-class sweep as the language oracle. `--repeat` measures the
first request that builds the shared index against later grammars that reuse it. No GPU, no server.

  python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b            # 22-token document
  python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --doc long # 200-token document
  python tools/grammar_cost.py --pack ~/.kindi/packs/bonsai-2b --repeat  # first vs cached index

Prints one row per grammar and exits non-zero if a grammar refused the document: a rejected document
makes the row meaningless, and this harness must not report a number it did not measure. The
2026-11-17 run on the 2B is recorded in
docs/benchmarks/2026-11-17-i7-8086k-rtx4060ti-grammar-cost/RESULTS.md.

`--doc long` is a 200-token document and a *pair* of constraints that both describe exactly it: an
ordered flat record (keys, types and order fixed) as a GBNF, and the same shape as a JSON schema.
Both accept the document, which is what makes the two rows comparable at a realistic reply length;
rows also carry a per-quarter breakdown, so growth with document length is visible rather than
inferred from one average.
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
REPEAT_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
DOC = '{"name": "alpha", "tags": ["x", "y"], "n": 42}'

# The realistic-length pair: one 200-token document (measured with the 2B tokenizer), an ordered flat
# record grammar that accepts exactly that language, and the equivalent schema. Flat, so no unroll
# depth is involved and the comparison is the constraint, not the recursion limit.
LONG_ORDER = (("id", "str"), ("type", "str"), ("name", "str"), ("count", "num"),
              ("items", "arr"), ("value", "str"), ("tags", "arr"), ("n", "num"))
_LONG_ITEMS = ("Amazon", "Nile", "Yangtze", "Mississippi", "Yenisei", "Yellow", "Ob", "Parana",
               "Congo", "Amur", "Lena", "Mekong", "Volga", "Danube", "Indus", "Ganges",
               "Brahmaputra", "Zambezi", "Orinoco", "Niger")
LONG_DOC = ('{"id": "req-2026-11-17-0001", "type": "river-report", '
            '"name": "A survey of the great rivers of the northern and southern basins", '
            '"count": 20, "items": [' + ", ".join('"%s"' % i for i in _LONG_ITEMS) + '], '
            '"value": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron", '
            '"tags": ["geography", "rivers", "hydrology", "survey", "report", "basin", "long-form"], '
            '"n": 7}')


def long_grammar(order=LONG_ORDER):
    """An ordered flat record: those keys, in that order, with those value types."""
    parts = ['"\\"%s\\"" ws ":" ws %s' % (k, kind) for k, kind in order]
    body = ' "," ws '.join(parts)
    return ('root ::= "{" ws ' + body + ' ws "}" ws\n'
            'arr ::= "[" ws str ( "," ws str )* "]" ws\n'
            'num ::= ( "-"? ( [0-9] | [1-9] [0-9]{0,15} ) ) ( "." [0-9]+ )? ws\n'
            'str ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" ( ["\\\\/bfnrt] | "u" '
            '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\\"" ws\n'
            'ws ::= [ \\t\\n\\r]*\n')


def tight_grammar(order=LONG_ORDER):
    """The same ordered record with **no whitespace** between tokens.

    Measured 2026-11-17: a grammar that permits `[ \\t\\n\\r]*` anywhere lets a greedy 2B spend all
    200 max_tokens on whitespace right after the first colon (9 characters of reply), and the schema
    automaton permits the same. Forbidding it is what makes a long constrained reply progress, so it
    is the variant to measure at length.
    """
    parts = ['"\\"%s\\"" ":" %s' % (k, kind) for k, kind in order]
    body = ' "," '.join(parts)
    return ('root ::= "{" ' + body + ' "}"\n'
            'arr ::= "[" str ( "," str )* "]"\n'
            'num ::= "-"? ( [0-9] | [1-9] [0-9]{0,15} )\n'
            'str ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" ( ["\\\\/bfnrt] | "u" '
            '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\\""\n')


LONG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [k for k, _ in LONG_ORDER],
    "properties": {k: ({"type": "array", "items": {"type": "string"}} if kind == "arr" else
                       {"type": "integer"} if kind == "num" else {"type": "string"})
                   for k, kind in LONG_ORDER},
}


def vocab_texts(pack):
    vocab = json.load(open(os.path.join(pack, "vocab.json"), encoding="utf-8"))
    tok = Tok(vocab, os.path.join(pack, "merges.txt"))
    return tok, [tok.decode_bytes(i).decode("utf-8", "replace") for i in range(len(vocab))]


def walk(g, tok, texts, doc, label):
    """Time every cold vocabulary sweep the document forces.

    Returns (ms_per_token, accepted, per_token_costs, misses). A cached state costs nothing, so the
    per-token list is the honest cost of that token; the quarter breakdown shows whether the price
    grows with the document, which one average cannot.
    """
    # hit detection via the cache's own miss counter: the cache is a StateMaskCache
    # (bounded LRU), not a dict, so `state in cache` no longer works
    cache = getattr(g, "_vocab_cache", None)
    ids = tok.encode(doc)
    t0 = time.perf_counter()
    g.vocab_allowed(g.start(), texts)
    t_start = time.perf_counter() - t0
    state, per_miss, costs, ok = g.start(), [], [], True
    t_walk = time.perf_counter()
    for i in ids:
        state = g.advance(state, texts[i])
        if state is None:
            ok = False
            break
        before = cache.misses if cache is not None else 0
        t1 = time.perf_counter()
        g.vocab_allowed(state, texts)
        dt = (time.perf_counter() - t1) * 1e3
        hit = cache is None or cache.misses == before
        costs.append(0.0 if hit else dt)
        if not hit:
            per_miss.append(dt)
    t_walk = time.perf_counter() - t_walk
    total = sum(per_miss)
    print("%-22s tokens=%3d states=%5d misses=%3d  filter=%8.0f ms  %7.0f ms/token  "
          "miss med=%6.0f max=%6.0f ms  walk=%7.0f ms  start=%5.0f ms  doc_ok=%s"
          % (label, len(costs), getattr(g, "states", 0), len(per_miss), total,
             total / max(1, len(costs)), statistics.median(per_miss) if per_miss else 0,
             max(per_miss) if per_miss else 0, t_walk * 1e3, t_start * 1e3, ok))
    if len(costs) >= 40:
        q = len(costs) // 4
        seg = [costs[i * q:(i + 1) * q] for i in range(4)]
        seg[-1].extend(costs[4 * q:])
        print("%-22s quarters ms/token: %s   (hits %d/%d)"
              % ("", " / ".join("%.0f" % (sum(s) / max(1, len(s))) for s in seg),
                 len(costs) - len(per_miss), len(costs)))
    return total / max(1, len(costs)), ok, costs, len(per_miss)


def build(text, depth, max_states="200000"):
    os.environ["KINDI_GBNF_MAX_DEPTH"] = str(depth)
    os.environ["KINDI_GBNF_MAX_STATES"] = max_states
    t0 = time.perf_counter()
    g = Grammar.from_string(text)
    return g, (time.perf_counter() - t0) * 1e3


def repeat_requests(texts):
    """Measure grammar setup plus the first mask across separate structured requests.

    The token texts are built once, as a server does.  The first grammar owns
    the token-class index build; the next two are new grammar instances over the
    same vocabulary, which is the repeat-request case the process cache covers.
    """
    rows = []
    for label, schema in (("first (same schema)", SCHEMA),
                          ("repeat (same schema)", SCHEMA),
                          ("new grammar", REPEAT_SCHEMA)):
        t0 = time.perf_counter()
        grammar = grammar_from_schema(schema)
        t1 = time.perf_counter()
        allowed = grammar.vocab_allowed(grammar.start(), texts)
        t2 = time.perf_counter()
        rows.append({"label": label, "grammar_ms": (t1 - t0) * 1e3,
                     "first_mask_ms": (t2 - t1) * 1e3, "allowed": len(allowed)})
        print("%-22s grammar=%8.1f ms  first-mask=%8.1f ms  allowed=%d"
              % (label, rows[-1]["grammar_ms"], rows[-1]["first_mask_ms"], len(allowed)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"),
                    help="pack directory with vocab.json and merges.txt (default: %(default)s)")
    ap.add_argument("--doc", choices=("short", "long"), default="short",
                    help="short: the 22-token document on every grammar; long: the 200-token "
                         "document with its own ordered grammar and equivalent schema")
    ap.add_argument("--repeat", action="store_true",
                    help="measure the first request against repeat requests sharing the "
                         "vocabulary index (no document walk)")
    args = ap.parse_args()

    tok, texts = vocab_texts(args.pack)
    print("pack: %s\nvocab entries: %d\n" % (args.pack, len(texts)))

    if args.repeat:
        repeat_requests(texts)
        return 0

    if args.doc == "long":
        print("document: %d tokens\n" % len(tok.encode(LONG_DOC)))
        bad = False
        g, ms = build(long_grammar(), 0)
        print("gbnf long (ordered record): states=%d compile=%.1f ms" % (g.states, ms))
        _, ok, _, _ = walk(g, tok, texts, LONG_DOC, "gbnf long")
        bad = bad or not ok
        _, ok, _, _ = walk(grammar_from_schema(LONG_SCHEMA), tok, texts, LONG_DOC, "json_schema long")
        bad = bad or not ok
        if bad:
            print("\na constraint refused the document: those rows are not a measurement")
        return 1 if bad else 0

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
        _, ok, _, _ = walk(g, tok, texts, DOC, label)
        bad = bad or not ok
    _, ok, _, _ = walk(grammar_from_schema(SCHEMA), tok, texts, DOC, "json_schema")
    bad = bad or not ok

    if bad:
        print("\na document was refused: those rows are not a measurement")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
