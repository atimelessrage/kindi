#!/usr/bin/env python3
"""What the accepted-character skip saves per mask miss, and that it changes no mask.

The 2026-11-25 trie-walk round left one automaton question per distinct first character of the
vocabulary on every trie node. This harness A/Bs the follow-up in one process, with the grammar's
per-state mask cache off so every state pays its first-visit mask:

  * `exact` - the table with no `state_chars`: every child edge asks the automaton;
  * `skip`  - the same table plus `state_chars`, so a node looks up only the characters its state
    can accept (`JsonGrammar._accepted_chars`).

Both arms read the same prewarmed vocabulary index and the same states. Identity is checked twice:
`skip` against `exact` on every state, and both against the pre-token-class `_vocab_allowed_scan`
on the first `--scan-states` states (the full 248,320-entry sweep).

  python docs/benchmarks/2026-11-25-i7-8086k-cpu-key-chars/mask_ab.py \
      --pack ~/.kindi/packs/bonsai-2b --out mask.json --rounds 3

No GPU, no server and no model: the mask's own work.
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import grammar_cost  # noqa: E402

from kindi import token_classes  # noqa: E402
from kindi.jsongrammar import grammar_from_schema  # noqa: E402


def states_for(grammar, tok, texts, doc):
    """The automaton state before each token of `doc` (None if the doc is refused)."""
    state, out = grammar.start(), [grammar.start()]
    for token in tok.encode(doc):
        state = grammar.advance(state, texts[token])
        if state is None:
            return None
        out.append(state)
    return out


def exact_table(grammar, texts):
    """The walk without the accepted-character hook: one automaton step per child edge."""
    return token_classes.TokenClassTable(texts, grammar._step, grammar._token_safe_sets)


def counting_pass(table, states):
    """Automaton steps and distinct edge states per mask miss."""
    original = table._step
    out = []
    for state in states:
        stats = {"step_calls": 0}
        seen = set()

        def step(st, ch, _original=original, _stats=stats, _seen=seen):
            _stats["step_calls"] += 1
            _seen.add(st)
            return _original(st, ch)

        table._step = step
        try:
            table.allowed(state)
        finally:
            table._step = original
        stats["distinct_edge_states"] = len(seen)
        out.append(stats)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"))
    parser.add_argument("--out", default="mask.json")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scan-states", type=int, default=5,
                        help="states also checked against the full-vocabulary sweep")
    args = parser.parse_args()

    tok, texts = grammar_cost.vocab_texts(args.pack)
    grammar = grammar_from_schema(grammar_cost.SCHEMA)
    states = states_for(grammar, tok, texts, grammar_cost.DOC)
    if states is None:
        raise SystemExit("the grammar refused the fixture document - not a measurement")

    skip = grammar._vocab_table(texts)
    exact = exact_table(grammar, texts)
    arms = {"skip": skip, "exact": exact}

    identity = {}
    for name, table in arms.items():
        identity[name] = all(table.allowed(state).tolist() == exact.allowed(state).tolist()
                             for state in states)
    scan_ok, scan_checked = True, 0
    for state in states[:max(0, args.scan_states)]:
        want = grammar._vocab_allowed_scan(state, texts).tolist()
        scan_checked += 1
        for table in arms.values():
            if table.allowed(state).tolist() != want:
                scan_ok = False

    times = {name: [[] for _ in states] for name in arms}
    pass_totals = {name: [] for name in arms}
    for _ in range(max(1, args.rounds)):
        for name in arms:
            pass_totals[name].append(0.0)
        for i, state in enumerate(states):
            for name, table in arms.items():
                t0 = time.perf_counter()
                table.allowed(state)
                dt = (time.perf_counter() - t0) * 1e3
                times[name][i].append(dt)
                pass_totals[name][-1] += dt

    medians = {name: [round(statistics.median(row), 3) for row in rows]
               for name, rows in times.items()}
    totals = {name: round(sum(row), 1) for name, row in medians.items()}
    counts = {name: counting_pass(table, states) for name, table in arms.items()}
    summary = {
        "host": "i7-8086K, Windows, Python 3.13, CPU engine, no GPU",
        "grammar": "json_schema (tools/grammar_cost.SCHEMA), 22-token document",
        "vocabulary": len(texts),
        "misses": len(states),
        "rounds": max(1, args.rounds),
        "median_ms": {name: round(statistics.median(row), 3) for name, row in medians.items()},
        "total_ms": totals,
        "speedup_median_miss": round(statistics.median(medians["exact"])
                                     / max(1e-9, statistics.median(medians["skip"])), 3),
        "speedup_total": round(totals["exact"] / max(1e-9, totals["skip"]), 3),
        "steps_per_miss_median": {name: statistics.median(c["step_calls"] for c in rows)
                                  for name, rows in counts.items()},
        "edge_states_per_miss_median": {name: statistics.median(c["distinct_edge_states"] for c in rows)
                                        for name, rows in counts.items()},
        "masks_identical_to_exact": identity["skip"] and identity["exact"],
        "full_scan_identical": scan_ok,
        "full_scan_states": scan_checked,
        "per_state_ms": medians,
        "modes": sorted({state[0] for state in states}),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("misses=%d  median skip=%5.3f ms exact=%5.3f ms  total %6.1f vs %6.1f ms  "
          "speedup %.2fx  identical=%s  full-scan=%s"
          % (summary["misses"], summary["median_ms"]["skip"], summary["median_ms"]["exact"],
             summary["total_ms"]["skip"], summary["total_ms"]["exact"],
             summary["speedup_median_miss"], summary["masks_identical_to_exact"],
             summary["full_scan_identical"]))


if __name__ == "__main__":
    main()
