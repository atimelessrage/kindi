#!/usr/bin/env python3
"""What one mask miss spends inside `TokenClassTable._trie_allowed`, and what the class memo saves.

The 2026-11-24 refinement left the trie walk as the largest remaining part of a
constrained mask miss. This harness attributes that cost and A/Bs the 2026-11-25
change in one process, so both arms see the same clock, caches and thermal state:

  * `exact` - a table built with no `char_class`: the automaton is asked once per
    child edge, which is what the walk did before the change;
  * `class` - the grammar's own table, whose walk shares a `character -> next
    state` memo between states that the grammar proves accept the same
    characters.

Both tables read the same prewarmed vocabulary index. Each state is timed in
both arms, alternating, with the grammar's per-state mask cache off, so every
state pays its first-visit mask. A counting pass reports how many trie nodes and
automaton steps a miss used, and cProfile attributes the time.

  python docs/benchmarks/2026-11-25-i7-8086k-cpu-trie-walk/profile_walk.py \
      --pack ~/.kindi/packs/bonsai-2b --out walk.json

No GPU and no server: this is the mask's own work.
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import grammar_cost  # noqa: E402

from kindi import token_classes  # noqa: E402


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
    """The same table without a char_class: every edge asks the automaton."""
    return token_classes.TokenClassTable(texts, grammar._step, grammar._token_safe_sets)


def ab_pass(tables, states, rounds=3):
    """Alternating per-state timings for every arm, plus the total of every pass.

    Pass 0 is the cold walk (the class memos are empty); later passes are what a later request
    over the same states costs, which is when a shared edge memo can pay off.
    """
    times = {name: [[] for _ in states] for name in tables}
    pass_totals = {name: [] for name in tables}
    for _ in range(rounds):
        for name in tables:
            pass_totals[name].append(0.0)
        for i, state in enumerate(states):
            for name, table in tables.items():
                t0 = time.perf_counter()
                ids = table.allowed(state)
                dt = (time.perf_counter() - t0) * 1e3
                times[name][i].append(dt)
                pass_totals[name][-1] += dt
                sizes = ids.size
    return ({name: [round(statistics.median(t), 3) for t in rows] for name, rows in times.items()},
            sizes, {name: [round(t, 1) for t in totals] for name, totals in pass_totals.items()})


def counting_pass(table, states):
    """Node and step counts per state, with `_step`, `_chars_under` and `_ids_under` wrapped."""
    counts = []
    original_step, original_chars, original_ids = table._step, table._chars_under, table._ids_under

    def run(state):
        stats = {"step_calls": 0, "chars_under_calls": 0, "ids_under_calls": 0}
        seen_states = set()

        def step(st, ch):
            stats["step_calls"] += 1
            seen_states.add(st)
            return original_step(st, ch)

        def chars(node):
            stats["chars_under_calls"] += 1
            return original_chars(node)

        def ids(node):
            stats["ids_under_calls"] += 1
            return original_ids(node)

        table._step, table._chars_under, table._ids_under = step, chars, ids
        try:
            table.allowed(state)
        finally:
            table._step, table._chars_under, table._ids_under = (
                original_step, original_chars, original_ids)
        stats["distinct_edge_states"] = len(seen_states)
        return stats

    for state in states:
        counts.append(run(state))
    return counts


def profile_pass(table, states, top=8):
    """cProfile tottime of the same walk (times are profiled, so inflated)."""
    pr = cProfile.Profile()
    pr.enable()
    for state in states:
        table.allowed(state)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(top)
    out = {}
    for line in buf.getvalue().splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0] != "ncalls":
            try:
                name = os.path.basename(parts[-1].replace("\\", "/"))
                out[name] = {"tottime_s": float(parts[1]), "calls": int(parts[0].split("/")[0])}
            except ValueError:
                continue
    return out


def measure(label, grammar, tok, texts, doc, rounds=3):
    states = states_for(grammar, tok, texts, doc)
    if states is None:
        print("%s: the grammar refused the document - not a measurement" % label)
        return None, False
    class_table = grammar._vocab_table(texts)
    class_table.allowed(states[0])                    # the served path builds the table first
    arms = {"class": class_table, "exact": exact_table(grammar, texts)}
    times, _, pass_totals = ab_pass(arms, states, rounds)
    counts = counting_pass(class_table, states)
    exact_counts = counting_pass(arms["exact"], states)
    prof = profile_pass(class_table, states)
    med = {name: statistics.median(v) for name, v in times.items()}
    summary = {
        "misses": len(states),
        "median_ms": {name: round(v, 3) for name, v in med.items()},
        "total_ms": {name: round(sum(v), 1) for name, v in times.items()},
        "pass_totals_ms": pass_totals,
        "first_pass_speedup": round(pass_totals["exact"][0] / max(1e-9, pass_totals["class"][0]), 3),
        "warm_pass_speedup": round(statistics.median(pass_totals["exact"][1:] or [1.0])
                                   / max(1e-9, statistics.median(pass_totals["class"][1:] or [1.0])), 3),
        "speedup_total": round(sum(times["exact"]) / max(1e-9, sum(times["class"])), 3),
        "speedup_median_miss": round(med["exact"] / max(1e-9, med["class"]), 3),
        "max_ms": {name: round(max(v), 3) for name, v in times.items()},
        "steps_per_miss_median": statistics.median(c["step_calls"] for c in counts),
        "steps_per_miss_median_exact": statistics.median(c["step_calls"] for c in exact_counts),
        "edge_states_per_miss_median": statistics.median(c["distinct_edge_states"] for c in counts),
        "edge_states_per_miss_median_exact": statistics.median(
            c["distinct_edge_states"] for c in exact_counts),
        "modes": sorted({(state[0] if isinstance(state, tuple) else "gbnf") for state in states}),
        "per_state_ms": {name: v for name, v in times.items()},
        "profile_tottime_class_arm": prof,
    }
    print("%-18s misses=%2d  median class=%5.2f ms exact=%5.2f ms  total %6.1f vs %6.1f ms  "
          "speedup %.2fx" % (label, summary["misses"], summary["median_ms"]["class"],
                             summary["median_ms"]["exact"], summary["total_ms"]["class"],
                             summary["total_ms"]["exact"], summary["speedup_total"]))
    return summary, True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.expanduser("~/.kindi/packs/bonsai-2b"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args(argv)

    tok, texts = grammar_cost.vocab_texts(args.pack)
    token_classes.prewarm(texts)                  # the server prewarms at startup
    rows = {"pack": os.path.basename(args.pack), "vocab": len(texts),
            "doc_tokens": len(tok.encode(grammar_cost.DOC)), "rounds": args.rounds,
            "host": {"cpu": os.cpu_count(), "python": sys.version.split()[0]},
            "grammars": {}}
    ok_all = True
    for label in ("json_schema", "gbnf_record"):
        if label == "json_schema":
            grammar = grammar_cost.grammar_from_schema(grammar_cost.SCHEMA)
        else:
            os.environ["KINDI_GBNF_MAX_DEPTH"] = "1"
            os.environ["KINDI_GBNF_MAX_STATES"] = "200000"
            grammar = grammar_cost.Grammar.from_string(grammar_cost.RECORD_GBNF)
        summary, ok = measure(label, grammar, tok, texts, grammar_cost.DOC, args.rounds)
        ok_all = ok_all and ok
        if summary:
            rows["grammars"][label] = summary
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print("wrote", args.out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
