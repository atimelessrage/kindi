# Evidence claims

A number in a doc is a claim about a run. This directory makes that claim explicit and checkable:
one `claims.json` per measured gate, a checker that re-derives every number from the run's tracked
artifact, and a test that fails when a doc and its measurement part ways.

```sh
python tools/check_claims.py          # exit 1 with every problem, exit 0 with a coverage line
python tools/check_claims.py --list   # gate -> measured date and number ids
python -m pytest tests/test_claims.py -q
```

`docs/BACKENDS.md`, `docs/ROADMAP.md` and `CURRENT-VS-KINDI-0.6.md` are the docs under this rule
(`REGISTRY.json` lists them). The checker enforces three things for every one of them:

1. **The quoted text is still there.** Each number names the exact `display` string and the docs that
   quote it. Editing a number in the prose without updating the claim fails with
   "`docs/ROADMAP.md` no longer quotes `471.7`".
2. **The artifact still says so.** The same number is re-read from the run's evidence (a JSON row, a
   CSV cell, a ratio of two cells, a max relative gap between two files, or a verbatim needle in the
   run's `RESULTS.md`) and compared with the quoted figure at the doc's own rounding (0.2% by default).
3. **The evidence is tracked.** A claim may not rest on gitignored `bench-logs/` output: a number
   nobody can reproduce is not evidence. Runs that only exist locally stay out of the registry and the
   doc says so.

A fourth rule keeps the registry from being optional: **every `docs/benchmarks/<dir>` the three docs
cite must be claimed, or listed under `unclaimed` in `REGISTRY.json` with a reason.** A new citation
with no claim and no entry fails the check, so evidence cannot appear in a doc without an owner.

## One file per gate

```json
{
  "gate": "kld-2b",
  "title": "Lossy-cache quality gate (KLD) on the 2B, q8_0 vs k8v4",
  "measured": "2026-11-14",
  "commit": "7c456bb",
  "command": "python -m kindi kld (plain-q8, tp-q8, tp-k8v4 at ctx 2048 and 8192)",
  "notes": "why this run exists and what it decided",
  "evidence": ["docs/benchmarks/.../kld-2b.json", "docs/benchmarks/.../RESULTS.md"],
  "numbers": [
    {"id": "q8-2k", "display": "2.74e-07", "quoted_in": ["docs/ROADMAP.md"],
     "check": {"kind": "json", "file": "docs/benchmarks/.../kld-2b.json",
               "pointer": "/1/kld_nats", "expect": 2.74e-07}}
  ]
}
```

Required top-level keys: `gate`, `title`, `measured`, `commit`, `command`, `evidence`, `numbers`.
Required per number: `id`, `display`, `quoted_in`, `check` (with `expect` unless the kind is `text`).
Optional: `notes`, and per number `tolerance` (relative, default `0.002`).

| `check.kind` | recomputes | extra fields |
|---|---|---|
| `json` | one JSON value | `file`, `pointer` (e.g. `/1/kld_nats`), `expect` |
| `csv` | one cell of the row matching `where` | `file`, `where`, `field`, `expect` |
| `ratio` | `a` / `b` | `a`, `b` (`{file, pointer}`), `expect` |
| `compare` | the **largest** relative gap in percent between value pairs, which must stay under `expect` | `pairs` (`[[file_a, pointer_a, file_b, pointer_b], ...]`), `expect` |
| `text` | nothing; the `needle` must appear verbatim in the artifact (whitespace-insensitive) | `file`, `needle` |
| `git` | a figure from a revision range: `commits` (`rev-list --count a..b`), `diffstat` (`files`/`insertions`/`deletions`), `deleted` (files removed), `treefiles` (tracked files at one revision) | `op`, `revs` or `rev`, optional `field`/`path`/`match`, `expect` |

`text` is for runs whose evidence is prose. It is the weakest kind: it proves the page still records
the figure, not that the figure was recomputed. Prefer a JSON or CSV row when the run wrote one.

`git` is for claims about the repository itself: a tag's diffstat, commits since a tag, tracked-file
counts at a commit. There is no artifact to read - the tag is the evidence - so the claim names it in
`evidence` as `ref:v0.6.1`, which the checker resolves to a commit; a moved or deleted tag fails as
missing evidence. `tools/check_claims.py` gates `CURRENT-VS-KINDI-0.6.md` this way.

## Adding or changing a gate

1. The run writes its artifact under `docs/benchmarks/<date>-<host>/` and the page records the
   command, the host and what was not verified.
2. Add `docs/claims/<gate>.json` quoting only numbers the page actually supports, and only numbers a
   doc actually uses. A number the docs do not quote does not need a claim.
3. Add the docs to the number's `quoted_in` and run the checker. A failure is either a wrong needle or
   a doc that overstated the run: fix whichever is wrong, not the checker.
4. A new measurement that supersedes an old one rewrites the claim and updates every `quoted_in` doc
   in the same commit. That is the point: the doc cannot outlive the run.

Retiring a gate is a docs decision, not a delete: drop the citations from the docs, then move the
file to `docs/development/AGENTS-HISTORY.md`-style history or delete it with the reason in the commit.

## Coverage today

21 gates, 116 numbers. Four citations are listed as debt in `REGISTRY.json` with the reason each one
cannot be claimed yet (a page that keeps the breakdown but not the cell the doc quotes, a page that
adds provenance rather than a measurement, architecture constants, and a summary line the run's page
never states). `python tools/check_claims.py` prints the debt count on every run, so the gap is
visible instead of forgotten.

The rule is deliberately narrow: it covers these three docs and the gates behind them. A number in
another page — `docs/AUTOTUNE.md`, `docs/FP8-BF16.md`, `docs/BENCHMARKS.md` — is still prose. Widening
the `docs` list in `REGISTRY.json` is what extends the harness; the checker does the rest.
