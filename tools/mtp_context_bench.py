# SPDX-License-Identifier: AGPL-3.0-only
"""Mode A at depth: plain decoding against `KINDI_SPEC=mtp` on one built engine.

`kindi bench` times single steps and `kindi verify` runs a 1K-context matrix, so neither answers
"does the MTP head pay at 8K/24K?". This harness builds the engine once (with the head attached),
then for each context position runs the same prompt and the same greedy loop twice - plain, then
mode A - switching only `KINDI_SPEC` and `spec_begin`, so the A/B shares weights, prompt and
kernel caches and the numbers differ only by the mode.

Generation time excludes the prefill: the prompt is fed before the clock starts. Reported per run:
generation tok/s over `--tokens` new tokens, and mode A's drafts and accepted count.

    KINDI_TC=1 KINDI_SPEC=mtp KINDI_MTP_HEAD=~/.kindi/models/kindi-mtp-2b-v1.npz \\
      python tools/mtp_context_bench.py --pack ~/.kindi/packs/bonsai-2b \\
      --ctx 24576 --positions 8192,24576 --tokens 128 --runs 2 --json DIR/mtp-context.json

The prompt text is the repository's `docs/*.md` joined in name order unless `--text` names a file;
a real prose prompt matters, because the head's acceptance on repeated or synthetic tokens is not
the acceptance a user's context gets.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # `python tools/...` from the root


def _prompt_text(path=None):
    if path:
        return open(path, encoding="utf-8").read()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = []
    for p in sorted(glob.glob(os.path.join(here, "docs", "*.md"))):
        try:
            text.append(open(p, encoding="utf-8").read())
        except OSError:
            continue
    return "\n".join(text)


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


def _positions(text):
    """Comma-separated positive prompt lengths, e.g. 8192,24576."""
    parts = [part.strip() for part in str(text).split(",")]
    try:
        if not parts or any(not part for part in parts):
            raise ValueError
        values = [int(part) for part in parts]
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated positive integers, e.g. 8192,24576 (got %r)"
            % (text,)) from None
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated positive integers, e.g. 8192,24576 (got %r)"
            % (text,))
    return values


def _generate(eng, ids, n, sync):
    """`tp.generate` without the plain shortcut: the prompt is fed, then `n` steps are timed."""
    eng.reset()
    if hasattr(eng, "feed_batch"):
        eng.feed_batch(ids[:-1], 0)          # the chunked prompt path the server uses (per-token feed is 30x slower)
    else:
        for i, t in enumerate(ids[:-1]):
            eng.feed(t, i)
    pos, out = len(ids), []
    sync()
    t0 = time.perf_counter()
    out.append(eng.step(ids[-1], pos - 1))
    while len(out) < n:
        out.append(eng.step(out[-1], pos))
        pos += 1
    sync()
    return out, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--head", default=os.environ.get("KINDI_MTP_HEAD"),
                    help="the registered MTP head (default: KINDI_MTP_HEAD)")
    ap.add_argument("--ctx", type=_positive_int, default=24576,
                    help="requested context (a positive integer; the loader may reduce it)")
    ap.add_argument("--positions", type=_positions, default="8192,24576",
                    help="comma-separated positive prompt lengths to measure (default: 8192,24576)")
    ap.add_argument("--tokens", type=_positive_int, default=128,
                    help="new greedy tokens per run (a positive integer; default: 128)")
    ap.add_argument("--runs", type=_positive_int, default=2,
                    help="runs per mode and position (a positive integer; default: 2)")
    ap.add_argument("--text", default=None, help="a text file for the prompt (default: docs/*.md)")
    ap.add_argument("--json", default=None, help="also write the rows here")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if not a.head:
        print("mtp_context_bench: no head (--head or KINDI_MTP_HEAD)", file=sys.stderr)
        return 2
    os.environ["KINDI_SPEC"] = "mtp"                 # attached at build; switched per run below
    os.environ["KINDI_MTP_HEAD"] = a.head
    os.environ.setdefault("KINDI_TC", "1")

    import cupy as cp

    from kindi.engine import make_engine
    from kindi.tokenizer import Tok

    eng = make_engine(a.pack, ctx=a.ctx, log=print)
    if getattr(eng, "mtp", None) is None:
        print("mtp_context_bench: the head did not attach; refusing to time the plain engine twice",
              file=sys.stderr)
        return 2
    tok = Tok(eng.vocab, os.path.join(a.pack, "merges.txt"))
    ids = tok.encode(_prompt_text(a.text))
    print("prompt: %d tokens available, ctx %d, %d new tokens, %d runs per mode"
          % (len(ids), eng.ctx, a.tokens, a.runs))
    if len(ids) < max(a.positions):
        print("mtp_context_bench: the prompt text is shorter than the deepest position", file=sys.stderr)
        return 2
    sync = cp.cuda.Device().synchronize
    rows = []
    for pos in a.positions:
        prompt = ids[:pos]
        for run in range(a.runs):
            for mode in ("off", "mtp"):
                os.environ["KINDI_SPEC"] = "mtp" if mode == "mtp" else "0"
                eng.spec_begin(mode == "mtp")
                try:
                    _out, dt = _generate(eng, prompt, a.tokens, sync)
                finally:
                    st = eng.spec_end() or {}
                row = {"position": pos, "run": run + 1, "mode": "mtp" if mode == "mtp" else "off",
                       "tokens": a.tokens, "seconds": round(dt, 3), "tok_s": round(a.tokens / dt, 1),
                       "drafted": int(st.get("drafted", 0)), "accepted": int(st.get("accepted", 0))}
                rows.append(row)
                note = ""
                if row["drafted"]:
                    note = "  %d/%d drafted (%.1f%%)" % (row["accepted"], row["drafted"],
                                                         100.0 * row["accepted"] / row["drafted"])
                print("pos %5d run %d %-3s %6.1f tok/s%s" % (pos, run + 1, row["mode"], row["tok_s"], note))
    # names only: the JSON is committed, and an absolute home path in a tracked artifact is a
    # personal-data hit for the secret-history gate
    out = {"pack": os.path.basename(os.path.normpath(a.pack)), "head": os.path.basename(a.head),
           "ctx": eng.ctx, "rows": rows}
    if a.json:
        parent = os.path.dirname(os.path.abspath(a.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
