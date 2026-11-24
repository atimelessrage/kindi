# SPDX-License-Identifier: AGPL-3.0-only
"""Every documented speculation knob must change an effective setting, or say why it cannot (REQ-0026/0018).

The bug this guards is not hypothetical: for four releases `/health` answered `enabled: true, method:
mtp` from `KINDI_SPEC` alone while the MTP head's graph never built, and every gate stayed green. A
documented knob whose value reaches nothing is indistinguishable from a working one to a reader.

The gate is three parts:
  1. every knob here is documented in the generated `docs/CONFIGURATION-REFERENCE.md`;
  2. every knob here is read somewhere in `kindi/` (a rename or a move to a doc string fails);
  3. setting it changes a value the code computes, not a value the test computed.
Part 4 is the coverage half: every documented speculation knob must appear in LIVE or PARKED, so a new
`KINDI_SPEC_*` cannot be documented and forgotten.

PARKED rows are not exempt from (1)-(2); they record why the effect cannot be observed on a CPU-only
host (a card, the vendored qwen4exp engine, or a whole model build), so the reason is visible instead
of the row quietly disappearing.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "docs", "CONFIGURATION-REFERENCE.md")


def _documented():
    text = open(REFERENCE, encoding="utf-8").read()
    return set(re.findall(r"KINDI_[A-Z0-9_]+", text))


def _readers(knob):
    """Files in kindi/ that read the knob as a string literal (a doc mention is not a read)."""
    hits = []
    for name in sorted(os.listdir(os.path.join(ROOT, "kindi"))):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(ROOT, "kindi", name), encoding="utf-8", errors="replace").read()
        # both quote styles, and the f-string/%-style forms used for the report strings
        if ('"%s"' % knob) in text or ("'%s'" % knob) in text:
            hits.append(name)
    return hits


# --- the live probes: each returns what the code decided, with the knob set ----------------------
def _engine(**kw):
    import types
    eng = types.SimpleNamespace(spec_begin=lambda *a, **k: None, drafter=None, dspine=None,
                                mtp=object(), _mtp_heads={}, W=1)
    eng.__dict__.update(kw)
    return eng


def _probe_spec(eng, monkeypatch):
    from kindi import spec
    return spec.resolve_mode(eng, os.environ.get("KINDI_SPEC", "0"))[0]


def _probe_spec_k(eng, monkeypatch):
    from kindi import spec
    spec.spec_begin(eng, True)
    return eng.spec_k


def _probe_spec_sample(eng, monkeypatch):
    from kindi import spec
    spec.spec_begin(eng, False, temperature=0.7, top_p=0.8, top_k=20, rng=object())
    return eng.spec_sp is not None


def _probe_dspine_k(eng, monkeypatch):
    from kindi import spec
    eng.dspine = object()                      # an attached block drafter: resolve_mode then picks dspine
    spec.spec_begin(eng, True)
    return eng.spec_k


def _probe_dspine(eng, monkeypatch):
    from kindi import dspine
    return dspine.wanted(eng)


def _probe_mtp_head(eng, monkeypatch):
    from kindi import mtp_heads
    return mtp_heads.report(spec="mtp")


# knob -> (default, trial, probe, what the trial must produce)
LIVE = {
    "KINDI_SPEC": ("0", "mtp", _probe_spec, "mtp"),
    "KINDI_SPEC_K": ("4", "2", _probe_spec_k, 2),
    "KINDI_SPEC_SAMPLE": ("1", "0", _probe_spec_sample, False),
    "KINDI_DSPINE": (None, "/nonexistent-dspine", _probe_dspine, True),
    "KINDI_DSPINE_K": ("7", "3", _probe_dspine_k, 3),
    # the exact wording is the loader's; the gate only needs the decision to change, which the
    # second assertion in the test enforces (documented head vs a refused path)
    "KINDI_MTP_HEAD": (None, "/nonexistent/head.npz", _probe_mtp_head, "REFUSED"),
}

# knob -> why its effect cannot be observed on this host (the row is still checked for (1) and (2))
# live knobs the generated reference does not list: a real gap, pinned so it stays visible. The
# generator derives its table from a fixed set of read sites; these are read elsewhere.
UNDOCUMENTED = {
    "KINDI_MTP_HEAD": "read by mtp_heads.check/report and spec's head lookup, documented by hand in "
                      "docs/MTP.md; `kindi.config_reference` does not list it",
}

PARKED = {
    "KINDI_SPEC_PACK": "builds a second TPEngine for the drafter; needs a card and a 2B pack",
    "KINDI_SPEC_CTX": "the drafter's context, read in the same engine build as KINDI_SPEC_PACK",
    "KINDI_SPEC_WINDOW": "the drafter's context fallback, same build",
    "KINDI_SPEC_DRAFT_HEAD": "needs the drafter build plus its head file",
    "KINDI_MTP_HEAD_ALLOW_UNREGISTERED": "covered live in tests/test_mtp_head.py against a real file",
    "KINDI_DSPINE_MB": "the drafter's VRAM reservation; read by reserve_mb() during a harness build",
    "KINDI_DSPINE_NARROW": "tries narrow products in the block drafter's own kernels; needs a card",
    "KINDI_SPEC_B2_MAX": "caps the 2B head's extension on greedy requests; read while a drafter runs",
    "KINDI_SPEC_B0_K2_MAX": "the guard's k=2 ceiling for the first window; runs inside a card request",
    "KINDI_SPEC_MTP_K2_MAX": "the same ceiling for mode A; runs inside a card request",
    "KINDI_SPEC_GUARD": "enables the speed guard, which only runs once a speculating engine steps",
    "KINDI_SPEC_GUARD_FROM": "the guard's first window; needs a step to reach the threshold",
    "KINDI_SPEC_GUARD_LOG": "prints the guard's windows; observed in a card run, not a unit test",
    "KINDI_SPEC_GUARD_BOUND": "enables the guard's bounded probe; needs a stepping engine",
    "KINDI_SPEC_GUARD_BOUND_DEPTH": "the bounded probe's depth; same run as GUARD_BOUND",
    "KINDI_SPEC_KNOWN": "the prompt-cache key's speculation tag, read in cache.py on a warm hit",
}

PREFIXES = ("KINDI_SPEC", "KINDI_DSPINE")
EXTRA = ("KINDI_MTP_HEAD", "KINDI_MTP_HEAD_ALLOW_UNREGISTERED")


def _in_scope(knob):
    return knob.startswith(PREFIXES) or knob in EXTRA


def test_every_live_knob_is_documented_and_read():
    docs = _documented()
    for knob in LIVE:
        if knob not in UNDOCUMENTED:
            assert knob in docs, "%s is in the liveness table but not documented" % knob
        assert _readers(knob), "%s has no reader in kindi/ (renamed, moved into prose, or dead)" % knob


def test_every_parked_knob_is_documented_and_read():
    docs = _documented()
    for knob in PARKED:
        assert knob in docs, "%s is parked but not documented" % knob
        assert _readers(knob), "%s has no reader in kindi/ (renamed, moved into prose, or dead)" % knob


def test_setting_a_live_knob_changes_what_the_code_decides(monkeypatch):
    """The strong half: set the knob, ask the code, and the answer differs from the default's."""
    from kindi import spec  # noqa: F401 - imported once; the probes reuse it
    for knob, (default, trial, probe, want) in sorted(LIVE.items()):
        monkeypatch.delenv(knob, raising=False)
        if default is not None:
            monkeypatch.setenv(knob, default)
        # the other rows need a mode selected; the KINDI_SPEC row is the mode, so it sets nothing else
        if knob in ("KINDI_SPEC_K", "KINDI_SPEC_SAMPLE", "KINDI_DSPINE_K", "KINDI_DSPINE"):
            monkeypatch.setenv("KINDI_SPEC", "dspine" if knob.startswith("KINDI_DSPINE") else "mtp")
        got_default = probe(_engine(), monkeypatch)
        monkeypatch.setenv(knob, trial)
        got_trial = probe(_engine(), monkeypatch)
        if isinstance(want, str):
            assert want in got_trial, "%s=%s did not reach the decision (got %r, wanted %r)" % (
                knob, trial, got_trial, want)
        else:
            assert got_trial == want, "%s=%s did not reach the decision (got %r, wanted %r)" % (
                knob, trial, got_trial, want)
        assert got_default != got_trial, "%s is inert: both values decide the same thing (%r)" % (
            knob, got_default)
        monkeypatch.delenv("KINDI_SPEC", raising=False)


def test_every_documented_speculation_knob_is_classified():
    """Part 4: the coverage half. A new documented speculation knob must be live or parked here."""
    documented = {k for k in _documented() if _in_scope(k)}
    classified = (set(LIVE) | set(PARKED) | set(UNDOCUMENTED))
    missing = sorted(documented - classified)
    assert not missing, ("documented speculation knobs with no liveness row: %s - add a probe "
                         "or a PARKED reason" % ", ".join(missing))


def test_the_table_does_not_list_knobs_the_reference_dropped():
    """A row for a knob nobody documents any more is dead weight, and usually a stale rename."""
    documented = _documented()
    stale = sorted(((set(LIVE) | set(PARKED) | set(UNDOCUMENTED)) - documented) - set(UNDOCUMENTED))
    assert not stale, "liveness rows for knobs the reference no longer documents: %s" % ", ".join(stale)
