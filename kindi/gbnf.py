# SPDX-License-Identifier: AGPL-3.0-only
"""GBNF grammars for constrained decoding (REQ-0018), as a character automaton.

``Grammar.from_string(text)`` parses llama.cpp's GBNF and exposes the same shape as
``jsongrammar.JsonGrammar`` — ``start()``, ``done(state)``, ``advance(state, text)`` and
``vocab_allowed(state, texts)`` — so the server's constrained sampler drives either one.

Supported: rules ``name ::= body`` (``root`` is the start rule), alternation ``|``,
sequences, grouping ``(...)``, literals ``"..."`` with C-style escapes, character
classes ``[a-z]`` / ``[^...]`` with ranges and escapes, ``.`` (any character),
repetition ``* + ? {m} {m,} {m,n}``, rule references, comments ``#`` and blank lines.

Not supported in this slice: a recursive rule (one that reaches itself). That is a
pushdown language, not a regular one, so the parser refuses it with the cycle's name
instead of quietly accepting a truncated language. Recursive JSON is what
``response_format: json_schema`` is for. Every refusal is a ``ValueError`` at compile
time, which the server turns into an enveloped 400 before any token is generated.

The automaton is forward-complete: a token is only accepted when the state it reaches
can still reach the accepting state, so constrained decoding never paints itself into
a dead end that would leave the allowed set empty.
"""
import collections
import os

# A grammar is compiled into one state per literal character, so an unbounded repetition or a
# megabyte literal is a memory request from whoever sent the request. Both limits are refusals
# (ValueError -> an enveloped 400), not silent truncation.
MAX_REPEAT = 10000
DEFAULT_MAX_STATES = 20000


def _state_limit():
    """KINDI_GBNF_MAX_STATES (read per grammar, so an operator can raise it without a restart of
    the module's constants). A bad value falls back to the default rather than failing a request."""
    try:
        return max(1, int(os.environ.get("KINDI_GBNF_MAX_STATES", str(DEFAULT_MAX_STATES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_STATES


# --- parsing ---------------------------------------------------------------

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
            "0": "\0", "[": "[", "]": "]", "-": "-", ".": "."}


class _Parser(object):
    """Recursive-descent parser for the GBNF subset above. Raises ValueError."""

    def __init__(self, text):
        self.s = text if isinstance(text, str) else ""
        self.i = 0
        self.n = len(self.s)
        self.rules = {}
        self.order = []

    def error(self, msg):
        line = self.s.count("\n", 0, self.i) + 1
        raise ValueError("GBNF line %d: %s" % (line, msg))

    # whitespace and comments

    def skip_inline(self):
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r":
                self.i += 1
            elif c == "#":
                while self.i < self.n and self.s[self.i] != "\n":
                    self.i += 1
            else:
                return

    def skip_lines(self):
        while True:
            self.skip_inline()
            if self.i < self.n and self.s[self.i] == "\n":
                self.i += 1
            else:
                return

    # grammar

    def parse(self):
        while True:
            self.skip_lines()
            if self.i >= self.n:
                break
            name = self.name()
            self.skip_inline()
            if not self.s.startswith("::=", self.i):
                self.error("expected '::=' after rule name %r" % name)
            self.i += 3
            body = self.alternation()
            self.skip_inline()
            if self.i < self.n and self.s[self.i] != "\n":
                self.error("unexpected %r after the body of %r" % (self.s[self.i], name))
            if name in self.rules:
                self.error("rule %r is defined twice" % name)
            self.rules[name] = body
            self.order.append(name)
        if not self.rules:
            raise ValueError("GBNF: no rules")
        if "root" not in self.rules:
            raise ValueError("GBNF: no 'root' rule (llama.cpp starts the grammar at 'root')")
        return self.rules

    def name(self):
        start = self.i
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        if start == self.i:
            self.error("expected a rule name or a reference")
        if self.s[start].isdigit():
            self.error("a name cannot start with a digit")
        return self.s[start:self.i]

    def alternation(self):
        branches = [self.sequence()]
        self.skip_inline()
        while self.i < self.n and self.s[self.i] == "|":
            self.i += 1
            branches.append(self.sequence())
            self.skip_inline()
        return branches

    def sequence(self):
        items = []
        while True:
            self.skip_inline()
            if self.i >= self.n or self.s[self.i] in "|\n)":
                return items
            items.append(self.item())

    def item(self):
        atom = self.atom()
        while True:
            self.skip_inline()
            if self.i >= self.n:
                return atom
            c = self.s[self.i]
            if c == "*":
                self.i += 1
                atom = ("rep", atom, 0, None)
            elif c == "+":
                self.i += 1
                atom = ("rep", atom, 1, None)
            elif c == "?":
                self.i += 1
                atom = ("rep", atom, 0, 1)
            elif c == "{":
                lo, hi = self.braces()
                atom = ("rep", atom, lo, hi)
            else:
                return atom

    def atom(self):
        c = self.s[self.i]
        if c == "(":
            self.i += 1
            alt = self.alternation()
            self.skip_inline()
            if self.i >= self.n or self.s[self.i] != ")":
                self.error("expected ')'")
            self.i += 1
            return ("group", alt)
        if c == '"':
            return ("lit", self.literal())
        if c == "[":
            ranges, negated = self.charclass()
            return ("range", ranges, negated)
        if c == ".":
            self.i += 1
            return ("any",)
        if c.isalnum() or c == "_":
            return ("ref", self.name())
        self.error("unexpected %r" % c)

    def literal(self):
        self.i += 1                                   # opening quote
        out = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == '"':
                self.i += 1
                return "".join(out)
            if c == "\n":
                self.error("unterminated string literal")
            if c == "\\":
                out.append(self.escape())
            else:
                self.i += 1
                out.append(c)
        self.error("unterminated string literal")

    def escape(self):
        self.i += 1                                   # backslash
        if self.i >= self.n:
            self.error("trailing backslash")
        c = self.s[self.i]
        if c == "x":
            h = self.s[self.i + 1:self.i + 3]
            if len(h) != 2 or any(d not in "0123456789abcdefABCDEF" for d in h):
                self.error("\\x needs two hex digits")
            self.i += 3
            return chr(int(h, 16))
        if c == "u":
            h = self.s[self.i + 1:self.i + 5]
            if len(h) != 4 or any(d not in "0123456789abcdefABCDEF" for d in h):
                self.error("\\u needs four hex digits")
            self.i += 5
            return chr(int(h, 16))
        self.i += 1
        return _ESCAPES.get(c, c)

    def charclass(self):
        self.i += 1                                   # opening bracket
        negated = self.i < self.n and self.s[self.i] == "^"
        if negated:
            self.i += 1
        ranges = []
        while self.i < self.n:
            if self.s[self.i] == "]":
                self.i += 1
                if not ranges:
                    self.error("empty character class")
                return tuple(ranges), negated
            lo = self.class_char()
            if (self.i + 1 < self.n and self.s[self.i] == "-"
                    and self.s[self.i + 1] != "]"):
                self.i += 1
                hi = self.class_char()
                if ord(hi) < ord(lo):
                    self.error("character class range is reversed")
                ranges.append((ord(lo), ord(hi)))
            else:
                ranges.append((ord(lo), ord(lo)))
        self.error("unterminated character class")

    def class_char(self):
        if self.i >= self.n:
            self.error("unterminated character class")
        if self.s[self.i] == "\\":
            return self.escape()
        c = self.s[self.i]
        self.i += 1
        return c

    def braces(self):
        self.i += 1                                   # opening brace

        def number():
            start = self.i
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            if start == self.i:
                self.error("expected a number inside {...}")
            return int(self.s[start:self.i])

        lo = number()
        hi = lo
        if self.i < self.n and self.s[self.i] == ",":
            self.i += 1
            hi = None if (self.i < self.n and self.s[self.i] == "}") else number()
        if self.i >= self.n or self.s[self.i] != "}":
            self.error("expected '}'")
        self.i += 1
        if hi is not None and hi < lo:
            self.error("{m,n} needs n >= m")
        if hi is not None and hi > MAX_REPEAT:
            self.error("{m,n} expands to at most %d copies (%d requested); narrower bounds or a "
                       "rule reference with '*' is cheaper" % (MAX_REPEAT, hi))
        if lo > MAX_REPEAT:
            self.error("{m} expands to at most %d copies (%d requested)" % (MAX_REPEAT, lo))
        return lo, hi


# --- the automaton ---------------------------------------------------------

class _NFA(object):
    """Thompson construction over labels ("ch", char) / ("range", ...) / ("any",)."""

    def __init__(self, rules, max_states=None):
        self.program = rules
        self.edges = {}
        self.count = 0
        self.built = {}
        self.active = []
        self.max_states = _state_limit() if max_states is None else int(max_states)

    def state(self):
        if self.count >= self.max_states:
            raise ValueError(
                "GBNF: grammar too large (%d automaton states, limit %d); shorten a literal or a "
                "repetition count, or use response_format json_schema "
                "(KINDI_GBNF_MAX_STATES raises the limit)" % (self.count, self.max_states))
        s = self.count
        self.count += 1
        self.edges[s] = []
        return s

    def edge(self, a, label, b):
        self.edges[a].append((label, b))

    def build_rule(self, name):
        if name in self.built:
            return self.built[name]
        if name in self.active:
            cycle = " -> ".join(self.active + [name])
            raise ValueError(
                "GBNF: recursive rule (%s); this slice compiles regular grammars only - "
                "use response_format json_schema for recursive JSON" % cycle)
        if name not in self.program:
            raise ValueError("GBNF: unknown rule %r" % name)
        self.active.append(name)
        pair = self.build_alt(self.program[name])
        self.active.pop()
        self.built[name] = pair
        return pair

    def build_alt(self, branches):
        start, end = self.state(), self.state()
        for seq in branches:
            s, e = self.build_seq(seq)
            self.edge(start, ("eps",), s)
            self.edge(e, ("eps",), end)
        return start, end

    def build_seq(self, items):
        start = end = self.state()
        for item in items:
            s, e = self.build_item(item)
            self.edge(end, ("eps",), s)
            end = e
        return start, end

    def build_item(self, item):
        kind = item[0]
        if kind == "lit":
            start = end = self.state()
            for ch in item[1]:
                nxt = self.state()
                self.edge(end, ("ch", ch), nxt)
                end = nxt
            return start, end
        if kind == "range":
            start, end = self.state(), self.state()
            self.edge(start, ("range", item[1], item[2]), end)
            return start, end
        if kind == "any":
            start, end = self.state(), self.state()
            self.edge(start, ("any",), end)
            return start, end
        if kind == "ref":
            return self.build_rule(item[1])
        if kind == "group":
            return self.build_alt(item[1])
        if kind == "rep":
            return self.build_rep(item[1], item[2], item[3])
        raise ValueError("GBNF: unknown node %r" % (kind,))

    def build_rep(self, sub, lo, hi):
        start = end = self.state()
        for _ in range(lo):
            s, e = self.build_item(sub)
            self.edge(end, ("eps",), s)
            end = e
        if hi is None:
            s, e = self.build_item(sub)
            nxt = self.state()
            self.edge(end, ("eps",), s)
            self.edge(e, ("eps",), s)                 # loop
            self.edge(end, ("eps",), nxt)
            self.edge(e, ("eps",), nxt)
            return start, nxt
        for _ in range(hi - lo):
            s, e = self.build_item(sub)
            nxt = self.state()
            self.edge(end, ("eps",), s)
            self.edge(e, ("eps",), nxt)
            self.edge(end, ("eps",), nxt)             # skip this copy
            end = nxt
        return start, end


def _accepts(label, ch):
    kind = label[0]
    if kind == "ch":
        return ch == label[1]
    if kind == "any":
        return True
    if kind == "range":
        cp = ord(ch)
        hit = any(lo <= cp <= hi for lo, hi in label[1])
        return (not hit) if label[2] else hit
    return False


class Grammar(object):
    """A parsed GBNF grammar with the same interface as ``jsongrammar.JsonGrammar``."""

    def __init__(self, rules, start_rule="root"):
        nfa = _NFA(rules)
        self._nfa = nfa
        self._start, self._accept = nfa.build_rule(start_rule)
        self._productive = self._productive_states()
        self._closures = {}
        self._vocab_cache = {}

    @classmethod
    def from_string(cls, text):
        """Parse GBNF text. Raises ValueError on syntax errors, unknown or recursive rules."""
        return cls(_Parser(text).parse())

    # --- states ---------------------------------------------------------

    def _closure(self, states):
        cached = self._closures.get(states)
        if cached is not None:
            return cached
        out = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for label, target in self._nfa.edges[s]:
                if label[0] == "eps" and target not in out:
                    out.add(target)
                    stack.append(target)
        closed = frozenset(out)
        self._closures[states] = closed
        return closed

    def _productive_states(self):
        """Every NFA state from which the accepting state is still reachable."""
        reverse = collections.defaultdict(list)
        for src, edges in self._nfa.edges.items():
            for _, target in edges:
                reverse[target].append(src)
        seen = set([self._accept])
        stack = [self._accept]
        while stack:
            s = stack.pop()
            for prev in reverse[s]:
                if prev not in seen:
                    seen.add(prev)
                    stack.append(prev)
        return seen

    def start(self):
        return self._closure(frozenset([self._start]))

    def done(self, state):
        return state is not None and self._accept in state

    def advance(self, state, text):
        """The state reached by ``text``, or None when the text leaves the language."""
        if state is None:
            return None
        for ch in (text or ""):
            state = self._step(state, ch)
            if state is None:
                return None
        return state

    def _step(self, state, ch):
        out = set()
        for s in state:
            for label, target in self._nfa.edges[s]:
                if label[0] != "eps" and _accepts(label, ch):
                    out.add(target)
        if not out:
            return None
        closed = self._closure(frozenset(out))
        if not (closed & self._productive):
            return None                                # dead end: the language cannot finish
        return closed

    def vocab_allowed(self, state, texts):
        """Token ids whose decoded text keeps ``state`` alive (cached per state)."""
        import numpy as np

        cached = self._vocab_cache.get(state)
        if cached is not None:
            return cached
        ids = np.fromiter((i for i, t in enumerate(texts) if self.advance(state, t) is not None),
                          dtype=np.int64)
        self._vocab_cache[state] = ids
        return ids


def parse(text):
    """The rule table of a GBNF string (mostly useful in tests and error messages)."""
    return _Parser(text).parse()


__all__ = ["Grammar", "parse"]
