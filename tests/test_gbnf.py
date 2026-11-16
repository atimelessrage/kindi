# SPDX-License-Identifier: AGPL-3.0-only
"""GBNF grammars for constrained decoding (REQ-0018).

The automaton is tested on its own (no model, no GPU): the language it accepts, the
dead-end rule that keeps the sampler's allowed set non-empty, and the parse errors the
server turns into an enveloped 400.
"""
import pytest

from kindi.gbnf import Grammar, parse


def _accepts(grammar, text):
    return grammar.done(grammar.advance(grammar.start(), text))


def test_literal_alternation():
    g = Grammar.from_string('root ::= "yes" | "no"')
    assert _accepts(g, "yes") and _accepts(g, "no")
    assert not _accepts(g, "ye")
    assert not _accepts(g, "yess")


def test_sequence_group_and_repetition():
    g = Grammar.from_string('root ::= ("ab")+ "c"?')
    assert _accepts(g, "ab")
    assert _accepts(g, "ababc")
    assert _accepts(g, "abc")
    assert not _accepts(g, "a")


def test_character_classes_and_negation():
    g = Grammar.from_string("root ::= [a-c]+ [^0-9]")
    assert _accepts(g, "abcx")
    assert not _accepts(g, "abc1")


def test_bounded_and_open_counts():
    bounded = Grammar.from_string('root ::= "a"{2,3}')
    assert not _accepts(bounded, "a")
    assert _accepts(bounded, "aa") and _accepts(bounded, "aaa")
    assert not _accepts(bounded, "aaaa")
    open_ended = Grammar.from_string('root ::= "a"{2,}')
    assert _accepts(open_ended, "a" * 5)


def test_escapes_comments_and_hex():
    g = Grammar.from_string('# a comment\nroot ::= "\\n" [\\t] "\\x41"')
    assert _accepts(g, "\n\tA")


def test_rule_references():
    g = Grammar.from_string(
        'root ::= greeting " " name\ngreeting ::= "hi" | "hello"\nname ::= [^ ]+')
    assert _accepts(g, "hello Ada")
    assert not _accepts(g, "hi")


def test_dot_matches_any_character():
    g = Grammar.from_string('root ::= . "!"')
    assert _accepts(g, "x!")
    assert not _accepts(g, "!")


def test_empty_body_is_the_empty_string():
    g = Grammar.from_string('root ::= ""')
    assert _accepts(g, "")


def test_a_dead_end_is_refused_before_the_sampler_reaches_it():
    """A live state that can no longer accept must not keep the allowed set open:
    otherwise the sampler picks a token the grammar can never finish."""
    g = Grammar.from_string('root ::= "ab" | "ac"')
    after_a = g.advance(g.start(), "a")
    assert after_a is not None
    assert g.advance(after_a, "x") is None


def test_vocab_allowed_keeps_only_tokens_inside_the_language():
    g = Grammar.from_string('root ::= "yes" | "no"')
    texts = ["y", "ye", "yes", "n", "no", "nope", "x"]
    ids = set(int(i) for i in g.vocab_allowed(g.start(), texts).tolist())
    assert ids == {0, 1, 2, 3, 4}


def test_parse_returns_the_rule_table():
    rules = parse('root ::= a\na ::= "x"')
    assert set(rules) == {"root", "a"}


def test_errors_name_the_problem():
    with pytest.raises(ValueError, match="root"):
        Grammar.from_string('start ::= "a"')
    with pytest.raises(ValueError, match="unknown rule"):
        Grammar.from_string('root ::= missing')
    with pytest.raises(ValueError, match="recursive"):
        Grammar.from_string('root ::= "(" root ")" | "x"')
    with pytest.raises(ValueError, match="GBNF line 2"):
        Grammar.from_string('root ::= "a"\nbogus')
    with pytest.raises(ValueError, match="unterminated"):
        Grammar.from_string('root ::= "a')
