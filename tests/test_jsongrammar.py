# SPDX-License-Identifier: AGPL-3.0-only
"""JsonGrammar: schema-ordered JSON automaton conformance (REQ-0018, CPU-only)."""
import json
import random

import pytest

from kindi.jsongrammar import JsonGrammar, grammar_from_schema, json_object_grammar

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "addr": {"type": "object", "properties": {"city": {"type": "string"}},
                 "required": ["city"]},
        "mood": {"enum": ["happy", "sad"]},
    },
    "required": ["name", "age", "mood"],
}


def walk(grammar, text):
    st = grammar.start()
    for ch in text:
        st = grammar.advance(st, ch)
        if st is None:
            return None
    return st


def test_valid_document_walks_to_done():
    g = grammar_from_schema(SCHEMA)
    st = walk(g, '{"name": " Ada ", "age": -36, "tags": ["a", "b"], '
                 '"addr": {"city": "London"}, "mood": "happy"}')
    assert st is not None and g.done(st)
    # and it really parses, with the required fields
    parsed = json.loads('{"name": " Ada ", "age": -36, "tags": ["a", "b"], '
                        '"addr": {"city": "London"}, "mood": "happy"}')
    assert parsed["name"] == " Ada "


def test_optionals_can_be_skipped():
    g = grammar_from_schema(SCHEMA)
    assert g.done(walk(g, '{"name": "x\\u00e9", "age": 0, "mood": "sad"}'))
    assert g.done(walk(g, '{"name": "x", "age": 2, "mood": "happy"}'))


@pytest.mark.parametrize("doc", [
    '{"age": 3, "name": "x", "mood": "happy"}',          # required out of order
    '{"name": "x", "mood": "happy"}',                     # missing a required property
    '{"name": "x", "age": 2, "mood": "meh"}',             # enum value not in the set
    '{"name": "x", "age": 1.5, "mood": "happy"}',         # integer needs canonical form
    '{"name": 5, "age": 2, "mood": "happy"}',             # wrong scalar type
    '{"name": "x", "age": 1.2.3, "mood": "happy"}',      # number prefix is dead after 1.2.3
    '{"name": "x", "age": 1e2.3, "mood": "happy"}',      # no fraction after an exponent
    '{"nam": "x", "age": 2, "mood": "happy"}',          # a proper prefix of a key is not the key
    '{"name": "x", "age": 2, "extras": 1, "mood": "happy"}',  # unknown key
    '{"name": "x", "age": 2, "mood": "happy",}',          # trailing comma
])
def test_schema_violations_are_rejected(doc):
    g = grammar_from_schema(SCHEMA)
    assert walk(g, doc) is None


def test_random_schema_instances_walk_and_parse():
    g = grammar_from_schema(SCHEMA)
    rng = random.Random(11)
    for _ in range(40):
        doc = {
            "name": "".join(rng.choice("ab \tλ🎉") for _ in range(rng.randrange(0, 12))),
            "age": rng.randrange(-999, 999),
            "mood": rng.choice(["happy", "sad"]),
        }
        if rng.random() < 0.5:
            doc["tags"] = [rng.choice(["x", "y"]) for _ in range(rng.randrange(0, 3))]
        if rng.random() < 0.5:
            doc["addr"] = {"city": rng.choice(["L", "M"])}
        # reorder to schema order (name, age, tags, addr, mood)
        ordered = {**{k: doc[k] for k in ("name", "age", "tags", "addr", "mood") if k in doc}}
        text = json.dumps(ordered, ensure_ascii=False)
        st = walk(g, text)
        assert st is not None and g.done(st), text
        assert json.loads(text) == ordered


def test_json_object_accepts_nested_generic():
    g = json_object_grammar()
    text = '{"a": 1.5e3, "b": [true, false, null, {"c": "d"}], "e": "x\\n\\u263A"}'
    assert g.done(walk(g, text))
    assert g.done(walk(g, '{"a\\u0041": [1]}'))
    assert g.done(walk(g, '{"a":{"b":{"c":{"d":[]}}}}'))
    assert walk(g, '{"a": 01}') is None
    assert walk(g, '{} x') is None
    assert not g.done(walk(g, '{"a": "unterminated}'))


@pytest.mark.parametrize("schema", [
    {"type": "frob"},
    {"type": "object", "properties": {"x": {"enum": [{"a": 1}]}}},
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["y"]},
    {"properties": "not-a-dict"},
    {"type": "array", "items": "not-a-dict"},
])
def test_invalid_schemas_are_refused(schema):
    with pytest.raises(ValueError):
        grammar_from_schema(schema)


# every char the automaton can ever consume: JSON structure, scalars, the schema's
# keys and enum literals, and a few generic word chars for free-form names
_ALPHABET = set('"{}[],:0123456789tfntruefalsn-.eE)') | set(' \t')
for _name, _sub in SCHEMA["properties"].items():
    _ALPHABET |= set(_name)
    _ALPHABET |= set("".join(_sub.get("enum", [])))
ALPHABET = sorted(_ALPHABET)


def _reachable_states(grammar, limit=3000):
    seen = {grammar.start()}
    frontier = [grammar.start()]
    while frontier and len(seen) < limit:
        st = frontier.pop()
        for ch in ALPHABET:
            nxt = grammar._step(st, ch)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def test_every_reachable_state_is_live():
    """Forward completeness: every reachable state has at least one continuation."""
    for grammar in (grammar_from_schema(SCHEMA), json_object_grammar()):
        for st in _reachable_states(grammar):
            assert any(grammar._step(st, ch) is not None for ch in ALPHABET), st


def test_no_comma_after_the_last_property():
    # after every named property is consumed, ',' is a dead end: it must be rejected
    # at the automaton level (the GPU gate found the empty allowed set this made).
    g = grammar_from_schema(SCHEMA)
    st = walk(g, '{"name": "x", "age": 2, "mood": "happy"')
    assert st is not None
    assert g.advance(st, ",") is None
    assert g.done(walk(g, '{"name": "x", "age": 2, "mood": "happy"}'))


def test_vocab_allowed_caches_and_filters():
    g = grammar_from_schema(SCHEMA)
    texts = ["{", '"', "}", "[", "]", ",", ":", "0", '"name"', '"age"', "x", " "]
    ids = g.vocab_allowed(g.start(), texts)
    assert ids.tolist() == [0, 11]
    again = g.vocab_allowed(g.start(), texts)
    assert again is ids


def test_vocab_allowed_after_open_brace_expects_a_key():
    g = grammar_from_schema(SCHEMA)
    st = walk(g, "{")
    texts = ["{", '"', "}", "[", "]", ",", ":", '"name"', '"age"', '"mood"', '"tags"', "0"]
    ids = g.vocab_allowed(st, texts)
    # after '{': a key-start or the name key; '"' alone is a key start
    assert ids.tolist() == [1, 7]
