"""Tokenizer ids on a synthetic Qwen3.5 vocabulary, without a model pack."""
import random

import pytest

from kindi.tokenizer import Tok, _B2U, bpe, gpt2_bytes

BASE = 256          # the 256 byte-level tokens
AB = BASE           # "ab"
ABC = BASE + 1      # "abc"
ADDED = 248044      # the first added-token slot Tok reads


def make_tok(tmp_path):
    """A 248k-entry vocabulary: the GPT-2 byte alphabet, two merges, two added tokens."""
    vocab = ([_B2U[b] for b in range(BASE)] + ["ab", "abc"]
             + [""] * (ADDED - (BASE + 2)) + ["<|im_start|>", "<|im_end|>"])
    assert len(vocab) == ADDED + 2
    merges = tmp_path / "merges.txt"
    merges.write_text("a b\nab c\n", encoding="utf-8")
    return Tok(vocab, str(merges))


def test_gpt2_bytes_maps_ascii_to_itself():
    assert gpt2_bytes("abc") == "abc"
    assert gpt2_bytes(" ") == _B2U[32]


def test_golden_ids(tmp_path):
    tok = make_tok(tmp_path)
    assert tok.encode("ab") == [AB]
    assert tok.encode("abc") == [ABC]
    assert tok.encode("Hi") == [72, 105]
    assert tok.encode("<|im_start|>") == [ADDED]


def test_decode_round_trip(tmp_path):
    tok = make_tok(tmp_path)
    assert tok.decode([AB]) == "ab"
    assert tok.decode(tok.encode("abc")) == "abc"


def test_added_token_is_split_out_of_text(tmp_path):
    tok = make_tok(tmp_path)
    ids = tok.encode("Hi<|im_start|>there")
    assert ids[:2] == [72, 105]
    assert ids[2] == ADDED
    assert tok.decode(ids) == "Hi<|im_start|>there"


def test_lowest_rank_pair_merges_first():
    ranks = {("a", "b"): 1, ("b", "c"): 0}
    assert bpe("abc", ranks) == ["a", "bc"]


@pytest.mark.parametrize("text", (
    "hello world",
    "  leading and trailing  ",
    "def f():\n\treturn 1\n",
    "caf\u00e9 \u65e5\u672c\u8a9e \U0001f642",
    "a\r\nb\r\n",
    "<|im_start|>system\n<|im_end|>",
    "x" * 300,
))
def test_unicode_and_whitespace_round_trip(tmp_path, text):
    tok = make_tok(tmp_path)
    assert tok.decode(tok.encode(text)) == text


def test_tokenizer_fuzz_corpus_round_trips(tmp_path):
    """A deterministic corpus over the boundaries that a hand-written pre-tokenizer can lose."""
    tok = make_tok(tmp_path)
    rng = random.Random(20261111)
    alphabet = ("a", "b", "c", " ", "\n", "\r", "\t", "\u00e9", "\u65e5", "\U0001f642",
                "<|im_start|>", "<|im_end|>")
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 60)))
        assert tok.decode(tok.encode(text)) == text


def test_missing_piece_fails_loudly(tmp_path):
    tok = make_tok(tmp_path)
    del tok.sid["c"]                 # a corrupt pack: the byte token is absent
    try:
        tok.encode("c")
    except ValueError as e:
        assert "c" in str(e)
    else:
        raise AssertionError("a missing token was silently encoded as id 0")
