"""Requirement-based tests for prompt-lookup drafting and the SentencePiece
tokenizer: the user-visible contracts (drafts copy earlier text; SPM encodes
like llama.cpp on small synthetic vocabularies), not implementation internals.
"""
from kindi.lookup import Lookup
from kindi.spm import SPM


def test_lookup_drafts_what_followed_an_earlier_match():
    lk = Lookup([10, 20, 30, 40, 10, 20])
    # The text now ends with 10,20 which earlier was followed by 30.
    assert lk.draft(1) == [30]


def test_lookup_no_history_no_draft():
    assert Lookup().draft(4) == []
    assert Lookup([1]).draft(4) == []


def test_lookup_length_prefers_no_draft_when_acceptance_is_low():
    lk = Lookup([1, 2, 3, 1, 2])
    lk.acc, lk.tries = 0.0, 10.0  # acceptance 0: a draft is expected to lose time
    assert lk.length(4) == 0


def test_lookup_kept_updates_acceptance():
    lk = Lookup([1, 2, 3, 1, 2])
    before = lk.acc / lk.tries
    lk.kept(2, 2)  # all drafted tokens kept
    assert lk.acc / lk.tries > before
    lk.kept(0, 3)  # nothing kept
    assert lk.acc / lk.tries < 1.0


def _spm():
    tokens = ["<s>", "</s>", "▁a", "b", "▁ab", "▁", "<0x61>", "<0x62>", "<0x63>"]
    scores = [0.0, 0.0, 3.0, 1.0, 5.0, 0.5, 0.0, 0.0, 0.0]
    types = [3, 3, 1, 1, 1, 1, 6, 6, 6]
    kv = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": scores,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_space_prefix": True,
    }
    return SPM(kv)


def test_spm_encodes_with_space_prefix_and_merges():
    spm = _spm()
    # "ab" -> leading space -> "▁ab", the highest-score piece.
    assert spm.encode("ab") == [4]
    # "a b" -> "▁a" + "▁b"? "▁b" is not a piece, so byte fallback for "b" after ▁.
    ids = spm.encode("a b")
    assert ids[0] == 2  # ▁a


def test_spm_round_trip_on_known_text():
    spm = _spm()
    assert spm.decode(spm.encode("ab")) == "ab"


def test_spm_byte_fallback_for_unknown_characters():
    spm = _spm()
    ids = spm.encode("c")  # "▁c" unknown -> ▁ + byte token for "c"
    assert 8 in ids  # <0x63>


def test_spm_special_tokens_split_first():
    spm = _spm()
    # Control tokens are matched in the text before BPE merging.
    ids = spm.encode("ab</s>")
    assert 1 in ids  # </s>
