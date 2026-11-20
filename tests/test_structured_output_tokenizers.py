# SPDX-License-Identifier: AGPL-3.0-only
"""Structured output over the real Lightning/Nanbeige tokenizer classes.

The older conformance tests drive the backends with a two-word fake vocab. These
tests construct the actual ``LightningTokenizer`` and ``SPM`` classes from
synthetic GGUF-KV records and then run the backend generate loops (fake engine
heads only), so the grammar path sees the same byte/special-token decoding the
servers use: byte-level BPE bytes for Lightning, SentencePiece byte fallback
plus control-token silence for Nanbeige, incremental UTF-8 across split
multibyte sequences, and the control-token stop sets.
"""
import codecs
import json
from types import SimpleNamespace

import numpy as np

from kindi.config import DEFAULTS
from kindi.jsongrammar import grammar_from_schema
from kindi.lightning_tokenizer import LightningTokenizer
from kindi.spm import SPM
from kindi.tokenizer import _B2U


def _lightning_tokenizer():
    specials = ["<|im_start|>", "<|im_end|>", "</s>"]
    merged = ["tr", "tru", "true", "fa", "fal", "fals", "false"]
    vocab = [_B2U[b] for b in range(256)] + specials + merged
    types = [1] * 256 + [3] * len(specials) + [1] * len(merged)
    merges = ["t r", "tr u", "tru e", "f a", "fa l", "fal s", "fals e"]
    kv = {
        "tokenizer.ggml.tokens": vocab,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": 256,
        "tokenizer.ggml.eos_token_id": 257,
        "tokenizer.ggml.add_bos_token": False,
    }
    return LightningTokenizer(kv)


def _nanbeige_tokenizer():
    specials = ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tokens = list(specials) + ["▁"] + ["<0x%02X>" % b for b in range(256)]
    types = [3] * len(specials) + [1] + [6] * 256
    scores = [0.0] * len(specials) + [0.5] + [0.0] * 256
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


def _lightning_stop_ids(tok):
    return {tok.eos} | {i for i, t in enumerate(tok.types) if t == 3}


def _nanbeige_stop_ids(tok):
    return {i for i, ty in enumerate(tok.types) if ty == 3} | {tok.eos}


class _Cp(object):
    asnumpy = staticmethod(lambda x: np.asarray(x))
    flatnonzero = staticmethod(np.flatnonzero)

    class cuda(object):
        class Device(object):
            def __init__(self, dev):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False


def _logits(vocab_len, script, index, illegal):
    z = np.zeros(vocab_len, dtype=np.float64)
    z[illegal] = 100.0
    if index < len(script):
        z[script[index]] = 50.0
    return z


def _fake_k():
    return SimpleNamespace(
        cfg=dict(DEFAULTS, max_tokens=64, temperature=0.0, top_p=1.0, top_k=0,
                 repeat_penalty=1.0, repeat_stop=0, kv_cache=None, lookup=0,
                 mtp_head=None),
        busy={},
        stats={"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
               "completion_tokens": 0})


SCHEMA = {"type": "json_schema", "json_schema": {"name": "x", "schema": {
    "type": "object", "properties": {"city": {"type": "string"}},
    "required": ["city"]}}}
OUT = '{"city":"café"}'


def _request():
    return {"messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
            "response_format": SCHEMA}


def test_lightning_special_tokens_split_and_stop():
    tok = _lightning_tokenizer()
    im_start, im_end = tok.sid["<|im_start|>"], tok.sid["<|im_end|>"]
    ids = tok.encode("hi<|im_start|>x")
    assert im_start in ids
    stop = _lightning_stop_ids(tok)
    assert {im_start, im_end, tok.sid["</s>"]} <= stop
    grammar = grammar_from_schema(SCHEMA["json_schema"]["schema"])
    texts = [tok.decode_bytes(i).decode("utf-8", "replace") for i in range(len(tok.vocab))]
    allowed = grammar.vocab_allowed(grammar.start(), texts)
    assert tok.sid["{"] in allowed
    assert im_start not in allowed and im_end not in allowed


def test_nanbeige_control_tokens_are_silent_and_stop():
    tok = _nanbeige_tokenizer()
    im_start = tok.tokens.index("<|im_start|>")
    im_end = tok.tokens.index("<|im_end|>")
    assert tok.piece(im_start) == b"" and tok.piece(im_end) == b""
    ids = tok.encode("hi<|im_start|>x")
    assert im_start in ids
    stop = _nanbeige_stop_ids(tok)
    assert im_start in stop and im_end in stop
    assert tok.piece(tok.byte_tok[0x7B]) == b"{"


def test_incremental_utf8_holds_a_split_multibyte_sequence():
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    assert dec.decode(b"\xc3") == ""
    piece = dec.decode(b"\xa9")
    assert piece == "é"
    assert "�" not in piece


def test_lightning_backend_structured_output_utf8_stop(monkeypatch):
    import kindi.lightning_chat as chat
    from kindi.lightning_gen import LightningBackend

    monkeypatch.setattr(chat, "render",
                        lambda *a, **k: "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    tok = _lightning_tokenizer()
    script = [tok.sid[_B2U[b]] for b in OUT.encode("utf-8")] + [tok.eos]
    illegal = tok.sid[_B2U[0]]

    class _Engine(object):
        ctx = 4096
        cp = _Cp
        pos = 0
        i = 0

        def __init__(self):
            self.logits = _logits(len(tok.vocab), script, 0, illegal)

        def reset(self):
            self.pos = 0

        def prefill(self, ids, pos=None):
            self.pos = (pos or 0) + len(ids)
            return 0

        def step(self, token=None):
            self.pos += 1
            self.i += 1
            self.logits = _logits(len(tok.vocab), script, self.i, illegal)
            return 0

        def snapshot(self):
            pass

        def restore(self):
            pass

    backend = LightningBackend.__new__(LightningBackend)
    backend.log = lambda *a: None
    backend.path = "x.gguf"
    backend.tok = tok
    backend.tok.tokens = tok.vocab
    backend.stop_ids = _lightning_stop_ids(tok)
    backend.im_start = tok.sid["<|im_start|>"]
    backend.eng = _Engine()
    backend.cached = []
    backend.snap = []

    items = list(backend.generate(_fake_k(), _request()))
    done = next(v for kind, v in items if kind == "done")
    assert json.loads(done["content"]) == {"city": "café"}
    assert done["finish"] == "stop"
    assert "<|im_start|>" not in done["content"]


def test_adapter_capabilities_name_json_grammar_for_the_server_variants():
    from kindi.backends.bonsai import BonsaiBackend
    from kindi.backends.fake import FakeBackend
    from kindi.backends.gemma4 import Gemma4Backend
    from kindi.backends.lightning import LightningBackend
    from kindi.backends.nanbeige import NanbeigeBackend

    for cls in (BonsaiBackend, Gemma4Backend, LightningBackend, NanbeigeBackend):
        assert cls.capabilities.supports_json_grammar is True
        assert cls.capabilities.supports_token_filter is True
    assert FakeBackend.capabilities.supports_json_grammar is False


def test_lightning_adapter_token_filter_reads_the_full_allowed_set():
    from kindi.backends.base import SamplingParams
    from kindi.backends.lightning import LightningBackend

    backend = LightningBackend.__new__(LightningBackend)
    legal, illegal = 7, 3
    logits = np.zeros(16)
    logits[illegal] = 100.0
    logits[legal] = -50.0  # outside lightning_gen's candidate window

    class _Eng(object):
        cp = _Cp

        def __init__(self):
            self.logits = logits

    backend.engine = _Eng()
    params = SamplingParams(max_tokens=1, temperature=0.0,
                            token_filter=lambda state: {legal})
    assert backend._sample_next(params, greedy=illegal, generated=[], rng=None,
                               prompt_ids=[1, 2]) == legal


def test_nanbeige_adapter_token_filter_reads_the_full_allowed_set():
    from kindi.backends.base import SamplingParams
    from kindi.backends.nanbeige import NanbeigeBackend

    tok = _nanbeige_tokenizer()
    backend = NanbeigeBackend.__new__(NanbeigeBackend)
    legal, illegal = tok.byte_tok[0x7B], tok.byte_tok[0]
    logits = np.zeros(len(tok.tokens))
    logits[illegal] = 100.0
    logits[legal] = -50.0  # outside the 30-temperature candidate window

    class _Card(object):
        dev = 0

        def __init__(self):
            self.logits = logits

    class _Eng(object):
        cp = _Cp

        def __init__(self):
            self.cards = [_Card()]

    backend.tok = tok
    backend.engine = _Eng()
    params = SamplingParams(max_tokens=1, temperature=0.0,
                            token_filter=lambda state: {legal})
    assert backend._sample_next(params, greedy=illegal, generated=[], rng=None,
                               prompt_ids=[1, 2]) == legal


def test_nanbeige_backend_structured_output_utf8_stop(monkeypatch):
    from kindi.nanbeige_gen import NanbeigeBackend

    tok = _nanbeige_tokenizer()
    script = [tok.byte_tok[b] for b in OUT.encode("utf-8")] + [tok.eos]
    illegal = tok.byte_tok[0]

    class _Card(object):
        dev = 0

        def __init__(self):
            self.logits = _logits(len(tok.tokens), script, 0, illegal)

    class _Engine(object):
        ctx = 4096
        cp = _Cp
        i = 0

        def __init__(self):
            self.cards = [_Card()]

        def prefill_batch(self, ids, pos0=0):
            return 0

        def step(self, token, pos):
            return 0

        def step_begin(self, token, pos):
            self.i += 1
            self.cards[0].logits = _logits(len(tok.tokens), script, self.i, illegal)

        def step_end(self):
            return 0

    backend = NanbeigeBackend.__new__(NanbeigeBackend)
    backend.log = lambda *a: None
    backend.tok = tok
    backend.stop_ids = _nanbeige_stop_ids(tok)
    backend.eng = _Engine()
    backend.cached = []
    backend.lookup = 0
    backend.lk_stats = [0] * 6
    backend.mtp = False

    items = list(backend.generate(_fake_k(), _request()))
    done = next(v for kind, v in items if kind == "done")
    assert json.loads(done["content"]) == {"city": "café"}
    assert done["finish"] == "stop"
    assert "<|im_start|>" not in done["content"] and "<|im_end|>" not in done["content"]
