# SPDX-License-Identifier: AGPL-3.0-only
"""The startup vocabulary prewarm covers every served generation path.

The token-class index is keyed by the identity of the token-text list, so a prewarm that built a
different list than the request path would be wasted work. These tests pin the shared accessor,
each tokenizer's text API (Lightning's byte decoder, Nanbeige's SentencePiece pieces), the opt-out
and the server's delegation to its wrapped backend. No GPU, pack or model is needed.
"""
from kindi import server, token_classes
from kindi.grammar_prewarm import GrammarPrewarm
from kindi.lightning_gen import LightningBackend
from kindi.nanbeige_gen import NanbeigeBackend


class _LightningTok(object):
    vocab = ["{", "}", "x"]

    def decode_bytes(self, i):
        return self.vocab[i].encode("utf-8")


class _NanbeigeTok(object):
    tokens = ["{", "}", "x"]

    def piece(self, i):
        return self.tokens[i].encode("utf-8")


def _bare(cls, tok):
    obj = cls.__new__(cls)
    obj.tok = tok
    obj.messages = []
    obj.log = obj.messages.append
    return obj


def test_every_generation_path_inherits_the_startup_prewarm():
    for cls in (server.Kindi, LightningBackend, NanbeigeBackend):
        assert issubclass(cls, GrammarPrewarm)
        assert callable(cls.start_grammar_prewarm)


def test_lightning_uses_its_byte_decoder_and_builds_the_list_once():
    backend = _bare(LightningBackend, _LightningTok())
    assert backend.grammar_texts() == ["{", "}", "x"]
    assert backend.grammar_texts() is backend.grammar_texts()


def test_nanbeige_uses_its_sentencepiece_pieces():
    backend = _bare(NanbeigeBackend, _NanbeigeTok())
    assert backend.grammar_texts() == ["{", "}", "x"]


def test_the_prewarm_builds_the_same_list_the_request_will_use(monkeypatch):
    seen = []
    monkeypatch.setattr(token_classes, "prewarm", lambda texts: seen.append(texts))
    backend = _bare(NanbeigeBackend, _NanbeigeTok())
    thread = backend.start_grammar_prewarm()
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()
    assert seen and seen[0] is backend.grammar_texts()
    assert any("ready" in message for message in backend.messages)


def test_the_prewarm_can_be_opted_out(monkeypatch):
    monkeypatch.setenv("KINDI_PREWARM_GRAMMAR", "0")
    backend = _bare(LightningBackend, _LightningTok())
    assert backend.start_grammar_prewarm() is None
    assert "_grammar_texts" not in backend.__dict__      # nothing was built at startup


def test_the_server_delegates_the_prewarm_to_the_wrapped_backend():
    class Backend(object):
        started = 0

        def start_grammar_prewarm(self):
            Backend.started += 1
            return "thread"

    K = server.Kindi.__new__(server.Kindi)
    K.nanbeige = Backend()
    K.variant = "nanbeige"
    assert K.start_grammar_prewarm() == "thread"
    assert Backend.started == 1


def test_the_server_skips_the_prewarm_for_a_variant_it_does_not_serve():
    K = server.Kindi.__new__(server.Kindi)
    K.nanbeige = None
    K.variant = "other"
    assert K.start_grammar_prewarm() is None
