# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only gates for the Inkling 2 Mini backend (no pack, no GPU, no compiled engine).

The engine itself (kindi/inkling2.py's ctypes C library) needs a real GGUF and is exercised by
`kindi verify` / tools/verify_inkling2.py on a host that has one; what is pinned here is the
host-side contract that decides whether a file is Inkling at all, which sampling values are
used, how a reply's tokens become text/reasoning/tool calls, and which execution mode a GPU
inventory selects.
"""
import json
import os

import pytest

from kindi import inkling2_cli
from kindi.inkling2 import Spec, _choose_mode
from kindi.inkling2_gen import Reply, model_sampling, parse_call


class FakeGGUF(object):
    def __init__(self, kv, tensors=()):
        self.kv = kv
        self.tensors = list(tensors)
        self.data_start = 0


def _kv(**over):
    kv = {
        "general.architecture": "inkling2_loop",
        "inkling2_loop.block_count": 24,
        "inkling2_loop.embedding_length": 2304,
        "inkling2_loop.expert_feed_forward_length": 1152,
        "inkling2_loop.expert_count": 128,
        "inkling2_loop.expert_used_count": 8,
        "inkling2_loop.vocab_size": 201024,
        "inkling2_loop.vocab_size_unpadded": 200058,
        "inkling2_loop.loop.start": 6,
        "inkling2_loop.loop.end": 18,
        "inkling2_loop.loop.count": 3,
        "inkling2_loop.attention.sliding_window": 1024,
        "inkling2_loop.attention.sliding_window_pattern": [True] * 24,
    }
    kv.update(over)
    return kv


def _tensor(name, ty=0):
    return (name, (8,), ty, 0)


# ---- recognition -----------------------------------------------------------

def test_is_inkling_matches_the_declared_architecture_only():
    assert inkling2_cli.is_inkling(FakeGGUF(_kv()))
    assert not inkling2_cli.is_inkling(FakeGGUF({"general.architecture": "qwen35"}))
    # the header-inventory name is not a runnable file: the engine requires inkling2_loop
    assert not inkling2_cli.is_inkling(FakeGGUF({"general.architecture": "inkling2"}))


def test_spec_builds_offsets_for_global_and_layer_tensors():
    tensors = [_tensor("token_embd.weight"), _tensor("output.weight")]
    tensors += [_tensor("blk.0.attn_q.weight"), _tensor("blk.23.ffn_down_exps.weight", ty=2)]
    # Spec prices every layer's expert bytes, so every layer needs its two expert tensors
    for ip in range(24):
        tensors += [_tensor("blk.%d.ffn_gate_up_exps.weight" % ip, ty=2),
                    _tensor("blk.%d.ffn_down_exps.weight" % ip, ty=2)]
    spec = Spec(FakeGGUF(_kv(), tensors))
    assert spec.n_layer == 24 and spec.n_embd == 2304 and spec.n_exp == 128 and spec.n_used == 8
    assert spec.tmap["token_embd.weight"][1] == 0
    assert spec.l_tys[0 * 42 + 0] == 0          # blk.0.attn_q.weight present
    assert spec.l_tys[23 * 42 + 27] == 2        # blk.23.ffn_down_exps.weight present
    assert spec.l_tys[5 * 42 + 0] == -1         # absent layers stay -1, not 0


def test_spec_rejects_a_wrong_architecture_and_wrong_dimensions():
    with pytest.raises(ValueError, match="inkling2_loop"):
        Spec(FakeGGUF({"general.architecture": "qwen35"}))
    with pytest.raises(ValueError, match="native kernels require dimensions"):
        Spec(FakeGGUF(_kv(**{"inkling2_loop.expert_count": 64})))


# ---- sampling --------------------------------------------------------------

def test_model_sampling_defaults_and_gguf_overrides():
    assert model_sampling({}) == {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    got = model_sampling({"general.sampling.temp": 0.7, "general.sampling.top_p": 0.9})
    assert got["temperature"] == 0.7 and got["top_p"] == 0.9 and got["top_k"] == 0


# ---- reply parsing ----------------------------------------------------------

class FakeTok(object):
    def __init__(self):
        self.sid = {"<|message_model|>": 1, "<|content_text|>": 2, "<|content_thinking|>": 3,
                    "<|content_invoke_tool_json|>": 4, "<|end_message|>": 5,
                    "<|content_model_end_sampling|>": 6}
        self.special = set(self.sid.values())
        self.words = {10: b"hello", 11: b" world"}

    def decode_bytes(self, i):
        return self.words.get(i, b"")

    def decode(self, ids):
        return b"".join(self.words.get(i, b"") for i in ids).decode()


def test_reply_routes_thinking_text_and_tool_calls_by_message_kind():
    tok = FakeTok()
    reply = Reply(tok, {"search"})
    assert reply.feed(1) == []                       # <|message_model|>
    assert reply.feed(3) == []                       # thinking header
    assert reply.feed(10) == [("reasoning", "hello")]
    assert reply.feed(5) == []                       # end of thinking message
    assert reply.feed(1) == []
    assert reply.feed(2) == []                       # text header
    assert reply.feed(11) == [("text", " world")]
    assert reply.feed(6) == [] and reply.ended       # EOS ends the reply
    assert reply.calls == []


def test_parse_call_accepts_an_offered_tool_and_rejects_an_unoffered_one():
    body = json.dumps({"name": "search", "args": {"q": "kindi"}})
    call = parse_call("search", body, {"search"})
    assert call and call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"q": "kindi"}
    assert parse_call("search", body, {"other"}) is None
    assert parse_call("search", "not json", {"search"}) is None


# ---- mode selection ----------------------------------------------------------

def test_choose_mode_prefers_full_vram_then_pinned_then_cpu(monkeypatch):
    monkeypatch.delenv("KINDI_INKLING2_MODE", raising=False)
    big = [(0, "RTX 6000", 89, 9000, 24000)]
    small = [(0, "K420", 42, 1500, 2000)]
    assert _choose_mode([], vram_mb=7000) == "cpu"
    assert _choose_mode(big, vram_mb=7000) == "gpu_full"
    assert _choose_mode(small, vram_mb=7000) == "hetero_k420"
    monkeypatch.setenv("KINDI_INKLING2_MODE", "cpu")
    assert _choose_mode(big, vram_mb=7000) == "cpu"


# ---- CLI ---------------------------------------------------------------------

def test_inkling_subcommand_declares_a_context_flag():
    from kindi.__main__ import _build_parser
    ns = _build_parser().parse_args(["inkling", "m.gguf", "hi"])
    assert ns.ctx == 2048 and ns.max_tokens == 128


def test_inkling_run_reports_a_missing_gguf_instead_of_a_traceback(tmp_path):
    class Args(object):
        gguf = str(tmp_path / "nope.gguf")
        prompt = "hi"
        max_tokens = 4
        ctx = 2048
        raw = False
    assert inkling2_cli.run(Args()) == 2


def test_bundled_reference_is_well_formed():
    path = os.path.join(os.path.dirname(os.path.abspath(inkling2_cli.__file__)),
                        "reference-inkling2-mini.json")
    ref = json.load(open(path, encoding="utf-8"))
    assert ref["model"] == "Inkling-2-Mini-UD-Q4_K_M"
    for case in ref["cases"]:
        assert len(case["expected"]) == len(case["gap"]) == len(case["alt"])
