"""Device-free tests for the v0.7.2 LoRA port and the unified OpenAI error body.

The kernels need a CUDA card; what can be checked here is the arithmetic the adapter is
converted with (the activation fold and the V-head permutation), the readers, the pack
layout `kindi.lora.convert` writes, the cache fingerprint it feeds, and the config/CLI
plumbing. `tests/test_server.py` covers the HTTP error bodies on the CPU.
"""
import json
import os

import numpy as np
import pytest

from kindi import engine, lora
from kindi.api import errors as api_errors


def _hadamard(n):
    """the n-point Hadamard matrix (n a power of two), built the way fold_rows folds"""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def test_fold_rows_is_the_activation_fold():
    rng = np.random.default_rng(7)
    hb, n, r = 8, 16, 3
    A = rng.standard_normal((r, n))
    signs = rng.choice([-1.0, 1.0], size=n)
    got = lora.fold_rows(A, signs, hb)
    # (blockdiag(H): each hb-wide block of a row goes through its own transform)
    want = np.concatenate([(A * signs)[:, i * hb:(i + 1) * hb] @ (_hadamard(hb) / np.sqrt(hb)).T
                           for i in range(n // hb)], axis=1)
    assert np.allclose(got, want)


def test_fold_rows_is_orthogonal():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 32))
    signs = np.ones(32)
    F = lora.fold_rows(A, signs, 16)
    assert np.allclose(F @ F.T, A @ A.T)


def test_vhead_permutation_is_a_bijection_in_gguf_order():
    assert list(lora.vhead_perm(2, 4)) == [0, 2, 1, 3]
    assert sorted(lora.vhead_perm(4, 12)) == list(range(12))
    rows = lora.vhead_rows(2, 4, 3)
    assert sorted(rows.tolist()) == list(range(12))


def test_gguf_and_hf_adapter_names_both_parse():
    assert lora._gguf_name("blk.3.attn_q.lora_A.weight") == (3, "attn_q", "A")
    assert lora._gguf_name("base_model.model.model.layers.5.self_attn.k_proj.lora_B.weight") == (5, "attn_k", "B")
    assert lora._gguf_name("model.layers.5.mlp.down_proj.lora_A.default.weight") == (5, "ffn_down", "A")
    assert lora._gguf_name("not-an-adapter") is None


def test_read_adapter_reads_npz_pairs_and_refuses_half_a_pair(tmp_path):
    good = tmp_path / "adapter.npz"
    np.savez(good,
             **{"blk.0.ffn_gate.lora_A.weight": np.zeros((2, 8), np.float32),
                "blk.0.ffn_gate.lora_B.weight": np.zeros((4, 2), np.float32),
                "r": np.array(2), "lora_alpha": np.array(4)})
    pairs, cfg, files = lora.read_adapter(str(good))
    assert list(pairs) == [(0, "ffn_gate")]
    assert pairs[(0, "ffn_gate")][0].shape == (2, 8)
    assert int(cfg["r"]) == 2 and int(cfg["lora_alpha"]) == 4
    assert files == [str(good)]

    bad = tmp_path / "half.npz"
    np.savez(bad, **{"blk.0.ffn_gate.lora_A.weight": np.zeros((2, 8), np.float32)})
    with pytest.raises(SystemExit, match="only one of lora_A / lora_B"):
        lora.read_adapter(str(bad))


def _tiny_pack(path, C=8, R=4):
    path.mkdir(parents=True)
    (path / "kindi_pack.json").write_text(json.dumps({
        "pack_format": 1, "hadamard_block": 4, "source_sha256": "a" * 64,
        "dimensions": {"groups": 2, "state_size": 2, "gdn_inner": 4},
    }), encoding="utf-8")
    tens = {"blk.0.ffn_gate.weight": {"rows": R, "ne0": C},
            "blk.0.attn_qkv.weight": {"rows": 4, "ne0": C}}
    (path / "bcodes_index.json").write_text(json.dumps({"tensors": tens}), encoding="utf-8")
    (path / "index.json").write_text(json.dumps({"tensors": tens}), encoding="utf-8")
    np.ones(C, np.float32).tofile(str(path / ("signs_%d.f32" % C)))


def test_convert_writes_folded_fp16_rows_and_resolve_finds_them(tmp_path):
    pack = tmp_path / "pack"
    _tiny_pack(pack)
    A = np.arange(16, dtype=np.float32).reshape(2, 8) / 10.0
    B = np.ones((4, 2), dtype=np.float32)
    adapter = tmp_path / "adapter.npz"
    np.savez(adapter, **{"blk.0.ffn_gate.lora_A.weight": A, "blk.0.ffn_gate.lora_B.weight": B,
                         "lora_alpha": np.array(2), "r": np.array(2)})
    key = lora.convert(str(adapter), str(pack), log=lambda *a: None)
    assert len(key) == 16
    meta = json.loads((pack / "lora" / key / "meta.json").read_text(encoding="utf-8"))
    assert meta["tensors"] == 1 and meta["ranks"] == [2]

    h, ts = lora.resolve(key, str(pack), log=lambda *a: None)
    assert h == key
    a, b = ts["blk.0.ffn_gate.weight"]
    assert a.dtype == np.float16 and a.shape == (2, 8)
    assert b.dtype == np.float16 and b.shape == (4, 2)
    want = lora.fold_rows(A.astype(np.float64), np.ones(8), 4)
    assert np.allclose(a.astype(np.float32), want, atol=1e-3)


def test_convert_refuses_a_tensor_kindi_does_not_adapt(tmp_path):
    pack = tmp_path / "pack"
    _tiny_pack(pack)
    adapter = tmp_path / "emb.npz"
    np.savez(adapter, **{"token_embd.lora_A.weight": np.zeros((2, 8), np.float32),
                         "token_embd.lora_B.weight": np.zeros((4, 2), np.float32)})
    with pytest.raises(SystemExit):
        lora.convert(str(adapter), str(pack), log=lambda *a: None)


def test_engine_registers_the_adapter_kernels():
    assert "k_lora" in engine.KERNELS and "k_lora2" in engine.KERNELS


def test_cache_fingerprint_changes_with_the_adapter(monkeypatch):
    from kindi.cache import engine_fingerprint

    class Fake(object):
        W = 1

    monkeypatch.setattr("kindi.cache._arith_files", lambda here: ())
    h0, d0 = engine_fingerprint(Fake(), {"head": "exact"})
    Fake.lora_hash = "0123456789abcdef"
    h1, d1 = engine_fingerprint(Fake(), {"head": "exact"})
    assert d1["lora"] == "0123456789abcdef" and "lora" not in d0
    assert h0 != h1


def test_config_reads_and_expands_the_adapter(monkeypatch, tmp_path):
    from kindi import config

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("KINDI_LORA", "~/adapter")
    cfg = config.load()
    assert cfg["lora"] == os.path.expanduser("~/adapter")
    assert config.DEFAULTS["lora"] == ""


def test_a_wrong_typed_messages_value_is_a_request_error_not_a_crash():
    from kindi import oai

    with pytest.raises(oai.RequestError, match="messages must be a list"):
        oai.check_messages({"messages": "hello"})
    with pytest.raises(oai.RequestError, match="string"):
        oai.check_messages({"messages": [{"role": 3}]})
    oai.check_messages({"messages": [{"role": "user", "content": "hi"}]})


def test_error_body_names_the_field_and_keeps_the_type_as_code():
    body = api_errors.error_body("messages cannot be a string", param="messages")
    assert body["error"]["param"] == "messages"
    assert body["error"]["code"] == "invalid_request_error"
