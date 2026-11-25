"""The vision projector (mmproj) tier: inventory, placement policy and host-RAM residency."""
import io
import os
import struct
from pathlib import Path

from kindi import mmproj

ALIGN = 32


def write_gguf(path, tensors, kv=()):
    """A minimal GGUF v3 whose tensor offsets are aligned the way the converter writes them."""
    def header(offsets):
        out = io.BytesIO()
        out.write(b"GGUF")
        out.write(struct.pack("<I", 3))
        out.write(struct.pack("<Q", len(tensors)))
        out.write(struct.pack("<Q", len(kv)))
        for key, value in kv:
            k = key.encode()
            out.write(struct.pack("<Q", len(k)))
            out.write(k)
            out.write(struct.pack("<I", 8))                 # string
            v = value.encode()
            out.write(struct.pack("<Q", len(v)))
            out.write(v)
        for (name, dims, kind, _data), off in zip(tensors, offsets, strict=True):
            n = name.encode()
            out.write(struct.pack("<Q", len(n)))
            out.write(n)
            out.write(struct.pack("<I", len(dims)))
            for d in dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", kind))
            out.write(struct.pack("<Q", off))
        return out.getvalue()

    offsets, at = [], 0
    for _name, _dims, _kind, data in tensors:
        offsets.append(at)
        at += len(data)
        at = (at + ALIGN - 1) // ALIGN * ALIGN
    head = header(offsets)
    start = (len(head) + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(b"\0" * (start - len(head)))
        for (_name, _dims, _kind, data), off in zip(tensors, offsets, strict=True):
            fh.seek(start + off)
            fh.write(data)
    return start


def projector(tmp_path, extra=()):
    """Two tensors: an f32 patch embedding and an f16 attention matrix."""
    tensors = [("v.patch_embd.weight", [8, 2], 0, bytes(64)),
               ("v.blk.0.attn_q.weight", [16], 1, bytes(32))] + list(extra)
    path = str(tmp_path / "mmproj.gguf")
    write_gguf(path, tensors, kv=(("clip.projector_type", "qwen3vl_merger"),
                                  ("general.architecture", "clip")))
    return path, tensors


def test_inventory_prices_every_tensor(tmp_path):
    path, _tensors = projector(tmp_path)
    report = mmproj.inventory(path)
    assert report["result"] == "pass", report["error"]
    assert report["total_bytes"] == 96
    assert report["types"] == {"f32": 64, "f16": 32}
    assert report["clip"]["clip.projector_type"] == "qwen3vl_merger"
    assert len(report["tensors"]) == 2


def test_ram_place_holds_the_weights_in_one_arena(tmp_path):
    path, tensors = projector(tmp_path)
    report = mmproj.load(path, place="ram")
    assert report["result"] == "pass", report["error"]
    span = mmproj.inventory(path)["total_bytes"]
    assert report["host_bytes"] >= span
    assert report["device_bytes"] == 0
    assert report["arena"].startswith("one host bytearray")


def test_vram_place_accounts_without_allocating(tmp_path):
    path, _tensors = projector(tmp_path)
    report = mmproj.load(path, place="vram")
    assert report["result"] == "pass"
    assert report["device_bytes"] == report["total_bytes"] == 96
    assert report["host_bytes"] == 0


def test_resolve_place_flag_then_config_then_default():
    assert mmproj.resolve_place("ram", {"mmproj_place": "vram"}) == ("ram", "--place")
    assert mmproj.resolve_place(None, {"mmproj_place": "vram"}) == ("vram", "mmproj_place")
    assert mmproj.resolve_place(None, {}) == ("ram", "default")
    place, source = mmproj.resolve_place("auto", {})
    assert place == "ram" and "auto" in source


def test_an_unknown_place_is_refused():
    place, error = mmproj.resolve_place("gpu", {})
    assert place is None
    assert "ram" in error and "vram" in error and "auto" in error


def test_an_unsupported_tensor_type_fails_closed(tmp_path):
    path = str(tmp_path / "odd.gguf")
    write_gguf(path, [("v.odd.weight", [32], 99, bytes(32))])
    report = mmproj.inventory(path)
    assert report["result"] == "fail"
    assert "unsupported tensor type 99" in report["error"]


def test_a_truncated_file_fails_with_a_reason(tmp_path):
    path, _tensors = projector(tmp_path)
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 40)
    report = mmproj.load(path, place="ram")
    assert report["result"] == "fail"
    assert report["error"]


def test_max_mb_refuses_a_projector_that_does_not_fit(tmp_path):
    big = bytes(400000 * 4)                      # 1.53 MiB of f32
    path, _tensors = projector(tmp_path, extra=[("v.big.weight", [400000], 0, big)])
    report = mmproj.load(path, place="ram", max_mb=1)
    assert report["result"] == "fail"
    assert "over the --max-mb" in report["error"]
    assert mmproj.load(path, place="ram", max_mb=4)["result"] == "pass"


def fake_pack(tmp_path, mmproj_block=None):
    """A pack directory with the manifest and the two indexes `kindi plan` reads."""
    import json

    pack = tmp_path / "pack"
    pack.mkdir(parents=True)
    man = {"model": "fake", "backend": "kindi", "source_sha256": "x", "f32_bytes": 0,
           "head": "token_embd.weight",
           "dimensions": {"layers": 1, "full_attention_interval": 2, "kv_heads": 2,
                          "head_dim": 16, "state_size": 4, "gdn_inner": 8, "groups": 2}}
    if mmproj_block is not None:
        man["mmproj"] = mmproj_block
    (pack / "kindi_pack.json").write_text(json.dumps(man))
    (pack / "bcodes_index.json").write_text(json.dumps({"tensors": {"blk.0.w": {"rows": 16,
                                                                              "ne0": 128}}}))
    (pack / "index.json").write_text(json.dumps({"tensors": {"token_embd.weight":
                                                             {"nfloats": 128}}}))
    return str(pack)


def test_record_writes_the_projector_size_into_the_manifest(tmp_path):
    import json

    path, _tensors = projector(tmp_path)
    pack = fake_pack(tmp_path)
    out = mmproj.record(pack, path)
    assert out["result"] == "pass", out["error"]
    block = json.loads((Path(pack) / "kindi_pack.json").read_text())["mmproj"]
    assert block["bytes"] == 96
    assert block["place"] == "ram"
    assert len(block["sha256"]) == 64
    assert block["tensors"] == 2
    assert mmproj.declared(pack)["bytes"] == 96
    assert mmproj.record(pack, path, place="vram")["place"] == "vram"


def test_record_refuses_a_pack_without_a_manifest(tmp_path):
    path, _tensors = projector(tmp_path)
    out = mmproj.record(str(tmp_path / "nope"), path)
    assert out["result"] == "fail"
    assert "manifest" in out["error"]


def test_policy_counts_the_card_memory_each_place_saves(tmp_path):
    path, _tensors = projector(tmp_path)
    pack = fake_pack(tmp_path)
    mmproj.record(pack, path)
    ram = mmproj.policy(pack, {"mmproj_place": "ram"})
    assert (ram["declared"], ram["place"]) == (True, "ram")
    assert (ram["host_bytes"], ram["card_bytes"], ram["card_saved_bytes"]) == (96, 0, 96)
    assert "host RAM" in ram["detail"]
    vram = mmproj.policy(pack, {"mmproj_place": "vram"})
    assert (vram["host_bytes"], vram["card_bytes"], vram["card_saved_bytes"]) == (0, 96, 0)
    assert "counted against the card" in vram["detail"]
    assert mmproj.policy(pack, {"mmproj_place": "gpu"})["error"]


def test_health_reads_the_same_policy_as_the_plan(tmp_path):
    """`/health`'s projector block is `mmproj.policy`, and `kindi plan` embeds the same call."""
    from kindi import plan, server

    path, _tensors = projector(tmp_path)
    pack = fake_pack(tmp_path)
    mmproj.record(pack, path)
    cfg = {"mmproj_place": "vram"}
    assert server._projector_health(pack, cfg) == mmproj.policy(pack, cfg)
    rep = plan.describe(pack, 100, "auto", assume_vram_gib=1, assume_ram_gib=4, assume_cores=2,
                        mmproj_place="vram")
    assert rep["projector"] == server._projector_health(pack, cfg)
    assert rep["projector"]["card_bytes"] == 96


def test_a_pack_that_declares_nothing_decides_nothing(tmp_path):
    import json

    pack = fake_pack(tmp_path)
    pol = mmproj.policy(pack, {"mmproj_place": "ram"})
    assert (pol["declared"], pol["bytes"], pol["card_saved_bytes"]) == (False, 0, 0)
    assert "decides nothing" in pol["detail"]
    assert mmproj.declared(pack) is None
    # a bare `vision` flag is still a declaration (the image refusal reads it) with no size
    from pathlib import Path

    vision = fake_pack(tmp_path / "vision")
    man = json.loads((Path(vision) / "kindi_pack.json").read_text())
    man["vision"] = True
    (Path(vision) / "kindi_pack.json").write_text(json.dumps(man))
    assert mmproj.declared(vision) == {"bytes": 0, "file": None, "sha256": None, "place": None}
