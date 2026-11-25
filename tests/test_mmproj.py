"""The vision projector (mmproj) tier: inventory, placement policy and host-RAM residency."""
import io
import os
import struct

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
