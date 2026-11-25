# SPDX-License-Identifier: AGPL-3.0-only
"""The vision projector (mmproj) as a host-RAM tier.

Bonsai 2.5 27B (and the other multimodal families in `docs/ROADMAP.md`) ship the vision tower as a
separate `mmproj-*.gguf`, about 600-930 MB, used only when a request carries an image.  On a card
that the language model already fills, uploading it costs VRAM that the K/V cache wants, for weights
that sit idle on every text request.  llama.cpp's `--no-mmproj-offload` is the same idea: keep the
projector off the device.

This module is the tier, not the encoder.  It reads the projector's GGUF header, prices every
tensor, applies a placement policy (`ram`, `vram`, `auto`), and for `ram` loads the weights into one
host arena and measures the resident bytes.  No Kindi build implements a vision projector yet, so
nothing here runs a forward pass, and the server still refuses image input; what exists is the
placement decision and the memory accounting the encoder will consume.

Placement policy is `KINDI_MMPROJ_PLACE` (config key `mmproj_place`), default `ram`: the projector is
off the decode path, so the default keeps the card to the language model and its cache.
"""
import ctypes
import hashlib
import json
import os
import struct
import time

from .gguf import GGUF

# GGML tensor types this reader can price: type id -> (bytes per block, elements per block).
# A projector file that uses a type outside this table fails closed and says which one.
TYPES = {
    0: ("f32", 4, 1),
    1: ("f16", 2, 1),
    2: ("q4_0", 18, 32),
    3: ("q4_1", 20, 32),
    6: ("q5_0", 22, 32),
    7: ("q5_1", 24, 32),
    8: ("q8_0", 34, 32),
    9: ("q8_1", 36, 32),
    10: ("q2_k", 84, 256),
    11: ("q3_k", 110, 256),
    12: ("q4_k", 144, 256),
    13: ("q5_k", 176, 256),
    14: ("q6_k", 210, 256),
    15: ("q8_k", 292, 256),
    30: ("bf16", 2, 1),
}

PLACES = ("ram", "vram", "auto")
DEFAULT_PLACE = "ram"


def resolve_place(requested=None, cfg=None):
    """The placement a run uses: the flag, else the config key, else `ram`.

    `auto` resolves to `ram` on purpose: the projector is not on the decode path, so the VRAM it
    would take is worth more to the language model's K/V cache.  Returns (place, source) or
    (None, error).
    """
    candidates = []
    if requested:
        candidates.append((str(requested).strip().lower(), "--place"))
    if cfg is not None and cfg.get("mmproj_place"):
        candidates.append((str(cfg["mmproj_place"]).strip().lower(), "mmproj_place"))
    candidates.append((DEFAULT_PLACE, "default"))
    for value, source in candidates:
        if value not in PLACES:
            return None, "%s must be one of %s (got %r)" % (source, ", ".join(PLACES), value)
        if value == "auto":
            return "ram", source + " (auto: the projector is off the decode path)"
        return value, source
    return DEFAULT_PLACE, "default"


def declared(pack):
    """The pack manifest's projector block, or None when the pack declares none.

    The field is `mmproj` in `kindi_pack.json`, written by `kindi mmproj FILE --record PACK`:
    `{"bytes": N, "file": name, "sha256": ..., "place": policy}`.  A bare `"vision": true` (or a
    non-object `mmproj`) counts as a declaration with no size, so the image refusal, `/health` and
    `kindi plan` read one field and cannot disagree about whether a pack has vision weights.
    """
    if not pack:
        return None
    try:
        with open(os.path.join(pack, "kindi_pack.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return None
    block = man.get("mmproj")
    if isinstance(block, dict):
        return block
    if block or man.get("vision"):
        return {"bytes": 0, "file": None, "sha256": None, "place": None}
    return None


def policy(pack=None, cfg=None, requested=None):
    """Where the declared projector's weights live and what that costs the card.

    One function, so `kindi plan` and `/health` report the same decision: `place` is the resolved
    policy (`--place`, `KINDI_MMPROJ_PLACE`/`mmproj_place`, default `ram`), `host_bytes` and
    `card_bytes` say where the weights are counted, and `card_saved_bytes` is the card memory the
    policy keeps for the language model and its K/V cache.
    """
    block = declared(pack) or {}
    n = int(block.get("bytes") or 0)
    place, source = resolve_place(requested, cfg)
    out = {"declared": bool(declared(pack)), "bytes": n, "file": block.get("file"),
           "place": place, "place_source": source, "error": None,
           "host_bytes": 0, "card_bytes": 0, "card_saved_bytes": 0, "detail": None}
    if place is None:
        out["error"] = source
        out["detail"] = ("the placement policy is refused, so no projector placement is decided: %s"
                         % source)
        return out
    if place == "ram":
        out["host_bytes"] = n
        out["card_saved_bytes"] = n
    else:
        out["card_bytes"] = n
    mib = n / 1048576.0
    if not out["declared"]:
        out["detail"] = ("no projector declared in this pack's manifest (`kindi mmproj FILE "
                         "--record PACK` records its size), so mmproj_place decides nothing here")
    elif place == "ram":
        out["detail"] = ("%.1f MiB of projector weights stay in host RAM, so %.1f MiB of card "
                         "memory stays with the language model and its K/V cache" % (mib, mib))
    else:
        out["detail"] = ("%.1f MiB of projector weights are counted against the card, out of the "
                         "room the language model and its K/V cache plan against" % mib)
    out["detail"] += "; no Kindi build implements the encoder, so image input is refused"
    return out


def record(pack, path, place=None):
    """Write a projector's size into a pack's manifest: `kindi mmproj FILE --record PACK`.

    The pack keeps no projector bytes (they live in the standalone `mmproj-*.gguf`), so what the
    manifest needs is the price and the identity: bytes, file name, sha256 and the policy.  Returns
    a small report; `result` is `fail` with `error` set when either side is unusable.
    """
    out = {"result": "fail", "error": None, "pack": pack, "path": os.path.abspath(path),
           "bytes": None, "file": os.path.basename(path), "sha256": None, "place": None,
           "manifest": os.path.join(pack or "", "kindi_pack.json")}
    report = inventory(path)
    if report["result"] != "pass":
        out["error"] = report["error"]
        return out
    resolved, source = resolve_place(place, None)
    if resolved is None:
        out["error"] = source
        return out
    try:
        with open(out["manifest"], encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as exc:
        out["error"] = "no readable manifest at %s: %s" % (out["manifest"], exc)
        return out
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    man["mmproj"] = {"bytes": int(report["total_bytes"]), "file": out["file"],
                     "sha256": digest.hexdigest(), "place": resolved,
                     "tensors": len(report["tensors"]), "clip": report["clip"],
                     "note": "the projector lives outside the pack; Kindi prices it and keeps it "
                             "off the card (mmproj_place), and no build implements its encoder"}
    tmp = out["manifest"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, out["manifest"])
    out.update({"result": "pass", "bytes": int(report["total_bytes"]),
                "sha256": man["mmproj"]["sha256"], "place": resolved})
    return out


def render_record(out):
    """Human-readable form of a `--record` result."""
    if out["result"] != "pass":
        return ["projector record: fail (%s)" % out["error"]]
    return ["projector record: pass",
            "  pack: %s" % out["manifest"],
            "  projector: %s, %.1f MiB, sha256 %s... (%s)"
            % (out["file"], out["bytes"] / 1048576.0, out["sha256"][:12], out["place"]),
            "  plan and /health read this size; the weights stay outside the pack"]


def _elements(dims):
    n = 1
    for d in dims:
        n *= int(d)
    return n


def tensor_bytes(dims, kind):
    """(bytes, type name) for one tensor, or (None, reason) for a type this reader cannot price."""
    entry = TYPES.get(int(kind))
    if entry is None:
        return None, "unsupported tensor type %d" % int(kind)
    name, per_block, elems = entry
    n = _elements(dims)
    if n % elems:
        return None, "%s tensor with %d elements is not a multiple of %d" % (name, n, elems)
    return n // elems * per_block, name


def inventory(path):
    """The projector's tensors and price, before any placement decision.

    Returns a report dict; `result` is `fail` with `error` set when the file cannot be read.
    """
    report = {"path": os.path.abspath(path), "exists": os.path.exists(path), "file_bytes": None,
              "version": None, "tensors": [], "total_bytes": 0, "types": {}, "clip": {},
              "result": "fail", "error": None}
    if not report["exists"]:
        report["error"] = "no such file: %s" % path
        return report
    try:
        report["file_bytes"] = os.path.getsize(path)
        g = GGUF(path)
    except (OSError, ValueError, struct.error) as exc:
        report["error"] = "not a readable GGUF: %s" % exc
        return report
    report["version"] = g.version
    for name, ne, kind, offset in g.tensors:
        nbytes, label = tensor_bytes(ne, kind)
        row = {"name": name, "dims": [int(d) for d in ne], "type": int(kind),
               "type_name": label, "bytes": nbytes, "offset": int(offset)}
        report["tensors"].append(row)
        if nbytes is None:
            report["error"] = "%s: %s" % (name, label)
            return report
        report["total_bytes"] += nbytes
        report["types"][label] = report["types"].get(label, 0) + nbytes
    for key in ("clip.projector_type", "clip.has_vision_encoder", "clip.vision.block_count",
                "clip.vision.embedding_length", "clip.vision.image_size", "clip.vision.patch_size",
                "general.name", "general.architecture"):
        if key in g.kv:
            value = g.kv[key]
            if isinstance(value, list) and len(value) > 8:
                value = "%d values" % len(value)
            report["clip"][key] = value
    if not report["tensors"]:
        report["error"] = "the file has no tensors"
        return report
    size = report["file_bytes"]
    if size is not None and report["total_bytes"] > size:
        report["error"] = ("the tensor table prices %d bytes but the file is %d bytes"
                           % (int(report["total_bytes"]), int(size)))
        return report
    report["result"] = "pass"
    return report


def rss_bytes():
    """The process's resident set size, or None where it cannot be read."""
    try:
        if os.name == "nt":
            class _Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            proc = ctypes.windll.kernel32.GetCurrentProcess()
            query = ctypes.windll.kernel32.K32GetProcessMemoryInfo
            query.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
            query.restype = ctypes.c_int
            if query(ctypes.c_void_p(proc), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return None
        with open("/proc/self/statm", "rb") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, IndexError):
        return None


def load(path, place="ram", max_mb=None):
    """Place the projector: price it, and for `ram` read its weights into one host arena.

    `vram` allocates nothing here: this probe never touches a device, so it reports the bytes a
    device placement would hold (`device_bytes`) and keeps the host arena empty.  That is the
    policy's accounting, and the report says so.
    """
    report = inventory(path)
    report.update({"probe": "mmproj", "schema": 1, "support_claim": False,
                   "scope": "projector tier: GGUF inventory and placement accounting; no vision "
                            "encoder, no image path, no forward pass",
                   "place": place, "place_source": None, "host_bytes": 0, "device_bytes": 0,
                   "load_ms": None, "rss_before": None, "rss_after": None, "rss_delta": None,
                   "arena": None})
    if report["result"] != "pass":
        return report
    if place == "vram":
        report["device_bytes"] = report["total_bytes"]
        report["arena"] = "device policy: this probe allocates nothing"
        return report
    limit = None if max_mb in (None, 0) else int(max_mb) * 1024 * 1024
    if limit is not None and report["total_bytes"] > limit:
        report["error"] = ("the projector is %.1f MiB, over the --max-mb %.1f MiB limit"
                           % (report["total_bytes"] / 1048576.0, limit / 1048576.0))
        report["result"] = "fail"
        return report
    try:
        g = GGUF(path)
        base = g.data_start
        span = max(row["offset"] + row["bytes"] for row in report["tensors"])
        started = time.perf_counter()
        before = rss_bytes()
        arena = bytearray(span)
        with open(path, "rb") as f:
            for row in report["tensors"]:
                begin = base + row["offset"]
                f.seek(begin)
                chunk = f.read(row["bytes"])
                if len(chunk) != row["bytes"]:
                    report["error"] = ("%s needs %d bytes at %d, the file ends after %d"
                                       % (row["name"], row["bytes"], begin, len(chunk)))
                    report["result"] = "fail"
                    return report
                arena[row["offset"]:row["offset"] + row["bytes"]] = chunk
        report["load_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        report["rss_before"] = before
        report["rss_after"] = rss_bytes()
        if before is not None and report["rss_after"] is not None:
            report["rss_delta"] = report["rss_after"] - before
        report["host_bytes"] = len(arena)
        report["arena"] = "one host bytearray, %d bytes" % len(arena)
        del arena
    except (OSError, ValueError, MemoryError) as exc:
        report["error"] = "could not hold the projector in host RAM: %s" % exc
        report["result"] = "fail"
    return report


def render(report):
    """Human-readable form of the projector inventory and placement."""
    mib = 1048576.0
    lines = ["vision projector (mmproj) tier: inventory and placement; no encoder, no image path",
             "  file: %s" % report["path"]]
    if report.get("file_bytes"):
        lines.append("  file: %.1f MiB, GGUF v%s, %d tensors"
                     % (report["file_bytes"] / mib, report.get("version"),
                        len(report.get("tensors") or [])))
    if report.get("types"):
        lines.append("  types: " + ", ".join("%s %.1f MiB" % (k, v / mib)
                                             for k, v in sorted(report["types"].items())))
    if report.get("clip"):
        lines.append("  clip: " + ", ".join("%s=%s" % (k, v)
                                            for k, v in sorted(report["clip"].items())))
    lines.append("  place: %s (%s)" % (report.get("place"),
                                       report.get("place_source") or "policy"))
    if report.get("host_bytes"):
        lines.append("  host RAM: %.1f MiB resident in %s (%.1f ms)%s"
                     % (report["host_bytes"] / mib, report.get("arena"), report.get("load_ms") or 0.0,
                        "" if report.get("rss_delta") is None
                        else "; RSS %+d MiB" % (report["rss_delta"] // 1048576)))
    if report.get("device_bytes"):
        lines.append("  device: %.1f MiB would be held on the card (policy accounting; this probe "
                     "allocates nothing)" % (report["device_bytes"] / mib))
    lines.append("  result: %s%s" % (report["result"],
                                     "" if not report.get("error") else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
