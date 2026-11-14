"""Per-machine calibration cache (`kindi calibrate`, roadmap gate 2).

On the paths that time them (non-TC plans, HIP), the 5-trit kernel shapes are timed at every
start (`tp.g5_tuned`) and the result discarded, and draft count / placement are set by hand. This
module persists the measured answers in one keyed file per (GPU, pack hash, engine build):

    KINDI_CALIB_DIR (default ~/.cache/kindi/calib) / <gpu>-<pack-sha12>.json

Sections:
  g5         per-shape best (threads, rows per thread, splits, kernel var) for the 5-trit GEMV
  draft_k    the effective draft length the calibration recorded (KINDI_SPEC_K still wins)
  ratio      snapshot of the split-ratio calibration (tandem.load_ratio), when present
  placement  the planner's picks (auto/rate) for a context, recorded as data

The engine build tag (version + a hash of the compiled kernel sources) stamps every file: a
kernel or version change is a miss, never stale configs. Every read tolerates a missing or
corrupt file, and every write is atomic and best-effort - calibration must never break a start.
"""
import hashlib
import json
import os
import re
import time

KIND = "kindi-calib/1"
_MEMO = {}
_BUILD_TAG = None


def calib_dir():
    return os.environ.get("KINDI_CALIB_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "kindi", "calib")


def gpu_key(name, arch):
    """A filename-safe identity for a GPU: model name + compute capability / gcn arch."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "gpu"
    return "%s-%s" % (slug[:48], str(arch).replace(" ", "").replace("/", "-"))


def gpu_identity():
    """gpu_key for the current CUDA device, or None without a GPU."""
    try:
        import cupy as cp
        rt = cp.cuda.runtime
        prop = rt.getDeviceProperties(cp.cuda.Device().id)
        name = prop.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        if getattr(rt, "is_hip", False):
            arch = prop.get("gcnArchName", b"")
            if isinstance(arch, bytes):
                arch = arch.decode("utf-8", "replace")
            arch = str(arch).split(":")[0]
        else:
            arch = "sm%d%d" % (prop["major"], prop["minor"])
        return gpu_key(name, arch)
    except Exception:
        return None


def build_tag():
    """kindi version + a hash of the CUDA kernel sources, so a kernel change is a cache miss."""
    global _BUILD_TAG
    if _BUILD_TAG is None:
        from . import __version__
        h = hashlib.sha256(__version__.encode())
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            names = sorted(n for n in os.listdir(here) if n.endswith(".cu"))
        except OSError:
            names = []
        for n in names:
            try:
                with open(os.path.join(here, n), "rb") as f:
                    h.update(f.read(1 << 20))
            except OSError:
                continue
        _BUILD_TAG = h.hexdigest()[:16]
    return _BUILD_TAG


def path_for(gpu, pack_sha):
    sha = (str(pack_sha or "pack"))[:12]
    return os.path.join(calib_dir(), "%s-%s.json" % (gpu, sha))


def _load(path):
    if path in _MEMO:
        return _MEMO[path]
    data = None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("kind") == KIND and d.get("build") == build_tag():
            data = d
    except (OSError, ValueError):
        data = None
    _MEMO[path] = data
    return data


def _save(path, data):
    data = dict(data, kind=KIND, build=build_tag(), saved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except OSError:
        pass
    _MEMO[path] = data


# ------------------------------------------------------------------ 5-trit kernel configs

def lookup_g5(gpu, pack_sha, lane, R, NG, rowpar):
    """The measured best (threads, rows per thread, splits, kernel var) for one shape, or None."""
    d = _load(path_for(gpu, pack_sha))
    if not d:
        return None
    sec = (d.get("g5") or {}).get("lane" if lane else "row", {})
    c = sec.get("%d,%d,%d" % (R, NG, 1 if rowpar else 0))
    if not isinstance(c, (list, tuple)) or len(c) < 3:
        return None
    return (int(c[0]), int(c[1]), int(c[2]), str(c[3]) if len(c) > 3 else "r")


def store_g5(gpu, pack_sha, lane, R, NG, rowpar, cfg):
    """Record one shape's measured config (nt, rpt, splits, kernel var); best-effort."""
    path = path_for(gpu, pack_sha)
    d = dict(_load(path) or {})
    g5 = dict(d.get("g5") or {})
    sec = dict(g5.get("lane" if lane else "row") or {})
    sec["%d,%d,%d" % (R, NG, 1 if rowpar else 0)] = [int(cfg[0]), int(cfg[1]), int(cfg[2]), str(cfg[3]) if len(cfg) > 3 else "r"]
    g5["lane" if lane else "row"] = sec
    d["g5"] = g5
    d["gpu"] = gpu
    d["pack_sha256"] = pack_sha
    _save(path, d)


# ------------------------------------------------------------------ the other measured answers

def record(gpu, pack_sha, section, data):
    """Record one measured section (draft_k / ratio / placement); best-effort."""
    path = path_for(gpu, pack_sha)
    d = dict(_load(path) or {})
    d[section] = data
    d["gpu"] = gpu
    d["pack_sha256"] = pack_sha
    _save(path, d)


def read(gpu, pack_sha):
    """The whole calibration for this (GPU, pack), or None."""
    return _load(path_for(gpu, pack_sha))


def forget_gpu(gpu):
    """Drop the in-process cache for one GPU's files (tests)."""
    drop = [p for p in _MEMO if os.path.basename(p).startswith(gpu + "-")]
    for p in drop:
        del _MEMO[p]
