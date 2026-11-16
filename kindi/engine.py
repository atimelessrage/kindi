"""Kindi engine for Ternary-Bonsai-2 Qwen3.5 hybrid models with ternary weights.

Per token: embedding row -> hybrid GDN/attention blocks with SwiGLU FFNs -> output norm -> tied head.
Every matrix product is a ternary GEMV (k_g4, lookup tables, see docs/HOW-IT-WORKS.md); the activations entering it
are Hadamard-folded (the transform Prism ML folded into the weights), fused into the preceding norm.
Arithmetic follows llama.cpp's reference for this model (fp64 norm sums, fp16 K/V cache), so greedy output matches
it token for token (kindi verify). The K/V cache can also be q8_0 (KINDI_KV=q8_0, or kv="q8_0", which the
tandem's plain GPU owner uses): the same kernels and arithmetic as the plan engine's q8_0 cache (k_store_q8,
k_q8row, k_attn_sq8t, k_attn_vq8w), half the memory, and a different - lossy - cache than the fp16 reference,"""
import json
import os
import time

import numpy as np
try:
    import cupy as cp
except ImportError:                   # a CPU-only host (KINDI_DEVICE=cpu)
    cp = None


def _cupy_arch_fix():
    """CuPy 9.6 picks the NVRTC target with min() on strings: min("120", "75") is "120", which NVRTC 10.2 rejects
    (NVRTC_ERROR_INVALID_OPTION for every kernel on Blackwell, sm_120). The same choice, compared as numbers."""
    try:
        from cupy.cuda import compiler, device
        if cp.cuda.runtime.is_hip or getattr(compiler, "_kindi_arch_fix", False):
            return
        mx = compiler._get_max_compute_capability()
    except Exception:  # noqa: BLE001 - best-effort NVRTC arch fix; the default stays if it fails
        return

    def _get_arch():
        a = device.Device().compute_capability
        if a in compiler._tegra_archs:
            return a
        return a if int(a) <= int(mx) else mx

    compiler._get_arch = _get_arch
    compiler._kindi_arch_fix = True


if cp is not None:
    _cupy_arch_fix()

from . import layout

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
EOS = (248044, 248046)
FIRST_SPECIAL = 248044                 # Qwen3.5 vocabulary: ids 248044.. are the special / control tokens
CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kindi.cu")
KERNELS = ("k_g4", "k_g4l", "k_fold_fwht", "k_rmsfold", "k_smfold", "k_l2w", "k_rmsw", "k_gdn_gates",
           "k_gdn_conv", "k_gdn_delta", "k_rope", "k_deq_emb3", "k_attn_s4w", "k_softmax_rows", "k_attn_v4p",
           "k_attn_comb", "k_argmax", "k_sumparts", "k_padd", "k_fwht_hi", "k_expand", "k_mask_neg", "k_band", "k_topk",
           "k_attn_sg", "k_attn_vg", "k_xchg",
           "k_xchg2", "k_attn_vgn", "k_attn_combn", "k_store_h", "k_cpq", "k_xchg3",
           "k_xrms", "k_gdn_pre", "k_gdn_deltan", "k_rmsw_s", "k_rope2", "k_store_q8", "k_store_fp8",
           "k_attn_sf8t", "k_attn_vf8w", "k_q8row",
           "k_attn_sq8", "k_attn_vq8", "k_store_k8v4",
           "k_attn_sq8f", "k_attn_vq8f", "k_deq_emb4",
           "k_attn_sq8t", "k_attn_vq8u", "k_attn_vq8g", "k_xrmsN", "k_attn_vq8w", "k_attn_vq4w", "k_g5r4", "k_g5r8", "k_attn_sq8d", "k_g5r1", "k_g5r2", "k_g5n4",
           "k_g5r1s", "k_g5r2s", "k_g5r4s", "k_g5r8s",
           "k_g5a1", "k_g5a2", "k_g5a4", "k_g5a8",
           "k_g5a1s", "k_g5a2s", "k_g5a4s", "k_g5a8s")



def _kv_mode(kv=None):
    """The plain engine's K/V cache format: fp16 by default, q8_0 on request. See Engine.__init__."""
    if kv is not None:
        if kv not in ("f16", "q8_0"):
            raise SystemExit("kindi: kv must be f16 or q8_0 (got %r)" % (kv,))
        return kv
    env = os.environ.get("KINDI_KV", "").strip().lower()
    if env in ("q8", "q8_0"):
        return "q8_0"
    if env in ("k8v4", "k8_v4"):
        raise SystemExit("kindi: KINDI_KV=k8v4 is not supported by the plain engine; use the plan "
                         "engine (KINDI_TP=1) or KINDI_KV=q8_0")
    if env not in ("", "f16", "fp16"):
        raise SystemExit("kindi: KINDI_KV must be f16 or q8_0 for the plain engine (got %r)" % env)
    return "f16"


def _no_thp():
    """ROCm: no khugepaged collapses in this process (they evict the GPU queues mid-kernel; see AMD-NOTES.md)"""
    if cp is None or not getattr(cp.cuda.runtime, "is_hip", False) or os.environ.get("KINDI_THP", "0") == "1":
        return
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)       # PR_SET_THP_DISABLE, inherited by children
    except (OSError, AttributeError):
        pass


_no_thp()

def _upload(a):
    """host array -> card with a plain synchronous copy: cp.asarray stages it in a pinned host buffer of the same size,
    and a GB-sized cudaHostAlloc fails (cudaErrorInvalidValue) on a host that is short of free RAM"""
    g = cp.empty(a.shape, dtype=a.dtype)
    g.set(np.ascontiguousarray(a), stream=cp.cuda.Stream.null)
    return g


class Kernels(object):
    def __init__(self):
        mod = cp.RawModule(code=open(CU, encoding="utf-8").read())
        for k in KERNELS:
            setattr(self, k, mod.get_function(k))



def file_to_card(path, dtype, chunk=1 << 28):
    """a binary file -> device array, read 256 MB at a time; each chunk's page cache is dropped after the copy"""
    dt = np.dtype(dtype)
    n = os.path.getsize(path) // dt.itemsize
    out = cp.empty(n, dtype=dt)
    step = max(1, chunk // dt.itemsize)
    with open(path, "rb") as f:
        o = 0
        while o < n:
            a = np.fromfile(f, dtype=dt, count=min(step, n - o))
            out[o:o + a.size].set(a)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), o * dt.itemsize, a.size * dt.itemsize, os.POSIX_FADV_DONTNEED)
            o += a.size
            del a
    return out

class Engine(object):
    """One sequence at a time. step(tok, pos) -> next token (greedy) with the logits left in self.ylog;
    feed(tok, pos) processes a prompt token without the output head."""

    def __init__(self, pack, ctx=4096, clusters=None, log=print, layers=None, embed=True, head=True, kv=None):
        t0 = time.time()
        self.log = log
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        if man.get("pack_format") != 1:
            raise SystemExit("kindi: pack format %r is not supported - re-run kindi convert" % man.get("pack_format"))
        self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))          # Hadamard block of the folded inputs (2B/4B 512, 9B 4096)
        self.head_name = man.get("head", "token_embd.weight")   # the tied embedding, or a separate output.weight
        self.D, self.NL, self.NFF = self.dims["width"], self.dims["layers"], self.dims["ffn"]
        self.NH, self.NHKV, self.HD = self.dims["heads"], self.dims["kv_heads"], self.dims["head_dim"]
        self.NK, self.HK = self.dims["groups"], self.dims["state_size"]
        self.NV = self.dims["gdn_inner"] // self.HK
        self.CONV_CH = 2 * self.NK * self.HK + self.NV * self.HK
        self.NATT = self.NL // self.dims["full_attention_interval"]
        # the K/V cache format: fp16 by default (the 2B/4B/9B references are fp16, so a q8 cache would not
        # reproduce them); KINDI_KV=q8_0 opts in, and the tandem's plain GPU owner passes kv="q8_0" so a
        # split frees the offloaded layers' K/V as well as their weights. k8v4 is not served here.
        kv = _kv_mode(kv)
        if kv == "q8_0" and self.HD != 256:
            raise SystemExit("kindi: the q8_0 K/V cache needs a head size of 256 (this pack has %d)" % self.HD)
        self.kv = kv
        if self.NV % self.NK:
            raise SystemExit("kindi: unsupported grouped head dimensions")
        self.G = self._gqa_groups(self.NH, self.NHKV, self.HD)
        # A tandem split loads only the layers it runs on this backend: the other layers' tensors never reach
        # the card, so the GPU side needs less VRAM. The full-path arrays (M, conv, Kc, Vc) stay indexed by the
        # original layer number and have None entries for the layers another backend owns.
        if layers is None:
            self.layers = tuple(range(self.NL))
            self._full_layers = True
        else:
            raw = tuple(int(x) for x in layers)
            if not raw or len(set(raw)) != len(raw) or raw != tuple(sorted(raw)):
                raise SystemExit("kindi: a layer range must be a non-empty, sorted list of distinct layer numbers")
            if raw[0] < 0 or raw[-1] >= self.NL:
                raise SystemExit("kindi: layer range %r is outside 0..%d" % (raw, self.NL - 1))
            self.layers = raw
            self._full_layers = raw == tuple(range(self.NL))
        if not self._full_layers and clusters is not None:
            raise SystemExit("kindi: a partial layer range cannot use the fast head")
        self._layer_set = frozenset(self.layers)
        self.need_embed, self.need_head = bool(embed), bool(head)
        if self.need_embed and 0 not in self._layer_set:
            raise SystemExit("kindi: embedding needs layer 0 in the layer range")
        if self.need_head and self.NL - 1 not in self._layer_set:
            raise SystemExit("kindi: the output head needs the last layer in the layer range")
        self._kvi = {}
        kvi = 0
        for il in range(self.NL):
            if (il + 1) % self.dims["full_attention_interval"] == 0:
                self._kvi[il] = kvi
                kvi += 1
        self.inv_freq = np.float64(self.dims["rope_base"]) ** (-2.0 * np.arange(self.dims["rope_dim"] // 2) / self.dims["rope_dim"])
        self.k = Kernels()
        self._f32 = {}
        self.split_groups = int(os.environ.get("KINDI_SPLIT_GROUPS", "128"))   # column-split a single GEMV below this many 16-row groups per SM
        # one GEMV block per multiprocessor (K420: 1 SMX, K1200: 4 SMMs); a 16-row group stays in one warp, so the
        # sums and the output are the same for any count
        prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.nsm = int(os.environ.get("KINDI_GEMV_BLOCKS") or prop["multiProcessorCount"])
        unit = {3: "SMX", 5: "SMM"}.get(prop["major"], "SM")       # Kepler SMX, Maxwell SMM, later generations SM
        log("GPU: %s, sm_%d%d, %d %s%s, %d MiB" % (prop["name"].decode(), prop["major"], prop["minor"],
                                                   prop["multiProcessorCount"], unit,
                                                   "" if prop["multiProcessorCount"] == 1 else "s",
                                                   prop["totalGlobalMem"] // 2 ** 20))
        self.idx = json.load(open(os.path.join(pack, "index.json")))
        self.bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        self._all_bi = self.bi
        if layers is not None:
            self.bi = self._select_layers(self.bi, self.layers, self.need_embed, self.need_head, self.head_name)
        self.hsl = None
        self.hs_force = None
        if clusters is not None:
            # the special tokens (end of text, chat markers, <tool_call>, </think>, ...) go in one extra cluster that the
            # fast head always scores: in k-means clusters they can stay out of the shortlist for good (the 9B never
            # closed its thinking with the fast head)
            assign = np.array(clusters["assign"], dtype=np.int64)
            cent = np.asarray(clusters["centroids"], dtype=np.float32)
            self.hs_force = cent.shape[0]
            assign[FIRST_SPECIAL:] = self.hs_force
            clusters = {"assign": assign, "centroids": np.vstack([cent, np.zeros((1, cent.shape[1]), np.float32)])}
            bc = np.fromfile(os.path.join(pack, "bcodes.bin"), dtype=np.uint8)
            dp = np.fromfile(os.path.join(pack, "dpm.bin"), dtype=np.uint16)
            bc, dp, self.hsl = layout.reorder_head(self.bi, bc, dp, clusters, self.head_name)
            dims = layout.repack(self.bi, bc, dp)
            self.bc, self.dp = _upload(bc), _upload(dp)
            del bc, dp
        else:
            # tensor by tensor from memory maps of the pack, repacked on the host and copied to the card: the host
            # holds one tensor, not the weights (a 2 GiB container runs the 9B)
            bcm = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
            dpm = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.uint16, mode="r")
            dims = {}
            if self._full_layers:
                self.bc, self.dp = cp.zeros(bcm.size, dtype=cp.uint8), cp.zeros(dpm.size, dtype=cp.uint16)
                for name, rec in self.bi.items():
                    ns, P = rec["ne0"] // 4, rec["ne0"] // 128
                    o, d = rec["bc_off"], rec["dpm_off"] // 2
                    b, q = np.array(bcm[o:o + ns * rec["rows"]]), np.array(dpm[d:d + P * rec["rows"]])
                    dims.update(layout.repack({name: dict(rec, bc_off=0, dpm_off=0)}, b, q))
                    self.bc[o:o + b.size].set(b)
                    self.dp[d:d + q.size].set(q)
            else:
                # Compact arrays: only the selected tensors are uploaded, and the local offsets below are
                # what self.w indexes. This is the VRAM saving of a tandem split, not a view of the full pack.
                total_b = sum((rec["ne0"] // 4) * rec["rows"] for rec in self.bi.values())
                total_d = sum((rec["ne0"] // 128) * rec["rows"] for rec in self.bi.values())
                self.bc, self.dp = cp.zeros(total_b, dtype=cp.uint8), cp.zeros(total_d, dtype=cp.uint16)
                local, ob, od = {}, 0, 0
                for name, rec in self.bi.items():
                    ns, P = rec["ne0"] // 4, rec["ne0"] // 128
                    o, d = rec["bc_off"], rec["dpm_off"] // 2
                    b, q = np.array(bcm[o:o + ns * rec["rows"]]), np.array(dpm[d:d + P * rec["rows"]])
                    dims.update(layout.repack({name: dict(rec, bc_off=0, dpm_off=0)}, b, q))
                    self.bc[ob:ob + b.size].set(b)
                    self.dp[od:od + q.size].set(q)
                    local[name] = dict(rec, bc_off=ob, dpm_off=od * 2)
                    ob += b.size
                    od += q.size
                self.bi = local
            del bcm, dpm
        self.w = {n: (self.bc[self.bi[n]["bc_off"]:], self.dp[self.bi[n]["dpm_off"] // 2:], np.int32(R), np.int32(C))
                  for n, (R, C) in dims.items()}
        self.fl = file_to_card(os.path.join(pack, "f32.bin"), cp.float32)
        # Hadamard sign vectors, one per folded input width: D (norm outputs), NV*HK (GDN output), NH*HD (attention
        # output; 2048 = D on the 2B, 4096 != D on the 4B) and NFF (FFN down)
        sg = lambda w: cp.asarray(np.fromfile(os.path.join(pack, "signs_%d.f32" % w), dtype=np.float32))  # noqa: E731
        self.sD, self.sV, self.sA, self.sF = sg(self.D), sg(self.NV * self.HK), sg(self.NH * self.HD), sg(self.NFF)
        # cp.asarray stages each upload through pinned host memory and CuPy keeps those blocks: give them back (the
        # 9B weights left ~4 GB of host RAM pinned)
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
        Rh = VOCAB if self.hsl is None else self.hsl["Rp"]
        work = max(self.D, self.NFF, self.NV * self.HK)
        self.y, self.xp, self.h, self.xn, self.zv = z(max(Rh, self.D)), z(work), z(self.D), z(self.D), z(self.NV * self.HK)
        self.h2 = z(self.D)
        self.part = z(self.nsm * max(self.D, self.NFF, self.NV * self.HK))   # column-split GEMV partial sums
        self.qkv, self.cs = z(self.CONV_CH), z(self.CONV_CH)
        self.qkbuf, self.qkvbuf = z(2 * self.NK * self.HK), z(self.NHKV * self.HD)
        self.kn, self.qn, self.qr, self.kro = z(self.NHKV * self.HD), z(self.NH * 2 * self.HD), z(self.NH * self.HD), z(self.NHKV * self.HD)
        self.fo, self.fo6 = z(self.NH * self.HD), z(self.NFF)
        self.outgdn, self.att, self.ffg, self.ffu = z(self.NV * self.HK), z(self.NH * self.HD), z(self.NFF), z(self.NFF)
        self.gb, self.gg, self.gout = z(self.NV), z(self.NV), z(self.NV * self.HK)
        if self.kv == "q8_0":
            self.qq, self.qd = cp.zeros(self.NH * self.HD, dtype=cp.int8), z(self.NH * self.HD // 32)
        # fit the context into the memory that is left (a 9B on a 4 GB card, a 2B on a 1 GB card): bytes per position
        # = K/V cache (fp16) + attention scores + attention partials + RoPE tables
        from . import chunk1 as _chunk1
        if self._full_layers and self.kv != "q8_0":
            _chunk1.reserve(self)                  # the chunked prompt path's buffers, before the context fit
        elif self.kv == "q8_0":
            # the chunked prompt path (chunk1) writes an fp16 cache; with q8 the prompt is fed one token
            # per call through attn() (correct, at the decode rate), exactly as a tandem partial engine is
            self.chunk_ok, self.chunk, self.batch_max = False, {}, 1
        else:
            self.batch_max = 1                     # a tandem engine feeds one token per backend call
        # Only the attention layers this engine owns hold K/V here. A tandem's GPU owner loads a subset, so
        # charging it for every layer of the model reduced the context more than the memory actually allowed
        # (the 27B at 24,576 with 8 CPU layers was cut to 17,664 tokens that way).
        owned_attn = self.NATT if self._full_layers else sum(1 for il in self.layers if il in self._kvi)
        # q8_0: one int8 per value plus one fp16 scale per 32 values, for K and V (the plan engine's bytes)
        kv_per_pos = owned_attn * 2 * self.NHKV * (self.HD + self.HD // 32 * 2) if self.kv == "q8_0" \
            else owned_attn * 2 * self.NHKV * self.HD * 2
        per_pos = (kv_per_pos + self.NH * 4 + (self.NH * self.HD * 4) // 256
                   + self.dims["rope_dim"] * 4)
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        budget = free - int(os.environ.get("KINDI_MEM_RESERVE_MB", "96")) * 2 ** 20
        fit = max(0, budget) // per_pos // 256 * 256
        if fit < self.ctx:
            if fit < 1024:
                raise SystemExit("kindi: not enough GPU memory for this model: %d MiB free after the weights"
                                 % (free // 2 ** 20))
            log("context reduced from %d to %d tokens to fit %d MiB of free GPU memory" % (self.ctx, fit, free // 2 ** 20))
            self.ctx = fit
        self.sbuf = z(self.NH * self.ctx)
        self.apart = z(((self.ctx + 255) // 256) * self.NH * self.HD)   # attention partials: NH x HD per 256-position chunk
        self.aout = cp.zeros(1, dtype=cp.int32)
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * self.inv_freq[None, :]
        self.cos_t = cp.asarray(np.cos(th).astype(np.float32))
        self.sin_t = cp.asarray(np.sin(th).astype(np.float32))
        # recurrent (GDN) state for the 18 GDN layers; fp16 K/V cache for the 6 attention layers
        interval = self.dims["full_attention_interval"]
        self.M = [None] * self.NL
        self.conv = [None] * self.NL
        for il in self.layers:
            self.M[il] = cp.zeros((self.NV, self.HK, self.HK) if (il + 1) % interval else (1,), dtype=cp.float32)
            self.conv[il] = z(3 * self.CONV_CH)
        self.Kc = [None] * self.NATT
        self.Vc = [None] * self.NATT
        self.Kq = self.Kd = self.Vq = self.Vd = None
        if self.kv == "q8_0":
            from .tp import Dist
            self.Kq, self.Kd, self.Vq, self.Vd = [None] * self.NATT, [None] * self.NATT, [None] * self.NATT, [None] * self.NATT
            for il in self.layers:
                if il in self._kvi:
                    kvi = self._kvi[il]
                    self.Kq[kvi] = cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.int8)
                    self.Kd[kvi] = cp.zeros((self.ctx, self.NHKV, self.HD // 32), dtype=cp.float16)
                    self.Vq[kvi] = cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.int8)
                    self.Vd[kvi] = cp.zeros((self.ctx, self.NHKV, self.HD // 32), dtype=cp.float16)
                    # the cache copies K/V as one byte stream per layer (checkpoints, parked conversations):
                    # a Dist over the quantised row and its scales, as the plan engine exposes them
                    self.Kc[kvi] = Dist([(cp.cuda.Device().id, self.Kq[kvi]), (cp.cuda.Device().id, self.Kd[kvi])], rows=True)
                    self.Vc[kvi] = Dist([(cp.cuda.Device().id, self.Vq[kvi]), (cp.cuda.Device().id, self.Vd[kvi])], rows=True)
        else:
            for il in self.layers:
                if il in self._kvi:
                    self.Kc[self._kvi[il]] = cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.float16)
                    self.Vc[self._kvi[il]] = cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.float16)
        self.shortlist = 0                 # > 0: fast head over this many clusters
        if self.hsl is not None:
            m = self.hsl
            self.hs_cent = cp.asarray(m["cent"])
            self.hs_permtok, self.hs_permtok_h, self.hs_permpos = cp.asarray(m["permtok"]), m["permtok"], m["permpos"]
            self.hs_gstart, self.hs_gcount = cp.asarray(m["gstart"]), cp.asarray(m["gcount"])
            self.hs_glist = cp.zeros(m["Rp"] // 16, dtype=cp.int32)
            self.hs_yc = z(m["Rp"])
            self.hs_tokc = cp.zeros(m["Rp"], dtype=cp.int32)
        self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
        log("engine ready: ctx %d, head %s (%.1fs)" % (self.ctx, "reordered" if self.hsl else "plain", time.time() - t0))

    # ------------------------------------------------------------------------------------------- helpers
    def f32(self, name):
        v = self._f32.get(name)              # cached: a fresh view per call costs Python time on every launch
        if v is None:
            rec = self.idx["tensors"][name]
            off = rec["f32_off"] // 4
            v = self._f32[name] = self.fl[off:off + rec["rows"] * rec["ne0"]]
        return v

    def gemv(self, names, x, outs, parts_ok=False):
        """Up to three ternary tensors with the same input in one launch."""
        a = [self.w[n] for n in names]
        args = []
        for i in range(3):
            if i < len(a):
                args += [a[i][0], a[i][1], outs[i], a[i][2]]
            else:
                args += [a[0][0], a[0][1], outs[0], np.int32(0)]
        R0, C = int(a[0][2]), int(a[0][3])
        if len(a) == 1 and self.nsm > 1 and (R0 >> 4) < self.split_groups * self.nsm and (C >> 9) >= self.nsm:
            # few row groups for many columns (the 4B's 2560-row outputs): each block takes 1/nsm of the columns of
            # every row, so a phase has 4x the work for the same table build and barriers; then add the parts
            args[2] = self.part
            self.k.k_g4((1, self.nsm), (1024,), tuple(args) + (x, a[0][3]))
            if parts_ok:                          # the next rmsfold adds the parts (a block output in self.part)
                return self.nsm
            self.k.k_sumparts(((R0 + 255) // 256,), (256,), (self.part, outs[0], np.int32(R0), np.int32(self.nsm)))
            return 1
        self.k.k_g4((self.nsm,), (1024,), tuple(args) + (x, a[0][3]))
        return 1

    def fold_args(self, width):
        """Hadamard blocks above 512: the fold kernels do 512-point chunks on all SMs (HB = -512, unscaled) and
        k_fwht_hi finishes the block. Returns (grid, HB argument)."""
        if self.HB > 512:
            return (width // 512,), np.int32(-512)
        return (width // self.HB,), np.int32(self.HB)

    def fold_hi(self, buf, width):
        if self.HB > 512:
            self.k.k_fwht_hi(((width // 512 // (self.HB // 512) * 512 + 255) // 256,), (256,),
                             (buf, np.int32(self.HB), np.int32(width)))

    def rmsfold(self, x, wname):
        """RMS norm + Hadamard fold of the residual stream (self.h; x is ignored). If a block left its output in
        self.y, the kernel adds it first (that block's residual add, fused here) and writes the sum to the second
        buffer, which then becomes self.h."""
        grid, hb = self.fold_args(self.D)
        self.k.k_rmsfold(grid, (512,), (self.h, self.part if self.pending > 1 else self.y, self.h2,
                                        self.f32(wname), self.sD, self.xn,
                                        self.xp, np.int32(self.D), np.int32(self.pending), hb))
        self.fold_hi(self.xp, self.D)
        if self.pending:
            self.h, self.h2 = self.h2, self.h
        self.pending = 0
        return self.xn

    def rms_rows(self, x, wname, rows, n, out):
        self.k.k_rmsw(((rows + 7) // 8,), (256,), (x, self.f32(wname), out, np.int32(n), np.int32(rows)))
        return out

    @staticmethod
    def _gqa_groups(nh, nhkv, hd):
        """Query heads per K/V head. The fp16 decode kernels cover a whole group: 4:1 is the fast path of the
        2B/4B/9B, and the generic kernels cover G <= 8 at head size 256 (the 27B's 6:1 attention)."""
        if nhkv <= 0 or nh % nhkv:
            raise SystemExit("kindi: unsupported grouped head dimensions")
        g = nh // nhkv
        if not 1 <= g <= 8 or (g != 4 and hd != 256):
            raise SystemExit("kindi: unsupported grouped head dimensions")
        return g

    @staticmethod
    def _select_layers(bi, layers, embed, head, head_name):
        """The tensor records this backend needs, for a partial layer range."""
        prefixes = tuple("blk.%d." % il for il in layers)
        out = {name: rec for name, rec in bi.items() if name.startswith(prefixes)}
        if embed:
            out["token_embd.weight"] = bi["token_embd.weight"]
        if head:
            out[head_name] = bi[head_name]
        return out

    def reset(self):
        state = self.M + self.conv
        if self.kv == "q8_0":
            state += self.Kq + self.Kd + self.Vq + self.Vd
        else:
            state += self.Kc + self.Vc
        for a in state:
            if a is not None:
                a.fill(0)

    def _require_full_layers(self, what):
        if not self._full_layers:
            raise RuntimeError("kindi: %s needs the full layer set; use begin/layer/finish" % what)

    def embed(self, tok):
        if not self.need_embed:
            raise RuntimeError("kindi: this engine does not own the embedding")
        a = self.w["token_embd.weight"]
        # the embedding rows are in cluster order only when the head is the tied embedding
        row = int(self.hs_permpos[tok]) if self.hsl is not None and self.head_name == "token_embd.weight" else int(tok)
        self.k.k_deq_emb3((1,), (1024,), (a[0], a[1], self.sD, self.h, np.int32(row), np.int32(self.D), np.int32(self.HB)))
        return self.h

    # ------------------------------------------------------------------------------------------- blocks
    def gdn(self, il, h):
        p = "blk.%d." % il
        xn = self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_qkv.weight", p + "attn_gate.weight"], self.xp, [self.qkv, self.zv])
        self.k.k_gdn_gates((self.NV,), (256,), (xn, self.f32(p + "ssm_beta.weight"), self.f32(p + "ssm_alpha.weight"),
                                           self.f32(p + "ssm_dt.bias"), self.f32(p + "ssm_a"), self.gb, self.gg,
                                           np.int32(self.D)))
        self.k.k_gdn_conv(((self.CONV_CH + 255) // 256,), (256,),
                          (self.qkv, self.conv[il], self.f32(p + "ssm_conv1d.weight"), self.cs, np.int32(self.CONV_CH)))
        self.k.k_l2w((4,), (256,), (self.cs, self.qkbuf, np.int32(self.HK), np.int32(2 * self.NK)))
        self.k.k_gdn_delta((self.NV,), (self.HK,), (self.M[il], self.qkbuf[:self.NK * self.HK], self.qkbuf[self.NK * self.HK:], self.cs[2 * self.NK * self.HK:],
                                          self.gg, self.gb, self.gout, np.float32(1.0 / np.sqrt(self.HK)), np.int32(self.NK)))
        outn = self.rms_rows(self.gout, p + "ssm_norm.weight", self.NV, self.HK, self.outgdn)
        grid, hb = self.fold_args(self.NV * self.HK)
        self.k.k_smfold(grid, (512,), (self.zv, outn, self.sV, self.xp, np.int32(self.HK), np.int32(self.NK), np.int32(self.NV // self.NK), hb))
        self.fold_hi(self.xp, self.NV * self.HK)
        npart = self.gemv([p + "ssm_out.weight"], self.xp, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def attn(self, il, h, pos, kvi):
        p = "blk.%d." % il
        self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], self.xp,
                  [self.qn, self.kro, self.qkvbuf])
        Kn = self.rms_rows(self.kro, p + "attn_k_norm.weight", self.NHKV, self.HD, self.kn)
        qc = self.qn.reshape(self.NH, 2 * self.HD)[:, :self.HD].copy()
        q = self.rms_rows(qc, p + "attn_q_norm.weight", self.NH, self.HD, self.fo)
        cs, sn = self.cos_t[pos], self.sin_t[pos]
        self.k.k_rope((self.NH,), (256,), (q, cs, sn, self.qr, np.int32(self.HD)))
        self.k.k_rope((self.NHKV,), (256,), (Kn, cs, sn, self.kro, np.int32(self.HD)))
        n = pos + 1
        if self.kv == "q8_0":
            nblk = self.NHKV * self.HD // 32
            self.k.k_store_q8(((64 * nblk + 1023) // 1024,), (min(1024, 64 * nblk),),
                              (self.kro, self.qkvbuf, self.Kq[kvi][pos], self.Kd[kvi][pos],
                               self.Vq[kvi][pos], self.Vd[kvi][pos], np.int32(nblk)))
        else:
            self.Kc[kvi][pos] = self.kro.reshape(self.NHKV, self.HD)
            self.Vc[kvi][pos] = self.qkvbuf.reshape(self.NHKV, self.HD)
        self._attn_core(self.Kc[kvi][:n] if self.kv != "q8_0" else None,
                        self.Vc[kvi][:n] if self.kv != "q8_0" else None, n, (n + 255) // 256, kvi)
        grid, hb = self.fold_args(self.NH * self.HD)
        self.k.k_fold_fwht(grid, (512,), (self.att, self.sA, self.xp, hb))
        self.fold_hi(self.xp, self.NH * self.HD)
        npart = self.gemv([p + "attn_output.weight"], self.xp, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def _attn_core(self, Kall, Vall, n, nch, kvi=None):
        """Scores, softmax, values and the gated output of one attention layer into self.att. The 4:1 models use the
        fast kernels; the generic GQA kernels run one warp per position (8 positions per block)."""
        if getattr(self, "kv", "f16") == "q8_0":
            # the plan engine's q8_0 decode attention (k_attn_sq8t + k_attn_vq8w + k_attn_combn), for every G
            self.k.k_q8row(((self.NH * self.HD + 1023) // 1024,), (1024,),
                           (self.qr, self.qq, self.qd, np.int32(self.NH * self.HD // 32)))
            self.k.k_attn_sq8t((self.NHKV, (self.ctx + 255) // 256), (256,),
                               (self.Kq[kvi], self.Kd[kvi], self.qq, self.qd, self.sbuf,
                                np.int32(n), np.int32(self.NHKV), np.int32(self.G)))
            self.k.k_softmax_rows((self.NH,), (256,), (self.sbuf, np.int32(n)))
            self.k.k_attn_vq8w((self.NHKV, 1, nch), (128,),
                               (self.sbuf, self.Vq[kvi], self.Vd[kvi], self.apart,
                                np.int32(n), np.int32(self.NH), np.int32(self.NHKV), np.int32(self.G)))
            self.k.k_attn_combn((self.NH,), (256,), (self.apart, self.qn, self.att, np.int32(n), np.int32(self.NH)))
            return
        if self.G == 4:
            self.k.k_attn_s4w((self.NHKV, min(32, (n + 31) // 32)), (256,),
                              (Kall, self.qr, self.sbuf, np.int32(n), np.int32(self.NHKV)))
            self.k.k_softmax_rows((self.NH,), (256,), (self.sbuf, np.int32(n)))
            self.k.k_attn_v4p((self.NHKV, 1, nch), (128,),
                              (self.sbuf, Vall, self.apart, np.int32(n), np.int32(self.NH), np.int32(self.NHKV)))
            self.k.k_attn_comb((self.NH,), (256,),
                               (self.apart, self.qn, self.att, np.int32(nch), np.int32(self.NH)))
            return
        self.k.k_attn_sg((self.NHKV, min(32, (n + 7) // 8)), (256,),
                         (Kall, self.qr, self.sbuf, np.int32(n), np.int32(self.NHKV), np.int32(self.G)))
        self.k.k_softmax_rows((self.NH,), (256,), (self.sbuf, np.int32(n)))
        self.k.k_attn_vgn((self.NHKV, 1, nch), (128,),
                          (self.sbuf, Vall, self.apart, np.int32(n), np.int32(self.NH), np.int32(self.NHKV),
                           np.int32(self.G)))
        self.k.k_attn_combn((self.NH,), (256,), (self.apart, self.qn, self.att, np.int32(n), np.int32(self.NH)))

    def ffn(self, il, h):
        p = "blk.%d." % il
        self.rmsfold(h, p + "post_attention_norm.weight")
        self.gemv([p + "ffn_gate.weight", p + "ffn_up.weight"], self.xp, [self.ffg, self.ffu])
        grid, hb = self.fold_args(self.NFF)
        self.k.k_smfold(grid, (512,), (self.ffg, self.ffu, self.sF, self.fo6, np.int32(0), np.int32(0), np.int32(0), hb))
        self.fold_hi(self.fo6, self.NFF)
        npart = self.gemv([p + "ffn_down.weight"], self.fo6, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def body(self, tok, pos):
        self._require_full_layers("body")
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        h = self.embed(tok)
        self.pending = 0
        kvi = 0
        for il in range(self.NL):
            if (il + 1) % self.dims["full_attention_interval"] == 0:
                h = self.attn(il, h, pos, kvi)
                kvi += 1
            else:
                h = self.gdn(il, h)
            h = self.ffn(il, h)
        return h

    def feed(self, tok, pos):
        """A prompt token whose logits are not needed."""
        self.body(tok, pos)

    def feed_batch(self, toks, pos):
        """Prompt tokens pos ..: whole chunks through kindi.chunk1 (KINDI_CHUNK=0: off), the rest one by one."""
        from . import chunk1 as _chunk1
        _chunk1.feed_batch(self, toks, pos)

    def step(self, tok, pos):
        h = self.body(tok, pos)
        self.rmsfold(h, "output_norm.weight")
        return self.head()

    # ------------------------------------------------------------------------------------------- tandem handoff
    def begin(self, tok):
        """Embed one token into self.h; the layer() calls that follow run a partial layer range."""
        if not self.need_embed:
            raise RuntimeError("kindi: this engine does not own the embedding")
        self.h = self.embed(tok)
        self.pending = 0

    def layer(self, il, pos):
        """Run one complete block on self.h, which holds the full residual entering the block."""
        il = int(il)
        if il not in self._layer_set:
            raise IndexError("layer %d is not owned by this engine" % il)
        if not 0 <= pos < self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        if il in self._kvi:
            self.attn(il, self.h, pos, self._kvi[il])
        else:
            self.gdn(il, self.h)
        self.ffn(il, self.h)

    def flush(self):
        """Materialize the residual add left in self.y by the last GDN/attention/FFN output."""
        if self.pending:
            if self.pending > 1:
                self.k.k_sumparts(((self.D + 255) // 256,), (256,), (self.part, self.y, np.int32(self.D),
                                                                         np.int32(self.pending)))
            self.h += self.y[:self.D]
            self.pending = 0

    def finish(self):
        """Apply the output norm and exact head to the residual after a partial layer range."""
        if not self.need_head:
            raise RuntimeError("kindi: this engine does not own the output head")
        self.flush()
        self.rmsfold(self.h, "output_norm.weight")
        return self.head()

    def get_hidden(self):
        """Copy the residual after the last layer to a NumPy array for the other backend."""
        self.flush()
        return cp.asnumpy(self.h).copy()

    def set_hidden(self, values):
        """Accept the other backend's residual into this backend's layer input."""
        x = np.asarray(values, dtype=np.float32)
        if x.shape != (self.D,):
            raise ValueError("hidden state has shape %r, expected (%d,)" % (x.shape, self.D))
        self.h[:] = cp.asarray(x)
        self.pending = 0

    # ------------------------------------------------------------------------------------------- output head
    def head(self):
        """-> greedy next token id. Leaves the logits in self.ylog[:self.ylog_n], their argmax slot in self.aout and
        (reordered head) each slot's token id in self.ylog_map."""
        if not self.need_head:
            raise RuntimeError("kindi: this engine does not own the output head")
        a = self.w[self.head_name]
        if self.hsl is None:
            self.gemv([self.head_name], self.xp, [self.y])
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(VOCAB), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
            return int(self.aout[0])
        Rp = self.hsl["Rp"]
        M = self.shortlist
        if M <= 0 or M >= self.hsl["K"]:
            self.gemv([self.head_name], self.xp, [self.y])
            self.k.k_mask_neg((64,), (256,), (self.y, self.hs_permtok, np.int32(Rp)))
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(Rp), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, Rp, self.hs_permtok
            return int(self.hs_permtok_h[int(self.aout[0])])
        cs = self.hs_cent @ self.xp[:self.D]
        cs[self.hs_force] = 3.0e38                      # the special-token cluster is always in the shortlist
        sel = cp.argsort(-cs)[:M].astype(cp.int32)
        cnt = self.hs_gcount[sel]
        off = (cp.cumsum(cnt) - cnt).astype(cp.int32)
        ng = int(cnt.sum())
        self.k.k_expand((M,), (128,), (sel, self.hs_gstart, self.hs_gcount, off, self.hs_glist))
        zr = np.int32(0)
        self.k.k_g4l((self.nsm,), (1024,), (a[0], a[1], self.hs_yc, a[2], a[0], a[1], self.hs_yc, zr, a[0], a[1], self.hs_yc,
                                     zr, self.xp, a[3], self.hs_glist, np.int32(ng), self.hs_permtok, self.hs_tokc))
        n = ng * 16
        self.k.k_mask_neg((16,), (256,), (self.hs_yc, self.hs_tokc, np.int32(n)))
        self.k.k_argmax((1,), (1024,), (self.hs_yc, np.int32(n), self.aout))
        self.ylog, self.ylog_n, self.ylog_map = self.hs_yc, n, self.hs_tokc
        return int(self.hs_tokc[int(self.aout[0])])

    def generate(self, ids, n):
        """Greedy: the n tokens after the prompt ids (fresh state)."""
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out


PLAN_SMALL_HIP = ("gfx1151",)      # the AMD GPUs where the 2B / 4B / 9B take the plan engine (measured)


def tc_hp_for(man):
    """The activation precision the plan engine should use for this pack: "1", or None to leave the
    environment alone.

    A pack folded with Hadamard block 4096 (the 9B here) passes the tensor-core verify 4/4 only with the
    high-precision activations (about 14 bits instead of 8, same weight traffic); with the default ones one
    borderline prompt flips at token 36 (3/4), which is why `kindi/spec.py` refuses HB 4096 without them.
    Measured on the RTX 4060 Ti: 9B 4/4 at 122.6 tok/s against 74.6-80.0 on the plain engine. So for those
    packs the exact setting is the default rather than a flag every user has to find. An explicit
    `KINDI_TC_HP` always wins, including `0` (which then stops with the measured explanation).

    It is read once, when `kindi.spec` is imported, and caps the verify pass at T = 4, so a process that
    loads an HB-4096 pack first keeps that cap for any later pack.
    """
    if "KINDI_TC_HP" in os.environ:
        return None
    try:
        hb = int(man.get("hadamard_block", 512))
    except (TypeError, ValueError):
        return None
    return "1" if hb == 4096 else None


# the `backend` values make_engine_legacy can build for. A dense Bonsai pack writes none: only the
# Gemma 4, Nanbeige, Lightning, Cascade and DeepSeek V4.1 converters/writers set the key, so an
# absent/empty value is the dense family.
DENSE_BACKENDS = (None, "")
# families that route expert work instead of running every weight on every token. The dense tandem
# layer split has no meaning for them: their CPU/GPU split is the expert tiering inside their engine.
MOE_BACKENDS = ("kindi_lightning", "kindi_dsv41")


def adapter_model_id(man):
    """The registry model id a backend adapter claims for this pack manifest, or None.

    A manifest two adapters claim is contradictory, so it is refused rather than resolved to
    whichever hook runs first (that would silently build the other family's engine). The silent
    version of that same failure is a placeholder `model` field: the pack's own adapter misses its
    pack and the gateway quietly falls back to the legacy builder (T4.2's "Hf" 27B bug).
    """
    from .modelids import adapter_model_id as _resolve

    return _resolve(man)


def make_engine(pack, ctx=4096, clusters=None, log=print, tandem="off", cpu_layers=None, gpu=None):
    """Engine on one GPU (or the tandem engine): the T4.2 gateway.

    A pack whose manifest names a family resolves through that family's adapter in the backend
    registry (``bonsai``/``nanbeige``/``lightning``/``gemma4``/``dsv41``); ``KINDI_LEGACY_ENGINE_PATH=1``
    forces the dense engine builder for the dense family. Each Bonsai/Nanbeige/Lightning adapter
    builds the same engine object its family's legacy builder built, so serve/verify/bench numbers
    do not change; the Gemma 4 and DeepSeek V4.1 adapters are the only ``make_engine`` path for
    those families (their pre-port builders are inline in their servers).

    The legacy builder only builds the dense engines, so this refuses a pack that names another
    family it cannot build — the valve does not help there either — and it logs which path every
    pack took: a dense pack whose own adapter does not claim it (a placeholder `model`, the "Hf"
    27B case) is the one fallback that stays valid but must be visible.
    """
    man = None
    try:
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            man = json.load(f)
    except Exception:  # noqa: BLE001 - leave broken packs to the legacy builder's error
        man = None
    family = man.get("backend") if isinstance(man, dict) else None
    if os.environ.get("KINDI_LEGACY_ENGINE_PATH") == "1":
        if family not in DENSE_BACKENDS:
            raise SystemExit(
                "kindi: %s is a %r pack and KINDI_LEGACY_ENGINE_PATH=1 asks for the dense engine "
                "builder, which cannot build that family (the pre-port path lives in its own "
                "server, not in make_engine_legacy). Unset the variable to use its adapter."
                % (pack, family))
        log("kindi: KINDI_LEGACY_ENGINE_PATH=1: the legacy dense engine builder, no backend adapter")
        return make_engine_legacy(
            pack, ctx=ctx, clusters=clusters, log=log, tandem=tandem,
            cpu_layers=cpu_layers, gpu=gpu,
        )
    if str(tandem or "off").strip().lower().replace("_", "-") != "off" and family in MOE_BACKENDS:
        # Not a no-op to hide: the flag asked for a split, and the family's split is its expert tiering.
        log("kindi: %s is a mixture-of-experts pack (%s): the tandem runtime splits complete dense "
            "layers, which does not apply. This family's CPU/GPU split is the expert-tier path inside "
            "its own engine (dense core and shared expert on the card, routed experts host-resident "
            "with last-used device slots). Tandem stays off; `kindi plan --pack %s` prints that "
            "placement (`--dcache` sizes its report, KINDI_LT_DCACHE sizes the run)."
            % (pack, family, pack))
        tandem = "off"
    if isinstance(man, dict):
        model_id = adapter_model_id(man)
        if model_id:
            from .backends import resolve

            backend = resolve(
                model_id,
                config=dict(
                    pack=pack, ctx=ctx, clusters=clusters, log=log, tandem=tandem,
                    cpu_layers=cpu_layers, gpu=gpu,
                ),
            )
            log("kindi: %s -> the %s backend (%s)" % (pack, backend.capabilities.name, model_id))
            return backend.engine
        if family not in DENSE_BACKENDS:
            raise SystemExit(
                "kindi: %s is a %r pack and no backend adapter serves that family, so the gateway "
                "would build the dense Bonsai engine for it (the wrong engine). `kindi serve` and "
                "`kindi verify` route by `backend`." % (pack, family))
        if "bcodes_bytes" not in man:
            raise SystemExit(
                "kindi: %s has no `backend` field and no dense Bonsai weight fields "
                "(bcodes_bytes): it is not a pack this version can build - re-run "
                "`kindi convert` or `kindi use`" % (pack,))
        log("kindi: %s declares no backend and no adapter claims it (model %r): the legacy dense "
            "builder, which has no registry identity - check the manifest's `model`/`source_file`"
            % (pack, man.get("model")))
    return make_engine_legacy(
        pack, ctx=ctx, clusters=clusters, log=log, tandem=tandem,
        cpu_layers=cpu_layers, gpu=gpu,
    )


def make_engine_legacy(pack, ctx=4096, clusters=None, log=print, tandem="off", cpu_layers=None, gpu=None):
    """The pre-T4.2 engine builder (the ``KINDI_LEGACY_ENGINE_PATH=1`` path, and what the
    backend adapters delegate to): an engine on one GPU, or the two-card tensor parallel
    engine (kindi.tp) when the weights do not fit one card and two GPUs are present.
    KINDI_TP=1 / 0 forces the choice. tandem="cpu-first" or "gpu-first" splits the dense
    model between the CPU engine and one GPU engine."""
    if os.environ.get("KINDI_DEVICE", "gpu") == "intel":
        # Intel integrated/discrete graphics (REQ-0020/REQ-0021): discovery, eligibility and the
        # shared-memory budget exist (kindi/intel.py, the G0 slice), but the SYCL kernel backend
        # does not. Refuse loudly instead of silently running the CPU engine under an Intel name.
        raise SystemExit("kindi: KINDI_DEVICE=intel is not implemented yet - the SYCL backend "
                         "(REQ-0020/REQ-0021) needs hardware to verify. Use KINDI_DEVICE=cpu for "
                         "the CPU engine or the default CUDA/ROCm path for a discrete GPU.")
    mode = str(tandem or "off").strip().lower().replace("_", "-")
    if mode != "off":
        if os.environ.get("KINDI_DEVICE", "gpu") == "cpu" or cp is None:
            raise SystemExit("kindi: tandem needs a CUDA/ROCm GPU; KINDI_DEVICE=cpu is set or CuPy is missing")
        from .tandem import TandemEngine
        if clusters is not None:
            log("the fast head is not available in tandem mode: exact head")
        return TandemEngine(pack, ctx=ctx, cpu_layers=cpu_layers, mode=mode, log=log, gpu=gpu)
    if os.environ.get("KINDI_DEVICE", "gpu") == "cpu" or cp is None:
        from .cpu import Engine as CPUEngine
        if clusters is not None:
            log("the fast head is not available on the CPU: exact head")
        return CPUEngine(pack, ctx=ctx, log=log)
    from .sched import blocking_sync
    blocking_sync(log)                      # before the first CUDA call (KINDI_BLOCKING_SYNC=0: off)
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    force = os.environ.get("KINDI_TP", "")
    weights = int(man.get("bcodes_bytes", 0)) + int(man.get("dpm_bytes", 0))
    two = cp.cuda.runtime.getDeviceCount() >= 2
    dims = man.get("dimensions", {})
    # the plan engine (kindi.tp) also runs on one card: it is the one that has the 27B's kernels (6 q heads per K/V
    # head, Hadamard blocks of 1024, q8_0 K/V). It is also the faster engine on the models that have it:
    # 2B tc read 471 vs 190 plain, 4B 214 vs 129, 9B 134 vs 90 tok/s (CROSS-MODE tc rows). So by default the
    # plan engine is used for every model it supports, not only the 27B; KINDI_TP=0 keeps the plain engine.
    plan_model = True
    # AMD GPUs with many CUs (Strix Halo, gfx1151): the 2B, 4B and 9B (512- and 4096-point folds) also run faster on
    # the plan engine's 5-trit products (2B write 173 -> 259, 9B 52 -> 89 tokens/s); measured there only
    if cp.cuda.runtime.is_hip and int(man.get("hadamard_block", 512)) in (512, 4096) and not two:
        arch = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id).get("gcnArchName", b"")
        arch = (arch.decode() if isinstance(arch, bytes) else str(arch)).split(":")[0]
        plan_model = plan_model or arch in PLAN_SMALL_HIP
    if force == "1" or (force != "0" and (plan_model or (two and weights > 0.9 * cp.cuda.runtime.memGetInfo()[1]))):
        hp = tc_hp_for(man)
        if hp is not None:
            os.environ["KINDI_TC_HP"] = hp         # before kindi.spec is imported: it reads this once
            log("Hadamard block 4096: high-precision activations on (measured 4/4 at 122 tok/s; "
                "KINDI_TC_HP=0 asks for the default activations, measured 3/4)")
        from .tp import TPEngine
        from . import spec
        spec.install(log)                       # KINDI_TC=1: tensor-core plans (one sm_80+ card); no-op otherwise
        if clusters is not None:
            log("the fast head is not available with two cards: exact head")
        return TPEngine(pack, ctx=ctx, log=log)
    return Engine(pack, ctx=ctx, clusters=clusters, log=log)
