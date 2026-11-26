# SPDX-License-Identifier: AGPL-3.0-only
"""Run one real decode token on an Intel GPU, layer by layer, and compare it with the CPU engine.

`docs/INTEL-ARC-KERNELS.md` puts the kernels cheapest-first and leaves **K3 (the ternary GEMV wiring)**
as the remaining Phase A item: the matvec exists, the norm/fold exists (K1, `kindi/intel_layer.py`), but no
loop sequences them into a layer.  This probe is that loop.

What runs where, per layer (2B geometry: 24 layers, every 4th attention, the rest GDN):

- **Device:** K1's `rms_sum` + `fold_sign` (residual add, RMS, signed FWHT), and `pack_gemv4` for every
  ternary matrix (qkv+gate, ssm_out, ffn gate/up/down, q/k/v/o, head).  One enqueue per kernel, one
  blocking read per matrix; the residual stream itself stays on the host (uploads are 2 KB, the
  `h += y` adds are host numpy), so every enqueue is counted and timed.
- **Host (numpy, fp32, mirroring `kcpu.c`):** the GDN gates (beta/alpha), conv + silu, the per-head L2
  norms and the delta rule, the gate+permute, FFN `silu(g)*u`, attention norms/RoPE/KV/softmax/gated V,
  and the argmax.  The GDN `M`/`conv` state and the K/V cache are host arrays, reset per run.

The oracle is `kindi.cpu.Engine` decode on the same pack: the Arc token's argmax and logits are compared
against the CPU's at each of a few positions, with teacher forcing (both sides see the CPU's token), so a
divergence is local rather than compounded.  This is **not** a support claim: it is the plan's "first
honest token", measured, against the CPU engine, on the 2B pack that fits this host.

    python -m kindi intel-step --pack ~/.kindi/packs/bonsai-2b --device-name Arc --steps 4 --json out.json
"""
import ctypes
import json
import os
import time

import numpy as np

from .intel_cl import (
    CL_CONTEXT_PLATFORM,
    CL_MEM_COPY_HOST_PTR,
    CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY,
    CL_SUCCESS,
    DEFAULT_LOADER,
    _bind,
    _build_log,
    _enumerate,
    _finish,
    _hex,
    _open,
    _read_back,
    _set_arg,
    pick_device,
)
from .intel_layer import FOLD_LANES, RMS_LANES, SOURCE as NORM_SOURCE, _launch, _local_arg, _row, _signs
from .intel_pack import SOURCE as GEMV_SOURCE, _set_args as _set_gemv_args, dpm_view
from .intel_ternary import _launch_1d, _launch_2d

DEFAULT_STEPS = 4
MAX_STEPS = 32


def _silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float32)).astype(np.float32))


def _sigm(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)).astype(np.float32))


def _rms_rows(x, w, n):
    """One `rms_rows` row: x[:n] normalised by its own RMS and scaled by w.

    Matches kcpu.c's rms_rows(): the mean square is (double sum)/n cast to float32, then
    1/sqrt(mean + 1e-6f) in float32, and the row is scaled elementwise as (x * sc) * w."""
    x = np.ascontiguousarray(x[:n], dtype=np.float32)
    acc = np.sum(x.astype(np.float64) * x.astype(np.float64))
    ms = np.float32(acc / np.float64(n))
    sc = np.float32(1.0) / np.float32(np.sqrt(ms + np.float32(1e-6)))
    return (((x * sc) * w[:n].astype(np.float32))).astype(np.float32)


class _Device(object):
    """The OpenCL plumbing shared by the kernels: two programs, counted enqueues."""

    def __init__(self, loader, opener, device, device_name, report):
        self.report = report
        self.enqueues = 0
        lib, error = _open(loader, opener)
        if lib is None:
            raise ValueError(error or "OpenCL loader unavailable")
        self.lib = lib
        try:
            _bind(lib)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("OpenCL loader is missing an API: %s" % exc) from exc
        devices, error = _enumerate(lib, report)
        if devices is None:
            raise ValueError(error or "OpenCL device enumeration failed")
        picked, info, error = pick_device(lib, devices, device, device_name)
        if error or picked is None or info is None:
            raise ValueError((error or "the selected device could not be read"))
        platform, device_handle, platform_entry = devices[picked]
        report["platform"] = platform_entry["name"]
        report["device"] = info
        report["fp64"] = "cl_khr_fp64" in (info.get("extensions") or "")
        if "arc" not in (info.get("name") or "").lower():
            report["warnings"].append(
                "the selected device is %r, not an Arc: pass --device-name Arc for a run whose "
                "numbers may be quoted as Arc" % (info.get("name"),))
        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        code = ctypes.c_int(0)
        self.context = lib.clCreateContext(properties, 1, handles, None, None, ctypes.byref(code))
        report["returncodes"]["clCreateContext"] = _hex(code.value)
        if not self.context or code.value != CL_SUCCESS:
            raise ValueError("clCreateContext returned %s" % _hex(code.value))
        code = ctypes.c_int(0)
        self.queue = lib.clCreateCommandQueue(self.context, device_handle, 0, ctypes.byref(code))
        report["returncodes"]["clCreateCommandQueue"] = _hex(code.value)
        if not self.queue or code.value != CL_SUCCESS:
            raise ValueError("clCreateCommandQueue returned %s" % _hex(code.value))
        self.device_handle = device_handle
        self.handles = handles
        self.rms_program = self._build(NORM_SOURCE, "rms_sum")
        self.rms = self._kernel(self.rms_program, b"rms_sum")
        self.fold = self._kernel(self.rms_program, b"fold_sign")
        gemv_program = self._build(GEMV_SOURCE, "pack_gemv4")
        self.gemv = self._kernel(gemv_program, b"pack_gemv4")
        self.buffers = []

    def _build(self, source, kernel_name):
        code = ctypes.c_int(0)
        src = source.encode("utf-8")
        lengths = ctypes.c_size_t(len(src))
        strings = (ctypes.c_char_p * 1)(src)
        program = self.lib.clCreateProgramWithSource(self.context, 1, strings, ctypes.byref(lengths),
                                                     ctypes.byref(code))
        self.report["returncodes"]["clCreateProgramWithSource"] = _hex(code.value)
        if not program or code.value != CL_SUCCESS:
            raise ValueError("clCreateProgramWithSource returned %s" % _hex(code.value))
        build = int(self.lib.clBuildProgram(program, 1, self.handles, None, None, None))
        self.report["returncodes"]["clBuildProgram"] = _hex(build)
        if build != CL_SUCCESS:
            raise ValueError("the kernel did not build: %s"
                             % _build_log(self.lib, program, self.device_handle))
        return program

    def _kernel(self, program, name):
        code = ctypes.c_int(0)
        kernel = self.lib.clCreateKernel(program, name, ctypes.byref(code))
        self.report["returncodes"]["clCreateKernel/" + name.decode()] = _hex(code.value)
        if not kernel:
            raise ValueError("clCreateKernel failed")
        return kernel

    def buffer(self, data, flags, key):
        wait = self.enqueues
        from .intel_ternary import _buffer
        buf = _buffer(self.lib, self.context, flags, data, self.report, key)
        self.buffers.append(buf)
        return buf

    def write(self, buf, data, key):
        """Enqueue a blocking host write; counted like every other enqueue."""
        array = np.ascontiguousarray(data, dtype=np.float32)
        code = int(self.lib.clEnqueueWriteBuffer(
            ctypes.c_void_p(int(self.queue)), ctypes.c_void_p(int(buf)), ctypes.c_uint(1),
            ctypes.c_size_t(0),
            ctypes.c_size_t(int(array.nbytes)),
            ctypes.cast(array.ctypes.data, ctypes.c_void_p), ctypes.c_uint(0), None, None))
        self.enqueues += 1
        self.report["returncodes"][key] = _hex(code)
        if code != CL_SUCCESS:
            raise ValueError("%s returned %s" % (key, _hex(code)))

    def read(self, buf, elements, key):
        out = _read_back(self.lib, self.queue, buf, elements, self.report, key)
        self.enqueues += 1
        return np.frombuffer(out, dtype=np.float32).copy()


class ArcStep(object):
    """One decode token on the Arc: device folds + GEMVs, host conv/delta/attention state."""

    def __init__(self, pack_dir, loader=None, opener=None, device=0, device_name=None, ctx=2048):
        self.report = {"returncodes": {}, "warnings": [], "platforms": [], "pack": {"dir": pack_dir},
                       "loader": {"available": True}, "ok": False}
        self.pack_dir = pack_dir
        manifest = json.load(open(os.path.join(pack_dir, "kindi_pack.json"), encoding="utf-8"))
        self.manifest = manifest
        d = manifest["dimensions"]
        self.D, self.NFF = int(d["width"]), int(d["ffn"])
        self.NH, self.NKV, self.HD = int(d["heads"]), int(d["kv_heads"]), int(d["head_dim"])
        self.NK, self.HK = int(d["groups"]), int(d["state_size"])
        self.NV = int(d["gdn_inner"]) // self.HK
        self.HB = int(manifest.get("hadamard_block") or 512)
        self.NL = int(d["layers"])
        self.rh = int(d["rope_dim"]) // 2
        self.rope_dim = int(d["rope_dim"])
        self.rope_base = float(d["rope_base"])
        self.CC = 2 * self.NK * self.HK + self.NV * self.HK
        self.vocab = int(manifest["vocab"])
        self.ctx = ctx
        self.dev = _Device(loader or DEFAULT_LOADER, opener, device, device_name, self.report)
        self._load_pack()
        self._allocate()
        self.reset()
        inv = np.float64(self.rope_base) ** (-2.0 * np.arange(self.rh) / self.rope_dim)
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)

    # -- pack and device setup --
    def _load_pack(self):
        pack_dir = self.pack_dir
        bc = json.load(open(os.path.join(pack_dir, "bcodes_index.json"), encoding="utf-8"))
        idx = json.load(open(os.path.join(pack_dir, "index.json"), encoding="utf-8"))
        self.bc = bc["tensors"]
        self.f32_index = idx["tensors"]
        self.bcodes = np.fromfile(os.path.join(pack_dir, "bcodes.bin"), dtype=np.uint8)
        self.dpm16 = np.fromfile(os.path.join(pack_dir, "dpm.bin"), dtype=np.uint8).view("<f2")
        self.f32bin = np.fromfile(os.path.join(pack_dir, "f32.bin"), dtype=np.float32)
        self.signs = {int(k): np.fromfile(os.path.join(pack_dir, "signs_%d.f32" % int(k)),
                                          dtype=np.float32).copy()
                      for k in _signs(pack_dir, idx)}
        self.attn_layers = sorted({int(n.split(".")[1]) for n in self.bc
                                   if n.endswith(".attn_q.weight")})

    def _f32(self, name):
        entry = self.f32_index[name]
        off, n = int(entry["f32_off"]) // 4, int(entry["nfloats"])
        return self.f32bin[off:off + n].copy()

    def _allocate(self):
        dev = self.dev
        self.codes_buf = dev.buffer(self.bcodes, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    "clCreateBufferCodes")
        self.scales = {}
        for name, entry in self.bc.items():
            phases = int(entry["ne0"]) // 128
            tensor = {"rows": int(entry["rows"]), "phases": phases,
                      "dpm_off": int(entry["dpm_off"])}
            self.scales[name] = dev.buffer(
                np.ascontiguousarray(dpm_view(self.dpm16, tensor).astype(np.float32).reshape(-1)),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, "clCreateBufferScales")
            entry["_phases"] = phases
        W = max(self.D, self.NFF, self.NV * self.HK, self.NH * self.HD)
        self.W = W
        z = np.zeros(W, dtype=np.float32)
        self.h = dev.buffer(z, CL_MEM_WRITE_ONLY, "clCreateBufferH")
        self.res = dev.buffer(z, CL_MEM_WRITE_ONLY, "clCreateBufferRes")
        self.xn = dev.buffer(z, CL_MEM_WRITE_ONLY, "clCreateBufferXn")
        self.xp = dev.buffer(z, CL_MEM_WRITE_ONLY, "clCreateBufferXp")
        self.ybuf = dev.buffer(np.zeros(self.vocab, dtype=np.float32), CL_MEM_WRITE_ONLY,
                               "clCreateBufferY")
        self.sig = dev.buffer(np.ones(W, dtype=np.float32), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              "clCreateBufferSig")
        self.ones = dev.buffer(np.ones(W, dtype=np.float32), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               "clCreateBufferOnes")
        self.sign_buf = {w: dev.buffer(np.ascontiguousarray(s), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       "clCreateBufferSign%d" % w)
                         for w, s in self.signs.items()}
        self.norm_buf = {}
        for name in ("output_norm.weight",):
            self.norm_buf[name] = dev.buffer(
                np.ascontiguousarray(self._f32(name)), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                "clCreateBufferNorm")
        for il in range(self.NL):
            for suf in ("attn_norm.weight", "post_attention_norm.weight"):
                nm = "blk.%d.%s" % (il, suf)
                self.norm_buf[nm] = dev.buffer(
                    np.ascontiguousarray(self._f32(nm)), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    "clCreateBufferNorm")

    def reset(self):
        self.M = {il: np.zeros((self.NV, self.HK, self.HK), dtype=np.float32) for il in range(self.NL)}
        self.conv = {il: np.zeros(3 * self.CC, dtype=np.float32) for il in range(self.NL)}
        self.Kc = {il: np.zeros((self.ctx, self.NKV, self.HD), dtype=np.float16) for il in self.attn_layers}
        self.Vc = {il: np.zeros((self.ctx, self.NKV, self.HD), dtype=np.float16) for il in self.attn_layers}

    # -- device calls --
    def _norm_fold(self, h, w_name):
        """rms_sum + fold_sign for width D; leaves the folded input in xp.

        The residual is folded in on the host (h += y after every block), so this is always the
        CPU's add=0 path with h already carrying the sum -- the same xf values either way."""
        dev = self.dev
        dev.write(self.h, h, "clEnqueueWriteH")
        lib, rms, fold = dev.lib, dev.rms, dev.fold
        D, HB = self.D, self.HB
        _set_arg(lib, rms, 0, ctypes.c_void_p(int(self.h)))
        _set_arg(lib, rms, 1, ctypes.c_void_p(int(self.h)))
        _set_arg(lib, rms, 2, ctypes.c_void_p(int(self.sig)))
        _set_arg(lib, rms, 3, ctypes.c_uint32(D))
        _set_arg(lib, rms, 4, ctypes.c_uint32(0))
        _local_arg(lib, rms, 5, RMS_LANES * 4)
        _local_arg(lib, rms, 6, RMS_LANES * 4)
        _launch(lib, dev.queue, rms, RMS_LANES, RMS_LANES, self.report, "clEnqueueRmsSum")
        dev.enqueues += 1
        _set_arg(lib, fold, 0, ctypes.c_void_p(int(self.h)))
        _set_arg(lib, fold, 1, ctypes.c_void_p(int(self.h)))
        _set_arg(lib, fold, 2, ctypes.c_void_p(int(self.norm_buf[w_name])))
        _set_arg(lib, fold, 3, ctypes.c_void_p(int(self.sign_buf[D])))
        _set_arg(lib, fold, 4, ctypes.c_void_p(int(self.sig)))
        _set_arg(lib, fold, 5, ctypes.c_void_p(int(self.xn)))
        _set_arg(lib, fold, 6, ctypes.c_void_p(int(self.xp)))
        _set_arg(lib, fold, 7, ctypes.c_uint32(D))
        _set_arg(lib, fold, 8, ctypes.c_uint32(HB))
        _set_arg(lib, fold, 9, ctypes.c_uint32(0))
        _local_arg(lib, fold, 10, HB * 4)
        _launch(lib, dev.queue, fold, (D // HB) * FOLD_LANES, FOLD_LANES, self.report,
                "clEnqueueFoldSign")
        dev.enqueues += 1

    def _fold_only(self, src, width, sign_width):
        """Fold a host vector with unit weight/scale (the FFN/GDN/attn m -> xp step)."""
        dev = self.dev
        dev.write(self.h, src, "clEnqueueWriteSrc")
        lib, fold = dev.lib, dev.fold
        _set_arg(lib, fold, 0, ctypes.c_void_p(int(self.h)))
        _set_arg(lib, fold, 1, ctypes.c_void_p(int(self.res)))
        _set_arg(lib, fold, 2, ctypes.c_void_p(int(self.ones)))
        _set_arg(lib, fold, 3, ctypes.c_void_p(int(self.sign_buf[sign_width])))
        _set_arg(lib, fold, 4, ctypes.c_void_p(int(self.ones)))
        _set_arg(lib, fold, 5, ctypes.c_void_p(int(self.xn)))
        _set_arg(lib, fold, 6, ctypes.c_void_p(int(self.xp)))
        _set_arg(lib, fold, 7, ctypes.c_uint32(width))
        _set_arg(lib, fold, 8, ctypes.c_uint32(self.HB))
        _set_arg(lib, fold, 9, ctypes.c_uint32(0))
        _local_arg(lib, fold, 10, self.HB * 4)
        _launch(lib, dev.queue, fold, (width // self.HB) * FOLD_LANES, FOLD_LANES,
                self.report, "clEnqueueFoldOnly")
        dev.enqueues += 1

    def _gemv(self, name):
        """One ternary GEMV with the folded input in xp; returns the output row."""
        dev = self.dev
        entry = self.bc[name]
        rows, cols = int(entry["rows"]), int(entry["ne0"])
        lib = dev.lib
        _set_gemv_args(lib, dev.gemv, int(self.codes_buf), int(self.scales[name]),
                       int(self.xp), int(self.ybuf), rows, cols, 1,
                       int(entry["bc_off"]), 0)
        _launch_2d(lib, dev.queue, dev.gemv, (rows + 3) // 4, 1, self.report, "clEnqueueGemv")
        dev.enqueues += 1
        return dev.read(self.ybuf, rows, "clEnqueueReadY")[:rows]

    def _xn(self):
        return self.dev.read(self.xn, self.D, "clEnqueueReadXn")[: self.D]

    # -- host layer math, mirroring kcpu.c --
    def _gdn(self, il, h, pos):
        D, NV, NK, HK, CC = self.D, self.NV, self.NK, self.HK, self.CC
        p = "blk.%d." % il
        self._norm_fold(h, p + "attn_norm.weight")
        qkv = self._gemv(p + "attn_qkv.weight")
        zv = self._gemv(p + "attn_gate.weight")
        xn = self._xn()
        beta_w = self._f32(p + "ssm_beta.weight").reshape(NV, D)
        alpha_w = self._f32(p + "ssm_alpha.weight").reshape(NV, D)
        gb = _sigm(beta_w @ xn)
        a = alpha_w @ xn + self._f32(p + "ssm_dt.bias")
        gg = np.where(a > 20.0, a, np.log1p(np.exp(np.minimum(a, 20.0)))).astype(np.float32) \
            * self._f32(p + "ssm_a")
        conv_w = self._f32(p + "ssm_conv1d.weight").reshape(CC, 4)
        cv = self.conv[il]
        nwent = qkv[:CC]
        o = cv[:CC] * conv_w[:, 0] + cv[CC:2 * CC] * conv_w[:, 1] \
            + cv[2 * CC:] * conv_w[:, 2] + nwent * conv_w[:, 3]
        cv[:CC] = cv[CC:2 * CC].copy()
        cv[CC:2 * CC] = cv[2 * CC:].copy()
        cv[2 * CC:] = nwent
        cs = _silu(o.astype(np.float32))
        qk = np.empty((2 * NK, HK), dtype=np.float32)
        for r in range(2 * NK):
            xr = cs[r * HK:(r + 1) * HK].astype(np.float32)
            acc = np.sum(xr.astype(np.float64) * xr.astype(np.float64))
            sc = np.float32(1.0) / np.maximum(np.float32(np.sqrt(np.float32(acc))), np.float32(1e-6))
            qk[r] = (xr * sc).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(HK))
        gout = np.empty((NV, HK), dtype=np.float32)
        outn = np.empty((NV, HK), dtype=np.float32)
        M = self.M[il]
        for hh in range(NV):
            kh = hh % NK
            ks = qk[NK + kh]
            qs = qk[kh]
            v = cs[2 * NK * HK + hh * HK: 2 * NK * HK + (hh + 1) * HK]
            e = np.exp(gg[hh]).astype(np.float32)
            sk = np.sum(M[hh] * e * ks[:, None], axis=0).astype(np.float32)
            dd = ((v - sk) * gb[hh]).astype(np.float32)
            M[hh] = (M[hh] * e + ks[:, None] * dd[None, :]).astype(np.float32)
            o_h = np.sum(M[hh] * qs[:, None], axis=0).astype(np.float32) * scale
            gout[hh] = o_h
            outn[hh] = _rms_rows(o_h, self._f32(p + "ssm_norm.weight"), HK)
        rep = NV // NK
        if rep > 1:
            grp = np.arange(NV * HK) // HK
            hd = np.arange(NV * HK) % HK
            src_idx = hd + HK * (grp // rep + NK * (grp % rep))
        else:
            src_idx = np.arange(NV * HK)
        m = (_silu(zv[src_idx]) * outn.reshape(-1)[src_idx]).astype(np.float32)
        self._fold_only(m, NV * HK, NV * HK)
        y = self._gemv(p + "ssm_out.weight")
        return h, y

    def _attn(self, il, h, pos):
        D, NH, NKV, HD, rh = self.D, self.NH, self.NKV, self.HD, self.rh
        G = NH // NKV
        p = "blk.%d." % il
        self._norm_fold(h, p + "attn_norm.weight")
        qn = self._gemv(p + "attn_q.weight")
        kr = self._gemv(p + "attn_k.weight")
        vv = self._gemv(p + "attn_v.weight")
        cs_, sn_ = self.cos[pos], self.sin[pos]
        qr = np.empty((NH, HD), dtype=np.float32)
        kro = np.empty((NKV, HD), dtype=np.float32)
        for r in range(NH + NKV):
            if r < NH:
                tmp = _rms_rows(qn[r * 2 * HD:r * 2 * HD + HD], self._f32(p + "attn_q_norm.weight"), HD)
                dst = qr[r]
            else:
                tmp = _rms_rows(kr[(r - NH) * HD:(r - NH + 1) * HD], self._f32(p + "attn_k_norm.weight"), HD)
                dst = kro[r - NH]
            a_, b_ = tmp[:rh], tmp[rh:2 * rh]
            dst[:rh] = a_ * cs_ - b_ * sn_
            dst[rh:2 * rh] = a_ * sn_ + b_ * cs_
            dst[2 * rh:] = tmp[2 * rh:]
            if r >= NH:
                kv = r - NH
                self.Kc[il][pos, kv] = dst.astype(np.float16)
                self.Vc[il][pos, kv] = vv[kv * HD:(kv + 1) * HD].astype(np.float16)
        n = pos + 1
        K = self.Kc[il][:n].astype(np.float32)     # (n, NKV, HD)
        V = self.Vc[il][:n].astype(np.float32)
        att = np.empty((NH, HD), dtype=np.float32)
        for hh in range(NH):
            kv = hh // G
            q = qr[hh]
            sc = np.sum(K[:, kv, :] * q[None, :], axis=1).astype(np.float32) * np.float32(0.0625)
            mx = np.max(sc)
            ex = np.exp(sc - mx)
            pr = (ex / np.sum(ex)).astype(np.float32)
            acc = np.sum(V[:, kv, :] * pr[:, None], axis=0).astype(np.float32)
            att[hh] = acc * _sigm(qn[hh * 2 * HD + HD:hh * 2 * HD + 2 * HD])
        self._fold_only(att.reshape(-1), NH * HD, NH * HD)
        y = self._gemv(p + "attn_output.weight")
        return h, y

    def _ffn(self, il, h):
        p = "blk.%d." % il
        self._norm_fold(h, p + "post_attention_norm.weight")
        g = self._gemv(p + "ffn_gate.weight")
        u = self._gemv(p + "ffn_up.weight")
        m = (_silu(g) * u).astype(np.float32)
        self._fold_only(m, self.NFF, self.NFF)
        return self._gemv(p + "ffn_down.weight")

    def _embed(self, tok):
        """kc embed(): h = sD * FWHT_blocks(row), the sign applied after the transform."""
        row = _row(self.bcodes, self.dpm16, self.bc["token_embd.weight"], tok).astype(np.float32)
        o = row.reshape(-1, self.HB).copy()
        for i in range(o.shape[0]):
            b = o[i]
            h_ = 1
            while h_ < self.HB:
                v = b.reshape(-1, 2 * h_)
                a = v[:, :h_].copy()
                c = v[:, h_:].copy()
                v[:, :h_] = (a + c).astype(np.float32)
                v[:, h_:] = (a - c).astype(np.float32)
                h_ <<= 1
            b *= np.float32(1.0 / np.sqrt(self.HB))
            o[i] = b
        return (o.reshape(-1) * self.signs[self.D]).astype(np.float32)

    def step(self, tok, pos, head=True):
        """One decode token: returns (greedy token, logits) or just the residual after the stack."""
        h = self._embed(tok)
        for il in range(self.NL):
            p = "blk.%d." % il
            if il in self.attn_layers:
                _, y = self._attn(il, h, pos)
            else:
                _, y = self._gdn(il, h, pos)
            h = (h + y).astype(np.float32)
            y = self._ffn(il, h)
            h = (h + y).astype(np.float32)
        if not head:
            return h
        self._norm_fold(h, "output_norm.weight")
        logits = self._gemv("token_embd.weight")
        return int(np.argmax(logits)), logits

    def finish(self):
        _finish(self.dev.lib, self.dev.queue, self.report, "clFinish")


def run_step(pack_dir, prefix_ids, nsteps, loader=None, opener=None, device=0, device_name=None,
             ctx=2048):
    if prefix_ids is None:
        prefix_ids = (128, 2562, 2027, 31823)
    """Verify the Arc decode step against the CPU engine, teacher-forced, and time it."""
    report = {"returncodes": {}, "warnings": [], "ok": False,
              "pack": {"dir": pack_dir}, "loader": {"available": True}, "steps": [],
              "platforms": [], "prefix": list(prefix_ids), "nsteps": nsteps}
    runner = None
    try:
        runner = ArcStep(pack_dir, loader=loader, opener=opener, device=device,
                         device_name=device_name, ctx=ctx)
        report.update(runner.report)
        report["device"] = runner.report["device"]
        dev = runner.dev
        # Oracle: the CPU engine, fed the same prefix.
        from . import cpu
        eng = cpu.Engine(pack_dir, ctx=ctx, threads=2, log=None)
        refs = []
        for pos, tok in enumerate(prefix_ids[:-1]):
            eng.feed(tok, pos)
        toks = list(prefix_ids)
        pos = len(prefix_ids) - 1
        # Warm the Arc state on the prefix (layers only, like the CPU's feed).
        for p0, tok in enumerate(prefix_ids[:-1]):
            runner.step(tok, p0, head=False)
        dev.enqueues = 0
        for _ in range(nsteps):
            tok = toks[-1]
            ref_tok = eng.step(tok, pos)
            ref_logits = np.array(eng.logits, dtype=np.float32)
            t0 = time.perf_counter()
            got_tok, got_logits = runner.step(tok, pos)
            dt = (time.perf_counter() - t0) * 1e3
            toks.append(ref_tok)
            diff = np.abs(got_logits - ref_logits)
            rel = diff / (np.abs(ref_logits) + 1e-6)
            report["steps"].append({"pos": pos, "tok": tok, "ref_tok": ref_tok, "got_tok": got_tok,
                                     "match": bool(ref_tok == got_tok),
                                     "max_abs_diff": float(diff.max()),
                                     "max_rel_diff": float(rel.max()),
                                     "ms": round(dt, 3), "enqueues": dev.enqueues})
            dev.enqueues = 0
            pos += 1
        eng.close()
        report["ok"] = all(s["match"] for s in report["steps"])
        report["ms_per_step"] = round(sum(s["ms"] for s in report["steps"]) / len(report["steps"]), 3)
        report["enqueues_per_step"] = round(sum(s["enqueues"] for s in report["steps"]) / len(report["steps"]), 1)
    except Exception as exc:                       # noqa: BLE001 - report the failure with the trace
        report["error"] = repr(exc)
        import traceback
        report["traceback"] = traceback.format_exc()
    finally:
        if runner is not None:
            try:
                runner.finish()
            except Exception:
                pass
    return report


def render(report):
    lines = ["Arc decode step (K3 wiring): device folds + GEMVs, host conv/delta/attention; "
             "compared against the CPU engine, teacher-forced"]
    pack = report["pack"]
    lines.append("  pack: %s" % pack.get("dir"))
    if report.get("device"):
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB" % (
            device.get("name"), device.get("compute_units"), device.get("global_mem_mib")))
    if report.get("error"):
        lines.append("  error: %s" % report["error"])
        return "\n".join(lines)
    for s in report["steps"]:
        lines.append("  pos %d: tok %d -> ref %d / arc %d%s  max|dlogit| %.4g (%.3g rel)  %.2f ms, %d enqueues"
                     % (s["pos"], s["tok"], s["ref_tok"], s["got_tok"],
                        "" if s["match"] else "  MISMATCH", s["max_abs_diff"], s["max_rel_diff"],
                        s["ms"], s["enqueues"]))
    lines.append("  verdict: %s; %.2f ms/step, %.1f enqueues/step" % (
        "token-exact" if report["ok"] else "DIVERGED", report["ms_per_step"],
        report["enqueues_per_step"]))
    return "\n".join(lines)


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
