"""The Bonsai 2.5 MTP head on the CPU engine: the ``-mtp`` GGUF's 65th block drafts tokens.

Prism ML ships the 27B draft head in the same file as the model: one Qwen3.5 decoder block
(``blk.64``: gated attention with q/k norms and a 17,408-wide FFN, stored Q4_0) plus the four
``nextn`` tensors that turn it into a multi-token-prediction head (``enorm``, ``hnorm``,
``eh_proj`` and ``shared_head_norm``).  The head predicts with the main model's own output head
(``output.weight``), so its argmax is over the main model's vocabulary, not a second one.

Row convention, the same as ``kindi/mtp.py``: the head's row for position ``q`` reads the main
model's residual at ``q`` and the token at ``q + 1``, and predicts the token at ``q + 2``.  A
draft of k tokens therefore needs one row at the verified position and then k - 1 rows that feed
the head's own output vector and the token it has just predicted - the way the head was trained.

What this does and does not do on the CPU.  ``kindi/cpu.py`` decodes one token at a time: ``step``
streams the whole pack for a single row, so a draft can only be confirmed by the same forward pass
plain decoding would have run.  The head here drafts and scores correctly, and ``score_drafts``
compares its drafts with the model's own greedy tokens, but it cannot make CPU decoding faster
until verification itself is batched - that is the road-map row "batched verify for MTP on the
CPU".  Use it for acceptance measurements and for the head's plumbing; do not expect tokens per
second from it yet.
"""
import os
import time

import numpy as np

from . import cpu as _cpu
from .convert import trits_from_raw
from .gguf import GGUF

F32 = np.float32
Q4_0 = 2
PTQ1_0 = 143

MATS = (("fc", "nextn.eh_proj.weight"), ("q", "attn_q.weight"), ("k", "attn_k.weight"),
        ("v", "attn_v.weight"), ("o", "attn_output.weight"), ("gate", "ffn_gate.weight"),
        ("up", "ffn_up.weight"), ("down", "ffn_down.weight"))
NORMS = (("e_norm", "nextn.enorm.weight"), ("h_norm", "nextn.hnorm.weight"), ("in_norm", "attn_norm.weight"),
         ("q_norm", "attn_q_norm.weight"), ("k_norm", "attn_k_norm.weight"),
         ("post_norm", "post_attention_norm.weight"), ("out_norm", "nextn.shared_head_norm.weight"))


def fwht(x, hb):
    """The engine's Hadamard block transform (kcpu.c:fwht_block): no sign flip, 1/sqrt(hb) at the end."""
    y = np.array(x, dtype=F32, copy=True)
    for b in range(0, y.size, hb):
        z = y[b:b + hb]
        h = 1
        while h < hb:
            z = z.reshape(-1, 2, h)
            z = np.stack([z[:, 0] + z[:, 1], z[:, 0] - z[:, 1]], axis=1).reshape(-1)
            h *= 2
        y[b:b + hb] = (z / np.sqrt(hb)).astype(F32)
    return y


def fold(x, signs, hb):
    """fold(v) = FWHT_blocks(signs * v) = H (S v), the transform kcpu.c applies to folded inputs.

    A pack matrix stored folded is W (S H) for that same H and S, so dot(W, x) is
    dot(W (S H), H (S x)): folding the input with this call is what makes the pack's matrix and
    the head's plain float32 output agree.
    """
    return fwht(signs * np.asarray(x, dtype=F32), hb)


def q4_dequant(raw, rows, cols):
    """(rows * cols/32, 18) Q4_0 blocks -> (rows, cols) float32 (the reference ``kc_q4gemv`` matches)."""
    nb = cols // 32
    raw = np.asarray(raw, dtype=np.uint8).reshape(rows * nb, 18)
    d = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.float16).astype(F32).reshape(rows, nb)
    qs = raw[:, 2:].reshape(rows, nb, 16)
    lo = (qs & 15).astype(F32) - 8.0
    hi = (qs >> 4).astype(F32) - 8.0
    v = np.concatenate([lo, hi], axis=2) * d[:, :, None]
    return v.reshape(rows, cols)


def trits(raw):
    """PTQ1_0 (nb, 28) blocks -> (nb, 128) trits in {0, 1, 2}.

    One implementation, the converter's: a block is three byte groups (16 + 8 + 2 bytes, 5 trits per
    byte), and an earlier local copy here decoded only the first 80 of the 128 columns - the rest were
    left uninitialised.  That is what kept the head's acceptance near random: every embedding row it
    built was 37% garbage.
    """
    return trits_from_raw(raw)


class Head(object):
    """The MTP draft head of a Ternary-Bonsai ``-mtp`` GGUF, on a loaded CPU engine.

    ``engine`` supplies the main model's output head (the head's argmax) and the pack's Hadamard
    signs (the output head is stored folded).  ``window`` is how many recent positions the head's
    own K/V ring keeps; a draft round writes at most ``k`` rows, so 64 is generous.
    """

    def __init__(self, gguf_path, engine, k=4, window=64, log=print):
        self.eng, self.path, self.k, self.W = engine, gguf_path, int(k), int(window)
        g = GGUF(gguf_path)
        self.data0 = int(g.data_start)
        by = {t[0]: t for t in g.tensors}
        block = None
        for name in by:
            if name.endswith(".nextn.eh_proj.weight"):
                block = name.split(".")[1]
                break
        if block is None:
            raise ValueError("kindi: %s carries no MTP head (no blk.N.nextn.eh_proj.weight)" % gguf_path)
        pre = "blk.%s." % block
        kv = g.kv
        self.dims = d = {"D": int(kv["qwen35.embedding_length"]), "NFF": int(kv["qwen35.feed_forward_length"]),
                         "NH": int(kv["qwen35.attention.head_count"]),
                         "NHKV": int(kv["qwen35.attention.head_count_kv"]),
                         "HD": int(kv["qwen35.attention.key_length"]),
                         "RD": int(kv["qwen35.rope.dimension_count"]),
                         "base": float(kv["qwen35.rope.freq_base"]), "block": int(block)}
        with open(gguf_path, "rb") as f:
            self.w, self.n, nbytes = {}, {}, 0
            for key, suffix in MATS:
                name = pre + suffix
                ne, ty, off = by[name][1], by[name][2], by[name][3]
                cols, rows = int(ne[0]), int(np.prod(ne[1:]))
                if ty != Q4_0:
                    raise ValueError("kindi: %s is GGUF type %d, expected Q4_0" % (name, ty))
                if cols % 32:
                    raise ValueError("kindi: %s has %d columns, not a multiple of 32" % (name, cols))
                f.seek(self.data0 + off)
                w = np.frombuffer(f.read(rows * (cols // 32) * 18), dtype=np.uint8).copy()
                if w.size != rows * (cols // 32) * 18:
                    raise ValueError("kindi: %s is truncated" % name)
                self.w[key] = (w, rows, cols)
                nbytes += w.size
            for key, suffix in NORMS:
                name = pre + suffix
                ne, ty, off = by[name][1], by[name][2], by[name][3]
                cols, rows = int(ne[0]), int(np.prod(ne[1:]))
                if ty != 0 or rows != 1:
                    raise ValueError("kindi: %s is GGUF type %d, expected F32" % (name, ty))
                f.seek(self.data0 + off)
                self.n[key] = np.frombuffer(f.read(4 * cols), dtype=F32).copy()
                if self.n[key].size != cols:
                    raise ValueError("kindi: %s is truncated" % name)
                nbytes += 4 * cols
        self.bytes = nbytes
        ne, ty, off = by["token_embd.weight"][1], by["token_embd.weight"][2], by["token_embd.weight"][3]
        if ty != PTQ1_0:
            raise ValueError("kindi: token_embd.weight is GGUF type %d, expected PTQ1_0" % ty)
        self._emb = (off, int(ne[0]), int(np.prod(ne[1:])))
        # the output head lives in the pack, folded: the input needs the same signs and Hadamard block
        self.hb = int(getattr(engine, "HB", 1024))
        path = os.path.join(getattr(engine, "pack", ""), "signs_%d.f32" % d["D"])
        self.signs = np.fromfile(path, dtype=F32) if os.path.exists(path) else None
        if self.signs is None or self.signs.size != d["D"]:
            raise ValueError("kindi: the pack has no signs_%d.f32, the folded output head cannot be read" % d["D"])
        ne, ty, off = by["output_norm.weight"][1], by["output_norm.weight"][2], by["output_norm.weight"][3]
        with open(gguf_path, "rb") as f:
            f.seek(self.data0 + off)
            self.out_w = np.frombuffer(f.read(4 * int(ne[0])), dtype=F32).copy()
        self._emb_cache = {}                     # token id -> raw PTQ1_0 row (a draft re-reads the same rows a lot)
        self.order = "eh"                        # eh_proj input order: [emb, hidden] (probe-switchable)
        self.hidden = "raw"                      # "raw": residual as get_hidden() returns it; "normed": rms * output_norm (what the GPU head's XN is)
        self.emb_mode = "raw"                    # "raw": the GGUF embedding row; "folded": the pack's fold of it (what the engine embeds)
        self.rope_off = 0                        # RoPE position offset vs the row position (probe-switchable)
        self.seed_enabled = False                # seed the window with verified rows (llama.cpp does)
        self.seed_last = False                   # also seed the prompt's last row (its next token is the first generated one)
        self.last_u = None                       # the last row's pre-attention vector (u3 - o), for a probe dump
        self.K = np.zeros((self.W, d["NHKV"], d["HD"]), dtype=F32)
        self.V = np.zeros((self.W, d["NHKV"], d["HD"]), dtype=F32)
        self.kp = np.full(self.W, -(1 << 30), dtype=np.int64)
        self.stats = {"rows": 0, "drafts": 0}
        self.reset()
        (log or (lambda *a: None))("MTP head ready: %s blk.%s, %.0f MB Q4_0, %d heads x %d, window %d" % (
            os.path.basename(gguf_path), block, nbytes / 1e6, d["NH"], d["HD"], self.W))

    # ------------------------------------------------------------------ tensors
    def _gemv(self, key, x):
        w, rows, cols = self.w[key]
        return _cpu.q4_gemv(w, x, rows, cols)

    def _embed(self, tok):
        off, cols, rows = self._emb
        if not 0 <= tok < rows:
            raise ValueError("kindi: token %d is outside the embedding (%d rows)" % (tok, rows))
        row = self._emb_cache.get(tok)
        if row is None:
            if len(self._emb_cache) > 8192:
                self._emb_cache.clear()
            row = self._emb_cache[tok] = self._embed_read(tok, off, cols)
        return fold(row, self.signs, self.hb) if self.emb_mode == "folded" else row

    def _embed_read(self, tok, off, cols):
        bpr, bsize = cols // 128, 28
        with open(self.path, "rb") as f:
            f.seek(self.data0 + off + tok * bpr * bsize)
            buf = np.frombuffer(f.read(bpr * bsize), dtype=np.uint8).reshape(bpr, bsize)
        tr = trits(buf).astype(F32) - 1.0
        d = np.frombuffer(buf[:, 26:28].tobytes(), dtype=np.float16).astype(F32)
        return (tr * d[:, None]).reshape(cols)

    @staticmethod
    def _rms(x, w):
        s = 1.0 / np.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-6)
        return (x * F32(s) * w).astype(F32)

    def _rope(self, x, pos):
        rd, base = self.dims["RD"], self.dims["base"]
        j = np.arange(rd // 2, dtype=np.float64)
        ang = float(pos) * base ** (-2.0 * j / rd)
        c, s = np.cos(ang).astype(F32), np.sin(ang).astype(F32)
        a, b = x[:, :rd // 2].copy(), x[:, rd // 2:rd].copy()
        x[:, :rd // 2] = a * c - b * s
        x[:, rd // 2:rd] = a * s + b * c

    def _logits(self, out):
        rows = int(self.eng.ylog_n)
        best = int(np.argmax(self.eng.gemv_rows(self.eng.head_name, (0, rows), fold(out, self.signs, self.hb))))
        return best + int(getattr(self.eng, "head_off", 0) or 0)

    # ------------------------------------------------------------------ one row
    def row(self, emb, hin, pos):
        """The head's row at ``pos``: returns (the token it predicts, its output vector)."""
        d, n = self.dims, self.n
        NH, NHKV, HD = d["NH"], d["NHKV"], d["HD"]
        parts = {"eh": (self._rms(emb, n["e_norm"]), self._rms(hin, n["h_norm"])),
                 "he": (self._rms(hin, n["h_norm"]), self._rms(emb, n["e_norm"]))}
        x = np.concatenate(parts[self.order])
        u = self._gemv("fc", x)
        self.last_u = u
        a = self._rms(u, n["in_norm"])
        qg = self._gemv("q", a).reshape(NH, 2 * HD)
        kk = self._gemv("k", a).reshape(NHKV, HD)
        vv = self._gemv("v", a).reshape(NHKV, HD)
        for h in range(NH):
            qg[h, :HD] = self._rms(qg[h, :HD], n["q_norm"])
        for h in range(NHKV):
            kk[h] = self._rms(kk[h], n["k_norm"])
        self._rope(qg[:, :HD], pos + self.rope_off)
        self._rope(kk, pos + self.rope_off)
        slot = pos % self.W
        self.K[slot], self.V[slot], self.kp[slot] = kk, vv, pos
        ok = (self.kp <= pos) & (self.kp > pos - self.W)
        kvh = np.minimum(np.arange(NH) * NHKV // NH, NHKV - 1)
        q = qg[:, :HD]
        att = np.empty((NH, HD), dtype=F32)
        for h in range(NH):
            s = self.K[:, kvh[h]] @ q[h] * F32(1.0 / np.sqrt(HD))
            s = np.where(ok, s, F32(-1e30))
            e = np.exp(s - s.max())
            att[h] = (e / e.sum()) @ self.V[:, kvh[h]]
        gate = 1.0 / (1.0 + np.exp(-qg[:, HD:]))
        o = self._gemv("o", (att * gate).reshape(NH * HD))
        u2 = u + o
        p = self._rms(u2, n["post_norm"])
        f = self._silu(self._gemv("gate", p)) * self._gemv("up", p)
        u3 = u2 + self._gemv("down", f)
        out = self._rms(u3, n["out_norm"])
        self.stats["rows"] += 1
        return self._logits(out), out

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------ drafting
    def _hin(self, hin):
        if self.hidden == "normed":
            # the GPU head is fed XN = rms(residual) * output_norm, and its chained rows are already
            # rms(u3) * shared_head_norm; the raw residual is the odd one out. Only rows from the
            # engine need this; chained rows already arrive normed.
            return self._rms(np.asarray(hin, dtype=F32), self.out_w)
        return hin

    def seed(self, rows):
        """Write verified rows (pos, residual, next token) into the head's K/V window, llama.cpp-style.

        The reference drafts against a cache that holds every verified position's MTP row, not just
        the head's own draft rows. score_drafts calls this with the rows it has just verified; each
        row overwrites the draft row at the same position.
        """
        if not self.seed_enabled:
            return
        for pos, hin, tok in rows:
            self.row(self._embed(int(tok)), self._hin(hin), int(pos))

    def grow_window(self, n):
        """Grow the K/V ring to at least ``n`` positions, so a long prompt fits (it never shrinks)."""
        n = int(n)
        if n <= self.W:
            return False
        d = self.dims
        self.W = n
        self.K = np.zeros((n, d["NHKV"], d["HD"]), dtype=F32)
        self.V = np.zeros((n, d["NHKV"], d["HD"]), dtype=F32)
        self.kp = np.full(n, -(1 << 30), dtype=np.int64)
        self.reset()
        return True

    def reset(self):
        self.K[:] = 0
        self.V[:] = 0
        self.kp.fill(-(1 << 30))

    def draft(self, prefix, hin, k=None):
        """Up to ``k`` tokens after ``prefix[-1]``.

        ``hin`` is the main model's residual at the position of ``prefix[-1]`` (``len(prefix) - 1``):
        the head's row for that position reads the residual and the token after it, and predicts the
        token after that, so the row index is ``len(prefix) - 2``.
        """
        k = self.k if k is None else int(k)
        if k < 1 or len(prefix) < 2:
            return []
        q, tok = len(prefix) - 2, int(prefix[-1])
        pred, out = self.row(self._embed(tok), self._hin(hin), q)
        drafts = [pred]
        for i in range(1, k):
            pred, out = self.row(self._embed(drafts[-1]), out, q + i)
            drafts.append(pred)
        self.stats["drafts"] += 1
        return drafts


class PackDrafter(object):
    """An external drafter on the CPU: a second engine on a smaller converted pack (the 2B or 4B).

    It ignores the main model's residual and replays the known prefix through its own engine, then
    steps greedily for ``k`` tokens.  A replay per round is O(n) forwards on the small pack, which is
    the honest cost of drafting on this engine until verification is batched; it is the same
    acceptance experiment as the integrated head, with a drafter that is already converted here
    (``~/.kindi/packs/bonsai-2b``) instead of the ``-mtp`` file's 65th block.
    """

    def __init__(self, pack, ctx=2048, threads=None, k=4, log=print, incremental=False):
        self.eng = _cpu.Engine(pack, ctx=ctx, threads=threads, log=log or (lambda *a: None))
        self.path, self.k = pack, int(k)
        self.incremental = bool(incremental)
        self.stats = {"drafts": 0}
        self.reset()

    def reset(self):
        self.eng.reset()
        self.stats["drafts"] = 0
        self.consumed = []      # the tokens whose rows this engine has consumed (incremental mode)
        self.snap = None        # its GDN state at len(consumed)
        self._snap_buf = None   # the reusable buffer snap is copied into (no per-round allocation)

    def draft(self, prefix, hin, k=None):
        """Up to ``k`` tokens after ``prefix[-1]``, from the drafter's own greedy decode of the prefix.

        ``incremental=True`` keeps this engine's state across rounds: a round feeds only the tokens
        the prefix has gained since the last one, from a snapshot of the state at ``prefix[:-1]``, so
        drafting costs ``k`` rows plus the tokens the verify pass accepted instead of a replay of the
        whole prefix (the replay is what made the measurement in docs/MTP.md cost more than it saved).
        """
        k = self.k if k is None else int(k)
        if k < 1 or len(prefix) < 2:
            return []
        if not self.incremental:
            self.eng.reset()
            for i, t in enumerate(prefix[:-1]):
                self.eng.feed(int(t), i)
        else:
            want = [int(t) for t in prefix[:-1]]
            if self.snap is None or want[:len(self.consumed)] != self.consumed:
                self.eng.reset()
                self.consumed = []
                for i, t in enumerate(want):
                    self.eng.feed(t, i)
                    self.consumed.append(t)
            else:
                self.eng.load_state(self.snap)          # back to the state at prefix[:-1]
                for t in want[len(self.consumed):]:
                    self.eng.feed(t, len(self.consumed))
                    self.consumed.append(t)
            if self._snap_buf is None or self._snap_buf.size != self.eng.state_len():
                self._snap_buf = np.empty(self.eng.state_len(), dtype=np.float32)
            self.snap = self.eng.save_state(into=self._snap_buf)
        tok, pos, drafts = int(prefix[-1]), len(prefix) - 1, []
        for _ in range(k):
            tok = self.eng.step(tok, pos)
            drafts.append(tok)
            pos += 1
        self.stats["drafts"] += 1
        self.last = list(drafts)
        return drafts


def speculative_decode(eng, ids, n, drafter, k=4):
    """Greedy decode with a batched verify pass: ``drafter`` proposes, one T-row pass confirms.

    The loop keeps one *pending* token (the model's own argmax, not yet consumed) and a round is
    ``[pending] + drafts[:k]`` at consecutive positions, so the pass that would have cost k + 1
    weight reads costs one (``kindi/kcpu.c``'s kc_step_batch).  The drafts that match are accepted,
    the model's next argmax becomes the new pending token, and ``kc_batch_rollback`` puts the GDN
    state back at the last accepted row.  Returns ``(tokens, stats)`` like
    ``eng.generate(ids, n)`` - ``n`` tokens after the prompt, and the same ones, because every token
    is the model's own greedy pick.

    ``stats``: ``rounds``, ``offered`` (k per round), ``matched``, ``rate``, ``rows`` (batch rows
    fed), ``kept`` (rows the state kept), ``depth[i]`` (rounds where draft i + 1 was accepted) and
    ``batch_ms`` / ``rollback_ms``.

    Measurement only: this loop is not on a serving path (no `kindi bench --spec` CPU cell, no
    server wiring), because the CP1 gate it was built for was missed at 0.51x on the 27B
    (docs/ROADMAP.md CP1). It stays so the reopen (CP3) has an exactness-correct loop to run.
    """
    drafter.reset()
    eng.reset()
    for i, t in enumerate(ids[:-1]):
        eng.feed(t, i)
    pending = eng.step(ids[-1], len(ids) - 1)      # the model's token at position len(ids)
    out = list(ids)                                # tokens the model has committed (positions 0..q-1)
    q = len(ids)                                   # the position `pending` occupies
    stats = {"rounds": 0, "offered": 0, "matched": 0, "depth": [0] * max(0, int(k)),
             "rows": 0, "kept": 0, "batch_ms": 0.0, "rollback_ms": 0.0}
    while len(out) - len(ids) < n:
        prefix = list(out)                         # tokens at positions 0..q-1
        h = eng.batch_hidden(0).copy() if hasattr(eng.lib, "kc_batch_hidden") and stats["rounds"] else None
        # the drafter must see `pending` too: row 0's argmax is the token after it, so draft 0 is
        # compared with a row that already consumed `pending`
        drafts = drafter.draft(prefix + [pending], h, k)
        rows = [pending] + drafts
        t0 = time.time()
        preds = eng.step_batch(rows, q)
        stats["batch_ms"] += (time.time() - t0) * 1000.0
        matched = 0
        while matched < len(drafts) and preds[matched] == drafts[matched]:
            matched += 1
        accepted = drafts[:matched]
        stats["rounds"] += 1
        stats["offered"] += len(drafts)
        stats["matched"] += matched
        stats["rows"] += len(rows)
        for i in range(matched):
            if i < len(stats["depth"]):
                stats["depth"][i] += 1
        keep = matched + 1
        t0 = time.time()
        eng.batch_rollback(keep)
        stats["rollback_ms"] += (time.time() - t0) * 1000.0
        stats["kept"] += keep
        out.extend([pending] + accepted)           # the rows this round commits
        pending = preds[matched]
        q += keep
    stats["rate"] = (stats["matched"] / float(stats["offered"])) if stats["offered"] else 0.0
    return out[len(ids):len(ids) + n], stats


def score_drafts(eng, ids, n, drafter, k=4):
    """Plain greedy decode of ``n`` tokens with ``drafter`` drafting after every step.

    ``drafter`` is anything with ``reset()`` and ``draft(prefix, hin, k)``: the integrated
    :class:`Head`, or :class:`PackDrafter` with a second, smaller engine.  Returns
    ``(tokens, stats)``; ``tokens`` is exactly ``eng.generate(ids, n)``, because the drafts are
    scored, never substituted.  ``stats`` has ``steps``, ``offered``, ``matched``, the match count per
    draft depth (``depth[i]`` = how often draft i + 1 matched) and ``rate`` = matched/offered.
    """
    drafter.reset()
    eng.reset()
    hiddens = {}
    for i, t in enumerate(ids[:-1]):
        eng.feed(t, i)
        hiddens[i] = eng.get_hidden()
    out = [eng.step(ids[-1], len(ids) - 1)]        # the model's token at len(ids)
    hiddens[len(ids) - 1] = eng.get_hidden()
    seed = getattr(drafter, "seed", None)
    if seed is not None:
        rows = [(p, hiddens[p], ids[p + 1]) for p in range(len(ids) - 1)]
        if getattr(drafter, "seed_last", False):
            # the whole prompt: the last row's next token is the one the model just produced
            rows.append((len(ids) - 1, hiddens[len(ids) - 1], out[0]))
        seed(rows)
    scored = []
    while len(out) < n:
        prefix = list(ids) + out
        base = len(prefix)                          # the position of the first drafted token
        # The head's row q reads the residual at q and embeds the token at q + 1; it drafts rows
        # len(prefix) - 2 .. and predicts from len(prefix) on, so the residual to pass is the one at
        # len(prefix) - 2. After the step above (at base - 1) `get_hidden()` is exactly that.
        drafts = drafter.draft(prefix, eng.get_hidden(), k)
        scored += [(base + i, i, d) for i, d in enumerate(drafts)]
        out.append(eng.step(out[-1], base - 1))
        if seed is not None:
            seed([(base - 1, eng.get_hidden(), out[-1])])
    at = {len(ids) + i: t for i, t in enumerate(out)}
    stats = {"steps": len(out) - 1, "offered": 0, "matched": 0, "depth": [0] * max(0, int(k))}
    for pos, depth, cand in scored:
        if pos not in at:                           # a draft past the end of the run is not offered
            continue
        stats["offered"] += 1
        if at[pos] == cand:
            stats["matched"] += 1
            if depth < len(stats["depth"]):
                stats["depth"][depth] += 1
    stats["rate"] = (stats["matched"] / float(stats["offered"])) if stats["offered"] else 0.0
    return out, stats
