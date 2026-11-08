# Per-Slot-Refactor der Plain-Engine — konkreter Code-Plan

Status: **NO-GO/archiviert (2026-11-08).** **Kein Produktivcode geändert.** Das ist der Phase-1-Code-Plan aus
`docs/CONTINUOUS-BATCHING.md` (Abschnitt 5); der Phase-0-Spike hat das 1,5x-Gate mit 0,69x verfehlt. Der Plan ist
nicht autorisiert und bleibt nur als Referenz liegen, bis ein neuer Spike das Gate besteht.

## 0. Kurzfassung

- Die Plain-Engine (`kindi/engine.py`, Bonsai 2B/4B/9B, eine GPU) bekommt **feste Slots**: K/V-Cache,
  GDN-Rekurrenz (`M`) und Conv-State (`conv`) erhalten eine führende Slot-Dimension.
- **Slot 0 bleibt exakt der heutige Ein-Sequenz-Zustand.** `Engine.Kc`, `Engine.Vc`, `Engine.M` und
  `Engine.conv` bleiben als Views auf Slot 0 bestehen, damit `cache.Runner`, `chunk1` und `spec`
  unverändert weiterlaufen.
- Ein neuer Adapter `kindi/batchengine.py` (`BatchEngine`) verwaltet Slot-Besitz, Token-Positionen
  und Logit-Snapshots pro Slot. Der Decode läuft in diesem Refactor **sequenziell pro Slot**
  (B einzelne Kernel-Launches) — der Durchsatzhebel (ein GEMM für B Zeilen) ist Phase 2.
- Sampling bekommt einen optionalen Pfad für **explizite Logits**, damit ein Snapshot später
  gesampelt werden kann, ohne dass `eng.ylog` noch dem Slot gehört.
- Neue Tests: `tests/test_engine_slots.py`, `tests/test_batchengine.py`,
  `tests/test_sampling_logits.py` (CPU, CI) und `tests/gpu/test_slot_parity.py` (manuell, GPU).

Nicht Teil dieses Refactors: Scheduler/Server, Paged KV, Prefix-Sharing, Speculative Decoding,
`TPEngine`, CPU-Engine, Gemma 4, Nanbeige, Fast Head im Batch, neue Kernel.

## 1. Ist-Zustand: was genau stört

| Ort | Zustand heute | Problem für Slots |
|---|---|---|
| `Engine.__init__` (`engine.py:88`) | `self.M`, `self.conv`, `self.Kc`, `self.Vc` werden einmal pro Engine alloziert (`engine.py:184-188`) | ein Satz State für alle Sequenzen |
| `Engine.reset` (`engine.py:262`) | `for a in self.M + self.conv + self.Kc + self.Vc: a.fill(0)` | löscht alle Sequenzen; keine Slot-ID |
| `Engine.gdn` (`engine.py:274`) | nutzt `self.conv[il]` (`282`) und `self.M[il]` (`284`) | fester State |
| `Engine.attn` (`engine.py:294`) | schreibt `self.Kc[kvi][pos]` / `self.Vc[kvi][pos]` (`305/306`), liest `self.Kc[kvi][:n]` (`309`) | Position global, ein K/V-Bereich |
| `Engine.body` / `feed` / `step` / `generate` | reichen `pos` durch, keinen Slot | ein Sequenzpfad |
| `Engine.feed_batch` (`engine.py:351`) | ruft `chunk1.feed_batch(self, toks, pos)` | Prompt-Chunk einer Sequenz |
| `chunk1.feed_chunk` | schreibt `eng.Kc[kvi][pos:pos+T]`, nutzt `eng.M[il]`, `eng.conv[il]` | dito |
| `Sampler.__call__` (`sampling.py:32`) | liest `eng.ylog`, `eng.ylog_n`, `eng.ylog_map`, `eng.aout` | Logits sind ein globaler Puffer |
| `cache.Runner` (`cache.py`) | hostet `eng.M`, `eng.conv`, `eng.Kc`, `eng.Vc`; `prepare` (`560`) ruft `feed_batch`/`feed`/`step`; `step` (`684`) | erwartet genau einen aktiven Zustand |

Wichtig: Die **Workspace-Puffer** (`self.h`, `self.xp`, `self.qkv`, `self.sbuf`, `self.apart`, …)
sind heute schon geteilt und bleiben es. Zwei Slots dürfen sich innerhalb eines Layers nicht
verschränken; pro Scheduler-Tick wird Slot für Slot gerechnet. Das ist für Phase 1 gewollt.

## 2. Design in einem Satz

Persistenter State bekommt eine führende Slot-Dimension; Slot 0 wird als View exportiert, damit
jeder bestehende Aufrufer unverändert bleibt; alles Neue liegt hinter `BatchEngine` und ist ohne
`slots > 1` inert.

Invarianten:

1. `Engine.Kc[kvi] is Engine.KcS[kvi][0]` (und analog `Vc`, `M`, `conv`): Schreiben in die View
   schreibt in den Slot-0-Speicher.
2. `Runner` sieht weiterhin **einen** Slot: `nbytes` der Views, `kv_tok`, Checkpoints und
   `engine_fingerprint` bleiben unverändert groß.
3. Eine Slot-Nummer wird nur an `reset`, `gdn`, `attn`, `body`, `feed`, `feed_batch`, `step`,
   `generate` und `chunk1.*` durchgereicht — nie an `head()` (der Kopf arbeitet auf dem Workspace).
4. Slot 0 verhält sich bitgleich zu heute; `kindi verify` muss PASS bleiben.
5. `slots=1` ist Default; `BatchEngine` wird nur explizit konstruiert.

## 3. Datenmodell

### 3.1 Allokation mit Slot-Views

```python
# kindi/engine.py — neue reine Helfer, von np.zeros und cp.zeros nutzbar
def slot_state_bytes(dims):
    """Bytes des rekursiven States (M + conv) für EINEN Slot."""
    interval = dims["full_attention_interval"]
    nv, hk = dims["gdn_inner"] // dims["state_size"], dims["state_size"]
    conv_ch = 2 * dims["groups"] * hk + nv * hk
    per_layer = [nv * hk * hk if (il + 1) % interval else 1 for il in range(dims["layers"])]
    return 4 * (sum(per_layer) + dims["layers"] * 3 * conv_ch)


def kv_bytes_per_token(dims):
    """Bytes K/V (fp16, K und V) für EINEN Slot und EINE Position."""
    return 2 * (dims["layers"] // dims["full_attention_interval"]) * dims["kv_heads"] * dims["head_dim"] * 2


def alloc_state(zeros, dims, ctx, slots):
    """(MS, convS, KcS, VcS) mit führender Slot-Dimension; zeros(shape, dtype) ist np.zeros oder cp.zeros."""
    interval = dims["full_attention_interval"]
    nv, hk, nhkv, hd = dims["gdn_inner"] // dims["state_size"], dims["state_size"], dims["kv_heads"], dims["head_dim"]
    conv_ch = 2 * dims["groups"] * hk + nv * hk
    MS = [zeros((slots, nv, hk, hk) if (il + 1) % interval else (slots, 1), np.float32)
          for il in range(dims["layers"])]
    convS = [zeros((slots, 3 * conv_ch), np.float32) for _ in range(dims["layers"])]
    natt = dims["layers"] // interval
    KcS = [zeros((slots, ctx, nhkv, hd), np.float16) for _ in range(natt)]
    VcS = [zeros((slots, ctx, nhkv, hd), np.float16) for _ in range(natt)]
    return MS, convS, KcS, VcS
```

Im Konstruktor (nach dem Kontext-Fit, an der Stelle der heutigen Allokation):

```python
self.nslots = max(1, int(slots))
self.MS, self.convS, self.KcS, self.VcS = alloc_state(cp.zeros, self.dims, self.ctx, self.nslots)
self.M = [a[0] for a in self.MS]         # Slot 0: Runner/chunk1/spec wie bisher
self.conv = [a[0] for a in self.convS]
self.Kc = [a[0] for a in self.KcS]
self.Vc = [a[0] for a in self.VcS]
```

`self.sbuf`, `self.apart`, `self.cos_t`, `self.sin_t` und die `chunk`-Puffer bleiben **einmal**
vorhanden (Workspace, keine Slot-Dimension). Sie werden pro Slot nacheinander benutzt; Phase 2
braucht für den echten Batch-Decode `sbuf`/`apart` mit Slot- oder Zeilen-Dimension.

### 3.2 Speicher pro Slot (zur Kontrolle der Fit-Formel)

| Modell | K/V pro Slot bei ctx=4096 | M + conv pro Slot | Summe |
|---|---:|---:|---:|
| 2B (6 Attention-Layer, NHKV=2, HD=256) | 48 MiB | ~18 MiB | ~66 MiB |
| 9B (Attention-Anteil, NHKV/HD laut Pack) | ~128 MiB | ~24 MiB | ~152 MiB |

Die Zahlen folgen aus `kv_bytes_per_token(dims) * ctx` und `slot_state_bytes(dims)`; sie sind eine
Rechnung, keine Messung. `slot_state_bytes` und `kv_bytes_per_token` werden in den CPU-Tests gegen
kleine Dims geprüft, damit die Fit-Formel nicht still von der Allokation abdriftet.

## 4. Betroffene Funktionen

### 4.1 Signatur-Änderungen (alle mit Default, damit jeder Altaufrufer gilt)

| Funktion | Datei | heute | neu |
|---|---|---|---|
| `Engine.__init__` | `engine.py:88` | `(pack, ctx=4096, clusters=None, log=print)` | `(..., slots=1)`; setzt `self.nslots` |
| `Engine.reset` | `engine.py:262` | `reset()` | `reset(slot=0)`; neu `reset_slot(slot)` |
| `Engine.gdn` | `engine.py:274` | `(il, h)` | `(il, h, slot=0)` |
| `Engine.attn` | `engine.py:294` | `(il, h, pos, kvi)` | `(il, h, pos, kvi, slot=0)` |
| `Engine.ffn` | `engine.py` | `(il, h)` | unverändert (kein persistenter State) |
| `Engine.body` | `engine.py` | `(tok, pos)` | `(tok, pos, slot=0)` |
| `Engine.feed` | `engine.py` | `(tok, pos)` | `(tok, pos, slot=0)` |
| `Engine.feed_batch` | `engine.py:351` | `(toks, pos)` | `(toks, pos, slot=0)` |
| `Engine.step` | `engine.py:356` | `(tok, pos)` | `(tok, pos, slot=0)` |
| `Engine.generate` | `engine.py:395` | `(ids, n)` | `(ids, n, slot=0)` |
| `make_engine` | `engine.py:408` | `(pack, ctx=4096, clusters=None, log=print)` | `(..., slots=1)` |
| `chunk1.feed_chunk` | `chunk1.py` | `(eng, toks, pos)` | `(eng, toks, pos, slot=0)` |
| `chunk1.feed_batch` | `chunk1.py:195` | `(eng, toks, pos)` | `(eng, toks, pos, slot=0)` |
| `Sampler.__call__` | `sampling.py:32` | `(greedy_tok, temperature, top_p, top_k, rng, min_p=0.0, penalty=None)` | zusätzlich `logits=None, logits_map=None, logits_n=None, aout=None` |
| `BatchEngine` | **neu** `batchengine.py` | — | `claim/free/reset_slot/prefill/decode/snapshot` |

`Engine.embed`, `Engine.head` und `Engine.ffn` bleiben unverändert. `head()` schreibt weiterhin
`eng.ylog`/`ylog_n`/`ylog_map`/`aout`; der Aufrufer muss die Logits **unmittelbar danach**
schnappschussen (siehe `BatchEngine._snapshot`).

### 4.2 `reset`, `gdn`, `attn` — die drei echten Änderungen im Rechenpfad

```python
def reset_slot_arrays(arrays, slot):
    """arrays: MS, convS, KcS, VcS (oder eine dieser Listen). Nur `slot` wird genullt."""
    for a in arrays:
        a[slot].fill(0)

def reset(self, slot=0):
    reset_slot_arrays(self.MS + self.convS + self.KcS + self.VcS, slot)

def reset_slot(self, slot):
    """Slot freigeben: der State ist danach wie nach einem frischen Laden."""
    self.reset(slot)
```

```python
def gdn(self, il, h, slot=0):
    ...
    self.k.k_gdn_conv(..., self.convS[il][slot], ...)
    self.k.k_gdn_delta((self.NV,), (self.HK,), (self.MS[il][slot], ...))
```

```python
def attn(self, il, h, pos, kvi, slot=0):
    ...
    K = self.KcS[kvi][slot]
    V = self.VcS[kvi][slot]
    K[pos] = self.kro.reshape(self.NHKV, self.HD)
    V[pos] = self.qkvbuf.reshape(self.NHKV, self.HD)
    n = pos + 1
    Kall, Vall = K[:n], V[:n]
```

`body`, `feed`, `step`, `generate` reichen `slot` nur durch:

```python
def body(self, tok, pos, slot=0):
    ...
    h = self.attn(il, h, pos, kvi, slot) if (il + 1) % interval == 0 else self.gdn(il, h, slot)

def generate(self, ids, n, slot=0):
    self.reset(slot)
    ...
```

### 4.3 `chunk1` — Prefill eines Slots

```python
def feed_chunk(eng, toks, pos, slot=0):
    ...
    eng.KcS[kvi][slot, pos:pos + T] = KRO[:T * L].reshape(T, NHKV, HD)
    eng.VcS[kvi][slot, pos:pos + T] = VB[:T * L].reshape(T, NHKV, HD)
    ...
    eng.ck.k_gdn_conv_T(..., eng.convS[il][slot], ...)
    eng.ck.k_gdn_delta_T((NV,), (HK,), (eng.MS[il][slot], ...))

def feed_batch(eng, toks, pos, slot=0):
    ...
    feed_chunk(eng, list(toks[i:i + n]), pos + i, slot)
    ...
    eng.feed(toks[i], pos + i, slot)
```

Der Chunk-Pfad bleibt **eine Sequenz pro Aufruf**. Die `chunk`-Puffer (`eng.chunk`) werden nicht
pro Slot vervielfacht; der Scheduler liest Prefills seriell (Phase 2 kann Zeilen-/Slot-Offsets in
den Puffern ergänzen).

## 5. Slot-Zahl und Kontext-Fit

Heute (`engine.py:165-169`):

```python
per_pos = (self.NATT * 2 * self.NHKV * self.HD * 2 + self.NH * 4
           + (self.NH * self.HD * 4) // 256 + self.dims["rope_dim"] * 4)
free = memGetInfo()[0] + pool.free_bytes()
budget = free - KINDI_MEM_RESERVE_MB * 2**20
fit = max(0, budget) // per_pos // 256 * 256
```

Neu, als reine Funktion (testbar ohne GPU):

```python
def fit_slots(free, reserve, ctx, max_slots, state_one, kv_tok, per_pos_shared, min_ctx=1024):
    """(slots, fit_ctx). Erst Slots reduzieren, dann den Kontext kürzen.

    Slot 0 zahlt wie heute nur über `reserve` (die 96 MiB Reserve deckten den State schon immer);
    jeder ZUSÄTZLICHE Slot zahlt seinen State und sein K/V pro Position.
    """
    budget = max(0, int(free) - int(reserve))
    slots = max(1, int(max_slots))
    while slots > 1:
        per_pos = per_pos_shared + slots * kv_tok
        fit = max(0, budget - (slots - 1) * state_one) // per_pos // 256 * 256
        if fit >= ctx:
            return slots, ctx
        slots -= 1
    per_pos = per_pos_shared + kv_tok
    fit = max(0, budget) // per_pos // 256 * 256
    return 1, fit
```

Aufruf im Konstruktor:

```python
per_pos_shared = self.NH * 4 + (self.NH * self.HD * 4) // 256 + self.dims["rope_dim"] * 4
state_one = slot_state_bytes(self.dims)
kv_tok = kv_bytes_per_token(self.dims)
self.nslots, fit = fit_slots(free, reserve, self.ctx, slots, state_one, kv_tok, per_pos_shared)
if fit < 1024:
    raise SystemExit("kindi: not enough GPU memory for this model: %d MiB free after the weights"
                     % (free // 2 ** 20))
if fit < self.ctx:
    log("context reduced from %d to %d tokens to fit %d MiB of free GPU memory" % (self.ctx, fit, free // 2 ** 20))
    self.ctx = fit
if self.nslots > 1:
    log("slots: %d (%.0f MB per slot at ctx %d)" % (self.nslots, (state_one + kv_tok * self.ctx) / 2 ** 20, self.ctx))
```

Eigenschaften:

- `slots=1` liefert **exakt** die heutige Formel (`budget // per_pos`), also keine Kontextänderung
  für bestehende Nutzer.
- Bei `slots>1` wird zuerst die Slot-Zahl gesenkt, bevor der Kontext leidet; das entspricht der
  Empfehlung in `docs/CONTINUOUS-BATCHING.md` („Slots aus freiem VRAM, nicht die Kontextlänge
  reduzieren“).
- Unter 1024 Token Kontext bricht der Start mit derselben Meldung wie heute ab.
- `KINDI_BATCH_SLOTS` wird **noch nicht** im Server gelesen. Der Parser existiert für Phase 3:

```python
def requested_slots(env=None):
    """KINDI_BATCH_SLOTS: 1..16; alles andere fällt auf 1 zurück (kein Verhalten ändern)."""
    raw = (os.environ if env is None else env).get("KINDI_BATCH_SLOTS", "1")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(n, 16))
```

`make_engine(..., slots=1)` reicht den Wert an `Engine` weiter; `verify`, `bench` und `chat`
bleiben bei 1. GPU-Tests konstruieren `make_engine(pack, ctx=4096, slots=4)` direkt.

## 6. `BatchEngine` (neu: `kindi/batchengine.py`)

Der Adapter kapselt Slot-Besitz und Buchführung. Er enthält **keinen** Kernel-Code; die
Kernänderung in Phase 2 ersetzt nur den Rumpf von `decode`.

```python
import collections

import numpy as np
try:
    import cupy as cp
except ImportError:               # die CPU-Engine / ein Host ohne CuPy
    cp = None

SlotLogits = collections.namedtuple("SlotLogits", "slot logits n ymap greedy")


class SlotError(RuntimeError):
    """Slot doppelt belegt/freigegeben oder außerhalb von 0..nslots-1."""


class BatchEngine(object):
    def __init__(self, eng, nslots=None):
        self.eng = eng
        self.nslots = int(nslots or getattr(eng, "nslots", 1))
        self.owner = [None] * self.nslots       # Sequence/Request-ID, nur für Invarianten
        self.tokens = [[] for _ in range(self.nslots)]   # Token, deren Wirkung im State ist
        self.logits = [None] * self.nslots      # Snapshot je Slot (Device oder NumPy)
        # eigener Argmax-Puffer pro Slot: k_topk/k_band lesen den Maximalwert aus aout, das nach
        # einem Batch-Tick sonst zum zuletzt gerechneten Slot gehört
        self.aout = (cp.zeros(self.nslots, dtype=cp.int32) if cp is not None and hasattr(eng, "k")
                     else np.zeros(self.nslots, dtype=np.int32))
        if getattr(eng, "hsl", None) is not None:
            eng.shortlist = 0                   # MVP: exakter Head, keine Shortlist pro Slot

    def claim(self, owner=None):
        """Ersten freien Slot nehmen und nullen; ohne freien Slot SlotError."""
        for s in range(self.nslots):
            if self.owner[s] is None:
                self.owner[s], self.tokens[s] = owner or s, []
                self.eng.reset(s)
                return s
        raise SlotError("no free slot (%d)" % self.nslots)

    def free(self, slot):
        """Slot nullen und freigeben; doppeltes Freigeben ist ein Fehler."""
        self._check(slot)
        if self.owner[slot] is None:
            raise SlotError("slot %d is already free" % slot)
        self.eng.reset(slot)
        self.owner[slot], self.tokens[slot] = None, []

    def reset_slot(self, slot):
        self._check(slot)
        self.eng.reset(slot)
        self.tokens[slot] = []

    def prefill(self, slot, toks):
        """State = toks; gibt die Logits für das erste neue Token zurück."""
        self._check(slot)
        if not toks:
            raise ValueError("prefill needs at least one token")
        self.eng.feed_batch(toks[:-1], 0, slot)
        greedy = self.eng.step(toks[-1], len(toks) - 1, slot)
        self.tokens[slot] = list(toks)
        return self._snapshot(slot, greedy)

    def decode(self, batch):
        """batch: [(slot, tok), ...] — tok ist das zuletzt gesampelte Token des Slots.
        Gibt die Logits des jeweils nächsten Tokens zurück; Phase 2 ersetzt nur diesen Rumpf."""
        out = []
        for slot, tok in batch:
            self._check(slot)
            pos = len(self.tokens[slot])
            greedy = self.eng.step(tok, pos, slot)
            self.tokens[slot].append(tok)
            out.append(self._snapshot(slot, greedy))
        return out

    def _snapshot(self, slot, greedy):
        e = self.eng
        n = int(e.ylog_n)
        buf = self.logits[slot]
        if buf is None or buf.shape[0] != n:
            buf = self.logits[slot] = _copy_of(e.ylog[:n])
        else:
            _copy_into(buf, e.ylog[:n])
        self.aout[slot] = e.aout[0]             # pro Slot für k_topk/k_band
        return SlotLogits(slot, buf, n, e.ylog_map, greedy)
```

Die Kopier-Helfer halten den Adapter frei von CuPy-Zwang (CPU-Tests nutzen NumPy):

```python
def _copy_of(src):
    return np.array(src, copy=True) if isinstance(src, np.ndarray) else cp.array(src, copy=True)


def _copy_into(dst, src):
    if isinstance(dst, np.ndarray):
        np.copyto(dst, src)
    else:
        cp.copyto(dst, src)
```

Wichtige Verträge:

- `prefill` speist `toks[:-1]` per `feed_batch` (Chunk-Pfad) und ruft `step(toks[-1], len-1, slot)`;
  danach ist `len(self.tokens[slot]) == len(toks)` und die Position des nächsten `decode` ist
  `len(toks)` — identisch zu `Runner.step`.
- `decode` fügt das **gespeiste** Token an, nicht das neue. Der Aufrufer sampelt aus den
  zurückgegebenen Logits und reicht das neue Token beim nächsten Tick an `decode`.
- `_snapshot` muss direkt nach `step` laufen; `eng.ylog`/`ylog_n`/`ylog_map`/`aout` gehören sonst
  schon dem nächsten Slot. Der Snapshot ist eine Kopie, keine View.
- `free` nullt den State, gibt aber keinen Speicher zurück (feste Slots, MVP).
- `BatchEngine` ist ohne `KINDI_BATCHING` ungenutzt; `slots=1` erlaubt trotzdem den B=1-Paritätstest.

## 7. Sampling pro Slot

`Sampler.__call__` bekommt optionale Logits, damit der Snapshot gesampelt werden kann:

```python
def __call__(self, greedy_tok, temperature, top_p, top_k, rng, min_p=0.0, penalty=None,
             logits=None, logits_map=None, logits_n=None, aout=None):
    e = self.eng
    yl = e.ylog if logits is None else logits
    nl = int(e.ylog_n if logits_n is None else logits_n)
    ymap = e.ylog_map if logits is None else logits_map
    ao = e.aout if aout is None else aout
    ...
    e.k.k_topk((1,), (1024,), (yl, ao, np.int32(nl), ...))
```

Analog bekommen `_logits()` und `_penalized()` die drei optionalen Argumente. Der Default-Pfad
(`logits is None`) bleibt zeichengleich; `tests/test_sampling.py` und der Server sind unberührt.
Der `aout`-Parameter ist nötig, weil `k_topk`/`k_band` den Argmax-Wert aus `eng.aout` lesen, nach
einem Batch-Tick aber der letzte Slot dort steht.

Aufruf im Scheduler (Phase 3):

```python
row = be.decode([(s, next_tok[s]) for s in active])
for r in row:
    t = sample(r.greedy, ..., logits=r.logits, logits_n=r.n, logits_map=r.ymap, aout=be.aout[r.slot:r.slot + 1])
```

## 8. Kompatibilität

| Nutzer | Verhalten nach dem Refactor |
|---|---|
| `cache.Runner` | unverändert; `eng.M`/`conv`/`Kc`/`Vc` sind Slot-0-Views, `_set_host` schreibt durch |
| `cache._save_ckpt`/`_park` | hostet weiterhin nur Slot 0; `state_bytes`/`kv_tok` bleiben gleich |
| `chunk1` (Single-Pfad) | Default `slot=0`; identische Kernel-Argumente wie heute |
| `spec.py` | ruft `eng.feed_batch`/`step` ohne Slot → Slot 0; `TPEngine` unberührt |
| `kindi/tp.py`, `cpu.py`, `gemma4*`, `nanbeige*` | keine Änderung |
| `engine_fingerprint` | `_ARITH` enthält `engine.py`/`chunk1.py`: gespeicherte Openings werden **einmalig** verworfen (sicher, aber im Bericht erwähnen) |
| `kindi verify` | unverändert, weil `slots=1` Default; muss PASS bleiben |
| `kindi bench` | unverändert; keine Batch-Messung in diesem Refactor |

## 9. Umsetzung in Commits

| # | Dateien | Inhalt | Tests vor dem Commit |
|---|---|---|---|
| 1 | `kindi/engine.py`, `tests/test_engine_slots.py` | `slot_state_bytes`, `kv_bytes_per_token`, `alloc_state`, `fit_slots`, `requested_slots`; Slot-Allokation und Views; `reset(slot)`, `gdn/attn/body/feed/step/generate(..., slot=0)` | neue CPU-Tests, `pytest tests -q`, `compileall` |
| 2 | `kindi/chunk1.py`, `kindi/engine.py` | `feed_chunk`/`feed_batch` mit Slot; `Engine.feed_batch(toks, pos, slot=0)` | GPU: Prefill-Parität; CPU: `chunk1.feed_batch` mit Fake (nur wenn `chunk_ok=False`, kein Kernel) |
| 3 | `kindi/batchengine.py`, `tests/test_batchengine.py` | Adapter inkl. `SlotLogits`, Claim/Free, Logit-Snapshots | neue CPU-Tests |
| 4 | `kindi/sampling.py`, `tests/test_sampling_logits.py` | explizite Logits + `aout`; Default unverändert | neue CPU-Tests, `tests/test_sampling.py` |
| 5 | `tests/gpu/test_slot_parity.py`, `AGENTS.md` | GPU-Parität/Isolation, Handoff-Notiz | manuell auf der 8-GB-Karte |
| 6 | `docs/CONFIGURATION.md`, `docs/INSTALL.md` | `KINDI_BATCH_SLOTS` dokumentieren, **erst wenn die Lease frei ist** (siehe AGENTS.md) | Doku-Review |

Nach Commit 1–4 muss die CPU-Suite grün sein, ohne dass ein GPU-Pack vorhanden ist. Commit 5 ist
der einzige, der Hardware braucht.

## 10. Tests

### 10.1 CPU (CI, kein GPU-Pack)

`tests/test_engine_slots.py`

- `test_slot_state_bytes_matches_alloc_state`: für kleine `dims` (z. B. `layers=4`,
  `full_attention_interval=2`, `state_size=8`, `gdn_inner=16`, `groups=2`) gilt
  `slot_state_bytes(dims) == sum(a[0].nbytes for a in MS) + sum(c[0].nbytes for c in convS)`.
- `test_kv_bytes_per_token_matches_alloc_state`: analog gegen `KcS[0][0].nbytes // ctx`.
- `test_slot_views_write_through`: mit `alloc_state(np.zeros, ...)`; schreibe in
  `KcS[0][1, 5, 0, 0]`, lese über `Kc[0][5, 0, 0]`; schreibe in Slot 2 und prüfe, dass Slot 1
  unverändert bleibt.
- `test_fit_slots_single_slot_keeps_the_old_formula`: `fit_slots(free, 0, 4096, 1, ...)[1]` ist
  `free // per_pos // 256 * 256` (exakt die heutige Zeile).
- `test_fit_slots_reduces_slots_before_the_context`: Budget, bei dem 2 Slots 4096 schaffen, 4 nicht
  → `(2, 4096)`.
- `test_fit_slots_shortens_the_context_for_one_slot`: `(1, 2048)`.
- `test_fit_slots_too_little_memory_returns_below_min`: `fit < 1024`; der Konstruktor wirft.
- `test_requested_slots_parsing`: fehlend/`"0"`/`"abc"` → 1; `"4"` → 4; `"99"` → 16.
- `test_reset_slot_only_clears_one_slot`: NumPy-Arrays aus `alloc_state`, Slot 1 füllen,
  `reset_slot_arrays(MS + convS + KcS + VcS, 0)`, Slot 1 unverändert.

`tests/test_batchengine.py` (Fake-Engine, NumPy, keine GPU)

- `test_claim_returns_the_first_free_slot_and_resets_it`
- `test_free_resets_and_rejects_double_free`
- `test_prefill_feeds_all_but_the_last_token` (prüft `feed_batch(toks[:-1], 0, slot)` und
  `step(toks[-1], len(toks)-1, slot)`)
- `test_decode_steps_at_the_slot_position_and_appends_the_fed_token`
- `test_slots_keep_independent_tokens_and_positions` (verschachtelt Slot 0/1)
- `test_snapshot_copies_logits_per_slot` (nach zwei `decode`-Aufrufen sind beide Snapshots
  unverändert, obwohl `eng.ylog` weiterläuft)
- `test_prefill_rejects_an_empty_prompt`
- `test_slot_out_of_range_raises`

`tests/test_sampling_logits.py` (Fake-Engine ohne `k`/`aout`, `Sampler.gpu == False`)

- `test_explicit_logits_override_the_engine_buffer` (Engine-Logits sagen Token 0, explizite Logits
  Token 3, `top_k=1`, `temperature=1` → 3)
- `test_explicit_logits_map_returns_engine_token_ids` (`ymap=[10,11,12,13]`, bester Index 2 → 12)
- `test_default_path_still_reads_the_engine_buffer` (Regression zu `tests/test_sampling.py`)

### 10.2 GPU (manuell, `KINDI_TEST_GPU=1`, 8-GB-Karte)

`tests/gpu/test_slot_parity.py` (`pytest.importorskip("cupy")`, Skip ohne Env/Pack)

- `test_slots_one_is_unchanged`: `make_engine(pack, ctx=1024, slots=1).generate(reference.ids, 40)`
  gegen die heutigen Referenztokens (oder gegen `make_engine(..., slots=1)` vor/nach dem Refactor);
  zusätzlich `python -m kindi verify` als Gate.
- `test_prefill_slot_equals_feed`: `feed_batch(toks, 0, slot=2)` + `step` vs. `feed`-Schleife +
  `step` auf demselben Slot → gleiches Token; danach `KcS[kvi][2, :len(toks)]` gleich.
- `test_b1_batch_engine_matches_generate`: Prompt aus `reference.json`, 16 Token, `BatchEngine`
  mit 1 Slot vs. `Engine.generate`.
- `test_four_slots_interleaved_match_serial`: vier Prompts, vier Slots, abwechselnd je ein Token;
  jedes Ergebnis token-gleich zum seriellen Lauf (frischer Slot, `temperature=0`).
- `test_free_does_not_touch_other_slots`: Slot 1 füllen, Slot 0 freigeben, Slot 1 weiter dekodieren
  → gleiches Ergebnis wie vor dem Free.
- `test_slot_fit_on_the_real_pack`: `make_engine(pack, ctx=4096, slots=4)` lädt beim 2B; `nslots`
  und `ctx` werden geloggt (keine harte Erwartung an die Karte).

Vakuum-Nachweis: in einer Kopie `KcS[kvi][slot]` durch `KcS[kvi][0]` in `Engine.attn` ersetzen →
`test_four_slots_interleaved_match_serial` und `test_free_does_not_touch_other_slots` müssen
fehlschlagen. Ebenso `fit_slots`-Test: die `(slots-1)`-Subtraktion entfernen → der
Reduziert-Slots-vor-Kontext-Test muss fehlschlagen.

### 10.3 Bestehende Gates

- `python -m pytest tests/ -q` (CPU-Suite, inkl. `test_cache_decisions.py`, `test_server.py`,
  `test_sampling.py`) bleibt grün.
- `python -m compileall -q kindi`, `bash -n install.sh`.
- `python -m kindi verify` (GPU, 2B) muss PASS bleiben; `kindi bench` muss sich im Rahmen der
  Messstreuung nicht ändern.

## 11. Abnahmekriterien

1. `slots=1` ist verhaltensgleich: `verify` PASS, CPU-Suite grün, `bench` unverändert (Streuung).
2. `BatchEngine` mit einem Slot liefert für die vier Verify-Prompts dieselben 40 Token wie
   `Engine.generate`.
3. Vier Slots, verschachtelt dekodiert, liefern pro Slot dieselben Token wie vier serielle Läufe
   (16 Token, `temperature=0`).
4. `reset`/`free` eines Slots ändert keinen anderen Slot (Token-Parität + State-Isolation).
5. 2B lädt mit `ctx=4096` und 4 Slots auf der 8-GB-Karte; 9B mit 2 Slots; bei zu wenig Speicher
   sinkt die Slot-Zahl, dann der Kontext, nie ein OOM.
6. `Runner`-Cache (Checkpoints, Parken, Openings) funktioniert unverändert; der Fingerprint-Wechsel
   ist dokumentiert.
7. Keine neue Abhängigkeit, kein neuer Kernel, kein Serververhalten in diesem Refactor.

## 12. Risiken und offene Entscheidungen

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| Slot-View schreibt nicht durch (CuPy/NumPy) | Runner-Cache korrumpiert Slot 0 | `test_slot_views_write_through`; `_set_host` bleibt `copyto`/`set` |
| Gemeinsamer Workspace wird zwischen Slots verschränkt | falsche Logits, Datenleck | pro Tick Slot für Slot; `BatchEngine.decode` ist die einzige Schleife |
| Kontext-Fit ändert sich für `slots=1` | bestehende Nutzer verlieren Kontext | Formel für 1 Slot exakt beibehalten; Test pinnt sie |
| Chunk-Prefill mit falschem Slot-Offset | K/V eines Slots überschrieben | GPU-Prefill-Parität + Isolationstest |
| Logit-Snapshot zu spät | Sampling mit falschem Slot | `_snapshot` direkt nach `step`; Test mit zwei Slots |
| `eng.shortlist` (Fast Head) pro Slot | Shortlist des letzten Slots | MVP `shortlist=0`; Fast Head bleibt Phase 3 |
| Fingerprint-Wechsel | gespeicherte Openings einmalig ungültig | im Commit/Report nennen; `Openings` verwirft fremde Fingerprints selbst |
| Kein GPU-CI | Parity-Regressionen erst auf Hardware | CPU-Buchführungstests + manueller GPU-Lauf + Vakuum-Nachweis |

Offene Entscheidungen (vor Commit 1 zu treffen):

- **Reserve-Politik:** Slot 0 über die bestehende `KINDI_MEM_RESERVE_MB` decken (empfohlen, hält
  `slots=1` exakt) oder `slots * state_one` immer abziehen (sicherer, ändert aber den Kontext für
  bestehende Nutzer).
- **`KINDI_BATCH_SLOTS`:** jetzt nur Parser + Tests, im Server erst mit dem Scheduler (Phase 3)
  lesen — oder schon in `make_engine` lesen, damit ein GPU-Test die Fit-Policy ohne Codeänderung
  treiben kann.
- **Slot-Zahl 4 vs. dynamisch:** MVP feste Obergrenze 4 beim 2B; beim 9B 2. Die Fit-Formel erlaubt
  beides, die Defaults gehören in den Scheduler-Plan.
- **Chunk-Größe pro Slot:** `eng.chunk["T"]` bleibt global; bei langen Prompts und vielen Slots
  kann ein kleinerer Chunk (`KINDI_CHUNK`) die Latenz der anderen Slots senken. Messung in Phase 2.

## 13. Was danach kommt

- **Phase 2:** `BatchEngine.decode` durch einen echten Batch-Pfad ersetzen (ein GEMM für B Zeilen,
  `k_att_flash16`/GDN mit Slot-Argument, `k_argmax_rows`). Der Vertrag (`SlotLogits` pro Slot)
  bleibt; nur der Rumpf ändert sich.
- **Phase 3:** `BatchScheduler`, `Kindi.submit()`, SSE pro Sequenz, `/health`-Metriken,
  `KINDI_BATCHING=1`; dann `KINDI_BATCH_SLOTS` in `docs/CONFIGURATION.md` dokumentieren.
- **Phase 4:** Per-Slot-Prefix-Cache (LCP pro Slot), Checkpoints pro Slot, 9B-Speicherprofil.
