# Continuous Batching in Kindi — Architektur und Umsetzungsplan

Status: **NO-GO (2026-11-08).** Der Phase-0-Spike erreichte B=4 mit 131,8 tok/s = 0,69x Single-Decode; das
1,5x-Gate ist FAIL. Kein Produktivcode geändert. Dieses Dokument ist ein Mess- und Designprotokoll, kein
freigegebenes MVP. Archiviert in [ROADMAP.md](ROADMAP.md).

## 0. Kurzfassung

Continuous Batching ist in Kindi machbar, aber **kein reines Server-Feature**. Der Server kann heute
nur eine Anfrage gleichzeitig bedienen, weil die Engine genau einen Satz aus KV-Cache, GDN-Rekurrenz
und Workspace besitzt. Die vorhandenen Batch-Kernel verarbeiten mehrere **Token einer Sequenz**
(Prompt-Chunks, Spec-Verify), nicht mehrere Sequenzen. Der eigentliche Aufwand liegt deshalb in:

1. einem Sequenz-/Slot-Modell für KV, GDN-State und Conv-State,
2. einem Batch-Decode, der die Gewichte einmal für mehrere Sequenzen liest,
3. einem Scheduler und Streaming-Umbau im Server,
4. später der Integration des bestehenden Prompt-Caches.

**Empfehlung:** Zuerst einen 4–6-Tage-Spike für den Batch-GEMM machen. Nur wenn dieser bei vier
unabhängigen Sequenzen mindestens ~1,5× Gesamtdurchsatz gegenüber einem einzelnen Decode zeigt,
lohnt der Ausbau. Danach ein MVP für die **2B/4B/9B-Engine auf einer GPU**, feste Slots, ohne
Speculative Decoding, ohne Plan-Engine und zunächst ohne Cross-Request-Prompt-Cache. **Dieses Gate ist inzwischen
FAIL (0,69x); die Empfehlung ist nicht freigegeben.**

**Aufwandsschätzung** (ein erfahrener Engineer, der Repo und CUDA/CuPy kennt):

| Umfang | Personentage | Kalender |
|---|---:|---:|
| Spike + MVP (2B, 4 Slots, Decode-Batch, Scheduler, Streaming, keine Cache-/Spec-/Plan-Integration) | 26–38 | 5–8 Wochen |
| Production v1 (zusätzlich Per-Slot-Prefix-Cache, 9B-Speicherprofil, Fehlerpfade, Doku, Benchmarks) | 38–55 | 8–11 Wochen |
| Zusätzlich Plan-Engine/Speculative Decoding (27B/Multi-Card, pro-Sequenz-Spec) | +25–35 | 13–18 Wochen gesamt |

Das ist eine Schätzung, keine Messung. Der größte Unsicherheitsfaktor sind die neuen Decode-Kernel,
nicht der Scheduler.

## 1. Befund im heutigen Code

### 1.1 Server: eine Anfrage, eine GPU-Queue

- `kindi/server.py` dokumentiert ausdrücklich „One request at a time“.
- `GPU_THREAD = ThreadPoolExecutor(max_workers=1)` führt jede Engine-Arbeit auf genau einem Thread aus.
- `K.lock` umschließt den gesamten `generate`-Lauf; weitere Anfragen warten im Threadpool.
- `K.busy` ist ein einzelner Statusdatensatz; `/health` kann nur eine aktive Anfrage melden.
- Streaming und Non-Streaming iterieren beide über `K.generate(req)` und halten den Lock bis zum Ende.

Das ist kein Scheduler, sondern eine FIFO-Warteschlange vor einem globalen Lock.

### 1.2 Engine: ein globaler Sequenzzustand

`kindi/engine.py` sagt es im Docstring: „One sequence at a time.“ Konkret global bzw. pro Engine:

- `self.Kc[kvi]`, `self.Vc[kvi]`: fp16-K/V-Cache je Attention-Layer, Form `[ctx, NHKV, HD]`.
- `self.M[il]`: GDN-Rekurrenzstate je Layer.
- `self.conv[il]`: Conv-Shiftregister je Layer.
- `self.h`, `self.xp`, `self.qkv`, `self.att`, `self.ffg`, `self.ffu`, `self.ylog`, ...: ein gemeinsamer
  Workspace.
- `reset()` löscht **alle** Sequenzen; es gibt keine Slot-ID.

Die K/V-Allokation wird beim Start anhand des freien Speichers auf `ctx` angepasst. Mehrere Sequenzen
würden heute denselben K/V-Bereich überschreiben.

### 1.3 Vorhandene Batch-Fähigkeit: Token, nicht Sequenzen

`kindi/chunk1.py` und `kindi/chunk.py` verarbeiten Prompt-Chunks:

- `k_mm_g4` / `k_mm_g4k` berechnen `Y[t][r] = sum_k X[t][k] W[r][k]` für `T` Zeilen. Die Gewichte
  werden pro Tile einmal gelesen; `T` ist heute die Zahl der Prompt-Token **einer** Sequenz.
- `k_gdn_conv_T` und `k_gdn_delta_T` laufen `T` Token **in Reihenfolge** mit genau einem Conv-/GDN-State.
- `k_att_flash16_T` liest `Kc`/`Vc` ab `pos` für **eine** Sequenz; `pos` und `T` sind global.
- `Engine.feed_batch(toks, pos)` ist ein Prompt-Pfad, kein Multi-Sequenz-Pfad.

Damit existiert bereits der wichtigste Baustein für den Durchsatz: ein GEMM, das die Gewichte für
mehrere Zeilen teilt. Es fehlt die **Sequenz-Dimension** und die per-Sequenz-Attention/Rekurrenz.

### 1.4 Cache: ein aktiver Zustand, geparkte Kopien im Host-RAM

`kindi/cache.py` (`Runner`) hält:

- genau eine aktive Konversation auf der Karte (`self.tokens`, `self.ckpt`),
- geparkte Konversationen als Host-Kopien von State und K/V (`self.parked`),
- Checkpoints des rekursiven States, teils im VRAM-Slot (`KINDI_CKPT_VRAM`),
- die Erwartung, dass `Runner.prepare()` den **einen** Engine-State umschaltet.

Der Cache ist gut für serielle Chat-/Agent-Turns, aber nicht für mehrere gleichzeitig aktive
Sequenzen. Ein naiver Umbau würde Checkpoints, Parken und K/V-Kopien gegeneinander korrumpieren.

### 1.5 Plan- und Spec-Engine: ebenfalls eine Sequenz

- `TPEngine` (`kindi/tp.py`) hat einen Satz K/V pro Karte und ein `ctx`; `KINDI_BATCH` bedeutet dort
  Prompt-Token pro Batch, nicht parallele Sequenzen.
- `spec.py` verarbeitet `T=1..TMAX` Token einer Sequenz pro Verify-Pass; die CUDA-Graph-Pläne sind
  auf genau diese eine Sequenz zugeschnitten.
- Die 2B-Plan-Engine ist nur mit `KINDI_TC_TMAX=4` token-exakt; der Default T=8 divergiert. Die
  9B-Plan-Engine lädt mit TC, stürzt aber im Graph-Launch ab. Beides ist ein separates Baustelle und
  darf nicht mit Continuous Batching vermischt werden.

### 1.6 Getrennte Backends

`gemma4.py`, `gemma4_tp.py` und `nanbeige.py` haben eigene Engines, eigene Caches und teils eigene
Prefill-Batches. Ein Multi-Sequenz-Scheduler müsste für jedes Backend eine eigene Implementierung
bekommen. Für den ersten Ausbau gilt: **nur Bonsai (qwen35), nur eine GPU, nur die Plain-Engine**.

## 2. Was der Durchsatzhebel wirklich ist

Der Decode ist bei diesen Modellen weitgehend bandbreitengebunden: Pro Token wird das gesamte
Gewichtspack gelesen. Eine grobe Rechnung mit den gemessenen Packs und Einzeltoken-Raten auf der
RTX 4060 Ti (8 GB):

| Modell | Pack | gemessene Rate | grober Gewichtsverkehr | Bandbreitenkopfraum bei ~288 GB/s |
|---|---:|---:|---:|---:|
| 2B, Plain TC=1 | 506 MB | 191,5 tok/s | ~97 GB/s | ~3,0× |
| 2B, Plan TC T=4 | 506 MB | 347,1 tok/s | ~176 GB/s | ~1,6× |
| 9B, Plain TC=0 | 2.406 MB | 86,5 tok/s | ~208 GB/s | ~1,4× |

Die Zahlen sind eine **grobe obere Abschätzung**, keine Messung des echten DRAM-Verkehrs (K/V,
Aktivierungen und Overhead fehlen). Sie zeigen aber die Größenordnung:

- Beim **2B** kann ein Batch aus 4 Sequenzen theoretisch maximal ~3× aggregierten Durchsatz
  bringen (Bandbreitengrenze; die 1,5× sind das reale MVP-Ziel).
- Beim **9B** ist die Karte bereits nahe an der Bandbreite; realistisch sind dort eher 1,1–1,3×,
  nicht 4×. Der 9B ist kein guter erster Batch-Kandidat.
- Beim **27B** passt das Pack auf 8 GB nicht; Continuous Batching ändert daran nichts.
- Bei **einer** Anfrage bringt Continuous Batching exakt nichts. Der Hebel gilt für mehrere
  gleichzeitige Clients/Agenten, nicht für einen einzelnen langen Stream.

Ohne einen Batch-Decode, der die Gewichte wirklich einmal für mehrere Sequenzen liest, ist der
Scheduler nur eine bessere Warteschlange. Deshalb ist der GEMM-Spike das erste Gate.

## 3. Scope des ersten Ausbaus

### MVP (Ziel)

- Eine GPU, Bonsai-2B/4B/9B, Plain-Engine (`engine.py`, nicht `TPEngine`).
- Feste Slots: standardmäßig 4 beim 2B, automatisch speicherangepasst beim 9B (2–4).
- Getrennter Per-Slot-State für K/V, GDN und Conv.
- Serieller Prefill (ein Sequenz-Chunk pro Scheduler-Tick), **gebatchter Decode** über alle aktiven
  Slots.
- Streaming und Non-Streaming, Cancellation, Stop-Sequenzen, Sampling pro Sequenz.
- Exakter Head; kein Fast Head im Batch (Fast Head später, da er pro Sequenz eigene kurze Listen hat).
- Feature-Flag `KINDI_BATCHING=1`; Default zunächst `0`.

### Ausdrücklich nicht im MVP

- 27B/Multi-Card, `TPEngine`, `KINDI_TP=1`.
- Speculative Decoding (`ngram`, `draft2b`, MTP), CUDA-Graph-Pläne.
- Gemma 4 / Nemotron, Nanbeige, CPU-Engine.
- Paged KV, Prefix-Sharing über Sequenzen, Cross-Request-Prefix-Cache und Checkpoints über
  Sequenzen hinweg.
- Dynamische Batch-Größen über die Slot-Zahl hinaus.
- Änderung der Token-Exaktheitsgarantie.

## 4. Zielarchitektur

### 4.1 Überblick

```text
Client
  │  HTTP / SSE
  ▼
FastAPI (async) ──► Render + Tokenize (CPU-Thread) ──► Admission
                                                        │
                                                        ▼
                                              BatchScheduler (1 GPU-Thread)
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                               ▼                               ▼
                   SlotPool                       Prefill-Pfad                    Decode-Pfad
              (K/V, GDN, conv)                (chunk1, 1 Sequenz)          (Batch über B Slots)
                        │                               │                               │
                        └───────────────────────────────┴───────────────┬───────────────┘
                                                                        ▼
                                                                  BatchEngine
                                                                        │
                                                                        ▼
                                                              Sampler pro Sequenz
                                                                        │
                                                                        ▼
                                                          per-Sequenz-Ausgabewarteschlange
                                                                        │
                                                                        ▼
                                                                     SSE / JSON
```

Der GPU-Thread besitzt weiterhin die einzige Engine und alle CUDA-Aufrufe. Der Unterschied ist, dass
er nicht mehr einen Request-Generator ausführt, sondern eine Schleife über viele `Sequence`-Objekte.

### 4.2 Datenmodell

```python
class Sequence:
    id: str
    tokens: list[int]           # Prompt + bereits erzeugte Token
    pos: int                    # nächste Position
    slot: int
    phase: str                  # "prefill" | "decode" | "done" | "cancelled"
    prompt_len: int
    max_new: int
    sampling: SamplingSpec      # temperature, top_p, top_k, min_p, seed, penalty
    stop: list[str]
    cancel: threading.Event
    out: queue.Queue            # ("text"|"reasoning"|"done"|"error", payload)
    stats: dict
```

```python
class Slot:
    id: int
    seq: Sequence | None
    Kc: list[cp.ndarray]        # je Attention-Layer: [ctx, NHKV, HD]
    Vc: list[cp.ndarray]
    M: list[cp.ndarray]         # je GDN-Layer: [NV, HK, HK]
    conv: list[cp.ndarray]      # je Layer: [3 * CONV_CH]
    tokens: list[int]           # nur für spätere Prefix-Erkennung
    kv_used: int
```

```python
class BatchEngine:
    nslots: int
    def reset_slot(self, slot: int) -> None
    def prefill(self, slot: int, toks: list[int], pos: int) -> logits | None
    def decode(self, slots: list[int], toks: list[int], positions: list[int]) -> list[logits]
    def free_slot(self, slot: int) -> None
```

Die bestehende `Engine` bleibt für `kindi verify`, `bench`, `chat` und den Single-Request-Pfad
unverändert. `BatchEngine` ist ein Wrapper/Adapter mit eigener Slot-Allokation; der Single-Pfad wird
nicht durch Batch-Code ersetzt. Das hält die Token-Exaktheit und die bestehenden Tests intakt.

### 4.3 KV- und State-Layout (MVP: feste Slots)

Empfohlen für den ersten Ausbau: **feste Slots mit vollem Kontext pro Slot**, keine Pages.

```text
Kc[layer][slot][ctx][NHKV][HD]   fp16
Vc[layer][slot][ctx][NHKV][HD]   fp16
M[layer][slot][NV][HK][HK]       fp32
conv[layer][slot][3 * CONV_CH]   fp32
```

Ein Slot wird beim Laden der Engine allokiert; die Zahl der Slots wird aus dem freien Speicher
bestimmt, nicht die Kontextlänge reduziert. Jede Sequenz erhält einen Slot für ihre gesamte
Lebensdauer; beim Ende wird der Slot zurückgesetzt und neu vergeben. Das ist echtes Continuous
Batching (Zu-/Abgang zwischen Decode-Schritten), aber ohne Paging und ohne Prefix-Sharing.

Speicherbedarf pro Slot bei `ctx=4096` (aus den Dimensionen berechnet):

| Modell | K/V | GDN-State | Conv | Summe/Slot | 4 Slots |
|---|---:|---:|---:|---:|---:|
| 2B | 48 MiB | 18 MiB | 1,7 MiB | ~68 MiB | ~271 MiB |
| 9B | 128 MiB | 24 MiB | 3,0 MiB | ~155 MiB | ~620 MiB |

Die Gewichte (2B ~506 MB, 9B ~2,4 GB) bleiben geteilt. Auf der 8-GB-Karte sind 4 Slots beim 2B
unproblematisch; beim 9B sind 2–4 Slots realistisch, abhängig von Desktop-Belegung, Workspace und
Prompt-Puffern. Die Engine muss die Slot-Zahl zur Laufzeit bestimmen und bei zu wenig Speicher
auf weniger Slots zurückfallen, nicht mit OOM starten.

**Warum nicht sofort Paged KV?** Feste Slots sind deutlich einfacher: keine Block-Tables, keine
Gather-/Scatter-Kernel, keine Fragmentierung, keine Eviction. Der Preis ist ein fester
Kontext pro Slot und keine gemeinsame Prefix-Nutzung. Für den ersten Produktivpfad ist das der
richtige Trade-off. Paged KV und Prefix-Sharing sind Phase 2/3, wenn der Batch-Durchsatz bewiesen ist.

### 4.4 Kernel-Strategie

Der Batch-Decode nutzt die vorhandene Zeilen-Dimension der Chunk-Kernel:

- **Lineare Produkte:** `k_mm_g4`/`k_mm_g4k` mit `T = B` Zeilen. Jede Zeile ist ein Token einer
  anderen Sequenz; `X`/`Y` werden mit `ldx`/`ldy` gestridet. Die Gewichte werden für alle B Zeilen
  einmal gelesen. Das ist der zentrale Hebel.
- **Norm/Fold/RoPE/SwiGLU:** `k_rmsfold_T`, `k_smfold_T`, `k_fold_fwht_T`, `k_rope_T` können bereits
  T Zeilen; sie brauchen für unterschiedliche Positionen entweder eine Positionsliste oder B
  separate Launches. B separate Launches sind im MVP akzeptabel; die teuren GEMMs bleiben gebatcht.
- **GDN:** `k_gdn_conv_T` und `k_gdn_delta_T` sind heute sequenziell über T Token **einer** Sequenz.
  Für B Sequenzen braucht es `k_gdn_conv_slot`/`k_gdn_delta_slot` mit State-Pointer pro Slot, oder
  im MVP B einzelne Launches mit T=1. Die GDN-Kosten sind klein gegenüber den GEMMs; B Launches sind
  ein vertretbarer erster Schritt.
- **Attention:** `k_att_flash16_T` liest eine K/V-Basis und `pos`. Im MVP entweder B separate
  Launches mit dem jeweiligen Slot-Pointer oder ein Kernel mit Slot-Dimension. Die Attention ist
  nicht der Weight-Engpass, aber bei langem Kontext relevant; ein Slot-Kernel ist Phase 2.
- **Embedding:** `k_deq_emb3` pro Token; B Launches sind unkritisch.
- **Head:** Der exakte Head ist ein großes GEMV (`gemv`, `k_g4` in `engine.py`), nicht der
  Chunk-GEMM. Für B Zeilen braucht es einen Zeilenkopf (`gemv_rows`/`k_g4r`) oder einen neuen
  `k_head_batch`; `k_argmax` braucht eine Zeilenversion (`k_argmax_rows`, im MTP-Code gibt es
  bereits `k_argmax_T` als Vorlage). Der Fast Head bleibt im MVP aus, weil er pro Sequenz eine
  andere Cluster-Shortlist hat.
- **5-Trit/Plan:** `k_mm_g5n` und die TC-Pläne sind eine separate Engine. Nicht im MVP.

Wichtig: Die Zeilen eines Batch-GEMM müssen **unabhängig** sein; bei Attention und GDN darf keine
Zeile den State einer anderen lesen. Die bestehenden Kernel sind genau dann korrekt, wenn jede Zeile
ihren eigenen Slot-/Positionsbezug bekommt. Das ist die zentrale Testachse.

### 4.5 Scheduler

Ein `BatchScheduler` läuft auf dem GPU-Thread und besitzt alle Slots. Die HTTP-Threads rendern und
tokenisieren nur; sie fassen die Engine nicht an.

```text
loop:
  1. Cancels und fertige Sequenzen einsammeln, Slots freigeben.
  2. Wartende Sequenzen admissionieren, solange freie Slots existieren.
  3. Wenn eine Sequenz im Prefill ist: einen Chunk (z. B. 64–256 Token) lesen.
     - Prompt fertig: letztes Token + Head, ersten Token sampeln, in Decode wechseln.
  4. Alle aktiven Decode-Sequenzen (bis max_batch) in einem BatchEngine.decode(...) verarbeiten.
  5. Pro Sequenz sampeln, Token anhängen, Stop/EOS/max_new prüfen, Text in die Ausgabe-Queue legen.
  6. Wiederholen; wenn nichts zu tun ist, auf neue Arbeit warten.
```

Eigenschaften:

- **Continuous:** Admission und Entlassung passieren zwischen zwei Decode-Schritten, nicht erst am
  Ende eines Batch-Fensters.
- **Chunked Prefill:** Ein langer Prompt blockiert den Decode nicht für seine gesamte Länge; pro Tick
  wird nur ein begrenzter Prefill-Chunk gelesen. Im MVP ist Prefill serialisiert (eine Sequenz),
  Decode gebatcht; gemischter Prefill+Decode-Batch ist Phase 2.
- **Fairness:** Round-Robin über wartende Sequenzen; Prefill bekommt ein Budget pro Tick; Decode hat
  Vorrang, damit laufende Streams nicht verhungern.
- **Backpressure:** `max_queue`; bei Überlauf HTTP 503 statt unbegrenztem Wachstum.
- **Cancellation:** `cancel`-Event pro Sequenz; der Scheduler prüft es an jeder Tick-Grenze, gibt den
  Slot frei und sendet kein `done`.
- **Kein `K.lock` mehr im Request-Pfad:** Der Scheduler ist der einzige Mutator; die HTTP-Schicht
  konsumiert nur die per-Sequenz-Queue.

### 4.6 Server-Integration

- `Kindi` bekommt einen `BatchScheduler`, wenn `KINDI_BATCHING=1` und der Variant unterstützt wird.
- `POST /v1/chat/completions` erzeugt eine `Sequence`, ruft `submit()` und liest die Ausgabe-Queue.
- Streaming: pro Sequenz eine `asyncio.Queue`/`queue.Queue`; der bestehende SSE-Relay wird auf die
  Sequenz-Queue umgestellt. Non-Streaming sammelt bis `done`.
- Der bestehende Single-Pfad bleibt als Fallback und für `KINDI_BATCHING=0` erhalten.
- `/health` erweitern: `active`, `waiting`, `slots_free`, `batch_size`, `queue_depth`, `kv_used`,
  `prefill_tokens`, `decode_tokens`.
- `usage`/`timing` pro Request bleiben; `cached_tokens` kommt in Phase 2 dazu.
- Die OpenAI-kompatible API, Tool-Calls, Thinking-Budget und Stop-Sequenzen bleiben unverändert;
  die Logik wandert pro Sequenz in den Scheduler.

### 4.7 Prompt-Cache (Phase 2)

Der bestehende `Runner` ist bewusst auf eine aktive Konversation ausgelegt. Im MVP wird der
Cross-Request-Cache **deaktiviert** (jede neue Sequenz liest ihren Prompt ab Position 0). Das ist
eine bewusste Regression für Agenten-Workloads, aber es hält den ersten Batch-Pfad testbar.

Phase 2:

- `SlotCache` pro Slot: `tokens`, `ckpt`, `kv_used`; beim Admission LCP mit dem neuen Prompt.
- Ein Slot mit passendem Prefix wird wiederverwendet; sonst `reset_slot`.
- Checkpoints/`KINDI_CKPT_VRAM` pro Slot; Host-Budget; keine Host-Kopie während eines aktiven
  Decode-Batches.
- Parken/Öffnen aus `Runner` später; zuerst nur der In-Memory-Prefix pro Slot.
- Ein gemeinsamer Prefix-Pool/Paged KV ist Phase 3, nicht MVP.

### 4.8 Plan-Engine und Speculative Decoding (Phase 3)

Nicht Teil des MVP. Beide haben eigene, feste Ein-Sequenz-Strukturen:

- `TPEngine` hält K/V pro Karte und `ctx`; die `Plan`-Replays und CUDA-Graphen sind auf eine Sequenz
  und feste `T` ausgelegt.
- `spec.py` hat einen globalen Draft-/Verify-Zustand; pro-Sequenz-Spec würde pro Slot eigene
  Draft-Historien, Guards und Graph-Pläne brauchen.
- Die 2B-Plan-Engine ist nur mit T=4 token-exakt, die 9B-Plan-Engine stürzt ab. Diese Baustellen
  müssen zuerst separat gelöst werden, bevor sie mit Batching kombiniert werden.

## 5. Umsetzungsplan

### Phase 0 — Spike und Go/No-Go (4–6 Personentage)

Ziel: Beweisen, dass ein Batch-GEMM über unabhängige Sequenzen auf dieser Hardware überhaupt einen
Durchsatzgewinn bringt, bevor Server- oder Cache-Code angefasst wird.

- Mikrobenchmark: ein Transformer-Block (QKV, FFN, Head) mit `B=1,2,4` unabhängigen Zeilen über
  `k_mm_g4k`; Gewichtsread einmal; X/Y mit Zeilenstrides.
- 2B bei ctx 4096, 9B bei ctx 4096; Zeit pro Batch, aggregierte Token/s, GPU-Speicher.
- Prüfen: sind die Zeilen unabhängig und token-exakt gegenüber B=1?
- Vergleich mit dem heutigen Einzel-Decode (2B 191,5; 9B 86,5 tok/s).
- **Gate:** Bei B=4 mindestens ~1,5× aggregierter Durchsatz und keine Korrektheitsabweichung.
  Sonst stoppen oder nur den 2B weiterverfolgen.

Deliverable: Messprotokoll, Entscheidung, keine Produktionsänderung.

**Ergebnis (2026-11-08, RTX 4060 Ti, 2B):** Der Spike läuft (`bench-logs/spike_batch_gemm.py`);
B=1/2/4 sind zeilen-exakt (max. Differenz 0,0; Head-Argmax 1/1, 2/2, 4/4), und die GEMM-Zeilen
amortisieren sich (3,90× relativ zu B=1). Absolut erreicht B=4 aber nur 131,8 tok/s = 0,69× des
Einzel-Decodes (191,5) und damit das 1,5×-Gate nicht; der Chunk-GEMM-Pfad ist bei kleinem T zu
langsam. **No-Go für den 4-Slot-MVP auf diesem Pfad.** Details:
`bench-logs/phase0-spike/RESULTS.md`.

### Phase 1 — Slot-Engine (7–10 Personentage)

Der konkrete Code-Plan (betroffene Funktionen, Signaturen, Speicher-Fit, Tests) steht in
[`PER-SLOT-REFACTOR.md`](PER-SLOT-REFACTOR.md).

- `BatchEngine`/`SlotPool` mit `nslots`, `reset_slot`, `free_slot`.
- `Kc/Vc` mit Slot-Dimension; `M/conv` pro Slot; Workspace bleibt geteilt.
- Speicher-Fit: Slots aus freiem VRAM bestimmen; `KINDI_BATCH_SLOTS` als Override.
- Prefill über den bestehenden `chunk1`-Pfad, aber mit Slot-Offset; Decode zunächst B=1.
- Golden-Tests: B=1 über den neuen Pfad muss exakt dieselben Token liefern wie die alte `Engine`.
- `KINDI_BATCHING=0` und der Single-Pfad bleiben unverändert.

Done when: 2B und 9B laden mit Slots, eine Sequenz pro Slot kann lesen/generieren, alte Tests grün.

### Phase 2 — Batch-Decode-Kernel (9–14 Personentage)

- Batch-GEMM für alle linearen Produkte; X/Y-Zeilen; Gewicht einmal lesen.
- GDN-/Conv-/Attention-Slot-Kernel oder B Einzel-Launches als Zwischenschritt.
- `k_argmax_rows`/Head-Batch; exakter Head.
- Pro-Sequenz-Sampling: Logits-Zeile, Temperatur, Top-P/Top-K, Stop, EOS.
- Korrektheitstests: B=1/2/4 gegen serielle Einzelausführung; dieselben Prompts.
- Profiling: Wo geht die Zeit verloren (GEMM, Attention, Head, Sampling)? Tile-Größen tunen.

Done when: vier unabhängige 2B-Sequenzen laufen in einem Batch, die Ausgabe ist für die Testprompts
token-gleich, und der aggregierte Durchsatz liegt über dem Single-Pfad.

### Phase 3 — Scheduler und Server (6–8 Personentage)

- `BatchScheduler` mit Admission, Prefill-Chunk, Decode-Batch, Finish, Cancel, Backpressure.
- `Kindi.submit()`; Streaming- und Non-Streaming-Pfad; per-Sequenz-Queues.
- `/health`-Metriken; Fehlerbehandlung; Timeouts; 400/503.
- Concurrency-Tests: 2/4 Streams, Abbruch, Stop, max_tokens, Kontextgrenze.
- Feature-Flag; Single-Pfad als Fallback.

Done when: vier gleichzeitige SSE-Streams auf dem 2B sauber laufen und `KINDI_BATCHING=0` unverändert
ist.

### Phase 4 — Cache und 9B-Profil (7–10 Personentage)

- `SlotCache` mit LCP, Per-Slot-Prefix, Checkpoints, Host-Budget.
- 9B-Speicherprofil: 2–4 Slots bei 4096; OOM-Fallback; Desktop-Belegung.
- Agenten-Test: mehrere Turns, Prefix-Reuse, keine Korruption.
- Benchmark: 2B/9B, B=1/2/4, TTFT, Gesamtdurchsatz, p95.

### Phase 5 — Härtung und Doku (5–7 Personentage)

- `pytest`, `kindi verify`, `bench`, `bash -n`; neue Batch-Tests.
- Doku: `docs/CONFIGURATION.md`, `docs/CARDS.md`, `README.md` (Limits), `AGENTS.md`.
- Release Notes; `KINDI_BATCHING`-Default; Rollback.
- Windows/WSL-Shim und Linux/Docker prüfen; keine Regression im Single-Pfad.

### Zeitplan

| Phase | Personentage | Meilenstein |
|---|---:|---|
| 0 Spike | 4–6 | Go/No-Go |
| 1 Slot-Engine | 7–10 | B=1 exakt, Slots laden |
| 2 Batch-Kernel | 9–14 | B=4 korrekt und schneller |
| 3 Scheduler/Server | 6–8 | 4 Streams, Cancel, 503 |
| 4 Cache/9B | 7–10 | Prefix-Reuse, 9B stabil |
| 5 Härtung | 5–7 | Release-fähig |
| **MVP (0–3)** | **26–38** | **5–8 Wochen** |
| **Production v1 (0–5)** | **38–55** | **8–11 Wochen** |
| Plan/Spec zusätzlich | +25–35 | 13–18 Wochen gesamt |

Voraussetzungen: ein Senior-Engineer mit CUDA/CuPy- und Repo-Erfahrung, die 8-GB-Karte zum Testen,
kein GPU-CI. Ohne diese Erfahrung oder mit paralleler 9B-/27B-Arbeit eher +30–50 %.

## 6. Abnahmekriterien

### Korrektheit

- `kindi verify` bleibt PASS (Single-Pfad).
- Batch B=1 ist token-gleich zum alten Single-Pfad.
- Batch B=2/4 ist für die vier Verify-Prompts token-gleich zur seriellen Ausführung. Falls die
  Batch-Arithmetik bei Gleichstand abweicht, wird das dokumentiert und **nicht** als Default
  freigegeben; die Exaktheitsgarantie des Projekts darf nicht stillschweigend fallen.
- Stop, EOS, max_tokens, Thinking-Budget, Tool-Calls und Sampling funktionieren pro Sequenz.

### Durchsatz und Latenz

- 2B, ctx 4096, 4 gleiche 512-Token-Prompts, 128 neue Token: aggregierter Decode-Durchsatz
  mindestens 1,5× Single (Ziel; Messung entscheidet).
- 9B, ctx 4096, 2 Slots: mindestens 1,1× aggregiert; sonst wird das 9B-Batching nicht als Gewinn
  ausgeliefert.
- TTFT p95 und Queue-Zeit werden gemessen; ein langer Prefill darf laufende Decodes nicht
  unbounded blockieren (Chunked Prefill).
- Kein OOM bei 2B/4 Slots und 9B/2 Slots auf der 8-GB-Karte; sauberer Fallback auf weniger Slots.

### Robustheit

- Vier gleichzeitige SSE-Streams, Abbruch mitten im Stream, Stop-Sequenz, Kontextgrenze (400),
  Queue-Overflow (503), Server-Shutdown.
- `KINDI_BATCHING=0` verhält sich unverändert zum heutigen Single-Pfad.
- `python -m pytest tests/ -q` und die bestehenden No-GPU-Gates bleiben grün.

## 7. Risiken und Gegenmaßnahmen

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| Batch-GEMM bringt keinen Gewinn (Bandbreite/Tile) | Ganzer Ansatz lohnt nicht | Phase-0-Spike, hartes Go/No-Go |
| Token-Exaktheit bricht durch andere Reduktionsreihenfolge | Vertrauensverlust, Verify rot | Goldens, B=1-Parität, notfalls Batch nur opt-in |
| GDN/Attention pro Slot fehlerhaft | Falsche Ausgaben, State-Leak | Per-Slot-Kernel, Isolationstests, B=1-Äquivalenz |
| 8-GB-Speicher reicht nicht für 4 Slots | OOM/Weniger Kontext | Slots automatisch begrenzen, 9B mit 2 Slots starten |
| Cache-Refactor korrumpiert Prefixe | Agenten falsche Antworten | MVP ohne Cross-Request-Cache; Phase 2 mit eigenen Tests |
| Scheduler-Cancel/Streaming-Leak | Hängende Requests, Slots belegt | Cancellation an Tick-Grenzen, Queue- und Slot-Invarianten |
| Plan-/Spec-Engine wird mit reingezogen | Unkontrollierbare Komplexität | Explizit außerhalb des MVP; separate Phase 3 |
| Gemma/Nanbeige/CPU fehlen | Feature-Lücke | Bewusst nur Bonsai/Plain/1 GPU; klar dokumentieren |
| Kein GPU-CI | Regressionen erst auf Hardware | CPU-Tests, Goldens, nächtlicher 8-GB-Lauf |
| Zwei Engines divergieren | Wartungsaufwand | `BatchEngine` als Adapter, Single-Pfad nicht ersetzen |

> **Nachtrag 2026-11-08:** Die MVP-Empfehlung unten ist durch das Phase-0-Ergebnis überholt. Phase 1 ist nicht
autorisiert; erst ein neuer Spike mit bestandenem Gate öffnet sie wieder.

## 8. Entscheidung und Empfehlung

1. **Phase 0 zuerst.** Ohne messbaren Batch-GEMM-Gewinn ist der Rest nicht gerechtfertigt.
2. **MVP nur für Bonsai Plain, eine GPU, 2B zuerst.** Der 2B hat den größten Bandbreitenkopfraum;
   der 9B ist nahe am Limit und wird nach dem Spike separat bewertet.
3. **Feste Slots, kein Paged KV, kein Prefix-Sharing im MVP.** Das hält die Kernel- und
   Cache-Komplexität beherrschbar; der Preis ist ein voller Kontext pro Slot und kein
   Cross-Request-Prefix.
4. **Kein Speculative Decoding und keine Plan-Engine im MVP.** Die 2B-T=8-Divergenz und der
   9B-Graph-Crash sind eigene Baustellen.
5. **Single-Pfad bleibt Default**, `KINDI_BATCHING=1` ist opt-in, bis die Abnahmekriterien erfüllt
   sind.

Für einen einzelnen Stream bleibt Speculative Decoding/TC der bessere Hebel (2B Plan T=4: 347 tok/s
gegen 191,5 Plain). Continuous Batching ist der Hebel für **mehrere gleichzeitige** Streams; der
2B ist dafür der richtige erste Kandidat, der 9B nur mit gedämpften Erwartungen.

## 9. Offene Fragen an den Betreiber

- Wie viele gleichzeitige Clients/Agenten sollen unterstützt werden (2, 4, 8)?
- Sind 2.048 oder 4.096 Token Kontext pro Slot akzeptabel, oder muss der volle Kontext pro Request
  erhalten bleiben?
- Ist Cross-Request-Prefix-Cache für den Start Pflicht (Agenten) oder erst Phase 2?
- Muss die Token-Exaktheit auch im Batch gelten, oder sind seltene Abweichungen bei Gleichstand
  akzeptabel?
- Ist der 9B für den Ziel-Workload wichtiger als der 2B, obwohl der Bandbreitenkopfraum dort klein
  ist?
