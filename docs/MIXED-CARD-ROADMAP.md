# Mixed-card pairing and asymmetric tensor parallel: corrected roadmap

Status: **proposal plus implemented engine path, 2026-11-12. The opt-in `KINDI_TP_SPLIT=auto`
layout is in `kindi/tp.py`; `equal` remains the default. There is no mixed-card GPU record and
no `mixedx2` installer profile. Current mixed-card offers remain experimental, not a support
claim.**

This page reviews the interrupted WIP's pairing and asymmetric-TP design against the current
tree. It replaces the mixed-architecture section of the WIP's
`docs/PAIR-SETUPS.md` as the working plan. That file exists only on the WIP branch
(`wip/main-interrupted-snapshot-20261109-045352`), never in this tree, so there is nothing here to
open. The WIP snapshot and backup remain evidence only.

## 1. What the WIP proposed

The WIP branch `wip/main-interrupted-snapshot-20261109-045352` contains:

- `docs/PAIR-SETUPS.md`: a 235-line pairing document. Its later section claims that
  “Starting with Kindi 0.4.0, mixed-architecture GPU pairs are supported”, with
  `KINDI_TP_SPLIT=auto`, per-card tensor-core backends, per-card memory reserves and a
  `mixedx2` profile.
- `kindi/tp.py`: a 223-line change with a `KINDI_TP_SPLIT` parser and an “asymmetric” branch.
- `kindi/spec.py`: a 30-line change adding per-card `KINDI_CARD_<n>_TC` overrides and removing
  the Turing two-card guard.
- `install.sh`: **zero bytes**. The WIP has no installer implementation to provide the
  `mixedx2` profile or any of the promised per-card variables.

The proposal is therefore internally inconsistent: the document claims installer-supported
mixed architecture, but the WIP installer is empty, and the engine change does not implement a
complete asymmetric layout.

## 2. What the current tree actually implements

Current `install.sh` remains the only mixed-card installer routing. The engine now also has
the opt-in layout in section 4.5; the table below describes the pre-layout baseline and the
state of the other paths.

| area | current behaviour | consequence |
|---|---|---|
| `pair_ok()` | any two cards from `maxwell`, `pascal8`, `turing`, `tc` or `p100`, each at least 3,800 MiB; Kepler, 2 GB and unsupported cards excluded | mixed-generation offers exist: P106+P4000, M4000+P4000, RTX 4000+4060 Ti and others; no total-memory or per-card shard check |
| `profile_of()` | two sm_80+ cards get `tc2`; all other pairs get `k1200x2` | mixed pairs use the plain two-card engine; there is no `mixedx2` profile |
| mixed weight path | a pre-6.0 + 6.0-or-newer pair adds `KINDI_G5=0` | both shards use the 2-bit path. `kindi/tp.py` reads this global override in each `Shard`; the path is real, but its token exactness on mixed cards is unmeasured |
| `kindi/tp.py` | **equal** is the default two-card split: `NKg`, `NVg`, `NHKVg` and `NHg` divided by `W`; FFN rows in whole Hadamard blocks; FFN columns half; output head half. Opt-in `KINDI_TP_SPLIT=auto` (section 4.5) uses free-VRAM shares and complete per-component ranges | equal remains the compatible path. Auto gives the larger card more FFN blocks and output rows; on the 27B the GDN/attention ranges stay equal because of HB alignment. No mixed-card run exists |
| `kindi/spec.py` | `KINDI_TC=1` installs one global tensor-core path; `_tc_backend(card)` can select `tc` or `tc_turing` per card, but the Turing backend rejects `W=2` | mixed `tc+turing` cannot use tensor cores on both cards; the installer therefore leaves such pairs on the plain path |
| tests | `tests/test_installer_cards.py` has 19 fake-table cases | routing is tested; no mixed-card engine, verify or server run exists |

The current `KINDI_G5=0` fallback is not a substitute for asymmetric TP. It makes both shards use
the same (2-bit) weight path; it does not give the larger card more layers, heads or FFN rows.
The current `tc2` profile selects tensor cores only when both cards are sm_80 or newer. A mixed
`tc+turing` pair is deliberately kept on the plain path.

## 3. Review findings

### 3.1 Pairing

1. **The WIP's “supported from 0.4.0” claim is false.** The WIP installer is empty; the current
   installer is a separate mixed-generation routing change, and no mixed-card GPU run is
   recorded.
2. **The WIP rule and the current rule disagree on cross-weight pairs.** The WIP requires both
   cards to use the same 5-trit or 2-bit path and rejects P4000+M4000. The current installer
   accepts that pair and sets `KINDI_G5=0` for both cards. The current fallback is coherent in
   code, but it changes the newer card's arithmetic path and has no token-exact measurement.
3. **The current broad rule is not yet a feasibility rule.** `pair_ok()` checks 3.8 GiB per card,
   not whether the selected model, its per-card shard, the K/V cache and the exchange buffers fit
   that card at the advertised context. A 27B offer can still fail at load or be silently reduced.
4. **The WIP's profiles are not the current profiles.** `pascal8x2`, `turingx2`, `pair` and
   `mixedx2` in the WIP document do not exist in current `profile_of()`. Current offers are
   `k1200x2` and `tc2` only.
5. **The WIP's 33B/24 GB claim is not a current offer.** Current mixed pairs offer 27B and
   Nemotron; 33B remains a separate Nanbeige host-layer path, not a Bonsai pair.

### 3.2 Asymmetric tensor parallel (findings about the interrupted WIP)

These findings describe the WIP snapshot, not the implemented layout in section 4.5.

1. **The WIP does not implement asymmetric model partitions.** Its `tp.py` branch changes
   `ffn_rows`, `ffn_cols` and `head_split` proportionally. It leaves `NKg`, `NVg`, `NHKVg` and
   `NHg` at the equal split and its own comment says full asymmetric TP “requires kernel
   modifications”.
2. **The exchange and plan shapes are not resized.** The plans and the mapped exchange buffers
   assume the current equal split. A larger card given more FFN rows does not get a matching
   exchange contract, so the result is not a valid two-card plan.
3. **Per-card tensor cores are only an idea.** The WIP adds env overrides and removes the
   `tc_turing` `W=2` guard, but supplies no test, no exchange proof and no installer variables.
   The current guard exists for a reason: the Turing two-card path was not verified.
4. **`KINDI_TP_SPLIT=auto` is VRAM-total based, not layout based.** Weights, K/V, activations and
   exchange buffers do not scale linearly with total VRAM. A ratio derived from capacity alone
   can produce an invalid or unbalanced shard.
5. **The WIP has no acceptance record.** `install.sh` is empty, the split changes have no tests,
   and the only claimed evidence is prose.

## 4. Corrected design

Keep the layers separate. Do not merge the WIP branch into `install.sh`, `tp.py` or `spec.py`.

### 4.1 Installer routing

The current `pair_ok()` may continue to expose candidate pairs, but an offer is **experimental**
until its runtime gate passes. The installer must report the selected pair mode and the weight
path in the setup description. It must not imply “24 GB combined” means a model will fit.

The first installer correction should be a feasibility check, not an asymmetric split:

- compute the actual per-card weights and K/V for the selected model and requested context;
- keep a per-card reserve for the exchange, CUDA context and allocator;
- reject a pair/model/context combination when either card cannot hold its share;
- keep `KINDI_G5=0` only as a same-weight-path fallback, and add a test that both shard
  environments receive it; and
- do not add `KINDI_TP_SPLIT` to `config.py` until the engine implements a complete layout.

### 4.2 Equal-split mixed validation

This is the next real milestone. It uses the existing `TPEngine` and the current
`KINDI_G5=0` fallback, with no asymmetric claims:

| pair | path under test | first hardware gate |
|---|---|---|
| P106-100 + P4000 | plain 5-trit, `k1200x2` | 2B/4B/9B/27B verify where memory permits |
| M4000 + P4000 | plain 2-bit, `k1200x2`, `KINDI_G5=0` | 2B/4B/9B/27B verify; confirm both shards load 2-bit |
| RTX 4000 + RTX 4060 Ti | plain 5-trit, no TC | 2B/4B/9B/27B verify; compare with each card alone |
| RTX 4060 Ti + RTX 5060 Ti | `tc2` (both sm_80+) | 27B two-card verify first, then server/cache |

A pair passes only when the model is token-exact against the same llama.cpp reference, both cards
participate in the exchange, `/health` reports the fitted context, and the server restart/cache
check is clean. This phase may still use equal work per card; it is not asymmetric TP.

### 4.3 Per-card tensor cores

Only after the equal-split mixed run:

- test the Turing `tc_turing` backend at `W=2` on two RTX 4000s, then mixed with an sm_80 card;
- keep the current `w_ok` guard until that run is clean;
- decide whether the installer may set `KINDI_TC=1` for a mixed pair, or whether `_tc_backend`
  per card is enough without new public per-card variables;
- do not add `KINDI_CARD_<n>_TC` until the engine and installer both use it and tests cover the
  fallback.

### 4.4 Asymmetric layout

This is a new design, not a WIP merge. The unit is a complete **per-card layout**, not a single
ratio:

- GDN groups, attention query/KV heads, convolution/state channels and output columns;
- FFN rows in whole Hadamard blocks and FFN columns in aligned units;
- output-head rows and vocabulary exchange;
- K/V cache slices and exchange buffers;
- per-card weight path (`g5`/2-bit) and tensor-core backend; and
- the context that fits the smaller shard after all of the above.

The design must state the alignment constraints and the exactness argument before code. The
existing equal-split path remains the reference. A valid result must be token-exact against the
full-GPU reference and the equal-split pair where both fit, and must record peak VRAM.

Only then may a `mixedx2` profile be added. `KINDI_TP_SPLIT=auto` is already available as an
engine opt-in; it does not create an installer offer or a support claim. The public name should
describe the mode, not claim a general “mixed architecture” support.

### 4.5 Implemented layout (2026-11-12; unverified)

`kindi/tp.py` now has the complete per-card layout described above. `KINDI_TP_SPLIT=auto`
(opt-in; `equal` remains the default) computes the first card's share from free VRAM after a
512 MiB reserve, rounds every component to its alignment unit, and uses `equal` when the usable
shares are within 10 percentage points of each other. `KINDI_TP_SPLIT=equal` keeps the
historical path. The layout covers:

- GDN q/k groups with their tiled v heads (`NV * HK` is HB-aligned);
- attention q and KV heads (`NH * HD` is HB-aligned);
- FFN gate/up rows and `ffn_down` columns, both on the same whole-HB interval; and
- output-head rows on 16-row units, with the vocabulary exchange and argmax offsets.

The per-card K/V caches and exchange buffers follow these head ranges; no separate ratio is used.

On the 27B (HB 1024, 16 GDN key groups with 3 value heads each, 4 KV heads) the alignment
unit forces GDN 8/8 (24/24 value heads) and attention 12/12; at a 2:1 free ratio only the FFN
(11/6 whole blocks) and the output head move. A geometry with two value heads per key group
exercises the GDN 12/4 case in the CPU-only test. This is why the split is per component, not one global ratio.

The layout is part of the `g5` cache key and the saved-opening engine fingerprint. A CPU-only
test covers the 27B geometry, the Shard slice offsets, the invalid-alignment refusal and the
fingerprint. There is still **no mixed-card GPU run**, so this is an implemented engine path,
not a support claim: token-exact comparison against the full-GPU and equal-split references and
a peak-VRAM record remain the P4 gates. The installer does not add a `mixedx2` profile, and
the automatic path is opt-in because `equal` remains the default.

## 5. Phases and gates

| phase | work | gate | status |
|---|---|---|---|
| P0 | freeze claims; keep mixed offers experimental; link this proposal | current 19 installer tests pass; no mixed-card support claim | **done 2026-11-11** (proposal only) |
| P1 | per-card feasibility and `KINDI_G5=0` test | fake-table and plan tests reject impossible offers; both shard paths checked | pending |
| P2 | equal-split mixed hardware validation | 27B verify plus server/cache on at least one mixed pair | pending |
| P3 | Turing/TC mixed path | `tc_turing` `W=2` verify, or an explicit plain-path decision | pending |
| P4 | asymmetric layout design and implementation | token-exact equal/full-GPU comparison; peak VRAM | **partial 2026-11-12**: complete layout, `KINDI_TP_SPLIT=auto|equal`, aligned fallbacks, cache/fingerprint coverage and CPU-only tests; no mixed-card run |
| P5 | installer/config `KINDI_TP_SPLIT` + `mixedx2` | generated config docs and fake-table tests | **partial 2026-11-12**: environment variable and generated docs; no `mixedx2` profile or installer fake-table change |
| P6 | performance, 24K+ and 33B/Nanbeige separation | measured, not a support claim until all gates pass | pending |

The order matters. P2 gives a usable mixed-card result without inventing a new split. P4 is a
different engine plan, not a flag on the existing equal split.

## 6. Acceptance matrix

| claim | required evidence | current |
|---|---|---|
| mixed pair is offered | 19 installer routing tests | **yes** |
| mixed pair is supported | two-card verify, server, cache, measured peak | **no** |
| `KINDI_G5=0` fallback is exact | both shards load 2-bit; token-exact verify on the pair | **not measured** |
| two sm_80 cards use `tc2` | two-card TC verify, no exchange race | **not measured** |
| mixed `tc+turing` uses both tensor-core backends | Turing `W=2` verify and exactness | **blocked by current guard** |
| asymmetric split is exact | complete per-card layout, exactness and peak-VRAM record | **layout implemented; no exactness/peak record** |
| 33B runs on a mixed 24 GB pair | measured Nanbeige or pair result | **not offered** |

## 7. Risks and non-goals

- **Do not merge the interrupted WIP `tp.py` or `spec.py`.** The implemented layout in section
  4.5 is a separate change; the WIP snapshot's layout, exchange, per-card backend and tests are
  incomplete.
- **Do not merge the WIP `PAIR-SETUPS` claims.** Its installer is empty and its
  “0.4.0 supported” statement is contradicted by the current tree.
- **Do not use total VRAM as a fit proof.** Equal split means the smaller card's share still has
  to hold its weights, K/V and exchange.
- **Do not describe `KINDI_G5=0` as asymmetric.** It is a global weight-path fallback.
- **Do not advertise 33B on a mixed pair.** The current mixed offer is 27B/Nemotron; 33B is a
  separate host-layer/Nanbeige target.
- **Keep the support contract.** An installer option is not support; [CARD-SUPPORT.md](CARD-SUPPORT.md#what-supported-means-here)
  still requires a real dual-card verify and a measured record.

## 8. Evidence and current references

- WIP commit: `wip/main-interrupted-snapshot-20261109-045352` (`81cf295`).
- WIP `install.sh` size: **0 bytes**.
- WIP `docs/PAIR-SETUPS.md`: 235 lines; see its “Asymmetric Multi-GPU Tensor Parallel” section.
- WIP `kindi/tp.py`: `_parse_tp_split`, `_apply_split_ratio` and the FFN/head-only branch; its
  comment says the remaining heads need kernel changes.
- WIP `kindi/spec.py`: `KINDI_CARD_<n>_TC` override and removal of the Turing `W=2` guard.
- Current `install.sh`: `pair_ok()` and `profile_of()`; current tests: `python -m pytest -q
  tests/test_installer_cards.py` -> 19 passed.
- Current `kindi/tp.py`: equal split by default, opt-in free-VRAM layout in section 4.5, and
  per-shard `KINDI_G5` handling.
- Current `kindi/spec.py`: `_tc_backend(card)` and the `tc_turing` `W=2` guard.
- Current status: [ROADMAP.md](ROADMAP.md), [CARD-SUPPORT.md](CARD-SUPPORT.md), [P106.md](P106.md),
  and [TANDEM-RUNTIME.md](TANDEM-RUNTIME.md).

This page now records an implemented engine path, but adds no installer option, `mixedx2`
profile or support claim. The mixed-card GPU gates in section 5 remain open.
