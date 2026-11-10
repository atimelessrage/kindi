# Long-term card support: single and dual GPUs

Status: **P0 implemented 2026-11-10; not a support claim yet.** `install.sh` now classifies every sm_80 card
as `tc` from its compute capability, pairs two cards of the same class with at least 3,800 MiB each, and gives
two sm_80 cards the `tc2` profile. `tests/test_installer_cards.py` runs the installer's own card functions
against a fake nvidia-smi table. The first GPU verify is recorded on the RTX 4060 Ti: the 2B tc plan is 4/4
token-exact, the 4B tc plan is 2/4, the 9B tc plan is refused (Hadamard 4096), and the plain 4B/9B paths are
4/4. The dual, 27B/33B and 5060 Ti rows remain open; P1-P3 are unimplemented.

## What “supported” means here

A card is supported when all of these hold:

1. The installer detects it and offers the advertised single-card or dual-card setup.
2. The advertised model loads, answers a chat request and `/health` reports the fitted context.
3. `kindi verify` passes with the documented reference (token-exact for Bonsai; the documented agreement
   rule for Nanbeige/Gemma).
4. A dual setup passes the same check with both cards visible and the plan engine using both.
5. The setup is in `docs/CARDS.md` with its settings and a measured result.

This is stricter than “the kernels compile”. A card that runs only after manual environment overrides is
not supported.

Kindi consumes its own Bonsai PTQ1_0 packs and Nanbeige GGUF. The NVFP4/FP8/AWQ/GPTQ/EXL2 boxes in a
generic GPU table are hardware capabilities, not Kindi formats; they do not imply support here. The
5060 Ti’s native FP4 is not used by Kindi today.

## Current code reality

| gate | current behaviour | consequence for this goal |
|---|---|---|
| install.sh `card_class()` | `tc` for every compute capability 8.0 and later; `turing` for sm_75 with tensor cores; `pascal8` for 61–79 at 7,500 MiB or more; `p100` for sm_60 at 12,000 MiB or more; below 50 is `kepler`; everything else is `maxwell` | **The RTX 4060 Ti 8 GB is now `tc`**, so the tensor-core profile is offered; `profile_of()` picks a 24,576-token context for an 8 GB tc card instead of the 131,072-token 10 GB+ default. |
| install.sh `pair_ok()` | two cards of the same class (maxwell, pascal8, turing, tc or p100) with at least 3,800 MiB each; 2 GB cards and mixed classes stay out | **Dual P106, P4000, RTX 4000, 4060 Ti and 5060 Ti setups are now offered.** The 2 GB GT 1030 pair and a 4060 Ti + RTX 4000 mix are rejected. |
| install.sh `profile_of()` dual branch | two sm_80+ cards get `tc2`: `KINDI_CARDS=2`, `KINDI_TC=1`, `KINDI_FD=1`, `KINDI_ATT_TC=1`, `KINDI_TC_TMAX=4`, a 512 MiB exchange reserve, and a context from the smaller card (24,576 below 9.5 GB, 65,536 above); speculative decoding is off | Two Ada/Blackwell cards now use the tensor-core plan path. Spec remains off until the P1 two-card verify passes. Legacy pairs keep `k1200x2`. |
| install.sh `card_models()` | `pascal8` offers 27B; `maxwell` offers 9B/Nemotron/4B/2B; `tc` offers 27B/Nemotron/9B/4B/2B | A 6 GB P106 is offered up to 9B alone and 27B as a pair. A 2 GB GT 1030 is offered 4B/2B and does not pair. |
| `kindi/tc.py` | compiles for compute_75 PTX, raises the PTX header to 7.1/sm_8x, lets the driver JIT; `supported()` accepts major >= 8 or major == 6 | Ada and Blackwell depend on this PTX patch; it is exactly the fragile path that needs a real test on the 4060 Ti and 5060 Ti. |
| `kindi/engine.py` | `_cupy_arch_fix()` caps CuPy 9.6’s NVRTC target to the toolkit maximum | Necessary for Blackwell with the CUDA 10.2 image, but it means the image is not a native sm_89/sm_120 toolchain. |
| `kindi/tp.py` | two-card plan engine; 27B path requires Hadamard blocks of 1024 | The 27B dual target is viable; the 2B tc plan is exact, the 4B tc plan is 2/4, and the 9B is refused (Hadamard 4096). |
| docs/CARDS.md | “All the cards use the same Docker image (CUDA 10.2, CuPy 9.6)” | A 4060 Ti/5060 Ti support claim needs a modern-toolkit image and a documented exception, or a verified PTX-JIT path. |

## Target matrix

“Dual” means two cards of the same class, both visible to one process; it is not SLI. Model entries are
targets pending verification, not current claims.

| Card | Arch / CC | Current installer class | Single target | Dual target | Required path |
|---|---|---|---|---|---|
| GT 1030 2 GB | Pascal 6.1 | `maxwell` | 2B; 4B only if the fitted context is acceptable | not useful | plain engine; 2 GB context/memory test |
| P106-100 6 GB | Pascal 6.1 | `maxwell` | 9B/4B/2B | 27B (12 GB total) now offered | plain plan engine, no TC; P106 has no display outputs |
| Quadro M4000 8 GB | Maxwell 5.2 | `maxwell` | 9B/4B/2B; 27B only if the 8 GB fit is verified | 27B | plain plan engine, no TC; already in `pair_ok` |
| Quadro P4000 8 GB | Pascal 6.1 | `pascal8` | 27B/9B/4B/2B | 27B now offered | plain plan engine, no TC; the dp4a kernels exist but `pascal8` does not enable `KINDI_TC=1` |
| Quadro RTX 4000 8 GB | Turing 7.5 | `turing` | 27B/9B/4B/2B | 27B now offered | Turing tensor-core backend (`KINDI_TC=1`); CUDA 10.2 supports sm_75 |
| RTX 4060 Ti 8 GB | Ada 8.9 | `tc` | 27B/9B/4B/2B; tc plan verify: 2B 4/4, 4B 2/4, 9B refused (plain 4/4) | 27B with `tc2`; 33B is not viable on 2×8 GB | `tc` profile, 24,576-token default; PTX patch; fix the 4B tc divergence and 9B HB=4096 before a tc support claim |
| RTX 5060 Ti 16 GB | Blackwell (confirm the driver’s exact CC; table says 10.0/12.8) | `tc` | 27B; 33B if the 16 GB fit is verified | 27B and 33B | `tc2`; Blackwell-capable toolkit or verified PTX JIT; 16 GB memory profile |

Notes:

- **RTX 4060 Ti tc verify (2026-11-10):** with `KINDI_TC=1 KINDI_TP=1 KINDI_FD=1 KINDI_ATT_TC=1
  KINDI_TC_TMAX=4 KINDI_SPEC=0`, the 2B is 4/4 token-exact at 354.5 tokens/s; the 4B is 2/4
  (prompts 2 and 3); the 9B is refused before graph build (Hadamard 4096). The plain 4B/9B paths are
  4/4 at 114.9/72.5 tokens/s. The tc plan is not a 4B/9B support claim yet.
- **33B needs about 15.5 GB of weights plus K/V and per-card overhead.** Two 8 GB cards (16 GB total)
  are not enough for the current Nanbeige estimator; a single 16 GB card is marginal and must be measured;
  two 16 GB cards are the safe dual target.
- **27B works on an 8 GB card today** (GTX 1080 is the existing proof). The 4060 Ti 8 GB should therefore
  be a single-card 27B target, not only a 2B card.
- **No tensor-core path on Turing (RTX 4000), Pascal (P4000/P106) or Maxwell (M4000).** They use the plain
  or plan engine. The 4060 Ti and 5060 Ti are the two tensor-core cards in this table. Pascal sm_61 also has
  the bit-identical dp4a kernels in `tcp.cu`, but the installer's `pascal8` profile leaves `KINDI_TC` off.
- **GT 1030 is not “unusable” for Kindi’s 2B**: the 2 GB K420 is the existing 2B proof point. The 2 GB
  card is a legitimate 2B target with a small context; 4B must be measured, and 9B/27B are out of scope.
  The generic LLM table’s “unusable” judgement is not the Kindi contract.

## Work packages

### P0 — make the installer classify and pair correctly

**Status 2026-11-10:** items 1-5 are implemented and covered by `tests/test_installer_cards.py` (12 cases:
4060 Ti 8 GB and 5060 Ti 16 GB are `tc`; 2×P106, 2×RTX 4000 and 2×4060 Ti pair; 2×GT 1030 and mixed classes do
not; `tc2` and the legacy dual profile are checked). No GPU verify is recorded, so this stays a roadmap, not a
support claim. Item 4 is satisfied by the existing model lists: the tc branch keeps `27b nemotron 9b 4b 2b`, and
the 2×8 GB pair offers 27B, not 33B.

1. In `card_class()`, separate **compute capability** from **memory profile**:
   - `tc`: CC >= 8.0 (4060 Ti 8 GB included), with a memory sub-profile for 8 GB and 16 GB.
   - `pascal8`: CC 6.1–7.9 at 7,500 MiB or more (smaller cards stay `maxwell`).
   - `p100`: the existing sm_60 exception.
   - `maxwell`: CC 5.0–6.0.
   - `kepler`: below 5.0.
   The 9,500 MiB test must not decide whether tensor cores exist.
2. In `pair_ok()`, allow same-class pairs from the supported range (at least 4 GB each), including
   sm_61, sm_75, sm_80, sm_86, sm_89 and Blackwell. Exclude the 2 GB GT 1030 pair. Keep a total-memory
   check for 27B and 33B.
3. Add a `tc2` profile in `profile_of()` for two CC >= 8.0 cards: `KINDI_CARDS=2`, `KINDI_TC=1`, context
   fitted to the smaller card, and a memory reserve that leaves room for the exchange. Keep speculative
   decoding off until the two-card verify test passes; add it in P1.
4. Update `card_models()` only from measured results: add 27B for dual P106/P4000/RTX 4000/4060 Ti, and
   do not offer 33B on a 2×8 GB pair.
5. Add installer routing tests with a fake `nvidia-smi` (the existing GRUB fixture is a good pattern):
   4060 Ti 8 GB → `tc`, 5060 Ti 16 GB → `tc`, 2×P106 → dual, 2×4060 Ti → dual, 2×GT 1030 → no pair.

### P1 — make the Ada/Blackwell path explicit

6. Add a CUDA 12.x / current-CuPy image for sm_80+ cards, while keeping the CUDA 10.2/CuPy 9.6 image for
   Kepler and other legacy cards. The 10.2 PTX patch in `tc.py` can stay as the compatibility path, but it
   must not be the only path for a new 5060 Ti.
7. [~] Test native sm_89 and sm_120 compilation against the same verify prompts. Confirm whether the current
   `tc.py` `ptx_target()` (sm_80 for everything after sm_86) works on Blackwell, or whether a native
   toolkit target is required. The 4060 Ti (sm_89) tc plan is 4/4 on the 2B but 2/4 on the 4B, and the
   9B is refused (Hadamard 4096); sm_120 remains untested.
8. Add the two-card `tc2` verify: 27B on 2×8 GB Ada, 27B/33B on 2×16 GB Blackwell, exact output and no
   exchange race. Only then turn on two-card spec.

### P2 — memory and context profiles

9. Measure the fitted context on 2 GB, 6 GB, 8 GB, 12 GB and 16 GB cards; set the installer defaults from
   the measurement, not from the card’s marketing name.
10. Add the 4B-on-2 GB result separately from the 2B result. If it is not stable, remove 4B from the GT 1030
    profile.
11. Update `docs/CARDS.md`, `docs/CONFIGURATION.md` and the support matrix with the measured settings.

### P3 — validation and performance

12. Run single and dual verify/bench for every row that is advertised. Record the commit, driver, CUDA,
    context and tokens/s; add the result to the existing benchmark tables.
13. Keep the legacy cards in the regression set: K420, K1200, K6000, GTX 1080, P100 and RTX 3080 must not
    lose their current paths.

## Acceptance matrix

| setup | 2B | 4B | 9B | 27B | 33B |
|---|---:|---:|---:|---:|---:|
| GT 1030 2 GB | single | measure | no | no | no |
| P106-100 6 GB | single | single | single | 2×P106 only | no |
| M4000 8 GB | single | single | single | single or 2× | no |
| P4000 8 GB | single | single | single | single or 2× | no |
| RTX 4000 8 GB | single | single | single | single or 2× | no |
| 4060 Ti 8 GB | single | single | single | single or 2× | no (2× = 16 GB) |
| 5060 Ti 16 GB | single | single | single | single or 2× | single if fit, 2× preferred |

A checked cell requires an installer option, a passing verify and a measured row in the docs. Empty cells
are not promises. The RTX 4060 Ti's tc results above are 2B 4/4, 4B 2/4 and 9B refused; the plain 4B/9B
paths are 4/4, so its 4B/9B cells are not tc-checked.

## Open decisions

- Is the 5060 Ti’s actual compute capability `10.0` or `12.0` on the target driver? Use `nvidia-smi
  --query-gpu=compute_cap` and record it; do not infer it from the marketing table.
- Does the current CUDA 10.2 PTX patch reach sm_120 reliably, or should the 5060 Ti require the modern
  image? P1 decides; until then the 5060 Ti is not a release-supported card.
- Is 2×P106 27B fast enough to be useful through PCIe x8/x4 with no display outputs? Measure before
  advertising it.
- Is 27B on a single 8 GB Ada card stable at a useful context, or does the TC graph memory reserve leave
  too little? Measure, then set the installer context.
- Is 2×4060 Ti 27B better as plain plan or TC plan? Start with TC on and spec off; add spec only after the
  two-card verify.
