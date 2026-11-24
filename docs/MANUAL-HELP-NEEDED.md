# Manual help needed (owner actions)

Work that needs a human, not the agent: downloads blocked by a broken upstream
file, hardware only the owner has, installations/publishing only the release
owner can do, and optional tooling that unblocks agent work. Each entry has a
state line — tick it off when done.

This is the **only** manual-help tracker. A local scratch copy is not kept:
anything it holds is folded in here, and a stray one is now visible to
`git status` instead of being ignored.

## Automated tasks (no owner action)

- ~~The Lightning T4.2 GPU gate is **the agent's**, not the owner's: do not
  use the old `<kindi-model-root>/kindi-lightning/models/...` file whose three hashes
  disagreed.~~ **Done 2026-11-16:** a clean re-download to
  `<kindi-model-root>/kindi-lightning-clean/models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`
  matches the catalog size/SHA-256/GGUF magic; a fresh pack was built and
  `kindi verify` on the RTX 4060 Ti is 5 exact + 7 ties, 0 fail, with bench
  read 788.3–821.3 tok/s and write 165.96–167.41 tok/s at ctx 8192.
  Evidence in `docs/benchmarks/2026-11-16-i7-8086k-rtx4060ti-lightning-30b/RESULTS.md`.

## 1. Download the Nanbeige 4.3 33B GGUF and resume the T4.2 Nanbeige gate

- **State:** blocked 2026-11-16, waiting on the owner.
- **What:** get a working `AH-Nanbeige4.3-33B-IQ3_XXS.gguf` (or an equivalent
  IQ3_XXS Nanbeige4.3-33B file) onto this host, then the Nanbeige adapter GPU
  gate (verify/bench parity on the RTX 4060 Ti) can run.
- **Why blocked:** the catalog file at
  `ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF` is corrupt as served by Hugging
  Face (2026-11-16 measurement): full 13,108,685,952-byte download, but its
  SHA-256 is `39d8e2d8a71377e8283b08437d753a39941a38a5f1d7881a2ca0f1f543615005`
  (catalog records `9d4f18e8…`), and only 5 of its 471 tensor-info entries
  parse as names — the rest are NUL-garbled (`blk.0.attn_q.weight` lookups
  raise `KeyError` in `kindi/nanbeige.py`). A fresh range re-download matched
  the same bytes, so HF is serving them as-is. The corrupt copy was deleted
  (13.1 GB on `L:`) at the owner's request.
- **Options:** (a) report the broken file to the repo author / wait for a
  fixed upload, then download with `kindi use nanbeige-33b` (note: `C:` has
  ~13 GB free, so put the pack on a bigger drive, e.g. `L:` with
  `--pack <kindi-model-root>/kindi-nb/packs/nanbeige-33b`); (b) use bartowski's
  `Nanbeige_Nanbeige4.3-33B-IQ3_XXS.gguf` (14.49 GB) from
  `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` — but that is a different quant,
  so the shipped `kindi/reference-nanbeige-33b*.json` will not reproduce and
  the references must be regenerated first (see §3).
- **When done, tell the agent:** "rerun the Nanbeige T4.2 gate". The adapter
  (`kindi/backends/nanbeige.py`) is already ported and CPU-tested; only the
  pack + `kindi verify`/`kindi bench` parity remain.

## 2. Download the PQ2_0 / Q2_0 files for 4B / 9B / 27B (the 142/42 import claim)

- **State:** open since 2026-11-15, waiting on the owner (agent-side HF was
  unreachable that day; recheck `curl https://huggingface.co` — it answers
  200 now, so an agent retry is viable, but no agent should spend ~20 GB of
  download without asking).
- **What:** `Ternary-Bonsai-2-4B-PQ2_0.gguf`, `…-9B-PQ2_0.gguf`,
  `…-27B-PQ2_0.gguf` (and the plain Q2_0 variants) from
  `prism-ml/Ternary-Bonsai-2-4B-gguf` etc. — roughly 5 GB + 19 GB + 19 GB
  in packs, on a drive that has room (L: or T:, not C:).
- **Why:** Kindi's import claim for those types (tensor types 142/42) is
  verified only on the 2B today. With the downloads: `kindi convert` each,
  `kindi verify` against llama.cpp, and — if they pass — the claim can be
  extended to all sizes and the AGENTS.md/SUPPORT-MATRIX notes updated.
  Without them, "2B-only" stays the honest claim.
- **When done, tell the agent:** "run the PQ2_0/Q2_0 import gate".

## 3. Make a pinned llama.cpp reference server available (llama-server)

- **State:** open, waiting on the owner (tooling).
- **What:** a working `llama-server` binary pinned to a known llama.cpp
  version, so reference-token generation (the thing `kindi verify` compares
  against) can be regenerated on demand.
- **Why:** several pending gates need fresh references: the bartowski Nanbeige
  quant (§1b), the Gemma 26B G0 reference, the Cascade 2 pinned reference,
  the Dory D1 goldens. Today `~/llama.cpp` has source and a build tree but
  no `llama-server` on PATH. A pinned binary (or a short build command +
  version pin committed to `docs/development/KNOWLEDGE.md`) unblocks all of
  them; right now each of those gates is stuck on "no reference".
- **When done, tell the agent:** the binary path (or that `llama-server` is
  on PATH), and the version.

## 3b. Put the installed C compiler on `PATH` for the CPU engine (Windows host)

- **State:** resolved in code on 2026-11-24; the owner action is now optional tidying.
- **What:** clang+LLVM 18.1.8 (with libomp) is installed under Program Files, but `clang` is not on
  `PATH`. Either add its `bin` to `PATH`, or set `KINDI_CC` to the `clang.exe` and `KINDI_LLVM` to
  the install root; the engine now also finds a standard LLVM install on its own.
- **Why this was open:** `kindi/kcpu.c` is compiled at first use into a content-keyed cache
  (`~/.cache/kindi/libkcpu-<key>.dll`). The engine only looked for the literal name `clang`, so
  every live CPU-engine run failed with `kindi: could not run clang` and the CPU path looked
  unavailable although the compiler was installed. `kindi/cpu.py:_find_cc` now resolves `KINDI_CC`,
  then `PATH`, then `Program Files/clang+llvm-*` or `LLVM*` (newest first), and `kindi caps` uses the
  same resolver.
- **Measured after the fix on this host:** a library builds from the current `kcpu.c` in 6.3 s;
  `KINDI_DEVICE=cpu kindi verify` is 4/4 token-exact on the 2B (24.6-29.6 tok/s) and on the 27B
  (3.7 tok/s, 340 s load); `kindi info` reports `AVX=yes, AVX2=yes, FMA3=yes, F16C=yes`.
- **Ran after the fix:** the live half of REQ-0028 A4 on the 2B pack (`kindi selftest`: `dialects agree`,
  exit 0, 23.5 s).
- **Still unrun here:** `kindi selftest` on the 27B pack (its two dialect requests pay the same 7-minute
  load) and native-hardware CPU rows on other machines.

## 3c. Decide four unbaselined history rows on `pr/07-intel-probe`

- **State:** open; a maintainer decision, not a code change.
- **What:** `python tools/check_secret_history.py` (its default range is `--all`, so it covers every ref) fails on
  `tests/test_intel.py: local drive path +8 (baseline 4)`. The eight rows come from three 2026-11-22 commits on
  `pr/07-intel-probe` (`b8d8b70`, `e7270c8`, `658a885`) that no other ref contains; this branch adds none
  (`--range 0d3c474..HEAD` is clean).
- **Why it is open:** either review those rows and re-record them with `--update-baseline` (endorsing them as
  examples, the way earlier rows were), or rewrite that branch before it is pushed or merged.
- **When done, tell the agent:** which of the two, and the baseline count afterwards.

## 4. Tests and runs on other hardware

All of these need machines the owner controls; the agent can only write the
commands and interpret the output. Each row states the gate it unlocks.

| # | rig | gate it unlocks | what is needed |
|---|---|---|---|
| 4a | **Intel Arc A770 (installed in the Windows test rig)** | REQ-0020 M1–M8: the SYCL/oneAPI backend (first non-NVIDIA GPU support) | **Install done 2026-11-24:** oneAPI 2026.2 is on the host with VS 2026 (`icx-cl`, `icpx`, `dpcpp`, `sycl-ls`). The earlier "no Level Zero device" note was environmental, not a hardware or driver gap: the Level Zero adapter is only loaded from the oneAPI `bin` directories, and with those on `PATH` `sycl-ls` lists the Arc A770 through Level Zero. The SYCL vector-add smoke now **passes** on Level Zero (0 mismatches, `result: pass`; `docs/benchmarks/2026-11-24-i7-8086k-arc-a770-sycl/`), so what is left needs no owner action: the same smoke on the Linux M2 target, plus recording the M1 arrival facts this shell cannot see (ReBAR/PCIe state, the CPU and llama.cpp SYCL baselines). The driver's own compilers already run every OpenCL probe on the card (`intel-smoke` Level Zero copies, `intel-cl` vector-add/GEMM, `intel-ternary` phase-parallel packed GEMV, `intel-gemm` the engine's q8 GEMM bit-exact at 4.2-67.8 GFLOP/s). |
| 4b | **P106-100 (6 GB, Pascal) single + dual** | Installer-offered pairs gain their first measured rows (single 2B/4B/9B; dual 27B) | `kindi verify` 4/4, sampled peak, read + decode at 100/2K/8K/24K — see `docs/TANDEM-RUNTIME.md` for the exact commands and cells. |
| 4c | **M4000 / P4000 / RTX 4000 (Maxwell/Pascal/Turing rig)** | CARDS.md pair offers on Pascal/Maxwell; the tandem-on-pre-tensor-core question (the exchange rate of the split differs there) | Same verify/bench matrix as 4b, plus the tandem-vs-full-GPU run at 2B/9B (2 packs) — the roadmap's dedicated pass. |
| 4d | **A Blackwell card (sm_120, e.g. RTX 5060 Ti)** | The native FP8 prompt GEMM, synthetic-tested on Ada but never on Blackwell | Run the manual Blackwell verification lane in `docs/ROADMAP.md` Formats; one real FP8 model run + a throughput number. |
| 4e | **i5-2500K (AVX-only) / Xeon W-2133 / i9-13900K** | The AVX-only CPU fallback and the other CPU rows of the matrix | Native CPU engine + CPU server chat re-run; timing numbers. |
| 4f | **A machine that cannot run full-GPU but can split (any 2-card pair)** | The mixed-card P2 phase and truly-larger-than-VRAM tandem rows | Two-card verify/bench on an unequal pair (e.g. 6 GB + 8 GB) for `KINDI_TP_SPLIT=auto`. |

**Owner's standing note (2026-11-24):** the rows above are parked on purpose, not
waiting on an action today — the current focus is the Arc A770 path (4a), and
another card or a different machine may follow later. No CUDA device is visible
from this shell at all (`nvidia-smi` reports insufficient permissions, CuPy
raises `cudaErrorNoDevice`), so even a single-card CUDA rerun cannot happen
here. The installed Arc A770 is visible through Level Zero/OpenCL and its probes
run, but it is not a CUDA second card and has no Kindi backend yet. When a rig
appears, the two-card cache runs use the local scratch scripts under `bench-logs/`
(`test_k8v4_cache.py` with `KINDI_CARDS=2`, then `test_k8v4_dist_cache.py`, which
then needs no stubbed device); the tracked `tests/test_k8v4.py` already pins the
quarantine and the CPU-side selection, so only the card run is missing.

## 5. Publications and packages only the release owner can do

- **Push the tree:** `git push origin` is 403 from this environment (house
  rule is also never-push). Write access to `ahtx/kindi` is **not** needed and
  should not be chased: this is a fork (`myfork` = `atimelessrage/kindi`), so
  contributions go upstream as pull requests. The fork (`myfork/main`) is
  behind; the owner must push. This is the single easiest owner action with the
  biggest unblock value (CI, installer refresh, ever). Read
  [SECURITY.md](SECURITY.md) first: the remotes still hold pre-redaction
  objects, so a plain fast-forward can republish what the 2026-11-17 rewrite
  removed.
- **Run the offsite backup:** the repository and its evidence are on one
  volume, and pushing has never succeeded. Run
  `python tools/backup_bundle.py --out <second-medium>/kindi-backups`, then
  copy the bundle and its `.sha256` off this machine. See
  [BACKUP.md](BACKUP.md).
- **Publish the 0.7.1-fork release bundle** (see `docs/RELEASE-CHECKLIST.md`): the tree
  now names **0.7.1-fork** in the package, both installers and the Docker tags; the local
  source archive and its hash exist in `dist/`. The source URL and checksum in
  `install.sh` point at the not-yet-created `v0.7.1-fork` release asset, so publication
  and the hosted checksum pair are still owner-only steps. Publishing/tagging needs the
  owner's explicit authorization (the checklist says so).
- **Retire the stale `v0.5.0-installer` one-liners** (2026-11-16 audit, F9): the docs
  and both installers now point at `v0.7.1-fork`; the old tag's asset predates this tree.
  Do not publish or link it as the current installer.
- **Publish the Windows and Linux installer pair together** (2026-11-17 real run, see
  [WINDOWS.md](WINDOWS.md#verified-installer-run-2026-11-17)): `install.ps1`
  (`$KindiWinVersion`) and `install.sh` now both say **0.7.1-fork**, so Settings > Apps
  and the installed tree will agree. The hosted pair is still unverified until the
  release owner uploads these exact assets.
- **Report the corrupt ahtx upload (§1) to the repo author** (or file/feedback
  on the HF repo) so a fixed file exists upstream; the agent cannot file
  it for you.
- **Register the self-hosted GPU CI runner** (steps in `docs/CI.md` §5.1):
  the agreed-cards gate (P4000/M4000/RTX4000 trio) and every GPU gate in
  this file want a machine that runs them; today they are hand-run and
  hand-recorded. This is the registration that turns "measured once" into
  "enforced".

## 6. Decide (maintainer calls, no hardware or downloads needed)

- **T4.4/T4.5 positioning (REQ-0019):** llama.cpp proxy (recommended) vs
  native generic backend (design only). There is no format-coverage claim
  until one of them lands and passes `kindi verify` or the proxy gate. Only
  the maintainer can make this call; the agent will not invent the product
  decision.
- **Whether to add old runtimes** (Python floor D3, queue default D5 —
  `docs/ROADMAP.md` Agent API bullet): these are product calls recorded in
  the same place.

## 7. Put the DSpine weights on this host (the `dspine` verify cell)

- **State:** blocked 2026-11-17, waiting on the owner.
- **What:** the DSpine drafter directory (`config.json` + `model.safetensors`) for the 27B, i.e.
  `KINDI_DSPINE=DIR` as `docs/DSPINE.md` describes.
- **Why blocked:** there is no `dspine` directory under `~/.kindi` (checked 2026-11-17), and the
  weights are a separate download from the model packs.
- **Unblocks:** `kindi verify --spec dspine` (REQ-0025 A3), the one shippable speculation mode with
  no exactness cell; `ngram`, `mtp`, `draft2b` and `both` are measured on the 2B (`docs/MTP.md`).

## Done

- **Nanbeige corrupt-file deletion** (2026-11-16): the 13.1 GB corrupt GGUF
  and `<kindi-model-root>/kindi-nb` were removed at the owner's request; the adapter port and
  the gate are recorded so the download (§1) resumes cleanly.
- **27B pack for the K8/V4 long-context gate** (2026-11-24): built as a **copy**
  (the source GGUF was not touched and its sha256 `53107f53…33ee3` re-checked
  after conversion). The pack is 7,242 MB, built in 357 s by
  `python -m kindi convert <gguf> --out <pack>`: 64 layers, width 5120,
  head_dim 256, Hadamard block 1024 (the tensor-core-friendly case). The
  planner accepts it — no split at ctx 8192 (7,266 MiB against a 7,680 MiB
  budget with 8 GiB assumed); at the default maximum context it plans 2 CPU
  layers. Keep the pack on the external model volume: the system drive holds
  the 2B/4B/9B packs and has under 6 GB free. What is left is GPU exactness
  (`kindi verify`, K8/V4), which is parked with §4 because it needs a CUDA
  device; the CPU engine is measured separately in `docs/CPU-SUPPORT.md`.
- **No history rewrite** (decided 2026-11-17, recorded in
  [SECURITY.md](SECURITY.md)): the reviewed 160-row / ~3,318-hit baseline is
  kept rather than rewritten again, so no approval or force-push is
  outstanding.
- **The `# api_key` line in `kindi.conf.example`** is the placeholder
  `choose-a-long-random-string`, not a credential — nothing to rotate or
  scrub.
