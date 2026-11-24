# Manual help needed (owner actions)

Work that needs a human, not the agent: downloads blocked by a broken upstream
file, hardware only the owner has, installations/publishing only the release
owner can do, and optional tooling that unblocks agent work. Each entry has a
state line — tick it off when done.

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

## 4. Tests and runs on other hardware

All of these need machines the owner controls; the agent can only write the
commands and interpret the output. Each row states the gate it unlocks.

| # | rig | gate it unlocks | what is needed |
|---|---|---|---|
| 4a | **Intel Arc A770 (installed in the Windows test rig)** | REQ-0020 M1–M8: the SYCL/oneAPI backend (first non-NVIDIA GPU support) | Install the oneAPI Base Toolkit (record the release, `icx-cl --version`, `sycl-ls --verbose` and its supported Visual Studio version; this host has VS 2026), run the M1 arrival checklist in `docs/INTEL-ARC.md` (device/driver, pinned oneAPI/SYCL/Level Zero, ReBAR/PCIe state, CPU + llama.cpp SYCL baselines), then tell the agent the checklist output. The planned next M2 step is the SYCL vector-add smoke with its acceptance checks in `docs/INTEL-ARC.md`. |
| 4b | **P106-100 (6 GB, Pascal) single + dual** | Installer-offered pairs gain their first measured rows (single 2B/4B/9B; dual 27B) | `kindi verify` 4/4, sampled peak, read + decode at 100/2K/8K/24K — see `docs/TANDEM-RUNTIME.md` for the exact commands and cells. |
| 4c | **M4000 / P4000 / RTX 4000 (Maxwell/Pascal/Turing rig)** | CARDS.md pair offers on Pascal/Maxwell; the tandem-on-pre-tensor-core question (the exchange rate of the split differs there) | Same verify/bench matrix as 4b, plus the tandem-vs-full-GPU run at 2B/9B (2 packs) — the roadmap's dedicated pass. |
| 4d | **A Blackwell card (sm_120, e.g. RTX 5060 Ti)** | The native FP8 prompt GEMM, synthetic-tested on Ada but never on Blackwell | Run the manual Blackwell verification lane in `docs/ROADMAP.md` Formats; one real FP8 model run + a throughput number. |
| 4e | **i5-2500K (AVX-only) / Xeon W-2133 / i9-13900K** | The AVX-only CPU fallback and the other CPU rows of the matrix | Native CPU engine + CPU server chat re-run; timing numbers. |
| 4f | **A machine that cannot run full-GPU but can split (any 2-card pair)** | The mixed-card P2 phase and truly-larger-than-VRAM tandem rows | Two-card verify/bench on an unequal pair (e.g. 6 GB + 8 GB) for `KINDI_TP_SPLIT=auto`. |

## 5. Publications and packages only the release owner can do

- **Push the tree:** `git push origin` is 403 from this environment (house
  rule is also never-push). The fork (`myfork/main`) is behind; the owner
  must push. This is the single easiest owner action with the biggest
  unblock value (CI, installer refresh, ever). Read
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
