# Manual help needed (owner actions)

Work that needs a human, not the agent: downloads blocked by a broken upstream
file, hardware only the owner has, and publications that need the release
owner's authorization. Each entry has a state line — tick it off when done.

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
  `--pack L:/kindi-nb/packs/nanbeige-33b`); (b) use bartowski's
  `Nanbeige_Nanbeige4.3-33B-IQ3_XXS.gguf` (14.49 GB) from
  `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` — but that is a different quant,
  so the shipped `kindi/reference-nanbeige-33b*.json` will not reproduce and
  the references must be regenerated from `~/llama.cpp` (llama-server) before
  `kindi verify` can pass.
- **When done, tell the agent:** "rerun the Nanbeige T4.2 gate". The adapter
  (`kindi/backends/nanbeige.py`) is already ported and CPU-tested; only the
  pack + `kindi verify`/`kindi bench` parity remain.
