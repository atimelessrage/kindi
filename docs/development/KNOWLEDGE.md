# Engineering knowledge

Durable operational facts that do not belong in a single decision, requirement or user
guide. Keep commands copy-pasteable and avoid local machine paths and credentials.

## Verification layers

| layer | command | what it proves |
|---|---|---|
| Parse and unit gate | `python -m compileall -q kindi && python -m pytest tests/ -q` | Python modules parse; CPU-only behavior and guards hold. |
| CPU build/smoke | `KINDI_CPU_SMOKE=1 KINDI_CPU_CACHE=<cache> python -m pytest tests/test_cpu_build.py tests/test_cpu_threads.py tests/test_cpu_smoke.py -q` | The native CPU library builds and loads (a compiler is required). |
| CPU exactness | `KINDI_CPU_PACK=<pack> python -m pytest tests/test_cpu_layers.py -q` | Per-layer CPU execution matches the reference. |
| GPU exactness | `python -m kindi --pack <pack> --ctx 1024 verify` | Greedy output matches the reference for the selected engine. |
| Generated docs | `python -m kindi.config_reference --check`, `python -m kindi.support_matrix --check`, `python docs/benchmarks/build_cross_mode.py --check` | Generated references match source. |
| Project memory | `python tools/project_memory.py --check` | Requirement history is current. |
| Docs links | `python -m pytest tests/test_docs_links.py -q` | Local Markdown links resolve. |

## Configuration and generated files

- Precedence is built-in defaults < config file < environment < command-line flags.
- `docs/CONFIGURATION-REFERENCE.md` is generated from `kindi/config.py` and the
  `KINDI_*` environment reads. Regenerate it after adding or moving a key.
- `docs/SUPPORT-MATRIX.md` is generated from the model catalog, conversion formats, K/V
  modes and CPU ISA checks. Regenerate it after a support change.
- `docs/benchmarks/CROSS-MODE.md`/`.csv` are generated from measured benchmark files.
- Generated files are checked by tests; a green unit suite does not excuse a stale
  generated reference if you have not run the generator.

## Support boundaries worth remembering

- The exact head and `kindi verify` are the exactness reference; benchmark throughput does
  not replace correctness.
- The 9B tensor-core plan is refused because its Hadamard block is 4096. The plain engine
  is the fallback.
- The 4B tensor-core path uses the standard decode attention because split flash-decode
  diverges on its 2560-wide plan. This is a conditional pass, not a claim about split
  decode on that geometry.
- The 27B plain engine refuses grouped head dimensions it does not support. A refusal is a
  known boundary, not a benchmark failure.
- `k8v4` is experimental, opt-in and quarantined; it is not token-exact. See the
  requirement and decision records before touching it.
- CPU support requires AVX and chooses one of five explicit ISA tiers (AVX2/FMA/F16C,
  AVX2/F16C, AVX2/FMA, AVX2, AVX-only). Forced-tier exactness on an AVX2 host is not a
  native Sandy Bridge/no-FMA hardware claim.
- The dense Bonsai CPU+GPU tandem runtime is implemented for the 2B/9B in `cpu-first`
  and `gpu-first` placements, with 27B also gated after generic GQA support; the
  Nanbeige host-layer path and MoE/Gemma tandem remain separate.

## Git archaeology

```sh
git log --follow --oneline -- docs/development/requirements/REQ-0001.md
git show <commit>:docs/development/requirements/REQ-0001.md
git log -S'REQ-0001' --all -- docs/development/requirements
git show <deletion-commit>^:docs/development/requirements/REQ-0001.md
```

The generated history records 12-character abbreviations. Use the full hash from
`git log` when a command needs it. Never rewrite a requirement ID to hide a reversal; add
a new decision or mark the old one superseded.

## External documentation

- User-facing behavior: `README.md`, `docs/*.md`, `examples/`.
- Agent-facing API usage: `docs/AGENTS.md`.
- Installation and release: `docs/INSTALL.md`, `docs/RELEASE-CHECKLIST.md`.
- Support claims: `docs/SUPPORT-MATRIX.md` and the per-card pages.
- Internal process: this directory.

A user-visible behavior change is incomplete until the external page and examples are
updated. Run the docs-link test after moving or renaming a page.
