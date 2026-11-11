# CI and the GPU smoke lane

The normal GitHub-hosted workflow is
[.github/workflows/ci.yml](../.github/workflows/ci.yml):

- `CPU tests`: the CPU-only pytest suite on Ubuntu;
- `CPU engine build and smoke`: compiles `kindi/kcpu.c` with AVX2/FMA/F16C and loads it;
- `Windows CPU and config boundary`: the Windows-safe config, CPU, startup and tokenizer tests;
- `Installer shell checks`: `bash -n` and ShellCheck on the committed LF installer.

These jobs do **not** run CUDA kernels, a model pack or `kindi verify` on a real card.

## GPU smoke

[.github/workflows/gpu-smoke.yml](../.github/workflows/gpu-smoke.yml) is a manual
`workflow_dispatch` lane. It requires a self-hosted runner with both labels:

- `self-hosted`
- `gpu`

The runner needs a working NVIDIA driver, the CuPy version used by the chosen image, and a
converted Kindi pack. Start the workflow with the absolute pack path as its `pack` input. The job:

1. runs `python -m kindi verify` (token-for-token, greedy);
2. starts `python -m kindi serve` on `127.0.0.1:8199`;
3. waits for `/health`;
4. sends one non-streaming chat request with `temperature: 0`;
5. uploads the server log if any step fails.

This is intentionally not an automatic job: a GPU and a 0.4-19 GB model pack are not available on
GitHub-hosted runners, and a skipped job is not verification. Record the runner, card, driver,
CuPy/CUDA versions and pack hash with the run. Until that run exists, the RTX 50/Blackwell and
dual-card rows remain unverified as described in [CARD-SUPPORT.md](CARD-SUPPORT.md).

## Adding a pack to a runner

The pack is produced once with `python -m kindi convert MODEL.gguf`; point the runner at that
directory, not at the GGUF, and keep it outside the workspace. Do not commit the pack or any model
file. The workflow uses `KINDI_PACK`, so the same script works with a path containing spaces.
