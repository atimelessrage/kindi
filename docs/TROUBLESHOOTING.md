# Troubleshooting

**`OutOfMemoryError` at start.** Something else holds GPU memory. `nvidia-smi` lists GPU processes (a desktop
session uses a few MB, which is fine; another CUDA or Vulkan program is not). Free the card, or lower `ctx`
(each 1,000 tokens of context is ~12 MB).

**`no model pack at ...`.** Run `kindi convert <model.gguf>` first, or point `pack` at the right directory.

**`needs the PTQ1_0 GGUF`.** Kindi reads the PTQ1_0 files of Ternary-Bonsai-2-2B, -4B, -9B and -27B, and since
2026-11-10 it also converts the `PQ2_0` (type 142) and `Q2_0` (type 42) files of the 2B to a byte-identical pack
(`kindi convert` records `source_format` in `kindi_pack.json`). Only the 2B was converted and verified; the
4B/9B/27B `PQ2_0`/`Q2_0` files use the same types but were not downloaded here.

**CUDA errors such as `no kernel image is available` or `CUDA driver version is insufficient`.** Use the NVIDIA 470
driver and the CUDA 10.2 stack in the Docker image.

**The first start is slow.** Kindi compiles its CUDA kernels with NVRTC on first use (about a minute) and caches them
under `state_dir/kernels`.

**A client times out or retries on the first request.** Kindi reads the first long prompt at about 25-33 tokens/s. Raise the
client's timeout and limit retries (see [AGENTS.md](AGENTS.md)); watch progress with `curl -s
localhost:8093/health`.

**Answers are slow on every turn of an agent.** Check that the agent sends the same tools and system prompt each
time. `/health` (send `Authorization: Bearer <api_key>` when a key is configured) and the server log show `prompt: N reused, M read` per request.

**`401 invalid API key`.** The server has `api_key` set; send `Authorization: Bearer <key>`.

**`the prompt has N tokens; the context is M`.** The conversation outgrew `ctx`. Let the client summarise older turns,
or raise `ctx` if memory allows.

**`could not select device driver "" with capabilities: [[gpu]]`.** The Docker daemon started before you installed the NVIDIA
Container Toolkit. Restart the daemon: `sudo systemctl restart docker`, or `systemctl --user restart docker`
for rootless Docker. Then `docker run --rm --gpus all ubuntu nvidia-smi` must show the cards.

**`PermissionError: [Errno 13] Permission denied: '/data/...'` on rootless Docker.** Remove `-u $(id -u):$(id -g)` from
the `docker run` command. Rootless Docker runs the container's root as your user (see [DOCKER.md](DOCKER.md)).

**`kindi verify` fails.** The pack is from a different model file, or the GPU stack differs from the tested one.
Re-run `kindi convert --force` from the official file and check the driver.

**The server answers `429` and `Retry-After: 1`.** Up to eight requests may wait behind the active one
(`KINDI_QUEUE_MAX`). Wait a second and retry once; do not open more parallel connections or retry in a loop.
Watch `queue.waiting`, `queue.max` and `queue.rejected` in `/health` (send `Authorization: Bearer <api_key>` when a key is configured).

**The server exits after a CUDA error, or keeps restarting.** Kindi exits instead of serving from a broken CUDA
context. Check `nvidia-smi`, `dmesg` and the container log; restart the container only after the GPU is healthy.
The installer uses `--restart unless-stopped`, so a persistent error becomes a restart loop rather than wrong output.

**`400 this pack has no vision weights; image input is not supported`.** The selected Bonsai pack is text-only.
Remove the image part or use a text-only prompt. A pack is vision-capable only when its `kindi_pack.json` names
`vision` or `mmproj` weights; this release does not load them.

**`/health` shows only `ok`, `kindi`, `commit`, `model` and `ctx`.** Detailed health fields are available with
`Authorization: Bearer <api_key>` when `api_key` is set. With no key, those details are loopback-only: a
non-loopback bind gets only liveness. If you need queue/GPU details from a monitor, set `api_key` and send the
header, or bind the server to loopback.

**The CPU engine cannot build.** The first CPU start needs `gcc` with OpenMP on Linux, or `clang` with `libomp` on
Windows. Check `gcc --version` or `clang --version`; the build uses `-march=native`, so a binary copied from another
CPU may not run. Kindi logs the exact compiler command it runs. Recovery knobs:

- `KINDI_CPU_CACHE=/path/to/writable/cache` - use a writable cache directory instead of `~/.cache/kindi`.
- `KINDI_CC=gcc` or `KINDI_CC=clang` - choose the compiler explicitly; it is part of the cached library key.
- `KINDI_LLVM=/path/to/llvm` - point at the LLVM installation that contains `lib/` and `bin/` on Windows.
- `KINDI_CPU_ISA=auto|avx2-fma-f16c|avx2-f16c-no-fma|avx2-fma|avx2|avx-only` - force an explicit ISA tier;
  an unsupported forced tier fails before building instead of running an illegal instruction.

`KINDI_CC` and `KINDI_LLVM` are operator-controlled local settings, not a security boundary: if the server process
inherits untrusted environment variables, they can choose which compiler runs at first CPU-library build.
