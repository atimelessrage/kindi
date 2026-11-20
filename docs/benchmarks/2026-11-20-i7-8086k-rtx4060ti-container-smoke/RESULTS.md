# Container smoke test: image builds, GPU passes, `/health` and chat pass on Python 3.6

This page records the repaired local build of the **fork** image
`kindi:0.7.0-fork`, its Python 3.6 compatibility check, and the
**unmodified-image** `/health` + one-chat smoke on the RTX 4060 Ti host.
The image runs Python 3.6.9, builds and loads the 2B pack on the GPU, and now
binds port 8093 normally.

| Check | Result | Evidence |
|---|---|---|
| Python 3.6 package compile | **PASS** | 94 files, `python36_compile_failures=0` |
| Image build | **PASS** | [`docker-build.log`](docker-build.log), `BUILD_EXIT=0` |
| GPU passthrough | **PASS** | [`nvidia-smi` and `kindi info`](#gpu-passthrough-preflight) |
| Pack + GPU engine load | **PASS** | [`unmodified-startup.log`](unmodified-startup.log) |
| `/health` on the unmodified image | **PASS, 200** | [`unmodified-health.json`](unmodified-health.json) |
| One chat on the unmodified image | **PASS, 200 in 0.2337 s** | [`unmodified-chat.json`](unmodified-chat.json) |

This is the fork image, not an official `v0.6.1` container. No official 0.6.1
image was built or pulled here, so this page does **not** establish official
image parity.

## Fix

The original image failed before binding because Python 3.6 cannot parse
`from __future__ import annotations`. The fix removes that import from the four
modules that carried it:

- `kindi/sessions.py`
- `kindi/cascade.py`
- `kindi/draft_policy.py`
- `kindi/moe_compat.py`

None of the four uses postponed annotations, so removing the import changes no
runtime annotation behavior. The whole package was compiled with the image's
own interpreter:

```sh
docker run --rm -v "$PWD/kindi:/src/kindi:ro" \
  --entrypoint python3 kindi:0.7.0-fork /tmp/check.py
# checking 94 files
# python36_compile_failures=0
```

The rebuilt image was also checked directly:

```sh
docker run --rm --entrypoint sh kindi:0.7.0-fork -c \
  "python3 --version; grep -n 'from __future__ import annotations' \
   /opt/kindi/kindi/sessions.py"
# Python 3.6.9
# (no match)
```

## Host and image

| | |
|---|---|
| Host | Windows 10 Pro 19045, Intel Core i7-8086K (6C/12T), 32 GB RAM |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8,188 MiB, driver 617.42 |
| Docker | Docker Desktop, Linux engine 29.9.3, WSL2 kernel 6.18.58.1 |
| Image | `kindi:0.7.0-fork` |
| Image ID | `sha256:b08916d7d25a19c87831196067a4407d0dac5d1143237b22c97dc59aa3b45af1` |
| Size | 2.56 GB (`docker image ls`); `docker image inspect` reports 2,559,096,710 bytes |
| Labels | `org.opencontainers.image.version=0.7.0-fork`, `...revision=1627b96312509e6e2bdaad805116f7f6459b88f3-dirty` |
| Python | 3.6.9 (GCC 8.4.0), from the image |
| CUDA / CuPy | CUDA 10.2, CuPy 9.6.0 (image contents) |

## Build

The image was built from the repository's `docker/Dockerfile` and the working
tree's `kindi/` directory:

```sh
cp -a kindi bench-logs/docker-build-context/kindi
cp docker/Dockerfile bench-logs/docker-build-context/Dockerfile
docker build --progress=plain -t kindi:0.7.0-fork \
  --build-arg KINDI_VERSION=0.7.0-fork \
  --build-arg KINDI_REVISION="$(git rev-parse HEAD)-dirty" \
  -f bench-logs/docker-build-context/Dockerfile \
  bench-logs/docker-build-context
```

`diff -qr kindi bench-logs/docker-build-context/kindi` exits 0, so the
`COPY kindi` input is identical to the working tree. The reduced context is not
the literal `docker build ... .` command in the Dockerfile: it omits `docs/`,
`tests/` and the rest of the worktree, none of which the Dockerfile copies.
There is no `.dockerignore` in the repository. The build ended `BUILD_EXIT=0`;
the CUDA installer's pinned SHA-256 check passed before installation. The full
transcript is [`docker-build.log`](docker-build.log).

## GPU passthrough preflight

```sh
docker run --rm --gpus all --entrypoint nvidia-smi kindi:0.7.0-fork \
  --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

```
NVIDIA GeForce RTX 4060 Ti, 8188 MiB, 617.42
```

`kindi info` was also run in the same image (ID `b08916d7…`) on 2026-11-20,
using the smoke's named volume; it exits 0 and reports the card and the 2B pack
at `/data/pack`:

```sh
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v kindi-smoke-data:/data \
  -e KINDI_PACK=/data/pack \
  -e KINDI_STATE_DIR=/data/state \
  --entrypoint python3 kindi:0.7.0-fork -m kindi info
```

```
Kindi 0.7.0-fork
…
pack: /data/pack - Ternary-Bonsai-2-2B, from Ternary-Bonsai-2-2B-PTQ1_0.gguf
…
GPU: NVIDIA GeForce RTX 4060 Ti, compute 89, 7039 of 8188 MiB free
```

The full output is [`kindi-info.txt`](kindi-info.txt). `kindi info` reports the
pack manifest, not that the engine loaded it; the pack load is the startup log
below. The engine log records the plan engine and the tensor-core configuration
used by the smoke.

## Unmodified-image run

The host's existing `~/.kindi/packs/bonsai-2b` pack was copied into a named
volume because the engine writes its GPU cache beside the pack. The host pack
was not modified.

```sh
# MSYS_NO_PATHCONV=1 stops Git Bash from rewriting the /data/... values.
MSYS_NO_PATHCONV=1 docker run -d --name kindi-smoke-fixed --gpus all \
  -p 127.0.0.1:18093:8093 \
  -v kindi-smoke-data:/data \
  -e KINDI_PACK=/data/pack \
  -e KINDI_STATE_DIR=/data/state \
  -e KINDI_HOST=0.0.0.0 \
  -e KINDI_API_KEY=<test-key> \
  -e KINDI_CTX=4096 \
  kindi:0.7.0-fork
```

The image was not modified at runtime; there was no bind mount and no overlay.
Startup reached the GPU and the pack:

```
[kindi 08:10:12] blocking sync: card 0 primary context flags 0x4 (set: 0)
[kindi 08:10:14] card 0: 0.41 GB of weights (1s)
[kindi 08:10:14] plan engine ready: 1 card, ctx 4096, K/V q8_0, weights 5-trit lane (2.8s)
[kindi 08:10:15] [kindi] engine ready for bonsai2-2b
[kindi 08:10:15] serving on http://0.0.0.0:8093/v1 (model bonsai2-2b)
```

The complete startup log is [`unmodified-startup.log`](unmodified-startup.log).

`/health` returned HTTP 200:

```json
{"ok":true,"kindi":"0.7.0-fork","commit":null,"model":"bonsai2-2b","ctx":4096}
```

`commit` is `null` because the working tree's `kindi/COMMIT` still holds the
archive placeholder; the image label carries the real revision.

One non-streaming request:

```sh
curl -sS -H 'Authorization: Bearer <test-key>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-2b","messages":[{"role":"user","content":"Reply with exactly: container smoke ok"}],"max_tokens":24,"temperature":0}' \
  http://127.0.0.1:18093/v1/chat/completions
```

returned HTTP 200 in **0.2337 s**:

```json
{
  "id": "chatcmpl-37fe1ce04ad44e888786dca2",
  "model": "bonsai2-2b",
  "choices": [{"index": 0,
               "message": {"role": "assistant", "content": "container smoke ok"},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 19, "completion_tokens": 3, "total_tokens": 22},
  "timing": {"prompt_s": 0.21, "prompt_new_tokens": 19,
             "gen_s": 0.01, "gen_tok_s": 211.01}
}
```

The full response is [`unmodified-chat.json`](unmodified-chat.json). The
container was still running when the request completed and was stopped
afterward.

## Initial blocked build (history)

The first image (`sha256:b2d9f1d7ec03ff1697b15b715e11197811fad211caa2f5100639ee9e3778d97d`)
built and loaded the pack, but `serve` exited before binding port 8093:

```
File "/opt/kindi/kindi/sessions.py", line 10
    from __future__ import annotations
    ^
SyntaxError: future feature annotations is not defined
```

The original failure log is [`startup-failure.log`](startup-failure.log). A
diagnostic overlay that removed only that one import returned `/health` 200 and
one chat 200, recorded in [`health.json`](health.json),
[`chat.json`](chat.json) and [`overlay-startup.log`](overlay-startup.log). That
overlay was the isolation step; the rebuilt image above is the evidence for the
shipped image.

## Exactly what was skipped, and why

| Step | Status | Why |
|---|---|---|
| `KINDI_MODEL` download and pack conversion | skipped | the existing `bonsai-2b` pack was copied into the volume, to test the server surface without a download |
| CPU-engine smoke | skipped | the host supports GPU passthrough and the GPU path was selected |
| ROCm / ROCm 7 images | skipped | this page covers only the CUDA image |
| Restart persistence, streaming, tool calls, auth failures, rate limits | skipped | out of scope for this health + one-chat smoke |
| Official `v0.6.1` container image | skipped | no official 0.6.1 image was available locally; this is the fork image |
| Full repository-context build | skipped | reduced context contains the same `docker/Dockerfile` and byte-identical `kindi/`; the Dockerfile only copies `kindi` |

## Limits

- One host, one card, one 2B pack, `ctx=4096`, one non-streaming chat request.
- The image was built from the working tree at `1627b96` with uncommitted
  changes, including the Python 3.6 fix; the label records that tree as `...-dirty`.
- This is not a speed result and not an official 0.6.1 image comparison.
- Raw build and startup transcripts live in the ignored `bench-logs/` tree;
  this page, the build log and the three unmodified-image outputs are the
  durable evidence.
