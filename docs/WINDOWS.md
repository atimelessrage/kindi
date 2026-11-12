# Windows quickstart

Status: **native Windows CPU path verified; native Windows GPU path not verified.**
The one-command installer is Linux-only. For NVIDIA/AMD GPU work on Windows, use the Linux
Docker path (or WSL2 with the Linux installer) and treat the result as a Linux GPU run, not a
native Windows GPU result. See [INSTALL.md](INSTALL.md) and [DOCKER.md](DOCKER.md).

## What works natively

The native Windows path runs the **CPU engine** on the 2B, 4B and 9B Bonsai packs. It was
verified on 2026-11-10 on an Intel Core i7-8086K with Python 3.13, clang 18.1.8, the MSVC 14.51
toolset and the Windows SDK. No `vcvars` shell is needed.

| Model | CPU verify | Approximate speed |
|---|---:|---:|
| 2B | 4/4 token-exact | 33.7 tok/s |
| 4B | 4/4 token-exact | 16.7 tok/s |
| 9B | 4/4 token-exact | 9.1 tok/s |

These are one-run measurements from the recorded Windows run, not a promise for other CPUs.
The CPU engine supports **2B/4B/9B only**; 27B and Nanbeige are GPU targets. The full CPU
details and the AVX/AVX-only limits are in [CPU-SUPPORT.md](CPU-SUPPORT.md).

## Prerequisites

- Windows 10 or 11 on x86-64.
- A CPU with **AVX**. AVX2 + F16C is the fast path; FMA is optional. An AVX-only CPU uses the
  slower compatibility path. A CPU without AVX cannot run the CPU engine.
- Python 3.11 or newer. Python 3.13 was used for the verified run.
- LLVM/clang 18.1.8 for Windows, including `libomp.dll`, plus the MSVC build tools and Windows
  SDK that clang finds through `vswhere`.
- Enough disk for the model and its converted pack. The 2B is the smallest starting point.

The CPU engine does not need CUDA or CuPy. Native Windows GPU execution is not part of this
quickstart and has not been validated.

## Install the Python dependencies

From PowerShell in the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy fastapi uvicorn pydantic
```

Add `httpx` only if you are running the test suite:

```powershell
python -m pip install -r tests/requirements-test.txt
```

## Point Kindi at clang

Set the two paths for the current PowerShell session. Adjust the LLVM directory if you
installed a different version or location.

```powershell
$env:KINDI_CC = "C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe"
$env:KINDI_LLVM = "C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc"
$env:KINDI_DEVICE = "cpu"
```

For `cmd.exe`, use `set` instead:

```bat
set KINDI_CC=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc\bin\clang.exe
set KINDI_LLVM=C:\Program Files\clang+llvm-18.1.8-x86_64-pc-windows-msvc
set KINDI_DEVICE=cpu
```

Confirm that clang is visible and that Kindi can see the CPU path:

```powershell
clang --version
python -m kindi info
```

The first CPU run builds a keyed `libkcpu-<key>.dll` under `KINDI_CPU_CACHE` (or the default
cache). `libomp.dll` is loaded from the LLVM `bin` directory. The build key includes the host
CPU, compiler and ISA tier, so changing the compiler or CPU force-rebuilds it.

## Get and verify a model

The 2B is the recommended first model:

```powershell
python -m kindi models
python -m kindi use bonsai-2b
python -m kindi verify
```

`kindi use` downloads the GGUF, builds the pack under the configured state directory and selects
it. The pack load is the slow part on Windows (about 23 s for the 2B in the recorded run; the 9B
took about 112 s). `kindi verify` compares greedy output token-for-token with the reference.

For the 4B or 9B, use `python -m kindi use bonsai-4b` or `python -m kindi use bonsai-9b`.
The 27B is not a native Windows CPU target.

## Run the server

```powershell
python -m kindi serve --host 127.0.0.1 --port 8093
```

The OpenAI-compatible endpoint is then `http://127.0.0.1:8093/v1/chat/completions`. The health
endpoint is `http://127.0.0.1:8093/health`. For a client-side smoke test:

```powershell
$body = @{
  messages = @(@{ role = "user"; content = "Reply with OK." })
  max_tokens = 8
  temperature = 0
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8093/v1/chat/completions" `
  -Method Post -ContentType "application/json" -Body $body
```

Use `chat_template_kwargs.enable_thinking = $false` only if you are working with a model whose
template supports it; the server's default sampling is not greedy. Set `temperature: 0` for a
token-exact comparison.

## Threads and context

The engine defaults to one thread per physical core and sets OpenMP affinity. On the verified
6-core/12-thread i7-8086K that is 6 threads; do not assume all 12 logical CPUs are faster. On
this CPU, 12 threads collapsed to roughly 1–2 tok/s in the recorded sweep. To override explicitly:

```powershell
$env:OMP_NUM_THREADS = "6"
```

The configured context is `24576` by default. Raise it only after checking the pack and available
RAM; the CPU engine's long-context throughput falls with depth.

## Run the tests

```powershell
python -m compileall -q kindi
python -m pytest -q
```

The optional coverage gate used by CI is:

```powershell
python -m pytest -q --cov=kindi --cov-report=term --cov-fail-under=27
```

The suite is CPU-only and uses stubs for CUDA. It does not prove that a GPU path works.

## Limits and non-claims

- **No native Windows GPU support is claimed.** CUDA/CuPy execution on Windows has not been
  validated here; use the Linux Docker image or WSL2 for GPU work.
- **AMD graphics are not supported natively on Windows.** Use the Linux ROCm images.
- **The installer does not run on Windows.** It checks Linux, Docker and Linux GPU access.
- **No 27B, Nemotron or Nanbeige CPU model.** Those are GPU targets.
- **No AVX means no CPU engine.** The installer and engine report this before building.
- **The native GPU rows in the main support matrix are not Windows rows.** They remain Linux
  results unless a Windows run is recorded.
