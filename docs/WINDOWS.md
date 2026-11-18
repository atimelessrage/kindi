# Kindi on Windows

Kindi runs on Windows 10 and Windows 11 through WSL2. WSL2 is a small Linux virtual machine that is a part of
Windows. One PowerShell command installs everything: it turns on WSL2, makes a Linux system only for Kindi, installs
Docker, checks the GPU, and runs the normal Kindi installer inside. Then Windows programs use Kindi at
`http://localhost:8093/v1`, the same as on Linux.

## Install

Open PowerShell (you do not need "Run as administrator") and run:

```powershell
irm https://amirhusain.com/kindi/install.ps1 | iex
```

To see each step before anything changes, do a dry run first:

```powershell
& ([scriptblock]::Create((irm https://amirhusain.com/kindi/install.ps1))) -DryRun
```

The installer prints each step and the reason for it. It asks before it changes something.

**If Microsoft Defender or your company's security software stops the one-line command:** some security tools
block every "download and run" PowerShell command. Download the script, check it, and run it as a file instead:

```powershell
Invoke-WebRequest https://amirhusain.com/kindi/install.ps1 -OutFile install.ps1
Get-FileHash install.ps1        # compare with the sha256 in the release notes
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

When WSL2 starts for the first time, Windows can show a "Welcome to Windows Subsystem for Linux" window. You
can close it: the install goes on in its own window. Every run writes a log to `%LOCALAPPDATA%\Kindi\install.log`.

| Step | What happens |
|---|---|
| 1. Check the PC | An x86-64 PC, Windows 10 2004 (build 19041) or newer, virtualization, RAM, disk, the GPU and its Windows driver, the proxy |
| 2. Turn on WSL2 | Only if it is off. Windows asks for administrator approval once, and the PC restarts. The installer continues by itself after you sign in again |
| 3. A Linux system for Kindi | A new WSL2 distribution with the name `Kindi`. It is Ubuntu 24.04 from Canonical, with a sha256 check. Your other WSL distributions do not change. If `Kindi` exists, the installer reuses it |
| 4. Docker and the GPU | If you have Docker Desktop, Kindi uses it. If not, the installer puts Docker Engine and the NVIDIA container toolkit into the `Kindi` distribution. Then the installer checks that the GPU is visible inside WSL2 |
| 5. The Kindi installer | `install.sh` runs inside WSL2. It shows the setups for your GPU, the recommended one first, downloads and checks the model, and starts the server |
| 6. Windows items | A check that `http://localhost:8093` answers on Windows, Start-menu items, a desktop shortcut, and an entry in Settings > Apps |

Run the same command again to upgrade or repair Kindi. The installer keeps the models that you downloaded.

## What you need

- **An x86-64 PC** (Intel or AMD). Windows on ARM (for example Snapdragon laptops) cannot run Kindi.
- **Windows 10 version 2004 or newer, or Windows 11.** For the GPU in WSL2, you need Windows 11 or Windows 10 21H2
  (build 19044) or newer. On older builds, Kindi uses the CPU engine.
- **Virtualization on** in the BIOS/UEFI (Intel VT-x or AMD SVM). Most PCs have it on.
- **An NVIDIA GPU with a current Windows driver** from nvidia.com (Game Ready or Studio). Do not install a Linux NVIDIA
  driver inside WSL2: the Windows driver serves WSL2. Without an NVIDIA GPU, Kindi uses the CPU engine.
- **Disk:** about 15-40 GB, for Linux, Docker, the models and their packs.

**AMD GPUs:** Kindi's AMD engine needs the Linux amdgpu compute driver (`/dev/kfd`). WSL2 does not have it. On a
Windows PC with an AMD GPU, Kindi uses the CPU engine. For the AMD GPU speed, use native Linux.

## Use Kindi

- **Kindi Console** (Start menu > Kindi > Kindi Console, and on the desktop): the status of each server, a chat window,
  the logs, a restart, and the settings for OpenAI clients.
- **Kindi Shell (WSL)**: a Linux shell in the `Kindi` distribution. The `kindi` commands and `docker` work there.
- **Any OpenAI client on Windows:** set the base URL to `http://localhost:8093/v1` and the API key to `none`.
  See [AGENTS.md](AGENTS.md) for agents and chat clients.

```powershell
$env:OPENAI_BASE_URL = 'http://localhost:8093/v1'; $env:OPENAI_API_KEY = 'none'
```

WSL2 stops an idle Linux system after a short time. With Docker Engine inside the `Kindi` distribution, a small hidden
process (`Kindi WSL keep-alive` in the Startup folder) keeps it running, so the server is always ready after you sign
in. With Docker Desktop, Docker Desktop keeps the server running.

## Options

Add options after `-File install.ps1`, or after the `scriptblock` form above.

| Option | What it does |
|---|---|
| `-DryRun` | Print every step and command, change nothing |
| `-Check` | Only check the PC and say what the install would do |
| `-Yes` | Answer yes to every question |
| `-LinuxArgs '...'` | Options for `install.sh`, for example `'--model 27b'`, `'--gpu cpu'` or `'--config 2'` (see [INSTALL.md](INSTALL.md)) |
| `-Distro NAME` | Use another WSL2 distribution. The installer uses a distribution that you have again, and never replaces it. The installer asks before it adds Docker or turns on systemd in it |
| `-DistroDir PATH` | Where a new distribution keeps its disk (default `%LOCALAPPDATA%\Kindi\wsl`). Use it to put Kindi on a larger drive |
| `-DockerMode MODE` | `auto` (default: Docker Desktop if you have it), `desktop` or `engine` |
| `-WslMemoryGB N` | Give the WSL2 virtual machine N GB of RAM (see below) |
| `-WithBridge` | Also install Kindi Bridge, the agent workspace (see the Kindi Bridge chapter). The installer adds Python 3, tmux and Node.js 22 to the `Kindi` distribution for it, and a Start-menu item "Kindi Bridge" that opens it in Windows Terminal. Without this option, an interactive run asks (the default is no); `-Yes` alone never adds it |
| `-NoBridge` | Do not install Kindi Bridge and do not ask |
| `-NoDesktopShortcut` | No desktop shortcut |
| `-NoReboot` | After the installer turns on WSL2, do not restart. You restart, and the install continues after you sign in |
| `-Uninstall`, `-Purge` | Remove Kindi (see below) |

## RAM in WSL2

Windows gives the WSL2 virtual machine about half of the PC's RAM by default. The installer shows both numbers. The
27B and the larger models need more. To give WSL2 more RAM, run the installer with `-WslMemoryGB N`, for example
`-WslMemoryGB 48` on a 64 GB PC. This sets `memory=` in `%UserProfile%\.wslconfig` (the installer keeps a copy of the old file)
and runs `wsl --shutdown` once. That command stops all WSL distributions and Docker Desktop, so the installer asks
first.

## Proxies

The installer uses the Windows proxy setting (including PAC and WPAD), or the `HTTPS_PROXY` variable. It sets the
proxy for the downloads, for `apt` and Docker inside WSL2, and for `docker build`. A proxy that runs on the PC's own
`localhost` (for example Px or Cntlm) is not visible from WSL2, unless `networkingMode=mirrored` is set in
`.wslconfig`. Otherwise, give the proxy's network address in `HTTPS_PROXY`.

## Verified installer run (2026-11-17)

The Windows installer had only ever been exercised through `-DryRun`/`-Check` on real
hardware. This is a real run on a Windows 10 Pro 19045 PC (i7-8086K, 32 GB RAM, RTX 4060 Ti,
WSL2 already on, Docker Desktop installed), done with a separate name and location so the
machine's own setup is untouched:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 `
  -AppName KindiInstallerTest -Distro KindiInstallerTest `
  -DistroDir '<wsl-root>\wsl' `
  -DockerMode skip -NoBridge -NoDesktopShortcut -Yes -LinuxArgs '--dry-run'
```

`-DockerMode skip` and `-LinuxArgs '--dry-run'` keep the run off Docker Desktop's settings and
off the multi-GB model download; everything the Windows half does (PC checks, the Ubuntu
image, the distribution, systemd, the GPU check, the Windows items) ran for real. Wall clock: **01:02:30 to 01:04:04** (94 s, transcript times).

| step | observed |
|---|---|
| PC check | Windows 10 Pro build 19045 x86-64, virtualization on, 12 threads, 32 GB RAM, 22 GB free on `S:`, RTX 4060 Ti on driver 617.42, no proxy, WSL2 on; noted that 22 GB is under the 15-40 GB guidance and that Docker setup was skipped |
| Ubuntu image | downloaded from `cloud-images.ubuntu.com/wsl/releases/24.04/current`, **sha256 `8251e27f…` matched Canonical's SHA256SUMS**, then deleted from the cache |
| distribution | `wsl --import KindiInstallerTest <wsl-root>\wsl` (WSL2), a Linux user named after the Windows account, `systemd=on` and that user as the default written to `/etc/wsl.conf`, distro restarted (other distributions kept running) |
| resources | WSL memory 16 GB of 32 (the script says `-WslMemoryGB 24` is the way to raise it) |
| GPU in WSL | `nvidia-smi -L` inside the distribution: `GPU 0: NVIDIA GeForce RTX 4060 Ti (UUID: GPU-78de4e15…)` — the Windows driver reaches WSL2 as documented |
| Linux installer | the *published* `install.sh` reports **0.6.1**, checked the machine (GPU 0 sm_89, 8188 MiB), listed the supported setups and picked setup 1 (27B on GPU 0, context 65536, profile `pascal8`, `kindi` container on port 8093), then stopped at its own `--dry-run` before building the image |
| Windows check | no server to poll, so the localhost check was skipped exactly as a dry run predicts |
| Windows items | Start-menu folder with `KindiInstallerTest Console`, `KindiInstallerTest Shell (WSL)` and `Uninstall KindiInstallerTest`; Settings > Apps entry `KindiInstallerTest (WSL2)`; no desktop shortcut (`-NoDesktopShortcut`); no scheduled task; `keepAlivePid: 0` in `state.json` |

Where it wrote, and nowhere else:

| location | contents |
|---|---|
| `<wsl-root>\ext4.vhdx` | the distribution's disk, 1.3 GB after provisioning |
| `%LOCALAPPDATA%\KindiInstallerTest\` | `state.json` (172 B: distro, `createdDistro`, `dockerMode`, user, `bridge: false`), `install.log` (the transcript, 3.8 KB), a copy of `install.ps1`, `kindi-console.ps1`, empty `download\` cache, `wsl-scripts\`; 85 KB total |
| `%APPDATA%\Microsoft\Windows\Start Menu\Programs\KindiInstallerTest\` | the three shortcuts (1.2-1.7 KB each) |
| `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\KindiInstallerTest` | `DisplayName KindiInstallerTest (WSL2)`, `DisplayVersion 0.6.0`, `InstallLocation` = the state folder, `UninstallString` = `install.ps1 -Uninstall -AppName … -Distro …` |

What it did **not** touch, verified after the run: the `Ubuntu` and `docker-desktop`
distributions (both still registered, `Ubuntu` still the default), Docker Desktop's
`settings-store.json` (mtime August 23), and port 8093 (nothing listening). The machine's own
Kindi setup was never a target: a different `-Distro`, `-AppName` and `-DistroDir` keep the
distribution, shortcuts, registry entry and state folder distinct.

Two things this run showed and did not hide:

- **The Windows and Linux installers report different versions.** The published `install.sh`
  says **0.6.1**; `install.ps1` (`$KindiWinVersion`) and this repository's `install.sh` say
  **0.6.0**, so Settings > Apps shows 0.6.0 while the distribution holds 0.6.1. The version
  pairing is a release-owner step (republish or pin the two together), tracked in
  [MANUAL-HELP-NEEDED.md](MANUAL-HELP-NEEDED.md).
- **The in-WSL disk check is not a host-space check.** Inside the distribution `install.sh`
  reported **954.5 GB free** for its own home directory, because it sees the virtual disk's capacity,
  while the host `S:` actually had 22 GB free. The Windows layer is the one that knows: it prints
  the host figure. On a nearly full drive the host-side note is the constraint to trust, and a
  model download can still fail after the Linux check passes.

To remove the test install (this deletes the distribution and its disk):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Uninstall -Purge `
  -AppName KindiInstallerTest -Distro KindiInstallerTest -Yes
```

## Remove Kindi

Use Settings > Apps > Kindi (WSL2) > Uninstall, or Start menu > Kindi > Uninstall Kindi. It removes the Kindi servers
and image, the keep-alive process, the Start-menu items, the shortcut and the Settings entry. Then it asks if it
should also delete the `Kindi` distribution with its models. With `-Purge`, it deletes them without a question. A
distribution that the installer did not make (`-Distro`) is never deleted: only Kindi is removed from it.

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\Kindi\install.ps1" -Uninstall
```

## Problems

| Message | What to do |
|---|---|
| "this PC has an ARM processor" | Kindi needs an x86-64 PC |
| "Windows build ... is too old for WSL2" | Settings > Windows Update: install the updates, then run the installer again |
| "virtualization is off in the BIOS/UEFI" | Restart the PC and open the firmware setup (often F2, F10, F12 or Del during start). Turn on Intel Virtualization Technology (VT-x) or AMD SVM Mode, save, and run the installer again |
| "the NVIDIA GPU does not reach WSL2" | Install the current NVIDIA Windows driver from nvidia.com and restart. Do not install a Linux driver inside WSL2. For the CPU engine only: `-LinuxArgs '--gpu cpu'` |
| "Docker Desktop does not answer in 'Kindi'" | Docker Desktop > Settings > Resources > WSL integration: turn on `Kindi`, then Apply & restart. Or use `-DockerMode engine` |
| "does not answer ... from Windows" | WSL2 forwards `localhost` by default. If `.wslconfig` has `localhostForwarding=false`, remove it and run `wsl --shutdown` |
| "systemd did not start" | Run `wsl --update` (WSL 0.67.6 or newer is needed), then run the installer again |

The Linux installer's own messages are in [INSTALL.md](INSTALL.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Native Windows CPU quickstart (without WSL2)


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
- **The Linux installer does not run on Windows.** Use `install.ps1` (above) for the WSL2 path, or run this native CPU path directly.
- **No 27B, Nemotron or Nanbeige CPU model.** Those are GPU targets.
- **No AVX means no CPU engine.** The installer and engine report this before building.
- **The native GPU rows in the main support matrix are not Windows rows.** They remain Linux
  results unless a Windows run is recorded.
