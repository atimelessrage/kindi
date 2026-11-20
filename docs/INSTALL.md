# The one-command installer

`install.sh` installs Kindi with one command. It finds your NVIDIA and AMD graphics, shows each setup that your
computer can run, with the best one first, and then downloads and checks the models, converts them, checks the
output against llama.cpp and starts the servers as services.

This installer is **Linux-only**. Windows users can follow [WINDOWS.md](WINDOWS.md) for the
verified native CPU path and [the measured 2B source-tree GPU run](WINDOWS.md#native-windows-gpu-validation-2026-11-19).
Native Windows GPU execution is not covered by this Linux installer and is not a general support claim: the 2026-11-19 result
covers one 2B pack on an RTX 4060 Ti 8 GB from the source tree; the 4B/9B/27B packs, other cards,
AMD and the installer's GPU profiles remain unmeasured. For a supported GPU setup, use the Linux
Docker image or WSL2.

## Install (Linux x86-64)

Download the script, compare its sha256 with the value below, read it, and then run it. This is
the path the project supports: the checksum covers the file you actually execute.

```bash
curl -fsSLO https://github.com/atimelessrage/kindi/releases/download/v0.7.0-fork/install.sh
echo "7fa65bdb8829a30e280645ebbc191ab9675118c4aaac509093dc2e868b8145e1  install.sh" | sha256sum -c -
bash install.sh --dry-run     # show every step and command first
bash install.sh
```

`curl ... | bash` is convenient but verifies nothing; if you use that form, it runs the hosted file
without the checksum block. The fork release tag is `v0.7.0-fork`; the older `v0.5.0-installer`
tag predates this tree and is not the current installer.

On Windows 10 or 11, use the PowerShell installer. It sets up WSL2 and then runs this installer
inside it (see [WINDOWS.md](WINDOWS.md)). Download it and compare the hash below with a reviewed
copy before running rather than piping it straight into `iex`:

```powershell
Invoke-WebRequest https://github.com/atimelessrage/kindi/releases/download/v0.7.0-fork/install.ps1 -OutFile install.ps1
Get-FileHash install.ps1      # compare with the table below
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

For Docker Compose, Podman, servers and automated installs, there are also ready-made images (no build):
[CONTAINERS.md](CONTAINERS.md). The installer builds the same images on your computer.

| File | sha256 |
|---|---|
| `install.sh` (Kindi 0.7.0-fork) | `7fa65bdb8829a30e280645ebbc191ab9675118c4aaac509093dc2e868b8145e1` |
| `install.ps1` (Kindi 0.7.0-fork) | `0c25abe5f6806ee57070b52b8dd04c1509f4a1cda583fef729b407a9a7ef2888` |
| `kindi-0.7.0-fork.tar.gz` (source archive) | `6086c3cc3c359dfad27a81aeb3215a9784a80284ff39c45e9d29e8b434000499` |

**What those values hash.** They are the exact release assets: the two installers and the source
archive. The archive is built from the release tree and excludes the installers, the two
checksum-bearing release pages and the secret-scanner baseline, so its hash does not depend on
itself. The installers are uploaded
as separate assets. Older tags, including `v0.5.0-installer`, hash differently; a mismatch means a
stale or modified artifact, not tampering.

If `sha256sum -c` does not print `install.sh: OK`, do not run the script. The checksum is for the
fork `install.sh` in the release bundle (the shared tree plus the fork's pinned source archive, card
classes and verify-speed fallback; LF line endings - a copy re-saved with CRLF hashes differently).
The upstream 0.6.0 release's own `install.sh` hashes to
`ec0f3f19cf5c5c2d6f0f660db1efb9a6e289a4cfaf0a7c8decfb996f5e4471dc`; that value is historical and
is not used by this fork release.

## What it does

| Step | What it does | sudo |
|---|---|---|
| 1. Check the machine | Linux x86-64, AVX for the CPU engine, RAM, free disk, each NVIDIA GPU with its compute capability, and each AMD RDNA3 or RDNA 3.5 GPU. Without AVX, GPU setups remain available but CPU setups are not offered. | no |
| 2. Docker and GPU access | Finds Docker, and the NVIDIA Container Toolkit or the AMD devices `/dev/kfd` and `/dev/dri`. If something is missing, it asks before it installs it (apt). Without the toolkit, it gives the container the NVIDIA devices and the driver libraries directly. | only to install |
| 3. Select the setup | Shows each setup for this computer, the recommended one first, and asks which one (below). | no |
| 4. Build the image | Builds `kindi:0.7.0-fork` (CUDA 10.2), `kindi-rocm:0.7.0-fork` (ROCm 6.3, RDNA3) or `kindi-rocm7:0.7.0-fork` (ROCm 7, RDNA 3.5) from the release source, and checks that the Kindi modules load. A re-run with the same source does not build again. | no |
| 5. Download the models | Downloads each GGUF from Hugging Face. The sha256 must match the value in the script (table below). | no |
| 6. Convert and verify | `kindi convert`, then `kindi verify` one time for each model on each kind of card. | no |
| 7. Start the servers | One Docker container for each server, with `--restart unless-stopped`, on `127.0.0.1`. It waits until each `/health` answers. | no |

At the end it shows each server's address, model, context and the speed of the check, and the settings for OpenAI
clients and for pi (one provider for each server). It writes no secrets and asks for no keys. The servers listen on
127.0.0.1 only.

If your user is not in the `docker` group, the script uses `sudo docker`.

## Select a setup

The installer lists each setup that your computer can run. A computer with an RTX 3060 and a GTX 1080 shows:

```text
    1) every GPU, one server each: 27B on GPU 0 + 27B on GPU 1 (recommended)
    2) Ternary-Bonsai-2 27B on GPU 0 (NVIDIA GeForce RTX 3060)
    3) NVIDIA Nemotron 3.5 Bolt E4B on GPU 0 (NVIDIA GeForce RTX 3060)
    ...
    7) Ternary-Bonsai-2 27B on GPU 1 (NVIDIA GeForce GTX 1080)
    ...
    12) Ternary-Bonsai-2 2B on the CPU (80 threads; slow, no GPU needed)
```

- One number gives one server.
- Two or more numbers, for example `2,8`, give one server for each setup, side by side.
- `NxK` gives setup N on K cards of the same kind, for example `5x2` for a 4B on each of two K1200 cards.

A kind of card shows one time in the list, also when the computer has many cards of that kind. A card has at most one
server. Each server gets its own container (`kindi`, `kindi-2`, ...), the next free port from 8093, and its own
cache folder when a model runs more than one time.

To select without questions, add `--auto`: the installer takes the recommended setup and continues. In a script, add
`--config` and the numbers, for example `--config 2,8`. `--dry-run` prints the list.

### The recommended setup

| Your computer | Recommended | Other setups in the list |
|---|---|---|
| No GPU | 2B on the CPU | none |
| Quadro K420 1 GB | 2B | 2B on the CPU |
| Quadro K420 2 GB | 4B | 2B, 2B on the CPU |
| One Quadro K1200 4 GB | 9B | Nemotron, 4B, 2B, 2B on the CPU |
| Two Quadro K1200 4 GB | 27B split over both cards (tensor parallel) | Nemotron split over both cards; 9B, Nemotron, 4B or 2B on each card; 2B on the CPU |
| GeForce GTX 1080 8 GB | 27B | Nemotron, 9B, 4B, 2B, 2B on the CPU |
| GeForce RTX 3060 / 3080 (sm_86, 9.5 GB or more) | 27B with tensor cores and the 2B draft model | Nemotron, 9B, 4B, 2B, 2B on the CPU |
| AMD Radeon 780M (gfx1103) | 27B | Nemotron, 2B, 2B on the CPU |
| AMD RDNA 3.5: Radeon 840M (gfx1152), Radeon 8060S (gfx1151) | 27B (12 GB of graphics memory or more) | 9B, 4B, 2B, 2B on the CPU |
| Two or more GPUs that do not split one model | the best model on every GPU, one server each | each setup of each card |

| Profile | Cards | Context asked |
|---|---|---|
| cpu | no GPU, or `--gpu cpu` | 24,576 |
| kepler | sm_30-sm_37 | 24,576; 65,536 at 1.5 GB or more; 23,808 for the 4B on a 2 GB card |
| maxwell | below sm_61, or sm_61-sm_79 below 7.5 GB | 24,576 |
| k1200x2 | two supported cards below sm_80, or a mixed pair that is not two sm_80+ cards; Kepler is single-card | 24,576 |
| pascal8 | sm_61-sm_79, 7.5 GB or more | 65,536 |
| tc | sm_80 or newer | 131,072; 24,576 on an 8 GB card |
| tc2 | two sm_80 or newer cards | 24,576 (8 GB pair) or 65,536 |
| rocm-gfx1103 | AMD Radeon 780M | 24,576 |
| rocm-gfx1151, rocm-gfx1152 | AMD Radeon 8060S, Radeon 840M (RDNA 3.5) | 24,576 |

Nemotron always asks for 24,576. The engine fits the context to the free memory and reports what it got. For
example, Nemotron split over two K1200 cards gets 8,448.

## AMD graphics

The Kindi ROCm images hold ROCm, CuPy 14.2.0 and Kindi, so you do not install ROCm on the computer. RDNA3 graphics
get `kindi-rocm` (ROCm 6.3.3). RDNA 3.5 graphics (gfx1150-gfx1152) get `kindi-rocm7` (ROCm 7, HIP 7.16). The
computer needs the `amdgpu` driver of Linux (kernel 6.2 or newer and `linux-firmware`) and Docker. The installer gives
the container `/dev/kfd` and `/dev/dri`. On the Radeon 780M it also sets `HSA_OVERRIDE_GFX_VERSION=11.0.2` and
checks the kernel option `amdgpu.cwsr_enable=0` (see [AMD.md](AMD.md)). It asks before it adds that option, and it
never adds it with `--yes` or `--auto`.

## Kernel changes

The installer changes the kernel only when you answer `y` at the keyboard. `--yes` and `--auto` never answer these
questions; they only print what to do.

- **gpu-unlock** (GeForce RTX 30 cards): the installer tells you which cards are not unlocked and offers to download
  and install the third-party [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module at the reviewed version.
  Kindi does not include it.
- **`amdgpu.cwsr_enable=0`** (Radeon 780M): see [AMD.md](AMD.md).

## Model files

The script pins these files. It does not use a file with a different sha256.

| Model | File | Size (bytes) | sha256 |
|---|---|---|---|
| 2B | `Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 429,183,456 | `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa` |
| 4B | `Ternary-Bonsai-2-4B-PTQ1_0.gguf` | 949,549,920 | `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60` |
| 9B | `Ternary-Bonsai-2-9B-PTQ1_0.gguf` | 1,997,486,976 | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |
| 27B | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| Nemotron | `NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf` (unsloth) | 4,219,133,120 | `3f8032b9d2c18536c4ac51b97959a563cc27434de99823640c98147edc62ce9b` |
| Nanbeige 33B | `AH-Nanbeige4.3-33B-IQ3_XXS.gguf` | 13,108,685,952 | `9d4f18e8a22573c94b18a455147e807a8b9d2cd4b30302f2a4639c2db0a6c239` |

### Use a model you already downloaded in Ollama

Ollama stores its weights as content-addressed blobs (`blobs/sha256-<hash>`), which Kindi cannot
name. This helper reads Ollama's manifests and links each model under a readable `.gguf` name,
without copying or downloading anything:

```sh
python tools/ollama_links.py --dry-run    # list what it would link
python tools/ollama_links.py              # $OLLAMA_MODELS, else ~/.ollama/models
python tools/ollama_links.py -m <ollama-root>/models -o <ollama-links>
```

It links into `~/.kindi/models/ollamaLinks` by default, named after the manifest (`gemma3/4b` ->
`gemma3-4b.gguf`), and then `python3 -m kindi convert ~/.kindi/models/ollamaLinks/gemma3-4b.gguf`
works like any other GGUF. A symbolic link is made when the host allows it; otherwise a hard link,
which needs the output on the same drive as the blobs (the script says so when they differ, and
`-o` moves it). Manifests whose blob is missing are reported, `--force` replaces a file that is
not the same model, and `--dry-run` changes nothing. On Windows, `tools/ollama_links.ps1` is the
same script for PowerShell.
## Options

| Option | What it does |
|---|---|
| `--dry-run` | Shows each step, each command and the list of setups. It changes nothing. |
| `--yes`, `-y` | Answers yes to each question (for scripts) and takes the recommended setup. Kernel changes are never answered. |
| `--auto` | Takes the recommended setup and continues with no questions. Kernel changes are only described. |
| `--config N` | Setup N of the list; `N,M` for several servers; `NxK` for setup N on K cards. |
| `--model M` | Lists only the setups with this model: `2b`, `4b`, `9b`, `27b`, `nemotron` or `nanbeige`. |
| `--ctx N` | The context length in tokens. |
| `--gpu G` | `N` (one NVIDIA card), `N,M` (two cards), `amd` or `amdN`, `all`, or `cpu`. |
| `--port P` | The port of the first server on 127.0.0.1 (default 8093). Later servers take the next free ports. |
| `--lan` | Also serve your local network, so that other computers can use Kindi. Off by default. The installer makes an API key, keeps it in `~/.kindi/api-key` (only you can read it), and every client must send it as `Authorization: Bearer <key>`. A later `--lan` run keeps the same key. Run the installer again without `--lan` to go back to this computer only. |
| `--api-key K` | The API key that clients must send. With `--lan`, `--api-key none` turns the key off (not recommended). Without `--lan`, it sets a key for local clients. |

### Use Kindi from other computers (`--lan`)

```sh
bash install.sh --lan
```

The installer publishes each server port on all the network addresses of this computer, not only 127.0.0.1, and
prints the network address (`http://<this computer's address>:8093/v1`). The containers keep their normal Docker
network, so the GPU stays visible to them (changing the Docker network mode by hand can hide the GPU from the
container). On the other computer, use that address and the key in `~/.kindi/api-key`:

```sh
curl http://<address>:8093/v1/chat/completions -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <key>" -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

- Use `--lan` only on a network that you trust. Anyone with the key can use the servers. Do not forward the ports
  from the internet.
- If a firewall is on (for example `ufw`), allow the port for your network:
  `sudo ufw allow from <your network>/24 to any port 8093`.
- `/health` stays open without the key. It shows the model and the server state, not the conversations.
| `--dir D` | The data folder (default `~/.kindi`). |
| `--name N` | The name of the first container (default `kindi`; later ones are `N-2`, `N-3`, ...). |
| `--threads N` | The CPU engine: the number of threads. |
| `--memory M` | The container memory limit, for example `8g`. |
| `--source S` | The Kindi source: a folder, or a `.tar.gz` path or URL (default: the 0.7.0-fork release). |
| `--with-bridge` | Also installs Kindi Bridge, the terminal workspace for agents on Kindi (see below). |
| `--no-bridge` | Does not ask about Kindi Bridge. |
| `--bridge-source D` | Installs Kindi Bridge from folder D (default: `bridge/` in the Kindi source). |
| `--uninstall` | Removes each server container and image that the installer made, and Kindi Bridge. The models stay. |
| `--uninstall --purge` | Also removes the data folder, and asks before it deletes Bridge's state. |

## Kindi Bridge (optional)

Kindi Bridge is a workspace in the terminal for agents on Kindi: conversations, scheduled prompts, completion
checks, folders of your own documents, and the live state of your Kindi servers. When the installer runs in a
terminal, it asks if you want it. `--with-bridge` installs it with no question. `--yes` and `--auto` alone do not
install it.

Bridge runs on the host, not in a container. It needs Python 3.10 or later with `venv`, tmux 3.2 or later, and the
pi coding agent 1.3.0 (Node.js 22.19 or later). The installer offers to install the missing parts: Python and tmux with
`apt-get`, and pi with `npm` into `~/.local`. It puts the program in `~/.kindi/bridge`, the command in
`~/.local/bin/kindi-bridge`, and the state in `~/.local/state/kindi-bridge`. Bridge has its own pi folder there, so
your own pi setup does not change. The installer connects Bridge's agents to the first Kindi server. It can also run
Bridge's scheduler as a systemd user service. Run the installer again to upgrade Bridge. The manual has a chapter
with each menu and option.

## Upgrade, repair and remove

Run the script again to upgrade, repair or change the setup. It keeps the steps that are already correct: the image
(same source), the models (sha256 already checked) and the packs (made from the same files). It always runs
`kindi verify` again and starts new containers. Servers of the earlier install that the new setup does not keep are
removed.

```bash
bash install.sh --uninstall          # the containers and the images; the models stay in ~/.kindi
bash install.sh --uninstall --purge  # also ~/.kindi
```

## Tested

These are the last recorded full installer tests: a real install, a chat request, a tool-call request and
`--uninstall --purge` (Kindi 0.3.0-dev, 2026-10-24). They have not been re-run for every 0.5.0 card; the
0.5.0 hardware evidence is in [CARDS.md](CARDS.md) and [BENCHMARKS.md](BENCHMARKS.md).

| Computer | Setup | Result |
|---|---|---|
| 2× Quadro K1200 | 27B split over both cards | PASS: verify 4 of 4 token-exact, 13.4 tokens/s |
| 2× Quadro K1200 | Nemotron split over both cards | PASS: verify 4 of 4 (16-40 of 40 tokens agree), 11.6 tokens/s; context 8,448 |
| 2× Quadro K1200 | `5x2`: a 4B on each card | PASS: verify 4 of 4 token-exact, 39.9 tokens/s; both servers answer |
| GeForce GTX 1080 8 GB, without the toolkit | 27B | PASS: verify 4 of 4 token-exact, 30.3 tokens/s |
| Quadro K6000 12 GB (NVIDIA 470, in a KVM virtual machine) | 27B, context 156,672 | PASS: verify 4 of 4 token-exact, 20.4 tokens/s; the cache answers the second request |
| GeForce GTX 1080 8 GB, without the toolkit | Nemotron | PASS: verify 4 of 4 (16-40 of 40 tokens agree), 24.1 tokens/s |
| AMD Radeon 780M (Ryzen 9 7940HS, 64 GB), without ROCm on the host | 27B, ROCm image built by the installer | PASS: verify 4 of 4 token-exact, 9.7 tokens/s; context 24,576 |

For Nemotron, `kindi verify` compares the first 40 tokens of four prompts with llama.cpp's output for the same file.
A prompt passes when the first 12 or more tokens agree. Kindi's Nemotron engine computes in a different order than
llama.cpp, so long outputs can part on a near tie.
