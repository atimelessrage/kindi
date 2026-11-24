# Security policy

Kindi runs a local inference server, and the installer can optionally build a Docker image, install a
third-party kernel module and change the host's kernel command line. Treat the installer and a
network-exposed server as privileged, security-sensitive components. The repository's own secrets,
history-redaction and push policy is [docs/SECURITY.md](docs/SECURITY.md).

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** form on the fork:

<https://github.com/atimelessrage/kindi/security/advisories/new>

(Upstream Kindi is at `ahtx/kindi`; this fork's reports go to the fork. If the fork form is unavailable, contact the repository owner privately instead of opening a public issue.)

Do not open a public issue with exploit details, credentials, model files or personal data. Include:

- the Kindi version (`python -m kindi --version` or `/health`),
- the operating system, GPU/driver and whether Docker was used,
- the smallest reproduction you can share,
- the impact you believe it has.

We will acknowledge a report as soon as we can and coordinate disclosure after a fix is available.
Please do not test against systems you do not own or have permission to test.

## Threat model and operational guidance

- **Network exposure.** The API key is a bearer token, not transport encryption. The default bind is
  `127.0.0.1`; if you set `host = 0.0.0.0`, put the server behind a TLS-terminating reverse proxy and
  a firewall. `/health` always answers liveness, version, model and context; its detailed view
  (GPU memory, queue, counters, live progress) is withheld unless the request carries the API key,
  or the bind is loopback. Do not expose it to an untrusted network either way.
- **Image input.** The current text-only packs have no vision weights. Image content is rejected with
  HTTP 400 instead of being silently ignored. Do not assume a future vision pack accepts remote URLs
  without an explicit, guarded fetch policy.
- **Request limits.** Request bodies are capped (16 MiB by default, `max_body_mb` / `KINDI_MAX_BODY_MB`) and the
  waiting queue is bounded. A full queue returns HTTP 429 with `Retry-After`; retry once rather than
  opening more connections.
- **Secret scanning.** `.pre-commit-config.yaml` and the CI `secret-scan` job run detect-secrets
  against `.secrets.baseline`. The hook and CI scan every tracked text file; only the baseline is
  excluded. CI also rejects a stale baseline. Update it with
  `python tools/check_secret_baseline.py --update` only after reviewing a newly flagged value; the
  baseline stores hashes, not the secret values.
- **Installer.** `install.sh` is designed to be reviewed before running. It downloads models with
  pinned SHA-256 hashes and asks before privileged changes. Prefer a release archive with a verified
  checksum over piping a moving URL into a shell. The AMD path changes GRUB only after confirmation
  and keeps a backup.
- **Docker.** The images run as root unless you pass `-u $(id -u):$(id -g)`. Prefer rootless Docker
  or an explicit unprivileged user, and do not publish port 8093 directly to the internet.
- **Model and pack integrity.** A pack's `kindi_pack.json` and tensor files are trusted input. Do not
  load packs from untrusted sources; the converter and verifier are not a sandbox.

## Supported versions

Security fixes target the current 0.7.x release of this fork. Older releases and unmodified forks may not receive
fixes. This policy does not promise a bounty or a response time.
