# Security policy

Kindi runs a local inference server, and the installer can optionally build a Docker image, install a
third-party kernel module and change the host's kernel command line. Treat the installer and a
network-exposed server as privileged, security-sensitive components.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** form:

<https://github.com/ahtx/kindi/security/advisories/new>

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
  a firewall. `/health` is intentionally unauthenticated and reports model, queue and GPU details;
  do not expose it to an untrusted network.
- **Image input.** The current text-only packs have no vision weights. Image content is rejected with
  HTTP 400 instead of being silently ignored. Do not assume a future vision pack accepts remote URLs
  without an explicit, guarded fetch policy.
- **Request limits.** Request bodies are capped (16 MiB by default, `KINDI_MAX_BODY_MB`) and the
  waiting queue is bounded. A full queue returns HTTP 429 with `Retry-After`; retry once rather than
  opening more connections.
- **Installer.** `install.sh` is designed to be reviewed before running. It downloads models with
  pinned SHA-256 hashes and asks before privileged changes. Prefer a release archive with a verified
  checksum over piping a moving URL into a shell. The AMD path changes GRUB only after confirmation
  and keeps a backup.
- **Docker.** The images run as root unless you pass `-u $(id -u):$(id -g)`. Prefer rootless Docker
  or an explicit unprivileged user, and do not publish port 8093 directly to the internet.
- **Model and pack integrity.** A pack's `kindi_pack.json` and tensor files are trusted input. Do not
  load packs from untrusted sources; the converter and verifier are not a sandbox.

## Supported versions

Security fixes target the current 0.5.x release. Older releases and unmodified forks may not receive
fixes. This policy does not promise a bounty or a response time.
