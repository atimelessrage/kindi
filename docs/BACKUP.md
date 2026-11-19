# Offsite backup

The repository, benchmark evidence and claims live on one volume. Pushing is not part of this
checkout's workflow (see [SECURITY.md](SECURITY.md)), so a second copy is an owner action, not a
git operation.

## Monthly bundle

```sh
python tools/backup_bundle.py --out /media/<second-medium>/kindi-backups
```

The tool writes `kindi-<YYYYMMDD-HHMMSS>.bundle` with `git bundle create --all`, runs
`git bundle verify`, and writes a `.sha256` sidecar next to it. It refuses to overwrite an
existing bundle and warns when the output directory is on the repository's own volume. Copy the
bundle and its sidecar to a second physical medium or an encrypted cloud folder; a bundle on the
same disk does not survive that disk failing.

## Restore

```sh
git bundle verify kindi-20261119-120000.bundle
git clone kindi-20261119-120000.bundle kindi-restore
```

A clone from the bundle contains every branch that was in `--all`, including the archival
branches. Restore into a new directory; do not clone over an existing checkout.

## Sensitivity

`--all` includes every local ref, including the archival `wip/` branches and the rewritten
history. The repository's secret-history baseline records legacy hits and is not an endorsement;
treat the bundle as sensitive material and store it encrypted. See [SECURITY.md](SECURITY.md) for
the redaction and push policy.
