"""The offsite backup tool must create, verify and hash a real bundle."""
import subprocess

from tools import backup_bundle


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo)] + list(args), check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "one")
    return repo


def test_bundle_is_created_verified_and_hashed(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "backups"
    assert backup_bundle.main(["--repo", str(repo), "--out", str(out),
                               "--timestamp", "20261119-120000"]) == 0
    bundle = out / "kindi-20261119-120000.bundle"
    assert bundle.is_file()
    assert subprocess.run(["git", "-C", str(repo), "bundle", "verify", str(bundle)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    sidecar = out / "kindi-20261119-120000.bundle.sha256"
    assert backup_bundle.sha256(bundle) in sidecar.read_text(encoding="utf-8")


def test_existing_bundle_is_not_overwritten(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "backups"
    args = ["--repo", str(repo), "--out", str(out), "--timestamp", "20261119-120000"]
    assert backup_bundle.main(args) == 0
    assert backup_bundle.main(args) == 2


def test_dry_run_writes_nothing(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "backups"
    assert backup_bundle.main(["--repo", str(repo), "--out", str(out),
                               "--timestamp", "20261119-120000", "--dry-run"]) == 0
    assert not out.exists()
