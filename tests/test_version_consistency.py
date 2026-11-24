"""Keep the release version and current image tags from drifting apart."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def _version(pattern, relative):
    match = re.search(pattern, _read(relative), re.MULTILINE)
    assert match, "no version in %s" % relative
    return match.group(1)


def test_package_and_installer_versions_match():
    package = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    installer = _version(r'^KINDI_VERSION="([^"]+)"', "install.sh")
    assert package == installer


def test_the_windows_installer_uses_the_package_version():
    """The 2026-11-24 audit found install.ps1 still pinning 0.7.0-fork.

    That literal is the version the script turns into the bundle download URL and
    the Programs & Features entry, so stale meant every Windows one-liner fetched
    the previous release.
    """
    package = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    windows = _version(r"^\$KindiWinVersion = '([^']+)'", "install.ps1")
    assert package == windows


def test_each_installer_points_at_its_own_release_tag():
    """A release URL may be derived from the version, never a second literal."""
    package = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    for relative in ("install.sh", "install.ps1"):
        tags = re.findall(r"releases/(?:download|tag)/v([^/'\"]+)/", _read(relative))
        assert tags, "%s names no release URL" % relative
        literal = {tag for tag in tags if not tag.startswith(("$", "{"))}
        assert literal <= {package}, "%s points at release(s) %s, expected %s" % (
            relative, sorted(literal - {package}), package)


def test_current_docs_and_images_use_the_package_version():
    version = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    tag = re.compile(r"\bkindi(?:-rocm7?)?:([0-9][0-9A-Za-z_.-]*)")
    offenders = []
    for relative in ("README.md", "docs/CARDS.md", "docs/DOCKER.md", "docs/AMD.md",
                     "docs/INSTALL.md", "docker/Dockerfile", "docker/Dockerfile.rocm",
                     "docker/Dockerfile.rocm7"):
        for match in tag.finditer(_read(relative)):
            if match.group(1) != version:
                offenders.append("%s: %s (expected %s)" % (relative, match.group(0), version))
    assert not offenders, "\n".join(offenders)


def test_readme_release_archive_matches_the_package_version():
    version = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    archives = re.findall(r"kindi-(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\.tar\.gz", _read("README.md"))
    assert archives, "README.md no longer documents a release archive"
    assert set(archives) == {version}, "README.md archive versions: %s" % sorted(set(archives))


def test_why_and_security_name_the_current_version():
    version = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    assert "Current release: Kindi %s" % version in _read("WHY.md")
    minor = ".".join(version.split(".")[:2])
    assert "current %s.x release" % minor in _read("SECURITY.md")


def test_amd_page_names_the_current_release():
    version = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    assert "Kindi %s runs on AMD" % version in _read("docs/AMD.md")
