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


def test_current_docs_and_images_do_not_use_old_kindi_tags():
    stale = re.compile(r"kindi:(0\.[0-3]\.[0-9]+)")
    offenders = []
    for relative in ("README.md", "docs/CARDS.md", "docs/DOCKER.md", "docs/AMD.md",
                     "docker/Dockerfile", "docker/Dockerfile.rocm"):
        for match in stale.finditer(_read(relative)):
            offenders.append("%s: %s" % (relative, match.group(0)))
    assert not offenders, "\n".join(offenders)


def test_amd_page_names_the_current_release():
    version = _version(r'^VERSION = "([^"]+)"', "kindi/__init__.py")
    assert "Kindi %s runs on AMD" % version in _read("docs/AMD.md")
