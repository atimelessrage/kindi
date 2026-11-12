# SPDX-License-Identifier: AGPL-3.0-only
"""The installer must verify the release source archive before extracting it.

The source-fetch block is extracted from install.sh and run with fake curl, sha256sum and tar
commands. This proves the order (download, hash, extract) and the mismatch path without needing
a network, Docker or a model pack.
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("KINDI_INSTALLER", ROOT / "install.sh"))
MARK_BEGIN = "# source-archive: begin"
MARK_END = "# source-archive: end"


def installer_text():
    return INSTALLER.read_text(encoding="utf-8").replace("\r\n", "\n")


def source_block():
    text = installer_text()
    start = text.index(MARK_BEGIN)
    end = text.index(MARK_END, start) + len(MARK_END)
    return text[start:end]


def run_block(fake_hash):
    bash = shutil.which("bash")
    if bash is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-source-")
    try:
        marker = Path(d) / "tar-called"
        script = """#!/usr/bin/env bash
set -euo pipefail
DRY=0
SOURCE=""
SOURCE_URL="https://example.invalid/kindi-v0.5.0.tar.gz"
SOURCE_SHA256="{expected}"
KINDI_SOURCE_SHA256=""
tarball="$SOURCE_URL"
DIR={dir}
DOCKERFILE="docker/Dockerfile"
SRC="$DIR/src"
TAR_MARKER={marker}
export TAR_MARKER
curl() {{ printf 'archive' > "$4"; }}
sha256sum() {{ printf '%s  %s\\n' "$FAKE_HASH" "$1"; }}
tar() {{ : > "$TAR_MARKER"; }}
run() {{ "$@"; }}
die() {{ echo "STOP: $*" >&2; exit 1; }}
{block}
""".format(expected="a" * 64, dir=shlex.quote(d), marker=shlex.quote(str(marker)),
           block=source_block())
        path = Path(d) / "case.sh"
        path.write_text(script, encoding="utf-8", newline="\n")
        env = dict(os.environ, FAKE_HASH=fake_hash)
        proc = subprocess.run([bash, str(path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60)
        return (proc.returncode, proc.stdout.decode("utf-8", "replace"),
                proc.stderr.decode("utf-8", "replace"), marker.exists(),
                (Path(d) / "src.tar.gz.part").exists())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_source_archive_has_a_pinned_fork_sha256():
    text = installer_text()
    match = re.search(r'^SOURCE_SHA256="([0-9a-f]{64})"$', text, re.MULTILINE)
    assert match, "install.sh has no pinned 64-hex SOURCE_SHA256"
    assert match.group(1) == "3ca1e79d371c87bea5471841db10b0509b90d3bd557512cf59fc83854b929a94"
    assert 'SOURCE_COMMIT="fd5414889e4acfd72b46bba3138a908b8e97c5be"' in text
    assert ('SOURCE_URL="https://github.com/atimelessrage/kindi/archive/'
            '${SOURCE_COMMIT}.tar.gz"') in text
    assert "ahtx/kindi/archive" not in text


def test_source_block_never_pipes_curl_into_tar():
    block = source_block()
    assert 'curl -fsSL "$tarball" -o "$part"' in block
    assert 'sha256sum "$part"' in block
    assert 'tar -xzf "$part"' in block
    assert "| tar" not in block
    assert block.index('sha256sum "$part"') < block.index('tar -xzf "$part"')


def test_port_probe_does_not_require_ss():
    """The installer may run on a minimal host without iproute2; it must still avoid stealing a port."""
    text = installer_text()
    start = text.index("port_free() {")
    end = text.index("\n}", start) + 2
    block = text[start:end]
    assert "command -v ss" in block
    assert "/dev/tcp/127.0.0.1/$p" in block
    assert "treating it as free" in block


def test_verify_speed_parse_has_a_fallback():
    """An older verify log may omit the speed line; pipefail must not kill the installer there."""
    text = installer_text()
    assert "grep -o '[0-9.]* tokens/s over the run' \"$VERIFY_OUT\" | awk '{print $1}' || true" in text
    assert '[ -n "$sp" ] || sp="?"' in text


def test_matching_source_archive_is_extracted():
    got = run_block("a" * 64)
    if got is None:
        return
    code, _out, err, tar_called, part_left = got
    assert code == 0, err
    assert tar_called
    assert not part_left


def test_source_archive_mismatch_is_not_extracted():
    got = run_block("b" * 64)
    if got is None:
        return
    code, _out, err, tar_called, part_left = got
    assert code != 0
    assert "STOP:" in err and "source archive sha256" in err
    assert not tar_called
    assert not part_left
