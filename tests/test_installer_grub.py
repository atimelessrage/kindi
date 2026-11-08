"""Fixture test for the amdgpu GRUB edit in install.sh.

install.sh adds amdgpu.cwsr_enable=0 to GRUB_CMDLINE_LINUX_DEFAULT with sed. The edit must keep
the kernel parameters that are already on that line. Kindi 0.4.0 had a literal 0x01 byte where
the sed backreference belongs, so the rewrite replaced every existing parameter with a control
character. This test runs the installer's own sed script (extracted from install.sh, so the two
cannot drift) against tests/fixtures/grub.default and checks the result.

    python tests/test_installer_grub.py          # no pytest needed
    python -m pytest tests/test_installer_grub.py -q
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("KINDI_INSTALLER", ROOT / "install.sh"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "grub.default"
OPTION = "amdgpu.cwsr_enable=0"
TARGET = "GRUB_CMDLINE_LINUX_DEFAULT="


def installer_text():
    return INSTALLER.read_text(encoding="utf-8")


def grub_sed_script(text=None):
    """The sed script that install.sh runs on the grub file for the cwsr option."""
    text = installer_text() if text is None else text
    for line in text.splitlines():
        if "sed -i" in line and OPTION in line:
            m = re.search(r"sed -i '([^']*)'", line)
            if m:
                return m.group(1)
    raise AssertionError("no 'sed -i ...' line for %s in %s" % (OPTION, INSTALLER))


def grub_value(text):
    line = next(l for l in text.splitlines() if l.startswith(TARGET))
    return line.split("=", 1)[1].strip()


def rewritten():
    """The fixture after the installer's sed, or None when this host has no sed."""
    if shutil.which("sed") is None:
        return None
    d = tempfile.mkdtemp(prefix="kindi-grub-")
    try:
        shutil.copyfile(FIXTURE, os.path.join(d, "grub"))
        subprocess.run(["sed", "-i", grub_sed_script(), "grub"], cwd=d, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(os.path.join(d, "grub"), encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


GUARD_FIRST = "grub_cmdline_value() {"
GUARD_EDIT = "cwsr_edit_grub() {"


def guarded_edit_block(text=None):
    """The command-line reader, the guard and the guarded edit, exactly as install.sh defines them.

    Extracted (as the sed script above is) so that the test cannot pass against a copy that drifted
    from the installer.
    """
    lines = (installer_text() if text is None else text).splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith(GUARD_FIRST)]
    edits = [i for i, l in enumerate(lines) if l.startswith(GUARD_EDIT)]
    assert starts and edits, "no %r / %r in %s" % (GUARD_FIRST, GUARD_EDIT, INSTALLER)
    end = next(i for i in range(edits[0], len(lines)) if lines[i] == "}")
    return "\n".join(lines[starts[0]:end + 1])


def run_guarded_edit(grub_bytes, sed_script=None):
    """Run install.sh's own guard on a copy of grub_bytes in a temp dir (never /etc).

    Returns (returncode, stderr, resulting bytes), or None when this host has no bash.
    """
    if shutil.which("bash") is None:
        return None
    block = guarded_edit_block()
    if sed_script is not None:
        real = grub_sed_script()
        assert real in block, "the extracted guard does not hold the installer's sed script"
        block = block.replace(real, sed_script)
    d = tempfile.mkdtemp(prefix="kindi-grub-guard-")
    try:
        grub = os.path.join(d, "grub")
        with open(grub, "wb") as f:
            f.write(grub_bytes)
        script = os.path.join(d, "run.sh")
        with open(script, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    'DRY=0; SUDO=""\n'
                    'die() { echo "stopped: $*" >&2; exit 1; }\n'
                    'run() { "$@"; }\n'
                    + block + "\n"
                    'cwsr_edit_grub "$1"\n')
        proc = subprocess.run([shutil.which("bash"), "run.sh", "grub"], cwd=d,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        with open(grub, "rb") as f:
            return proc.returncode, proc.stderr.decode("utf-8", "replace"), f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_installer_uses_a_backreference():
    script = grub_sed_script()
    assert "\\1" in script, "the replacement must keep the old value with the \\1 backreference"
    assert "\x01" not in script, "control byte (0x01) in the sed replacement"
    assert script.isascii()


def test_installer_has_no_control_bytes():
    data = INSTALLER.read_bytes()
    bad = sorted({b for b in data if b < 0x20 and b not in (0x09, 0x0a, 0x0d)})
    assert not bad, "install.sh contains control bytes: %r" % bad


def test_existing_kernel_parameters_survive():
    out = rewritten()
    if out is None:
        return
    value = grub_value(out)
    assert value.startswith('"') and value.endswith('"'), value
    params = value.strip('"').split()
    for wanted in ("quiet", "splash", "resume=/dev/nvme0n1p2", OPTION):
        assert wanted in params, "%r lost: %r" % (wanted, value)


def test_no_control_character_reaches_grub():
    out = rewritten()
    if out is None:
        return
    bad = [b for b in out.encode("utf-8") if b < 0x20 and b not in (0x09, 0x0a, 0x0d)]
    assert not bad, "control bytes after the rewrite: %r" % bad


def test_option_added_once_and_other_lines_unchanged():
    out = rewritten()
    if out is None:
        return
    assert out.count(OPTION) == 1
    before = FIXTURE.read_text(encoding="utf-8").splitlines()
    after = out.splitlines()
    assert len(before) == len(after)
    for old, new in zip(before, after):
        if old.startswith(TARGET):
            continue
        assert old == new, "unexpected change: %r -> %r" % (old, new)


def test_check_cwsr_uses_the_guarded_edit():
    text = installer_text()
    start = text.index("check_cwsr() {")
    body = text[start:text.index("\n}\n", start)]
    assert "cwsr_edit_grub /etc/default/grub" in body
    assert "sed -i" not in body, "the GRUB edit must go through the guarded helper"


def test_guard_accepts_a_good_rewrite():
    got = run_guarded_edit(FIXTURE.read_bytes())
    if got is None:
        return
    code, err, out = got
    assert code == 0, err
    value = grub_value(out.decode("utf-8"))
    params = value.strip('"').split()
    for wanted in ("quiet", "splash", "resume=/dev/nvme0n1p2", OPTION):
        assert wanted in params, "%r lost: %r" % (wanted, value)


def test_guard_restores_the_backup_when_a_parameter_would_be_lost():
    # the 0.4.0 bug: a literal 0x01 in place of the sed backreference wipes every parameter on the line
    buggy = grub_sed_script().replace("\\1", "\x01")
    got = run_guarded_edit(FIXTURE.read_bytes(), sed_script=buggy)
    if got is None:
        return
    code, err, out = got
    assert code != 0, "the guard accepted a rewrite that dropped every parameter"
    assert out == FIXTURE.read_bytes(), "the file was not restored from the backup"
    assert "back" in err or "restored" in err, err


def test_guard_aborts_when_the_rewrite_adds_nothing():
    # a line the sed does not match (single quotes): the edit would silently add nothing
    text = FIXTURE.read_text(encoding="utf-8").replace(
        'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash resume=/dev/nvme0n1p2"',
        "GRUB_CMDLINE_LINUX_DEFAULT='quiet splash resume=/dev/nvme0n1p2'")
    got = run_guarded_edit(text.encode("utf-8"))
    if got is None:
        return
    code, err, out = got
    assert code != 0, "the guard accepted an edit that added amdgpu.cwsr_enable=0 nowhere"
    assert out == text.encode("utf-8"), "the file was not restored from the backup"


if __name__ == "__main__":
    failed = 0
    for name in sorted(n for n in globals() if n.startswith("test_")):
        try:
            globals()[name]()
            print("PASS %s" % name)
        except Exception as e:  # noqa: BLE001 - the runner reports every test
            failed += 1
            print("FAIL %s: %s" % (name, e))
    print("%d failed" % failed if failed else "ALL PASS")
    sys.exit(1 if failed else 0)
