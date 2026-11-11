"""Fake-GPU tests for install.sh's card classification and two-card pairing.

The installer reads nvidia-smi into GIDX/GNAME/GMEM/GCC. These tests extract the installer's own
card functions (so the test cannot drift from install.sh) and run them in bash with a fake GPU
table: no NVIDIA driver, model pack or Linux kernel interface is needed.

    python tests/test_installer_cards.py
    python -m pytest tests/test_installer_cards.py -q

KINDI_INSTALLER points the tests at another copy, which is how the old-vs-fixed check is run.
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
FUNCS = ("card_tc", "card_class", "pair_ok", "card_models", "card_k", "profile_of")


def installer_text():
    return INSTALLER.read_text(encoding="utf-8")


def function_block(name):
    """The installer's own `name() { ... }`, by name."""
    lines = installer_text().splitlines()
    pat = re.compile(r"^%s\(\)\s*\{" % re.escape(name))
    start = next((i for i, line in enumerate(lines) if pat.match(line)), None)
    assert start is not None, "no %s() in %s" % (name, INSTALLER)
    end = next((i for i in range(start, len(lines)) if lines[i] == "}"), None)
    assert end is not None, "unterminated %s() in %s" % (name, INSTALLER)
    return "\n".join(lines[start:end + 1])


def run_case(body):
    """Run a bash body after the extracted card functions; None when this host has no bash."""
    bash = shutil.which("bash")
    if bash is None:
        return None
    script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              + "\n\n".join(function_block(n) for n in FUNCS) + "\n\n" + body)
    d = tempfile.mkdtemp(prefix="kindi-cards-")
    try:
        path = os.path.join(d, "case.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        proc = subprocess.run([bash, path], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60)
        return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def table(names, mems, ccs):
    """A fake nvidia-smi table: the arrays and count the installer fills."""
    n = len(names)
    return ('NG=%d; GIDX=(%s); GNAME=(%s); GMEM=(%s); GCC=(%s)\n'
            % (n, " ".join(str(i) for i in range(n)),
               " ".join('"%s"' % name for name in names),
               " ".join(str(m) for m in mems),
               " ".join(str(c) for c in ccs)))


def fields(names, mems, ccs, body):
    got = run_case(table(names, mems, ccs) + body)
    if got is None:
        return None
    code, out, err = got
    assert code == 0, err or out
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def one_card(name, mem, cc):
    return fields([name], [mem], [cc], """
echo "class=$(card_class 0)"
echo "models=$(card_models 0)"
""")


def pair(name_a, mem_a, cc_a, name_b, mem_b, cc_b):
    return fields([name_a, name_b], [mem_a, mem_b], [cc_a, cc_b], """
echo "class0=$(card_class 0)"
echo "class1=$(card_class 1)"
if pair_ok 0 1; then echo "pair=yes"; else echo "pair=no"; fi
""")


def profile(names, mems, ccs, model="27b", cards="0,1"):
    return fields(names, mems, ccs, """
SV_B=(cuda); SV_M=(%s); SV_C=("%s")
CTX=auto; MEMORY=""; THREADS=""; CORES=16; RAM_MB=65536
AGFX=(); AMEM=(); MTPF=""; XPACKS=(); ENV=(); NEED_MTP=0
mtp_ready() { return 1; }
die() { echo "die: $*" >&2; return 1; }
profile_of 0
echo "profile=$PROFILE"
echo "ctx=$C"
echo "env=${ENV[*]}"
""" % (model, cards))


def test_rtx_4060_ti_8gb_is_tensor_core():
    info = one_card("NVIDIA GeForce RTX 4060 Ti", 8188, "8.9")
    if info is None:
        return
    assert info["class"] == "tc", info
    assert "27b" in info["models"].split(), info


def test_rtx_5060_ti_16gb_is_tensor_core():
    info = one_card("NVIDIA GeForce RTX 5060 Ti", 16380, "12.0")
    if info is None:
        return
    assert info["class"] == "tc", info
    assert "27b" in info["models"].split(), info


def test_two_p106_cards_pair():
    # 6 GB Pascal is the maxwell memory class, but two of them are a valid 27B pair (12 GB total)
    info = pair("NVIDIA P106-100", 6077, "6.1", "NVIDIA P106-100", 6077, "6.1")
    if info is None:
        return
    assert info["class0"] == info["class1"] == "maxwell", info
    assert info["pair"] == "yes", info


def test_single_p106_6gb_is_offered_2b_4b_9b_but_not_27b():
    info = one_card("NVIDIA P106-100", 6077, "6.1")
    if info is None:
        return
    assert info["class"] == "maxwell", info
    models = info["models"].split()
    assert "9b" in models and "4b" in models and "2b" in models, info
    assert "27b" not in models, info


def test_single_p106_6gb_uses_the_24k_plain_profile():
    info = profile(["NVIDIA P106-100"], [6077], ["6.1"], model="9b", cards="0")
    if info is None:
        return
    assert info["profile"] == "maxwell", info
    assert info["ctx"] == "24576", info
    assert "CUDA_VISIBLE_DEVICES=0" in info["env"], info
    assert "KINDI_BLOCKING_SYNC=0" in info["env"], info
    assert "KINDI_TC=1" not in info["env"], info


def test_two_p106_cards_use_the_27b_dual_profile_without_tensor_cores():
    info = profile(["NVIDIA P106-100"] * 2, [6077, 6077], ["6.1", "6.1"], model="27b")
    if info is None:
        return
    assert info["profile"] == "k1200x2", info
    assert info["ctx"] == "24576", info
    for flag in ("KINDI_CARDS=2", "KINDI_REPEAT_STOP=1", "KINDI_CHECKPOINT_BUDGET_MB=4096"):
        assert flag in info["env"], (flag, info)
    assert "KINDI_TC=1" not in info["env"], info


def test_two_rtx_4060_ti_cards_pair():
    info = pair("NVIDIA GeForce RTX 4060 Ti", 8188, "8.9",
                "NVIDIA GeForce RTX 4060 Ti", 8188, "8.9")
    if info is None:
        return
    assert info["pair"] == "yes", info


def test_two_quadro_rtx_4000_cards_pair():
    info = pair("Quadro RTX 4000", 8192, "7.5", "Quadro RTX 4000", 8192, "7.5")
    if info is None:
        return
    assert info["pair"] == "yes", info


def test_two_maxwell_4gb_cards_still_pair():
    info = pair("Quadro K1200", 4096, "5.0", "Quadro K1200", 4096, "5.0")
    if info is None:
        return
    assert info["class0"] == "maxwell" and info["pair"] == "yes", info


def test_two_gt_1030_cards_do_not_pair():
    info = pair("NVIDIA GeForce GT 1030", 2048, "6.1", "NVIDIA GeForce GT 1030", 2048, "6.1")
    if info is None:
        return
    assert info["pair"] == "no", info


def test_different_card_classes_do_not_pair():
    info = pair("NVIDIA GeForce RTX 4060 Ti", 8188, "8.9", "Quadro RTX 4000", 8192, "7.5")
    if info is None:
        return
    assert info["pair"] == "no", info


def test_p106_and_p4000_do_not_pair():
    # Both are sm_61 Pascal, but the 6 GB P106 is in the maxwell memory class and the
    # 8 GB P4000 is pascal8; the documented policy pairs only cards of the same class.
    info = pair("NVIDIA P106-100", 6077, "6.1", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["class0"] == "maxwell", info
    assert info["class1"] == "pascal8", info
    assert info["pair"] == "no", info


def test_tc2_profile_for_two_8gb_ada_cards():
    info = profile(["NVIDIA GeForce RTX 4060 Ti"] * 2, [8188, 8188], ["8.9", "8.9"])
    if info is None:
        return
    assert info["profile"] == "tc2", info
    assert info["ctx"] == "24576", info
    for flag in ("KINDI_CARDS=2", "KINDI_TC=1", "KINDI_SPEC=0"):
        assert flag in info["env"], (flag, info)


def test_tc2_context_follows_the_smaller_card():
    info = profile(["NVIDIA GeForce RTX 5060 Ti", "NVIDIA GeForce RTX 5060 Ti"],
                   [16380, 16380], ["12.0", "12.0"])
    if info is None:
        return
    assert info["profile"] == "tc2" and info["ctx"] == "65536", info
    small = profile(["NVIDIA GeForce RTX 4060 Ti", "NVIDIA GeForce RTX 5060 Ti"],
                    [8188, 16380], ["8.9", "12.0"])
    if small is None:
        return
    assert small["profile"] == "tc2" and small["ctx"] == "24576", small


def test_legacy_dual_profile_has_no_tensor_cores():
    info = profile(["Quadro K1200"] * 2, [4096, 4096], ["5.0", "5.0"])
    if info is None:
        return
    assert info["profile"] == "k1200x2", info
    assert "KINDI_TC=1" not in info["env"], info


def test_single_8gb_ada_uses_the_tc_profile_without_the_2b_drafter():
    info = fields(["NVIDIA GeForce RTX 4060 Ti"], [8188], ["8.9"], """
SV_B=(cuda); SV_M=(27b); SV_C=("0")
CTX=auto; MEMORY=""; THREADS=""; CORES=16; RAM_MB=65536
AGFX=(); AMEM=(); MTPF=""; XPACKS=(); ENV=(); NEED_MTP=0
mtp_ready() { return 1; }
die() { echo "die: $*" >&2; return 1; }
profile_of 0
echo "profile=$PROFILE"
echo "ctx=$C"
echo "env=${ENV[*]}"
""")
    if info is None:
        return
    assert info["profile"] == "tc" and info["ctx"] == "24576", info
    assert "KINDI_TC=1" in info["env"], info
    assert "KINDI_SPEC" not in info["env"], info      # no room for the 2B drafter next to the 27B


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
