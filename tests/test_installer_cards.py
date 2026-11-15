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
FUNCS = ("card_tc", "card_class", "pair_ok", "pair_fits", "card_models", "card_k", "profile_of")


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
for m in 27b nemotron; do if pair_fits 0 1 "$m"; then echo "$m=yes"; else echo "$m=no"; fi; done
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


def test_single_m5000_is_the_maxwell_8gb_class():
    info = one_card("Quadro M5000", 8192, "5.2")
    if info is None:
        return
    assert info["class"] == "maxwell", info
    assert info["models"] == "9b nemotron 4b 2b", info


def test_single_p4000_uses_the_pascal8_64k_profile():
    info = one_card("Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["class"] == "pascal8", info
    assert info["models"] == "27b lightning nemotron 9b 4b 2b", info
    profile_info = profile(["Quadro P4000"], [8192], ["6.1"], model="27b", cards="0")
    if profile_info is None:
        return
    assert profile_info["profile"] == "pascal8" and profile_info["ctx"] == "65536", profile_info


def test_single_rtx_4000_uses_the_turing_tensor_core_profile():
    info = one_card("Quadro RTX 4000", 8192, "7.5")
    if info is None:
        return
    assert info["class"] == "turing", info
    assert info["models"] == "27b lightning nemotron 9b 4b 2b", info
    profile_info = profile(["Quadro RTX 4000"], [8192], ["7.5"], model="27b", cards="0")
    if profile_info is None:
        return
    assert profile_info["profile"] == "turing" and profile_info["ctx"] == "65536", profile_info
    for flag in ("KINDI_CARDS=1", "KINDI_TC=1", "KINDI_ATT_TC=1", "KINDI_SPEC=draft2b",
                 "CUDA_VISIBLE_DEVICES=0"):
        assert flag in profile_info["env"], (flag, profile_info)


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


def test_two_rtx_4000_cards_use_the_plain_dual_profile():
    info = profile(["Quadro RTX 4000"] * 2, [8192, 8192], ["7.5", "7.5"], model="27b")
    if info is None:
        return
    assert info["profile"] == "k1200x2" and info["ctx"] == "24576", info
    assert "KINDI_CARDS=2" in info["env"], info
    assert "KINDI_TC=1" not in info["env"], info  # the Turing backend is one card only


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


def test_kepler_does_not_pair_even_with_a_supported_card():
    # Kepler remains a single-card profile; the two-card plan engine has no Kepler record.
    info = pair("Quadro K5200", 8192, "3.5", "Quadro M4000", 8192, "5.2")
    if info is None:
        return
    assert info["class0"] == "kepler", info
    assert info["pair"] == "no", info


def test_turing_and_ada_cards_pair_on_the_plain_engine():
    info = pair("Quadro RTX 4000", 8192, "7.5", "NVIDIA GeForce RTX 4060 Ti", 8188, "8.9")
    if info is None:
        return
    assert info["class0"] == "turing" and info["class1"] == "tc", info
    assert info["pair"] == "yes", info
    info = profile(["Quadro RTX 4000", "NVIDIA GeForce RTX 4060 Ti"], [8192, 8188], ["7.5", "8.9"])
    if info is None:
        return
    assert info["profile"] == "k1200x2", info
    assert "KINDI_TC=1" not in info["env"], info


def test_p106_and_p4000_pair_on_the_plain_engine():
    info = pair("NVIDIA P106-100", 6077, "6.1", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["class0"] == "maxwell" and info["class1"] == "pascal8", info
    assert info["pair"] == "yes", info
    info = profile(["NVIDIA P106-100", "Quadro P4000"], [6077, 8192], ["6.1", "6.1"])
    if info is None:
        return
    assert info["profile"] == "k1200x2", info
    assert "KINDI_G5=0" not in info["env"], info
    assert "KINDI_TC=1" not in info["env"], info


def test_m5000_and_p4000_pair_with_the_two_bit_fallback():
    info = pair("Quadro M5000", 8192, "5.2", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["class0"] == "maxwell" and info["class1"] == "pascal8", info
    assert info["pair"] == "yes", info
    profile_info = profile(["Quadro M5000", "Quadro P4000"], [8192, 8192], ["5.2", "6.1"])
    if profile_info is None:
        return
    assert profile_info["profile"] == "k1200x2", profile_info
    assert "KINDI_G5=0" in profile_info["env"], profile_info


def test_m4000_and_p4000_pair_with_the_two_bit_fallback():
    info = pair("Quadro M4000", 8192, "5.2", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["class0"] == "maxwell" and info["class1"] == "pascal8", info
    assert info["pair"] == "yes", info
    info = profile(["Quadro M4000", "Quadro P4000"], [8192, 8192], ["5.2", "6.1"])
    if info is None:
        return
    assert info["profile"] == "k1200x2", info
    assert "KINDI_G5=0" in info["env"], info
    assert "KINDI_TC=1" not in info["env"], info


def test_mixed_pair_still_needs_four_gb_each():
    info = pair("NVIDIA GeForce GT 1030", 2048, "6.1", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["pair"] == "no", info


def test_profile_of_refuses_a_three_card_split():
    # Bonsai/Gemma tensor parallel is a two-card engine; --gpu 0,1,2 means three independent servers.
    got = run_case(table(["Quadro P4000"] * 3, [8192] * 3, ["6.1"] * 3) + """
SV_B=(cuda); SV_M=(27b); SV_C=("0,1,2")
CTX=auto; MEMORY=""; THREADS=""; CORES=16; RAM_MB=65536
AGFX=(); AMEM=(); MTPF=""; XPACKS=(); ENV=(); NEED_MTP=0
mtp_ready() { return 1; }
die() { echo "die: $*" >&2; return 1; }
profile_of 0
""")
    if got is None:
        return
    code, _out, err = got
    assert code != 0, err
    assert "at most two cards" in err, err


def test_unequal_pair_uses_the_proportional_tensor_split():
    # 6 GB + 8 GB: the pair is offered, and the installer switches the engine to the uneven
    # proportional layout instead of the historical equal split.
    info = pair("NVIDIA P106-100", 6077, "6.1", "Quadro P4000", 8192, "6.1")
    if info is None:
        return
    assert info["pair"] == "yes" and info["27b"] == "yes", info
    prof = profile(["NVIDIA P106-100", "Quadro P4000"], [6077, 8192], ["6.1", "6.1"])
    assert prof["profile"] == "k1200x2", prof
    assert "KINDI_TP_SPLIT=auto" in prof["env"], prof
    # Equal-capacity pairs do not opt in to the proportional layout.
    same = profile(["NVIDIA GeForce RTX 4060 Ti"] * 2, [8188, 8188], ["8.9", "8.9"])
    assert "KINDI_TP_SPLIT=auto" not in same["env"], same


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


def test_single_8gb_ada_uses_the_tc_profile_with_speculation():
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
    # 0.6: the tc 27B profile always drafts (DSpine when its files are present, else the 2B drafter
    # with TMAX=4); this harness has no dspine files, so the 2B drafter env must be set
    assert "KINDI_SPEC=auto" in info["env"], info
    assert "KINDI_SPEC_K=3" in info["env"] and "KINDI_TC_TMAX=4" in info["env"], info


def test_pascal_maxwell_turing_rig_classifies_every_card():
    # the user's rig: Quadro P4000 (Pascal 6.1) + Quadro M4000 (Maxwell 5.2) + Quadro RTX 4000 (Turing 7.5)
    names = ["Quadro P4000", "Quadro M4000", "Quadro RTX 4000"]
    info = fields(names, [8192, 8192, 8192], ["6.1", "5.2", "7.5"], """
echo "class0=$(card_class 0)"
echo "class1=$(card_class 1)"
echo "class2=$(card_class 2)"
echo "models0=$(card_models 0)"
echo "models1=$(card_models 1)"
echo "models2=$(card_models 2)"
for p in "0 1" "0 2" "1 2"; do set -- $p; if pair_ok "$1" "$2"; then echo "pair$1$2=yes"; else echo "pair$1$2=no"; fi; done
""")
    if info is None:
        return
    assert (info["class0"], info["class1"], info["class2"]) == ("pascal8", "maxwell", "turing"), info
    assert info["pair01"] == info["pair02"] == info["pair12"] == "yes", info


def test_the_rig_pair_setups_fit_and_the_profiles_match():
    # every pairing of the rig fits a 27b split on 8 GB cards; the mixed Pascal/Maxwell pair
    # forces KINDI_G5=0 (2-bit on both), while Turing+Pascal keeps the 5-trit path
    info = pair("Quadro P4000", 8192, "6.1", "Quadro M4000", 8192, "5.2")
    if info is None:
        return
    assert info["27b"] == "yes" and info["nemotron"] == "yes", info
    prof = profile(["Quadro P4000", "Quadro M4000"], [8192, 8192], ["6.1", "5.2"])
    assert prof["profile"] == "k1200x2" and "KINDI_G5=0" in prof["env"], prof
    prof2 = profile(["Quadro RTX 4000", "Quadro P4000"], [8192, 8192], ["7.5", "6.1"])
    assert prof2["profile"] == "k1200x2" and "KINDI_G5=0" not in prof2["env"], prof2


# two sub-3.8GB cards: Nemotron (whose tensor-parallel requirement is 2,400 MiB per card) is offered,
# the 27B (3,800 MiB) is not -- the smaller offer remains usable
def test_pair_fits_offers_nemotron_but_not_27b_on_smaller_pairs():
    info = pair("Quadro P1000", 2500, "6.1", "Quadro P1000", 2500, "6.1")
    if info is None or info.get("class0") != "maxwell":
        return
    assert info["pair"] == "yes", info
    assert info["27b"] == "no" and info["nemotron"] == "yes", info


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
