# SPDX-License-Identifier: AGPL-3.0-only
"""The CP1 batch arena is one process-global allocation, so its reuse key is a safety property.

The 2026-11-26 audit's HIGH finding: `batch_prepare` in `kindi/kcpu.c` reused the arena whenever
the row count, width and context matched, but it sizes `B.soff`/`B.SNAP` (and a dozen scratch rows)
from the *whole* geometry - layer count, head counts, the per-layer GDN state width, the vocabulary.
`kc_step_batch` then copies each GDN layer's state at the offsets of whoever allocated the arena:

    memcpy(B.SNAP + B.soff[il], L->M, mb); memcpy(... + mb, L->conv, cb);

A second Model with the same width and context and a larger layer count or per-layer state takes
those offsets past the end of `B.SNAP` - a silent out-of-bounds write, not an error. Two same-width
engines in one process is the reachable case: a verifier beside a second engine, or a server holding
two packs. (A recycled heap address for a freed Model makes the pointer alone insufficient too.)

These tests pin the reuse contract with two synthetic Models and no pack load; the arena probes
(`kc_batch_prepare`, `kc_batch_reuse_ok`, `kc_batch_release`) exist for exactly this.
"""
import ctypes
import os
import shutil
from pathlib import Path

import pytest

from kindi import cpu

SRC = Path(__file__).resolve().parents[1] / "kindi" / "kcpu.c"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """The CPU library built from this source, with the arena probes bound."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("arena")
    cc = cpu._find_cc()
    if not os.path.isabs(cc) and not shutil.which(cc):
        pytest.skip("no compiler to build the CPU library")
    if os.name == "nt":
        mp.setenv("KINDI_CC", cc)
    mp.setenv("KINDI_CPU_CACHE", str(tmp))
    isa = "avx2-fma-f16c" if cpu.cpu_tier() in ("avx2-fma-f16c", "unknown") else "avx2-f16c-no-fma"
    mp.setenv("KINDI_CPU_ISA", isa)
    cpu._build(str(SRC), str(cpu._so_path(str(tmp), isa)), isa=isa)
    library = cpu._lib()
    try:
        library.kc_batch_prepare.argtypes = [ctypes.POINTER(cpu.Model), ctypes.c_int]
        library.kc_batch_prepare.restype = ctypes.c_int
        library.kc_batch_reuse_ok.argtypes = [ctypes.POINTER(cpu.Model), ctypes.c_int]
        library.kc_batch_reuse_ok.restype = ctypes.c_int
        library.kc_batch_release.argtypes = []
        library.kc_batch_release.restype = ctypes.c_int
        library.kc_state_len.argtypes = [ctypes.POINTER(cpu.Model)]
        library.kc_state_len.restype = ctypes.c_int
    except AttributeError:
        mp.undo()
        pytest.skip("this CPU library predates the arena probes")
    yield library
    mp.undo()


@pytest.fixture(autouse=True)
def _fresh_arena(lib):
    """One arena per test: the module-scoped library keeps it alive across tests, and a leftover
    arena from a test with the same geometry would make a mutation look safe."""
    lib.kc_batch_release()
    yield
    lib.kc_batch_release()


def _model(nl, d=64, nff=64, nh=2, nhkv=2, nv=8, hk=4, cc=32, ctx=8, vocab=16, attn_every=4,
           alloc=None):
    """A tiny synthetic Model: no matrices, just the geometry the arena sizes itself from.

    Keeps its own Layer array alive through the returned tuple; `batch_prepare` reads only `is_attn`
    and `gate.R` per layer, plus `head.R` for the vocabulary. `alloc` sizes the Layer array larger
    than `NL`, so a test that raises `NL` later reads layers it really allocated.
    """
    layers = (cpu.Layer * (alloc or nl))()
    for i in range(nl):
        layers[i].is_attn = 1 if attn_every and (i + 1) % attn_every == 0 else 0
        layers[i].gate.R = 4
    m = cpu.Model()
    for name, value in (("D", d), ("NL", nl), ("NFF", nff), ("NH", nh), ("NHKV", nhkv),
                        ("HD", d // 8), ("NK", 2), ("HK", hk), ("NV", nv), ("CONV_CH", cc),
                        ("HB", 1), ("ctx", ctx), ("vocab", vocab)):
        setattr(m, name, value)
    m.L = ctypes.cast(layers, ctypes.POINTER(cpu.Layer))
    m.head.R = vocab
    return m, layers


def test_two_live_engines_never_share_the_arena(lib):
    """Two Models of the same width and context are two addresses, so the pointer already separates
    them; that is what makes the arena usable at all for a draft/verify pair."""
    a, keep_a = _model(2)
    b, keep_b = _model(3)                    # same D/ctx, one more GDN layer
    assert ctypes.addressof(a) != ctypes.addressof(b)
    assert lib.kc_batch_prepare(ctypes.byref(a), 5) == 1
    assert lib.kc_batch_reuse_ok(ctypes.byref(a), 5) == 1, "the arena it was built for must reuse"
    assert lib.kc_batch_reuse_ok(ctypes.byref(b), 5) == 0


def test_a_recycled_address_with_a_different_layer_count_does_not_reuse(lib):
    """The HIGH finding. A freed Engine hands its Model address back to the allocator, so the next
    Engine can be a *different* model at the *same* address - exactly what the arena sees here:
    one address, geometry changed under it. Layer count is part of the key today; keep it."""
    m, keep = _model(2, alloc=3)             # room for the third layer the mutation claims
    assert lib.kc_batch_prepare(ctypes.byref(m), 5) == 1
    before = lib.kc_state_len(ctypes.byref(m))
    m.NL = 3                                 # the arena's SNAP/soff sized 2 layers, not 3
    assert lib.kc_state_len(ctypes.byref(m)) > before
    assert lib.kc_batch_reuse_ok(ctypes.byref(m), 5) == 0, (
        "a batch would memcpy the third GDN layer's state at offsets from a 2-layer table, past "
        "the end of B.SNAP")


def test_a_recycled_address_with_a_wider_state_does_not_reuse(lib):
    """Same address, same layer count, wider SSM: every SNAP block grows, and the old rule (rows,
    width, context) accepted it - the arena then holds too few floats per layer."""
    m, keep = _model(3, nv=8)
    assert lib.kc_batch_prepare(ctypes.byref(m), 5) == 1
    before = lib.kc_state_len(ctypes.byref(m))
    m.NV = 16
    assert lib.kc_state_len(ctypes.byref(m)) > before
    assert lib.kc_batch_reuse_ok(ctypes.byref(m), 5) == 0


def test_a_recycled_address_with_another_geometry_does_not_reuse(lib):
    """Every field batch_prepare() sizes from: head width (MM/QN/ATT rows), the vocabulary (LOG),
    the FFN width, the conv/GDN widths."""
    for change, value in (("head.R", 64), ("NH", 4), ("NFF", 128), ("CONV_CH", 64), ("HK", 8),
                          ("HD", 16), ("NK", 8), ("NHKV", 4), ("HB", 2)):
        m, keep = _model(3)
        assert lib.kc_batch_prepare(ctypes.byref(m), 5) == 1
        target, _, field = change.partition(".")
        if field:
            setattr(getattr(m, target), field, value)     # the head rows size B.LOG/B.vocab
        else:
            setattr(m, change, value)
        assert lib.kc_batch_reuse_ok(ctypes.byref(m), 5) == 0, (
            "%s changed under the arena; a batch would index scratch sized for the old value" % change)


def test_more_rows_than_allocated_is_never_reused(lib):
    """`B.nt >= nt` is the one part of the old key that was right; keep it."""
    m, keep = _model(3)
    assert lib.kc_batch_prepare(ctypes.byref(m), 2) == 1
    assert lib.kc_batch_reuse_ok(ctypes.byref(m), 2) == 1
    assert lib.kc_batch_reuse_ok(ctypes.byref(m), 8) == 0


def test_release_frees_the_arena_and_the_next_prepare_rebuilds_it(lib):
    """Engine teardown calls kc_batch_release; the arena must come back for whoever batches next."""
    a, keep_a = _model(2)
    b, keep_b = _model(3)
    assert lib.kc_batch_prepare(ctypes.byref(a), 5) == 1
    assert lib.kc_batch_release() == 0
    assert lib.kc_batch_reuse_ok(ctypes.byref(a), 5) == 0, "a released arena fits nobody"
    assert lib.kc_batch_release() == 0, "release is idempotent"
    assert lib.kc_batch_prepare(ctypes.byref(b), 5) == 1
    assert lib.kc_batch_reuse_ok(ctypes.byref(b), 5) == 1
    assert lib.kc_batch_prepare(ctypes.byref(a), 5) == 1, "and it is re-sized for the older model"
