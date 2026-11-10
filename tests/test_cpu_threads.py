"""The CPU engine's thread default: physical cores, never the hyperthread siblings.

The i7-8086K runs the 2B at 39.2 tokens/s with 6 threads and 1.3 with 12 (OpenMP active
spinning on 6 pairs), so the default must resolve to one thread per physical core, while an
explicit `threads=` or `OMP_NUM_THREADS` stays untouched.
"""
import pytest

from kindi import cpu


@pytest.fixture
def no_omp(monkeypatch):
    for k in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_cpu_list_parses_ranges_and_singles():
    assert cpu._cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert cpu._cpu_list("5") == [5]
    assert cpu._cpu_list("") == []


def test_physical_cores_counts_sibling_groups(no_omp, monkeypatch):
    monkeypatch.setattr(cpu, "_affinity", lambda: list(range(12)))
    monkeypatch.setattr(cpu, "_siblings", lambda c: [c - c % 2, c - c % 2 + 1])
    assert cpu._physical_cores() == 6


def test_physical_cores_only_counts_the_cpus_it_may_run_on(no_omp, monkeypatch):
    # one CPU of each of three pairs: three cores, not six
    monkeypatch.setattr(cpu, "_affinity", lambda: [0, 2, 4])
    monkeypatch.setattr(cpu, "_siblings", lambda c: [c, c + 1])
    assert cpu._physical_cores() == 3


def test_physical_cores_falls_back_to_cpuinfo(no_omp, monkeypatch):
    info = (
        "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n\n"
        "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 1\n\n"
        "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 0\n\n"
        "processor\t: 3\nphysical id\t: 0\ncore id\t\t: 1\n"
    )
    monkeypatch.setattr(cpu, "_affinity", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(cpu, "_read", lambda path: None if path.startswith("/sys") else info)
    assert cpu._physical_cores() == 2


def test_physical_cores_uses_the_logical_count_when_topology_is_unknown(no_omp, monkeypatch):
    monkeypatch.setattr(cpu, "_affinity", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(cpu, "_read", lambda path: None)
    assert cpu._physical_cores() == 4


def test_explicit_threads_wins_and_does_not_bind(no_omp):
    assert cpu._configure_threads(3) == 3
    assert cpu.os.environ["OMP_NUM_THREADS"] == "3"
    assert "OMP_PROC_BIND" not in cpu.os.environ
    assert "OMP_PLACES" not in cpu.os.environ


def test_existing_omp_num_threads_is_preserved(no_omp, monkeypatch):
    monkeypatch.setattr(cpu, "_physical_cores", lambda: 6)
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    assert cpu._configure_threads() == 12
    assert cpu.os.environ["OMP_NUM_THREADS"] == "12"
    assert "OMP_PROC_BIND" not in cpu.os.environ


def test_default_is_the_physical_core_count(no_omp, monkeypatch):
    monkeypatch.setattr(cpu, "_physical_cores", lambda: 6)
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 12)
    assert cpu._configure_threads() == 6
    assert cpu.os.environ["OMP_NUM_THREADS"] == "6"
    assert cpu.os.environ["OMP_PROC_BIND"] == "close"
    assert cpu.os.environ["OMP_PLACES"] == "cores"


def test_default_keeps_an_existing_binding(no_omp, monkeypatch):
    monkeypatch.setattr(cpu, "_physical_cores", lambda: 6)
    monkeypatch.setenv("OMP_PROC_BIND", "spread")
    assert cpu._configure_threads() == 6
    assert cpu.os.environ["OMP_PROC_BIND"] == "spread"
