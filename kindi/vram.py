"""Sampled-peak VRAM monitoring for `kindi bench --sample-vram`.

Device-wide `memory.used` from nvidia-smi (so the peak includes every
allocation, not just the CuPy pool), sampled on a background thread while
the benchmark runs. Free-after-load numbers alone cannot prove a
larger-than-VRAM split fits; a sampled peak can.
"""
import subprocess
import threading
import time

SMI_CMD = ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
           "--format=csv,noheader,nounits"]


def parse_smi(text):
    """Parse one nvidia-smi csv row `used, total, free` (MiB) or return None."""
    try:
        row = str(text).strip().splitlines()[0]
        used, total, free = [int(x.strip()) for x in row.split(",")[:3]]
    except (ValueError, IndexError):
        return None
    return {"used_mib": used, "total_mib": total, "free_mib": free}


def gpu_memory(runner=None):
    """Device-wide VRAM from nvidia-smi, or None when it cannot be read."""
    try:
        if runner is not None:
            out = runner(SMI_CMD)
        else:
            out = subprocess.run(SMI_CMD, capture_output=True, text=True, timeout=5)
            out = out.stdout if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return parse_smi(out) if out is not None else None


class VramMonitor(object):
    """Sample device-wide VRAM while work runs; report the peak and its delta.

    `sampler` is injectable so tests can drive the monitor without a GPU.
    """

    def __init__(self, interval=0.5, sampler=None):
        self.interval = max(0.05, float(interval))
        self.sampler = sampler or gpu_memory
        self.samples = []
        self.phase = "startup"
        self.baseline = None
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.perf_counter()

    def _sample(self, phase=None):
        mem = self.sampler()
        if mem is None:
            return None
        rec = dict(mem)
        rec["at_s"] = round(time.perf_counter() - self._t0, 3)
        rec["phase"] = phase or self.phase
        self.samples.append(rec)
        return rec

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._t0 = time.perf_counter()
        self.baseline = self._sample("startup")
        self._thread = threading.Thread(target=self._run, name="vram-monitor", daemon=True)
        self._thread.start()

    def mark(self, phase):
        self.phase = phase
        self._sample(phase)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._sample("stop")

    def summary(self):
        if not self.samples:
            return {"available": False, "interval_s": self.interval}
        peak = max(self.samples, key=lambda r: r["used_mib"])
        base = self.baseline or self.samples[0]
        return {
            "available": True,
            "source": "nvidia-smi memory.used (device-wide)",
            "interval_s": self.interval,
            "baseline_used_mib": base["used_mib"],
            "peak_used_mib": peak["used_mib"],
            "peak_delta_mib": peak["used_mib"] - base["used_mib"],
            "peak_free_mib": peak["free_mib"],
            "peak_phase": peak["phase"],
            "sample_count": len(self.samples),
        }
