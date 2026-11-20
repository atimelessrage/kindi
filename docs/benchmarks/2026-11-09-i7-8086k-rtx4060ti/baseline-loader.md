# How the v0.4.0 baseline was made runnable on Windows

The baseline is the exact `v0.4.0` tree (`cee4488`) plus the minimum native-Windows loader change. No engine, attention, quantization, chunking, scheduling or benchmark logic was edited.

1. Copy `kindi/gpudrv.py` from current main (`2abf9ad`) into the v0.4.0 tree. It selects `nvcuda.dll` on Windows and `libcuda.so.1` elsewhere.
2. In `kindi/sched.py`, replace `cu = ctypes.CDLL("libcuda.so.1")` with `cu = gpudrv.driver()`.
3. In `kindi/chunk1.py`, replace `drv = ctypes.CDLL("libcuda.so.1")` with `from . import gpudrv` followed by `drv = gpudrv.driver()`.

`kindi/bench.py` and `kindi/engine.py` are byte-identical between `v0.4.0` and current main, so this patch only changes how the driver is opened. Original v0.4.0 cannot load its CUDA driver on native Windows without it.
