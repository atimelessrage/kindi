"""Run unmodified stock Kindi v0.6.1 code on native Windows.

Stock v0.6.1 hard-codes ctypes.CDLL("libcuda.so.1") in a few driver-loading
paths. Windows exposes the same CUDA driver API as nvcuda.dll. This shim only
rewrites that library name before importing stock Kindi; it does not patch or
replace any Kindi module or kernel.
"""
import ctypes
import os
import sys

if os.name == "nt":
    _cdll = ctypes.CDLL

    def _cdll_windows(name, *args, **kwargs):
        if name == "libcuda.so.1":
            name = "nvcuda.dll"
        return _cdll(name, *args, **kwargs)

    ctypes.CDLL = _cdll_windows

from kindi.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
