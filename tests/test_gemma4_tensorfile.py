# SPDX-License-Identifier: AGPL-3.0-only
"""Gemma 4 tensor uploads must not keep pinned host buffers (upstream issue #44).

``cupy.asarray`` stages the whole tensor in a pinned host buffer and leaves that buffer in CuPy's
pool, so a loaded model keeps host RAM it never uses again.  ``TensorFile`` now copies with
``empty`` + ``set``, like ``engine._upload``, and returns the pool after the resident load.
"""
import collections
import types

import numpy as np

from kindi import gemma4


class _NullStream(object):
    def synchronize(self):
        pass


class _Stream(object):
    null = _NullStream()


class _Device(object):
    def __init__(self, card=0):
        self.id = card


class _Array(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.data = None

    def set(self, data, stream=None):
        assert stream is _Stream.null
        self.data = np.array(data, copy=True)


class _Pool(object):
    def __init__(self):
        self.freed = 0

    def free_all_blocks(self):
        self.freed += 1


class _FakeCupy(object):
    """The small CuPy surface TensorFile uses, with asarray() as a tripwire."""

    def __init__(self):
        self.pool = _Pool()
        self.uploads = []
        self.cuda = types.SimpleNamespace(
            Stream=_Stream,
            Device=_Device,
            runtime=types.SimpleNamespace(
                getDeviceProperties=lambda card: {"name": b"fake"},
                memGetInfo=lambda: (2 << 30, 4 << 30)))

    def empty(self, shape, dtype):
        value = _Array(shape, dtype)
        self.uploads.append(value)
        return value

    def asarray(self, *args, **kwargs):
        raise AssertionError("cupy.asarray stages a pinned host buffer; use empty + set")

    def get_default_pinned_memory_pool(self):
        return self.pool


def _tensor_file(raw):
    tf = object.__new__(gemma4.TensorFile)
    tf.tensors = {"w": {"shape": (len(raw),), "type": 0}}
    tf._gpu_cache = collections.OrderedDict()
    tf._gpu_cache_bytes = 0
    tf._resident = {}
    tf.host = set()
    tf.raw = lambda name: memoryview(raw)
    return tf


def test_gpu_uses_a_plain_copy_and_not_cupy_asarray():
    cp = _FakeCupy()
    tf = _tensor_file(b"\x01\x02\x03\x04")
    value = tf.gpu("w", cp)
    assert value.data.tolist() == [1, 2, 3, 4]
    assert "w" in tf._gpu_cache


def test_load_resident_uses_the_same_copy_and_returns_pinned_blocks():
    cp = _FakeCupy()
    tf = _tensor_file(b"\x05\x06")
    info = tf.load_resident(cp)
    assert tf._resident["w"].data.tolist() == [5, 6]
    assert info["weight_bytes"] == 2
    assert cp.pool.freed == 1
