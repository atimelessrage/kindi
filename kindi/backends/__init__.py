# SPDX-License-Identifier: AGPL-3.0-only
"""Backend plugin interface (T4.1).

Importing this package registers the fake backend and the T4.2 bonsai and
nanbeige adapters (``kindi.backends.bonsai``, ``kindi.backends.nanbeige``).
``resolve`` serves the fake backend, the adapter backends and any
externally registered backend, and says so when asked for anything else.
"""
import os

from .base import Backend, BackendCapabilities, CancelToken, GenerationEvent, SamplingParams
from .registry import (
    BackendNotFound,
    BackendUnavailable,
    backend_classes,
    get_registration,
    list_backends,
    load_external_backends,
    register_backend,
    resolve,
    unregister_backend,
)
from .fake import FakeBackend
from .bonsai import BonsaiBackend
from .nanbeige import NanbeigeBackend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BonsaiBackend",
    "NanbeigeBackend",
    "BackendNotFound",
    "BackendUnavailable",
    "CancelToken",
    "FakeBackend",
    "GenerationEvent",
    "SamplingParams",
    "backend_classes",
    "get_registration",
    "list_backends",
    "load_external_backends",
    "register_backend",
    "resolve",
    "unregister_backend",
]


def load_configured_backends():
    """Load KINDI_BACKEND_MODULES once. Safe to call repeatedly."""
    spec = os.environ.get("KINDI_BACKEND_MODULES", "")
    if not spec:
        return []
    return load_external_backends(spec)
