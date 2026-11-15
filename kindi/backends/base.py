# SPDX-License-Identifier: AGPL-3.0-only
"""Backend plugin contract for Kindi (T4.1).

A backend is the boundary between the OpenAI-compatible API surface and one engine
implementation. This module is deliberately dependency-free and importable on a
CPU-only host: no CuPy, no FastAPI, no model pack and no GPU are needed to define or
test the contract.

Compatibility note: the CUDA image still runs Python 3.6, so this module avoids
dataclasses, typing.Protocol, walrus operators, positional-only parameters and
``from __future__ import annotations``. It uses abc.ABC and namedtuple instead.
"""
import abc
import collections
import threading


class CancelToken(object):
    """A cancellation flag a backend checks during prompt processing and decode.

    The HTTP layer owns the threading.Event; backends only poll it. ``is_set`` is the
    single method backends depend on, so a plain threading.Event also satisfies this
    contract.
    """

    def __init__(self, event=None):
        self._event = event if event is not None else threading.Event()

    def is_set(self):
        return self._event.is_set()

    def set(self):
        self._event.set()


class BackendCapabilities(object):
    """What a backend can do, in the vocabulary the API layer uses to route work."""

    def __init__(self, name, model_ids=(), architectures=(), devices=(), max_context=0,
                 supports_thinking=False, supports_tools=False, tool_format=None,
                 supports_speculation=False, supports_token_filter=False,
                 cache_kind="none", max_concurrent_sequences=1,
                 deterministic_greedy=False):
        self.name = name
        self.model_ids = tuple(model_ids or ())
        self.architectures = tuple(architectures or ())
        self.devices = tuple(devices or ())
        self.max_context = int(max_context or 0)
        self.supports_thinking = bool(supports_thinking)
        self.supports_tools = bool(supports_tools)
        self.tool_format = tool_format
        self.supports_speculation = bool(supports_speculation)
        self.supports_token_filter = bool(supports_token_filter)
        self.cache_kind = cache_kind or "none"
        self.max_concurrent_sequences = max(1, int(max_concurrent_sequences or 1))
        self.deterministic_greedy = bool(deterministic_greedy)

    def to_dict(self):
        return {
            "name": self.name,
            "model_ids": list(self.model_ids),
            "architectures": list(self.architectures),
            "devices": list(self.devices),
            "max_context": self.max_context,
            "supports_thinking": self.supports_thinking,
            "supports_tools": self.supports_tools,
            "tool_format": self.tool_format,
            "supports_speculation": self.supports_speculation,
            "supports_token_filter": self.supports_token_filter,
            "cache_kind": self.cache_kind,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "deterministic_greedy": self.deterministic_greedy,
        }


# Sampling parameters as the API layer passes them to a backend. ``token_filter`` is
# part of the contract now (T5.4 implements constrained decoding) so the signature
# does not have to change a second time. It is a callable that maps the current
# generation state to the set of token ids allowed at the next step, or None.
SamplingParams = collections.namedtuple(
    "SamplingParams",
    "max_tokens temperature top_p top_k min_p stop seed token_filter",
)
SamplingParams.__new__.__defaults__ = (512, 0.0, 1.0, 0, 0.0, (), None, None)


# One yielded item from Backend.generate. ``kind`` is "token" for a generated piece,
# "reasoning" for a thinking piece and "done" for the final event. The done event
# carries ``finish_reason`` (stop, length, tool_calls, cancelled) and any timing and
# speculation statistics the backend measured.
GenerationEvent = collections.namedtuple(
    "GenerationEvent",
    "kind token_id text logprobs finish_reason timing drafted accepted",
)
GenerationEvent.__new__.__defaults__ = (None, "", None, None, None, 0, 0)


class Backend(object, metaclass=abc.ABCMeta):
    """The contract every Kindi backend implements.

    Subclasses must set ``capabilities`` (a BackendCapabilities) and implement the
    abstract methods. The optional hooks (park, reference_gguf, verify_prompts) have
    defaults so a minimal backend can start with lifecycle, text primitives and
    generation only.
    """

    capabilities = None

    @abc.abstractmethod
    def load(self, config):
        """Load the model named by config. Raises a clear error when unavailable."""

    @abc.abstractmethod
    def close(self):
        """Release device and host resources. Safe to call more than once."""

    @abc.abstractmethod
    def info(self):
        """A JSON-able dict describing this backend instance (feeds /health)."""

    @abc.abstractmethod
    def tokenize(self, text, add_special_tokens=True):
        """Turn text into token ids."""

    @abc.abstractmethod
    def detokenize(self, token_ids):
        """Turn token ids back into text."""

    @abc.abstractmethod
    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        """Render OpenAI-style messages to the prompt text this backend expects."""

    @abc.abstractmethod
    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Yield GenerationEvent items for one request.

        ``cancel`` is a CancelToken (or any object with is_set()). The backend must
        stop at the next decode step or prompt chunk after it is set.
        """

    @abc.abstractmethod
    def cache_stats(self):
        """A JSON-able dict describing prompt-cache state."""

    def park(self, session):
        """Park a session's state. Returns True when this backend supports it."""
        return False

    def reference_gguf(self):
        """The GGUF path used as the verification reference, or None."""
        return None

    def verify_prompts(self):
        """Prompt fixtures for generalized verification: dicts with prompt and
        expected token ids. Empty means this backend has no verify gate yet."""
        return []
