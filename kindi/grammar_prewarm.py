# SPDX-License-Identifier: AGPL-3.0-only
"""Startup prewarm for the structured-output vocabulary index, shared by the generation paths.

The token-class index is cached by the *identity* of the token-text list, so the prewarm and the
first structured request must pass the same list object. Each generation path (the dense/Gemma
engine, Lightning and Nanbeige) owns a tokenizer whose text API differs, so it implements
``grammar_token_texts()`` and inherits the cached list, the background build and the opt-out here.

``KINDI_PREWARM_GRAMMAR=0`` leaves the build to the first constrained request.
"""
import os
import threading
import time


class GrammarPrewarm(object):
    """Mixin: one cached token-text list plus the background token-class build."""

    @staticmethod
    def log(message):
        """Startup diagnostics; every engine class sets its own ``log``."""
        del message

    def grammar_token_texts(self):
        """The vocabulary as text, in id order. The subclass owns its tokenizer's API."""
        raise NotImplementedError

    def grammar_texts(self):
        """The shared token-text list, built once (the prewarm and every grammar request use it)."""
        lock = self.__dict__.get("_grammar_texts_lock")
        if lock is None:                     # a test double built without __init__
            lock = self.__dict__["_grammar_texts_lock"] = threading.Lock()
        with lock:
            texts = self.__dict__.get("_grammar_texts")
            if texts is None:
                texts = self.__dict__["_grammar_texts"] = self.grammar_token_texts()
            return texts

    def start_grammar_prewarm(self):
        """Build the vocabulary index and trie in a daemon thread; None when opted out."""
        if os.environ.get("KINDI_PREWARM_GRAMMAR", "1") == "0":
            return None
        thread = threading.Thread(target=self._prewarm_grammar, name="kindi-grammar-prewarm",
                                  daemon=True)
        thread.start()
        return thread

    def _prewarm_grammar(self):
        try:
            from .token_classes import prewarm

            started = time.time()
            prewarm(self.grammar_texts())
            self.log("structured-output vocabulary ready in %.1fs" % (time.time() - started))
        except Exception as e:  # noqa: BLE001 - prewarm is an optimization, not a startup gate
            self.log("structured-output prewarm skipped: %s" % e)
