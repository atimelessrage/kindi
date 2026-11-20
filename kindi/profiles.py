"""Model-aware sampling profiles.

The publisher's guidance for the Bonsai 2 models (docs/BONSAI-2-SUPPORT.md, section 4.2):

- non-thinking: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5;
- thinking: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.0, and up to 8K output tokens.

The profiles are opt-in (``sampling_profile = publisher``), so the built-in defaults stay compatible. A
request's own values always win, and so does a key the user set in the config file, the environment or
on the command line: the profile only fills values that are still built-in defaults. No publisher
sampling profile is recorded for the Gemma 4 or Nanbeige backends; they keep the config values.
"""

PROFILES = {
    "bonsai2-": {
        False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.0, "max_tokens": 8192},
    },
}

# the config keys a profile may fill (the request-level keys; `g` in server.generate resolves those first)
SAMPLING_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "repeat_last_n",
                 "presence_penalty")


def profile_for(model_name, thinking=False):
    """The publisher values for a model id, or {} when none are recorded. The Bonsai 2B/4B/9B/27B share the
    Bonsai guidance; the thinking profile is selected by the request's enable_thinking flag."""
    name = (model_name or "").strip().lower()
    for prefix, modes in PROFILES.items():
        if name.startswith(prefix):
            return dict(modes[bool(thinking)])
    return {}


def enabled(cfg):
    return str(cfg.get("sampling_profile") or "default").lower() == "publisher"


def sampling_defaults(cfg, thinking=False):
    """The config-side sampling values of one request: the profile fills every key the user did not set
    explicitly (config.load records those in cfg["explicit"])."""
    prof = profile_for(cfg.get("model_name"), thinking) if enabled(cfg) else {}
    explicit = set(cfg.get("explicit") or ())
    return {k: prof[k] if k in prof and k not in explicit else cfg.get(k) for k in SAMPLING_KEYS}
