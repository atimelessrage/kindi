"""Model-aware sampling profiles: publisher values, the opt-in switch and precedence."""
from kindi import profiles

BASE = {"model_name": "bonsai2-2b", "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "repeat_penalty": 1.0, "repeat_last_n": 64, "presence_penalty": 0.0, "max_tokens": 1024}


def cfg(**more):
    out = dict(BASE, sampling_profile="publisher", explicit=frozenset())
    out.update(more)
    return out


def test_publisher_non_thinking_profile():
    sd = profiles.sampling_defaults(cfg(), thinking=False)
    assert sd["temperature"] == 0.7 and sd["top_p"] == 0.8 and sd["top_k"] == 20
    assert sd["presence_penalty"] == 1.5
    assert sd["max_tokens"] == 1024              # the publisher recommends no larger non-thinking budget


def test_publisher_thinking_profile():
    sd = profiles.sampling_defaults(cfg(), thinking=True)
    assert sd["temperature"] == 1.0 and sd["top_p"] == 0.95 and sd["top_k"] == 20
    assert sd["presence_penalty"] == 1.0
    assert sd["max_tokens"] == 8192


def test_profiles_are_opt_in():
    sd = profiles.sampling_defaults(cfg(sampling_profile="default"), thinking=True)
    assert sd["temperature"] == 0.7 and sd["top_p"] == 0.8
    assert sd["presence_penalty"] == 0.0
    assert sd["max_tokens"] == 1024


def test_explicit_config_values_beat_the_profile():
    sd = profiles.sampling_defaults(cfg(explicit=frozenset({"temperature", "presence_penalty"}),
                                        temperature=0.2, presence_penalty=0.0), thinking=True)
    assert sd["temperature"] == 0.2              # set by the user
    assert sd["presence_penalty"] == 0.0
    assert sd["top_p"] == 0.95                   # still the profile
    assert sd["max_tokens"] == 8192


def test_every_bonsai_size_shares_the_profile():
    for name in ("bonsai2-2b", "bonsai2-4b", "bonsai2-9b", "bonsai2-27b", "Bonsai2-27B"):
        assert profiles.profile_for(name)["presence_penalty"] == 1.5


def test_backends_without_a_publisher_profile_fall_back():
    for name in ("gemma4", "nanbeige-4.3-33b", ""):
        assert profiles.profile_for(name) == {}
        sd = profiles.sampling_defaults(cfg(model_name=name), thinking=True)
        assert sd["temperature"] == 0.7 and sd["presence_penalty"] == 0.0
