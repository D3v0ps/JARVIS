"""Choosing a model that is actually installed.

config.yaml saying qwen3:8b on a machine where qwen3:14b was pulled is not an
exotic case - it is what happens whenever the installer's last step does not run.
JARVIS should use what is there and say so, not sit mute waiting for someone to
edit a YAML file.
"""

from __future__ import annotations

import pytest

from jarvis.core.wiring import _parameter_size, choose_installed_model


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("qwen3:14b", 14.0),
        ("qwen3:8b", 8.0),
        ("qwen3:4b", 4.0),
        ("qwen3:0.6b", 0.6),
        ("qwen3:30b-a3b", 30.0),
        ("llama3.1:8b-instruct-q4_0", 8.0),
        ("mistral", 0.0),
        ("", 0.0),
    ],
)
def test_parameter_size_is_read_from_the_tag(tag, expected):
    assert _parameter_size(tag) == expected


def test_the_configured_model_wins_when_it_is_installed():
    assert choose_installed_model("qwen3:8b", ["qwen3:8b", "qwen3:14b"], 15.9) == "qwen3:8b"


def test_the_same_family_is_preferred_over_a_stranger():
    chosen = choose_installed_model("qwen3:8b", ["llama3.1:70b", "qwen3:4b"], 15.9)
    assert chosen == "qwen3:4b"


def test_the_largest_that_fits_the_card_is_chosen():
    installed = ["qwen3:4b", "qwen3:8b", "qwen3:14b"]
    assert choose_installed_model("qwen3:32b", installed, 15.9) == "qwen3:14b"
    assert choose_installed_model("qwen3:32b", installed, 8.0) == "qwen3:8b"
    assert choose_installed_model("qwen3:32b", installed, 4.0) == "qwen3:4b"


def test_an_unknown_card_does_not_second_guess_the_user():
    """They pulled it deliberately; without a VRAM figure, trust that."""
    assert choose_installed_model("qwen3:8b", ["qwen3:30b-a3b"], None) == "qwen3:30b-a3b"


def test_anything_installed_beats_nothing_to_think_with():
    assert choose_installed_model("qwen3:8b", ["llama3.1:8b"], 15.9) == "llama3.1:8b"


def test_no_models_at_all_returns_none():
    assert choose_installed_model("qwen3:8b", [], 15.9) is None


def test_a_model_too_big_for_the_card_is_still_used_if_it_is_all_there_is():
    """Better slow than silent."""
    assert choose_installed_model("qwen3:8b", ["qwen3:70b"], 6.0) == "qwen3:70b"


def test_resolve_model_points_the_client_at_what_exists(config):
    from jarvis.core.logging import get_logger
    from jarvis.core.wiring import resolve_model

    class FakeClient:
        model = "qwen3:8b"

        def available(self):
            return True

        def has_model(self, name=None):
            return (name or self.model) == "qwen3:14b"

        def models(self):
            return ["qwen3:14b"]

    client = FakeClient()
    config.set("brain.model", "qwen3:8b")
    config.set("system.vram_gb", 15.9)

    assert resolve_model(client, config, get_logger("test")) == "qwen3:14b"
    assert client.model == "qwen3:14b", "the client must actually be repointed"


def test_resolve_model_leaves_config_alone(config, tmp_path):
    """It is a runtime decision. Nothing is written to disk behind the user's back."""
    from jarvis.core.logging import get_logger
    from jarvis.core.wiring import resolve_model

    class FakeClient:
        model = "qwen3:8b"

        def available(self):
            return True

        def has_model(self, name=None):
            return False

        def models(self):
            return ["qwen3:14b"]

    before = config.path.read_text(encoding="utf-8")
    resolve_model(FakeClient(), config, get_logger("test"))
    assert config.path.read_text(encoding="utf-8") == before
