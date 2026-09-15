"""Configuration behaviour: dotted lookups, defaults, merging and path resolution.

Everything here works on a throwaway config file under ``tmp_path``; the shipped
``config.yaml`` in the repository is only ever read, never written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jarvis.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULTS,
    Config,
    project_root,
    resolve_language,
)


def write_config(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    """Write ``text`` as a config file under ``tmp_path`` and return its path."""
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


# --- dotted lookups -------------------------------------------------------------------
def test_get_reads_a_top_level_key(config):
    assert config.get("language") == "auto"


def test_get_reads_a_nested_key(config):
    assert config.get("brain.model") == "qwen3:8b"


def test_get_reads_a_three_level_key(config):
    assert config.get("ui.theme.idle") == "#1b3a4b"


def test_get_reads_an_app_mapping_entry(config):
    assert config.get("tools.apps.spotify") == "spotify"


def test_get_reads_a_key_that_contains_a_space(config):
    """`tools.apps` is keyed by what the user says, so 'task manager' has a space in it."""
    assert config.get("tools.apps.task manager") == "taskmgr"


def test_get_returns_the_supplied_default_for_an_unknown_key(config):
    assert config.get("brain.no_such_key", "fallback") == "fallback"


def test_get_returns_none_by_default_for_an_unknown_key(config):
    assert config.get("nothing.at.all") is None


def test_get_returns_the_default_for_an_empty_dotted_key(config):
    assert config.get("", "fallback") == "fallback"


def test_get_falls_back_to_defaults_when_the_data_lacks_the_key():
    """A user who deleted a key from their file still gets the shipped value."""
    bare = Config({})
    assert bare.get("brain.model") == "qwen3:8b"
    assert bare.get("ui.theme.listening") == "#29b6f6"


def test_get_prefers_an_explicit_null_over_the_default(tmp_path):
    path = write_config(tmp_path, "brain:\n  temperature: null\n")
    cfg = Config.load(path)
    assert cfg.get("brain.temperature") is None
    assert cfg.get("brain.temperature", 0.6) is None


def test_section_returns_the_mapping(config):
    section = config.section("stt")
    assert section["beam_size"] == 1
    assert section["model"] == "auto"


def test_section_returns_an_empty_dict_for_a_scalar(tmp_path):
    path = write_config(tmp_path, "stt: nonsense\n")
    assert Config.load(path).section("stt") == {}


# --- loading and merging --------------------------------------------------------------
def test_partial_file_keeps_default_siblings(tmp_path):
    path = write_config(tmp_path, "brain:\n  model: llama3\n")
    cfg = Config.load(path)
    assert cfg.get("brain.model") == "llama3"
    assert cfg.get("brain.num_ctx") == DEFAULTS["brain"]["num_ctx"]
    assert cfg.get("tts.engine") == DEFAULTS["tts"]["engine"]


def test_partial_file_merges_deep_sections_key_by_key(tmp_path):
    path = write_config(tmp_path, "ui:\n  theme:\n    idle: '#000000'\n")
    cfg = Config.load(path)
    assert cfg.get("ui.theme.idle") == "#000000"
    assert cfg.get("ui.theme.speaking") == DEFAULTS["ui"]["theme"]["speaking"]
    assert cfg.get("ui.ring_size") == DEFAULTS["ui"]["ring_size"]


def test_unknown_keys_in_the_file_are_kept(tmp_path):
    path = write_config(tmp_path, "experimental: true\nbrain:\n  my_own_knob: 7\n")
    cfg = Config.load(path)
    assert cfg.get("experimental") is True
    assert cfg.get("brain.my_own_knob") == 7


def test_a_list_in_the_file_replaces_the_default_list(tmp_path):
    path = write_config(tmp_path, "tools:\n  file_search_dirs: [Skrivbord]\n")
    assert Config.load(path).get("tools.file_search_dirs") == ["Skrivbord"]


def test_missing_file_yields_the_defaults(tmp_path):
    cfg = Config.load(tmp_path / "not-there.yaml")
    assert cfg.data == DEFAULTS
    assert cfg.get("brain.model") == "qwen3:8b"


def test_missing_file_does_not_create_it(tmp_path):
    missing = tmp_path / "not-there.yaml"
    Config.load(missing)
    assert not missing.exists()


def test_empty_file_yields_the_defaults(tmp_path):
    path = write_config(tmp_path, "")
    assert Config.load(path).data == DEFAULTS


def test_loading_the_shipped_config_does_not_mutate_defaults(config):
    config.set("brain.model", "something-else")
    assert DEFAULTS["brain"]["model"] == "qwen3:8b"


def test_unicode_values_survive_loading(tmp_path):
    path = write_config(tmp_path, "tools:\n  default_city: Göteborg\n")
    assert Config.load(path).get("tools.default_city") == "Göteborg"


def test_malformed_yaml_raises_value_error_naming_the_file(tmp_path):
    path = write_config(tmp_path, "brain:\n  model: [unclosed\n  temperature: :\n")
    with pytest.raises(ValueError) as excinfo:
        Config.load(path)
    assert str(path) in str(excinfo.value)


def test_a_top_level_list_raises_value_error_naming_the_file(tmp_path):
    path = write_config(tmp_path, "- brain\n- tts\n")
    with pytest.raises(ValueError) as excinfo:
        Config.load(path)
    assert str(path) in str(excinfo.value)


def test_a_non_utf8_file_raises_value_error_naming_the_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes("tools:\n  default_city: Göteborg\n".encode("cp1252"))
    with pytest.raises(ValueError) as excinfo:
        Config.load(path)
    assert str(path) in str(excinfo.value)


# --- set / save / reload --------------------------------------------------------------
def test_set_and_save_round_trip_through_the_file(tmp_path):
    path = write_config(tmp_path, "brain:\n  model: qwen3:8b\n")
    cfg = Config.load(path)
    cfg.set("brain.model", "qwen3:14b")
    cfg.set("tools.default_city", "Malmö")
    cfg.save()

    reloaded = Config.load(path)
    assert reloaded.get("brain.model") == "qwen3:14b"
    assert reloaded.get("tools.default_city") == "Malmö"


def test_set_creates_missing_intermediate_sections(tmp_path):
    cfg = Config.load(tmp_path / "absent.yaml")
    cfg.set("brand.new.leaf", 42)
    assert cfg.get("brand.new.leaf") == 42


def test_set_rejects_an_empty_key(config):
    with pytest.raises(ValueError):
        config.set("", "value")


def test_save_writes_readable_yaml(tmp_path):
    path = write_config(tmp_path, "brain:\n  model: qwen3:8b\n")
    cfg = Config.load(path)
    cfg.set("tools.default_city", "Göteborg")
    cfg.save()

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["tools"]["default_city"] == "Göteborg"


def test_save_leaves_no_temporary_files_behind(tmp_path):
    path = write_config(tmp_path, "brain:\n  model: qwen3:8b\n")
    cfg = Config.load(path)
    cfg.save()
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]


def test_save_returns_falsy_when_the_file_cannot_be_written(tmp_path):
    """A read-only config must not kill the assistant: report, do not raise."""
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    cfg = Config({"language": "en"}, blocker / "config.yaml")
    assert not cfg.save()


def test_save_without_a_path_is_a_no_op_returning_falsy():
    assert not Config({"language": "en"}).save()


def test_path_defaults_to_the_shipped_location():
    assert Config({}).path == project_root() / DEFAULT_CONFIG_PATH


# --- path resolution ------------------------------------------------------------------
def test_resolve_path_expands_a_dotted_key(config):
    assert config.resolve_path("tts.kokoro_model") == project_root() / "models" / "kokoro-v1.0.onnx"


def test_resolve_path_anchors_a_relative_literal_at_the_project_root(config):
    assert config.resolve_path("models/voices-v1.0.bin") == project_root() / "models" / "voices-v1.0.bin"


def test_resolve_path_leaves_an_absolute_path_alone(config, tmp_path):
    absolute = tmp_path / "elsewhere" / "kokoro.onnx"
    assert config.resolve_path(absolute) == absolute


def test_resolve_path_ignores_the_working_directory(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.resolve_path("models/voices-v1.0.bin").parent.parent == project_root()


def test_project_root_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    before = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == before
    assert (project_root() / "jarvis" / "config.py").is_file()


# --- language -------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["auto", "AUTO", "  ", ""])
def test_resolve_language_returns_none_for_auto(raw):
    """None means 'let Whisper detect it', which is what the pipeline expects."""
    assert resolve_language(Config({"language": raw})) is None


@pytest.mark.parametrize(
    "raw,expected",
    [("en", "en"), ("English", "en"), ("engelska", "en"), ("sv", "sv"), ("Svenska", "sv"), ("sv-SE", "sv")],
)
def test_resolve_language_maps_known_spellings(raw, expected):
    assert resolve_language(Config({"language": raw})) == expected


def test_resolve_language_falls_back_to_auto_for_nonsense():
    assert resolve_language(Config({"language": "klingon"})) is None


def test_resolve_language_falls_back_to_auto_for_a_null_language():
    assert resolve_language(Config({"language": None})) is None


def test_resolve_language_uses_the_shipped_default_when_the_key_is_gone():
    assert resolve_language(Config({})) is None
