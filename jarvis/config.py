"""Configuration for JARVIS.

The shipped ``config.yaml`` is mirrored key for key in :data:`DEFAULTS`, so a
lookup can never raise for a key the user happened to delete: :meth:`Config.load`
deep-merges the file on top of the defaults and :meth:`Config.get` falls back to
the defaults for anything still missing.

Only the standard library plus PyYAML is used here — this module must import on
a bare Linux box with no audio hardware and no Windows-only packages.
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import yaml

logger = logging.getLogger("jarvis.config")

DEFAULT_CONFIG_PATH: Final[Path] = Path("config.yaml")

#: Sentinel telling :func:`_dig` apart from a key that legitimately holds ``None``.
_MISSING: Final[object] = object()

#: Mirror of the shipped ``config.yaml``. Every key present in the file is
#: present here with the same default value, so missing keys are impossible.
DEFAULTS: Final[dict[str, Any]] = {
    "language": "auto",  # auto | en | sv
    "assistant": {
        "wake_greeting": True,
        "acknowledge_phrase": "Sir?",
        "startup_greeting": True,
        "conversation_timeout": 20,
        "confirmation_timeout": 10,
        "safety_mode": "normal",  # normal | strict
        "max_tool_rounds": 4,
    },
    "audio": {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "block_size": 1280,
        "chime_volume": 0.35,
        "output_volume": 1.0,
        "barge_in": True,
        "barge_in_speech_ms": 400,
        "barge_in_grace_ms": 350,
    },
    "wake": {
        "enabled": True,
        "model": "hey_jarvis",
        "sensitivity": 0.5,
        "cooldown": 2.0,
        "framework": "onnx",  # onnx | tflite
    },
    "vad": {
        "threshold": 0.5,
        "silence_ms": 700,
        "min_speech_ms": 250,
        "max_utterance_s": 15,
        "pre_roll_ms": 300,
    },
    "stt": {
        "model": "auto",  # auto | tiny | base | small | medium | large-v3
        "device": "auto",  # auto | cuda | cpu
        "compute_type": "auto",  # auto | float16 | int8 | int8_float16
        "beam_size": 1,
        "vad_filter": False,
    },
    "brain": {
        "host": "http://127.0.0.1:11434",
        "model": "qwen3:8b",
        "deep_model": None,
        "think": False,
        "keep_alive": -1,
        "num_ctx": 8192,
        "temperature": 0.6,
        "history_turns": 12,
        "request_timeout": 120,
        "warm_on_start": True,
    },
    "tts": {
        "engine": "kokoro",  # kokoro | piper | sapi
        "voice": "bm_george",  # bm_george | bm_lewis | bf_emma
        "speed": 1.0,
        "kokoro_model": "models/kokoro-v1.0.onnx",
        "kokoro_voices": "models/voices-v1.0.bin",
        "piper_model": "models/sv_SE-nst-medium.onnx",
        "fallback": "sapi",
    },
    "ui": {
        "overlay": True,
        "tray": True,
        "ring_size": 180,
        "opacity": 0.92,
        "always_on_top": True,
        "position": None,  # [x, y]
        "theme": {
            "idle": "#1b3a4b",
            "listening": "#29b6f6",
            "thinking": "#ffb300",
            "speaking": "#4dd0e1",
            "paused": "#616161",
        },
    },
    "tools": {
        "default_city": "Stockholm",
        "screenshot_dir": None,  # None = Pictures\\Jarvis
        "powershell_timeout": 30,
        "search_results": 3,
        "file_search_dirs": ["Desktop", "Documents", "Downloads"],
        "apps": {
            "spotify": "spotify",
            "chrome": "chrome",
            "edge": "msedge",
            "firefox": "firefox",
            "code": "code",
            "vscode": "code",
            "steam": "steam",
            "discord": "discord",
            "explorer": "explorer",
            "terminal": "wt",
            "powershell": "powershell",
            "settings": "ms-settings:",
            "calculator": "calc",
            "notepad": "notepad",
            "task manager": "taskmgr",
            "obs": "obs64",
            "youtube": "https://www.youtube.com",
        },
    },
    "system": {
        "vram_gb": None,
        "gpu_name": None,
    },
    "logging": {
        "level": "INFO",  # DEBUG | INFO | WARNING | ERROR
        "file": "logs/jarvis.log",
        "max_bytes": 5242880,
        "backups": 3,
        "color": True,
    },
}

#: Language aliases accepted in ``language:`` (and in spoken/CLI overrides).
_LANGUAGE_ALIASES: Final[dict[str, str]] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "engelska": "en",
    "sv": "sv",
    "swe": "sv",
    "swedish": "sv",
    "svenska": "sv",
    "sv-se": "sv",
}


def project_root() -> Path:
    """Return the repository root — the directory containing the ``jarvis`` package.

    Resolved from ``__file__`` rather than the process working directory, so the
    assistant finds ``config.yaml``, ``models/`` and ``prompts/`` no matter where
    it was launched from.
    """
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dictionaries merge key by key; scalars and lists replace the default
    outright. Keys only present in ``override`` are kept.
    """
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _dig(data: Any, parts: list[str]) -> Any:
    """Walk ``parts`` through nested mappings, returning ``_MISSING`` if absent."""
    node: Any = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


class Config:
    """Dot-path access over the parsed ``config.yaml``, with defaults baked in."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self._data: dict[str, Any] = data if isinstance(data, dict) else {}
        self._path: Path | None = Path(path) if path is not None else None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load ``path`` (default ``<project root>/config.yaml``) over :data:`DEFAULTS`.

        A missing file is not an error — the defaults are used and the fact is
        logged at INFO. A malformed file raises :class:`ValueError` naming both
        the file and the YAML error.
        """
        resolved = Path(path).expanduser() if path is not None else project_root() / DEFAULT_CONFIG_PATH
        if not resolved.is_absolute():
            # A relative path is honoured from the working directory when it
            # exists there (``--config my.yaml``), otherwise anchored at the
            # repository root so launching from anywhere still finds the file.
            cwd_candidate = (Path.cwd() / resolved).resolve()
            resolved = cwd_candidate if cwd_candidate.is_file() else (project_root() / resolved).resolve()

        if not resolved.is_file():
            logger.info("Config file %s not found; using built-in defaults.", resolved)
            return cls(copy.deepcopy(DEFAULTS), resolved)

        try:
            raw_text = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Cannot read config file %s: %s", resolved, exc)
            raise ValueError(f"Cannot read config file {resolved}: {exc}") from exc

        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            logger.error("Malformed YAML in config file %s: %s", resolved, exc)
            raise ValueError(f"Malformed YAML in config file {resolved}: {exc}") from exc

        if parsed is None:
            logger.info("Config file %s is empty; using built-in defaults.", resolved)
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Malformed config file {resolved}: expected a YAML mapping at the "
                f"top level, got {type(parsed).__name__}"
            )

        merged = _deep_merge(DEFAULTS, parsed)
        logger.debug("Loaded configuration from %s (%d top-level keys).", resolved, len(merged))
        return cls(merged, resolved)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return the value at ``dotted`` (e.g. ``"brain.model"``).

        Falls back to :data:`DEFAULTS` before returning ``default``, so a key the
        user deleted still yields the shipped value. A key explicitly set to
        ``null`` yields ``None``, not the default.
        """
        if not dotted:
            return default
        parts = dotted.split(".")
        value = _dig(self._data, parts)
        if value is _MISSING:
            value = _dig(DEFAULTS, parts)
        if value is _MISSING:
            return default
        return value

    def set(self, dotted: str, value: Any) -> None:
        """Set ``dotted`` in memory, creating intermediate sections as needed."""
        if not dotted:
            raise ValueError("Config.set() requires a non-empty dotted key.")
        parts = dotted.split(".")
        node: dict[str, Any] = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        """Return the mapping at ``name`` (e.g. ``"stt"``), or ``{}`` if it is not one."""
        value = self.get(name)
        if isinstance(value, dict):
            return value
        if value is not None:
            logger.warning("Config section %r is %s, not a mapping.", name, type(value).__name__)
        return {}

    def resolve_path(self, dotted_or_value: str | Path) -> Path:
        """Resolve a configured path against :func:`project_root`.

        ``dotted_or_value`` may be a dotted config key whose value is a path
        (``"logging.file"`` -> ``<root>/logs/jarvis.log``) or a literal path
        (``"models/kokoro-v1.0.onnx"``). Absolute paths and ``~`` are honoured
        as-is; everything else is anchored at the repository root so the working
        directory never matters.
        """
        value: str | Path = dotted_or_value
        if isinstance(dotted_or_value, str):
            looked_up = self.get(dotted_or_value, _MISSING)
            if isinstance(looked_up, (str, Path)) and str(looked_up):
                value = looked_up
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (project_root() / candidate).resolve()

    def save(self) -> None:
        """Write the current data back to :attr:`path` as YAML, atomically.

        Written to a temporary file in the same directory and moved into place
        with :func:`os.replace`, so a crash mid-write cannot truncate the user's
        configuration. Failures (read-only file, missing directory, permission
        denied) are logged and reported by returning ``False`` rather than
        raising — saving config is never worth killing the assistant over.
        Returns ``None`` on success.
        """
        if self._path is None:
            logger.warning("Config has no file path; nothing was saved.")
            return False  # type: ignore[return-value]

        target = Path(self._path)
        tmp_name: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(target.parent),
                prefix=target.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                yaml.safe_dump(
                    self._data,
                    handle,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
            tmp_name = None
            logger.debug("Configuration saved to %s.", target)
            return None
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Could not save configuration to %s: %s", target, exc)
            return False  # type: ignore[return-value]
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError as exc:  # pragma: no cover - cleanup best effort
                    logger.debug("Could not remove temporary file %s: %s", tmp_name, exc)

    @property
    def path(self) -> Path:
        """The file this configuration came from (the default location if unknown)."""
        if self._path is None:
            return project_root() / DEFAULT_CONFIG_PATH
        return self._path

    @property
    def data(self) -> dict[str, Any]:
        """The live, merged configuration mapping."""
        return self._data

    def __repr__(self) -> str:
        return f"Config(path={str(self.path)!r}, sections={sorted(self._data)!r})"


def resolve_language(cfg: Config) -> str | None:
    """Return ``"en"``, ``"sv"`` or ``None`` when ``language: auto``.

    ``None`` means "let the speech recogniser decide", which is what the rest of
    the pipeline expects for automatic language detection.
    """
    raw = cfg.get("language", "auto")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    mapped = _LANGUAGE_ALIASES.get(text)
    if mapped is not None:
        return mapped
    if len(text) == 2 and text.isalpha():
        logger.info("Unrecognised language %r; passing it through as-is.", raw)
        return text
    logger.warning("Unrecognised language %r in config; falling back to auto-detect.", raw)
    return None


__all__ = [
    "DEFAULTS",
    "DEFAULT_CONFIG_PATH",
    "Config",
    "project_root",
    "resolve_language",
]
