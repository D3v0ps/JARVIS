"""Speech-synthesis engine protocol, the silent fallback, and engine selection.

Every voice JARVIS can use implements :class:`TTSEngine`: a sample rate, a name, a
blocking :meth:`~TTSEngine.synthesize` that returns float32 mono audio, an
:meth:`~TTSEngine.available` probe and a :meth:`~TTSEngine.close`.

:func:`create_engine` builds the engine named in ``tts.engine``, catches every
construction failure, logs it in a single readable line and falls back through
``tts.fallback`` and finally to :class:`NullEngine` — the assistant must never die for
want of a voice.

Nothing heavy is imported here: the concrete engines (and with them ``kokoro_onnx``,
``piper`` and ``pyttsx3``) are imported inside :func:`_build_engine`, so importing this
module works on a bare Linux box.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import Config

__all__ = [
    "TTSEngine",
    "NullEngine",
    "create_engine",
    "select_engine_for_language",
    "reset_engine_cache",
    "peak_limit",
    "to_float32_mono",
    "silence",
    "KNOWN_ENGINES",
    "DEFAULT_SAMPLE_RATE",
    "NULL_SILENCE_S",
]

logger = logging.getLogger("jarvis.tts.engine")
#: Module logger under its own name, since the public functions take a ``logger`` argument.
_LOG = logger

#: Kokoro's native rate, and the rate the silent engine reports.
DEFAULT_SAMPLE_RATE = 24_000
#: How much silence :class:`NullEngine` returns per utterance, in seconds.
NULL_SILENCE_S = 0.2
#: Engine names understood by :func:`create_engine`.
KNOWN_ENGINES = ("kokoro", "piper", "sapi", "null")

#: Divisors turning integer PCM into [-1, 1].
_INT_SCALES: dict[Any, float] = {
    np.dtype(np.int16): 32_768.0,
    np.dtype(np.int32): 2_147_483_648.0,
}


@runtime_checkable
class TTSEngine(Protocol):
    """What the :class:`~jarvis.tts.speaker.Speaker` needs from a voice."""

    sample_rate: int
    name: str

    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]:
        """Render ``text`` to float32 mono audio, returning ``(audio, sample_rate)``."""
        ...

    def available(self) -> bool:
        """True when this engine can actually speak right now."""
        ...

    def close(self) -> None:
        """Release models, handles and temporary files. Safe to call twice."""
        ...


# --- shared audio helpers -------------------------------------------------------------
def to_float32_mono(audio: Any) -> np.ndarray:
    """Convert int16/int32/uint8/float audio, mono or ``(n, channels)``, to float32 mono."""
    data = np.asarray(audio)
    if data.ndim == 2:
        data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
    elif data.ndim > 2:
        raise ValueError(f"Audio must be 1-D or 2-D, got shape {data.shape}")
    scale = _INT_SCALES.get(data.dtype)
    if scale is not None:
        data = data.astype(np.float32) / scale
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    return np.ascontiguousarray(data, dtype=np.float32)


def peak_limit(audio: Any, peak: float = 1.0) -> np.ndarray:
    """Return float32 mono audio scaled so its loudest sample is at most ``peak``.

    Non-finite samples become silence rather than a speaker-destroying click, and the
    result is clipped as a final guarantee — every engine returns audio through here.
    """
    data = to_float32_mono(audio)
    ceiling = float(peak) if peak > 0.0 else 1.0
    if data.size == 0:
        return data
    data = np.nan_to_num(data, nan=0.0, posinf=ceiling, neginf=-ceiling)
    loudest = float(np.max(np.abs(data)))
    if loudest > ceiling:
        data = data * np.float32(ceiling / loudest)
    return np.clip(data, -ceiling, ceiling).astype(np.float32)


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    """Return ``seconds`` of float32 silence at ``sample_rate`` (never negative length)."""
    count = int(round(max(0.0, float(seconds)) * max(1, int(sample_rate))))
    return np.zeros(count, dtype=np.float32)


class NullEngine:
    """The voice of last resort: it logs once and returns silence.

    Returning ``NULL_SILENCE_S`` of silence instead of an empty array keeps the
    pipeline's timing intact — the player still "plays" a clip, the state bus still
    passes through SPEAKING, and the latency tracker still sees a first-audio moment.

    :meth:`available` deliberately returns ``True``: this engine always works, it simply
    has nothing to say. A caller that wants to know whether a real voice was found can
    compare ``engine.name`` with ``"null"`` or read :attr:`reason`.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silence_s: float = NULL_SILENCE_S,
        reason: str = "",
    ) -> None:
        self.name = "null"
        self.sample_rate = int(sample_rate) if int(sample_rate) > 0 else DEFAULT_SAMPLE_RATE
        self.silence_s = float(silence_s)
        #: Why we ended up here, for the log line and for callers that want to explain it.
        self.reason = str(reason or "")
        self.log = logger if logger is not None else logging.getLogger("jarvis.tts.null")
        self._warned = False
        self._closed = False

    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]:
        """Return :data:`NULL_SILENCE_S` seconds of silence, logging the first time only."""
        if not self._warned:
            self._warned = True
            detail = f" ({self.reason})" if self.reason else ""
            self.log.warning(
                "No speech engine is available%s; JARVIS will answer silently. "
                "Run 'python scripts/fetch_models.py' to install the British voice.", detail,
            )
        return silence(self.silence_s, self.sample_rate), self.sample_rate

    def available(self) -> bool:
        """Always ``True`` — silence never fails."""
        return True

    def close(self) -> None:
        """Mark the engine closed. Nothing to release."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NullEngine(sample_rate={self.sample_rate}, reason={self.reason!r})"


# --- configuration helpers ------------------------------------------------------------
def _cfg_get(cfg: "Config | None", dotted: str, default: Any) -> Any:
    """Read a dotted key from a :class:`~jarvis.config.Config` (or anything with ``get``)."""
    if cfg is None:
        return default
    try:
        value = cfg.get(dotted, default)
    except Exception:
        logger.debug("Could not read config key %r; using %r", dotted, default, exc_info=True)
        return default
    return default if value is None else value


def _resolve_path(cfg: "Config | None", dotted: str, default: str) -> Path:
    """Resolve a configured, possibly relative, path against the project root."""
    raw = str(_cfg_get(cfg, dotted, default) or default)
    resolver = getattr(cfg, "resolve_path", None)
    if callable(resolver):
        try:
            return Path(resolver(raw))
        except Exception:
            logger.debug("Config.resolve_path failed for %r", raw, exc_info=True)
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    try:
        from jarvis.config import project_root

        return (project_root() / candidate).resolve()
    except Exception:  # pragma: no cover - defensive
        return candidate.resolve()


def _is_swedish(language: str | None) -> bool:
    """True for ``"sv"``, ``"sv-SE"``, ``"swedish"``, ``"svenska"`` and friends."""
    text = str(language or "").strip().lower().replace("_", "-")
    return text.startswith("sv") or text in {"swedish", "svenska"}


def _as_name_list(value: Any) -> list[str]:
    """Normalise ``tts.fallback`` — a single name, a comma-separated string, or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, Iterable):
        names: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip().lower()
            if text:
                names.append(text)
        return names
    return [str(value).strip().lower()]


def _candidate_names(cfg: "Config | None", language: str | None, log: logging.Logger) -> list[str]:
    """Engine names to try, in order, for ``language``.

    The language rule lives here: when the resolved language is Swedish and the Piper
    Swedish model is actually installed, Piper goes first even though ``tts.engine`` says
    ``kokoro`` — Kokoro has no Swedish voice, so the British one would read Swedish text
    with an English phoniser. When the model is missing we stay with the configured
    engine rather than falling silent.
    """
    primary = str(_cfg_get(cfg, "tts.engine", "kokoro")).strip().lower() or "kokoro"
    names: list[str] = []

    if _is_swedish(language) and primary != "piper":
        piper_model = _resolve_path(cfg, "tts.piper_model", "models/sv_SE-nst-medium.onnx")
        if piper_model.is_file():
            log.info("Swedish requested: using the Piper voice %s ahead of %s.", piper_model.name, primary)
            names.append("piper")
        else:
            log.debug(
                "Swedish requested but the Piper model %s is missing; staying with %s. "
                "Install it with 'python scripts/fetch_models.py --swedish'.", piper_model, primary,
            )

    names.append(primary)
    names.extend(_as_name_list(_cfg_get(cfg, "tts.fallback", "sapi")))

    ordered: list[str] = []
    for name in names:
        if name in ordered or name == "null":
            continue
        if name not in KNOWN_ENGINES:
            log.warning("Unknown TTS engine %r in config; known engines are %s.", name, ", ".join(KNOWN_ENGINES))
            continue
        ordered.append(name)
    return ordered


# --- construction ---------------------------------------------------------------------
def _build_engine(name: str, cfg: "Config | None", log: logging.Logger) -> TTSEngine | None:
    """Construct one engine by name. Returns ``None`` (after one log line) on any failure."""
    try:
        engine: TTSEngine
        if name == "kokoro":
            from jarvis.tts.kokoro_tts import KokoroTTS

            engine = KokoroTTS(
                model_path=_cfg_get(cfg, "tts.kokoro_model", "models/kokoro-v1.0.onnx"),
                voices_path=_cfg_get(cfg, "tts.kokoro_voices", "models/voices-v1.0.bin"),
                voice=str(_cfg_get(cfg, "tts.voice", "bm_george")),
                speed=float(_cfg_get(cfg, "tts.speed", 1.0)),
                logger=log,
            )
        elif name == "piper":
            from jarvis.tts.piper_tts import PiperTTS

            engine = PiperTTS(
                model_path=_cfg_get(cfg, "tts.piper_model", "models/sv_SE-nst-medium.onnx"),
                speed=float(_cfg_get(cfg, "tts.speed", 1.0)),
                logger=log,
            )
        elif name == "sapi":
            from jarvis.tts.sapi_tts import SapiTTS

            engine = SapiTTS(speed=float(_cfg_get(cfg, "tts.speed", 1.0)), logger=log)
        elif name == "null":
            engine = NullEngine(log)
        else:  # pragma: no cover - filtered by _candidate_names
            log.warning("Unknown TTS engine %r; skipping it.", name)
            return None
    except Exception as exc:
        log.error("Speech engine '%s' could not be created: %s", name, exc, exc_info=True)
        return None

    try:
        if engine.available():
            log.info("Speech engine '%s' ready at %d Hz.", engine.name, engine.sample_rate)
            return engine
        reason = str(getattr(engine, "error", "") or "no reason given")
        log.warning("Speech engine '%s' is unavailable: %s", name, reason)
    except Exception as exc:
        log.error("Speech engine '%s' failed its availability check: %s", name, exc, exc_info=True)
    try:
        engine.close()
    except Exception:  # pragma: no cover - defensive
        log.debug("Closing the unusable '%s' engine failed.", name, exc_info=True)
    return None


# --- engine cache ---------------------------------------------------------------------
# The Speaker calls select_engine_for_language() per utterance, so built engines are
# cached by (name + the config values that shaped them). Loading the Kokoro model takes
# seconds; doing it per sentence would be unusable.
_CACHE_LOCK = threading.RLock()
_ENGINE_CACHE: dict[tuple[Any, ...], TTSEngine] = {}


def _cache_key(name: str, cfg: "Config | None") -> tuple[Any, ...]:
    return (
        name,
        str(_cfg_get(cfg, "tts.voice", "bm_george")),
        str(_cfg_get(cfg, "tts.speed", 1.0)),
        str(_cfg_get(cfg, "tts.kokoro_model", "")),
        str(_cfg_get(cfg, "tts.kokoro_voices", "")),
        str(_cfg_get(cfg, "tts.piper_model", "")),
    )


def _cached(key: tuple[Any, ...]) -> TTSEngine | None:
    with _CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            return None
        if getattr(engine, "closed", False):
            _ENGINE_CACHE.pop(key, None)
            return None
        return engine


def reset_engine_cache() -> None:
    """Close and drop every cached engine. Used by tests and by a config reload."""
    with _CACHE_LOCK:
        engines = list(_ENGINE_CACHE.values())
        _ENGINE_CACHE.clear()
    for engine in engines:
        try:
            engine.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Closing cached engine %r failed.", getattr(engine, "name", "?"), exc_info=True)


def select_engine_for_language(
    cfg: "Config", language: str | None, logger: logging.Logger | None = None
) -> TTSEngine:
    """Return the best engine for ``language``, building and caching it on first use.

    Order of preference:

    1. Piper, when ``language`` resolves to Swedish **and** the Piper model file exists —
       even if ``tts.engine`` is ``kokoro``, because Kokoro has no Swedish voice.
    2. ``tts.engine``.
    3. ``tts.fallback`` (a name, a comma-separated string, or a list).
    4. :class:`NullEngine`, which speaks silence rather than raising.

    This never raises: an engine that cannot be built is logged and skipped.
    """
    log = logger if logger is not None else _LOG
    tried: list[str] = []
    for name in _candidate_names(cfg, language, log):
        tried.append(name)
        key = _cache_key(name, cfg)
        engine = _cached(key)
        if engine is not None:
            return engine
        engine = _build_engine(name, cfg, log)
        if engine is not None:
            with _CACHE_LOCK:
                _ENGINE_CACHE[key] = engine
            return engine

    reason = f"tried {', '.join(tried)}" if tried else "no engine configured"
    null_key = ("null", reason)
    cached_null = _cached(null_key)
    if cached_null is not None:
        return cached_null
    log.error("No speech engine could be started (%s); falling back to silence.", reason)
    null_engine = NullEngine(log, reason=reason)
    with _CACHE_LOCK:
        _ENGINE_CACHE[null_key] = null_engine
    return null_engine


def create_engine(cfg: "Config", logger: logging.Logger | None = None) -> TTSEngine:
    """Build the configured speech engine, falling back until something works.

    Equivalent to :func:`select_engine_for_language` with the language from
    ``config.yaml`` (``None`` when ``language: auto``). Never raises.
    """
    log = logger if logger is not None else _LOG
    language: str | None = None
    try:
        from jarvis.config import resolve_language

        language = resolve_language(cfg)
    except Exception:
        log.debug("Could not resolve the configured language; assuming automatic.", exc_info=True)
    try:
        return select_engine_for_language(cfg, language, log)
    except Exception as exc:  # pragma: no cover - select_* already swallows failures
        log.error("Speech engine selection failed unexpectedly: %s", exc, exc_info=True)
        return NullEngine(log, reason=str(exc))
