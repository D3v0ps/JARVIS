"""Kokoro ONNX speech synthesis — the British voice JARVIS speaks English with.

``kokoro-onnx`` runs a 310 MB ONNX model plus a voice bank entirely on the CPU (or
DirectML/CUDA when onnxruntime is built for it) and returns 24 kHz float samples, which
is why it is the default engine: no network, no licence, no account.

The library is imported inside :meth:`KokoroTTS._load`, never at module import time, so
this module imports on a machine that has never seen ``kokoro_onnx``. Missing model files
are reported as a plain, actionable sentence pointing at ``scripts/fetch_models.py``
rather than a traceback, and :meth:`available` then returns ``False`` so
:func:`~jarvis.tts.engine.create_engine` can move on to the next engine.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.tts.engine import DEFAULT_SAMPLE_RATE, NULL_SILENCE_S, peak_limit, silence

__all__ = ["KokoroTTS", "MAX_CHUNK_CHARS", "chunk_text"]

logger = logging.getLogger("jarvis.tts.kokoro")
#: Module logger alias, because ``__init__`` takes a ``logger`` argument that shadows it.
_LOG = logger

#: Kokoro's output rate. Fixed by the model.
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
#: Kokoro degrades on long inputs (it starts rushing and swallowing words), so text is
#: split at sentence boundaries into chunks no longer than this and concatenated.
MAX_CHUNK_CHARS = 400
#: Default phoniser language. The ``bm_*`` voices are British, so British phonemes.
DEFAULT_LANG = "en-gb"

_INSTALL_HINT = (
    "The Kokoro voice needs the 'kokoro-onnx' package: pip install kokoro-onnx"
)
_MODEL_HINT = "run 'python scripts/fetch_models.py' to download it"

#: Sentence end followed by whitespace — the preferred place to cut a long text.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
#: Clause separators, used when a single sentence is longer than the chunk limit.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into speakable chunks of at most ``max_chars`` characters.

    Cuts at sentence boundaries first, then at clause separators, and only as a last
    resort at a space, so a chunk never ends mid-word. Returns ``[]`` for empty input.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return []
    limit = max(1, int(max_chars))
    if len(body) <= limit:
        return [body]

    chunks: list[str] = []
    current = ""
    for piece in _split_pieces(body, limit):
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_pieces(body: str, limit: int) -> list[str]:
    """Break ``body`` into sentence-ish pieces, none longer than ``limit``."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(body):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= limit:
                pieces.append(clause)
            else:
                pieces.extend(_split_on_spaces(clause, limit))
    return pieces


def _split_on_spaces(text: str, limit: int) -> list[str]:
    """Hard-wrap ``text`` at spaces (mid-word only when a single word is too long)."""
    parts: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = f"{current} {word}"
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


class KokoroTTS:
    """Kokoro ONNX voice.

    Args:
        model_path: ``kokoro-v1.0.onnx``; relative paths resolve against the project root.
        voices_path: ``voices-v1.0.bin`` voice bank, resolved the same way.
        voice: Voice name from the bank, e.g. ``bm_george`` (British male).
        speed: Speaking rate multiplier passed straight to Kokoro (1.0 is natural).
        lang: Phoniser language; ``en-gb`` keeps the British pronunciation.
        logger: Optional logger.

    Construction never raises: a missing model file or a missing package leaves
    :meth:`available` ``False`` and :attr:`error` set to one actionable sentence.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
        voice: str = "bm_george",
        speed: float = 1.0,
        lang: str = DEFAULT_LANG,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = "kokoro"
        self.sample_rate = SAMPLE_RATE
        self.voice = str(voice or "bm_george")
        self.speed = float(speed) if speed and float(speed) > 0 else 1.0
        self.lang = str(lang or DEFAULT_LANG)
        self.log = logger if logger is not None else _LOG

        self.model_path = _resolve(model_path, "models/kokoro-v1.0.onnx")
        self.voices_path = _resolve(voices_path, "models/voices-v1.0.bin")

        #: One plain sentence explaining why the engine cannot speak ("" when it can).
        self.error: str = ""
        self._kokoro: Any | None = None
        self._closed = False
        # Kokoro's session and its phoniser are not thread-safe: one synthesis at a time.
        self._lock = threading.Lock()
        self._warned_language = False

        self._load()

    # --- loading ----------------------------------------------------------------------
    def _missing_files(self) -> list[Path]:
        return [path for path in (self.model_path, self.voices_path) if not path.is_file()]

    def _load(self) -> None:
        """Import ``kokoro_onnx`` and build the model, recording any failure in :attr:`error`."""
        missing = self._missing_files()
        if missing:
            names = ", ".join(str(path) for path in missing)
            self.error = f"Kokoro model file(s) not found: {names} — {_MODEL_HINT}."
            self.log.warning("%s", self.error)
            return

        try:
            from kokoro_onnx import Kokoro  # local import: heavy, optional, Windows/Linux
        except ImportError as exc:
            self.error = f"{_INSTALL_HINT} ({exc})."
            self.log.warning("%s", self.error)
            return

        try:
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        except Exception as exc:
            self.error = f"Kokoro could not load {self.model_path.name}: {exc}"
            self.log.error("%s", self.error, exc_info=True)
            self._kokoro = None
            return

        self.error = ""
        self.log.info(
            "Kokoro ready: voice %s at speed %.2f from %s.", self.voice, self.speed, self.model_path.name
        )

    # --- protocol ---------------------------------------------------------------------
    def available(self) -> bool:
        """True when the model, the voice bank and the package all loaded."""
        return self._kokoro is not None and not self._closed

    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]:
        """Render ``text`` with the configured British voice.

        Long text is chunked at sentence boundaries under :data:`MAX_CHUNK_CHARS` and the
        pieces are concatenated, because the model starts rushing on long inputs.

        Returns ``(float32 mono peak-limited to 1.0, 24000)``. A synthesis failure returns
        a short silence instead of raising — a broken sentence must not end the turn.
        """
        chunks = chunk_text(text, MAX_CHUNK_CHARS)
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        if language is not None and not self._warned_language and str(language).lower().startswith("sv"):
            self._warned_language = True
            self.log.info("Kokoro has no Swedish voice; speaking Swedish text with the %s voice.", self.voice)
        if not self.available():
            self.log.error("Kokoro was asked to speak but is unavailable: %s", self.error or "not loaded")
            return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate

        rendered: list[np.ndarray] = []
        rate = self.sample_rate
        with self._lock:
            kokoro = self._kokoro
            if kokoro is None:  # closed between the check and the lock
                return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate
            for index, chunk in enumerate(chunks):
                try:
                    samples, chunk_rate = kokoro.create(
                        chunk, voice=self.voice, speed=self.speed, lang=self.lang
                    )
                except Exception as exc:
                    self.log.error(
                        "Kokoro failed on chunk %d of %d (%d characters): %s",
                        index + 1, len(chunks), len(chunk), exc, exc_info=True,
                    )
                    continue
                audio = np.asarray(samples, dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    continue
                if index == 0:
                    rate = int(chunk_rate) if chunk_rate else self.sample_rate
                elif chunk_rate and int(chunk_rate) != rate:
                    self.log.warning(
                        "Kokoro changed sample rate mid-utterance (%s Hz after %s Hz); keeping the first.",
                        chunk_rate, rate,
                    )
                rendered.append(audio)

        if not rendered:
            return silence(NULL_SILENCE_S, rate), rate
        combined = rendered[0] if len(rendered) == 1 else np.concatenate(rendered)
        return peak_limit(combined), rate

    def close(self) -> None:
        """Drop the ONNX session. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            kokoro, self._kokoro = self._kokoro, None
        if kokoro is not None:
            for method in ("close", "release"):
                closer = getattr(kokoro, method, None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # pragma: no cover - best effort
                        self.log.debug("Kokoro %s() failed on close.", method, exc_info=True)
                    break
        self.log.debug("Kokoro engine closed.")

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "ready" if self.available() else f"unavailable: {self.error}"
        return f"KokoroTTS(voice={self.voice!r}, speed={self.speed}, {state})"


def _resolve(value: str | Path | None, default: str) -> Path:
    """Resolve a model path against the project root, honouring absolute paths and ``~``."""
    raw = str(value) if value not in (None, "") else default
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    try:
        from jarvis.config import project_root

        return (project_root() / candidate).resolve()
    except Exception:  # pragma: no cover - defensive
        return candidate.resolve()
