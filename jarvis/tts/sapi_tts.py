"""Windows SAPI 5 speech synthesis through pyttsx3 — the voice that is always there.

Every Windows 11 machine ships with SAPI voices, so this is the fallback that lets JARVIS
talk before anyone has downloaded a model. pyttsx3 cannot hand back samples, so an
utterance is written to a temporary wav with ``save_to_file`` + ``runAndWait``, read back
with the standard-library :mod:`wave` module (no soundfile dependency), converted to
float32 and the file deleted.

``pyttsx3`` is imported inside :meth:`SapiTTS._load`, never at module import time, and the
engine reports itself unavailable with a clear sentence on anything that is not Windows.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from typing import Any

import numpy as np

from jarvis.tts.engine import NULL_SILENCE_S, peak_limit, read_wav_file, silence

__all__ = ["SapiTTS", "BRITISH_VOICE_HINTS", "DEFAULT_SAMPLE_RATE"]

logger = logging.getLogger("jarvis.tts.sapi")
#: Module logger alias, because ``__init__`` takes a ``logger`` argument that shadows it.
_LOG = logger

#: What SAPI usually produces; corrected from the wav header after the first utterance.
DEFAULT_SAMPLE_RATE = 22_050
#: Substrings that mark a British voice, most specific first. "George" and "Hazel" are the
#: Microsoft en-GB voices; the language tags catch third-party ones.
BRITISH_VOICE_HINTS = (
    "george", "hazel", "english (great britain)", "en-gb", "en_gb", "great britain", "united kingdom",
)
#: SAPI's neutral rate is 200 words per minute; JARVIS is a touch brisker than that.
DEFAULT_RATE_FACTOR = 1.08


class SapiTTS:
    """Windows SAPI 5 voice via pyttsx3.

    Args:
        voice: Optional voice id or name substring; overrides the British-voice search.
        rate: Optional absolute words-per-minute; overrides ``speed``.
        speed: Multiplier applied to the voice's own rate (config ``tts.speed``).
        volume: SAPI volume in [0, 1].
        logger: Optional logger.

    Construction never raises: on a non-Windows box, or without pyttsx3, :meth:`available`
    is ``False`` and :attr:`error` holds one plain sentence.
    """

    def __init__(
        self,
        voice: str | None = None,
        rate: int | None = None,
        speed: float = 1.0,
        volume: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = "sapi"
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.requested_voice = str(voice) if voice else None
        self.requested_rate = int(rate) if rate else None
        self.speed = float(speed) if speed and float(speed) > 0 else 1.0
        self.volume = min(1.0, max(0.0, float(volume)))
        self.log = logger if logger is not None else _LOG

        #: One plain sentence explaining why the engine cannot speak ("" when it can).
        self.error: str = ""
        #: Name of the SAPI voice actually in use, for the log and for diagnostics.
        self.voice_name: str = ""
        self._engine: Any | None = None
        self._closed = False
        # The SAPI event loop is not reentrant: runAndWait() from two threads deadlocks.
        self._lock = threading.Lock()

        self._load()

    # --- loading ----------------------------------------------------------------------
    def _load(self) -> None:
        """Initialise pyttsx3 on the SAPI 5 driver, recording failures in :attr:`error`."""
        if sys.platform != "win32":
            self.error = (
                f"The Windows voice needs Windows; this is {sys.platform}. "
                "Use the Kokoro engine instead."
            )
            self.log.debug("%s", self.error)
            return

        try:
            import pyttsx3  # local import: Windows-only COM wrapper
        except ImportError as exc:
            self.error = f"The Windows voice needs the 'pyttsx3' package: pip install pyttsx3 ({exc})."
            self.log.warning("%s", self.error)
            return

        try:
            engine = pyttsx3.init("sapi5")
        except Exception as exc:
            self.error = f"pyttsx3 could not start the SAPI 5 driver: {exc}"
            self.log.error("%s", self.error, exc_info=True)
            return

        self._engine = engine
        self.error = ""
        try:
            self._configure(engine)
        except Exception:  # pragma: no cover - configuration is best effort
            self.log.warning("Could not fully configure the Windows voice; using its defaults.", exc_info=True)
        self.log.info(
            "Windows SAPI voice ready: %s.", self.voice_name or "system default",
        )

    def _configure(self, engine: Any) -> None:
        """Pick a British voice when one is installed and set a slightly brisk rate."""
        chosen = self._pick_voice(engine)
        if chosen is not None:
            voice_id, self.voice_name = chosen
            try:
                engine.setProperty("voice", voice_id)
            except Exception:
                self.log.warning("SAPI refused the voice %s; keeping the default.", self.voice_name, exc_info=True)
                self.voice_name = ""

        try:
            base = int(engine.getProperty("rate") or 200)
        except Exception:
            base = 200
        target = self.requested_rate or int(round(base * DEFAULT_RATE_FACTOR * self.speed))
        target = max(80, min(400, target))
        try:
            engine.setProperty("rate", target)
        except Exception:
            self.log.debug("SAPI refused rate %d; keeping %d.", target, base, exc_info=True)
        try:
            engine.setProperty("volume", self.volume)
        except Exception:
            self.log.debug("SAPI refused volume %.2f.", self.volume, exc_info=True)

    def _pick_voice(self, engine: Any) -> tuple[str, str] | None:
        """Return ``(voice_id, name)`` for the requested or the best British voice."""
        try:
            voices = list(engine.getProperty("voices") or [])
        except Exception:
            self.log.debug("Could not enumerate SAPI voices.", exc_info=True)
            return None

        if self.requested_voice:
            wanted = self.requested_voice.lower()
            for voice in voices:
                if wanted in _voice_text(voice):
                    return str(getattr(voice, "id", "")), str(getattr(voice, "name", "") or self.requested_voice)
            self.log.warning("No SAPI voice matches %r; looking for a British one instead.", self.requested_voice)

        for hint in BRITISH_VOICE_HINTS:
            for voice in voices:
                if hint in _voice_text(voice):
                    name = str(getattr(voice, "name", "") or hint)
                    self.log.debug("Selected the British SAPI voice %s (matched %r).", name, hint)
                    return str(getattr(voice, "id", "")), name
        self.log.info("No British SAPI voice is installed; using the system default voice.")
        return None

    # --- protocol ---------------------------------------------------------------------
    def available(self) -> bool:
        """True when pyttsx3 holds a live SAPI engine."""
        return self._engine is not None and not self._closed

    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]:
        """Render ``text`` to float32 mono via a temporary wav file.

        ``language`` is accepted for protocol compatibility; the SAPI voice speaks whatever
        language it was installed for. Returns ``(audio peak-limited to 1.0, sample_rate)``,
        or a short silence when synthesis fails — never an exception.
        """
        body = " ".join(str(text or "").split())
        if not body:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        if not self.available():
            self.log.error("The Windows voice was asked to speak but is unavailable: %s", self.error or "not loaded")
            return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate

        handle, wav_path = tempfile.mkstemp(prefix="jarvis-sapi-", suffix=".wav")
        os.close(handle)
        try:
            with self._lock:
                engine = self._engine
                if engine is None:  # closed between the check and the lock
                    return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate
                if not self._render(engine, body, wav_path):
                    return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate
            result = read_wav_file(wav_path)
            if result is None:
                self.log.error("The Windows voice wrote an empty wav file for %d characters.", len(body))
                return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate
            audio, rate = result
            if rate > 0:
                self.sample_rate = rate
            return peak_limit(audio), self.sample_rate
        except Exception as exc:
            self.log.error("Reading the Windows voice's wav file failed: %s", exc, exc_info=True)
            return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                self.log.debug("Could not remove the temporary file %s.", wav_path)

    def _render(self, engine: Any, text: str, wav_path: str) -> bool:
        """Drive save_to_file + runAndWait once, recovering from a stuck event loop."""
        for attempt in (1, 2):
            try:
                engine.save_to_file(text, wav_path)
                engine.runAndWait()
                return True
            except RuntimeError as exc:
                # "run loop already started" — end it and try once more.
                self.log.warning("SAPI event loop was busy (%s); resetting it.", exc)
                try:
                    engine.endLoop()
                except Exception:
                    self.log.debug("endLoop() failed on the SAPI engine.", exc_info=True)
                if attempt == 2:
                    return False
            except Exception as exc:
                self.log.error("The Windows voice failed to synthesize: %s", exc, exc_info=True)
                return False
        return False

    def close(self) -> None:
        """Stop the SAPI engine. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # pragma: no cover - best effort
                self.log.debug("Stopping the SAPI engine failed.", exc_info=True)
        self.log.debug("Windows SAPI engine closed.")

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "ready" if self.available() else f"unavailable: {self.error}"
        return f"SapiTTS(voice={self.voice_name or 'default'!r}, {state})"


def _voice_text(voice: Any) -> str:
    """All searchable text of a pyttsx3 voice (name, id and language tags), lower-cased."""
    parts: list[str] = [str(getattr(voice, "name", "") or ""), str(getattr(voice, "id", "") or "")]
    languages = getattr(voice, "languages", None) or []
    for language in languages:
        if isinstance(language, bytes):
            parts.append(language.decode("utf-8", errors="replace"))
        else:
            parts.append(str(language))
    return " ".join(parts).lower().replace("_", "-")
