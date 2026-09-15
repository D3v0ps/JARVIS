"""Piper speech synthesis — the Swedish voice.

Kokoro has no Swedish voice, so when JARVIS is spoken to in Swedish the Speaker asks
:func:`~jarvis.tts.engine.select_engine_for_language` for this engine instead. Piper runs
a small ONNX voice (``sv_SE-nst-medium``) and emits raw 16-bit PCM at the voice's native
rate, usually 22050 Hz.

Two back-ends, tried in this order and both imported/located lazily:

1. the ``piper`` Python package (``PiperVoice.load`` + ``synthesize_stream_raw``), which
   keeps the model resident and is by far the faster path;
2. the ``piper`` command-line executable, which writes a wav to a temporary file that is
   read back with the standard-library :mod:`wave` module.

A missing model file is reported as one actionable sentence naming
``scripts/fetch_models.py --swedish``; :meth:`available` is then ``False`` and the engine
chain moves on.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.tts.engine import NULL_SILENCE_S, peak_limit, read_wav_file, silence, to_float32_mono

__all__ = ["PiperTTS", "DEFAULT_SAMPLE_RATE"]

logger = logging.getLogger("jarvis.tts.piper")
#: Module logger alias, because ``__init__`` takes a ``logger`` argument that shadows it.
_LOG = logger

#: Native rate of the shipped Swedish voice; overridden by the voice's own JSON config.
DEFAULT_SAMPLE_RATE = 22_050
#: How long the command-line back-end may take for one utterance.
CLI_TIMEOUT_S = 60.0

_MODEL_HINT = "run 'python scripts/fetch_models.py --swedish' to download it"


class PiperTTS:
    """Piper voice, used for Swedish.

    Args:
        model_path: ``*.onnx`` voice; relative paths resolve against the project root.
        config_path: Optional ``*.onnx.json``; defaults to ``<model>.json`` beside it.
        speed: Speaking rate multiplier (Piper's ``length_scale`` is its reciprocal).
        executable: Name or path of the Piper CLI used when the package is missing.
        logger: Optional logger.

    Construction never raises: failures leave :meth:`available` ``False`` and
    :attr:`error` set to one plain sentence.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        speed: float = 1.0,
        executable: str = "piper",
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = "piper"
        self.speed = float(speed) if speed and float(speed) > 0 else 1.0
        self.log = logger if logger is not None else _LOG
        self.executable = str(executable or "piper")

        self.model_path = _resolve(model_path, "models/sv_SE-nst-medium.onnx")
        self.config_path = (
            Path(str(config_path)).expanduser()
            if config_path
            else Path(f"{self.model_path}.json")
        )
        self.sample_rate = _read_sample_rate(self.config_path, self.log)
        #: Spoken language ("sv" for the shipped voice), read by the Speaker when it
        #: decides whether the current voice fits the utterance.
        self.language = _read_language(self.config_path, self.model_path)

        #: One plain sentence explaining why the engine cannot speak ("" when it can).
        self.error: str = ""
        self._voice: Any | None = None
        self._cli: str | None = None
        self._closed = False
        # One synthesis at a time: the ONNX session is shared, the CLI writes a temp file.
        self._lock = threading.Lock()

        self._load()

    # --- loading ----------------------------------------------------------------------
    def _load(self) -> None:
        """Find a usable back-end, recording the reason in :attr:`error` when none is."""
        if not self.model_path.is_file():
            self.error = f"Piper Swedish voice not found at {self.model_path} — {_MODEL_HINT}."
            self.log.warning("%s", self.error)
            return

        voice = self._load_python_voice()
        if voice is not None:
            self._voice = voice
            self.error = ""
            self.log.info("Piper ready: %s at %d Hz (Python package).", self.model_path.name, self.sample_rate)
            return

        self._cli = shutil.which(self.executable)
        if self._cli:
            self.error = ""
            self.log.info("Piper ready: %s at %d Hz (command line %s).", self.model_path.name, self.sample_rate, self._cli)
            return

        self.error = (
            "Piper is installed nowhere I can see it: neither the 'piper-tts' Python package "
            f"nor a '{self.executable}' executable on PATH."
        )
        self.log.warning("%s", self.error)

    def _load_python_voice(self) -> Any | None:
        """Load ``PiperVoice`` from the ``piper`` package, or return ``None``."""
        piper_voice_cls: Any | None = None
        for module_name in ("piper", "piper.voice"):
            try:
                module = __import__(module_name, fromlist=["PiperVoice"])
            except ImportError:
                continue
            piper_voice_cls = getattr(module, "PiperVoice", None)
            if piper_voice_cls is not None:
                break
        if piper_voice_cls is None:
            self.log.debug("The 'piper' Python package is not installed; trying the executable.")
            return None

        try:
            if self.config_path.is_file():
                voice = piper_voice_cls.load(str(self.model_path), config_path=str(self.config_path))
            else:
                voice = piper_voice_cls.load(str(self.model_path))
        except Exception as exc:
            self.log.warning("Piper could not load %s: %s", self.model_path.name, exc, exc_info=True)
            return None

        rate = _voice_sample_rate(voice)
        if rate:
            self.sample_rate = rate
        return voice

    # --- protocol ---------------------------------------------------------------------
    def available(self) -> bool:
        """True when the model exists and either back-end is usable."""
        if self._closed or not self.model_path.is_file():
            return False
        return self._voice is not None or self._cli is not None

    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]:
        """Render ``text`` with the Swedish voice.

        ``language`` is accepted for protocol compatibility and ignored: a Piper voice
        speaks exactly the one language it was trained on.

        Returns ``(float32 mono peak-limited to 1.0, sample_rate)``; a failure returns a
        short silence rather than raising.
        """
        body = " ".join(str(text or "").split())
        if not body:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        if not self.available():
            self.log.error("Piper was asked to speak but is unavailable: %s", self.error or "not loaded")
            return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate

        with self._lock:
            if self._voice is not None:
                audio = self._synthesize_package(body)
                if audio is not None:
                    return peak_limit(audio), self.sample_rate
                self.log.info("Falling back to the Piper executable for this utterance.")
                if self._cli is None:
                    self._cli = shutil.which(self.executable)
            if self._cli:
                result = self._synthesize_cli(body)
                if result is not None:
                    audio, rate = result
                    return peak_limit(audio), rate

        self.log.error("Piper produced no audio for %d characters of text.", len(body))
        return silence(NULL_SILENCE_S, self.sample_rate), self.sample_rate

    # --- back-ends --------------------------------------------------------------------
    def _length_scale(self) -> float:
        """Piper's ``length_scale``: how long each phoneme lasts, so 1 / speed."""
        return 1.0 / self.speed if self.speed > 0 else 1.0

    def _synthesize_package(self, text: str) -> np.ndarray | None:
        """Use the in-process voice. Returns float32 samples, or ``None`` to fall back."""
        voice = self._voice
        if voice is None:
            return None
        length_scale = self._length_scale()

        streamer = getattr(voice, "synthesize_stream_raw", None)
        if callable(streamer):
            try:
                try:
                    stream = streamer(text, length_scale=length_scale)
                except TypeError:  # older signatures without keyword arguments
                    stream = streamer(text)
                raw = b"".join(bytes(chunk) for chunk in stream)
                if raw:
                    return to_float32_mono(np.frombuffer(raw, dtype=np.int16))
                self.log.warning("Piper returned an empty PCM stream.")
            except Exception as exc:
                self.log.warning("Piper stream synthesis failed: %s", exc, exc_info=True)

        # piper >= 1.3: synthesize() yields AudioChunk objects instead of raw bytes.
        chunked = getattr(voice, "synthesize", None)
        if callable(chunked):
            try:
                pieces: list[np.ndarray] = []
                for chunk in chunked(text):
                    raw_bytes = getattr(chunk, "audio_int16_bytes", None)
                    if raw_bytes:
                        pieces.append(to_float32_mono(np.frombuffer(raw_bytes, dtype=np.int16)))
                        continue
                    array = getattr(chunk, "audio_float_array", None)
                    if array is not None:
                        pieces.append(to_float32_mono(array))
                    rate = int(getattr(chunk, "sample_rate", 0) or 0)
                    if rate:
                        self.sample_rate = rate
                if pieces:
                    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            except Exception as exc:
                self.log.warning("Piper chunk synthesis failed: %s", exc, exc_info=True)
        return None

    def _synthesize_cli(self, text: str) -> tuple[np.ndarray, int] | None:
        """Run the Piper executable, writing a wav to a temp file and reading it back."""
        cli = self._cli
        if not cli:
            return None
        handle, wav_path = tempfile.mkstemp(prefix="jarvis-piper-", suffix=".wav")
        os.close(handle)
        command = [
            cli, "--model", str(self.model_path), "--output_file", wav_path,
            "--length-scale", f"{self._length_scale():.4f}",
        ]
        if self.config_path.is_file():
            command[3:3] = ["--config", str(self.config_path)]
        try:
            completed = subprocess.run(
                command, input=text.encode("utf-8"), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=CLI_TIMEOUT_S, check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                self.log.error("The Piper executable exited with %d: %s", completed.returncode, detail[:400])
                return None
            return read_wav_file(wav_path)
        except subprocess.TimeoutExpired:
            self.log.error("The Piper executable timed out after %.0f seconds.", CLI_TIMEOUT_S)
        except Exception as exc:
            self.log.error("Running the Piper executable failed: %s", exc, exc_info=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                self.log.debug("Could not remove the temporary file %s.", wav_path)
        return None

    def close(self) -> None:
        """Release the voice. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            voice, self._voice = self._voice, None
        closer = getattr(voice, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover - best effort
                self.log.debug("Closing the Piper voice failed.", exc_info=True)
        self.log.debug("Piper engine closed.")

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "ready" if self.available() else f"unavailable: {self.error}"
        return f"PiperTTS(model={self.model_path.name!r}, rate={self.sample_rate}, {state})"


# --- helpers --------------------------------------------------------------------------
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


def _read_language(config_path: Path, model_path: Path) -> str:
    """Two-letter language of the voice, from its JSON config or its file name."""
    try:
        if config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            code = str(data.get("language", {}).get("code", "") or data.get("espeak", {}).get("voice", ""))
            if code:
                return code.replace("_", "-").split("-")[0].lower()
    except Exception:
        logger.debug("Could not read the language from %s.", config_path, exc_info=True)
    stem = model_path.stem.replace("_", "-").split("-")[0].lower()
    return stem[:2] if len(stem) >= 2 else "sv"


def _read_sample_rate(config_path: Path, log: logging.Logger) -> int:
    """Read ``audio.sample_rate`` from the voice's JSON config, defaulting to 22050 Hz."""
    try:
        if config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            rate = int(data.get("audio", {}).get("sample_rate", 0))
            if rate > 0:
                return rate
    except Exception:
        log.debug("Could not read the sample rate from %s.", config_path, exc_info=True)
    return DEFAULT_SAMPLE_RATE


def _voice_sample_rate(voice: Any) -> int:
    """Best-effort read of the loaded voice's native rate across piper versions."""
    for holder in (getattr(voice, "config", None), voice):
        rate = getattr(holder, "sample_rate", None)
        try:
            if rate and int(rate) > 0:
                return int(rate)
        except (TypeError, ValueError):
            continue
    return 0
