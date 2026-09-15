"""Speech-to-text with faster-whisper.

The transcriber tries CUDA first and falls back to CPU ``int8`` without drama: a broken
GPU driver, a missing cuDNN DLL or an out-of-memory error must never stop the assistant
from hearing its operator. ``faster_whisper`` itself is imported lazily inside
:meth:`Transcriber.load`, so this module imports fine on a machine that has no CUDA, no
model files and no faster-whisper installed at all.

Whisper is famous for inventing confident nonsense out of silence — "Thank you.",
"Tack för att du tittade!", "Undertexter av ...", "[Music]" — because those sentences are
everywhere in its subtitle training data. :data:`HALLUCINATION_PATTERNS` plus the
duration/confidence gate in :func:`is_hallucination` throw those away, which is the
difference between an assistant that feels calm and one that feels possessed.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "Transcript", "Transcriber", "pick_model", "add_cuda_dll_paths",
    "is_hallucination", "HALLUCINATION_PATTERNS",
]

logger = logging.getLogger("jarvis.stt.transcriber")

#: Whisper expects 16 kHz mono audio; anything else is resampled before decoding.
WHISPER_SAMPLE_RATE = 16000

#: Segments whose "no speech" probability is above this are treated as silence by whisper.
NO_SPEECH_THRESHOLD = 0.6

#: An utterance shorter than this is too short to carry a real subtitle credit.
HALLUCINATION_MAX_DURATION_S = 1.2

#: Above this "no speech" probability the decoder itself doubts there was any speech.
HALLUCINATION_NO_SPEECH_PROB = 0.6

#: Below this average log probability the decoder was guessing rather than hearing.
HALLUCINATION_AVG_LOGPROB = -0.8

# --- hallucination patterns ----------------------------------------------------------
# Matched against the *normalised* transcript: lower-cased, stripped of surrounding
# quotes and trailing ".,!?…", whitespace collapsed. Every entry is a full match.
HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    # Bracketed sound tags: "[Music]", "(upbeat music)", "[BLANK_AUDIO]", "♪ ♪".
    re.compile(r"[\[\(\*].{0,60}[\]\)\*]"),
    re.compile(r"[♪♫~\-_.·•]+"),
    # English: the single most common silence hallucination of them all.
    re.compile(r"(thank you|thanks|thank you so much|thanks a lot)"),
    # English YouTube outro credits baked into the training subtitles.
    re.compile(r"thank(s| you) for watching.{0,30}"),
    re.compile(r"(please )?(don't forget to )?(like,? )?(and )?subscribe.{0,40}"),
    re.compile(r"see you (in the )?next (time|video|one).{0,20}"),
    # English subtitle-credit boilerplate ("Subtitles by the Amara.org community").
    re.compile(r"(subtitles?|captions?|transcription|translation)( by| from|:).{0,60}"),
    re.compile(r".{0,40}amara\.org.{0,40}"),
    # Short English filler whisper emits over room tone.
    re.compile(r"(you|bye|okay|ok|hello|hi|yeah|yes|no|uh|um|hmm|mm|so|the end|amen|goodbye)"),
    re.compile(r"(bye bye|good bye|okay okay|you know|all right|alright)"),
    # Swedish: the direct counterparts, equally common in Nordic subtitle corpora.
    re.compile(r"tack( så mycket| ska du ha)?"),
    re.compile(r"tack för att (du|ni) (tittade|såg på|lyssnade).{0,30}"),
    re.compile(r"tack för (tittandet|tittningen|att du tittar).{0,30}"),
    # Swedish subtitling credits: "Undertexter av ...", "Textning: ...", "Översättning: ...".
    re.compile(r"(undertexter|undertextning|textning|översättning|svensk text|synk)( av| från|:| ).{0,60}"),
    re.compile(r".{0,40}(btistudios|nordisk undertext|sdi media|iyuno|svt text).{0,40}"),
    # Swedish short filler and sign-offs.
    re.compile(r"(hej|hej då|hejdå|ja|nej|jaha|okej|vi ses|ha det bra|god natt|förlåt|precis)"),
    re.compile(r"(vi ses i nästa (video|avsnitt)|prenumerera.{0,30})"),
    re.compile(r"(musik|musik spelar|applåder|skratt)"),
]

_CUDA_DLL_LOCK = threading.Lock()
_CUDA_DLL_DONE = False


# --- helpers -------------------------------------------------------------------------
def pick_model(vram_gb: float | None) -> str:
    """Choose a whisper model size for the available VRAM.

    ``>= 10 GB`` gets ``medium``, ``>= 6 GB`` gets ``small``, anything smaller gets
    ``base``. ``None`` (no GPU detected) gets ``small`` — CPU ``int8`` is still fast
    enough for one short utterance at a time.
    """
    if vram_gb is None:
        return "small"
    try:
        gigabytes = float(vram_gb)
    except (TypeError, ValueError):
        logger.debug("Unusable vram_gb=%r; assuming no GPU", vram_gb)
        return "small"
    if gigabytes >= 10:
        return "medium"
    if gigabytes >= 6:
        return "small"
    return "base"


def _site_package_dirs() -> list[Path]:
    """Every directory that may hold the ``nvidia`` wheel packages of this interpreter."""
    raw: list[str] = []
    try:
        import site

        raw += [str(entry) for entry in (site.getsitepackages() or [])]
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            raw.append(user_site)
    except Exception:
        logger.debug("site module gave no usable package directories", exc_info=True)
    try:
        import sysconfig

        paths = sysconfig.get_paths()
        raw += [entry for entry in (paths.get("purelib"), paths.get("platlib")) if entry]
    except Exception:
        logger.debug("sysconfig gave no usable package directories", exc_info=True)
    seen: set[str] = set()
    return [Path(entry) for entry in raw if not (entry.lower() in seen or seen.add(entry.lower()))]


def add_cuda_dll_paths(logger: logging.Logger | None = None) -> None:
    """Make the pip-installed CUDA libraries visible to faster-whisper on Windows.

    ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` drop their DLLs in
    ``site-packages/nvidia/<lib>/bin``, which is not on the DLL search path. Every
    existing directory is registered with :func:`os.add_dll_directory` and appended to
    ``PATH`` so child processes see it too. No-op on every other platform and silent
    about anything that goes wrong — a missing CUDA install is not an error here.
    """
    global _CUDA_DLL_DONE
    log = logger if logger is not None else globals()["logger"]
    if sys.platform != "win32":
        log.debug("add_cuda_dll_paths: not Windows, nothing to do")
        return
    with _CUDA_DLL_LOCK:
        if _CUDA_DLL_DONE:
            return
        added: list[str] = []
        try:
            for base in _site_package_dirs():
                for library in ("cublas", "cudnn"):
                    dll_dir = base / "nvidia" / library / "bin"
                    try:
                        if not dll_dir.is_dir():
                            continue
                        add_dll_directory = getattr(os, "add_dll_directory", None)
                        if add_dll_directory is not None:
                            add_dll_directory(str(dll_dir))
                        current = os.environ.get("PATH", "")
                        if str(dll_dir).lower() not in current.lower():
                            os.environ["PATH"] = f"{current}{os.pathsep}{dll_dir}" if current else str(dll_dir)
                        added.append(str(dll_dir))
                    except Exception as exc:
                        log.debug("Could not register CUDA DLL directory %s: %s", dll_dir, exc)
            if added:
                log.info("Registered %d CUDA DLL directory/directories: %s", len(added), ", ".join(added))
            else:
                log.debug("No nvidia/cublas/bin or nvidia/cudnn/bin found in site-packages")
        except Exception as exc:  # pragma: no cover - defensive, must never raise
            log.debug("add_cuda_dll_paths failed: %s", exc, exc_info=True)
        _CUDA_DLL_DONE = True


def _normalise(text: str) -> str:
    """Lower-case, unquote and strip trailing punctuation for pattern matching."""
    clean = " ".join(str(text or "").split()).strip().strip("\"'“”«»")
    clean = clean.strip(" .,!?…:-—–")
    return clean.lower()


def is_hallucination(
    text: str,
    *,
    duration_s: float,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
) -> bool:
    """True when ``text`` looks like whisper talking to itself over silence.

    A known hallucination phrase is only discarded when the evidence is weak as well:
    the utterance was shorter than :data:`HALLUCINATION_MAX_DURATION_S`, the decoder's
    own ``no_speech_prob`` was high, or its average log probability was poor. A real
    "thank you" spoken inside a normal-length, confident utterance survives.
    """
    normalised = _normalise(text)
    if not normalised:
        return True
    matched = any(pattern.fullmatch(normalised) for pattern in HALLUCINATION_PATTERNS)
    if not matched:
        return False
    too_short = float(duration_s or 0.0) < HALLUCINATION_MAX_DURATION_S
    silent = no_speech_prob is not None and float(no_speech_prob) >= HALLUCINATION_NO_SPEECH_PROB
    unsure = avg_logprob is not None and float(avg_logprob) <= HALLUCINATION_AVG_LOGPROB
    return bool(too_short or silent or unsure)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resampler — good enough for speech heading into whisper."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    target_length = max(1, int(round(audio.size * (float(dst_rate) / float(src_rate)))))
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Whatever the audio loop hands us, give faster-whisper contiguous mono float32."""
    array = np.asarray(audio)
    if array.ndim > 1:
        array = array.mean(axis=tuple(range(1, array.ndim)))
    if array.dtype != np.float32:
        if np.issubdtype(array.dtype, np.integer):
            max_value = float(np.iinfo(array.dtype).max) or 1.0
            array = array.astype(np.float32) / max_value
        else:
            array = array.astype(np.float32)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)


# --- data ----------------------------------------------------------------------------
@dataclass
class Transcript:
    """One decoded utterance. ``text`` is "" when nothing usable was heard."""

    text: str = ""
    language: str = ""
    duration_s: float = 0.0
    latency_ms: float = 0.0


# --- transcriber ---------------------------------------------------------------------
class Transcriber:
    """faster-whisper. CUDA first, CPU int8 fallback without drama."""

    def __init__(
        self,
        model: str = "auto",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
        language: str | None = None,
        vram_gb: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = str(model or "auto")
        self.device = str(device or "auto")
        self.compute_type = str(compute_type or "auto")
        self.beam_size = max(1, int(beam_size or 1))
        self.language = None if not language or str(language).lower() == "auto" else str(language)
        self.vram_gb = vram_gb
        self.log = logger if logger is not None else globals()["logger"]

        self._backend: Any | None = None
        self._lock = threading.RLock()
        self._device_in_use = ""
        self._compute_in_use = ""
        self._model_in_use = ""
        self._load_seconds = 0.0
        self._load_error_logged = False

    # --- properties ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """True once a whisper model is loaded and able to transcribe."""
        return self._backend is not None

    @property
    def device_in_use(self) -> str:
        """``"cuda"`` or ``"cpu"`` once loaded, ``""`` before the first :meth:`load`."""
        return self._device_in_use

    @property
    def model_in_use(self) -> str:
        """The resolved model size, e.g. ``"small"`` when ``model`` was ``"auto"``."""
        return self._model_in_use

    @property
    def load_seconds(self) -> float:
        """How long the last successful model load took, in seconds."""
        return self._load_seconds

    # --- loading ---------------------------------------------------------------------
    def _resolve_settings(self) -> tuple[str, str, str]:
        """Turn the ``auto`` values into a concrete (model, device, compute_type)."""
        size = self.model if self.model and self.model != "auto" else pick_model(self.vram_gb)
        device = self.device.lower()
        if device not in ("cuda", "cpu"):
            # "auto": only reach for CUDA when preflight actually found a GPU.
            device = "cuda" if self.vram_gb else "cpu"
        compute = self.compute_type.lower()
        if compute in ("", "auto"):
            compute = "float16" if device == "cuda" else "int8"
        return size, device, compute

    def load(self) -> None:
        """Load the whisper model. Blocking, thread-safe and a no-op when already loaded."""
        with self._lock:
            if self._backend is not None:
                return
            size, device, compute = self._resolve_settings()
            started = time.perf_counter()
            add_cuda_dll_paths(self.log)

            from faster_whisper import WhisperModel  # local import: heavy, optional on Linux

            backend: Any | None = None
            if device == "cuda":
                try:
                    backend = WhisperModel(size, device="cuda", compute_type=compute)
                except Exception as exc:
                    self.log.warning(
                        "CUDA unavailable (%s) - falling back to CPU int8", self._reason(exc)
                    )
                    device, compute, backend = "cpu", "int8", None
            if backend is None:
                backend = WhisperModel(size, device="cpu", compute_type=compute)

            self._backend = backend
            self._device_in_use = device
            self._compute_in_use = compute
            self._model_in_use = size
            self._load_seconds = time.perf_counter() - started
            self.log.info(
                "Whisper model '%s' loaded on %s/%s in %.1f s",
                size, device, compute, self._load_seconds,
            )

    @staticmethod
    def _reason(exc: BaseException) -> str:
        """One short line describing why CUDA said no."""
        message = " ".join(str(exc).split())
        if not message:
            message = exc.__class__.__name__
        return message if len(message) <= 160 else f"{message[:160]}…"

    # --- transcription ---------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        """Decode one utterance. Never raises — on failure the text comes back empty."""
        started = time.perf_counter()
        duration_s = 0.0
        try:
            samples = _as_mono_float32(audio)
            rate = int(sample_rate or WHISPER_SAMPLE_RATE)
            duration_s = float(samples.size) / float(rate) if rate > 0 else 0.0
            if samples.size == 0:
                return Transcript("", self.language or "", 0.0, self._elapsed_ms(started))
            samples = _resample(samples, rate, WHISPER_SAMPLE_RATE)

            if self._backend is None:
                self.load()
            backend = self._backend
            if backend is None:  # pragma: no cover - load() raises instead
                raise RuntimeError("whisper model is not loaded")

            segments, info = backend.transcribe(
                samples,
                beam_size=self.beam_size,
                language=self.language,          # None -> whisper detects the language
                vad_filter=False,                # Silero already segmented the utterance
                condition_on_previous_text=False,  # stop it continuing the previous sentence
                no_speech_threshold=NO_SPEECH_THRESHOLD,
            )
            text, no_speech_prob, avg_logprob = self._collect(segments)
            language = str(getattr(info, "language", "") or self.language or "")

            if text and is_hallucination(
                text, duration_s=duration_s, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob
            ):
                self.log.debug(
                    "Discarded likely hallucination %r (%.2f s, no_speech=%s, logprob=%s)",
                    text, duration_s, no_speech_prob, avg_logprob,
                )
                text = ""

            latency_ms = self._elapsed_ms(started)
            self.log.debug(
                "Transcribed %.2f s of audio in %.0f ms (%s): %r",
                duration_s, latency_ms, language or "?", text,
            )
            return Transcript(text, language, duration_s, latency_ms)
        except Exception as exc:
            self._log_failure(exc)
            return Transcript("", self.language or "", duration_s, self._elapsed_ms(started))

    def _collect(self, segments: Iterable[Any]) -> tuple[str, float | None, float | None]:
        """Join the segment texts and keep the worst confidence numbers seen."""
        parts: list[str] = []
        no_speech: float | None = None
        logprob: float | None = None
        for segment in segments:
            piece = str(getattr(segment, "text", "") or "")
            if piece.strip():
                parts.append(piece.strip())
            value = getattr(segment, "no_speech_prob", None)
            if value is not None:
                no_speech = float(value) if no_speech is None else max(no_speech, float(value))
            value = getattr(segment, "avg_logprob", None)
            if value is not None:
                logprob = float(value) if logprob is None else min(logprob, float(value))
        return " ".join(" ".join(parts).split()).strip(), no_speech, logprob

    def _log_failure(self, exc: BaseException) -> None:
        """Report a decoding failure loudly once, quietly afterwards."""
        if not self._load_error_logged:
            self._load_error_logged = True
            self.log.error("Transcription failed: %s", self._reason(exc), exc_info=True)
        else:
            self.log.debug("Transcription failed again: %s", self._reason(exc))

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000.0
