"""Voice activity detection and utterance segmentation.

Two segmenters with one identical interface — ``push()`` / ``reset()`` / ``speaking`` /
``available`` / ``last_prob`` — so the assistant can swap one for the other without
knowing which it holds:

* :class:`SpeechSegmenter` runs Silero VAD. It is accurate about the difference
  between a voice and a door, and it is what JARVIS uses when the model loads.
* :class:`EnergySegmenter` is plain RMS against a calibrated noise floor. No model, no
  torch, numpy only — the fallback when Silero is missing.

Both share the same timing rules, implemented once in :class:`_BaseSegmenter`:

1. a pre-roll ring buffer of ``pre_roll_ms`` keeps the audio from *before* speech was
   detected, so the first syllable survives;
2. speech starts when the probability crosses ``threshold``;
3. an utterance needs ``min_speech_ms`` of speech to count — a cough is discarded;
4. it ends after ``silence_ms`` of continuous quiet;
5. and it is cut off at ``max_utterance_s`` no matter what.

The returned array is one float32 utterance, pre-roll included, and the segmenter
resets itself so the next :meth:`push` starts clean.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger("jarvis.stt.vad")

__all__ = ["SpeechSegmenter", "EnergySegmenter", "SILERO_CHUNK"]

#: Silero VAD accepts exactly this many samples per call at 16 kHz.
SILERO_CHUNK = 512

#: Calibration window for the energy fallback's noise floor.
_CALIBRATION_MS = 500

#: The noise floor is clamped into this range so neither digital silence nor someone
#: talking through the calibration window can produce a useless threshold.
_FLOOR_MIN_DB = -70.0
_FLOOR_MAX_DB = -40.0

#: How far above the noise floor counts as speech (the 0.5 probability point).
_SPEECH_MARGIN_DB = 10.0

#: Weight of each new quiet frame in the slow re-calibration of the noise floor.
_FLOOR_ADAPT = 0.02

_EPS = 1e-7


def _mono(frame: np.ndarray) -> np.ndarray:
    """Coerce anything frame-shaped into a private, contiguous 1-D float32 copy.

    The copy matters: buffered frames are kept until the utterance is complete, and the
    caller is free to reuse the array it handed us in the meantime.
    """
    data = np.asarray(frame)
    if data.ndim > 1:
        data = data.mean(axis=1) if data.shape[-1] > 1 else data.reshape(-1)
    return np.array(data, dtype=np.float32, copy=True).reshape(-1)


class _BaseSegmenter:
    """Timing and buffering shared by the Silero and energy segmenters."""

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15,
        pre_roll_ms: int = 300,
        sample_rate: int = 16000,
        logger: logging.Logger | None = None,
        *,
        name: str = "vad",
    ) -> None:
        self.threshold = float(threshold)
        self.silence_ms = int(silence_ms)
        self.min_speech_ms = int(min_speech_ms)
        self.max_utterance_s = float(max_utterance_s)
        self.pre_roll_ms = int(pre_roll_ms)
        self.sample_rate = int(sample_rate) or 16000
        self.log = logger if logger is not None else logging.getLogger(f"jarvis.stt.{name}")

        per_ms = self.sample_rate / 1000.0
        self._silence_target = max(1, int(self.silence_ms * per_ms))
        self._min_speech_target = max(1, int(self.min_speech_ms * per_ms))
        self._max_samples = max(1, int(self.max_utterance_s * self.sample_rate))
        self._pre_roll_target = max(0, int(self.pre_roll_ms * per_ms))

        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._speaking = False
        self._last_prob = 0.0

    # ----------------------------------------------------------------- internals

    def _probability(self, frame: np.ndarray) -> float:
        """Speech probability in ``0..1`` for one frame. Implemented by subclasses."""
        raise NotImplementedError

    def _reset_state(self) -> None:
        """Drop everything about the current utterance, keep any learned calibration."""
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._utterance = []
        self._utterance_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0
        self._speaking = False

    def reset(self) -> None:
        """Forget the current utterance and start listening from scratch."""
        self._reset_state()
        self._last_prob = 0.0

    def _remember(self, frame: np.ndarray) -> None:
        """Append ``frame`` to the pre-roll ring, keeping the newest ``pre_roll_ms``."""
        keep = max(self._pre_roll_target, frame.size)
        buffer = np.concatenate((self._pre_roll, frame)) if self._pre_roll.size else frame
        self._pre_roll = buffer[-keep:] if buffer.size > keep else buffer

    def _finish(self) -> np.ndarray:
        """Assemble the utterance, reset, and hand it to the caller."""
        parts = self._utterance
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self._reset_state()
        self.log.debug(
            "Utterance captured: %.2f s (%d samples).", audio.size / self.sample_rate, audio.size
        )
        return audio.astype(np.float32, copy=False)

    # --------------------------------------------------------------------- input

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame of float32 mono audio.

        Returns:
            The full utterance — pre-roll included — when the end of speech is reached
            or the hard cap is hit; otherwise ``None``.
        """
        data = _mono(frame)
        if data.size == 0:
            return None

        probability = self._probability(data)
        self._last_prob = probability
        is_speech = probability >= self.threshold

        if not self._speaking:
            self._remember(data)
            if not is_speech:
                return None
            # Seed the utterance with the pre-roll, which already ends with this frame.
            start = self._pre_roll
            self._pre_roll = np.zeros(0, dtype=np.float32)
            self._utterance = [start]
            self._utterance_samples = int(start.size)
            self._speech_samples = int(data.size)
            self._silence_samples = 0
            self._speaking = True
            self.log.debug("Speech started (probability %.2f).", probability)
            return None

        self._utterance.append(data)
        self._utterance_samples += int(data.size)
        if is_speech:
            self._speech_samples += int(data.size)
            self._silence_samples = 0
        else:
            self._silence_samples += int(data.size)

        if self._utterance_samples >= self._max_samples:
            self.log.info("Utterance hit the %.0f second cap; cutting it off.", self.max_utterance_s)
            return self._finish()

        if self._silence_samples >= self._silence_target:
            if self._speech_samples >= self._min_speech_target:
                return self._finish()
            self.log.debug(
                "Discarding %d ms of noise — shorter than the %d ms minimum.",
                int(self._speech_samples * 1000 / self.sample_rate),
                self.min_speech_ms,
            )
            self._reset_state()
            return None

        return None

    # ---------------------------------------------------------------- properties

    @property
    def speaking(self) -> bool:
        """True while an utterance is being collected."""
        return self._speaking

    @property
    def available(self) -> bool:
        """True when this segmenter can actually judge speech."""
        return True

    @property
    def last_prob(self) -> float:
        """Speech probability of the most recent frame."""
        return self._last_prob


class SpeechSegmenter(_BaseSegmenter):
    """Silero VAD over 16 kHz float32 frames. Feed frames, get a finished utterance.

    Silero insists on exactly 512-sample chunks, while the microphone delivers 1280
    (80 ms). Frames are therefore buffered internally, the model is run over every
    complete 512-sample slice, and the frame's probability is the maximum across its
    slices — the caller's block size never has to match the model's.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15,
        pre_roll_ms: int = 300,
        sample_rate: int = 16000,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            threshold, silence_ms, min_speech_ms, max_utterance_s, pre_roll_ms,
            sample_rate, logger, name="vad",
        )
        self._model: Any | None = None
        self._torch: Any | None = None
        self._available = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._warned = False
        self._load()

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        """Import Silero lazily. Never raises; failure only means ``available`` is False."""
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:
            self.log.warning("torch is unavailable (%s); falling back to energy VAD.", exc)
            return

        model: Any | None = None
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-not-found]

            model = load_silero_vad()
            source = "silero-vad package"
        except Exception as exc:
            self.log.debug("silero_vad package unavailable (%s); trying torch.hub.", exc)
            try:
                model = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
                source = "torch.hub"
            except Exception as hub_exc:
                self.log.warning(
                    "Silero VAD could not be loaded (%s); falling back to energy VAD.", hub_exc
                )
                return

        if isinstance(model, tuple):  # torch.hub returns (model, utils)
            model = model[0]
        if model is None:
            return

        self._torch = torch
        self._model = model
        self._available = True
        self.log.info(
            "Silero VAD ready (%s): threshold %.2f, silence %d ms, pre-roll %d ms.",
            source, self.threshold, self.silence_ms, self.pre_roll_ms,
        )

    # ---------------------------------------------------------------- inference

    def _infer(self, chunk: np.ndarray) -> float:
        """Run one 512-sample chunk through the model and return its probability."""
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            return 0.0
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
                output = model(tensor, self.sample_rate)
            item = getattr(output, "item", None)
            return float(item()) if callable(item) else float(output)
        except Exception as exc:
            self._available = False
            self.log.error("Silero VAD inference failed: %s", exc, exc_info=True)
            return 0.0

    def _probability(self, frame: np.ndarray) -> float:
        """Maximum Silero probability across the complete 512-sample slices of ``frame``."""
        if not self._available:
            if not self._warned:
                self._warned = True
                self.log.warning("Silero VAD is unavailable; use EnergySegmenter instead.")
            return 0.0

        buffer = np.concatenate((self._pending, frame)) if self._pending.size else frame
        best: float | None = None
        offset = 0
        while buffer.size - offset >= SILERO_CHUNK:
            probability = self._infer(buffer[offset:offset + SILERO_CHUNK])
            offset += SILERO_CHUNK
            best = probability if best is None else max(best, probability)
        self._pending = buffer[offset:].copy()
        # Frames shorter than one chunk keep the previous probability rather than
        # dropping to zero and cutting an utterance in half.
        return self._last_prob if best is None else best

    def reset(self) -> None:
        """Clear the buffers and the model's own recurrent state."""
        super().reset()
        self._pending = np.zeros(0, dtype=np.float32)
        model = self._model
        reset_states = getattr(model, "reset_states", None) if model is not None else None
        if callable(reset_states):
            try:
                reset_states()
            except Exception as exc:  # pragma: no cover - defensive
                self.log.debug("Silero reset_states() failed: %s", exc)

    def _finish(self) -> np.ndarray:
        audio = super()._finish()
        self.reset()
        return audio

    @property
    def available(self) -> bool:
        """False when Silero could not be loaded — the caller falls back to RMS."""
        return self._available


class EnergySegmenter(_BaseSegmenter):
    """RMS-threshold fallback with the same push()/reset()/speaking interface.

    Used when silero-vad is missing or fails to load. The noise floor is calibrated
    from the first ~500 ms of audio and then nudged slowly during every quiet frame, so
    a fan spinning up does not slowly deafen JARVIS. Speech is anything
    ``_SPEECH_MARGIN_DB`` above that floor; the reported probability reaches 0.5 exactly
    at that point, so the same ``threshold`` value means the same thing as for Silero.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15,
        pre_roll_ms: int = 300,
        sample_rate: int = 16000,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            threshold, silence_ms, min_speech_ms, max_utterance_s, pre_roll_ms,
            sample_rate, logger, name="vad.energy",
        )
        self._calibration: list[float] = []
        self._calibration_samples = 0
        self._calibration_target = max(1, int(_CALIBRATION_MS * self.sample_rate / 1000))
        self._floor_db = _FLOOR_MAX_DB
        self._calibrated = False

    @staticmethod
    def _decibels(frame: np.ndarray) -> float:
        """RMS of a frame in dBFS, floored so digital silence stays finite."""
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, _EPS))

    def _probability(self, frame: np.ndarray) -> float:
        level = self._decibels(frame)

        if not self._calibrated:
            self._calibration.append(level)
            self._calibration_samples += int(frame.size)
            if self._calibration_samples < self._calibration_target:
                return 0.0
            median = float(np.median(np.asarray(self._calibration, dtype=np.float64)))
            self._floor_db = min(max(median, _FLOOR_MIN_DB), _FLOOR_MAX_DB)
            self._calibrated = True
            self._calibration.clear()
            self.log.debug("Noise floor calibrated at %.1f dBFS.", self._floor_db)

        probability = 0.5 * (level - self._floor_db) / _SPEECH_MARGIN_DB
        probability = min(1.0, max(0.0, probability))

        # Re-calibrate slowly, but only on quiet frames outside an utterance.
        if not self._speaking and level < self._floor_db + _SPEECH_MARGIN_DB:
            adapted = (1.0 - _FLOOR_ADAPT) * self._floor_db + _FLOOR_ADAPT * level
            self._floor_db = min(max(adapted, _FLOOR_MIN_DB), _FLOOR_MAX_DB)

        return probability

    @property
    def noise_floor_db(self) -> float:
        """The current noise floor in dBFS, for the logs and the preflight report."""
        return self._floor_db
