"""Wake-word detection with openWakeWord.

:class:`WakeWord` wraps openWakeWord's ``Model`` around the bundled ``hey_jarvis``
network and turns the stream of microphone frames into a single boolean event: "JARVIS
was just addressed". The library wants 16 kHz **int16** PCM while the rest of the
pipeline speaks float32, so the conversion happens here and nowhere else.

Three details matter for a detector that has to behave in a living room:

* the score is read from the prediction key openWakeWord actually used — which may
  carry a version suffix (``hey_jarvis_v0.1``) or a full path — resolved once by
  matching the configured model name inside the keys;
* ``wake.sensitivity`` is applied as the score threshold and ``wake.cooldown`` keeps a
  single spoken "hey Jarvis" from firing twice;
* the model's internal feature buffers are reset after every detection, so the tail of
  the wake word cannot leak into the next one.

Nothing here raises. A missing library, a missing model file or a failed download all
end in :attr:`WakeWord.available` being ``False`` and :meth:`WakeWord.process`
returning ``False`` forever, so the assistant still starts in text mode.

``openwakeword`` and ``onnxruntime`` are imported inside the methods that need them,
never at module import time.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("jarvis.wake.detector")

__all__ = ["WakeWord", "DEFAULT_MODEL", "DEFAULT_FRAMEWORK", "INT16_SCALE"]

#: Bundled openWakeWord model used by JARVIS.
DEFAULT_MODEL = "hey_jarvis"

#: Inference backend. ONNX runs everywhere we care about; "tflite" is the alternative.
DEFAULT_FRAMEWORK = "onnx"

#: float32 [-1, 1] -> int16 conversion factor.
INT16_SCALE = 32767.0

#: Shared feature extractors openWakeWord needs next to any wake-word model.
_FEATURE_MODELS = ("melspectrogram", "embedding_model")

#: Consecutive inference failures tolerated before the detector switches itself off.
_MAX_ERRORS = 5


def _framework_suffix(framework: str) -> str:
    """Return the model file extension used by ``framework``."""
    return ".tflite" if str(framework).strip().lower() == "tflite" else ".onnx"


def _openwakeword_model_dir() -> Path | None:
    """Return the directory openWakeWord downloads its models into, if importable."""
    try:
        import openwakeword  # type: ignore[import-not-found]
    except Exception:
        return None
    package_file = getattr(openwakeword, "__file__", None)
    if not package_file:
        return None
    return Path(package_file).resolve().parent / "resources" / "models"


class WakeWord:
    """openWakeWord with the bundled 'hey_jarvis' model, ONNX backend.

    Args:
        model: Model name as bundled with openWakeWord (``"hey_jarvis"``), or a path
            to a custom ``.onnx``/``.tflite`` model.
        sensitivity: Score threshold in ``0..1``. Raise it if the TV wakes JARVIS,
            lower it if he misses you.
        framework: ``"onnx"`` or ``"tflite"``.
        cooldown: Seconds during which a second detection is suppressed.
        logger: Optional logger; a module logger is used when omitted.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        sensitivity: float = 0.5,
        framework: str = DEFAULT_FRAMEWORK,
        cooldown: float = 2.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = str(model or DEFAULT_MODEL)
        self.sensitivity = float(sensitivity)
        self.framework = str(framework or DEFAULT_FRAMEWORK).strip().lower()
        self.cooldown = max(0.0, float(cooldown))
        self.log = logger if logger is not None else logging.getLogger("jarvis.wake.detector")

        self._model: Any | None = None
        self._available = False
        self._model_key: str | None = None
        self._key_logged = False
        self._last_score = 0.0
        self._last_detection = 0.0
        self._models_checked = False
        self._errors = 0

        self._load()

    # ------------------------------------------------------------------ loading

    def ensure_models(self) -> None:
        """Download the openWakeWord models on first run. Idempotent, never raises.

        The download is skipped entirely when the model file (and the shared feature
        extractors) are already on disk, so a normal start costs one ``listdir``.
        Any failure is logged and leaves :attr:`available` ``False``.
        """
        if self._models_checked:
            return
        self._models_checked = True

        try:
            custom = Path(self.model).expanduser()
            if custom.suffix and custom.is_file():
                self.log.debug("Using custom wake-word model file %s.", custom)
                return

            directory = _openwakeword_model_dir()
            if directory is None:
                self.log.warning(
                    "openWakeWord is not installed; wake-word models cannot be prepared."
                )
                self._available = False
                return

            if self._models_present(directory):
                self.log.debug("Wake-word models already present in %s.", directory)
                return

            self.log.info("Downloading openWakeWord models into %s (first run only).", directory)
            from openwakeword.utils import download_models  # type: ignore[import-not-found]

            try:
                download_models(model_names=[self.model])
            except TypeError:
                # Older/newer signatures: fall back to downloading the full bundle.
                download_models()
            self.log.info("openWakeWord models downloaded.")
        except Exception as exc:
            self.log.error("Could not prepare the wake-word models: %s", exc, exc_info=True)
            self._available = False

    def _models_present(self, directory: Path) -> bool:
        """True when the wake-word model and both feature extractors exist locally."""
        try:
            if not directory.is_dir():
                return False
            names = [path.name for path in directory.iterdir() if path.is_file()]
        except OSError as exc:
            self.log.debug("Cannot inspect the wake-word model directory %s: %s", directory, exc)
            return False

        suffix = _framework_suffix(self.framework)
        wanted = Path(self.model).stem.lower()
        has_wake = any(name.lower().startswith(wanted) and name.lower().endswith(suffix) for name in names)
        has_features = all(
            any(name.lower().startswith(feature) for name in names) for feature in _FEATURE_MODELS
        )
        return has_wake and has_features

    def _load(self) -> None:
        """Import openWakeWord and build the model. Never raises."""
        try:
            from openwakeword.model import Model  # type: ignore[import-not-found]
        except Exception as exc:
            self.log.warning(
                "openWakeWord is unavailable (%s); JARVIS will run without a wake word.", exc
            )
            return

        self.ensure_models()

        for spec in self._model_specs():
            try:
                self._model = Model(wakeword_models=[spec], inference_framework=self.framework)
            except Exception as exc:
                self.log.debug("Could not load wake-word model %r: %s", spec, exc)
                continue
            self._available = True
            self.log.info(
                "Wake word ready: model=%s, framework=%s, sensitivity=%.2f, cooldown=%.1f s.",
                spec,
                self.framework,
                self.sensitivity,
                self.cooldown,
            )
            return

        self.log.error(
            "Could not load the wake-word model %r; wake-word detection is disabled.", self.model
        )

    def _model_specs(self) -> list[str]:
        """Candidate model identifiers: the configured name, then a resolved file path."""
        specs: list[str] = [self.model]
        custom = Path(self.model).expanduser()
        if custom.suffix and custom.is_file():
            specs.insert(0, str(custom))
            return specs

        directory = _openwakeword_model_dir()
        if directory is None:
            return specs
        suffix = _framework_suffix(self.framework)
        wanted = Path(self.model).stem.lower()
        try:
            for path in sorted(directory.glob(f"*{suffix}")):
                if path.name.lower().startswith(wanted):
                    specs.append(str(path))
        except OSError:
            pass
        return specs

    # ----------------------------------------------------------------- detection

    def process(self, frame: np.ndarray) -> bool:
        """Feed one frame and report whether the wake word just fired.

        Args:
            frame: float32 mono audio at 16 kHz, nominally 1280 samples (80 ms).

        Returns:
            ``True`` exactly once per detection — the cooldown swallows the repeats,
            and the model's buffers are cleared so the next detection starts clean.
        """
        if not self._available or self._model is None:
            return False

        chunk = self._to_int16(frame)
        if chunk.size == 0:
            return False

        try:
            predictions = self._model.predict(chunk)
            self._errors = 0
        except Exception as exc:
            self._errors += 1
            self.log.error("Wake-word inference failed: %s", exc, exc_info=self._errors == 1)
            if self._errors >= _MAX_ERRORS:
                self._available = False
                self.log.error(
                    "Wake-word detection disabled after %d consecutive failures.", self._errors
                )
            return False

        score = self._score(predictions)
        self._last_score = score
        if score < self.sensitivity:
            return False

        now = time.monotonic()
        if self._last_detection and (now - self._last_detection) < self.cooldown:
            self.log.debug("Wake word suppressed by the cooldown (score %.2f).", score)
            return False

        self._last_detection = now
        self.log.info("Wake word detected (score %.2f).", score)
        self._reset_model()
        return True

    def _score(self, predictions: Any) -> float:
        """Read the score for the configured model out of a prediction mapping."""
        if not isinstance(predictions, dict) or not predictions:
            return 0.0
        key = self._resolve_key(predictions)
        if key is None:
            return 0.0
        try:
            return float(predictions[key])
        except (TypeError, ValueError):
            return 0.0

    def _resolve_key(self, predictions: dict) -> str | None:
        """Match the configured model name against openWakeWord's prediction keys.

        The key can be the bare name, a versioned name (``hey_jarvis_v0.1``) or a full
        path, so the name is matched as a substring of the key's stem. The result is
        cached and logged once.
        """
        cached = self._model_key
        if cached is not None and cached in predictions:
            return cached

        wanted = Path(self.model).stem.lower()
        chosen: str | None = None
        if self.model in predictions:
            chosen = self.model
        else:
            for key in predictions:
                stem = Path(str(key)).stem.lower()
                if wanted in stem or stem in wanted:
                    chosen = str(key)
                    break
        if chosen is None and len(predictions) == 1:
            chosen = str(next(iter(predictions)))

        if chosen is None:
            if not self._key_logged:
                self._key_logged = True
                self.log.error(
                    "No wake-word score for %r; openWakeWord reported %s.",
                    self.model,
                    sorted(str(key) for key in predictions),
                )
            return None

        self._model_key = chosen
        if not self._key_logged:
            self._key_logged = True
            self.log.info("Wake-word prediction key resolved to %r.", chosen)
        return chosen

    def _reset_model(self) -> None:
        """Clear openWakeWord's internal buffers so detections cannot overlap."""
        model = self._model
        if model is None:
            return
        reset = getattr(model, "reset", None)
        if callable(reset):
            try:
                reset()
                return
            except Exception as exc:
                self.log.debug("openWakeWord reset() failed: %s", exc)
        buffer = getattr(model, "prediction_buffer", None)
        if isinstance(buffer, dict):
            for scores in buffer.values():
                try:
                    scores.clear()
                except Exception:  # pragma: no cover - defensive
                    pass

    def reset(self) -> None:
        """Forget the last detection and clear the model buffers."""
        self._last_score = 0.0
        self._last_detection = 0.0
        self._errors = 0
        self._reset_model()

    @staticmethod
    def _to_int16(frame: np.ndarray) -> np.ndarray:
        """Convert a float32 mono frame in [-1, 1] to the int16 PCM openWakeWord wants."""
        data = np.asarray(frame)
        if data.ndim > 1:
            data = data.mean(axis=1) if data.shape[-1] > 1 else data.reshape(-1)
        data = np.ascontiguousarray(data).reshape(-1)
        if data.size == 0:
            return np.zeros(0, dtype=np.int16)
        if data.dtype == np.int16:
            return data
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / INT16_SCALE
        return (np.clip(data.astype(np.float32), -1.0, 1.0) * INT16_SCALE).astype(np.int16)

    # ---------------------------------------------------------------- properties

    @property
    def last_score(self) -> float:
        """Score of the most recent frame, for the overlay and the logs."""
        return self._last_score

    @property
    def available(self) -> bool:
        """False when the library or the model could not be loaded."""
        return self._available

    @property
    def model_key(self) -> str | None:
        """The prediction key openWakeWord uses for this model, once resolved."""
        return self._model_key

    def __repr__(self) -> str:
        return (
            f"WakeWord(model={self.model!r}, framework={self.framework!r}, "
            f"sensitivity={self.sensitivity:.2f}, available={self._available})"
        )
