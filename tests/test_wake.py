"""The wake word: does JARVIS look up when he is addressed, and stay quiet otherwise.

``openwakeword`` is not installed here, which is itself one of the cases under test: the
detector must come up disabled rather than raise, so the assistant still boots. Every
other test injects a fake ``openwakeword`` package into ``sys.modules`` and drives it
frame by frame — no microphone, no ONNX runtime, no model download.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from jarvis.wake.detector import INT16_SCALE, WakeWord

FRAME = 1280  # 80 ms at 16 kHz, the block size the audio loop uses


# --- test doubles ---------------------------------------------------------------------
class WakeStub:
    """Controls the fake openWakeWord model and records everything it was given."""

    def __init__(self) -> None:
        self.key = "hey_jarvis"
        self.score = 0.0
        self.scores: list[float] | None = None
        self.predictions: dict | None = None
        self.predict_error: Exception | None = None
        self.constructions: list[dict] = []
        self.chunks: list[np.ndarray] = []
        self.resets = 0
        self.downloads: list[dict] = []

    def next_prediction(self) -> dict:
        if self.predictions is not None:
            return self.predictions
        if self.scores:
            return {self.key: self.scores.pop(0)}
        return {self.key: self.score}

    def model_class(self) -> type:
        stub = self

        class FakeModel:
            def __init__(self, **kwargs):
                stub.constructions.append(kwargs)

            def predict(self, chunk):
                stub.chunks.append(np.asarray(chunk))
                if stub.predict_error is not None:
                    raise stub.predict_error
                return stub.next_prediction()

            def reset(self):
                stub.resets += 1

        return FakeModel


class LogSpy:
    """A logger with its own handler, independent of the root configuration."""

    def __init__(self, name: str) -> None:
        self.records: list[logging.LogRecord] = []
        spy = self

        class Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                spy.records.append(record)

        self.logger = logging.getLogger(name)
        self.handler = Collector()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def close(self) -> None:
        self.logger.removeHandler(self.handler)

    def at_least(self, level: int) -> list[str]:
        return [record.getMessage() for record in self.records if record.levelno >= level]


@pytest.fixture
def log():
    spy = LogSpy("tests.wake.detector")
    try:
        yield spy
    finally:
        spy.close()


def install_openwakeword(monkeypatch, tmp_path, stub: WakeStub, *, models_on_disk: bool = True):
    """Put a fake ``openwakeword`` package on ``sys.modules`` and return the stub."""
    package_dir = tmp_path / "openwakeword"
    models_dir = package_dir / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if models_on_disk:
        for name in ("hey_jarvis_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
            (models_dir / name).write_bytes(b"not a real model")

    package = types.ModuleType("openwakeword")
    package.__file__ = str(package_dir / "__init__.py")

    model_module = types.ModuleType("openwakeword.model")
    model_module.Model = stub.model_class()

    utils_module = types.ModuleType("openwakeword.utils")

    def download_models(**kwargs):
        stub.downloads.append(kwargs)
        for name in ("hey_jarvis_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
            (models_dir / name).write_bytes(b"not a real model")

    utils_module.download_models = download_models

    package.model = model_module
    package.utils = utils_module
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_module)
    return stub


@pytest.fixture
def wake_lib(monkeypatch, tmp_path):
    return install_openwakeword(monkeypatch, tmp_path, WakeStub())


def frame(value: float = 0.1, size: int = FRAME) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


# --- the library is missing -----------------------------------------------------------
def test_a_missing_openwakeword_leaves_the_detector_unavailable(monkeypatch, log):
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)

    detector = WakeWord(logger=log.logger)

    assert detector.available is False
    assert detector.process(frame()) is False
    assert detector.last_score == 0.0


def test_a_missing_openwakeword_is_a_warning_not_a_traceback(monkeypatch, log):
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)

    WakeWord(logger=log.logger)

    assert any("openWakeWord is unavailable" in message for message in log.at_least(logging.WARNING))


def test_an_unavailable_detector_keeps_returning_false_frame_after_frame(monkeypatch, log):
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)
    detector = WakeWord(logger=log.logger)

    assert [detector.process(frame()) for _ in range(20)] == [False] * 20


# --- detection ------------------------------------------------------------------------
def test_a_score_above_the_sensitivity_fires_the_wake_word(wake_lib, log):
    wake_lib.score = 0.82
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert detector.available is True
    assert detector.process(frame()) is True
    assert detector.last_score == pytest.approx(0.82)


def test_a_score_below_the_sensitivity_does_not_fire(wake_lib, log):
    wake_lib.score = 0.31
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is False
    assert detector.last_score == pytest.approx(0.31)


def test_a_score_exactly_at_the_sensitivity_fires(wake_lib, log):
    wake_lib.score = 0.5
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is True


def test_one_spoken_wake_word_fires_exactly_once(wake_lib, log):
    """The score stays high for several frames; the operator said it once."""
    wake_lib.score = 0.9
    detector = WakeWord(sensitivity=0.5, cooldown=2.0, logger=log.logger)

    fired = [detector.process(frame()) for _ in range(6)]

    assert fired == [True, False, False, False, False, False]


def test_the_cooldown_stops_a_second_detection_immediately_after_the_first(wake_lib, log):
    wake_lib.score = 0.9
    detector = WakeWord(sensitivity=0.5, cooldown=2.0, logger=log.logger)

    assert detector.process(frame()) is True
    assert detector.process(frame()) is False


def test_the_wake_word_fires_again_once_the_cooldown_has_passed(wake_lib, log, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("jarvis.wake.detector.time.monotonic", lambda: clock["now"])
    wake_lib.score = 0.9
    detector = WakeWord(sensitivity=0.5, cooldown=2.0, logger=log.logger)

    assert detector.process(frame()) is True
    clock["now"] += 1.9
    assert detector.process(frame()) is False
    clock["now"] += 0.2
    assert detector.process(frame()) is True


def test_reset_clears_the_cooldown(wake_lib, log):
    wake_lib.score = 0.9
    detector = WakeWord(sensitivity=0.5, cooldown=60.0, logger=log.logger)

    assert detector.process(frame()) is True
    detector.reset()

    assert detector.last_score == 0.0
    assert detector.process(frame()) is True


def test_the_model_buffers_are_cleared_after_a_detection(wake_lib, log):
    """Otherwise the tail of one "hey Jarvis" triggers the next one."""
    wake_lib.score = 0.9
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    detector.process(frame())

    assert wake_lib.resets == 1


def test_a_quiet_stream_never_fires_and_never_resets(wake_lib, log):
    wake_lib.score = 0.02
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert not any(detector.process(frame(0.001)) for _ in range(30))
    assert wake_lib.resets == 0


# --- the int16 conversion -------------------------------------------------------------
def test_float32_frames_reach_openwakeword_as_int16(wake_lib, log):
    detector = WakeWord(logger=log.logger)

    detector.process(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))

    chunk = wake_lib.chunks[0]
    assert chunk.dtype == np.int16
    assert chunk.tolist() == [0, int(INT16_SCALE), -int(INT16_SCALE), int(0.5 * INT16_SCALE)]


def test_a_too_loud_frame_is_clipped_rather_than_wrapped_around(wake_lib, log):
    """Without the clip, a shout would arrive as a negative sample and be heard as silence."""
    detector = WakeWord(logger=log.logger)

    detector.process(np.array([2.5, -2.5], dtype=np.float32))

    chunk = wake_lib.chunks[0]
    assert chunk.tolist() == [32767, -32767]
    assert chunk.min() >= -32768 and chunk.max() <= 32767


def test_a_whole_frame_stays_inside_the_int16_range(wake_lib, log):
    rng = np.random.default_rng(7)
    detector = WakeWord(logger=log.logger)

    detector.process(rng.uniform(-1.0, 1.0, FRAME).astype(np.float32))

    chunk = wake_lib.chunks[0]
    assert chunk.dtype == np.int16
    assert chunk.size == FRAME
    assert chunk.min() >= -32768 and chunk.max() <= 32767


def test_a_stereo_frame_is_mixed_down_before_inference(wake_lib, log):
    detector = WakeWord(logger=log.logger)

    detector.process(np.zeros((FRAME, 2), dtype=np.float32))

    assert wake_lib.chunks[0].shape == (FRAME,)


def test_an_empty_frame_is_ignored(wake_lib, log):
    detector = WakeWord(logger=log.logger)

    assert detector.process(np.zeros(0, dtype=np.float32)) is False
    assert wake_lib.chunks == []


# --- prediction keys ------------------------------------------------------------------
def test_a_versioned_prediction_key_still_yields_a_detection(wake_lib, log):
    """openWakeWord reports 'hey_jarvis_v0.1'; the configured name is 'hey_jarvis'."""
    wake_lib.key = "hey_jarvis_v0.1"
    wake_lib.score = 0.88
    detector = WakeWord(model="hey_jarvis", sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is True
    assert detector.model_key == "hey_jarvis_v0.1"


def test_a_full_model_path_as_prediction_key_still_yields_a_detection(wake_lib, log, tmp_path):
    wake_lib.key = str(tmp_path / "models" / "hey_jarvis_v0.1.onnx")
    wake_lib.score = 0.77
    detector = WakeWord(model="hey_jarvis", sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is True
    assert detector.last_score == pytest.approx(0.77)


def test_the_resolved_key_is_reused_for_later_frames(wake_lib, log):
    wake_lib.key = "hey_jarvis_v0.1"
    wake_lib.score = 0.2
    detector = WakeWord(sensitivity=0.9, logger=log.logger)

    for _ in range(5):
        detector.process(frame())

    key_logs = [message for message in log.at_least(logging.INFO) if "prediction key" in message]
    assert len(key_logs) == 1


def test_another_assistants_wake_word_does_not_wake_jarvis(wake_lib, log):
    wake_lib.predictions = {"alexa_v0.1": 0.99, "hey_mycroft_v0.1": 0.97}
    detector = WakeWord(model="hey_jarvis", sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is False
    assert detector.last_score == 0.0


def test_an_empty_prediction_mapping_is_not_a_detection(wake_lib, log):
    wake_lib.predictions = {}
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is False


def test_a_non_numeric_score_is_not_a_detection(wake_lib, log):
    wake_lib.predictions = {"hey_jarvis": "very likely"}
    detector = WakeWord(sensitivity=0.5, logger=log.logger)

    assert detector.process(frame()) is False


# --- failures -------------------------------------------------------------------------
def test_an_inference_failure_is_swallowed(wake_lib, log):
    wake_lib.predict_error = RuntimeError("onnxruntime session is gone")
    detector = WakeWord(logger=log.logger)

    assert detector.process(frame()) is False
    assert detector.available is True
    assert any("Wake-word inference failed" in message for message in log.at_least(logging.ERROR))


def test_the_detector_switches_itself_off_after_repeated_inference_failures(wake_lib, log):
    wake_lib.predict_error = RuntimeError("onnxruntime session is gone")
    detector = WakeWord(logger=log.logger)

    for _ in range(5):
        assert detector.process(frame()) is False

    assert detector.available is False
    calls_before = len(wake_lib.chunks)
    assert detector.process(frame()) is False
    assert len(wake_lib.chunks) == calls_before


def test_a_recovered_failure_does_not_count_towards_the_limit(wake_lib, log):
    wake_lib.predict_error = RuntimeError("transient glitch")
    detector = WakeWord(logger=log.logger)

    for _ in range(4):
        detector.process(frame())
    wake_lib.predict_error = None
    detector.process(frame())
    wake_lib.predict_error = RuntimeError("transient glitch")
    for _ in range(4):
        detector.process(frame())

    assert detector.available is True


def test_a_model_that_cannot_be_built_leaves_the_detector_unavailable(monkeypatch, tmp_path, log):
    stub = WakeStub()
    install_openwakeword(monkeypatch, tmp_path, stub)
    broken = types.ModuleType("openwakeword.model")

    class ExplodingModel:
        def __init__(self, **kwargs):
            raise OSError("hey_jarvis_v0.1.onnx is corrupt")

    broken.Model = ExplodingModel
    monkeypatch.setitem(sys.modules, "openwakeword.model", broken)
    sys.modules["openwakeword"].model = broken

    detector = WakeWord(logger=log.logger)

    assert detector.available is False
    assert detector.process(frame()) is False


# --- model files ----------------------------------------------------------------------
def test_the_models_are_downloaded_once_when_they_are_missing(monkeypatch, tmp_path, log):
    stub = install_openwakeword(monkeypatch, tmp_path, WakeStub(), models_on_disk=False)

    detector = WakeWord(logger=log.logger)
    detector.ensure_models()

    assert len(stub.downloads) == 1
    assert detector.available is True


def test_present_models_are_not_downloaded_again(wake_lib, log):
    detector = WakeWord(logger=log.logger)
    detector.ensure_models()

    assert wake_lib.downloads == []


def test_the_configured_model_name_is_what_openwakeword_is_asked_for(wake_lib, log):
    WakeWord(model="hey_jarvis", framework="onnx", logger=log.logger)

    assert wake_lib.constructions[0]["wakeword_models"] == ["hey_jarvis"]
    assert wake_lib.constructions[0]["inference_framework"] == "onnx"
