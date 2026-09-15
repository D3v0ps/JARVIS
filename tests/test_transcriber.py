"""Speech-to-text: model choice, the CUDA-to-CPU fallback and the hallucination filter.

``faster_whisper`` is not installed on the machine that runs these tests, so every test
that needs a backend injects a fake one into ``sys.modules``. Nothing here touches a GPU,
a model file or the network.

The hallucination filter gets the most attention because getting it wrong in either
direction is what a person actually feels: too eager and JARVIS answers ghosts, too shy
and he goes deaf in the middle of a sentence.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import types

import numpy as np
import pytest

from jarvis.stt.transcriber import (
    Transcriber,
    Transcript,
    add_cuda_dll_paths,
    is_hallucination,
    pick_model,
)

SAMPLE_RATE = 16000


# --- test doubles ---------------------------------------------------------------------
class FakeSegment:
    """One whisper segment: text plus the confidence numbers the filter looks at."""

    def __init__(self, text: str, no_speech_prob: float, avg_logprob: float) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class FakeInfo:
    def __init__(self, language: str) -> None:
        self.language = language


class WhisperStub:
    """Controls what the fake ``faster_whisper`` backend does and records how it is used."""

    def __init__(self) -> None:
        self.text = "Open Spotify and turn the volume up."
        self.language = "en"
        self.no_speech_prob = 0.05
        self.avg_logprob = -0.15
        self.cuda_fails = True
        self.transcribe_error: Exception | None = None
        self.constructions: list[dict] = []
        self.calls: list[dict] = []
        self.audio: list[np.ndarray] = []

    def model_class(self) -> type:
        stub = self

        class FakeWhisperModel:
            def __init__(self, size, device="cpu", compute_type="int8", **kwargs):
                stub.constructions.append(
                    {"size": size, "device": device, "compute_type": compute_type}
                )
                if device == "cuda" and stub.cuda_fails:
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                self.device = device

            def transcribe(self, audio, **kwargs):
                stub.calls.append(kwargs)
                stub.audio.append(np.asarray(audio))
                if stub.transcribe_error is not None:
                    raise stub.transcribe_error
                segments = iter(
                    [FakeSegment(stub.text, stub.no_speech_prob, stub.avg_logprob)]
                    if stub.text
                    else []
                )
                return segments, FakeInfo(stub.language)

        return FakeWhisperModel


class LogSpy:
    """A logger with its own handler, so records are captured whatever the root does."""

    def __init__(self, name: str) -> None:
        self.records: list[logging.LogRecord] = []
        self.logger = logging.getLogger(name)
        spy = self

        class Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                spy.records.append(record)

        self.handler = Collector()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def close(self) -> None:
        self.logger.removeHandler(self.handler)

    def messages(self, level: int) -> list[str]:
        return [record.getMessage() for record in self.records if record.levelno == level]

    def at_least(self, level: int) -> list[str]:
        return [record.getMessage() for record in self.records if record.levelno >= level]


@pytest.fixture
def log():
    spy = LogSpy("tests.stt.transcriber")
    try:
        yield spy
    finally:
        spy.close()


@pytest.fixture
def whisper(monkeypatch):
    """Install a fake ``faster_whisper`` module and hand back its controller."""
    stub = WhisperStub()
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = stub.model_class()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return stub


def speech(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A block of plausible float32 microphone audio of the requested length."""
    samples = int(seconds * sample_rate)
    time_axis = np.arange(samples, dtype=np.float32) / float(sample_rate)
    return (0.2 * np.sin(2 * np.pi * 180.0 * time_axis)).astype(np.float32)


# --- pick_model -----------------------------------------------------------------------
@pytest.mark.parametrize("vram_gb", [10, 10.0, 12, 24.0])
def test_pick_model_chooses_medium_from_ten_gigabytes_of_vram(vram_gb):
    assert pick_model(vram_gb) == "medium"


@pytest.mark.parametrize("vram_gb", [6, 6.0, 8, 9.99])
def test_pick_model_chooses_small_between_six_and_ten_gigabytes(vram_gb):
    assert pick_model(vram_gb) == "small"


@pytest.mark.parametrize("vram_gb", [5.99, 4, 2.0, 0])
def test_pick_model_chooses_base_below_six_gigabytes(vram_gb):
    assert pick_model(vram_gb) == "base"


def test_pick_model_chooses_small_when_no_gpu_was_detected():
    """No GPU still means CPU int8 small, not the tiny model."""
    assert pick_model(None) == "small"


@pytest.mark.parametrize("vram_gb", ["lots", "", object()])
def test_pick_model_survives_an_unusable_vram_value(vram_gb):
    assert pick_model(vram_gb) == "small"


def test_auto_model_resolves_through_pick_model(whisper, log):
    stt = Transcriber(model="auto", vram_gb=12.0, device="cpu", logger=log.logger)
    stt.load()

    assert stt.model_in_use == "medium"
    assert whisper.constructions[0]["size"] == "medium"


# --- loading --------------------------------------------------------------------------
def test_device_in_use_is_empty_before_the_model_is_loaded(whisper, log):
    stt = Transcriber(logger=log.logger)

    assert stt.device_in_use == ""
    assert stt.ready is False


def test_a_failed_cuda_attempt_falls_back_to_cpu_int8(whisper, log):
    whisper.cuda_fails = True
    stt = Transcriber(model="small", device="auto", vram_gb=12.0, logger=log.logger)

    stt.load()

    assert stt.ready is True
    assert stt.device_in_use == "cpu"
    assert [item["device"] for item in whisper.constructions] == ["cuda", "cpu"]
    assert whisper.constructions[-1]["compute_type"] == "int8"


def test_the_cuda_failure_is_logged_once_with_its_reason(whisper, log):
    whisper.cuda_fails = True
    stt = Transcriber(model="small", device="cuda", vram_gb=12.0, logger=log.logger)

    stt.load()
    stt.load()

    warnings = log.messages(logging.WARNING)
    assert len(warnings) == 1
    assert "cublas64_12.dll" in warnings[0]
    assert "CPU int8" in warnings[0]


def test_loading_twice_does_not_build_a_second_model(whisper, log):
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    stt.load()
    stt.load()
    stt.load()

    assert len(whisper.constructions) == 1


def test_two_threads_loading_at_once_build_exactly_one_model(whisper, log):
    stt = Transcriber(model="base", device="cpu", logger=log.logger)
    ready = threading.Barrier(2, timeout=2.0)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            ready.wait()
            stt.load()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "load() deadlocked between two threads"

    assert errors == []
    assert len(whisper.constructions) == 1


def test_cuda_is_not_attempted_when_no_gpu_was_detected(whisper, log):
    stt = Transcriber(model="base", device="auto", vram_gb=None, logger=log.logger)

    stt.load()

    assert [item["device"] for item in whisper.constructions] == ["cpu"]
    assert stt.device_in_use == "cpu"


def test_load_raises_when_faster_whisper_is_not_installed(monkeypatch, log):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    with pytest.raises(ImportError):
        stt.load()

    assert stt.ready is False


# --- transcription --------------------------------------------------------------------
def test_transcribe_returns_the_decoded_text(whisper, log):
    whisper.text = "Open the workshop, please."
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(2.0), SAMPLE_RATE)

    assert isinstance(result, Transcript)
    assert result.text == "Open the workshop, please."
    assert result.language == "en"
    assert result.duration_s == pytest.approx(2.0, abs=0.01)
    assert result.latency_ms >= 0.0


def test_transcribe_loads_the_model_on_first_use(whisper, log):
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    stt.transcribe(speech(1.5), SAMPLE_RATE)

    assert stt.ready is True
    assert len(whisper.constructions) == 1


def test_transcribe_returns_an_empty_transcript_when_the_backend_raises(whisper, log):
    whisper.transcribe_error = RuntimeError("CUDA out of memory")
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(2.0), SAMPLE_RATE)

    assert result.text == ""
    assert "CUDA out of memory" in " ".join(log.messages(logging.ERROR))


def test_repeated_decoding_failures_are_only_shouted_about_once(whisper, log):
    whisper.transcribe_error = RuntimeError("model is toast")
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    for _ in range(3):
        assert stt.transcribe(speech(1.0), SAMPLE_RATE).text == ""

    assert len(log.messages(logging.ERROR)) == 1


def test_transcribe_returns_an_empty_transcript_when_faster_whisper_is_missing(monkeypatch, log):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(2.0), SAMPLE_RATE)

    assert result.text == ""
    assert stt.ready is False


def test_empty_audio_is_reported_as_an_empty_transcript_without_loading(whisper, log):
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(np.zeros(0, dtype=np.float32), SAMPLE_RATE)

    assert result.text == ""
    assert result.duration_s == 0.0
    assert whisper.constructions == []


def test_audio_is_resampled_to_sixteen_kilohertz_before_decoding(whisper, log):
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    stt.transcribe(speech(1.0, sample_rate=8000), 8000)

    assert whisper.audio[0].shape == (SAMPLE_RATE,)
    assert whisper.audio[0].dtype == np.float32


def test_a_stereo_frame_is_mixed_down_to_mono(whisper, log):
    stereo = np.stack([speech(1.0), speech(1.0)], axis=1)
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    stt.transcribe(stereo, SAMPLE_RATE)

    assert whisper.audio[0].ndim == 1
    assert whisper.audio[0].size == SAMPLE_RATE


def test_a_forced_language_is_passed_to_whisper(whisper, log):
    stt = Transcriber(model="base", device="cpu", language="sv", logger=log.logger)

    stt.transcribe(speech(1.5), SAMPLE_RATE)

    assert whisper.calls[0]["language"] == "sv"


def test_automatic_language_lets_whisper_detect_it(whisper, log):
    whisper.language = "sv"
    stt = Transcriber(model="base", device="cpu", language="auto", logger=log.logger)

    result = stt.transcribe(speech(1.5), SAMPLE_RATE)

    assert whisper.calls[0]["language"] is None
    assert result.language == "sv"


# --- the hallucination filter ---------------------------------------------------------
HALLUCINATIONS = [
    "Thank you.",
    "Thanks for watching!",
    "Tack för att du tittade!",
    "Undertexter av Amara.org-gemenskapen",
    "[Music]",
    "Textning: BTI Studios",
    "♪♪♪",
    "(upbeat music)",
]


@pytest.mark.parametrize("phrase", HALLUCINATIONS)
def test_whisper_talking_to_itself_over_silence_is_discarded(whisper, log, phrase):
    """A subtitle credit hallucinated over half a second of room tone must not reach the brain."""
    whisper.text = phrase
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(0.5), SAMPLE_RATE)

    assert result.text == ""


def test_a_real_utterance_of_normal_length_is_never_discarded(whisper, log):
    whisper.text = "Jarvis, open Spotify and turn the volume down a little."
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(3.0), SAMPLE_RATE)

    assert result.text == "Jarvis, open Spotify and turn the volume down a little."


def test_a_swedish_utterance_with_accented_letters_is_never_discarded(whisper, log):
    whisper.text = "Öppna spellistan och höj volymen lite grann."
    whisper.language = "sv"
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(3.0), SAMPLE_RATE)

    assert result.text == "Öppna spellistan och höj volymen lite grann."


def test_a_confidently_spoken_thank_you_survives_the_filter(whisper, log):
    """The user really did say it: long enough, and the decoder was sure."""
    whisper.text = "Thank you."
    whisper.no_speech_prob = 0.02
    whisper.avg_logprob = -0.2
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(2.5), SAMPLE_RATE)

    assert result.text == "Thank you."


def test_a_long_utterance_the_decoder_doubted_is_discarded(whisper, log):
    whisper.text = "Thank you."
    whisper.no_speech_prob = 0.9
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(3.0), SAMPLE_RATE)

    assert result.text == ""


def test_a_long_utterance_the_decoder_guessed_at_is_discarded(whisper, log):
    whisper.text = "Tack så mycket"
    whisper.avg_logprob = -1.4
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(3.0), SAMPLE_RATE)

    assert result.text == ""


def test_a_silent_decode_produces_an_empty_transcript(whisper, log):
    whisper.text = ""
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    assert stt.transcribe(speech(1.0), SAMPLE_RATE).text == ""


@pytest.mark.parametrize("phrase", ["Yes.", "Ja."])
def test_a_short_spoken_confirmation_is_not_thrown_away(whisper, log, phrase):
    whisper.text = phrase
    whisper.no_speech_prob = 0.02
    whisper.avg_logprob = -0.2
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    result = stt.transcribe(speech(0.6), SAMPLE_RATE)

    assert result.text == phrase


def test_spaced_music_notes_over_silence_are_discarded(whisper, log):
    whisper.text = "♪ ♪"
    stt = Transcriber(model="base", device="cpu", logger=log.logger)

    assert stt.transcribe(speech(0.5), SAMPLE_RATE).text == ""


def test_is_hallucination_treats_empty_text_as_nothing_heard():
    assert is_hallucination("", duration_s=3.0) is True
    assert is_hallucination("   ", duration_s=3.0) is True


# --- CUDA DLL paths -------------------------------------------------------------------
def test_add_cuda_dll_paths_is_a_silent_no_op_off_windows(monkeypatch, log):
    monkeypatch.setattr(sys, "platform", "linux")
    calls: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", calls.append, raising=False)
    before = os.environ.get("PATH", "")

    add_cuda_dll_paths(log.logger)
    add_cuda_dll_paths(log.logger)

    assert calls == []
    assert os.environ.get("PATH", "") == before
    assert log.at_least(logging.INFO) == []


def test_add_cuda_dll_paths_works_without_a_logger(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert add_cuda_dll_paths() is None
