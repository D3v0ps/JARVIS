"""Utterance segmentation — where a sentence starts and, crucially, where it ends.

Everything here is synthetic audio: a tone is "speech", near-digital silence is
"quiet". That is enough to pin down the behaviour a person actually feels — a
sentence comes back whole, a cough does not become a turn, a monologue gets cut off,
the first syllable is not clipped off the front, and no more silence than necessary
is stapled to the back (Whisper transcribes trailing silence as "Thank you." and
charges latency for the privilege).

Silero needs torch, which is not installed here, so :class:`SpeechSegmenter` is
tested both ways: honestly unavailable, and with a recording stub in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from jarvis.stt.vad import SILERO_CHUNK, EnergySegmenter, SpeechSegmenter

SR = 16000
BLOCK = 1280  # 80 ms, the microphone's frame size
SPEECH_AMPLITUDE = 0.3


# --------------------------------------------------------------- audio fixtures


def silence(ms: float, seed: int = 0) -> np.ndarray:
    """Near-silence: a whisper of dither, so the noise floor is finite and realistic."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * ms / 1000)) * 1e-4).astype(np.float32)


def tone(ms: float, freq: float = 440.0, amplitude: float = SPEECH_AMPLITUDE) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000), dtype=np.float64) / SR
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def feed(segmenter, audio: np.ndarray, block: int = BLOCK) -> list[np.ndarray]:
    """Push ``audio`` through in fixed blocks, collecting every returned utterance."""
    out = []
    for start in range(0, audio.size - block + 1, block):
        result = segmenter.push(audio[start : start + block])
        if result is not None:
            out.append(result)
    return out


def loud_span(audio: np.ndarray, floor: float = SPEECH_AMPLITUDE / 6) -> tuple[int, int]:
    """First and last sample index above ``floor``. Raises if the clip is all quiet."""
    loud = np.flatnonzero(np.abs(audio) > floor)
    assert loud.size, "the utterance contains no audible speech at all"
    return int(loud[0]), int(loud[-1])


def ms_of(samples: int) -> float:
    return samples * 1000.0 / SR


def dominant_hz(audio: np.ndarray) -> float:
    windowed = audio.astype(np.float64) * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed, n=1 << 15))
    return float(np.fft.rfftfreq(1 << 15, 1.0 / SR)[int(np.argmax(spectrum))])


# ============================================================ EnergySegmenter


def test_a_phrase_between_silences_comes_back_as_one_utterance():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate([silence(800), tone(1200), silence(1200)])
    utterances = feed(segmenter, audio)

    assert len(utterances) == 1
    speech = utterances[0]
    assert speech.dtype == np.float32
    assert speech.ndim == 1
    first, last = loud_span(speech)
    assert ms_of(last - first) == pytest.approx(1200, abs=120)


def test_a_hundred_millisecond_blip_is_discarded():
    """A cough, a door, a keyboard — shorter than min_speech_ms is not a turn."""
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate([silence(800), tone(100), silence(1200)])
    assert feed(segmenter, audio) == []
    assert segmenter.speaking is False


def test_a_blip_does_not_poison_the_real_utterance_that_follows():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate(
        [silence(800), tone(100), silence(1000), tone(1000), silence(1000)]
    )
    utterances = feed(segmenter, audio)
    assert len(utterances) == 1
    first, last = loud_span(utterances[0])
    assert ms_of(last - first) == pytest.approx(1000, abs=150)


def test_the_hard_cap_cuts_off_a_monologue():
    """max_utterance_s must fire even though the speaker never pauses."""
    segmenter = EnergySegmenter(
        silence_ms=700, min_speech_ms=250, pre_roll_ms=300, max_utterance_s=1.0
    )
    audio = np.concatenate([silence(600), tone(5000)])
    utterances = feed(segmenter, audio)

    assert len(utterances) >= 1
    capped = utterances[0]
    assert capped.size >= SR * 1.0
    assert capped.size < SR * 1.0 + BLOCK * 2


def test_a_capped_utterance_is_followed_by_more_utterances_not_silence():
    """After the cap the segmenter must keep listening, not wedge."""
    segmenter = EnergySegmenter(
        silence_ms=700, min_speech_ms=250, pre_roll_ms=300, max_utterance_s=1.0
    )
    audio = np.concatenate([silence(600), tone(5000)])
    assert len(feed(segmenter, audio)) >= 3


def test_the_pre_roll_keeps_audio_from_before_speech_was_detected():
    """Without it, the front of "Jarvis, what's..." is gone before detection fires."""
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate([silence(800), tone(1000), silence(1000)])
    utterances = feed(segmenter, audio)

    first, _ = loud_span(utterances[0])
    lead_in_ms = ms_of(first)
    assert 100 <= lead_in_ms <= 320, f"only {lead_in_ms:.0f} ms of pre-roll survived"


def test_without_a_pre_roll_the_utterance_starts_at_the_onset():
    """The contrast test: the lead-in above really is the pre-roll doing its job."""
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=0)
    audio = np.concatenate([silence(800), tone(1000), silence(1000)])
    utterances = feed(segmenter, audio)

    first, _ = loud_span(utterances[0])
    assert ms_of(first) < 80


def test_the_first_hundred_milliseconds_of_speech_survive_intact():
    """The opening syllable is distinctive; if the pre-roll dropped it, it is gone."""
    marker_hz, body_hz = 300.0, 1200.0
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate(
        [silence(830), tone(100, marker_hz), tone(1000, body_hz), silence(1000)]
    )
    utterances = feed(segmenter, audio)
    assert len(utterances) == 1

    speech = utterances[0]
    first, _ = loud_span(speech)
    opening = speech[first : first + int(SR * 0.09)]
    assert dominant_hz(opening) == pytest.approx(marker_hz, rel=0.15)


def test_no_more_than_250ms_of_trailing_silence_is_returned():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate([silence(800), tone(1200), silence(1500)])
    utterances = feed(segmenter, audio)
    assert len(utterances) == 1

    speech = utterances[0]
    _, last = loud_span(speech)
    trailing_ms = ms_of(speech.size - 1 - last)
    assert trailing_ms <= 250, f"{trailing_ms:.0f} ms of silence is sent to Whisper"


def test_speaking_is_true_only_while_an_utterance_is_being_collected():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    feed(segmenter, silence(800))
    assert segmenter.speaking is False

    feed(segmenter, tone(400))
    assert segmenter.speaking is True

    feed(segmenter, silence(1200, seed=1))
    assert segmenter.speaking is False


def test_reset_throws_away_a_half_collected_utterance():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    feed(segmenter, np.concatenate([silence(800), tone(600)]))
    assert segmenter.speaking is True

    segmenter.reset()
    assert segmenter.speaking is False
    assert segmenter.last_prob == 0.0
    # The abandoned audio must not reappear glued to the next utterance.
    utterances = feed(segmenter, np.concatenate([tone(800), silence(1200, seed=2)]))
    assert len(utterances) == 1
    first, last = loud_span(utterances[0])
    assert ms_of(last - first) < 1000


def test_two_utterances_in_a_row_are_both_returned():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    audio = np.concatenate(
        [silence(800), tone(800), silence(1200, seed=1), tone(800), silence(1200, seed=2)]
    )
    assert len(feed(segmenter, audio)) == 2


def test_an_empty_frame_is_ignored():
    segmenter = EnergySegmenter()
    assert segmenter.push(np.zeros(0, dtype=np.float32)) is None
    assert segmenter.speaking is False


def test_a_stereo_frame_is_mixed_down_rather_than_rejected():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    mono = np.concatenate([silence(800), tone(1000), silence(1200)])
    stereo = np.stack([mono, mono], axis=1)

    utterances = []
    for start in range(0, stereo.shape[0] - BLOCK + 1, BLOCK):
        result = segmenter.push(stereo[start : start + BLOCK])
        if result is not None:
            utterances.append(result)
    assert len(utterances) == 1
    assert utterances[0].ndim == 1


def test_the_caller_may_reuse_the_frame_buffer_it_handed_over():
    """The mic loop is free to recycle its array; the utterance must be a copy."""
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    scratch = np.zeros(BLOCK, dtype=np.float32)
    source = np.concatenate([silence(800), tone(1000), silence(1200)])

    utterances = []
    for start in range(0, source.size - BLOCK + 1, BLOCK):
        scratch[:] = source[start : start + BLOCK]
        result = segmenter.push(scratch)
        if result is not None:
            utterances.append(result)
        scratch[:] = 0.0  # the caller recycles its buffer immediately

    assert len(utterances) == 1
    assert float(np.max(np.abs(utterances[0]))) > SPEECH_AMPLITUDE / 2


def test_last_prob_rises_with_the_level_of_the_frame():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    feed(segmenter, silence(800))
    quiet_prob = segmenter.last_prob

    segmenter.push(tone(80))
    assert segmenter.last_prob > quiet_prob
    assert 0.0 <= segmenter.last_prob <= 1.0


def test_the_noise_floor_is_calibrated_from_the_first_half_second():
    """A room with a loud fan must not be heard as one long sentence."""
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    hum = (np.random.default_rng(7).standard_normal(SR) * 0.02).astype(np.float32)
    assert feed(segmenter, hum) == []
    assert segmenter.noise_floor_db > -70.0
    assert segmenter.speaking is False


def test_digital_silence_does_not_produce_a_math_domain_error():
    segmenter = EnergySegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)
    assert feed(segmenter, np.zeros(SR, dtype=np.float32)) == []
    assert np.isfinite(segmenter.noise_floor_db)


def test_the_energy_segmenter_always_reports_itself_available():
    """It is the fallback; if it claimed unavailability there would be nothing left."""
    assert EnergySegmenter().available is True


# ============================================================ SpeechSegmenter


@pytest.fixture
def no_torch(monkeypatch):
    """Make ``import torch`` and ``import silero_vad`` fail, whatever is installed."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "silero_vad", None)


class RecordingModel:
    """A stand-in Silero model that records the exact chunks it was asked to score."""

    def __init__(self, probability=0.0):
        self.chunks: list[np.ndarray] = []
        self.rates: list[int] = []
        self.resets = 0
        self.probability = probability
        self.error: BaseException | None = None

    def __call__(self, tensor, sample_rate):
        if self.error is not None:
            raise self.error
        array = np.asarray(tensor, dtype=np.float32)
        self.chunks.append(array.copy())
        self.rates.append(sample_rate)
        value = (
            self.probability(array) if callable(self.probability) else float(self.probability)
        )
        return types.SimpleNamespace(item=lambda value=value: value)

    def reset_states(self):
        self.resets += 1


def install_fake_torch(monkeypatch, model, *, via_hub=False):
    """Put a minimal torch (and optionally silero_vad) into sys.modules."""

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = _NoGrad
    torch.from_numpy = lambda array: array
    torch.hub = types.SimpleNamespace(
        load=lambda *args, **kwargs: (model, ("utils",)) if via_hub else None
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    if via_hub:
        monkeypatch.setitem(sys.modules, "silero_vad", None)
    else:
        silero = types.ModuleType("silero_vad")
        silero.load_silero_vad = lambda: model
        monkeypatch.setitem(sys.modules, "silero_vad", silero)
    return torch


def rms_probability(array: np.ndarray) -> float:
    return 1.0 if float(np.sqrt(np.mean(array.astype(np.float64) ** 2))) > 0.05 else 0.0


def test_speech_segmenter_reports_unavailable_without_torch(no_torch):
    assert SpeechSegmenter().available is False


def test_an_unavailable_speech_segmenter_returns_none_instead_of_raising(no_torch):
    segmenter = SpeechSegmenter()
    for _ in range(20):
        assert segmenter.push(tone(80)) is None
    assert segmenter.last_prob == 0.0
    assert segmenter.speaking is False


def test_an_unavailable_speech_segmenter_still_accepts_an_empty_frame(no_torch):
    assert SpeechSegmenter().push(np.zeros(0, dtype=np.float32)) is None


def test_speech_segmenter_becomes_available_with_the_silero_package(monkeypatch):
    model = RecordingModel()
    install_fake_torch(monkeypatch, model)
    assert SpeechSegmenter().available is True


def test_speech_segmenter_falls_back_to_torch_hub(monkeypatch):
    """torch.hub returns (model, utils); only the model is kept."""
    model = RecordingModel()
    install_fake_torch(monkeypatch, model, via_hub=True)
    segmenter = SpeechSegmenter()
    assert segmenter.available is True
    segmenter.push(tone(80))
    assert model.chunks, "the hub model was never called"


def test_frames_are_buffered_into_exactly_512_sample_chunks(monkeypatch):
    """The mic delivers 1280 samples; Silero accepts only 512. The gap is ours to fix."""
    model = RecordingModel()
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()

    audio = tone(80 * 5)  # five 1280-sample frames = 6400 samples
    for start in range(0, audio.size, BLOCK):
        segmenter.push(audio[start : start + BLOCK])

    assert model.chunks, "the model was never called"
    assert {chunk.size for chunk in model.chunks} == {SILERO_CHUNK}
    assert len(model.chunks) == audio.size // SILERO_CHUNK
    assert set(model.rates) == {SR}


def test_the_remainder_of_a_frame_is_carried_into_the_next_one(monkeypatch):
    """1280 is not a multiple of 512, so samples must not be dropped at the seam."""
    model = RecordingModel()
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()

    audio = tone(80 * 4)
    for start in range(0, audio.size, BLOCK):
        segmenter.push(audio[start : start + BLOCK])

    stitched = np.concatenate(model.chunks)
    np.testing.assert_allclose(stitched, audio[: stitched.size], atol=1e-6)


def test_a_frame_shorter_than_one_chunk_keeps_the_previous_probability(monkeypatch):
    """Otherwise a short final frame would read as silence and cut a word in half."""
    model = RecordingModel(probability=0.9)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()

    segmenter.push(tone(80))
    assert segmenter.last_prob == pytest.approx(0.9)

    calls_before = len(model.chunks)
    segmenter.push(tone(5))  # 80 samples, far less than one 512-sample chunk
    assert len(model.chunks) == calls_before
    assert segmenter.last_prob == pytest.approx(0.9)


def test_the_probability_of_a_frame_is_the_loudest_of_its_chunks(monkeypatch):
    model = RecordingModel(probability=rms_probability)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter(threshold=0.5)

    mixed = np.concatenate([np.zeros(BLOCK - SILERO_CHUNK, dtype=np.float32), tone(32)])
    segmenter.push(mixed[:BLOCK])
    assert segmenter.last_prob == pytest.approx(1.0)


def test_a_stubbed_silero_returns_a_whole_utterance(monkeypatch):
    model = RecordingModel(probability=rms_probability)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter(silence_ms=700, min_speech_ms=250, pre_roll_ms=300)

    audio = np.concatenate([silence(800), tone(1200), silence(1200)])
    utterances = feed(segmenter, audio)

    assert len(utterances) == 1
    first, last = loud_span(utterances[0])
    assert ms_of(last - first) == pytest.approx(1200, abs=150)


def test_inference_failure_disables_silero_without_raising(monkeypatch):
    """A broken ONNX runtime must degrade to "unavailable", not crash the mic loop."""
    model = RecordingModel(probability=1.0)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()
    assert segmenter.available is True

    model.error = RuntimeError("ONNX runtime exploded")
    assert segmenter.push(tone(80)) is None
    assert segmenter.available is False
    # And it keeps answering politely afterwards.
    assert segmenter.push(tone(80)) is None


def test_reset_clears_the_models_recurrent_state(monkeypatch):
    model = RecordingModel(probability=rms_probability)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()

    segmenter.push(tone(80))
    segmenter.reset()
    assert model.resets >= 1
    assert segmenter.speaking is False
    assert segmenter.last_prob == 0.0


def test_reset_drops_the_partial_chunk_buffer(monkeypatch):
    """Leftover samples from the previous turn must not leak into the next one."""
    model = RecordingModel(probability=0.0)
    install_fake_torch(monkeypatch, model)
    segmenter = SpeechSegmenter()

    segmenter.push(tone(80))  # leaves 256 samples pending
    segmenter.reset()
    model.chunks.clear()

    fresh = tone(32, freq=1000.0)  # exactly 512 samples
    segmenter.push(fresh)
    assert len(model.chunks) == 1
    np.testing.assert_allclose(model.chunks[0], fresh, atol=1e-6)


def test_both_segmenters_expose_the_same_interface(no_torch):
    """The assistant holds one or the other and must never have to ask which."""
    expected = {"push", "reset", "speaking", "available", "last_prob"}
    assert expected <= set(dir(SpeechSegmenter))
    assert expected <= set(dir(EnergySegmenter))
