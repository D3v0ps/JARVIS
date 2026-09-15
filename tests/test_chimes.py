"""The sounds JARVIS makes for himself.

These are the only audio a user hears before a single word is spoken, so the
contract is strict: nothing clicks, nothing clips, nothing hums with DC, and the
wake chime goes *up* while the sleep chime goes *down*. That direction is the
whole emotional point of the pair — up means "I am listening", down means "I have
gone away — and it is worth a test that would actually notice if someone swapped
the two note lists.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.chimes import (
    DEFAULT_SAMPLE_RATE,
    PEAK,
    apply_volume,
    confirm_chime,
    error_chime,
    sleep_chime,
    tone,
    wake_chime,
)

# name -> (generator, documented length in ms)
CHIMES = {
    "wake": (wake_chime, 150.0),
    "sleep": (sleep_chime, 200.0),
    "confirm": (confirm_chime, 90.0),
    "error": (error_chime, 260.0),
}

ALL_CHIMES = list(CHIMES.items())


def dominant_hz(audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """The loudest frequency in ``audio``, via a zero-padded windowed FFT."""
    windowed = audio.astype(np.float64) * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed, n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def zero_crossings(audio: np.ndarray) -> int:
    """How many times ``audio`` crosses zero — a cheap proxy for pitch."""
    return int(np.count_nonzero(np.diff(np.signbit(audio))))


def halves(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = audio.size // 2
    return audio[:mid], audio[mid:]


# --------------------------------------------------------------------- basics


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_is_float32_mono(name, entry):
    audio = entry[0]()
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_has_the_documented_length(name, entry):
    generator, expected_ms = entry
    audio = generator(DEFAULT_SAMPLE_RATE)
    actual_ms = audio.size / DEFAULT_SAMPLE_RATE * 1000.0
    assert actual_ms == pytest.approx(expected_ms, abs=5.0)


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_stays_inside_full_scale_without_clipping(name, entry):
    audio = entry[0]()
    peak = float(np.max(np.abs(audio)))
    assert 0.0 < peak <= 1.0
    # Normalised to PEAK, so there is headroom left for the caller's own volume.
    assert peak == pytest.approx(PEAK, abs=1e-3)
    # A clipped chime would sit flat against the rail for many samples.
    assert np.count_nonzero(np.abs(audio) >= 0.999) == 0


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_no_chime_carries_a_dc_offset(name, entry):
    """A DC offset thumps the speaker cone and eats headroom for nothing."""
    audio = entry[0]()
    assert abs(float(np.mean(audio))) < 0.01


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_starts_and_ends_at_silence(name, entry):
    """Non-zero first/last samples are exactly what a click is."""
    audio = entry[0]()
    assert abs(float(audio[0])) < 1e-6
    assert abs(float(audio[-1])) < 1e-6


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_fades_in_and_out_rather_than_jumping(name, entry):
    """The first and last milliseconds must ramp, not leap to full amplitude."""
    audio = entry[0]()
    edge = int(DEFAULT_SAMPLE_RATE * 0.001)  # 1 ms
    peak = float(np.max(np.abs(audio)))
    assert float(np.max(np.abs(audio[:edge]))) < peak * 0.9
    assert float(np.max(np.abs(audio[-edge:]))) < peak * 0.9


@pytest.mark.parametrize("name,entry", ALL_CHIMES)
def test_every_chime_follows_the_requested_sample_rate(name, entry):
    generator, expected_ms = entry
    audio = generator(16000)
    assert audio.dtype == np.float32
    assert audio.size / 16000 * 1000.0 == pytest.approx(expected_ms, abs=5.0)


# ---------------------------------------------------------------- the gesture


def test_wake_chime_rises_in_pitch():
    """A5 then E6: the second half must be the higher note."""
    first, second = halves(wake_chime())
    assert dominant_hz(second) > dominant_hz(first) * 1.2


def test_sleep_chime_falls_in_pitch():
    """E6 then A5: the mirror of the wake chime, or it means the wrong thing."""
    first, second = halves(sleep_chime())
    assert dominant_hz(second) < dominant_hz(first) / 1.2


def test_wake_chime_rises_by_zero_crossing_count_too():
    """An independent measure of the same claim, in case the FFT is flattering us."""
    first, second = halves(wake_chime())
    assert zero_crossings(second) > zero_crossings(first)


def test_sleep_chime_falls_by_zero_crossing_count_too():
    first, second = halves(sleep_chime())
    assert zero_crossings(second) < zero_crossings(first)


def test_sleep_chime_is_the_mirror_of_the_wake_chime():
    """The wake chime ends where the sleep chime begins, and vice versa."""
    wake_first, wake_second = halves(wake_chime())
    sleep_first, sleep_second = halves(sleep_chime())
    assert dominant_hz(sleep_first) == pytest.approx(dominant_hz(wake_second), rel=0.05)
    assert dominant_hz(sleep_second) == pytest.approx(dominant_hz(wake_first), rel=0.05)


def test_confirm_chime_holds_a_single_pitch():
    first, second = halves(confirm_chime())
    assert dominant_hz(first) == pytest.approx(dominant_hz(second), rel=0.02)


def test_error_chime_is_two_separate_buzzes():
    """A double buzz needs an audible gap, otherwise it is one long groan."""
    audio = error_chime()
    window = int(DEFAULT_SAMPLE_RATE * 0.005)
    envelope = np.array(
        [np.sqrt(np.mean(audio[i : i + window] ** 2)) for i in range(0, audio.size - window, window)]
    )
    loud = envelope > envelope.max() * 0.3
    # Count runs of loud windows: two bursts means the loud mask switches on twice.
    bursts = int(np.count_nonzero(np.diff(loud.astype(np.int8)) == 1)) + int(loud[0])
    assert bursts == 2


def test_error_chime_is_lower_than_the_wake_chime():
    """The buzz has to read as "no", which means well below the cheerful tones."""
    assert dominant_hz(error_chime()) < dominant_hz(wake_chime()) / 2


# ----------------------------------------------------------------------- tone


def test_tone_is_a_sine_at_the_requested_frequency():
    audio = tone(1000.0, 100.0, 24000)
    assert audio.dtype == np.float32
    assert audio.size == 2400
    assert dominant_hz(audio, 24000) == pytest.approx(1000.0, rel=0.02)


def test_tone_above_nyquist_is_clamped_instead_of_aliasing():
    audio = tone(20000.0, 50.0, 24000)
    assert dominant_hz(audio, 24000) < 12000.0
    assert float(np.max(np.abs(audio))) <= 1.0


def test_tone_at_zero_hz_is_silence_not_a_crash():
    audio = tone(0.0, 50.0, 24000)
    assert audio.size == 1200
    assert np.all(audio == 0.0)


def test_tone_of_zero_milliseconds_still_returns_one_sample():
    """Never an empty array: downstream code concatenates these without checking."""
    assert tone(440.0, 0.0, 24000).size == 1


def test_tone_volume_is_clamped_into_range():
    assert float(np.max(np.abs(tone(440.0, 50.0, 24000, volume=5.0, fade_ms=0)))) <= 1.0
    assert np.all(tone(440.0, 50.0, 24000, volume=-2.0) == 0.0)


def test_tone_rejects_a_nonsense_sample_rate():
    with pytest.raises(ValueError):
        tone(440.0, 50.0, 0)


def test_tone_without_fades_starts_immediately():
    """fade_ms=0 is the building block _voice() relies on; it must not fade."""
    audio = tone(1000.0, 50.0, 24000, volume=0.5, fade_ms=0.0)
    assert float(np.max(np.abs(audio[:24]))) > 0.0


# --------------------------------------------------------------- apply_volume


def test_apply_volume_scales_without_changing_dtype():
    scaled = apply_volume(wake_chime(), 0.5)
    assert scaled.dtype == np.float32
    assert float(np.max(np.abs(scaled))) == pytest.approx(PEAK * 0.5, abs=1e-3)


def test_apply_volume_clamps_instead_of_wrapping_on_a_loud_gain():
    scaled = apply_volume(wake_chime(), 100.0)
    assert float(np.max(np.abs(scaled))) <= 1.0
    assert not np.any(np.isnan(scaled))


def test_apply_volume_treats_a_negative_gain_as_silence():
    assert np.all(apply_volume(wake_chime(), -1.0) == 0.0)


def test_apply_volume_treats_a_nan_gain_as_silence():
    assert np.all(apply_volume(wake_chime(), float("nan")) == 0.0)


def test_apply_volume_scrubs_non_finite_samples():
    dirty = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    cleaned = apply_volume(dirty, 1.0)
    assert np.all(np.isfinite(cleaned))
    assert cleaned[0] == 0.0
    assert cleaned[1] == pytest.approx(1.0)
    assert cleaned[2] == pytest.approx(-1.0)


def test_apply_volume_accepts_an_empty_clip():
    out = apply_volume(np.zeros(0, dtype=np.float32), 1.0)
    assert out.size == 0
    assert out.dtype == np.float32
