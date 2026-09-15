"""Generated interface chimes.

Every sound JARVIS makes for itself is synthesised here with plain numpy: no sample
files, no downloads, nothing to license. The wake chime is the signature sound — two
rising tones (A5 then E6) that overlap enough to read as a single gesture, with a soft
attack, a short decay tail and a gentle second harmonic for warmth. The sleep chime is
its descending mirror.

All generators return ``float32`` mono in ``[-1, 1]``, peak-normalised to about
:data:`PEAK` (0.6) so the caller still has headroom for its own volume scaling, and
every sound starts and ends at exactly zero so nothing clicks.

This module is pure numpy and imports on any platform.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("jarvis.audio.chimes")

# Kokoro speaks at 24 kHz, so the chimes default to the same rate and the player can
# keep a single output stream open for both.
DEFAULT_SAMPLE_RATE = 24000

#: Peak amplitude every chime is normalised to before the caller's volume scaling.
PEAK = 0.6

# Note frequencies (equal temperament, A4 = 440 Hz).
A5 = 880.00
E6 = 1318.51
C6 = 1046.50
E5 = 659.25
BUZZ = 165.00

# Amplitude of the second harmonic mixed into the melodic chimes. Small enough to stay
# a colour rather than a separate note.
_HARMONIC = 0.22

# Global fade applied to a finished chime so the very first and last sample are zero.
_EDGE_FADE_MS = 2.0


def _samples(ms: float, sample_rate: int) -> int:
    """Number of samples in ``ms`` milliseconds, at least one."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    return max(1, int(round(float(ms) * sample_rate / 1000.0)))


def _raised_cosine(n: int, rising: bool) -> np.ndarray:
    """Half a Hann window: a click-free ramp from 0 to 1 (or back down)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.array([0.0 if rising else 1.0], dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float64))
    if not rising:
        ramp = ramp[::-1]
    return ramp.astype(np.float32)


def _edge_fades(audio: np.ndarray, fade_ms: float, sample_rate: int) -> np.ndarray:
    """Apply raised-cosine fades to both ends of ``audio`` (returns a new array)."""
    out = np.array(audio, dtype=np.float32, copy=True)
    n = out.size
    if n == 0:
        return out
    fade = min(_samples(fade_ms, sample_rate), n // 2)
    if fade <= 0:
        return out
    out[:fade] *= _raised_cosine(fade, rising=True)
    out[-fade:] *= _raised_cosine(fade, rising=False)
    return out


def tone(
    freq: float,
    ms: float,
    sample_rate: int,
    *,
    volume: float = 0.3,
    fade_ms: float = 8,
) -> np.ndarray:
    """A single sine tone with click-free fades at both ends.

    Args:
        freq: Frequency in Hz. Values at or above Nyquist are clamped with a warning.
        ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.
        volume: Peak amplitude before fading, clamped to [0, 1].
        fade_ms: Length of the raised-cosine fade at each end.

    Returns:
        float32 mono samples in [-1, 1].
    """
    n = _samples(ms, sample_rate)
    nyquist = sample_rate / 2.0
    f = float(freq)
    if f <= 0.0:
        logger.warning("Tone frequency %.1f Hz is not audible; returning silence.", f)
        return np.zeros(n, dtype=np.float32)
    if f >= nyquist:
        logger.warning(
            "Tone frequency %.1f Hz is above Nyquist for %d Hz; clamping.", f, sample_rate
        )
        f = nyquist * 0.99
    vol = float(np.clip(volume, 0.0, 1.0))

    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    wave = (np.sin(2.0 * np.pi * f * t) * vol).astype(np.float32)
    return _edge_fades(wave, fade_ms, sample_rate)


def _voice(
    freq: float,
    ms: float,
    sample_rate: int,
    *,
    volume: float = 1.0,
    attack_ms: float = 12.0,
    release_ms: float = 8.0,
    harmonic: float = _HARMONIC,
    tail: float = 0.10,
) -> np.ndarray:
    """One chime partial: sine + second harmonic, soft attack, exponential decay tail.

    Args:
        freq: Fundamental frequency in Hz.
        ms: Total duration in milliseconds.
        sample_rate: Sample rate in Hz.
        volume: Peak amplitude of the partial.
        attack_ms: Length of the soft attack ramp.
        release_ms: Final fade so the partial ends at exactly zero.
        harmonic: Amplitude of the second harmonic relative to the fundamental.
        tail: Relative amplitude left at the end of the exponential decay.
    """
    n = _samples(ms, sample_rate)
    base = tone(freq, ms, sample_rate, volume=1.0, fade_ms=0.0)
    if harmonic > 0.0:
        base = base + tone(freq * 2.0, ms, sample_rate, volume=1.0, fade_ms=0.0) * float(harmonic)
        base = base / (1.0 + float(harmonic))

    duration_s = n / float(sample_rate)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    decay_rate = -np.log(max(float(tail), 1e-4)) / max(duration_s, 1e-6)
    env = np.exp(-decay_rate * t).astype(np.float32)

    attack = min(_samples(attack_ms, sample_rate), n)
    if attack > 0:
        env[:attack] *= _raised_cosine(attack, rising=True)
    release = min(_samples(release_ms, sample_rate), max(1, n - attack))
    if release > 0:
        env[-release:] *= _raised_cosine(release, rising=False)

    return (base.astype(np.float32) * env * float(volume)).astype(np.float32)


def _mix(parts: list[tuple[float, np.ndarray]], sample_rate: int) -> np.ndarray:
    """Sum partials placed at millisecond offsets onto one float32 canvas."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    length = 0
    placed: list[tuple[int, np.ndarray]] = []
    for offset_ms, part in parts:
        start = int(round(float(offset_ms) * sample_rate / 1000.0))
        placed.append((start, part))
        length = max(length, start + part.size)
    canvas = np.zeros(length, dtype=np.float32)
    for start, part in placed:
        canvas[start : start + part.size] += part
    return canvas


def _normalize(audio: np.ndarray, peak: float = PEAK) -> np.ndarray:
    """Scale ``audio`` so its loudest sample sits at ``peak``."""
    out = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(out))) if out.size else 0.0
    if current <= 1e-9:
        return np.zeros_like(out, dtype=np.float32)
    return (out * (float(peak) / current)).astype(np.float32)


def _finish(parts: list[tuple[float, np.ndarray]], sample_rate: int) -> np.ndarray:
    """Mix, normalise to :data:`PEAK`, fade the edges and clamp to [-1, 1]."""
    mixed = _mix(parts, sample_rate)
    mixed = _normalize(mixed, PEAK)
    mixed = _edge_fades(mixed, _EDGE_FADE_MS, sample_rate)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def wake_chime(sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """The signature wake sound: ~150 ms, A5 rising into E6.

    The two partials overlap by about a third of their length so the pair reads as one
    upward gesture rather than two separate beeps.
    """
    parts = [
        (0.0, _voice(A5, 95.0, sample_rate, volume=0.90, attack_ms=14.0, tail=0.14)),
        (55.0, _voice(E6, 95.0, sample_rate, volume=1.00, attack_ms=10.0, tail=0.07)),
    ]
    return _finish(parts, sample_rate)


def sleep_chime(sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """The descending mirror of the wake chime: ~200 ms, E6 falling to A5."""
    parts = [
        (0.0, _voice(E6, 120.0, sample_rate, volume=0.95, attack_ms=12.0, tail=0.12)),
        (80.0, _voice(A5, 120.0, sample_rate, volume=0.85, attack_ms=16.0, tail=0.05)),
    ]
    return _finish(parts, sample_rate)


def confirm_chime(sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """A short single blip (~90 ms at C6) acknowledging an action."""
    parts = [
        (0.0, _voice(C6, 90.0, sample_rate, volume=1.0, attack_ms=6.0, harmonic=0.15, tail=0.06)),
    ]
    return _finish(parts, sample_rate)


def error_chime(sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """A low double buzz (~260 ms) for a refusal or a failure."""
    buzz_a = _voice(BUZZ, 110.0, sample_rate, volume=1.0, attack_ms=5.0, harmonic=0.45, tail=0.15)
    buzz_b = _voice(BUZZ, 110.0, sample_rate, volume=0.9, attack_ms=5.0, harmonic=0.45, tail=0.08)
    # A third harmonic gives the buzz its slightly rough, unmistakably negative edge.
    rough_a = _voice(BUZZ * 3.0, 110.0, sample_rate, volume=0.18, attack_ms=5.0, harmonic=0.0)
    rough_b = _voice(BUZZ * 3.0, 110.0, sample_rate, volume=0.15, attack_ms=5.0, harmonic=0.0)
    parts = [(0.0, buzz_a), (0.0, rough_a), (150.0, buzz_b), (150.0, rough_b)]
    return _finish(parts, sample_rate)


def apply_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    """Scale ``audio`` by ``volume`` without ever clipping.

    Args:
        audio: Any numeric array; it is converted to float32 mono samples as-is.
        volume: Linear gain. Negative values are treated as 0 (silence); the result is
            always clamped into [-1, 1], and non-finite samples are zeroed.

    Returns:
        A new float32 array in [-1, 1].
    """
    out = np.asarray(audio, dtype=np.float32)
    gain = float(volume)
    if not np.isfinite(gain) or gain < 0.0:
        logger.warning("Invalid volume %r; treating it as silence.", volume)
        gain = 0.0
    scaled = out * np.float32(gain)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)
