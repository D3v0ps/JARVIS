"""The earcon family and the reactor's boot sweep.

``tests/test_chimes.py`` already pins the four original chimes. This file covers what
came later: the three brief-mode earcons (:func:`ack`, :func:`done`, :func:`awaiting`)
and the iris-open animation the reactor plays on startup.

Two claims are worth testing hard. The first is that the earcons are one *family* — they
are built from the wake chime's A5/E6 motif, and a future edit that drops one of them
onto an arbitrary frequency should fail here rather than be discovered by ear. The
second is that the boot sweep ends exactly where the settled ring begins: if the last
boot frame differs from the first live frame by even one pixel, the handover visibly
snaps.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.chimes import (
    A4,
    A5,
    AWAITING_PEAK,
    AWAITING_PERIOD_MS,
    DEFAULT_SAMPLE_RATE,
    E5,
    E6,
    PEAK,
    ack,
    awaiting,
    done,
    error_chime,
    wake_chime,
)
from jarvis.core.state import AssistantState
from jarvis.ui.reactor import (
    BOOT_SECONDS,
    COIL_COUNT,
    COIL_IN,
    COIL_OUT,
    RING_IN,
    ReactorRenderer,
)

# name -> (generator, documented length in ms, documented peak)
EARCONS = {
    "ack": (ack, 70.0, PEAK),
    "done": (done, 220.0, PEAK),
    "awaiting": (awaiting, 400.0, AWAITING_PEAK),
}
ALL_EARCONS = list(EARCONS.items())


def dominant_hz(audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """The loudest frequency in ``audio``, via a zero-padded windowed FFT."""
    windowed = audio.astype(np.float64) * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed, n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def halves(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = audio.size // 2
    return audio[:mid], audio[mid:]


def envelope(audio: np.ndarray, window_ms: float = 5.0,
             sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """A coarse RMS envelope, for counting bursts and finding gaps."""
    window = max(1, int(sample_rate * window_ms / 1000.0))
    return np.array([
        np.sqrt(np.mean(audio[i : i + window] ** 2))
        for i in range(0, max(1, audio.size - window), window)
    ])


# ----------------------------------------------------------------- shape and safety
@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_is_float32_mono(name, entry):
    audio = entry[0]()
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_has_the_documented_length(name, entry):
    generator, expected_ms, _ = entry
    audio = generator(DEFAULT_SAMPLE_RATE)
    assert audio.size / DEFAULT_SAMPLE_RATE * 1000.0 == pytest.approx(expected_ms, abs=5.0)


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_stays_inside_full_scale(name, entry):
    generator, _, expected_peak = entry
    audio = generator()
    peak = float(np.max(np.abs(audio)))
    assert 0.0 < peak <= 1.0
    assert peak == pytest.approx(expected_peak, abs=1e-3)
    assert np.count_nonzero(np.abs(audio) >= 0.999) == 0


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_starts_and_ends_at_silence(name, entry):
    """A non-zero first or last sample is exactly what a click is."""
    audio = entry[0]()
    assert abs(float(audio[0])) < 1e-6
    assert abs(float(audio[-1])) < 1e-6


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_fades_rather_than_jumping(name, entry):
    """The first and last milliseconds must ramp, or the speaker clicks."""
    audio = entry[0]()
    edge = int(DEFAULT_SAMPLE_RATE * 0.001)
    peak = float(np.max(np.abs(audio)))
    assert float(np.max(np.abs(audio[:edge]))) < peak * 0.9
    assert float(np.max(np.abs(audio[-edge:]))) < peak * 0.9


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_no_earcon_carries_a_dc_offset(name, entry):
    audio = entry[0]()
    assert abs(float(np.mean(audio))) < 0.01


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_follows_the_requested_sample_rate(name, entry):
    generator, expected_ms, _ = entry
    audio = generator(16000)
    assert audio.dtype == np.float32
    assert audio.size / 16000 * 1000.0 == pytest.approx(expected_ms, abs=5.0)


@pytest.mark.parametrize("name,entry", ALL_EARCONS)
def test_every_earcon_is_finite(name, entry):
    assert np.all(np.isfinite(entry[0]()))


# ------------------------------------------------------------------------- the motif
def test_ack_is_a_single_short_blip_on_the_motifs_top_note():
    audio = ack()
    assert dominant_hz(audio) == pytest.approx(E6, rel=0.02)
    first, second = halves(audio)
    assert dominant_hz(first) == pytest.approx(dominant_hz(second), rel=0.02)
    # Shorter than anything else in the set: it fires on every accepted command.
    assert audio.size < done().size


def test_done_rises_from_a5_to_e6():
    first, second = halves(done())
    assert dominant_hz(first) == pytest.approx(A5, rel=0.02)
    assert dominant_hz(second) == pytest.approx(E6, rel=0.02)


def test_done_rises_like_the_wake_chime_does():
    """Same gesture, same family — rising is what 'it worked' sounds like here."""
    first, second = halves(done())
    assert dominant_hz(second) > dominant_hz(first) * 1.2


def test_done_separates_its_two_notes_instead_of_overlapping_them():
    """The wake chime slurs its notes together; done articulates them."""
    values = envelope(done())
    quiet = values < values.max() * 0.25
    # A real gap somewhere in the middle third of the sound.
    middle = quiet[len(quiet) // 3 : 2 * len(quiet) // 3]
    assert np.any(middle)


def test_awaiting_is_the_same_fifth_an_octave_down():
    first, second = halves(awaiting())
    assert dominant_hz(first) == pytest.approx(E5, rel=0.02)
    assert dominant_hz(second) == pytest.approx(A4, rel=0.02)
    # Exactly an octave below the notes done() uses, so it is the same interval.
    assert E5 == pytest.approx(E6 / 2.0, rel=0.001)
    assert A4 == pytest.approx(A5 / 2.0, rel=0.001)


def test_awaiting_is_lower_and_slower_than_the_success_earcons():
    """Low and unhurried is what makes it read amber instead of green."""
    assert dominant_hz(awaiting()) < dominant_hz(done())
    assert awaiting().size > done().size > ack().size


def test_awaiting_is_quieter_because_it_repeats():
    """Five repeats at full level over a ten-second window would be unbearable."""
    assert AWAITING_PEAK < PEAK
    assert float(np.max(np.abs(awaiting()))) < float(np.max(np.abs(done())))


def test_awaiting_fits_inside_its_repeat_period_with_silence_to_spare():
    """Repeats must not overlap, or the amber note turns into a drone."""
    length_ms = awaiting().size / DEFAULT_SAMPLE_RATE * 1000.0
    assert length_ms < AWAITING_PERIOD_MS / 2.0
    # Five repeats cover the ten-second confirmation window.
    assert AWAITING_PERIOD_MS * 5 >= 10_000


def test_the_family_shares_the_wake_chimes_notes():
    """Every earcon lands on a note the wake chime already uses, or its octave."""
    allowed = {A4, E5, A5, E6}
    for generator in (ack, done, awaiting):
        audio = generator()
        for part in halves(audio):
            pitch = dominant_hz(part)
            assert any(abs(pitch - note) / note < 0.03 for note in allowed), (
                f"{generator.__name__} strayed to {pitch:.1f} Hz"
            )


def test_the_earcons_are_all_clearly_above_the_refusal_buzz():
    """Binary by design: the three of these mean yes or wait, the buzz means no."""
    buzz = dominant_hz(error_chime())
    for generator in (ack, done, awaiting):
        assert dominant_hz(generator()) > buzz * 2


def test_done_is_not_simply_the_wake_chime_again():
    """They share a motif; they must not be the same recording."""
    a, b = done(), wake_chime()
    assert a.size != b.size


# ------------------------------------------------------------------- the boot sweep
@pytest.fixture
def renderer() -> ReactorRenderer:
    return ReactorRenderer(96)


def alpha_sum(frame: np.ndarray) -> int:
    return int(frame[..., 3].sum())


def test_boot_frames_differ_from_the_settled_frame(renderer):
    settled = renderer.render(AssistantState.IDLE, 0.0)
    for boot_t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        assert not np.array_equal(frame, settled), f"boot_t={boot_t} already settled"


def test_the_boot_ends_exactly_on_the_settled_frame(renderer):
    """One differing pixel at the handover and the ring visibly snaps."""
    settled = renderer.render(AssistantState.IDLE, 0.0)
    for boot_t in (BOOT_SECONDS, BOOT_SECONDS + 0.001, 2.0, 30.0):
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        assert np.array_equal(frame, settled), f"boot_t={boot_t} is not the settled ring"


def test_the_iris_starts_completely_dark(renderer):
    frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=0.0)
    assert alpha_sum(frame) == 0


def test_the_iris_opens_monotonically(renderer):
    """Brightness only ever grows during the boot; a dip would read as a flicker."""
    sums = [
        alpha_sum(renderer.render(AssistantState.IDLE, 0.0, boot_t=step / 12.0 * BOOT_SECONDS))
        for step in range(13)
    ]
    assert sums == sorted(sums)
    assert sums[0] == 0
    assert sums[-1] > 0


def test_the_coil_ring_opens_outward_from_the_centre(renderer):
    """An iris opens; it does not fade a finished ring in at full size.

    Between ``COIL_IN`` and ``COIL_OUT`` there is nothing but the coils — no inner ring,
    no rim, and the halo is clipped to outside ``RIM_IN`` — so that annulus is a clean
    probe. Early in the boot the coils have not reached it yet and it is completely
    black, while the region further in is already lit.
    """
    size = renderer.size
    centre = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    radius = np.hypot(xs - centre, ys - centre) / centre
    annulus = (radius > COIL_IN + 0.02) & (radius < COIL_OUT - 0.02)
    further_in = (radius > 0.10) & (radius < RING_IN - 0.02)

    def alpha(boot_t: float | None, where: np.ndarray) -> float:
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        return float(frame[..., 3][where].max())

    early = 0.3 * BOOT_SECONDS
    assert alpha(early, annulus) == 0.0, "the coils are already at their final radius"
    assert alpha(early, further_in) > 0.0, "nothing is lit at all this early"
    assert alpha(0.5 * BOOT_SECONDS, annulus) == 0.0
    assert alpha(None, annulus) > 0.0, "the settled ring should fill the coil band"


def test_the_lit_radius_only_ever_grows(renderer):
    size = renderer.size
    centre = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    radius = np.hypot(xs - centre, ys - centre) / centre

    def outer_edge(boot_t: float) -> float:
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        lit = frame[..., 3] > 8
        return float(radius[lit].max()) if lit.any() else 0.0

    edges = [outer_edge(step / 10.0 * BOOT_SECONDS) for step in range(11)]
    assert edges == sorted(edges)
    assert edges[0] == 0.0
    assert edges[-1] > edges[1] > 0.0


def test_the_core_brightens_last(renderer):
    """The coils light first; the core arriving early ruins the reveal."""
    size = renderer.size
    centre = size // 2
    window = slice(centre - 2, centre + 2)

    def core_alpha(boot_t: float) -> float:
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        return float(frame[window, window, 3].mean())

    settled = renderer.render(AssistantState.IDLE, 0.0)
    settled_core = float(settled[window, window, 3].mean())

    assert core_alpha(0.5) == 0.0                 # still dark while the coils open
    assert core_alpha(0.9) > core_alpha(0.5)
    assert core_alpha(1.1) < settled_core         # not finished arriving until the end


def test_only_the_first_n_coils_are_lit(renderer):
    """Counted on the coil field itself, sector by sector around the ring."""
    size = renderer.size
    centre = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    angle = (np.arctan2(ys - centre, xs - centre) / (2 * np.pi)) % 1.0
    sector = np.floor(angle * COIL_COUNT).astype(int) % COIL_COUNT
    style = renderer.theme.style(AssistantState.IDLE)

    def lit_sectors(lit_count: int) -> int:
        field = renderer._coils(0.0, style, None, lit_count=lit_count)
        return len({int(s) for s in sector[field > 0.05]})

    for count in range(COIL_COUNT + 1):
        assert lit_sectors(count) == count


def test_the_boot_lights_the_coils_one_at_a_time(renderer):
    """Eight coils lit evenly is symmetric under a half turn; a partial ring is not.

    This is the frame-level version of the claim above: it would fail if the boot faded
    a complete ring in rather than filling it coil by coil.
    """
    settled = renderer.render(AssistantState.IDLE, 0.0)
    assert np.array_equal(settled, np.rot90(settled, 2))

    for boot_t in (0.3, 0.5, 0.7):
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t * BOOT_SECONDS)
        assert not np.array_equal(frame, np.rot90(frame, 2)), (
            f"the ring is already complete at {boot_t:.0%} of the boot"
        )


def test_a_negative_boot_time_is_treated_as_the_very_start(renderer):
    """Clocks are not always monotonic on the first frame after a resume."""
    frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=-5.0)
    assert alpha_sum(frame) == 0


def test_boot_frames_are_still_well_formed_rgba(renderer):
    for boot_t in (0.0, 0.5, 1.0, 1.3):
        frame = renderer.render(AssistantState.IDLE, 0.0, boot_t=boot_t)
        assert frame.shape == (renderer.size, renderer.size, 4)
        assert frame.dtype == np.uint8


def test_the_boot_works_for_every_state(renderer):
    """The overlay may already be listening before the sweep finishes."""
    for state in AssistantState:
        settled = renderer.render(state, 0.4)
        mid = renderer.render(state, 0.4, boot_t=0.5)
        after = renderer.render(state, 0.4, boot_t=BOOT_SECONDS)
        assert not np.array_equal(mid, settled)
        assert np.array_equal(after, settled)


def test_existing_calls_are_untouched_by_the_new_parameter(renderer):
    """``boot_t=None`` must be bit-for-bit what the renderer did before."""
    for state in AssistantState:
        assert np.array_equal(
            renderer.render(state, 0.7, amplitude=0.5, levels=[0.1, 0.9, 0.4]),
            renderer.render(state, 0.7, amplitude=0.5, levels=[0.1, 0.9, 0.4], boot_t=None),
        )
