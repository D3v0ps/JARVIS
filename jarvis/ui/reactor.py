"""The arc reactor, drawn properly.

tkinter cannot do this. Its canvas has no anti-aliasing, and Windows'
``-transparentcolor`` is a colour key rather than real transparency, so a soft
glow drawn that way ends up with a visible fringe. The result looks like a
widget, not like a presence in the room.

So the ring is rendered here instead: every layer is an analytic function of
radius and angle evaluated over a numpy grid, with ``smoothstep`` edges exactly
one pixel wide. That gives perfect anti-aliasing and a genuine per-pixel alpha
channel, which :mod:`jarvis.ui.layered` hands straight to Windows'
``UpdateLayeredWindow``. A 256 px frame costs about two milliseconds, so sixty
frames a second uses a few percent of one core.

Nothing here touches Windows or any GUI toolkit, which is what makes it
testable - and previewable - on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from jarvis.core.state import AssistantState

__all__ = ["ReactorTheme", "ReactorRenderer", "STATE_STYLES", "StateStyle"]

# --- geometry, as fractions of the half-width ----------------------------------------
CORE_R = 0.150          # the white-hot centre
CORE_BLOOM = 0.285      # how far the core's light spills
WELL_IN = 0.300         # dark gap
RING_IN, RING_OUT = 0.352, 0.412        # thin inner ring
COIL_IN, COIL_OUT = 0.462, 0.700        # the segmented coil ring
RIM_IN, RIM_OUT = 0.716, 0.772          # outer rim
HALO_OUT = 1.000        # soft falloff to nothing

COIL_COUNT = 8
COIL_GAP = 0.20         # fraction of each segment that is dark


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    text = str(value).lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (0.16, 0.71, 0.96)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite interpolation between two edges - this is where the AA comes from."""
    if edge1 == edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class StateStyle:
    """How one assistant state looks and moves."""

    color: str
    intensity: float            # overall brightness, 0..1
    breathe_period: float       # seconds for one breath; 0 disables
    breathe_depth: float        # how much the breath changes intensity
    spin: float                 # coil rotations per second
    sweep: float                # 0 = all coils lit, 1 = a single comet sweeping round
    reactive: float             # how much live audio amplitude modulates the ring
    pulse: float = 0.0          # rings travelling outward from the core, as when speaking


#: Motion is what separates "alive" from "a picture of a ring".
STATE_STYLES: dict[AssistantState, StateStyle] = {
    AssistantState.IDLE: StateStyle(
        color="#1b6fa8", intensity=0.42, breathe_period=4.5, breathe_depth=0.16,
        spin=0.02, sweep=0.0, reactive=0.0,
    ),
    AssistantState.LISTENING: StateStyle(
        color="#29b6f6", intensity=0.95, breathe_period=0.0, breathe_depth=0.0,
        spin=0.06, sweep=0.0, reactive=1.0,
    ),
    AssistantState.THINKING: StateStyle(
        color="#ffb300", intensity=0.82, breathe_period=0.0, breathe_depth=0.0,
        spin=0.85, sweep=1.0, reactive=0.0,
    ),
    AssistantState.SPEAKING: StateStyle(
        color="#3ee0c8", intensity=0.90, breathe_period=0.0, breathe_depth=0.0,
        spin=-0.14, sweep=0.0, reactive=0.85, pulse=1.0,
    ),
    AssistantState.PAUSED: StateStyle(
        color="#5c6670", intensity=0.30, breathe_period=0.0, breathe_depth=0.0,
        spin=0.0, sweep=0.0, reactive=0.0,
    ),
}


@dataclass
class ReactorTheme:
    """Colours pulled from ``config.yaml``, with the film's palette as the default."""

    colors: dict[str, str] = field(default_factory=dict)

    def style(self, state: AssistantState) -> StateStyle:
        base = STATE_STYLES.get(state, STATE_STYLES[AssistantState.IDLE])
        override = self.colors.get(state.value)
        if not override:
            return base
        return StateStyle(
            color=str(override), intensity=base.intensity,
            breathe_period=base.breathe_period, breathe_depth=base.breathe_depth,
            spin=base.spin, sweep=base.sweep, reactive=base.reactive, pulse=base.pulse,
        )


class ReactorRenderer:
    """Renders one RGBA frame of the arc reactor.

    The radius and angle grids are computed once; each frame is then a handful of
    vectorised expressions over them.
    """

    def __init__(self, size: int = 200, theme: ReactorTheme | None = None) -> None:
        self.size = max(48, int(size))
        self.theme = theme or ReactorTheme()
        self._build_grids()

    # --- setup -----------------------------------------------------------------------
    def _build_grids(self) -> None:
        n = self.size
        centre = (n - 1) / 2.0
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
        dx = (xs - centre) / centre
        dy = (ys - centre) / centre
        self._r = np.hypot(dx, dy).astype(np.float32)
        self._theta = np.arctan2(dy, dx).astype(np.float32)
        #: One pixel, in the same units as ``_r`` - the width every soft edge uses.
        self._px = float(1.0 / centre)
        # Static layers that never change shape, only brightness.
        self._core = np.exp(-((self._r / CORE_R) ** 2) * 1.6).astype(np.float32)
        self._core_bloom = np.exp(-((self._r / CORE_BLOOM) ** 2) * 2.2).astype(np.float32)
        self._inner_ring = self._band(RING_IN, RING_OUT)
        self._rim = self._band(RIM_IN, RIM_OUT)
        self._coil_mask = self._band(COIL_IN, COIL_OUT)
        halo = np.clip((HALO_OUT - self._r) / (HALO_OUT - RIM_OUT), 0.0, 1.0)
        self._halo = (halo ** 2.4).astype(np.float32) * (self._r > RIM_IN)
        self._disc = (1.0 - _smoothstep(1.0 - 2 * self._px, 1.0, self._r)).astype(np.float32)

    def _band(self, inner: float, outer: float) -> np.ndarray:
        """A ring between two radii, with one-pixel soft edges."""
        soft = self._px
        rise = _smoothstep(inner - soft, inner + soft, self._r)
        fall = 1.0 - _smoothstep(outer - soft, outer + soft, self._r)
        return (rise * fall).astype(np.float32)

    # --- per-frame -------------------------------------------------------------------
    def _coils(self, t: float, style: StateStyle, levels: Sequence[float] | None) -> np.ndarray:
        """The eight coils: lit evenly, sweeping like a comet, or riding the audio."""
        angle = self._theta + math.tau * (style.spin * t)
        position = (angle / math.tau) % 1.0                  # 0..1 around the ring
        segment = position * COIL_COUNT
        index = np.floor(segment).astype(np.int32) % COIL_COUNT
        within = segment - np.floor(segment)

        # Soft-edged gap between neighbouring coils.
        half_gap = COIL_GAP / 2.0
        edge = 0.035
        lit = _smoothstep(half_gap - edge, half_gap + edge, within) * (
            1.0 - _smoothstep(1.0 - half_gap - edge, 1.0 - half_gap + edge, within)
        )

        brightness = np.ones_like(lit, dtype=np.float32)

        if style.sweep > 0.0:
            # A comet sweeping round the ring. The floor matters as much as the head:
            # drop it too low and the ring stops reading as a ring and starts reading
            # as something broken.
            head = (style.spin * t) % 1.0
            distance = (position - head) % 1.0
            tail = np.clip(1.0 - distance / 0.55, 0.0, 1.0) ** 1.6
            glow = np.exp(-((distance / 0.055) ** 2)) + np.exp(-(((1.0 - distance) / 0.045) ** 2))
            brightness = (0.34 + 0.85 * tail + 0.75 * glow).astype(np.float32)
            brightness = ((1.0 - style.sweep) + style.sweep * brightness).astype(np.float32)

        if style.reactive > 0.0 and levels is not None and len(levels) > 0:
            # Live audio mapped around the ring, so the coils actually answer the room.
            values = np.asarray(list(levels), dtype=np.float32)
            per_coil = np.interp(
                np.arange(COIL_COUNT, dtype=np.float32),
                np.linspace(0, COIL_COUNT - 1, num=values.size, dtype=np.float32),
                values,
            )
            per_coil = np.clip(per_coil, 0.0, 1.0)
            modulation = 0.45 + 0.55 * per_coil[index]
            brightness = brightness * (
                (1.0 - style.reactive) + style.reactive * modulation
            ).astype(np.float32)

        return (self._coil_mask * lit * brightness).astype(np.float32)

    def render(
        self,
        state: AssistantState,
        t: float,
        *,
        amplitude: float = 0.0,
        levels: Sequence[float] | None = None,
        style: StateStyle | None = None,
    ) -> np.ndarray:
        """One frame as ``(size, size, 4)`` uint8 RGBA, straight alpha.

        ``t`` is seconds since the overlay started; ``amplitude`` is the current
        audio level (0..1) and ``levels`` an optional short history mapped around
        the ring.
        """
        style = style or self.theme.style(state)
        rgb = np.array(_hex_to_rgb(style.color), dtype=np.float32)

        intensity = float(style.intensity)
        if style.breathe_period > 0:
            phase = math.sin(math.tau * t / style.breathe_period)
            intensity += style.breathe_depth * phase
        intensity += 0.22 * float(np.clip(amplitude, 0.0, 1.0)) * style.reactive
        intensity = float(np.clip(intensity, 0.0, 1.35))

        coils = self._coils(t, style, levels)

        # Brightness of each layer, in "light units" that are summed then tone-mapped.
        light = (
            self._core * 1.55
            + self._core_bloom * 0.42
            + self._inner_ring * 1.05
            + coils * 1.20
            + self._rim * 0.70
            + self._halo * 0.38
        ) * intensity

        if style.pulse > 0.0:
            # A wave leaving the core and fading as it reaches the rim: a voice,
            # drawn. Two overlapping waves keep it continuous rather than strobing.
            speed, width = 0.62, 0.085
            amount = 0.30 + 0.55 * float(np.clip(amplitude, 0.0, 1.0))
            wave = np.zeros_like(self._r)
            for offset in (0.0, 0.5):
                front = ((t * speed + offset) % 1.0) * (RIM_OUT - CORE_R) + CORE_R
                ring = np.exp(-(((self._r - front) / width) ** 2))
                fade = np.clip(1.0 - (front - CORE_R) / (RIM_OUT - CORE_R), 0.0, 1.0)
                wave += ring * fade
            light = light + wave * amount * style.pulse * self._disc

        # The core reads white-hot while the rest keeps the state's colour.
        whiteness = np.clip(self._core * 1.25 + self._core_bloom * 0.18, 0.0, 1.0)[..., None]
        colour = rgb[None, None, :] * (1.0 - whiteness) + np.float32(1.0) * whiteness

        # Filmic-ish tone map: bright areas bloom toward white instead of clipping.
        exposure = np.clip(light, 0.0, None)[..., None]
        rgb_out = 1.0 - np.exp(-exposure * colour * 1.9)

        alpha = np.clip(light * 0.95, 0.0, 1.0) * self._disc

        frame = np.empty((self.size, self.size, 4), dtype=np.uint8)
        frame[..., :3] = np.clip(rgb_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        frame[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return frame

    # --- interop ---------------------------------------------------------------------
    @staticmethod
    def to_premultiplied_bgra(frame: np.ndarray) -> bytes:
        """The byte layout ``UpdateLayeredWindow`` wants: BGRA, alpha premultiplied.

        Windows composites layered windows with ``AC_SRC_ALPHA``, which assumes the
        colour channels have already been multiplied by alpha. Skip this and every
        glow gets a bright halo.
        """
        alpha = frame[..., 3:4].astype(np.uint16)
        rgb = (frame[..., :3].astype(np.uint16) * alpha // 255).astype(np.uint8)
        bgra = np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], frame[..., 3]])
        return np.ascontiguousarray(bgra).tobytes()


def blend_styles(a: StateStyle, b: StateStyle, t: float) -> StateStyle:
    """Ease between two states so the ring never snaps from one look to another."""
    t = float(np.clip(t, 0.0, 1.0))
    eased = t * t * (3.0 - 2.0 * t)
    ca, cb = np.array(_hex_to_rgb(a.color)), np.array(_hex_to_rgb(b.color))
    mixed = ca + (cb - ca) * eased
    color = "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in mixed))

    def lerp(x: float, y: float) -> float:
        return float(x + (y - x) * eased)

    return StateStyle(
        color=color,
        intensity=lerp(a.intensity, b.intensity),
        breathe_period=b.breathe_period,
        breathe_depth=lerp(a.breathe_depth, b.breathe_depth),
        spin=lerp(a.spin, b.spin),
        sweep=lerp(a.sweep, b.sweep),
        reactive=lerp(a.reactive, b.reactive),
        pulse=lerp(a.pulse, b.pulse),
    )
