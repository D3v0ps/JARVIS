"""The readout beside the ring: what JARVIS heard, what he is doing, what he said.

A ring alone is a status light. What makes the films' JARVIS feel like a presence
is that you can watch him work - the words arriving, the tool firing, the answer
coming back, and how long all of it took. That is what this panel shows.

It is drawn with Pillow into the same RGBA bitmap as the reactor, so the whole
overlay is one per-pixel-alpha surface with no window chrome anywhere. When there
is nothing to report the panel fades out completely and only the ring remains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from jarvis.core.state import AssistantState
from jarvis.ui.reactor import ReactorRenderer, ReactorTheme, StateStyle, _hex_to_rgb

__all__ = ["HudModel", "HudRenderer", "ToolEvent"]

PANEL_GAP = 26          # between the ring and the readout
PANEL_WIDTH = 400
MARGIN = 16
FADE_S = 0.35           # how long text takes to appear or leave
HOLD_S = 9.0            # how long the last exchange stays before fading

#: Preferred faces, best first. Segoe UI Variable ships with Windows 11.
FONT_CANDIDATES = (
    "SegoeUIVariableText-Regular.ttf", "segoeuivar.ttf", "segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
FONT_BOLD_CANDIDATES = (
    "SegoeUIVariableText-Semibold.ttf", "segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

STATE_LABELS = {
    AssistantState.IDLE: "STANDING BY",
    AssistantState.LISTENING: "LISTENING",
    AssistantState.THINKING: "WORKING",
    AssistantState.SPEAKING: "SPEAKING",
    AssistantState.PAUSED: "PAUSED",
}


@dataclass
class ToolEvent:
    """One tool call, as the panel shows it."""

    name: str
    done: bool = False
    ok: bool = True
    duration_ms: float = 0.0
    refused: bool = False


@dataclass
class HudModel:
    """Everything the panel knows. The assistant updates it; the renderer reads it."""

    state: AssistantState = AssistantState.IDLE
    heard: str = ""                      # what the user said
    reply: str = ""                      # what JARVIS is saying, as it streams
    tools: list[ToolEvent] = field(default_factory=list)
    latency_ms: float | None = None
    note: str = ""                       # "Say confirm", "The microphone is gone", ...
    updated_at: float = field(default_factory=time.perf_counter)

    def touch(self) -> None:
        self.updated_at = time.perf_counter()

    def begin_turn(self, heard: str) -> None:
        self.heard = heard
        self.reply = ""
        self.tools = []
        self.latency_ms = None
        self.note = ""
        self.touch()

    def add_reply(self, sentence: str) -> None:
        self.reply = (self.reply + " " + sentence).strip()
        self.touch()

    def tool_started(self, name: str) -> ToolEvent:
        event = ToolEvent(name=name)
        self.tools.append(event)
        self.touch()
        return event

    def tool_finished(self, name: str, *, ok: bool, duration_ms: float, refused: bool = False) -> None:
        for event in reversed(self.tools):
            if event.name == name and not event.done:
                event.done, event.ok, event.duration_ms, event.refused = True, ok, duration_ms, refused
                break
        self.touch()

    @property
    def age(self) -> float:
        return time.perf_counter() - self.updated_at

    def is_empty(self) -> bool:
        return not (self.heard or self.reply or self.tools or self.note)


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class HudRenderer:
    """Ring plus readout, composited into one RGBA frame."""

    def __init__(self, ring_size: int = 180, theme: ReactorTheme | None = None) -> None:
        self.ring_size = int(ring_size)
        self.reactor = ReactorRenderer(self.ring_size, theme)
        self.width = MARGIN * 2 + self.ring_size + PANEL_GAP + PANEL_WIDTH
        self.height = MARGIN * 2 + self.ring_size
        self._fonts: dict[tuple[int, bool], object] = {}
        self._scrim_base: np.ndarray | None = None

    def _scrim(self, alpha: float):
        """The panel's backing gradient, built once and reused every frame."""
        from PIL import Image

        if self._scrim_base is None:
            w, h = PANEL_WIDTH, self.ring_size
            ramp = (1.0 - np.arange(w, dtype=np.float32) / w) ** 1.5
            self._scrim_base = np.tile(ramp, (h, 1))
        scaled = np.clip(self._scrim_base * 150.0 * alpha, 0, 255).astype(np.uint8)
        rgba = np.zeros((*scaled.shape, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 6, 10, 14
        rgba[..., 3] = scaled
        return Image.fromarray(rgba, "RGBA")

    @property
    def theme(self) -> ReactorTheme:
        """The reactor's theme, so a HudRenderer is a drop-in for a ReactorRenderer."""
        return self.reactor.theme

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = _load_font(size, bold)
        return self._fonts[key]

    # --- panel visibility -------------------------------------------------------------
    @staticmethod
    def _panel_alpha(model: HudModel) -> float:
        """Fade in on arrival, hold, then fade away so idle is just the ring."""
        if model.is_empty():
            return 0.0
        age = model.age
        if age < FADE_S:
            return age / FADE_S
        if model.state in (AssistantState.LISTENING, AssistantState.THINKING, AssistantState.SPEAKING):
            return 1.0
        if age > HOLD_S:
            return max(0.0, 1.0 - (age - HOLD_S) / 1.2)
        return 1.0

    # --- drawing ----------------------------------------------------------------------
    def render(
        self,
        model: HudModel,
        t: float,
        *,
        amplitude: float = 0.0,
        levels: Sequence[float] | None = None,
        style: StateStyle | None = None,
        boot_t: float | None = None,
    ) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        ring = Image.fromarray(
            self.reactor.render(model.state, t, amplitude=amplitude, levels=levels,
                                style=style, boot_t=boot_t),
            "RGBA",
        )
        canvas.alpha_composite(ring, (MARGIN, MARGIN))

        alpha = self._panel_alpha(model)
        if alpha > 0.01:
            panel = self._draw_panel(model, alpha, style)
            canvas.alpha_composite(panel, (MARGIN + self.ring_size + PANEL_GAP, MARGIN))

        return np.asarray(canvas, dtype=np.uint8)

    def _draw_panel(self, model: HudModel, alpha: float, style: StateStyle | None):
        from PIL import Image, ImageDraw

        w, h = PANEL_WIDTH, self.ring_size
        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        accent = _hex_to_rgb(style.color if style else "#29b6f6")
        accent_rgb = tuple(int(c * 255) for c in accent)

        def fade(colour: tuple[int, int, int], strength: float) -> tuple[int, int, int, int]:
            return (*colour, int(max(0, min(255, 255 * strength * alpha))))

        # A scrim, so white text stays readable over a pale wallpaper without ever
        # looking like a window. Darkest at the hairline, gone by the right edge.
        # The gradient never changes shape, so it is built once and only scaled.
        panel.alpha_composite(self._scrim(alpha))

        # The hairline that ties the panel to the ring.
        draw.line([(0, 6), (0, h - 6)], fill=fade(accent_rgb, 0.55), width=1)

        y = 6
        # --- status line ---------------------------------------------------------------
        label = STATE_LABELS.get(model.state, model.state.value.upper())
        label_font = self.font(12, bold=True)
        draw.text((14, y), " ".join(label), font=label_font, fill=fade(accent_rgb, 0.95))

        if model.latency_ms is not None:
            timing = f"{model.latency_ms / 1000:.2f} s"
            tw = draw.textlength(timing, font=label_font)
            draw.text((w - 14 - tw, y), timing, font=label_font, fill=fade((128, 148, 165), 0.9))
        y += 26

        # --- what he heard --------------------------------------------------------------
        if model.heard:
            y = self._wrapped(draw, f"›  {model.heard}", 14, y, w - 28,
                              self.font(15), fade((150, 170, 188), 1.0), max_lines=2)
            y += 8

        # --- what he is doing ------------------------------------------------------------
        for event in model.tools[-3:]:
            mark, colour, strength = "▸", accent_rgb, 0.9
            suffix = ""
            if event.refused:
                mark, colour, suffix = "⊘", (244, 106, 106), "  refused"
            elif event.done and not event.ok:
                mark, colour, suffix = "×", (244, 106, 106), "  failed"
            elif event.done:
                mark, colour = "✓", (94, 214, 154)
                suffix = f"  {event.duration_ms:.0f} ms" if event.duration_ms else ""
            draw.text((14, y), f"{mark}  {event.name}{suffix}",
                      font=self.font(13), fill=fade(colour, strength))
            y += 20
        if model.tools:
            y += 4

        # --- what he said ----------------------------------------------------------------
        if model.reply:
            y = self._wrapped(draw, model.reply, 14, y, w - 28,
                              self.font(16), fade((228, 240, 250), 1.0), max_lines=3)

        # --- anything that needs saying ---------------------------------------------------
        if model.note:
            note_font = self.font(13, bold=True)
            draw.text((14, h - 24), model.note, font=note_font, fill=fade((255, 196, 86), 1.0))

        return panel

    def _wrapped(self, draw, text: str, x: int, y: int, width: int, font,
                 fill, max_lines: int = 3) -> int:
        """Word-wrap, truncating with an ellipsis rather than overflowing the panel."""
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
                if len(lines) == max_lines:
                    break
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) == max_lines and (current not in lines or len(words) > sum(len(l.split()) for l in lines)):
            last = lines[-1]
            while last and draw.textlength(last + "…", font=font) > width:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"

        line_height = int(font.size * 1.35) if hasattr(font, "size") else 20
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        return y
