"""The overlay's logic, on a machine with no Windows and no display.

Three bugs in a row reached the user's machine because nothing here was covered:
a ctypes argument that overflowed, a renderer attribute that did not exist, and a
latency mark taken outside a turn. The Win32 calls themselves cannot be exercised
off Windows - but everything around them can, and that is where those bugs were.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.core.state import AssistantState, StateBus
from jarvis.ui.hud import HudModel, HudRenderer, ToolEvent
from jarvis.ui.layered import LayeredOverlay
from jarvis.ui.reactor import STATE_STYLES, ReactorRenderer, ReactorTheme, blend_styles

ALL_STATES = list(AssistantState)


@pytest.fixture
def overlay(config):
    """A LayeredOverlay that was never started - construction touches no Win32."""
    return LayeredOverlay(config, StateBus())


# --- the contract between the overlay and its renderer ---------------------------------
def test_the_renderer_has_everything_the_overlay_reads_off_it(overlay):
    """This is the test that would have caught 'HudRenderer has no attribute theme'."""
    for attribute in ("width", "height", "theme", "render", "reactor"):
        assert hasattr(overlay.renderer, attribute), f"renderer is missing {attribute!r}"


def test_hud_renderer_answers_the_reactor_renderer_api_the_overlay_uses():
    """Not identical to ReactorRenderer - but identical where the overlay looks."""
    hud = HudRenderer(180)

    assert isinstance(hud.theme, ReactorTheme)
    assert hud.theme is hud.reactor.theme
    assert callable(hud.render)
    assert hud.width >= hud.ring_size and hud.height >= hud.ring_size


@pytest.mark.parametrize("state", ALL_STATES)
def test_a_style_can_be_resolved_for_every_state(overlay, state):
    overlay._previous = AssistantState.IDLE
    overlay._current = state
    style = overlay._style_now()

    assert style.color.startswith("#")
    assert 0.0 <= style.intensity <= 2.0


@pytest.mark.parametrize("state", ALL_STATES)
def test_a_frame_can_be_painted_for_every_state(overlay, state):
    """Everything the paint path does except handing the bytes to Windows."""
    overlay._current = state
    overlay.hud.state = state
    frame = overlay.renderer.render(overlay.hud, 0.5, amplitude=0.4,
                                    levels=[0.3] * 8, style=overlay._style_now())

    assert frame.shape == (overlay.height, overlay.width, 4)
    assert frame.dtype == np.uint8

    data = ReactorRenderer.to_premultiplied_bgra(frame)
    assert len(data) == overlay.width * overlay.height * 4


def test_premultiplied_bytes_never_exceed_their_alpha():
    """Windows composites with AC_SRC_ALPHA; a colour above its alpha haloes."""
    frame = ReactorRenderer(64).render(AssistantState.SPEAKING, 0.0)
    data = np.frombuffer(ReactorRenderer.to_premultiplied_bgra(frame), dtype=np.uint8)
    pixels = data.reshape(-1, 4)

    assert (pixels[:, 0] <= pixels[:, 3]).all()
    assert (pixels[:, 1] <= pixels[:, 3]).all()
    assert (pixels[:, 2] <= pixels[:, 3]).all()


# --- state transitions -----------------------------------------------------------------
def test_a_state_change_is_recorded_for_the_cross_fade(overlay):
    overlay._on_state(AssistantState.THINKING)

    assert overlay._current is AssistantState.THINKING
    assert overlay._previous is AssistantState.IDLE


def test_the_same_state_twice_does_not_restart_the_cross_fade(overlay):
    overlay._on_state(AssistantState.THINKING)
    first_change = overlay._changed_at
    overlay._on_state(AssistantState.THINKING)

    assert overlay._changed_at == first_change


def test_styles_blend_from_one_state_to_the_next():
    start = STATE_STYLES[AssistantState.IDLE]
    end = STATE_STYLES[AssistantState.THINKING]

    assert blend_styles(start, end, 0.0).color == start.color
    assert blend_styles(start, end, 1.0).color == end.color
    middle = blend_styles(start, end, 0.5)
    assert middle.color not in (start.color, end.color), "the midpoint is a real mix"


# --- the panel -------------------------------------------------------------------------
def test_the_panel_is_invisible_until_there_is_something_to_report():
    renderer = HudRenderer(120)
    assert renderer._panel_alpha(HudModel()) == 0.0


def test_the_panel_appears_once_a_turn_begins():
    renderer = HudRenderer(120)
    model = HudModel(state=AssistantState.LISTENING)
    model.begin_turn("what's the time")

    assert renderer._panel_alpha(model) > 0.0


def test_a_tool_is_shown_running_and_then_finished():
    model = HudModel()
    model.begin_turn("how's the system")
    model.tool_started("system_status")
    assert model.tools[-1].done is False

    model.tool_finished("system_status", ok=True, duration_ms=118)
    assert model.tools[-1].done is True
    assert model.tools[-1].duration_ms == 118


def test_a_refusal_is_marked_as_such_not_as_a_failure():
    model = HudModel()
    model.tool_started("run_powershell")
    model.tool_finished("run_powershell", ok=False, duration_ms=2, refused=True)

    assert model.tools[-1].refused is True


def test_a_long_reply_is_truncated_rather_than_overflowing_the_panel():
    renderer = HudRenderer(180)
    model = HudModel(state=AssistantState.SPEAKING)
    model.begin_turn("tell me everything")
    model.reply = "word " * 400

    frame = renderer.render(model, 0.0)
    assert frame.shape == (renderer.height, renderer.width, 4)


def test_swedish_text_renders_without_blowing_up():
    renderer = HudRenderer(180)
    model = HudModel(state=AssistantState.SPEAKING)
    model.begin_turn("hur mår systemet, är allt väl?")
    model.add_reply("Allt är lugnt, sir. Processorn går på fyra procent.")

    assert renderer.render(model, 0.0).shape[2] == 4


# --- position ---------------------------------------------------------------------------
def test_a_saved_position_is_used_when_it_is_on_screen(config, monkeypatch):
    config.set("ui.position", [100, 200])
    built = LayeredOverlay(config, StateBus())
    monkeypatch.setattr(built, "_u32", _FakeUser32(2560, 1440))

    assert built._initial_position() == (100, 200)


def test_a_position_from_a_monitor_that_is_gone_is_clamped_back(config, monkeypatch):
    config.set("ui.position", [9000, 9000])
    built = LayeredOverlay(config, StateBus())
    monkeypatch.setattr(built, "_u32", _FakeUser32(1920, 1080))

    x, y = built._initial_position()
    assert 0 <= x <= 1920 - built.width
    assert 0 <= y <= 1080 - built.height


class _FakeUser32:
    """Just enough of user32 for the position maths."""

    def __init__(self, width: int, height: int) -> None:
        self._metrics = {0: width, 1: height}

    def GetSystemMetrics(self, index: int) -> int:  # noqa: N802 - Win32 naming
        return self._metrics.get(index, 0)
