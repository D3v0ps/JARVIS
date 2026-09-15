"""The arc-reactor overlay: a frameless, always-on-top ring that shows JARVIS' state.

A borderless ~180 px tkinter window in the bottom-right corner, unless the user dragged
it elsewhere (that position is remembered in ``config.yaml``). One ``Canvas`` carries a
soft outer glow, a bright inner ring, a rotating arc segment and a pulsing core, animated
at about twenty frames per second in the ``ui.theme`` colours.

Threading is the trap here: :class:`~jarvis.core.state.StateBus` notifies observers on
whatever thread called ``set()`` — usually the audio thread — while Tk widgets may only be
touched from the thread running the mainloop. So :meth:`Overlay.on_state` only queues the
new state for a ``root.after`` pump, and :meth:`Overlay.stop` sets a flag that pump sees.
``tkinter`` is imported inside :meth:`start`; when it is missing or no display can be
opened, one warning is logged, :meth:`is_alive` stays ``False`` and JARVIS runs on faceless.
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ..core.state import AssistantState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config
    from ..core.state import StateBus

_logger = logging.getLogger("jarvis.ui.overlay")

DEFAULT_RING_SIZE = 180  #: Ring size in pixels when ``ui.ring_size`` is absent.
MARGIN = 24  #: Distance kept from the screen edges when placing the ring.
CAPTION_HEIGHT = 26  #: Pixels reserved below the ring for the caption line.
CAPTION_CHARS = 40  #: Longest caption shown before it is cut short.
FRAME_MS = 50  #: Animation period in milliseconds (~20 fps).
BG = "#010203"  #: Colour key for the transparent background.
_DECAY = 0.90  #: Per-frame decay applied to the fed-in amplitude.

#: Fallback ring colours, mirroring ``ui.theme`` in the shipped ``config.yaml``.
DEFAULT_THEME: Mapping[str, str] = {
    "idle": "#1b3a4b", "listening": "#29b6f6", "thinking": "#ffb300",
    "speaking": "#4dd0e1", "paused": "#616161",
}

#: Per-state animation, which is what makes the states visually distinct:
#: (phase step in radians, arc spin in degrees per frame, brightness base,
#: breathing weight, amplitude weight).
_MOTION: Mapping[AssistantState, tuple[float, float, float, float, float]] = {
    AssistantState.IDLE: (0.06, 0.8, 0.30, 0.16, 0.00),
    AssistantState.LISTENING: (0.22, 2.5, 0.55, 0.20, 0.25),
    AssistantState.THINKING: (0.18, 9.0, 0.65, 0.20, 0.00),
    AssistantState.SPEAKING: (0.38, 3.5, 0.55, 0.45, 0.00),
    AssistantState.PAUSED: (0.00, 0.0, 0.28, 0.00, 0.00),
}

#: Outer glow layers: (radius factor, line width, brightness, stipple).
_GLOW_LAYERS: tuple[tuple[float, int, float, str], ...] = (
    (1.00, 9, 0.30, "gray12"), (0.93, 7, 0.45, "gray25"), (0.86, 5, 0.65, "gray50"),
)


# --- pure helpers (unit-tested directly) ----------------------------------------


def parse_color(color: str) -> tuple[int, int, int]:
    """Parse ``#rgb`` / ``#rrggbb`` into an RGB triple, falling back to white."""
    digits = str(color or "").strip().lstrip("#")
    try:
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        if len(digits) == 6:
            return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    except ValueError:
        pass
    _logger.debug("Unparseable overlay colour %r; using white.", color)
    return (255, 255, 255)


def scale_color(color: str, factor: float) -> str:
    """Return ``color`` scaled towards black (``factor`` below one) or white (above)."""
    weight = max(0.0, float(factor))
    channels = (min(255, max(0, int(round(c * weight)))) for c in parse_color(color))
    return "#{:02x}{:02x}{:02x}".format(*channels)


def state_color(theme: Mapping[str, Any] | None, state: AssistantState) -> str:
    """The configured colour for ``state``, falling back to the shipped theme."""
    key = AssistantState.coerce(state).value
    if isinstance(theme, Mapping):
        value = theme.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_THEME.get(key, DEFAULT_THEME["idle"])


def motion(state: AssistantState) -> tuple[float, float, float, float, float]:
    """The animation parameters for ``state``; unknown states animate like IDLE."""
    return _MOTION.get(AssistantState.coerce(state), _MOTION[AssistantState.IDLE])


def pulse(state: AssistantState, phase: float, amplitude: float = 0.0) -> float:
    """Ring brightness for ``state`` at ``phase``, always within ``0..1``; ``amplitude``
    is a rough loudness in ``0..1`` that lets the listening pulse follow the voice."""
    _, _, base, breath, voice = motion(state)
    wave = (math.sin(float(phase)) + 1.0) / 2.0
    level = min(1.0, max(0.0, float(amplitude)))
    return min(1.0, max(0.0, base + breath * wave + voice * level))


def default_position(
    width: int, height: int, screen_w: int, screen_h: int, margin: int = MARGIN
) -> tuple[int, int]:
    """Bottom-right corner of the primary screen, inset by ``margin``."""
    return (max(0, int(screen_w) - int(width) - margin), max(0, int(screen_h) - int(height) - margin))


def clamp_position(
    position: Any, width: int, height: int, screen_w: int, screen_h: int, margin: int = MARGIN
) -> tuple[int, int]:
    """Turn a saved ``[x, y]`` into a position that is actually on screen; anything
    unusable, or off the current desktop (a monitor unplugged since the position was
    saved), falls back to :func:`default_position`."""
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        return default_position(width, height, screen_w, screen_h, margin)
    try:
        x, y = int(position[0]), int(position[1])
    except (TypeError, ValueError):
        _logger.debug("Ignoring unusable saved overlay position %r.", position)
        return default_position(width, height, screen_w, screen_h, margin)
    return (min(max(0, x), max(0, int(screen_w) - int(width))),
            min(max(0, y), max(0, int(screen_h) - int(height))))


def truncate_caption(text: str, limit: int = CAPTION_CHARS) -> str:
    """One short line for under the ring: whitespace collapsed, cut with an ellipsis."""
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)].rstrip() + "…"


class Overlay:
    """Frameless always-on-top tkinter arc-reactor ring. MUST run on the main thread;
    other threads call :meth:`on_state`, which marshals through a ``root.after`` pump."""

    def __init__(
        self,
        cfg: "Config",
        state: "StateBus",
        *,
        on_quit: Callable[[], None] | None = None,
        on_toggle_pause: Callable[[], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg, self._state = cfg, state
        self._on_quit, self._on_toggle_pause = on_quit, on_toggle_pause
        self._log = logger or _logger
        raw_size = cfg.get("ui.ring_size", DEFAULT_RING_SIZE)
        try:
            self._size = max(80, int(raw_size))
        except (TypeError, ValueError):
            self._log.warning("Invalid ui.ring_size %r; using %d px.", raw_size, DEFAULT_RING_SIZE)
            self._size = DEFAULT_RING_SIZE
        self._height = self._size + CAPTION_HEIGHT
        self._events: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=256)
        self._stop_flag = threading.Event()
        self._unsubscribe: Callable[[], None] | None = None
        self._root: Any = None
        self._canvas: Any = None
        self._menu: Any = None
        self._drag: tuple[int, int] | None = None
        self._alive = False
        self._current: AssistantState = state.state
        self._caption, self._amplitude = "", 0.0
        self._phase, self._spin = 0.0, 0.0

    # --- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        """Build the window and subscribe to the state bus. Never raises."""
        if self._alive:
            return
        try:
            import tkinter as tk  # noqa: PLC0415 - display bound, imported late on purpose
        except Exception as exc:  # ImportError on a slim Python build
            self._log.warning("Overlay disabled: tkinter is unavailable (%s).", exc)
            return
        try:
            root = tk.Tk()
            root.withdraw()
            root.title("JARVIS")
            root.configure(bg=BG)
            root.overrideredirect(True)
            if bool(self._cfg.get("ui.always_on_top", True)):
                root.attributes("-topmost", True)
            self._apply_transparency(root)
            x, y = clamp_position(self._cfg.get("ui.position"), self._size, self._height,
                                  root.winfo_screenwidth(), root.winfo_screenheight())
            root.geometry(f"{self._size}x{self._height}+{x}+{y}")
            canvas = tk.Canvas(root, width=self._size, height=self._height, bg=BG,
                               highlightthickness=0, bd=0, cursor="fleur")
            canvas.pack(fill="both", expand=True)
            canvas.bind("<ButtonPress-1>", self._on_press)
            canvas.bind("<B1-Motion>", self._on_drag)
            canvas.bind("<ButtonRelease-1>", self._on_release)
            canvas.bind("<Button-3>", self._on_menu)
            root.bind("<Button-3>", self._on_menu)
            self._menu = tk.Menu(root, tearoff=0)
            self._menu.add_command(label="Pause", command=self._toggle_pause)
            self._menu.add_command(label="Quit", command=self._quit)
            root.deiconify()
        except Exception as exc:  # TclError on a headless box or a locked session
            self._log.warning("Overlay disabled: no display available (%s).", exc)
            self._destroy()
            return
        self._root, self._canvas = root, canvas
        self._current = self._state.state
        self._caption = truncate_caption(self._state.text)
        self._alive = True
        self._unsubscribe = self._state.subscribe(self.on_state)
        root.after(FRAME_MS, self._tick)
        self._log.info("Overlay online at %d, %d (%d px).", x, y, self._size)

    def run(self) -> None:
        """Enter the Tk mainloop. Blocks; main thread only."""
        if not self._alive or self._root is None:
            self._log.debug("Overlay.run() has no window; returning immediately.")
            return
        try:
            self._root.mainloop()
        except Exception:
            self._log.exception("Overlay mainloop failed.")
        finally:
            self._destroy()

    def stop(self) -> None:
        """Ask the overlay to close. Safe from any thread, and safe to call twice."""
        self._stop_flag.set()
        if self._root is None:
            self._alive = False

    def is_alive(self) -> bool:
        """True while a window exists and has not been asked to close."""
        return bool(self._alive and self._root is not None and not self._stop_flag.is_set())

    # --- thread-safe inputs -----------------------------------------------------
    def on_state(self, state: AssistantState) -> None:
        """StateBus observer. Runs on the audio thread: queue only, never a widget."""
        self._push("state", AssistantState.coerce(state))

    def set_amplitude(self, level: float) -> None:
        """Feed a rough input loudness (``0..1``) so the listening pulse follows the voice."""
        try:
            self._push("amplitude", min(1.0, max(0.0, float(level))))
        except (TypeError, ValueError):
            self._log.debug("Ignoring non-numeric overlay amplitude %r.", level)

    def _push(self, kind: str, payload: Any) -> None:
        """Hand an event to the Tk pump, dropped rather than ever blocking a caller."""
        try:
            self._events.put_nowait((kind, payload))
        except queue.Full:  # pragma: no cover - only under a stalled mainloop
            self._log.debug("Overlay queue full; dropped %s event.", kind)

    # --- Tk thread --------------------------------------------------------------
    def _tick(self) -> None:
        """Drain the queue, advance the animation, repaint, reschedule."""
        if self._root is None:
            return
        if self._stop_flag.is_set():
            self._shutdown()
            return
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "state":
                self._current = payload
            elif kind == "amplitude":
                self._amplitude = payload
        try:
            self._caption = truncate_caption(self._state.text)
            phase_step, spin_step = motion(self._current)[:2]
            self._phase = (self._phase + phase_step) % (2.0 * math.pi)
            self._spin = (self._spin + spin_step) % 360.0
            self._amplitude *= _DECAY
            self._draw()
            self._root.after(FRAME_MS, self._tick)
        except Exception:
            self._log.exception("Overlay animation failed; closing the ring.")
            self._shutdown()

    def _draw(self) -> None:
        """Repaint the ring for the current state."""
        canvas = self._canvas
        if canvas is None:
            return
        canvas.delete("all")
        colour = state_color(self._cfg.get("ui.theme"), self._current)
        level = pulse(self._current, self._phase, self._amplitude)
        centre = self._size / 2.0
        radius = centre - 12.0
        for factor, width, brightness, stipple in _GLOW_LAYERS:
            glow = scale_color(colour, brightness * level)
            self._ring(canvas, centre, radius * factor, glow, width, stipple)
        inner = radius * 0.72
        self._ring(canvas, centre, inner, scale_color(colour, 0.55 + 0.75 * level), 3, None)
        if self._current is not AssistantState.PAUSED:
            arc = radius * 0.86
            canvas.create_arc(centre - arc, centre - arc, centre + arc, centre + arc,
                              start=self._spin, extent=78, style="arc", width=4,
                              outline=scale_color(colour, 0.9 + 0.8 * level))
        core = max(3.0, inner * (0.30 + 0.22 * level))
        canvas.create_oval(centre - core, centre - core, centre + core, centre + core,
                           fill=scale_color(colour, 0.7 + 0.9 * level), outline="")
        if self._caption:
            canvas.create_text(centre, self._size + CAPTION_HEIGHT / 2.0, text=self._caption,
                               fill=scale_color(colour, 1.1), font=("Segoe UI", 8), width=self._size - 6)

    @staticmethod
    def _ring(
        canvas: Any, centre: float, radius: float, colour: str, width: int, stipple: str | None
    ) -> None:
        """One circle of the glow; falls back to a solid outline without stipple support."""
        box = (centre - radius, centre - radius, centre + radius, centre + radius)
        if stipple:
            try:
                canvas.create_oval(*box, outline=colour, width=width, outlinestipple=stipple)
                return
            except Exception:
                pass
        canvas.create_oval(*box, outline=colour, width=width)

    # --- window plumbing --------------------------------------------------------
    def _apply_transparency(self, root: Any) -> None:
        """Colour-key the background on Windows, then apply the configured opacity."""
        if sys.platform == "win32":
            try:
                root.attributes("-transparentcolor", BG)
            except Exception as exc:
                self._log.debug("Transparent colour key unsupported here (%s).", exc)
        try:
            opacity = min(1.0, max(0.15, float(self._cfg.get("ui.opacity", 0.92))))
        except (TypeError, ValueError):
            opacity = 0.92
        try:
            root.attributes("-alpha", opacity)
        except Exception as exc:
            self._log.debug("Window opacity unsupported here (%s).", exc)

    def _on_press(self, event: Any) -> None:
        """Remember the grab offset so the drag does not jump."""
        if self._root is not None:
            self._drag = (event.x_root - self._root.winfo_x(), event.y_root - self._root.winfo_y())

    def _on_drag(self, event: Any) -> None:
        if self._drag is not None and self._root is not None:
            self._root.geometry(f"+{event.x_root - self._drag[0]}+{event.y_root - self._drag[1]}")

    def _on_release(self, _event: Any) -> None:
        """Remember where the user parked the ring."""
        if self._drag is None or self._root is None:
            return
        self._drag = None
        try:
            x, y = clamp_position([self._root.winfo_x(), self._root.winfo_y()], self._size, self._height,
                                  self._root.winfo_screenwidth(), self._root.winfo_screenheight())
            self._root.geometry(f"+{x}+{y}")
            self._cfg.set("ui.position", [x, y])
            self._cfg.save()
            self._log.debug("Overlay position saved: %d, %d.", x, y)
        except Exception as exc:
            self._log.warning("Could not save the overlay position: %s", exc)

    def _on_menu(self, event: Any) -> None:
        """Pop up the two-item context menu under the cursor."""
        if self._menu is None:
            return
        try:
            paused = self._current is AssistantState.PAUSED
            self._menu.entryconfigure(0, label="Resume" if paused else "Pause")
            self._menu.tk_popup(event.x_root, event.y_root)
        except Exception as exc:
            self._log.debug("Context menu failed to open (%s).", exc)
        finally:
            try:
                self._menu.grab_release()
            except Exception:
                pass

    # --- callbacks and teardown -------------------------------------------------
    def _toggle_pause(self) -> None:
        """Menu item: hand the pause/resume decision to the assistant."""
        self._invoke(self._on_toggle_pause, "pause toggle")

    def _quit(self) -> None:
        """Menu item: shut the assistant down, or just the ring when nothing is wired."""
        if self._on_quit is None:
            self.stop()
            return
        self._invoke(self._on_quit, "quit")

    def _invoke(self, callback: Callable[[], None] | None, what: str) -> None:
        """Run an injected callback, never letting the UI die with it."""
        if callback is None:
            self._log.debug("Overlay %s requested but no callback is wired.", what)
            return
        try:
            callback()
        except Exception:
            self._log.exception("Overlay %s callback failed.", what)

    def _shutdown(self) -> None:
        """Leave the mainloop from the Tk thread, then tear the window down."""
        self._alive = False
        if self._root is None:
            return
        try:
            self._root.quit()
        except Exception:
            self._log.debug("Overlay mainloop had already stopped.")
            self._destroy()

    def _destroy(self) -> None:
        """Unsubscribe and destroy the window. Idempotent."""
        self._alive = False
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                self._log.debug("Overlay was already unsubscribed from the state bus.")
            self._unsubscribe = None
        root, self._root, self._canvas, self._menu = self._root, None, None, None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                self._log.debug("Overlay window was already destroyed.")
