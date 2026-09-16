"""The system tray icon: a small glowing ring next to the clock.

``pystray`` owns a message loop of its own, so the icon runs in its own daemon thread
and the whole module stays optional. The image is drawn with Pillow at four times the
final size and downsampled, which is the cheapest way to get a smooth ring, and it is
redrawn in the state's colour every time :class:`~jarvis.core.state.StateBus` changes.

Neither ``pystray`` nor ``PIL`` is imported at module import time. When either is
missing, or the desktop has no tray to dock into, one warning is logged and JARVIS
simply runs without an icon — this is decoration, never a dependency.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ..core.state import AssistantState
from .overlay import DEFAULT_THEME, scale_color

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.state import StateBus

_logger = logging.getLogger("jarvis.ui.tray")

ICON_SIZE = 64  #: Final icon size in pixels; Windows scales it down as needed.
_SUPERSAMPLE = 4  #: Draw this many times larger, then downsample for smooth edges.
TOOLTIP = "JARVIS"  #: Hover text next to the clock.


def state_color(state: AssistantState, theme: Mapping[str, Any] | None = None) -> str:
    """The icon colour for ``state``, from ``theme`` when given, else the shipped theme."""
    key = AssistantState.coerce(state).value
    if isinstance(theme, Mapping):
        value = theme.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_THEME.get(key, DEFAULT_THEME["idle"])


def make_icon_image(color: str, size: int = ICON_SIZE) -> Any:
    """Draw a glowing ring in ``color`` and return a Pillow image, or ``None``.

    Pillow is imported here rather than at module scope; a missing or broken Pillow
    returns ``None`` so the caller can quietly skip the tray.
    """
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415 - optional, imported late
    except Exception as exc:  # ImportError, or a broken Pillow build
        _logger.debug("Tray icon cannot be drawn: Pillow is unavailable (%s).", exc)
        return None
    try:
        edge = max(16, int(size)) * _SUPERSAMPLE
        image = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        halo = scale_color(color, 0.55)
        margin = edge * 0.10
        draw.ellipse([margin, margin, edge - margin, edge - margin], outline=halo, width=int(edge * 0.14))
        ring = edge * 0.22
        draw.ellipse([ring, ring, edge - ring, edge - ring], outline=color, width=int(edge * 0.10))
        core = edge * 0.38
        draw.ellipse([core, core, edge - core, edge - core], fill=scale_color(color, 1.25))
        return image.resize((max(16, int(size)), max(16, int(size))), Image.LANCZOS)
    except Exception:
        _logger.exception("Tray icon could not be drawn.")
        return None


class Tray:
    """pystray icon with Open, Pause/Resume and Quit. Runs in its own thread."""

    def __init__(
        self,
        state: "StateBus",
        *,
        on_quit: Callable[[], None] | None = None,
        on_toggle_pause: Callable[[], None] | None = None,
        on_show_window: Callable[[], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._state = state
        self._on_quit = on_quit
        self._on_toggle_pause = on_toggle_pause
        #: Optional: without a desk window there is nothing to open, and the menu
        #: should not offer it.
        self._on_show_window = on_show_window
        self._log = logger or _logger
        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._running = False

    # --- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        """Create the icon and run it on a daemon thread. Never raises."""
        with self._lock:
            if self._running:
                return
            try:
                import pystray  # noqa: PLC0415 - desktop bound, imported late on purpose
            except Exception as exc:  # ImportError on a server or a slim install
                self._log.warning("Tray disabled: pystray is unavailable (%s).", exc)
                return
            image = make_icon_image(state_color(self._state.state))
            if image is None:
                self._log.warning("Tray disabled: Pillow is unavailable, so no icon can be drawn.")
                return
            try:
                self._icon = pystray.Icon("jarvis", image, TOOLTIP, menu=self._build_menu(pystray))
                self._thread = threading.Thread(target=self._run, name="jarvis-tray", daemon=True)
                self._running = True
                self._thread.start()
            except Exception as exc:
                self._log.warning("Tray disabled: the icon could not be created (%s).", exc)
                self._icon, self._thread, self._running = None, None, False
                return
        self._unsubscribe = self._state.subscribe(self.on_state)
        self._log.info("Tray icon online.")

    def stop(self) -> None:
        """Remove the icon and stop its thread. Safe from any thread, and twice."""
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                self._log.debug("Tray was already unsubscribed from the state bus.")
            self._unsubscribe = None
        with self._lock:
            icon, thread = self._icon, self._thread
            self._icon, self._thread, self._running = None, None, False
        if icon is not None:
            try:
                icon.stop()
            except Exception as exc:
                self._log.debug("Tray icon had already stopped (%s).", exc)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - pystray refusing to let go
                self._log.debug("Tray thread did not stop within two seconds; it is a daemon.")

    def is_alive(self) -> bool:
        """True while the icon thread is running."""
        with self._lock:
            return bool(self._running and self._thread is not None and self._thread.is_alive())

    # --- state ------------------------------------------------------------------
    def on_state(self, state: AssistantState) -> None:
        """StateBus observer: recolour the icon and refresh the menu's checkmark."""
        icon = self._icon
        if icon is None:
            return
        image = make_icon_image(state_color(state))
        try:
            if image is not None:
                icon.icon = image
            icon.update_menu()
        except Exception as exc:
            self._log.debug("Tray icon could not be updated (%s).", exc)

    # --- menu -------------------------------------------------------------------
    def _build_menu(self, pystray: Any) -> Any:
        """Open JARVIS (when there is a window), Pause/Resume, and Quit.

        The word is deliberate: what this opens is the desk application, not a console.
        The window is the default item as well as the first, so a double-click on the
        icon does the thing anyone who has just minimised JARVIS wants.
        """
        items = []
        if self._on_show_window is not None:
            items.append(
                pystray.MenuItem("Open JARVIS", self._show_window, default=True)
            )
        items.append(
            pystray.MenuItem(
                lambda _item: "Resume" if self._paused() else "Pause",
                self._toggle_pause,
                checked=lambda _item: self._paused(),
            )
        )
        items.append(pystray.MenuItem("Quit", self._quit))
        return pystray.Menu(*items)

    def _paused(self) -> bool:
        """Whether the assistant is currently paused, for the menu's label and checkmark."""
        try:
            return self._state.state is AssistantState.PAUSED
        except Exception:  # pragma: no cover - the bus is in-process and cannot fail
            return False

    def _show_window(self, _icon: Any = None, _item: Any = None) -> None:
        """Menu item: bring the desk window back, or open it for the first time."""
        self._invoke(self._on_show_window, "show window")

    def _toggle_pause(self, _icon: Any = None, _item: Any = None) -> None:
        """Menu item: hand the pause/resume decision to the assistant."""
        self._invoke(self._on_toggle_pause, "pause toggle")

    def _quit(self, _icon: Any = None, _item: Any = None) -> None:
        """Menu item: shut the assistant down, or just the icon when nothing is wired."""
        if self._on_quit is None:
            self.stop()
            return
        self._invoke(self._on_quit, "quit")

    def _invoke(self, callback: Callable[[], None] | None, what: str) -> None:
        """Run an injected callback, never letting the tray thread die with it."""
        if callback is None:
            self._log.debug("Tray %s requested but no callback is wired.", what)
            return
        try:
            callback()
        except Exception:
            self._log.exception("Tray %s callback failed.", what)

    # --- thread -----------------------------------------------------------------
    def _run(self) -> None:
        """The pystray message loop; ends when :meth:`stop` calls ``icon.stop()``."""
        icon = self._icon
        if icon is None:
            return
        try:
            icon.run()
        except Exception:
            self._log.exception("Tray icon stopped unexpectedly; JARVIS continues without it.")
        finally:
            with self._lock:
                self._running = False
