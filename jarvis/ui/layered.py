"""The Windows host for the arc reactor: a real per-pixel-alpha overlay.

``UpdateLayeredWindow`` composites a 32-bit bitmap with a genuine alpha channel
onto the desktop, which is the only way to get a soft glow with no fringe. It is
also, conveniently, the only Windows overlay mechanism that does not need to own
the main thread - so this runs in its own thread with its own message pump, and
the assistant loop keeps the main thread.

Everything Windows-specific is imported inside the methods that need it, so this
module still imports on Linux, where :class:`LayeredOverlay` simply reports that
it is unavailable and the caller falls back to the tkinter overlay.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from typing import Callable, Sequence

import numpy as np

from jarvis.core.logging import get_logger
from jarvis.core.state import AssistantState, StateBus
from jarvis.ui.hud import HudModel, HudRenderer
from jarvis.ui.reactor import STATE_STYLES, ReactorRenderer, ReactorTheme, blend_styles

__all__ = ["LayeredOverlay", "is_supported"]

# --- Win32 constants -----------------------------------------------------------------
WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

SW_SHOW = 5
SW_HIDE = 0
HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010

WM_DESTROY, WM_CLOSE, WM_TIMER = 0x0002, 0x0010, 0x0113
WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE = 0x0201, 0x0202, 0x0200
WM_RBUTTONUP, WM_APP = 0x0205, 0x8000
WM_QUIT_OVERLAY = WM_APP + 1

TPM_RIGHTBUTTON, TPM_RETURNCMD = 0x0002, 0x0100
MF_STRING, MF_SEPARATOR = 0x0000, 0x0800

IDM_PAUSE, IDM_QUIT, IDM_CENTRE, IDM_CLICKTHROUGH = 1001, 1002, 1003, 1004

DIB_RGB_COLORS = 0
FPS = 30.0
#: How long the iris-open takes when the overlay first appears.
BOOT_SECONDS = 1.2
TRANSITION_S = 0.28


def is_supported() -> bool:
    """True only on Windows, where layered windows exist."""
    return sys.platform == "win32"


class LayeredOverlay:
    """The arc reactor as a frameless, always-on-top, per-pixel-alpha window."""

    def __init__(
        self,
        cfg,
        state: StateBus,
        *,
        on_quit: Callable[[], None] | None = None,
        on_toggle_pause: Callable[[], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.on_quit = on_quit
        self.on_toggle_pause = on_toggle_pause
        self.log = logger or get_logger("ui.layered")

        self.ring_size = int(cfg.get("ui.ring_size", 180))
        self.opacity = float(cfg.get("ui.opacity", 0.92))
        theme = ReactorTheme(dict(cfg.get("ui.theme", {}) or {}))
        self.renderer = HudRenderer(self.ring_size, theme)
        #: What the panel shows. The assistant writes to this as a turn unfolds.
        self.hud = HudModel()
        self.width = self.renderer.width
        self.height = self.renderer.height
        self.click_through = bool(cfg.get("ui.click_through", False))

        self._u32 = self._g32 = self._k32 = None
        self._hwnd = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._failed = False

        self._amplitude = 0.0
        self._levels: list[float] = [0.0] * 8
        self._started_at = time.perf_counter()

        self._current = state.state
        self._previous = state.state
        self._changed_at = self._started_at
        self._unsubscribe: Callable[[], None] | None = None

        self._drag_origin: tuple[int, int] | None = None
        self._position: list[int] | None = None
        self._wndproc_ref = None  # keep the ctypes callback alive

    # --- public API ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return is_supported() and not self._failed

    def is_alive(self) -> bool:
        return bool(self._running.is_set() and self._thread and self._thread.is_alive())

    def set_amplitude(self, amplitude: float, levels: Sequence[float] | None = None) -> None:
        """Feed the live audio level in, so the ring answers the room."""
        self._amplitude = float(max(0.0, min(1.0, amplitude)))
        if levels:
            self._levels = [float(max(0.0, min(1.0, v))) for v in levels][:8]

    def start(self) -> bool:
        """Create the window on its own thread. Returns False if it could not."""
        if not is_supported():
            self.log.info("Layered overlay needs Windows; falling back.")
            return False
        if self.is_alive():
            return True
        self._unsubscribe = self.state.subscribe(self._on_state)
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="jarvis-overlay", daemon=True
        )
        self._thread.start()
        ready.wait(timeout=5.0)
        if self._failed:
            self.stop()
            return False
        return self.is_alive()

    def run(self) -> None:
        """Nothing to do: this backend owns its own thread. Kept so the layered and
        tkinter overlays present the same interface to the assistant."""
        return None

    #: The tkinter overlay must own the main thread; this one must not.
    needs_main_thread = False

    def stop(self) -> None:
        self._running.clear()
        if self._unsubscribe:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            self._unsubscribe = None
        hwnd = self._hwnd
        if hwnd and self._u32 is not None:
            try:
                self._u32.PostMessageW(hwnd, WM_QUIT_OVERLAY, 0, 0)
            except Exception:  # noqa: BLE001
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # --- state ----------------------------------------------------------------------
    def _on_state(self, new_state: AssistantState) -> None:
        """Called from the audio thread; only touches plain attributes."""
        if new_state is self._current:
            return
        self._previous = self._current
        self._current = new_state
        self._changed_at = time.perf_counter()

    def _style_now(self):
        elapsed = time.perf_counter() - self._changed_at
        blend = min(1.0, elapsed / TRANSITION_S) if TRANSITION_S > 0 else 1.0
        a = STATE_STYLES.get(self._previous, STATE_STYLES[AssistantState.IDLE])
        b = STATE_STYLES.get(self._current, STATE_STYLES[AssistantState.IDLE])
        theme_b = self.renderer.theme.style(self._current)
        b = theme_b if blend >= 1.0 else b
        return blend_styles(a, b, blend)

    # --- the window ------------------------------------------------------------------
    def _run(self, ready: threading.Event) -> None:
        try:
            self._create_window()
            self._running.set()
        except Exception as exc:  # noqa: BLE001 - any Win32 failure means fall back
            self._failed = True
            self.log.warning("Could not create the layered overlay (%s); using the fallback.", exc)
            ready.set()
            return
        ready.set()
        try:
            self._pump()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("The overlay stopped: %s", exc)
        finally:
            self._running.clear()
            self._destroy_window()

    def _create_window(self) -> None:
        from jarvis.ui import _win32 as w  # structures and prototypes live next door

        self._w = w
        # Declaring every signature is not optional. Without argtypes, ctypes marshals
        # each argument as a C int, and the first 64-bit handle - CreateWindowExW's
        # hInstance - raises "int too long to convert".
        self._u32, self._g32, self._k32 = w.bind()
        user32 = self._u32

        # Per-monitor DPI awareness, so the ring is crisp on a scaled display.
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            try:
                user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            except Exception:  # noqa: BLE001 - older Windows, harmless
                pass

        self._wndproc_ref = w.WNDPROC(self._wndproc)
        class_name = "JarvisReactorOverlay"

        wndclass = w.WNDCLASSEX()
        wndclass.cbSize = ctypes.sizeof(w.WNDCLASSEX)
        wndclass.lpfnWndProc = self._wndproc_ref
        wndclass.hInstance = self._k32.GetModuleHandleW(None)
        wndclass.lpszClassName = class_name
        wndclass.hCursor = user32.LoadCursorW(None, w.cursor_resource(32512))  # IDC_ARROW
        user32.RegisterClassExW(ctypes.byref(wndclass))  # a duplicate class is fine

        x, y = self._initial_position()
        self._hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            class_name, "JARVIS", WS_POPUP,
            x, y, self.width, self.height,
            None, None, wndclass.hInstance, None,
        )
        if not self._hwnd:
            raise OSError(f"CreateWindowExW failed: {ctypes.get_last_error()}")
        self._position = [x, y]
        if self.click_through:
            self._set_click_through(True)
        user32.ShowWindow(self._hwnd, SW_SHOW)
        user32.SetWindowPos(self._hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        self._paint()

    def _initial_position(self) -> tuple[int, int]:
        user32 = self._u32
        saved = self.cfg.get("ui.position")
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        if isinstance(saved, (list, tuple)) and len(saved) == 2:
            try:
                x, y = int(saved[0]), int(saved[1])
                # Clamp back on-screen: a monitor may have been unplugged since.
                x = max(0, min(x, screen_w - self.width))
                y = max(0, min(y, screen_h - self.height))
                return x, y
            except (TypeError, ValueError):
                pass
        margin = 28
        return screen_w - self.width - margin, screen_h - self.height - margin - 48

    def _destroy_window(self) -> None:
        if self._hwnd and self._u32 is not None:
            try:
                self._u32.DestroyWindow(self._hwnd)
            except Exception:  # noqa: BLE001
                pass
            self._hwnd = None

    # --- painting --------------------------------------------------------------------
    def _paint(self) -> None:
        """Render one frame and hand it to the compositor."""
        gdi32, user32 = self._g32, self._u32
        w = self._w

        t = time.perf_counter() - self._started_at
        self.hud.state = self._current
        # The ring irises open when it first appears. Hard-capped here rather than in
        # the renderer, so a slow first frame cannot leave it half-built.
        boot_t = t if t < BOOT_SECONDS else None
        frame = self.renderer.render(
            self.hud, t, boot_t=boot_t,
            amplitude=self._amplitude, levels=self._levels, style=self._style_now(),
        )
        bits = ReactorRenderer.to_premultiplied_bgra(frame)

        screen_dc = user32.GetDC(None)
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        header = w.BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(w.BITMAPINFOHEADER)
        header.biWidth = self.width
        header.biHeight = -self.height        # negative: top-down, matching numpy
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0              # BI_RGB
        info = w.BITMAPINFO()
        info.bmiHeader = header

        pixel_ptr = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(
            mem_dc, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(pixel_ptr), None, 0
        )
        old = gdi32.SelectObject(mem_dc, bitmap)
        ctypes.memmove(pixel_ptr, bits, len(bits))

        size = w.SIZE(self.width, self.height)
        src = w.POINT(0, 0)
        dst = w.POINT(*(self._position or (0, 0)))
        blend = w.BLENDFUNCTION(AC_SRC_OVER, 0, int(255 * self.opacity), AC_SRC_ALPHA)

        user32.UpdateLayeredWindow(
            self._hwnd, screen_dc, ctypes.byref(dst), ctypes.byref(size),
            mem_dc, ctypes.byref(src), 0, ctypes.byref(blend), ULW_ALPHA,
        )

        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(None, screen_dc)

    def _pump(self) -> None:
        """Message loop and frame clock in one, so nothing needs a Win32 timer."""
        user32 = self._u32
        w = self._w
        msg = w.MSG()
        frame_time = 1.0 / FPS
        next_frame = time.perf_counter()

        while self._running.is_set():
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
                if msg.message in (WM_QUIT_OVERLAY, 0x0012):  # WM_QUIT
                    self._running.clear()
                    return
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

            now = time.perf_counter()
            if now >= next_frame:
                self._paint()
                next_frame = max(now, next_frame + frame_time)
            time.sleep(0.004)

    # --- input -----------------------------------------------------------------------
    def _wndproc(self, hwnd, message, wparam, lparam):
        user32 = self._u32
        try:
            if message == WM_LBUTTONDOWN:
                user32.SetCapture(hwnd)
                self._drag_origin = self._cursor_pos()
                return 0
            if message == WM_MOUSEMOVE and self._drag_origin:
                cx, cy = self._cursor_pos()
                ox, oy = self._drag_origin
                if self._position:
                    self._position[0] += cx - ox
                    self._position[1] += cy - oy
                self._drag_origin = (cx, cy)
                return 0
            if message == WM_LBUTTONUP and self._drag_origin:
                user32.ReleaseCapture()
                self._drag_origin = None
                self._remember_position()
                return 0
            if message == WM_RBUTTONUP:
                self._context_menu(hwnd)
                return 0
            if message in (WM_CLOSE, WM_DESTROY):
                self._running.clear()
                return 0
        except Exception as exc:  # noqa: BLE001 - never let a handler kill the window
            self.log.debug("Overlay input handler: %s", exc)
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _cursor_pos(self) -> tuple[int, int]:
        point = self._w.POINT()
        self._u32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def _remember_position(self) -> None:
        if not self._position:
            return
        try:
            self.cfg.set("ui.position", [int(self._position[0]), int(self._position[1])])
            self.cfg.save()
        except Exception as exc:  # noqa: BLE001
            self.log.debug("Could not save the overlay position: %s", exc)

    def _context_menu(self, hwnd) -> None:
        user32 = self._u32
        menu = user32.CreatePopupMenu()
        paused = self.state.state is AssistantState.PAUSED
        user32.AppendMenuW(menu, MF_STRING, IDM_PAUSE, "Resume" if paused else "Pause")
        user32.AppendMenuW(menu, MF_STRING, IDM_CENTRE, "Move to the corner")
        user32.AppendMenuW(
            menu, MF_STRING, IDM_CLICKTHROUGH,
            "Catch clicks" if self.click_through else "Let clicks pass through",
        )
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, IDM_QUIT, "Quit JARVIS")

        x, y = self._cursor_pos()
        user32.SetForegroundWindow(hwnd)
        choice = user32.TrackPopupMenu(
            menu, TPM_RIGHTBUTTON | TPM_RETURNCMD, x, y, 0, hwnd, None
        )
        user32.DestroyMenu(menu)

        if choice == IDM_PAUSE and self.on_toggle_pause:
            self.on_toggle_pause()
        elif choice == IDM_CENTRE:
            self._position = list(self._initial_position_default())
            self._remember_position()
        elif choice == IDM_CLICKTHROUGH:
            self._set_click_through(not self.click_through)
        elif choice == IDM_QUIT:
            self._running.clear()
            if self.on_quit:
                self.on_quit()

    def _set_click_through(self, enabled: bool) -> None:
        """Add or drop WS_EX_TRANSPARENT so the overlay stops intercepting clicks.

        A layered window is hit-tested against its alpha channel, so the ring is
        already click-through where it is fully transparent; this covers the panel's
        faint backing as well, for anyone who wants the overlay to be purely
        something to look at.
        """
        user32 = self._u32
        GWL_EXSTYLE = -20
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        try:
            style = get_long(self._hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_TRANSPARENT) if enabled else (style & ~WS_EX_TRANSPARENT)
            set_long(self._hwnd, GWL_EXSTYLE, style)
            self.click_through = enabled
            self.cfg.set("ui.click_through", enabled)
            self.cfg.save()
        except Exception as exc:  # noqa: BLE001
            self.log.debug("Could not change the click-through style: %s", exc)

    def _initial_position_default(self) -> tuple[int, int]:
        user32 = self._u32
        margin = 28
        return (
            user32.GetSystemMetrics(0) - self.width - margin,
            user32.GetSystemMetrics(1) - self.height - margin - 48,
        )
