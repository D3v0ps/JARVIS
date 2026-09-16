"""Finding something on this machine willing to host the desk's page - and giving up well.

The page is the application, but a page needs a host, and the host worth having -
WebView2, through ``pywebview`` - is the one piece of this that may not be installed.
So the window is a chain rather than a dependency: WebView2 if it is there, Edge's own
app mode if it is not, the default browser if even Edge has been stripped out, and one
logged line if none of that is true. The ring keeps doing its job in every case, and
the assistant never learns which of the four it got, because nothing in here reaches
back into it: every failure ends in ``False``, never an exception.

Only two things here touch the world outside the process - :func:`load_webview` and
:func:`find_edge` - and they are deliberately module-level functions rather than
imports at the top of the file. That is what keeps the whole chain reachable from a
headless Linux box with no GUI, no Edge and no pywebview: replace the two seams and
every branch can be walked. It is also what keeps ``import jarvis.desk.window`` free.

Two behaviours are worth stating because they are easy to get backwards. Closing the
window **hides** it: the assistant is still listening, the tray still has it, and the
only thing allowed to end the process is :meth:`DeskWindow.stop`. And Edge is launched
against a throwaway profile under ``logs/``, never the operator's own - a browser
window we open must not inherit his cookies, his extensions or his history.

A third thing, because it looks like an omission rather than a decision: a frameless
window has no resize border. ``frameless=True`` is a ``FormBorderStyle`` of ``None`` on
WinForms, the operating system draws no grip, and pywebview exposes no hit-test hook to
put one back. So the page grows a grip of its own in its bottom-right corner and sends
``{"type": "window", "action": "resize", ...}``; the server validates it and calls
:meth:`DeskWindow.resize`, which is the only reason the remembered size is reachable at
all. :meth:`maximize`, :meth:`restore` and :meth:`minimize` exist for the same reason -
the buttons in the page's own title bar have nothing else to press.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable

from jarvis.core.logging import get_logger

__all__ = ["CHAIN", "MODES", "DeskWindow", "find_edge", "load_icon", "load_webview"]

#: What ``ui.window_mode`` may say. Anything else is treated as ``auto`` and logged.
MODES: tuple[str, ...] = ("auto", "webview", "edge", "browser", "off")

#: The fallback order, best first. ``auto`` walks it; a named mode picks one rung and
#: does not fall off it - if the operator asked for Edge, silently giving him the
#: default browser instead would be a lie about what he is looking at.
CHAIN: tuple[str, ...] = ("webview", "edge", "browser")

TITLE = "J.A.R.V.I.S."
#: The page's own background, so the frame does not flash white before the CSS lands.
BACKGROUND = "#05080c"
DEFAULT_SIZE: tuple[int, int] = (980, 720)
#: Narrower or shorter than this and the page stops being an application: the two
#: columns collapse over each other, the working line wraps into the telemetry and
#: the composer is pushed off the bottom. A minimum that renders a broken window is
#: not a minimum, so this is the smallest size that still keeps both columns.
MIN_SIZE: tuple[int, int] = (900, 620)

#: A drag fires a move event per frame. Writing YAML at sixty hertz would be absurd,
#: so the geometry is saved once the operator has stopped moving the window.
GEOMETRY_SAVE_DELAY = 0.75

#: Edge's profile lives beside the logs, inside the project, and is disposable.
EDGE_PROFILE = "logs/edge-profile"

#: The reactor, as ``installer/make_icon.py`` writes it. Without this the taskbar and
#: Alt-Tab show the Python logo, which is the wrong application entirely.
ICON_FILE = "assets/jarvis.ico"

#: Windows-only ``subprocess`` flag; absent on every other platform, hence the getattr.
_CREATE_NO_WINDOW = 0x08000000


def load_webview() -> Any | None:
    """Return the ``webview`` module, or None when pywebview is not installed.

    A seam, not a convenience: the tests hand back a fake module here, which is the
    only way to walk the webview branch on a machine with no GUI.
    """
    try:
        import webview  # noqa: PLC0415 - optional, and the whole point of this function
    except Exception:  # noqa: BLE001 - a half-installed pywebview raises far worse
        return None
    return webview


def load_icon(path: Path) -> Any | None:
    """Return a WinForms ``Icon`` for the file, or None where one cannot be made.

    The third seam, and there for the same reason as the other two: without it the
    only branch of :meth:`DeskWindow._set_window_icon` reachable from a machine with
    no pythonnet is the one that gives up, and the branch that matters is untestable.
    """
    try:
        from System.Drawing import Icon  # noqa: PLC0415 - pythonnet, Windows only

        return Icon(str(path))
    except Exception:  # noqa: BLE001 - no pythonnet, or a file that is not an icon
        return None


def find_edge() -> str | None:
    """Locate ``msedge.exe``: the PATH first, then both Program Files trees.

    The 32-bit tree is searched before the 64-bit one because that is where Edge
    actually installs itself on Windows 11. Returns None when Edge is not there.
    """
    for name in ("msedge", "msedge.exe"):
        found = shutil.which(name)
        if found:
            return found
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable)
        if not base:
            continue
        candidate = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:  # an unreadable drive is simply not a place Edge lives
            continue
    return None


class DeskWindow:
    """The desk's page in a window, by whichever of four means this machine allows."""

    def __init__(
        self,
        cfg: Any,
        url: str | Callable[[], str],
        *,
        logger: logging.Logger | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        #: A string, or something to ask. The ticket in the server's URL is single-use
        #: and expires after two minutes, so a URL captured when the window was built
        #: is worthless by the time the tray opens it an hour later. Whoever builds the
        #: window may therefore hand over ``lambda: server.url`` instead of a string,
        #: and every open asks for a fresh one.
        self._url_source: str | Callable[[], str] = url
        self.log = logger or get_logger("desk.window")
        #: Called when the operator closes the window himself, so the tray can say so.
        self.on_close = on_close

        self.mode = self._read_mode()
        self._backend = "none"
        self._lock = threading.RLock()
        self._visible = False
        self._save_delay = GEOMETRY_SAVE_DELAY

        self._webview: Any = None
        self._window: Any = None
        #: Set by :meth:`stop` so the closing handler stops vetoing the close.
        self._closing_for_real = threading.Event()

        self._edge_exe: str | None = None
        self._process: Any = None

        self._geometry_timer: threading.Timer | None = None
        self._pending_size: list[int] | None = None
        self._pending_pos: list[int] | None = None

    # --- what this machine can do -----------------------------------------------------
    @property
    def url(self) -> str:
        """The address to open, asked for afresh every time rather than remembered.

        Reading this from a callable source mints a new ticket when the last one has
        been spent, which is the whole point: a second open must not be refused by our
        own front door. ``""`` when the source will not answer - there is nothing
        useful to open, and a broken window is never worth an exception.
        """
        source = self._url_source
        if not callable(source):
            return str(source or "")
        try:
            return str(source() or "")
        except Exception as exc:  # noqa: BLE001 - a server that has stopped, most likely
            self.log.debug("The desk server would not give me a URL: %s", exc)
            return ""

    @property
    def backend(self) -> str:
        """``webview`` | ``edge`` | ``browser`` | ``none`` - what is hosting the page."""
        return self._backend

    @property
    def needs_main_thread(self) -> bool:
        """Only pywebview's loop does; everything else is a process or a no-op."""
        return self._backend == "webview"

    @property
    def visible(self) -> bool:
        """Whether a window is up, as far as this side of the glass can tell."""
        return self._visible

    def available(self) -> bool:
        """Whether anything here would host the window. Never raises, never starts one."""
        if self.mode == "off":
            return False
        if self._backend != "none":
            return True
        return any(self._probe(name) for name in self._order())

    # --- the chain ---------------------------------------------------------------------
    def start(self) -> bool:
        """Open the window with the best host available. False, never an exception."""
        with self._lock:
            if self._backend == "browser":
                # A tab is not a handle. There is nothing to show, nothing to raise and
                # no way to tell whether the operator closed it, so being asked to open
                # the window again means opening it again - from the top of the chain,
                # since the rung that was refused last time may be reachable now.
                self._backend = "none"
            if self._backend != "none":
                return True
            if self.mode == "off":
                self.log.info("The desk window is switched off; the ring will do, sir.")
                return False
            for name in self._order():
                try:
                    opened = self._attempt(name)
                except Exception as exc:  # noqa: BLE001 - a host that fails is not a crash
                    self.log.debug("The %s window host declined: %s", name, exc)
                    opened = False
                if opened:
                    self._backend = name
                    self._visible = True
                    if name == "webview":
                        self._set_window_icon()
                    self.log.info("The desk window is up, hosted by %s.", name)
                    return True
            self.log.info(
                "Nothing on this machine will host the desk window; the ring remains."
            )
            return False

    def run(self) -> None:
        """Block in pywebview's loop. Every other backend has nothing to block on."""
        if self._backend != "webview" or self._webview is None:
            return
        try:
            self._webview.start()
        except Exception as exc:  # noqa: BLE001 - the loop dying must not take the turn
            self.log.warning("The desk window closed unexpectedly: %s", exc)
        finally:
            self._visible = False

    def _order(self) -> tuple[str, ...]:
        """The hosts to try, in order: all of them, or the one that was demanded."""
        if self.mode == "auto":
            return CHAIN
        if self.mode in CHAIN:
            return (self.mode,)
        return ()

    def _attempt(self, name: str) -> bool:
        if name == "webview":
            return self._try_webview()
        if name == "edge":
            return self._try_edge()
        if name == "browser":
            return self._try_browser()
        return False

    def _probe(self, name: str) -> bool:
        """Is this host installed? Asked by :meth:`available`, which opens nothing."""
        try:
            if name == "webview":
                return load_webview() is not None
            if name == "edge":
                return find_edge() is not None
            return name == "browser"
        except Exception:  # noqa: BLE001 - an unanswerable question is a no
            return False

    # --- 1: pywebview ------------------------------------------------------------------
    def _try_webview(self) -> bool:
        """WebView2 in a frameless window of our own, remembered where he left it."""
        webview = load_webview()
        if webview is None:
            self.log.debug("pywebview is not installed; trying Edge's app mode.")
            return False
        if threading.current_thread() is not threading.main_thread():
            # A WebView2 window can only be built on the thread that owns the message
            # pump, and this is not it. Reporting success and deferring the window would
            # be a promise with nothing behind it - the tray would show a menu item that
            # opens nothing - so the rung is refused and Edge gets the page instead.
            self.log.info(
                "A window of my own needs the main thread, sir; I shall use Edge instead."
            )
            return False

        url = self.url  # fresh: the ticket in it is single-use and expires in two minutes
        if not url:
            self.log.debug("There is no address to open; there is no window to make.")
            return False

        self._closing_for_real.clear()  # a window reopened after a stop may be closed again
        width, height = self._configured_size()
        position = self._configured_pos()
        window = webview.create_window(
            TITLE,
            url=url,
            width=width,
            height=height,
            x=None if position is None else position[0],
            y=None if position is None else position[1],
            frameless=True,
            # Off, and it matters: easy_drag makes the *whole page* a drag handle, so no
            # text can be selected anywhere and dragging the voice-speed slider moves the
            # window instead of the slider. The page's own .pywebview-drag-region on its
            # title bar is registered separately by pywebview's customize.js and keeps
            # working without this - which is exactly what the page's comment assumes.
            easy_drag=False,
            resizable=True,
            min_size=MIN_SIZE,
            background_color=BACKGROUND,
            on_top=bool(self.cfg.get("ui.window_on_top", False)),
        )
        self._webview = webview
        self._window = window
        self._subscribe(window)
        return True

    def _subscribe(self, window: Any) -> None:
        """Wire the four events we care about, forgiving a pywebview that lacks them."""
        events = getattr(window, "events", None)
        for name, handler in (
            ("closing", self._on_closing),
            ("moved", self._on_moved),
            ("resized", self._on_resized),
            ("shown", self._on_shown),
        ):
            event = getattr(events, name, None)
            if event is None:
                continue
            try:
                event += handler  # pywebview's Event subscribes in place and returns itself
            except Exception as exc:  # noqa: BLE001 - an older pywebview, nothing worse
                self.log.debug("Could not subscribe to the %s event: %s", name, exc)

    def _on_closing(self) -> bool:
        """Veto the close and hide instead, unless :meth:`stop` asked for it in earnest.

        Returning False is how pywebview is told "not yet". Without it, the close
        button would end the window for the rest of the session and the tray would
        have nothing to bring back.
        """
        if self._closing_for_real.is_set():
            return True
        self._visible = False
        try:
            if self._window is not None:
                self._window.hide()
        except Exception as exc:  # noqa: BLE001
            self.log.debug("Could not hide the window on close: %s", exc)
        if self.on_close is not None:
            try:
                self.on_close()
            except Exception as exc:  # noqa: BLE001 - a listener is not the window's problem
                self.log.debug("The window's close listener raised: %s", exc)
        return False

    def _on_shown(self, *_args: Any) -> None:
        """The form exists now, which is the first moment its icon can be assigned.

        :meth:`start` tries as soon as the window is created, but on most pywebview
        backends nothing native exists until the loop has drawn it, so that attempt is
        a no-op and this is the one that lands. Both are guarded; the cost of missing
        is the default icon, not a window. The arguments are swallowed because
        pywebview has not always agreed with itself about how many it passes.
        """
        self._set_window_icon()

    def _on_moved(self, x: Any, y: Any) -> None:
        try:
            self._pending_pos = [int(x), int(y)]
        except (TypeError, ValueError):
            return
        self._schedule_geometry_save()

    def _on_resized(self, width: Any, height: Any) -> None:
        try:
            self._pending_size = [int(width), int(height)]
        except (TypeError, ValueError):
            return
        self._schedule_geometry_save()

    def _schedule_geometry_save(self) -> None:
        """Debounce: the last move of a drag is the only one worth writing down."""
        with self._lock:
            if self._geometry_timer is not None:
                self._geometry_timer.cancel()
            timer = threading.Timer(max(0.0, self._save_delay), self._save_geometry)
            timer.daemon = True
            self._geometry_timer = timer
        timer.start()

    def _save_geometry(self) -> None:
        """Write the remembered size and position back to ``config.yaml``."""
        with self._lock:
            size, position = self._pending_size, self._pending_pos
            self._pending_size = self._pending_pos = None
            self._geometry_timer = None
        if size is None and position is None:
            return
        try:
            if size is not None:
                self.cfg.set("ui.window_size", size)
            if position is not None:
                self.cfg.set("ui.window_pos", position)
            self.cfg.save()
        except Exception as exc:  # noqa: BLE001 - a read-only config is not worth a crash
            self.log.debug("Could not remember the window geometry: %s", exc)

    # --- 2: Edge in app mode -----------------------------------------------------------
    def _try_edge(self) -> bool:
        """``msedge --app=<url>``: the same page, no browser chrome, its own profile."""
        exe = self._edge_exe or find_edge()
        if not exe:
            self.log.debug("Edge is not installed here; falling back to the browser.")
            return False
        self._edge_exe = exe
        return self._launch_edge()

    def _launch_edge(self) -> bool:
        profile = self._edge_profile_dir()
        try:
            profile.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log.debug("Could not make Edge's profile directory: %s", exc)
            return False

        # Asked for here rather than carried from __init__: relaunching Edge after the
        # operator closed it needs a ticket that has not already been spent.
        url = self.url
        if not url:
            self.log.debug("There is no address to open; Edge has nothing to show.")
            return False
        width, height = self._configured_size()
        command = [
            str(self._edge_exe),
            f"--app={url}",
            # A dedicated profile, inside the project: never the operator's own, which
            # would drag his cookies, extensions and history into the assistant.
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--window-size={width},{height}",
        ]
        position = self._configured_pos()
        if position is not None:
            command.append(f"--window-position={position[0]},{position[1]}")

        extra: dict[str, Any] = {}
        if sys.platform == "win32":
            # No console flashes on the way to a windowless assistant (§24.7).
            extra["creationflags"] = _CREATE_NO_WINDOW
        try:
            self._process = subprocess.Popen(  # noqa: S603 - a path we found ourselves
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **extra,
            )
        except (OSError, ValueError) as exc:
            self.log.debug("Edge would not start: %s", exc)
            return False
        return True

    def _edge_profile_dir(self) -> Path:
        return _project_root(self.cfg) / EDGE_PROFILE

    def _edge_alive(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            return process.poll() is None
        except Exception:  # noqa: BLE001 - a process object we can no longer question
            return False

    def _stop_edge(self) -> None:
        """End the one process we started, and only that one."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired, or a fake
                    process.kill()
        except Exception as exc:  # noqa: BLE001 - it had already gone; that is the goal
            self.log.debug("Edge had already closed: %s", exc)

    # --- 3: the default browser --------------------------------------------------------
    def _try_browser(self) -> bool:
        """A tab, with chrome around it. Honest, and better than no window at all."""
        url = self.url  # fresh, for the same reason Edge's is: one ticket, one use
        if not url:
            return False
        opened = webbrowser.open(url)
        if opened is False:
            self.log.debug("No default browser answered.")
            return False
        return True

    # --- showing and hiding ------------------------------------------------------------
    def show(self) -> bool:
        """Bring the window back. False when this backend has no window to bring."""
        with self._lock:
            if self._backend == "webview" and self._window is not None:
                try:
                    self._window.show()
                except Exception as exc:  # noqa: BLE001
                    self.log.debug("Could not show the window: %s", exc)
                    return False
                self._visible = True
                return True
            if self._backend == "edge":
                # There is no honest way to raise a window we do not own a handle to,
                # and the operator may well have closed it: start a new one.
                if self._edge_alive():
                    self._visible = True
                    return True
                if not self._launch_edge():
                    return False
                self._visible = True
                return True
            return False

    def hide(self) -> bool:
        """Put the window away without ending the session. False where it means nothing."""
        with self._lock:
            if self._backend == "webview" and self._window is not None:
                try:
                    self._window.hide()
                except Exception as exc:  # noqa: BLE001
                    self.log.debug("Could not hide the window: %s", exc)
                    return False
                self._visible = False
                return True
            if self._backend == "edge":
                # Closing it is the only hiding Edge offers; :meth:`show` reopens it.
                self._stop_edge()
                self._visible = False
                return True
            return False

    def toggle(self) -> bool:
        """Whichever of :meth:`show` and :meth:`hide` the window is not already doing."""
        if self._backend == "edge":
            return self.hide() if self._edge_alive() else self.show()
        return self.hide() if self._visible else self.show()

    # --- what the page's own title bar drives -------------------------------------------
    def resize(self, width: Any, height: Any) -> bool:
        """Resize the window to what the page's grip dragged. False where it means nothing.

        This exists because ``frameless=True`` takes the operating system's resize
        border away and pywebview gives nothing back in its place, so without it the
        remembered size could be written but never changed, and the whole responsive
        layout would be unreachable. The size is clamped to :data:`MIN_SIZE` and to the
        screen: a grip dragged off the edge of the desk means "as big as you go".
        """
        size = _pair((width, height))
        if size is None:
            self.log.debug("The page asked for a size I cannot read: %r x %r", width, height)
            return False
        with self._lock:
            window = self._window if self._backend == "webview" else None
        if window is None:
            return False
        wide, high = self._clamp(*size)
        try:
            window.resize(wide, high)
        except Exception as exc:  # noqa: BLE001 - a grip is never worth a traceback
            self.log.debug("Could not resize the window: %s", exc)
            return False
        # Remembered here rather than left to the resized event: not every backend
        # fires one for a resize it was asked for, and this is the size he chose.
        self._on_resized(wide, high)
        return True

    def maximize(self) -> bool:
        """Fill the screen. False on every host that is not a window of our own."""
        return self._window_verb("maximize")

    def restore(self) -> bool:
        """Come back from maximised or minimised, to the size that was remembered."""
        return self._window_verb("restore")

    def minimize(self) -> bool:
        """To the taskbar - which is not hiding: :meth:`hide` is what the tray does."""
        return self._window_verb("minimize")

    def _window_verb(self, name: str) -> bool:
        """One of pywebview's own window verbs, guarded from the page that asked for it.

        The name never comes off a web page: the three callers above are the only
        things that reach this, for the same reason the server dispatches by table.
        """
        with self._lock:
            window = self._window if self._backend == "webview" else None
        if window is None:
            return False
        method = getattr(window, name, None)
        if not callable(method):
            self.log.debug("This pywebview cannot %s; the page's button is inert.", name)
            return False
        try:
            method()
        except Exception as exc:  # noqa: BLE001 - a title bar button, nothing more
            self.log.debug("The window would not %s: %s", name, exc)
            return False
        return True

    def _clamp(self, width: int, height: int) -> tuple[int, int]:
        """Between :data:`MIN_SIZE` and the screen, with the minimum winning a tie.

        A screen smaller than the minimum is a window that overflows it, which is the
        lesser of the two evils: too small is a broken layout, too large merely runs
        off the edge and can be dragged back.
        """
        width = max(MIN_SIZE[0], width)
        height = max(MIN_SIZE[1], height)
        screen = self._screen_size()
        if screen is not None:
            width = min(width, max(MIN_SIZE[0], screen[0]))
            height = min(height, max(MIN_SIZE[1], screen[1]))
        return width, height

    def _screen_size(self) -> tuple[int, int] | None:
        """The primary screen as pywebview reports it, or None when it will not say."""
        try:
            screens = list(getattr(self._webview, "screens", None) or ())
            if not screens:
                return None
            return int(screens[0].width), int(screens[0].height)
        except Exception:  # noqa: BLE001 - an older pywebview, or no display at all
            return None

    # --- the icon on the taskbar ----------------------------------------------------------
    def _set_window_icon(self) -> None:
        """Put the reactor in the taskbar and in Alt-Tab, in place of the Python logo.

        pywebview has no API for this. The icon belongs to the WinForms form it built,
        which it keeps in ``webview.gui.BrowserView.instances`` under the window's uid,
        so this reaches in - guarded at every step, because on any other backend, any
        other pywebview, or a machine without pythonnet, the correct outcome is the
        default icon and no noise.

        The other half of the job is not ours: Windows groups a taskbar button by the
        process's AppUserModelID, which ``jarvis/__main__.py`` must set before any
        window exists, or this icon appears on a button still labelled Python.
        """
        path = self._icon_path()
        webview, window = self._webview, self._window
        uid = getattr(window, "uid", None)
        if path is None or webview is None or uid is None:
            return
        try:
            form = webview.gui.BrowserView.instances[uid]
        except Exception as exc:  # noqa: BLE001 - most often: the form is not drawn yet
            self.log.debug("There is no form to put an icon on yet: %s", exc)
            return
        icon = load_icon(path)
        if icon is None:
            self.log.debug("Nothing here can read %s; the default icon stays.", path)
            return
        try:
            form.Icon = icon
        except Exception as exc:  # noqa: BLE001 - the window simply keeps its own icon
            self.log.debug("The window kept the default icon: %s", exc)

    def _icon_path(self) -> Path | None:
        """The shipped ``.ico``, or None when it was never generated."""
        try:
            candidate = _project_root(self.cfg) / ICON_FILE
            return candidate if candidate.is_file() else None
        except OSError:  # an unreadable path is simply not a place an icon lives
            return None

    # --- the end -----------------------------------------------------------------------
    def stop(self) -> None:
        """Close the window for real. Idempotent, and safe from any thread."""
        with self._lock:
            timer, self._geometry_timer = self._geometry_timer, None
            window, self._window = self._window, None
            backend, self._backend = self._backend, "none"
            self._visible = False
            self._closing_for_real.set()
        if timer is not None:
            timer.cancel()
        self._save_geometry()
        if window is not None:
            try:
                window.destroy()
            except Exception as exc:  # noqa: BLE001 - it may have gone already
                self.log.debug("The window had already closed: %s", exc)
        if backend == "edge":
            self._stop_edge()
        self._webview = None

    # --- configuration -----------------------------------------------------------------
    def _read_mode(self) -> str:
        raw = self.cfg.get("ui.window_mode", "auto") if hasattr(self.cfg, "get") else "auto"
        mode = str(raw or "auto").strip().lower()
        if mode not in MODES:
            self.log.warning("ui.window_mode is %r, which I do not know; using auto.", raw)
            return "auto"
        return mode

    def _configured_size(self) -> tuple[int, int]:
        width, height = _pair(self.cfg.get("ui.window_size", None)) or DEFAULT_SIZE
        return max(MIN_SIZE[0], width), max(MIN_SIZE[1], height)

    def _configured_pos(self) -> tuple[int, int] | None:
        return _pair(self.cfg.get("ui.window_pos", None))


def _pair(value: Any) -> tuple[int, int] | None:
    """Two integers out of a config value that the operator may have edited by hand."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _project_root(cfg: Any) -> Path:
    """Where relative paths in the config are anchored."""
    path = getattr(cfg, "path", None)
    if path:
        try:
            return Path(path).resolve().parent
        except (OSError, TypeError):
            pass
    from jarvis.config import project_root  # noqa: PLC0415 - avoids a cycle at import

    return project_root()
