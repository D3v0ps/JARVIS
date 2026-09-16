"""The desk's own front door: one loopback socket, one ticket, one window.

The phone is a guest and is treated like one — it pairs, it is rate-limited, and
the guard in :mod:`jarvis.remote.session` refuses anything guarded. The desk is not
a guest. It is the same person, at the same keyboard, who could already open
PowerShell by hand, so the window has exactly the rights the microphone has and
nothing in this module knows what a tier is.

That makes the *door* the only thing worth being paranoid about, and this module is
almost entirely door:

* the socket is bound to ``127.0.0.1`` on an OS-chosen port, and a configured host
  that is not a loopback literal is refused rather than quietly obeyed;
* the URL carries a single-use ticket that is exchanged for a session cookie and
  burned, so the copy of it left in Edge's command line and in the process list is
  worthless a moment later, and worthless anyway after two minutes;
* every request must come from a loopback peer, address this server by its own
  ``Host``, and — if it volunteers an ``Origin`` at all — volunteer ours. That is the
  DNS-rebinding defence: a page on the open web that resolves ``evil.example`` to
  ``127.0.0.1`` still cannot spend our port, because it cannot forge those.

Flask is imported inside :meth:`DeskServer.create_app`, so this module imports on a
bare Linux box with nothing installed, and a JARVIS started with ``--no-window``
never opens a socket at all. Nothing here raises into the assistant: a failure logs
and returns ``False`` or ``None``, and JARVIS keeps working with the whole window
absent.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

from jarvis.core.logging import get_logger
from jarvis.remote.session import Routine, load_routines

__all__ = [
    "COOKIE_NAME",
    "DeskServer",
    "LOCKED_OUT_HTML",
    "MAX_SAY",
    "SETTINGS",
    "STATUS_INTERVAL",
    "TICKET_TTL",
]

#: The session cookie. Deliberately not the phone's name: the two doors are separate.
COOKIE_NAME = "jarvis_desk"

#: How long the ticket in the URL stays good. It only has to survive one command line.
TICKET_TTL = 120.0

#: A typed turn is a sentence, not a novel. Anything longer is a paste accident.
MAX_SAY = 2000

#: Seconds between telemetry readings while at least one window is watching.
STATUS_INTERVAL = 2.0

#: How long a writer waits on a silent bus before sending a ping.
HEARTBEAT = 20.0

#: Frames queued for a window that has stopped reading. Oldest goes over the side.
_OUTBOX = 256

#: Bind addresses this server will accept. Not ``localhost``: a name can be pointed
#: somewhere else, and the whole point of this module is that the address cannot be.
_LOOPBACK_BINDS = {"127.0.0.1", "::1", "[::1]"}

#: Peers we will talk to, after ``::ffff:127.0.0.1`` has been folded onto IPv4.
_LOOPBACK_PEERS = {"127.0.0.1", "::1"}

#: Static file names we will serve. No slashes, no dots leading, no surprises.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
}

#: Pushed into a writer's bus queue to make it look at its outbox. It is never sent.
_WAKE = object()

#: ``jarvis.desk.bus.CLOSED``, repeated rather than imported so this module still
#: loads when the bus does not. A window that sees it has nothing left to wait for.
_BUS_CLOSED = "closed"

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}

#: What a browser gets when it has neither a live ticket nor our cookie. A stack
#: trace here would be both a leak and useless: the way back in is the tray.
LOCKED_OUT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>JARVIS</title>
<style>html{background:#07090c;color:#8fa3b0;font:15px/1.6 system-ui,sans-serif}
main{max-width:30rem;margin:18vh auto;padding:0 1.5rem;text-align:center}
h1{font-weight:400;font-size:1.3rem;color:#cfe3ee;margin:0 0 .6rem}</style></head>
<body><main><h1>Not just now, sir.</h1>
<p>This link has already been used. Open the window again from the JARVIS tray icon
and I shall be right with you.</p></main></body></html>
"""

#: What the operator gets when ``static/desk.html`` is not there. The ticket has
#: already been spent by the time we read the file, so refusing with a 500 would
#: lock him out of a window he is entitled to; this at least tells him what is wrong.
MISSING_PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>JARVIS</title>
<style>html{background:#07090c;color:#8fa3b0;font:15px/1.6 system-ui,sans-serif}
main{max-width:30rem;margin:18vh auto;padding:0 1.5rem;text-align:center}
h1{font-weight:400;font-size:1.3rem;color:#cfe3ee;margin:0 0 .6rem}</style></head>
<body><main><h1>The window is missing, sir.</h1>
<p>desk.html was not found next to the server. The voice side of me is unaffected.</p>
</main></body></html>
"""


def _clamped(low: float, high: float) -> Callable[[Any], float]:
    """A validator that accepts a finite number and pins it inside ``[low, high]``.

    Clamping rather than refusing is deliberate for the sliders: a window that sends
    1.4 for a speed capped at 1.3 meant "as fast as you go", and saying no to it
    would be pedantry. Nonsense — a string, a list, ``NaN``, ``True`` — is still a
    refusal, because it means the frame did not come from our page.
    """

    def check(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"expected a number, got {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("expected a finite number")
        return round(min(high, max(low, number)), 3)

    return check


def _flag(value: Any) -> bool:
    """A validator for a genuine on/off setting."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE:
            return True
        if word in _FALSE:
            return False
    raise ValueError(f"expected true or false, got {value!r}")


#: Every config key the window may touch, and the only shape each may take. This is
#: an allowlist and not a denylist on purpose: a new config key must be deliberately
#: added here before a web page can reach it, and ``brain.model`` never will be.
SETTINGS: dict[str, Callable[[Any], Any]] = {
    "tts.speed": _clamped(0.5, 1.6),
    "audio.chime_volume": _clamped(0.0, 1.0),
    "assistant.brief_mode": _flag,
    "wake.sensitivity": _clamped(0.05, 0.95),
    "ui.always_on_top": _flag,
}


class _NoBus:
    """Stands in for :class:`~jarvis.desk.bus.DeskBus` when that module is missing.

    It exists so a broken install loses the window and nothing else: the property
    still returns an object with the right shape, :meth:`DeskServer.start` sees it
    and refuses cleanly, and no caller anywhere gets an ``ImportError`` thrown at it.
    """

    def publish(self, type: str, /, **payload: Any) -> None:
        return None

    def subscribe(self) -> "queue.Queue[Any]":
        return queue.Queue(maxsize=1)

    def unsubscribe(self, q: Any) -> None:
        return None

    def replay(self) -> list:
        return []

    def close(self) -> None:
        return None


class _Window:
    """One connected window: its socket, its two queues and its closing flag.

    Reading and writing are split across two threads because ``ws.receive()`` blocks
    and the bus must keep flowing underneath it. Only the writer ever touches the
    socket; the reader hands its answers over through :attr:`outbox` and nudges the
    writer awake, so two threads can never be inside ``send()`` at once.
    """

    def __init__(self, ws: Any, subscriber: Any) -> None:
        self.ws = ws
        self.subscriber = subscriber
        self.outbox: "queue.Queue[dict]" = queue.Queue(maxsize=_OUTBOX)
        self.closed = threading.Event()

    def offer(self, payload: dict) -> None:
        """Queue one frame for the writer. Never blocks and never raises."""
        try:
            self.outbox.put_nowait(payload)
        except queue.Full:
            try:
                self.outbox.get_nowait()
            except queue.Empty:  # pragma: no cover - another thread drained it first
                pass
            try:
                self.outbox.put_nowait(payload)
            except queue.Full:  # pragma: no cover - a window this far behind is gone
                return
        self.wake()

    def wake(self) -> None:
        """Nudge the writer, which is parked on the bus queue, to look at the outbox."""
        try:
            self.subscriber.put_nowait(_WAKE)
        except Exception:  # noqa: BLE001 - a full queue already wakes it
            pass


class DeskServer:
    """Serves the desk window over loopback and carries events to and from it."""

    def __init__(
        self,
        cfg: Any,
        assistant: Any,
        logger: logging.Logger | None = None,
        *,
        bus: Any = None,
    ) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.log = logger or get_logger("desk")

        self.host = str(_cfg(cfg, "ui.window_host", "127.0.0.1")).strip() or "127.0.0.1"
        self.heartbeat = HEARTBEAT
        self.status_interval = STATUS_INTERVAL
        self.routines: list[Routine] = _routines(cfg, self.log)

        # Held in memory, never on disk, never in the log. Regenerated every run.
        self._ticket = secrets.token_urlsafe(32)
        self._ticket_expires = time.monotonic() + TICKET_TTL
        self._ticket_spent = False
        self._session = ""

        self._lock = threading.Lock()
        self._bus_lock = threading.Lock()
        self._bus: Any = bus
        self._running = threading.Event()
        self._closing = threading.Event()
        self._httpd: Any = None
        self._thread: threading.Thread | None = None
        self._poller: threading.Thread | None = None
        self._poll_wake = threading.Event()
        self._windows = 0
        self._bound_port = 0
        self._app: Any = None

        # An explicit table, built once. There is no dispatch by attribute name in
        # this file: a frame from a web page must never be able to choose a method.
        self._handlers: dict[str, Callable[[dict], dict | None]] = {
            "say": self._on_say,
            "listen": self._on_listen,
            "stop": self._on_stop,
            "confirm": self._on_confirm,
            "pause": self._on_pause,
            "resume": self._on_resume,
            "routine": self._on_routine,
            "set": self._on_set,
            "quit": self._on_quit,
        }

    # --- what the window is told ----------------------------------------------------
    @property
    def bus(self) -> Any:
        """The fan-out every face reads from, built on first use.

        Lazily, because a JARVIS started with ``--no-window`` should not allocate a
        two-hundred-event ring buffer for a page nobody will open.
        """
        with self._bus_lock:
            if self._bus is None:
                self._bus = self._make_bus()
            return self._bus

    def _make_bus(self) -> Any:
        try:
            from jarvis.desk.bus import DeskBus  # noqa: PLC0415 - lazy on purpose

            return DeskBus()
        except Exception as exc:  # noqa: BLE001 - a missing bus costs the window only
            self.log.error("The desk bus is unavailable, so no window can open: %s", exc)
            return _NoBus()

    # --- the door -------------------------------------------------------------------
    @property
    def url(self) -> str:
        """The one URL that opens a window, ticket and all. Do not log this."""
        return f"http://{self.display_host}:{self._bound_port}/?t={self._ticket}"

    @property
    def safe_url(self) -> str:
        """The same address without the ticket, which is what may be written down."""
        return f"http://{self.display_host}:{self._bound_port}/"

    @property
    def display_host(self) -> str:
        """How this server writes its own address."""
        return "[::1]" if self.host in ("::1", "[::1]") else "127.0.0.1"

    @property
    def host_header(self) -> str:
        """The only ``Host`` this server answers to."""
        return f"{self.display_host}:{self._bound_port}"

    @property
    def origin(self) -> str:
        """The only ``Origin`` this server accepts when one is offered."""
        return f"http://{self.host_header}"

    @property
    def port(self) -> int:
        """The port the OS actually gave us, or 0 before :meth:`start`."""
        return self._bound_port

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def windows(self) -> int:
        """How many windows are connected right now."""
        with self._lock:
            return self._windows

    # --- lifecycle ------------------------------------------------------------------
    def start(self) -> bool:
        """Open the socket. Returns False — never raises — when it cannot.

        The bind address is checked before anything else. A desk window has the
        rights of the keyboard, and a typo in ``config.yaml`` does not get to hand
        those to the local network.
        """
        if self._running.is_set():
            return True
        if self.host not in _LOOPBACK_BINDS:
            self.log.error(
                "Refusing to open the desk window: %r is not a loopback address. "
                "The window is only ever served to this machine.",
                self.host,
            )
            return False
        if isinstance(self.bus, _NoBus):
            self.log.error("Refusing to open the desk window: there is no event bus.")
            return False
        try:
            app = self.create_app()
        except ImportError as exc:
            self.log.warning(
                "The desk window needs Flask: %s. Install it with "
                "'pip install flask flask-sock'.",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - a broken window must not stop JARVIS
            self.log.exception("The desk window could not be prepared: %s", exc)
            return False

        try:
            self._httpd = self._make_server(app)
        except OSError as exc:
            self.log.error("The desk window could not bind %s (%s).", self.host, exc)
            return False
        except Exception as exc:  # noqa: BLE001
            self.log.exception("The desk window could not start: %s", exc)
            return False

        # The clock on the ticket starts when the door does, not when we were built.
        with self._lock:
            self._ticket_expires = time.monotonic() + TICKET_TTL
        self._closing.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._serve, name="jarvis-desk", daemon=True)
        self._thread.start()
        self.log.info("The desk window is served at %s.", self.safe_url)
        return True

    def stop(self) -> None:
        """Close the socket and let every window go. Safe to call twice, from anywhere."""
        self._closing.set()
        self._poll_wake.set()
        if not self._running.is_set():
            return
        self._running.clear()
        httpd = self._httpd
        for method in ("shutdown", "server_close", "close"):
            action = getattr(httpd, method, None)
            if not callable(action):
                continue
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.debug("The desk server's %s() failed: %s", method, exc)
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._httpd = None
        self._thread = None
        self.log.info("The desk window is closed.")

    def _serve(self) -> None:
        try:
            self._httpd.serve_forever()
        except Exception as exc:  # noqa: BLE001
            if self._running.is_set():
                self.log.exception("The desk server stopped unexpectedly: %s", exc)

    def _make_server(self, app: Any) -> Any:
        """Bind loopback on port 0 and read back what the OS handed us.

        Werkzeug's threaded server, because it is the only one in this stack that
        gives ``flask-sock`` the raw socket it needs for ``/ws/desk``.
        """
        from werkzeug.serving import make_server  # noqa: PLC0415 - lazy on purpose

        bind = "::1" if self.host in ("::1", "[::1]") else "127.0.0.1"
        server = make_server(bind, 0, app, threaded=True)
        self._bound_port = int(server.socket.getsockname()[1])
        return server

    # --- the app --------------------------------------------------------------------
    def create_app(self) -> Any:
        """Build the Flask app. Imports Flask here, so the module imports without it."""
        from flask import Flask  # noqa: PLC0415 - lazy on purpose
        from flask_sock import Sock  # noqa: PLC0415

        app = Flask(__name__, static_folder=None)
        app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
        sock = Sock(app)
        self._install_guard(app)
        self._register_routes(app)
        sock.route("/ws/desk")(self.desk_socket)
        self._app = app
        return app

    def _install_guard(self, app: Any) -> None:
        """Put the loopback, Host and Origin checks in front of every route."""
        from flask import Response  # noqa: PLC0415
        from werkzeug.exceptions import HTTPException  # noqa: PLC0415

        @app.before_request
        def _loopback_only() -> Any:
            reason = self.refusal()
            if not reason:
                return None
            self.log.error("Refusing a desk request: %s.", reason)
            return Response("Not for you.\n", status=403, mimetype="text/plain")

        @app.errorhandler(Exception)
        def _no_tracebacks(exc: Exception) -> Any:
            if isinstance(exc, HTTPException):
                return Response(f"{exc.code}\n", status=exc.code or 500, mimetype="text/plain")
            self.log.exception("A desk request failed: %s", exc)
            return Response("Something went wrong here, sir.\n", status=500, mimetype="text/plain")

    def _register_routes(self, app: Any) -> None:
        from flask import Response, request  # noqa: PLC0415

        @app.get("/")
        def index() -> Any:
            return self.open_window()

        @app.get("/static/<name>")
        def asset(name: str) -> Any:
            if not self.cookie_is_ours(request.cookies.get(COOKIE_NAME)):
                self.log.warning("An uncredentialed request asked for %r.", name)
                return Response("Not for you.\n", status=401, mimetype="text/plain")
            return self._static(name)

    # --- the checks every request and every socket goes through ------------------------
    def refusal(self) -> str:
        """Why the current request must not be served; ``""`` when it may be.

        Three questions, in the order that costs least. Failing to be in a request
        context at all is itself a refusal: default deny, so a future caller that
        forgets the context cannot accidentally open the door.
        """
        try:
            from flask import request  # noqa: PLC0415

            peer = _peer(request.remote_addr)
            host = (request.headers.get("Host") or "").strip()
            origin = (request.headers.get("Origin") or "").strip()
        except Exception:  # noqa: BLE001 - no request context, no entry
            return "there is no request to inspect"
        if peer not in _LOOPBACK_PEERS:
            return f"the peer {peer or 'unknown'!r} is not on the loopback interface"
        if host != self.host_header:
            return f"the Host header {host!r} is not {self.host_header!r}"
        if origin and origin != self.origin:
            return f"the Origin {origin!r} is not {self.origin!r}"
        return ""

    def cookie_is_ours(self, value: str | None) -> bool:
        """True when ``value`` is the session cookie this run handed out."""
        with self._lock:
            session = self._session
        if not value or not session:
            return False
        return secrets.compare_digest(str(value), session)

    def redeem(self, candidate: str) -> str:
        """Exchange a ticket for a session token, burning it. ``""`` when refused.

        One use, one run, two minutes. The ticket exists only to survive Edge's
        command line and the process list, both of which any other program on this
        machine can read, so it stops being a secret the moment it has done its job.
        """
        with self._lock:
            if not candidate:
                return ""
            if self._ticket_spent:
                self.log.warning("A desk ticket was offered twice; the second was refused.")
                return ""
            if time.monotonic() > self._ticket_expires:
                self.log.warning("A desk ticket was offered after it expired.")
                return ""
            if not secrets.compare_digest(str(candidate), self._ticket):
                self.log.warning("A desk ticket that is not ours was offered.")
                return ""
            self._ticket_spent = True
            self._session = secrets.token_urlsafe(32)
            self.log.info("The desk window redeemed its ticket.")
            return self._session

    # --- the page -----------------------------------------------------------------------
    def open_window(self) -> Any:
        """``GET /``: a cookie gets in, a live ticket buys a cookie, nothing else gets in."""
        from flask import Response, request  # noqa: PLC0415

        if self.cookie_is_ours(request.cookies.get(COOKIE_NAME)):
            return self._page()
        token = self.redeem(str(request.args.get("t", "") or ""))
        if not token:
            return Response(LOCKED_OUT_HTML, status=401, mimetype="text/html; charset=utf-8")
        response = self._page()
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="Strict",
            secure=False,  # this is http://127.0.0.1; Secure would make it unusable
            path="/",
        )
        return response

    def _page(self) -> Any:
        """The window itself, or an honest note when it is not on disk."""
        from flask import Response  # noqa: PLC0415

        try:
            body = (_STATIC_DIR / "desk.html").read_text(encoding="utf-8")
        except OSError as exc:
            self.log.error("The desk page is missing: %s", exc)
            body = MISSING_PAGE_HTML
        return Response(body, mimetype="text/html; charset=utf-8")

    def _static(self, name: str) -> Any:
        """One file from ``static/``, by a name that cannot walk out of it."""
        from flask import Response  # noqa: PLC0415

        if not _SAFE_NAME.fullmatch(name or ""):
            self.log.warning("Refusing the desk asset name %r.", name)
            return Response("No.\n", status=404, mimetype="text/plain")
        target = (_STATIC_DIR / name).resolve()
        if target.parent != _STATIC_DIR:
            self.log.error("Refusing the desk asset %r: it resolves outside static/.", name)
            return Response("No.\n", status=404, mimetype="text/plain")
        try:
            body = target.read_bytes()
        except OSError:
            return Response("Not here.\n", status=404, mimetype="text/plain")
        return Response(body, mimetype=_MIME.get(target.suffix.lower(), "application/octet-stream"))

    # --- the socket ---------------------------------------------------------------------
    def desk_socket(self, ws: Any) -> None:
        """``/ws/desk``: hello, then the replay, then everything as it happens.

        This thread reads; a second thread writes. Nothing the assistant publishes is
        ever written from the assistant's own thread, so a window that has stopped
        reading costs it a dropped event and not a single millisecond.
        """
        if not self._admit(ws):
            return
        bus = self.bus
        try:
            subscriber = bus.subscribe()
        except Exception as exc:  # noqa: BLE001
            self.log.error("A desk window could not subscribe to the bus: %s", exc)
            _send(ws, {"type": "error", "message": "I cannot reach my own events, sir."})
            return

        window = _Window(ws, subscriber)
        self._window_joined()
        writer = threading.Thread(
            target=self._write_loop, args=(window,), name="jarvis-desk-window", daemon=True
        )
        writer.start()
        try:
            self._read_loop(window)
        finally:
            window.closed.set()
            window.wake()
            writer.join(timeout=5)
            try:
                bus.unsubscribe(subscriber)
            except Exception:  # noqa: BLE001
                self.log.debug("Unsubscribing a desk window failed.", exc_info=True)
            self._window_left()

    def _admit(self, ws: Any) -> bool:
        """The same three checks as a request, plus the cookie, before a socket lives."""
        reason = self.refusal()
        if reason:
            self.log.error("Refusing a desk socket: %s.", reason)
            _send(ws, {"type": "error", "message": "Not for you."})
            _close(ws)
            return False
        try:
            from flask import request  # noqa: PLC0415

            cookie = request.cookies.get(COOKIE_NAME)
        except Exception:  # noqa: BLE001
            cookie = None
        if not self.cookie_is_ours(cookie):
            self.log.warning("An uncredentialed socket tried to open the desk window.")
            _send(ws, {"type": "error", "message": "Open the window from the tray, sir."})
            _close(ws)
            return False
        return True

    def _read_loop(self, window: _Window) -> None:
        """Take frames from the page and hand the answers to the writer."""
        while not window.closed.is_set() and not self._closing.is_set():
            try:
                message = window.ws.receive(timeout=self.heartbeat)
            except Exception:  # noqa: BLE001 - the window was closed
                return
            if message is None:
                continue  # a quiet moment; the writer does the pings
            reply = self.dispatch(message)
            if reply is not None:
                window.offer(reply)

    def _write_loop(self, window: _Window) -> None:
        """Hello, the replay, then the bus — all of it on this one thread."""
        if not _send(window.ws, self.hello()):
            window.closed.set()
            return
        try:
            history = list(self.bus.replay())
        except Exception:  # noqa: BLE001
            history = []
        for event in history:
            if not _send(window.ws, _frame(event)):
                window.closed.set()
                return
        # Subscribing happened before the replay so nothing could slip between them;
        # the price is that the queue may repeat what we just sent. ``history`` is
        # held for the life of the socket, so these identities stay unambiguous.
        already = {id(event) for event in history}

        while not window.closed.is_set() and not self._closing.is_set():
            if not self._flush(window):
                return
            try:
                item = window.subscriber.get(timeout=self.heartbeat)
            except queue.Empty:
                if not _send(window.ws, {"type": "ping"}):
                    window.closed.set()
                    return
                continue
            except Exception:  # noqa: BLE001 - the bus went away underneath us
                window.closed.set()
                return
            if item is _WAKE or id(item) in already:
                continue
            sent = _send(window.ws, _frame(item))
            if not sent or str(getattr(item, "type", "")) == _BUS_CLOSED:
                window.closed.set()
                return
        # One last flush: an error frame for the frame that closed the window is
        # still worth delivering, and the reader may have queued it after our
        # final pass round the loop.
        self._flush(window)

    def _flush(self, window: _Window) -> bool:
        """Send everything the reader has queued. False when the socket is gone."""
        while True:
            try:
                payload = window.outbox.get_nowait()
            except queue.Empty:
                return True
            if not _send(window.ws, payload):
                window.closed.set()
                return False

    # --- inbound frames -------------------------------------------------------------------
    def dispatch(self, message: Any) -> dict | None:
        """Route one inbound frame through the handler table. Never raises."""
        try:
            frame = json.loads(message)
        except (TypeError, ValueError):
            self.log.warning("A desk window sent something that is not JSON.")
            return {"type": "error", "message": "I expected JSON, sir."}
        if not isinstance(frame, dict):
            self.log.warning("A desk window sent a %s, not an object.", type(frame).__name__)
            return {"type": "error", "message": "I expected an object, sir."}
        kind = str(frame.get("type", "") or "").strip()
        handler = self._handlers.get(kind)
        if handler is None:
            self.log.warning("A desk window sent the unknown frame type %r.", kind)
            return {"type": "error", "message": f"I do not know the frame {kind or '(none)'}, sir."}
        try:
            return handler(frame)
        except Exception as exc:  # noqa: BLE001 - one bad frame is not a dead window
            self.log.exception("The desk frame %r failed: %s", kind, exc)
            return {"type": "error", "message": "That went wrong at my end, sir."}

    def _on_say(self, frame: dict) -> dict:
        text = str(frame.get("text", "") or "").strip()[:MAX_SAY]
        if not text:
            self.log.warning("An empty say frame from the desk window was refused.")
            return {"type": "error", "message": "There was nothing to say, sir."}
        found, accepted = self._ask("submit_desk_turn", text)
        if not found:
            return _no_hook("submit_desk_turn")
        if accepted is False:
            return {"type": "busy", "message": "I am in the middle of something, sir."}
        self.log.info("Desk turn (typed): %r", text[:80])
        return {"type": "ack", "for": "say"}

    def _on_listen(self, frame: dict) -> dict:
        found, armed = self._ask("arm_listening")
        if not found:
            return _no_hook("arm_listening")
        return {"type": "ack", "for": "listen", "ok": armed is not False}

    def _on_stop(self, frame: dict) -> dict:
        found, _ = self._ask("abort_turn")
        if not found:
            return _no_hook("abort_turn")
        return {"type": "ack", "for": "stop"}

    def _on_confirm(self, frame: dict) -> dict:
        granted = bool(frame.get("granted"))
        found, taken = self._ask("answer_confirmation", granted)
        if not found:
            return _no_hook("answer_confirmation")
        self.log.info("The desk window answered a confirmation: %s.", "yes" if granted else "no")
        return {"type": "ack", "for": "confirm", "granted": granted, "ok": taken is not False}

    def _on_pause(self, frame: dict) -> dict:
        found, _ = self._ask("pause")
        if not found:
            return _no_hook("pause")
        return {"type": "ack", "for": "pause"}

    def _on_resume(self, frame: dict) -> dict:
        found, _ = self._ask("resume")
        if not found:
            return _no_hook("resume")
        return {"type": "ack", "for": "resume"}

    def _on_routine(self, frame: dict) -> dict:
        name = " ".join(str(frame.get("name", "") or "").split())[:40]
        if not name:
            return {"type": "error", "message": "Which routine, sir?"}
        known = {routine.name.lower() for routine in self.routines}
        if name.lower() not in known:
            self.log.warning("The desk window asked for the unknown routine %r.", name)
            return {"type": "error", "message": f"I have no routine called {name}, sir."}
        found, said = self._ask("run_routine", name)
        if not found:
            return _no_hook("run_routine")
        self.log.info("Desk routine %r.", name)
        return {"type": "ack", "for": "routine", "name": name, "said": str(said or "")}

    def _on_set(self, frame: dict) -> dict:
        key = str(frame.get("key", "") or "").strip()
        rule = SETTINGS.get(key)
        if rule is None:
            self.log.warning("The desk window tried to set %r, which is not allowed.", key)
            return {
                "type": "error",
                "message": f"{key or 'That'} is not something I change from the window, sir.",
            }
        try:
            value = rule(frame.get("value"))
        except (TypeError, ValueError) as exc:
            self.log.warning("The desk window sent an unusable value for %s: %s", key, exc)
            return {"type": "error", "message": f"That is not a usable value for {key}, sir."}
        found, applied = self._ask("apply_setting", key, value)
        if not found:
            return _no_hook("apply_setting")
        if applied is False:
            return {"type": "error", "message": f"I could not change {key}, sir."}
        self.log.info("The desk window set %s to %r.", key, value)
        return {"type": "ack", "for": "set", "key": key, "value": value}

    def _on_quit(self, frame: dict) -> dict:
        stop = getattr(self.assistant, "stop", None)
        if not callable(stop):
            return _no_hook("stop")
        self.log.info("The desk window asked JARVIS to shut down.")
        # On its own thread: stopping joins the very threads this frame arrived on.
        threading.Thread(
            target=self._quit, args=(stop,), name="jarvis-desk-quit", daemon=True
        ).start()
        return {"type": "ack", "for": "quit"}

    def _quit(self, stop: Callable[[], Any]) -> None:
        try:
            stop()
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Shutting down from the desk window failed: %s", exc)

    def _ask(self, name: str, *args: Any) -> tuple[bool, Any]:
        """Call one of the assistant's six desk hooks. ``(found, result)``, never a raise."""
        handler = getattr(self.assistant, name, None)
        if not callable(handler):
            self.log.error("This build of JARVIS has no %s hook; the window asked for it.", name)
            return False, None
        try:
            return True, handler(*args)
        except Exception as exc:  # noqa: BLE001 - the window is not worth a crash
            self.log.exception("The assistant's %s hook failed: %s", name, exc)
            return True, False

    # --- what the window is told on arrival --------------------------------------------------
    def hello(self) -> dict:
        """The first frame: everything the page needs before a single event arrives."""
        return {
            "type": "hello",
            "version": _version(),
            "model": str(_cfg(self.cfg, "brain.model", "") or ""),
            "whisper": str(_cfg(self.cfg, "stt.model", "") or ""),
            "gpu": str(_cfg(self.cfg, "system.gpu_name", "") or ""),
            "phone_url": self._phone_url(),
            "paused": self.current_state() == "paused",
            "routines": [routine.as_json() for routine in self.routines],
            "tools": self._tools(),
        }

    def current_state(self) -> str:
        bus = self._state_bus()
        try:
            return str(bus.state) if bus is not None else "idle"
        except Exception:  # noqa: BLE001
            return "idle"

    def _state_bus(self) -> Any:
        parts = getattr(self.assistant, "parts", None)
        return getattr(parts, "state", None) or getattr(self.assistant, "state", None)

    def _dispatcher(self) -> Any:
        parts = getattr(self.assistant, "parts", None)
        return getattr(parts, "dispatcher", None) or getattr(self.assistant, "dispatcher", None)

    def _phone_url(self) -> str:
        remote = getattr(self.assistant, "remote", None)
        try:
            return str(getattr(remote, "phone_url", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _tools(self) -> list[dict]:
        """What JARVIS can do, for the window's capability list."""
        dispatcher = self._dispatcher()
        try:
            payload = dispatcher.tools_payload() if dispatcher is not None else []
        except Exception:  # noqa: BLE001
            return []
        tools: list[dict] = []
        for spec in payload or []:
            function = spec.get("function", {}) if isinstance(spec, dict) else {}
            name = str(function.get("name", "") or "")
            if name:
                tools.append({"name": name, "description": str(function.get("description", "") or "")})
        return tools

    # --- telemetry ------------------------------------------------------------------------
    def status_poller(self) -> None:
        """Publish one telemetry reading every :attr:`status_interval` while anyone looks.

        Nobody watching means nobody polled: ``nvidia-smi`` is the expensive part of a
        snapshot, and paying for it every two seconds for a window that is not open
        would be a background tax on a machine that is meant to be idle.
        """
        try:
            from jarvis.core.telemetry import snapshot  # noqa: PLC0415 - lazy on purpose
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Telemetry is unavailable, so the window gets no status: %s", exc)
            return
        while self.windows > 0 and not self._closing.is_set():
            try:
                reading = snapshot(self.assistant)
            except Exception as exc:  # noqa: BLE001 - one bad reading is not fatal
                self.log.debug("A telemetry snapshot failed: %s", exc)
                reading = None
            if isinstance(reading, dict):
                try:
                    self.bus.publish("status", **reading)
                except Exception:  # noqa: BLE001
                    self.log.debug("Publishing telemetry failed.", exc_info=True)
            if self._poll_wake.wait(self.status_interval):
                self._poll_wake.clear()
        self.log.debug("The telemetry poller stopped; no window is watching.")

    def _window_joined(self) -> None:
        poller: threading.Thread | None = None
        with self._lock:
            self._windows += 1
            alive = self._poller is not None and self._poller.is_alive()
            if self._windows == 1 and not alive:
                self._poll_wake.clear()
                poller = threading.Thread(
                    target=self.status_poller, name="jarvis-desk-status", daemon=True
                )
                self._poller = poller
        if poller is not None:  # started outside the lock; it publishes as it goes
            poller.start()

    def _window_left(self) -> None:
        with self._lock:
            self._windows = max(0, self._windows - 1)
            empty = self._windows == 0
        if empty:
            self._poll_wake.set()


# --- module helpers ---------------------------------------------------------------------
def _send(ws: Any, payload: dict) -> bool:
    """Send one JSON frame. Returns False when the socket is gone, and never raises."""
    try:
        ws.send(json.dumps(payload, ensure_ascii=False, default=str))
        return True
    except Exception:  # noqa: BLE001 - a closed window is the normal case
        return False


def _close(ws: Any) -> None:
    try:
        ws.close()
    except Exception:  # noqa: BLE001
        pass


def _frame(event: Any) -> dict:
    """Flatten a :class:`~jarvis.desk.bus.DeskEvent` into the wire shape.

    The event knows its own shape, so ask it. The fallback is here because this
    server must keep serving a bus that is one version behind it.
    """
    as_json = getattr(event, "as_json", None)
    if callable(as_json):
        try:
            shaped = as_json()
        except Exception:  # noqa: BLE001
            shaped = None
        if isinstance(shaped, dict):
            return shaped
    payload = getattr(event, "payload", None)
    frame = dict(payload) if isinstance(payload, dict) else {}
    frame["type"] = str(getattr(event, "type", "event") or "event")
    frame["at"] = float(getattr(event, "at", 0.0) or 0.0)
    return frame


def _no_hook(name: str) -> dict:
    return {"type": "error", "message": f"This build of JARVIS has no {name} hook, sir."}


def _peer(value: str | None) -> str:
    """Fold an IPv4-mapped IPv6 peer onto its IPv4 spelling."""
    raw = (value or "").strip()
    return raw[7:] if raw.startswith("::ffff:") else raw


def _cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None or not hasattr(cfg, "get"):
        return default
    try:
        value = cfg.get(key, default)
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _routines(cfg: Any, log: logging.Logger) -> list[Routine]:
    """The window shows the same macro buttons the phone does."""
    try:
        return load_routines(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("The routines could not be read: %s", exc)
        return []


def _version() -> str:
    try:
        from jarvis import __version__  # noqa: PLC0415

        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""
