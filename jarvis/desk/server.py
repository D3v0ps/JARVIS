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
  worthless a moment later, and worthless anyway after two minutes. Opening the
  window again mints a new one rather than reviving the old;
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
    "MAX_FIELD",
    "MAX_SAY",
    "REFUSAL_INTERVAL",
    "SETTINGS",
    "STATUS_INTERVAL",
    "TICKET_TTL",
    "WINDOW_ACTIONS",
    "WINDOW_BOUNDS",
]

#: The session cookie. Deliberately not the phone's name: the two doors are separate.
COOKIE_NAME = "jarvis_desk"

#: How long the ticket in the URL stays good. It only has to survive one command line.
TICKET_TTL = 120.0

#: A typed turn is a sentence, not a novel. Anything longer is a paste accident.
MAX_SAY = 2000

#: Seconds between telemetry readings while at least one window is watching.
STATUS_INTERVAL = 2.0

#: How often a burst of refusals may put a single line above DEBUG in the log.
REFUSAL_INTERVAL = 60.0

#: The longest a field off an inbound frame may be before it reaches a log line or
#: a reply. Nothing the page legitimately sends in one is longer than a config key.
MAX_FIELD = 64

#: The only things the page's own title bar may do to the native window. Anything
#: else - navigate, evaluate, move to another screen - is not a title bar's business.
WINDOW_ACTIONS = ("minimize", "maximize", "restore", "close", "resize")

#: ``(min width, min height, max width, max height)`` for a size the page asks for.
#: A frameless window cannot be resized by the operating system, so the page grows a
#: grip of its own and sends what it dragged; these are the bounds of belief.
WINDOW_BOUNDS = (320, 240, 7680, 4320)

#: How long a writer waits on a silent bus before sending a ping.
HEARTBEAT = 20.0

#: Frames queued for a window that has stopped reading. Oldest goes over the side.
_OUTBOX = 256

#: Bind addresses this server will accept. Not ``localhost``: a name can be pointed
#: somewhere else, and the whole point of this module is that the address cannot be.
_LOOPBACK_BINDS = {"127.0.0.1", "::1", "[::1]"}

#: Peers we will talk to, after ``::ffff:127.0.0.1`` has been folded onto IPv4.
_LOOPBACK_PEERS = {"127.0.0.1", "::1"}

#: A ticket as it appears in a URL. Used to take one back out of anything on its
#: way to the log: the secret is only a secret while it is nowhere on disk.
_TICKET_IN_TEXT = re.compile(r"([?&]t=)[A-Za-z0-9_-]+")

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


class _RefusalLog:
    """Counts refusals that cost the caller nothing, and lets one line a minute out.

    A page on the open web cannot get through this door — it cannot forge ``Host``,
    it cannot forge ``Origin`` and it has no cookie — but it can knock, and a loop of
    ``fetch('http://127.0.0.1:<port>/', {mode:'no-cors'})`` knocks several hundred
    times a second. One record apiece at ERROR is a quarter of a megabyte a second
    into ``logs/jarvis.log``: the whole shipped rotation budget, and with it the
    transcript and the tool-call audit trail CLAUDE.md requires, recycled in under
    two minutes. Worse, those records go to the window's bus as well, and a page that
    can fill a 256-slot queue can push a pending ``confirm`` off the back of it and
    hide the confirmation bar while a guarded tool is waiting.

    So the refusal itself — which is unremarkable, and which the attacker chose —
    goes to DEBUG, and what is genuinely worth knowing, that thousands arrived, comes
    out at WARNING at most once every :data:`REFUSAL_INTERVAL` seconds.
    """

    def __init__(self, log: logging.Logger, interval: float = REFUSAL_INTERVAL) -> None:
        self._log = log
        self._interval = float(interval)
        self._lock = threading.Lock()
        self._count = 0
        self._since = 0.0
        self._next = 0.0
        self._last = ""

    def note(self, what: str, reason: str) -> None:
        """Record one refusal. Always at DEBUG; sometimes, briefly, at WARNING."""
        self._log.debug("Refusing %s: %s.", what, reason)
        now = time.monotonic()
        with self._lock:
            self._count += 1
            self._last = f"{what}: {reason}"
            if now < self._next:
                return
            count, since, last = self._count, self._since, self._last
            self._count = 0
            self._since = now
            self._next = now + self._interval
        if count == 1:
            self._log.warning("Refused a desk request — %s.", last)
            return
        self._log.warning(
            "Refused %d desk requests in the last %.0f seconds, most recently %s.",
            count,
            max(0.0, now - since),
            last,
        )


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

        # A hostile page cannot get in, but it can knock as fast as it likes, and the
        # log is a finite resource. See :class:`_RefusalLog`.
        self._refusals = _RefusalLog(self.log)

        self._lock = threading.Lock()
        self._bus_lock = threading.Lock()
        #: The native host, once there is one. The page's own title bar has to drive
        #: it through here: a frameless window has no buttons of the system's own.
        self._window: Any = None
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
            "window": self._on_window,
        }

    def attach_window(self, window: Any) -> None:
        """Hand the server the native host, so the page's title bar can drive it.

        Attached rather than built here, and allowed to be ``None``, because the
        window may not exist at all: ``--no-window``, a browser fallback, a machine
        without WebView2. The frames still arrive; they simply have nothing to do.
        """
        with self._lock:
            self._window = window

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
        """The one URL that opens a window, ticket and all. Do not log this.

        Reading it mints a fresh ticket when the last one has been spent or has timed
        out. The alternative is what happens without it: the operator closes the
        window to the tray, asks for it again an hour later, and is told by his own
        assistant that his own link has already been used. A ticket is still good for
        exactly one use — this hands out a new one rather than reviving the old.
        """
        if self._ticket_stale():
            self.issue_ticket()
        with self._lock:
            ticket = self._ticket
        return f"http://{self.display_host}:{self._bound_port}/?t={ticket}"

    def issue_ticket(self) -> str:
        """Mint a new ticket, clear the spent flag and restart the two-minute clock.

        The tray calls this — through :attr:`url` — every time it opens the window
        again. It is not a way back into an old session: the ticket is new, so a copy
        of the previous one left in a command line or a process list stays worthless.
        """
        with self._lock:
            self._ticket = secrets.token_urlsafe(32)
            self._ticket_expires = time.monotonic() + TICKET_TTL
            self._ticket_spent = False
            ticket = self._ticket
        self.log.debug("A fresh desk ticket was issued; it is good for %.0f s.", TICKET_TTL)
        return ticket

    def _ticket_stale(self) -> bool:
        """True when the ticket in hand would be refused if it were offered now."""
        with self._lock:
            return self._ticket_spent or time.monotonic() > self._ticket_expires

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
        server = make_server(bind, 0, app, threaded=True, request_handler=self._handler_class())
        self._bound_port = int(server.socket.getsockname()[1])
        return server

    def _handler_class(self) -> Any:
        """A request handler that keeps the ticket off the console and out of a flood.

        Werkzeug's default prints every request line to ``stderr``, query string and
        all, which on a console run puts the one secret this module has in front of
        anyone reading over the operator's shoulder — and hands a page that cannot
        get in a second way to fill the screen, one our own rate limit does not cover.
        Its occasional real complaints are still worth keeping, so they come here at
        DEBUG, with any ticket taken out of them first.
        """
        from werkzeug.serving import WSGIRequestHandler  # noqa: PLC0415 - lazy on purpose

        logger = self.log

        class QuietHandler(WSGIRequestHandler):
            """Werkzeug's handler with its access log turned off."""

            def log_request(self, code: Any = "-", size: Any = "-") -> None:
                return None

            def log(self, type: str, message: str, *args: Any) -> None:  # noqa: A002
                try:
                    text = message % args if args else str(message)
                except Exception:  # noqa: BLE001 - a log line is never worth a request
                    text = str(message)
                logger.debug("The desk's HTTP server: %s", _redacted(text))

        return QuietHandler

    # --- the app --------------------------------------------------------------------
    def create_app(self) -> Any:
        """Build the Flask app. Imports Flask here, so the module imports without it."""
        from flask import Flask  # noqa: PLC0415 - lazy on purpose
        from flask_sock import Sock  # noqa: PLC0415

        app = Flask(__name__, static_folder=None)
        # A frame from this page is a sentence and a couple of numbers. 64 KiB is
        # already generous, and it is the only thing between a window that has gone
        # wrong and a message the size of memory.
        app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25, "max_message_size": 65536}
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
            self._refusals.note("a desk request", reason)
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
                self._refusals.note(
                    "a desk asset request", f"there is no session cookie (it asked for "
                    f"{str(name or '')[:MAX_FIELD]!r})"
                )
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
            self._refusals.note("a desk socket", reason)
            _send(ws, {"type": "error", "message": "Not for you."})
            _close(ws)
            return False
        try:
            from flask import request  # noqa: PLC0415

            cookie = request.cookies.get(COOKIE_NAME)
        except Exception:  # noqa: BLE001
            cookie = None
        if not self.cookie_is_ours(cookie):
            self._refusals.note("a desk socket", "it carried no session cookie")
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
        # Clamped before it can reach a log line or a reply: the field is under the
        # page's control, and neither the log file nor the window's feed is.
        kind = str(frame.get("type", "") or "").strip()[:MAX_FIELD]
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
        if not callable(getattr(self.assistant, "run_routine", None)):
            self.log.error("This build of JARVIS has no run_routine hook; the window asked for it.")
            return _no_hook("run_routine")
        self.log.info("Desk routine %r.", name)
        # On its own thread, like quit: a macro may contain a GUARDED tool, and the
        # confirmation it raises can only be answered by a frame arriving on this
        # reader thread. Running it here would mean the page shows a Confirm button
        # whose click nothing is left to read — the two would wait for each other
        # until the confirmation timed out. So the page is answered at once and what
        # the routine said reaches it the way every other spoken line does.
        threading.Thread(
            target=self._run_routine, args=(name,), name="jarvis-desk-routine", daemon=True
        ).start()
        return {"type": "ack", "for": "routine", "name": name}

    def _run_routine(self, name: str) -> None:
        """Run one macro off the socket's threads and put its line on the bus."""
        found, said = self._ask("run_routine", name)
        if not found:
            return
        line = str(said or "").strip()
        if line:
            self._publish_line(line)

    def _publish_line(self, said: str) -> None:
        """Publish what the routine said, unless the assistant has already said it.

        A macro speaks through the assistant, and the assistant narrates itself onto
        this same bus, so the line is usually already on its way to the window;
        publishing it again would print it twice. Publishing it never would leave a
        routine silent in a build where that narration is not wired up. So: look at
        what the bus has just seen, and only fill the gap.
        """
        try:
            recent = list(self.bus.replay())[-8:]
        except Exception:  # noqa: BLE001 - a bus that cannot be read is not our problem
            recent = []
        for event in recent:
            payload = getattr(event, "payload", None)
            if str(getattr(event, "type", "")) != "sentence" or not isinstance(payload, dict):
                continue
            if str(payload.get("text", "")).strip() == said:
                return
        try:
            self.bus.publish("sentence", text=said)
        except Exception:  # noqa: BLE001 - the window is never worth a routine
            self.log.debug("Publishing the routine's line failed.", exc_info=True)

    def _on_set(self, frame: dict) -> dict:
        key = str(frame.get("key", "") or "").strip()[:MAX_FIELD]
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

    def _on_window(self, frame: dict) -> dict:
        """The page's own title bar: minimise, maximise, restore, close, resize.

        The window is frameless, so the buttons at its top right are ours and this is
        the only way they can do anything at all. ``close`` means hide: the tray's
        ``Open JARVIS`` reverses it, and a title bar that could actually end the
        session would put shutting JARVIS down one mis-click away.
        """
        action = str(frame.get("action", "") or "").strip().lower()[:MAX_FIELD]
        if action not in WINDOW_ACTIONS:
            self.log.warning(
                "The desk window asked for %r, which is not one of my window actions.", action
            )
            return {"type": "error", "message": "I cannot do that to the window, sir."}
        size: tuple[int, int] | None = None
        if action == "resize":
            try:
                size = _window_size(frame.get("width"), frame.get("height"))
            except (TypeError, ValueError) as exc:
                self.log.warning("The desk window asked for an unusable size: %s", exc)
                return {"type": "error", "message": "That is not a usable window size, sir."}
        self._drive_window(action, size)
        return {"type": "ack", "for": "window", "action": action}

    def _drive_window(self, action: str, size: tuple[int, int] | None) -> None:
        """Do it to the native host, or write down that there was nothing to do it to.

        Never raises, whatever the host turns out to be: the page is entitled to press
        its own buttons in a browser tab, where there is no native window behind them,
        and a dead title bar is a better outcome than a machine error per click.
        """
        with self._lock:
            window = self._window
        if window is None:
            self.log.debug("The window was asked to %s, but none is attached.", action)
            return
        # An explicit table, for the same reason the frame table is one: the name of
        # a method never comes off a web page, not even one we have just checked.
        name = {
            "close": "hide",
            "minimize": "minimize",
            "maximize": "maximize",
            "restore": "restore",
            "resize": "resize",
        }[action]
        method = getattr(window, name, None)
        if not callable(method):
            self.log.debug("This window host cannot %s.", action)
            return
        try:
            if size is not None:
                method(*size)
            else:
                method()
        except Exception as exc:  # noqa: BLE001 - a title bar is not worth a traceback
            self.log.warning("The window would not %s: %s", action, exc)

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
            "settings": self._settings(),
        }

    def _settings(self) -> dict:
        """What the page's switches and sliders are actually set to, right now.

        Without this the window guesses: a switch drawn off first opened showed the
        opposite of the truth until the operator touched it, and touching it wrote
        the guess to ``config.yaml``. The same allowlist as ``set``, read in the same
        direction, so the window learns about exactly the keys it may change and not
        one line of the rest of the configuration.
        """
        return {key: _cfg(self.cfg, key, None) for key in SETTINGS}

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


def _window_size(width: Any, height: Any) -> tuple[int, int]:
    """Read a window size off a frame: two whole numbers, inside :data:`WINDOW_BOUNDS`.

    Clamped rather than refused for the same reason the sliders are: a grip dragged
    off the edge of a 4K screen means "as big as you go", not an attack. Nonsense —
    a string that is not a number, a list, ``NaN``, ``True`` — is still refused,
    because it means the frame did not come from our page.
    """
    low_w, low_h, high_w, high_h = WINDOW_BOUNDS
    return _whole(width, low_w, high_w), _whole(height, low_h, high_h)


def _whole(value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return int(min(high, max(low, round(number))))


def _redacted(text: str) -> str:
    """Take any ticket out of a line before anything writes it down."""
    return _TICKET_IN_TEXT.sub(r"\1...", str(text))


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
