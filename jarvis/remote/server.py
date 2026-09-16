"""The phone's half of JARVIS: a small Flask app, two WebSockets, one HTML file.

Nothing here runs unless ``remote.enabled`` is true, and nothing here imports Flask
until :meth:`RemoteServer.start` is called — the module itself imports on a bare Linux
box, and an installer that never enables remote access never opens a socket.

The shape of one turn:

    phone    tap -> vad-web -> onSpeechEnd(Float32Array @ 16 kHz)
             -> Int16 PCM -> binary WebSocket frame
    server   decode_pcm16 -> RemoteTurn -> the assistant's existing work queue
    worker   Whisper -> Brain.turn(on_sentence=...) -> sentences back down the socket
    phone    speaks sentence one while the model is still writing sentence two

A second socket carries StateBus changes so the phone's ring shows the same states as
the desk's HUD.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from jarvis.core.logging import get_logger
from jarvis.remote.session import (
    ToolReporter,
    COOKIE_NAME,
    Device,
    RemoteGuard,
    RemoteTurn,
    Routine,
    SessionStore,
    decode_pcm16,
    load_or_create_secret,
    load_routines,
)

__all__ = ["RemoteServer", "DEFAULTS", "MISSING_HOOK"]

#: Defaults for every ``remote.*`` key. They live here rather than in ``config.py``
#: so this package is self-contained until the integrator mirrors them into
#: ``DEFAULTS`` and ``config.yaml``.
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "host": "127.0.0.1",
    "port": 8765,
    "url": "",
    "allow_guarded": False,
    "secret_file": "logs/remote-secret.key",
    "session_ttl_hours": 720,
    "code_ttl": 900,
    "max_attempts": 5,
    "attempt_window": 300,
    "max_audio_seconds": 30,
    "speak_locally": False,
    "turn_timeout": 120,
    "heartbeat": 20,
    "server": "auto",  # auto | werkzeug | waitress
    "routines": [],
}

#: Said to the phone when this build has no ``submit_remote_turn`` hook. It is a plain
#: refusal on purpose: pretending the turn was queued would be worse than saying no.
MISSING_HOOK = (
    "This build of JARVIS cannot take turns from the phone yet: the assistant has no "
    "submit_remote_turn hook."
)

#: Bind addresses that would expose PowerShell to the whole network.
_FORBIDDEN_HOSTS = {"", "0.0.0.0", "::", "[::]", "*"}

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _cfg(cfg: Any, key: str) -> Any:
    """Read ``remote.<key>``, falling back to this module's own defaults."""
    default = DEFAULTS.get(key)
    if cfg is None or not hasattr(cfg, "get"):
        return default
    value = cfg.get(f"remote.{key}", default)
    return default if value is None else value


class RemoteServer:
    """Serves the PWA and carries audio, sentences and state to and from the phone."""

    def __init__(self, cfg: Any, assistant: Any, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.log = logger or get_logger("remote")

        self.enabled = bool(_cfg(cfg, "enabled"))
        self.host = str(_cfg(cfg, "host")).strip()
        self.port = int(_cfg(cfg, "port"))
        self.allow_guarded = bool(_cfg(cfg, "allow_guarded"))
        self.max_audio_seconds = float(_cfg(cfg, "max_audio_seconds"))
        # False keeps a remote reply on the phone; True also says it out of the desk
        # speakers, which is what you want when the phone is in the next room.
        self.speak_locally = bool(_cfg(cfg, "speak_locally"))
        self.turn_timeout = float(_cfg(cfg, "turn_timeout"))
        self.heartbeat = max(5.0, float(_cfg(cfg, "heartbeat")))
        self.routines: list[Routine] = load_routines(cfg)

        self._secret_path = Path(str(_cfg(cfg, "secret_file")))
        if not self._secret_path.is_absolute():
            self._secret_path = _project_root(cfg) / self._secret_path
        self._sessions: SessionStore | None = None
        self._sessions_lock = threading.Lock()

        self._running = threading.Event()
        self._closing = threading.Event()
        self._httpd: Any = None
        self._thread: threading.Thread | None = None
        self._bound_port = self.port
        self._app: Any = None

    # --- lifecycle --------------------------------------------------------------------
    @property
    def sessions(self) -> SessionStore:
        """The pairing store, built on first use.

        Lazily, because constructing it writes the cookie-signing secret to disk and a
        JARVIS with ``remote.enabled: false`` should leave no trace of a feature it
        never opened.
        """
        with self._sessions_lock:
            if self._sessions is None:
                self._sessions = SessionStore(
                    load_or_create_secret(self._secret_path, self.log),
                    ttl_hours=float(_cfg(self.cfg, "session_ttl_hours")),
                    max_attempts=int(_cfg(self.cfg, "max_attempts")),
                    attempt_window=float(_cfg(self.cfg, "attempt_window")),
                    code_ttl=float(_cfg(self.cfg, "code_ttl")),
                    logger=self.log,
                )
            return self._sessions

    @property
    def url(self) -> str:
        """Where this server actually listens."""
        host = self.host or "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self._bound_port}"

    @property
    def phone_url(self) -> str:
        """What the phone opens: the HTTPS door Enable-Phone.bat set up, else :attr:`url`."""
        public = str(_cfg(self.cfg, "url") or "").strip()
        return public or self.url

    @property
    def pairing_code(self) -> str:
        """The code currently accepted by ``/api/pair``."""
        return self.sessions.pairing_code

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> bool:
        """Open the socket. Returns False — never raises — when it cannot.

        Refuses to start on a wildcard bind address: something that can run
        PowerShell does not get to listen on every interface because of a typo.
        """
        if self._running.is_set():
            return True
        if not self.enabled:
            self.log.info("Remote access is off (remote.enabled is false).")
            return False
        if self.host.lower() in _FORBIDDEN_HOSTS:
            self.log.error(
                "Refusing to start remote access: remote.host is %r. Bind to a specific "
                "address, such as this machine's Tailscale address.",
                self.host,
            )
            return False
        try:
            app = self.create_app()
        except ImportError as exc:
            self.log.warning(
                "Remote access needs Flask: %s. Install it with "
                "'pip install flask flask-sock waitress'.",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - a broken remote must not stop JARVIS
            self.log.exception("Remote access could not be prepared: %s", exc)
            return False

        try:
            self._httpd = self._make_server(app)
        except OSError as exc:
            self.log.error("Remote access could not bind %s:%s (%s).", self.host, self.port, exc)
            return False
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Remote access could not start: %s", exc)
            return False

        self._closing.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._serve, name="jarvis-remote", daemon=True)
        self._thread.start()
        self._announce()
        return True

    def stop(self) -> None:
        """Close the socket. Safe to call twice, and from any thread."""
        if not self._running.is_set():
            return
        self._running.clear()
        self._closing.set()
        httpd = self._httpd
        for method in ("shutdown", "server_close", "close"):
            action = getattr(httpd, method, None)
            if not callable(action):
                continue
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.debug("Remote server %s() failed: %s", method, exc)
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._httpd = None
        self.log.info("Remote access stopped.")

    def _serve(self) -> None:
        try:
            self._httpd.serve_forever()
        except Exception as exc:  # noqa: BLE001
            if self._running.is_set():
                self.log.exception("The remote server stopped unexpectedly: %s", exc)

    def _make_server(self, app: Any) -> Any:
        """Build the WSGI server.

        Werkzeug's threaded server is the default because it is the only one in this
        stack that hands the raw socket to ``flask-sock``; waitress serves the page
        perfectly well but cannot carry a WebSocket, so choosing it is logged loudly.
        """
        choice = str(_cfg(self.cfg, "server")).strip().lower()
        if choice == "waitress":
            from waitress import create_server  # noqa: PLC0415 - lazy on purpose

            self.log.warning(
                "remote.server is 'waitress': the page will load but the microphone and "
                "state sockets will not connect. Use 'werkzeug' for a working phone."
            )
            server = create_server(app, host=self.host, port=self.port, threads=8)
            self._bound_port = self.port
            server.serve_forever = server.run  # type: ignore[attr-defined]
            return server

        from werkzeug.serving import make_server  # noqa: PLC0415 - lazy on purpose

        server = make_server(self.host, self.port, app, threaded=True)
        self._bound_port = int(server.socket.getsockname()[1])
        return server

    def _announce(self) -> None:
        """Print the pairing code where the operator can see it — console only."""
        banner = (
            f"\n  JARVIS remote is up at {self.url}\n"
            f"  Open on the phone: {self.phone_url}\n"
            f"  Pairing code: {self.sessions.pairing_code}\n"
        )
        self.log.info("Remote access listening on %s.", self.url)
        try:
            print(banner, flush=True)
        except Exception:  # noqa: BLE001 - no console, no problem
            pass

    # --- the app ------------------------------------------------------------------------
    def create_app(self) -> Any:
        """Build the Flask app. Imports Flask here, so the module imports without it."""
        from flask import Flask  # noqa: PLC0415 - lazy on purpose
        from flask_sock import Sock  # noqa: PLC0415

        app = Flask(__name__, static_folder=None)
        app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
        sock = Sock(app)
        self._register_routes(app)
        sock.route("/ws/audio")(self.audio_socket)
        sock.route("/ws/state")(self.state_socket)
        self._app = app
        return app

    def _register_routes(self, app: Any) -> None:
        from flask import jsonify, request  # noqa: PLC0415

        @app.get("/")
        def index() -> Any:
            return self._static("index.html", "text/html")

        @app.get("/manifest.webmanifest")
        def manifest() -> Any:
            return self._static("manifest.webmanifest", "application/manifest+json")

        @app.get("/icon.svg")
        def icon() -> Any:
            return self._static("icon.svg", "image/svg+xml")

        # Add to Home Screen asks for these by name, from the page and the manifest.
        # Without them the app on the phone gets a blank square for an icon.
        @app.get("/apple-touch-icon.png")
        def apple_icon() -> Any:
            return self._static("apple-touch-icon.png", "image/png")

        @app.get("/icon-192.png")
        def icon_192() -> Any:
            return self._static("icon-192.png", "image/png")

        @app.get("/icon-512.png")
        def icon_512() -> Any:
            return self._static("icon-512.png", "image/png")

        @app.get("/api/session")
        def session_info() -> Any:
            device = self.device_for(request.cookies.get(COOKIE_NAME))
            return jsonify(
                {
                    "paired": device is not None,
                    "device": device.name if device else "",
                    "state": self.current_state(),
                    "allow_guarded": self.allow_guarded,
                    "routines": [routine.as_json() for routine in self.routines],
                }
            )

        @app.post("/api/pair")
        def pair() -> Any:
            body = request.get_json(silent=True) or {}
            result = self.sessions.pair(
                str(body.get("code", "")),
                str(body.get("device", "")),
                request.remote_addr or "unknown",
            )
            payload = {"ok": result.ok, "status": result.status, "message": result.message}
            if not result.ok:
                if result.retry_after:
                    payload["retry_after"] = round(result.retry_after)
                return jsonify(payload), 429 if result.status == "rate_limited" else 401
            payload["device"] = result.device.name if result.device else ""
            response = jsonify(payload)
            response.set_cookie(
                COOKIE_NAME,
                result.token,
                httponly=True,
                samesite="Strict",
                max_age=int(self.sessions.ttl_seconds),
                path="/",
            )
            return response

        @app.post("/api/unpair")
        def unpair() -> Any:
            response = jsonify({"ok": True})
            response.delete_cookie(COOKIE_NAME, path="/")
            return response

        @app.get("/api/status")
        def api_status() -> Any:
            device = self.device_for(request.cookies.get(COOKIE_NAME))
            if device is None:
                return jsonify({"error": "Pair this phone first."}), 401
            return jsonify(self.status_snapshot())

        @app.post("/api/routine")
        def routine() -> Any:
            device = self.device_for(request.cookies.get(COOKIE_NAME))
            if device is None:
                return jsonify({"ok": False, "error": "not paired"}), 401
            body = request.get_json(silent=True) or {}
            ok, said = self.run_routine(str(body.get("name", "")), device)
            return jsonify({"ok": ok, "said": said}), 200 if ok else 404

    def _static(self, name: str, mimetype: str) -> Any:
        """Serve one file from ``static/``. A missing file is a 500 that says so.

        Read as bytes, always: the home-screen icons are PNGs, and a text read would
        fail on them. Flask adds its own ``charset`` to a textual mimetype, so this
        passes the bare type and never one that already carries a charset.
        """
        from flask import Response  # noqa: PLC0415

        try:
            body = (_STATIC_DIR / name).read_bytes()
        except OSError as exc:
            self.log.error("The remote file %s is missing: %s", name, exc)
            return Response(f"{name} is missing.", status=500, mimetype="text/plain")
        return Response(body, mimetype=mimetype)

    # --- helpers the routes and sockets share ---------------------------------------------
    def device_for(self, token: str | None) -> Device | None:
        """The paired device a cookie belongs to, or ``None``."""
        return self.sessions.verify_token(token)

    def status_snapshot(self) -> dict:
        """The desk at a glance: state, load, temperature, what is scheduled.

        The reading itself belongs to :mod:`jarvis.core.telemetry`, which caches the
        slow parts and is shared with the desk window - the two faces must never
        disagree about the one machine they are both watching. Added here: the phone's
        own view of the state, and the keys its page expects to exist.
        """
        from jarvis.core.telemetry import snapshot as read_machine  # noqa: PLC0415

        try:
            snapshot: dict[str, Any] = read_machine(self.assistant)
        except Exception:  # noqa: BLE001 - a glance at the phone must never throw
            self.log.debug("Telemetry could not be read.", exc_info=True)
            snapshot = {}
        snapshot["state"] = self.current_state()
        snapshot.setdefault("timers", [])
        snapshot["ok"] = True
        return snapshot

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

    def guard_for(self, device: Device | str) -> Callable[[Any], Any]:
        """A factory the worker thread uses to wrap the brain's dispatcher for one turn."""
        name = device.name if isinstance(device, Device) else str(device)

        def wrap(inner: Any) -> Any:
            return RemoteGuard(
                inner, self.cfg, allow_guarded=self.allow_guarded, device=name, logger=self.log
            )

        return wrap

    def run_routine(self, name: str, device: Device) -> tuple[bool, str]:
        """Run one named macro through the dispatcher. Returns ``(ok, spoken_line)``."""
        wanted = " ".join(str(name or "").split()).lower()
        routine = next((r for r in self.routines if r.name.lower() == wanted), None)
        if routine is None:
            self.log.warning("Remote routine %r from %s is not configured.", name, device)
            return False, ""
        inner = self._dispatcher()
        if inner is None:
            self.log.error("Remote routine %r cannot run: there is no dispatcher.", routine.name)
            return False, "I can't reach my tools just now, sir."
        dispatcher = self.guard_for(device)(inner)
        self.log.info("Remote routine %r from %s.", routine.name, device)
        summaries: list[str] = []
        ok = True
        for tool_name, args in routine.calls:
            result = dispatcher.execute(tool_name, dict(args))
            summary = str(getattr(result, "summary", "") or "").strip()
            if summary:
                summaries.append(summary)
            if not getattr(result, "ok", False):
                ok = False
                break
        said = routine.say if (ok and routine.say) else " ".join(summaries)
        return ok, said.strip()

    # --- the sockets -----------------------------------------------------------------------
    def audio_socket(self, ws: Any) -> None:
        """Carry utterances up and sentences back down. One turn at a time, by design."""
        device = self._authorise(ws)
        if device is None:
            return
        try:
            ws._jarvis_device = device  # so a typed "say" frame knows who is asking
        except Exception:  # noqa: BLE001 - a socket object that refuses attributes
            pass
        max_samples = int(self.max_audio_seconds * 16000)
        while not self._closing.is_set():
            try:
                message = ws.receive(timeout=self.heartbeat)
            except Exception:  # noqa: BLE001 - the phone went away
                return
            if message is None:
                if not _send(ws, {"type": "ping"}):
                    return
                continue
            if isinstance(message, (bytes, bytearray)):
                try:
                    audio = decode_pcm16(message, max_samples=max_samples)
                except ValueError as exc:
                    self.log.warning("Bad audio frame from %s: %s", device, exc)
                    if not _send(ws, {"type": "error", "message": f"Audio rejected: {exc}"}):
                        return
                    continue
                if not self._run_turn(ws, device, audio):
                    return
                continue
            if not self._handle_control(ws, message):
                return

    def _handle_control(self, ws: Any, message: Any) -> bool:
        """Answer a JSON control frame. Returns False when the socket should close."""
        try:
            data = json.loads(message)
        except (ValueError, TypeError):
            return _send(ws, {"type": "error", "message": "Expected JSON."})
        kind = str((data or {}).get("type", "")) if isinstance(data, dict) else ""
        if kind == "ping":
            return _send(ws, {"type": "pong"})
        if kind == "hello":
            return _send(ws, {"type": "ready", "state": self.current_state()})
        if kind == "say":
            text = str((data or {}).get("text", "") or "").strip()[:2000]
            if not text:
                return _send(ws, {"type": "error", "message": "Nothing to say."})
            device = getattr(ws, "_jarvis_device", None) or Device(name="phone", id="")
            return self._run_turn(ws, device, None, text=text)
        return _send(ws, {"type": "error", "message": f"Unknown message {kind or 'without a type'}."})

    def _run_turn(self, ws: Any, device: Device, audio: Any, *, text: str = "") -> bool:
        """Queue one turn and stream its sentences back. Returns False on a dead socket.

        Every send happens on this thread: the worker pushes into a queue and this
        loop drains it, so two threads never write to the same socket at once.
        """
        outbox: "queue.Queue[dict]" = queue.Queue()
        guard = self.guard_for(device)

        def wrap(inner: Any) -> Any:
            # The reporter sits outside the guard, so a refusal is reported too.
            return ToolReporter(guard(inner), outbox.put)

        turn = RemoteTurn(
            audio=audio,
            sample_rate=16000,
            device=device.name,
            text=text,
            on_sentence=lambda sentence: outbox.put({"type": "sentence", "text": sentence}),
            on_transcript=lambda heard: outbox.put({"type": "transcript", "text": heard}),
            on_done=lambda reply, error: outbox.put(
                {"type": "done", "reply": reply, "error": error}
            ),
            on_tool=outbox.put,
            wrap_dispatcher=wrap,
            speak_locally=self.speak_locally,
        )
        if text:
            self.log.info("Remote turn from %s (typed): %r", device, text[:80])
        else:
            self.log.info("Remote turn from %s: %.1f s of audio.", device, turn.duration_s)
        submit = getattr(self.assistant, "submit_remote_turn", None)
        if not callable(submit):
            self.log.error(MISSING_HOOK)
            return _send(ws, {"type": "error", "message": MISSING_HOOK})
        try:
            accepted = submit(turn)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Queueing the remote turn failed: %s", exc)
            return _send(ws, {"type": "error", "message": "I couldn't start that turn, sir."})
        if accepted is False:
            return _send(ws, {"type": "busy", "message": "I'm in the middle of something, sir."})

        deadline = time.monotonic() + self.turn_timeout
        while not self._closing.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.log.warning("Remote turn from %s timed out.", device)
                return _send(ws, {"type": "error", "message": "That took too long, sir."})
            try:
                event = outbox.get(timeout=min(remaining, self.heartbeat))
            except queue.Empty:
                if not _send(ws, {"type": "ping"}):
                    return False
                continue
            if not _send(ws, event):
                return False
            if event.get("type") == "done":
                return True
        return False  # the server is shutting down under the turn

    def state_socket(self, ws: Any) -> None:
        """Push every StateBus change to the phone so its ring matches the desk's."""
        device = self._authorise(ws)
        if device is None:
            return
        bus = self._state_bus()
        if bus is None:
            _send(ws, {"type": "error", "message": "No state to report."})
            return
        changes: "queue.Queue[str]" = queue.Queue()
        unsubscribe = bus.subscribe(lambda state: changes.put(str(state)))
        try:
            if not _send(ws, {"type": "state", "state": self.current_state()}):
                return
            while not self._closing.is_set():
                try:
                    state = changes.get(timeout=self.heartbeat)
                except queue.Empty:
                    if not _send(ws, {"type": "ping"}):
                        return
                    continue
                if not _send(ws, {"type": "state", "state": state}):
                    return
        finally:
            try:
                unsubscribe()
            except Exception:  # noqa: BLE001
                self.log.debug("Unsubscribing the remote state socket failed.", exc_info=True)

    def _authorise(self, ws: Any) -> Device | None:
        """Check the cookie on a WebSocket upgrade; close the socket when it is not ours."""
        try:
            from flask import request  # noqa: PLC0415

            token = request.cookies.get(COOKIE_NAME)
            client = request.remote_addr or "unknown"
        except Exception:  # noqa: BLE001 - no request context means no cookie
            token, client = None, "unknown"
        device = self.device_for(token)
        if device is None:
            self.log.warning("Unpaired device from %s tried to open a socket.", client)
            _send(ws, {"type": "unpaired", "message": "Pair this phone first."})
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
            return None
        return device


def _send(ws: Any, payload: dict) -> bool:
    """Send one JSON frame. Returns False when the socket is gone, and never raises."""
    try:
        ws.send(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:  # noqa: BLE001 - a phone in a pocket disconnects constantly
        return False


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
