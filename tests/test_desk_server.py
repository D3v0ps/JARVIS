"""The desk's door, tested the way an attacker would push on it.

The desk window has the rights of the keyboard, so almost everything worth testing
here is about who gets in: the loopback bind, the single-use ticket, the cookie, and
the three headers that stop a page on the open web from spending our port. The rest
checks that the window can never reach past the nine frames it is allowed to send,
and that a window which stops reading costs the assistant nothing.

No network beyond loopback, no Windows, no GPU, no Ollama. The assistant is a fake
with the six desk hooks on it, the bus is a small stand-in with the contract's shape,
and the WebSockets are objects with ``send``/``receive``/``close``.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.core.state import AssistantState, StateBus
from jarvis.desk.server import (
    COOKIE_NAME,
    MAX_FIELD,
    MAX_SAY,
    SETTINGS,
    TICKET_TTL,
    WINDOW_ACTIONS,
    WINDOW_BOUNDS,
    DeskServer,
    _RefusalLog,
    _Window,
)

SOURCE = (Path(__file__).resolve().parent.parent / "jarvis" / "desk" / "server.py").read_text(
    encoding="utf-8"
)

#: A scripted receive() that means "nothing arrived before the timeout".
TIMEOUT = object()

#: Every frame the page is allowed to send: the nine of contract 24.3 and the tenth
#: the title bar needs, because a frameless window has no buttons of the system's own.
FRAME_TYPES = {
    "say", "listen", "stop", "confirm", "pause", "resume", "routine", "set", "quit", "window",
}


@pytest.fixture(scope="module", autouse=True)
def leave_sys_modules_as_we_found_them():
    """Put the web stack back in its box when this module is done.

    The suite shares one interpreter, and other modules assert that importing JARVIS
    pulls in no Flask at all — which is true of the product and must stay testable.
    Building a real app here would otherwise fail a test that has nothing to do with
    the desk, purely because of the order pytest walks the files in.
    """
    before = set(sys.modules)
    yield
    web = {
        "blinker", "click", "flask", "flask_sock", "itsdangerous", "jinja2",
        "markupsafe", "simple_websocket", "werkzeug", "wsproto",
    }
    for name in sorted(set(sys.modules) - before):
        if name.split(".")[0] in web:
            sys.modules.pop(name, None)


# ----------------------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------------------
class Recorder(logging.Handler):
    """Collects log records so a test can assert on what JARVIS wrote down."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() % () if False else record.getMessage()
                         for record in self.records)

    def at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


class StandInBus:
    """The smallest object with :class:`~jarvis.desk.bus.DeskBus`'s shape.

    The real bus belongs to another module; the server's behaviour is what is under
    test here, so it gets a stand-in that cannot fail for reasons of its own.
    """

    def __init__(self, history: int = 200) -> None:
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=history)
        self._subscribers: list[queue.Queue] = []
        self.closed = False

    def publish(self, type: str, /, **payload):  # noqa: A002 - the contract's spelling
        event = SimpleNamespace(type=type, payload=dict(payload), at=time.time())
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass
        return event

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def replay(self) -> list:
        with self._lock:
            return list(self._history)

    def close(self) -> None:
        self.closed = True


class EchoingBus(StandInBus):
    """A bus that also hands a new window the history it is about to replay.

    The real one does not, but a window must not show the conversation twice merely
    because a bus was generous, so the server is tested against the worse of the two.
    """

    def subscribe(self) -> queue.Queue:
        q = super().subscribe()
        for event in self.replay():
            q.put_nowait(event)
        return q


class FakeSocket:
    """A WebSocket with a script of incoming messages and a record of outgoing ones.

    A callable in the script is run instead of being delivered — that is how a test
    publishes an event while the socket is open, without a sleep and a prayer.
    """

    def __init__(self, incoming: list | None = None, *, linger: float = 0.0) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[str] = []
        self.closed = False
        self.linger = linger
        self.lock = threading.Lock()

    def receive(self, timeout: float | None = None):
        if self.incoming:
            item = self.incoming.pop(0)
            if callable(item):
                item()
                return None
            return None if item is TIMEOUT else item
        if self.linger:
            time.sleep(self.linger)
            self.linger = 0.0
        raise ConnectionError("the window was closed")

    def send(self, data) -> None:
        if self.closed:
            raise ConnectionError("socket is closed")
        with self.lock:
            self.sent.append(data)

    def close(self) -> None:
        self.closed = True

    @property
    def events(self) -> list[dict]:
        with self.lock:
            return [json.loads(item) for item in self.sent]

    def kinds(self, skip_pings: bool = True) -> list[str]:
        return [e["type"] for e in self.events if not (skip_pings and e["type"] == "ping")]


class DeadSocket(FakeSocket):
    """A window that accepted the upgrade and then stopped listening."""

    def send(self, data) -> None:
        raise ConnectionError("nobody is reading")


class FakeWindow:
    """The native host, recording what the page's title bar asked of it.

    The window itself belongs to :mod:`jarvis.desk.window`; what matters here is
    which of its methods the server reaches for, that it reaches for nothing else,
    and that a host which says no costs the page nothing.
    """

    def __init__(self, *, explode: bool = False) -> None:
        self.calls: list[tuple] = []
        self.explode = explode

    def _did(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if self.explode:
            raise RuntimeError("the window manager said no")

    def hide(self) -> None:
        self._did("hide")

    def minimize(self) -> None:
        self._did("minimize")

    def maximize(self) -> None:
        self._did("maximize")

    def restore(self) -> None:
        self._did("restore")

    def resize(self, width: int, height: int) -> None:
        self._did("resize", width, height)


class FakeDispatcher:
    """The assistant's own dispatcher. The desk must run turns under this one."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict):
        self.executed.append((name, dict(args or {})))
        return SimpleNamespace(ok=True, summary=f"Ran {name}.", refused=False, data=None)

    def tools_payload(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": "get_time_date", "description": "The time."}},
            {"type": "function", "function": {"name": "run_powershell", "description": "A shell."}},
        ]


class FakeAssistant:
    """Stands in for jarvis.core.assistant.Assistant, all six desk hooks present."""

    def __init__(self, *, accept: bool = True) -> None:
        self.parts = SimpleNamespace(state=StateBus(), dispatcher=FakeDispatcher())
        self.remote = SimpleNamespace(phone_url="https://jarvis.example.ts.net")
        self.accept = accept
        self.turns: list[str] = []
        self.dispatchers: list[object] = []
        self.armed = 0
        self.aborted = 0
        self.confirmations: list[bool] = []
        self.routines: list[str] = []
        self.settings: list[tuple[str, object]] = []
        self.paused = False
        self.stopped = threading.Event()

    # --- the six from contract 24.3 ---
    def submit_desk_turn(self, text: str) -> bool:
        self.turns.append(text)
        # Whatever the desk runs, it runs under the assistant's own dispatcher.
        self.dispatchers.append(self.parts.dispatcher)
        return self.accept

    def arm_listening(self) -> bool:
        self.armed += 1
        return True

    def abort_turn(self) -> None:
        self.aborted += 1

    def answer_confirmation(self, granted: bool) -> bool:
        self.confirmations.append(granted)
        return True

    def run_routine(self, name: str) -> str:
        self.routines.append(name)
        return "Goodnight, sir."

    def apply_setting(self, key: str, value) -> bool:
        self.settings.append((key, value))
        return True

    # --- the ones the tray already has ---
    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stopped.set()


# ----------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------
class WaitingAssistant(FakeAssistant):
    """An assistant whose routine contains a GUARDED tool, so it waits to be told yes.

    This is the ordinary case for a macro that turns the lights off by way of
    PowerShell, and it is the case that deadlocked: the routine ran on the socket's
    reader thread, so the Confirm button on the bar it raised had nothing left to
    read it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.asked = threading.Event()
        self.answered = threading.Event()
        #: Was the confirmation delivered while the macro was still waiting for it?
        #: That is the whole question: an answer that only lands after the routine
        #: has given up is the deadlock, not the fix.
        self.confirmed = False

    def run_routine(self, name: str) -> str:
        self.routines.append(name)
        self.asked.set()
        self.confirmed = self.answered.wait(timeout=2)
        return "Goodnight, sir." if self.confirmed else ""

    def answer_confirmation(self, granted: bool) -> bool:
        self.confirmations.append(granted)
        self.answered.set()
        return True


@pytest.fixture
def desk_config(config):
    config.set(
        "remote",
        {
            "enabled": False,
            "routines": [
                {
                    "name": "Good night",
                    "say": "Goodnight, sir.",
                    "calls": [{"tool": "get_time_date", "args": {}}],
                }
            ],
        },
    )
    return config


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def assistant():
    return FakeAssistant()


@pytest.fixture
def bus():
    return StandInBus()


@pytest.fixture
def server(desk_config, assistant, bus, recorder):
    log = logging.getLogger(f"desk-test-{id(recorder)}")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.addHandler(recorder)
    made = DeskServer(desk_config, assistant, log, bus=bus)
    made.heartbeat = 0.05
    made.status_interval = 5.0  # the telemetry tests shorten this; the rest want quiet
    made._bound_port = 8123  # what the OS would have handed back from a real bind
    yield made
    made.stop()
    log.removeHandler(recorder)


@pytest.fixture
def app(server):
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    return server.create_app()


@pytest.fixture
def client(server, app):
    """A test client that looks like a browser on this machine."""
    return app.test_client()


def get(client, server, path: str = "/", **headers):
    """One GET with loopback credentials, unless a test forges one of them."""
    sent = {"Host": server.host_header}
    sent.update({key.replace("_", "-"): value for key, value in headers.items()})
    sent = {key: value for key, value in sent.items() if value is not None}
    return client.get(path, headers=sent, environ_base={"REMOTE_ADDR": "127.0.0.1"})


def socket_context(server, app, *, cookie: str | None = None, host: str | None = None,
                   origin: str | None = None, peer: str = "127.0.0.1"):
    """A request context for the WebSocket upgrade, forgeable one header at a time."""
    headers = {"Host": server.host_header if host is None else host}
    if cookie:
        headers["Cookie"] = f"{COOKIE_NAME}={cookie}"
    if origin:
        headers["Origin"] = origin
    return app.test_request_context(
        "/ws/desk", headers=headers, environ_base={"REMOTE_ADDR": peer}
    )


def admitted(server) -> str:
    """Walk the real ticket exchange and hand back the cookie it issued."""
    return server.redeem(server._ticket)


def run_socket(server, app, ws, *, cookie: str | None = None, **kwargs) -> FakeSocket:
    with socket_context(server, app, cookie=cookie, **kwargs):
        server.desk_socket(ws)
    return ws


def path_of(url: str) -> str:
    """The ``/?t=...`` half of a URL, which is what a test client asks for."""
    return "/" + url.split("/", 3)[3]


def waited_for(predicate, timeout: float = 5.0) -> bool:
    """True as soon as ``predicate`` holds; a background thread is not instant."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------------------
# The bind
# ----------------------------------------------------------------------------------
def test_the_server_binds_an_ephemeral_loopback_port_and_reads_the_real_one_back(server):
    """Port 0 means the OS chooses; the URL is useless unless we read back its answer."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    server._bound_port = 0

    assert server.start() is True
    try:
        assert server.port > 0
        assert server.running is True
        assert server.url.startswith(f"http://127.0.0.1:{server.port}/?t=")
        assert server.host_header == f"127.0.0.1:{server.port}"
    finally:
        server.stop()
    assert server.running is False


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "", "::", "[::]", "*", "localhost", "192.168.1.10", "0.0.0.0 "]
)
def test_a_host_that_is_not_a_loopback_literal_is_refused_before_any_socket_opens(
    server, recorder, host
):
    """A typo in config.yaml does not get to hand the keyboard's rights to the LAN.

    ``localhost`` is refused with the rest: a name can be pointed elsewhere, and the
    whole defence here rests on the address being one that cannot be.
    """
    server.host = host.strip() or host

    assert server.start() is False
    assert server.running is False
    assert any("not a loopback address" in line for line in recorder.at(logging.ERROR))


# ----------------------------------------------------------------------------------
# The ticket and the cookie
# ----------------------------------------------------------------------------------
def test_the_ticket_is_thirty_two_urlsafe_bytes_generated_per_run(desk_config, assistant, bus):
    """Two runs must never share a ticket, or a stale link would still open a window."""
    first = DeskServer(desk_config, assistant, bus=bus)
    second = DeskServer(desk_config, assistant, bus=bus)

    assert first._ticket != second._ticket
    assert len(first._ticket) >= 40  # token_urlsafe(32) is 43 characters
    assert "?t=" in first.url


def test_a_valid_ticket_sets_an_httponly_samesite_strict_session_cookie(server, client):
    """The ticket is visible in a process list; the cookie that replaces it is not."""
    response = get(client, server, f"/?t={server._ticket}")

    assert response.status_code == 200
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Secure" not in cookie  # this is plain http://127.0.0.1; Secure would break it
    assert COOKIE_NAME in cookie


def test_the_ticket_is_burned_so_a_second_window_cannot_use_the_same_link(server, app):
    """The link survives one command line. Anyone who read it afterwards gets nothing."""
    first = get(app.test_client(), server, f"/?t={server._ticket}")
    # A second browser, with no cookie of its own, offering the same link.
    second = get(app.test_client(), server, f"/?t={server._ticket}")

    assert first.status_code == 200
    assert second.status_code == 401
    assert server._ticket_spent is True


def test_a_ticket_older_than_two_minutes_is_refused(server, client):
    """It only has to survive Edge's start-up, so it stops being a key after 120 s."""
    server._ticket_expires = time.monotonic() - 1.0

    response = get(client, server, f"/?t={server._ticket}")

    assert response.status_code == 401
    assert TICKET_TTL == 120.0


def test_a_ticket_that_is_not_ours_is_refused(server, client):
    response = get(client, server, "/?t=" + "x" * 43)

    assert response.status_code == 401
    assert server._ticket_spent is False  # a wrong guess must not burn the real one


def test_a_window_that_already_holds_the_cookie_gets_in_without_a_ticket(server, client):
    """Reloading the page, or reopening it from the tray, must not need a new link."""
    get(client, server, f"/?t={server._ticket}")

    response = get(client, server, "/")

    assert response.status_code == 200


def test_a_request_with_neither_ticket_nor_cookie_gets_a_short_page_not_a_traceback(
    server, client
):
    """Whoever lands here is a person, not an exploit; tell them where the door is."""
    response = get(client, server, "/")
    body = response.get_data(as_text=True)

    assert response.status_code == 401
    assert "Traceback" not in body
    assert "tray" in body.lower()
    assert len(body) < 1500


def test_the_ticket_is_never_written_to_the_log(server, recorder):
    """It is only a secret while it is nowhere on disk, and the log is on disk."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    server._bound_port = 0

    assert server.start() is True
    server.stop()

    written = "\n".join(record.getMessage() for record in recorder.records)
    assert server._ticket not in written
    assert f"127.0.0.1:{server.port}" in written


# ----------------------------------------------------------------------------------
# The DNS-rebinding defence
# ----------------------------------------------------------------------------------
def test_a_request_from_a_peer_that_is_not_loopback_is_refused(server, client, recorder):
    """Nothing off this machine may reach a door that runs PowerShell without asking."""
    response = client.get(
        f"/?t={server._ticket}",
        headers={"Host": server.host_header},
        environ_base={"REMOTE_ADDR": "192.168.1.77"},
    )

    assert response.status_code == 403
    # A summary, not a record per knock: see the flood tests further down.
    assert any("loopback interface" in line for line in recorder.at(logging.WARNING))


def test_a_forged_host_header_is_refused(server, client, recorder):
    """evil.example resolving to 127.0.0.1 still cannot address us by our own name."""
    response = client.get(
        f"/?t={server._ticket}",
        headers={"Host": "jarvis.evil.example"},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert response.status_code == 403
    assert server._ticket_spent is False
    assert any("Host header" in line for line in recorder.at(logging.WARNING))


def test_a_forged_origin_is_refused(server, client, recorder):
    """A page from anywhere else that talks to our port is answered with a door."""
    response = get(client, server, f"/?t={server._ticket}", Origin="https://evil.example")

    assert response.status_code == 403
    assert any("Origin" in line for line in recorder.at(logging.WARNING))


def test_our_own_origin_is_accepted(server, client):
    response = get(client, server, f"/?t={server._ticket}", Origin=server.origin)

    assert response.status_code == 200


def test_a_request_with_no_origin_header_at_all_is_accepted(server, client):
    """A plain navigation sends no Origin; refusing it would refuse the window itself."""
    response = get(client, server, f"/?t={server._ticket}")

    assert response.status_code == 200


def test_the_socket_applies_the_same_three_checks_as_a_request(server, app, assistant, recorder):
    """An upgrade that skipped the HTTP guards must still die on the doorstep."""
    cookie = admitted(server)
    ws = run_socket(
        server, app, FakeSocket([json.dumps({"type": "say", "text": "delete everything"})]),
        cookie=cookie, host="jarvis.evil.example",
    )

    assert ws.closed is True
    assert assistant.turns == []
    assert any("Host header" in line for line in recorder.at(logging.WARNING))


def test_a_socket_with_a_forged_origin_is_closed(server, app, assistant):
    cookie = admitted(server)
    ws = run_socket(
        server, app, FakeSocket([json.dumps({"type": "quit"})]),
        cookie=cookie, origin="https://evil.example",
    )

    assert ws.closed is True
    assert assistant.stopped.is_set() is False


def test_a_socket_from_a_peer_that_is_not_loopback_is_closed(server, app, assistant):
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket([]), cookie=cookie, peer="10.0.0.4")

    assert ws.closed is True
    assert assistant.turns == []


def test_a_socket_without_the_session_cookie_is_closed(server, app, assistant):
    """The ticket opens the page; only the cookie it issued opens the socket."""
    admitted(server)
    ws = run_socket(server, app, FakeSocket([json.dumps({"type": "listen"})]), cookie=None)

    assert ws.closed is True
    assert assistant.armed == 0
    assert ws.kinds() == ["error"]


def test_an_asset_cannot_be_fetched_without_the_cookie(server, client):
    response = get(client, server, "/static/desk.css")

    assert response.status_code == 401


def test_an_asset_name_that_tries_to_walk_out_of_static_is_refused(server, client):
    get(client, server, f"/?t={server._ticket}")

    response = get(client, server, "/static/..%2f..%2fconfig.yaml")

    assert response.status_code == 404


# ----------------------------------------------------------------------------------
# The socket: hello, replay, then the live bus
# ----------------------------------------------------------------------------------
def test_a_window_is_greeted_with_hello_then_the_replay_then_live_events(
    server, app, bus, assistant
):
    """Opening the window an hour in must show the conversation, not an empty page."""
    bus.publish("heard", text="what is the weather")
    bus.publish("sentence", text="Cold, sir.")
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([lambda: bus.publish("state", state="listening")], linger=0.4),
        cookie=cookie,
    )

    kinds = ws.kinds()
    assert kinds[0] == "hello"
    assert kinds[1:3] == ["heard", "sentence"]
    assert "state" in kinds[3:]


def test_the_hello_frame_carries_what_the_page_needs_before_any_event(server, app, bus):
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket([]), cookie=cookie)

    hello = ws.events[0]
    assert hello["type"] == "hello"
    assert set(hello) >= {
        "version", "model", "whisper", "gpu", "phone_url", "paused", "routines", "tools"
    }
    assert hello["model"] == "qwen3:8b"
    assert hello["phone_url"] == "https://jarvis.example.ts.net"
    assert [r["name"] for r in hello["routines"]] == ["Good night"]
    assert {tool["name"] for tool in hello["tools"]} == {"get_time_date", "run_powershell"}


def test_a_replayed_event_is_not_sent_twice_when_it_is_also_in_the_queue(server, app):
    """Subscribing before the replay closes a race; the cost is a duplicate to skip.

    A bus that hands a new window both the backlog and the history is the worst case
    for that, so this one does, and the window must still see each line once.
    """
    doubling = EchoingBus()
    server._bus = doubling
    doubling.publish("state", state="thinking")
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([], linger=0.2), cookie=cookie)

    kinds = ws.kinds()
    assert kinds[0] == "hello"
    assert kinds.count("state") == 1


def test_the_paused_flag_in_hello_follows_the_state_bus(server, app, assistant):
    assistant.parts.state.set(AssistantState.PAUSED)
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([]), cookie=cookie)

    assert ws.events[0]["paused"] is True


# ----------------------------------------------------------------------------------
# The frame table
# ----------------------------------------------------------------------------------
def test_the_handler_table_is_an_explicit_dict_of_exactly_the_documented_frames(server):
    """Contract 24.3 lists nine frames, and the title bar adds ``window``.

    An eleventh must be a deliberate edit here: this table is the only thing that
    decides what a string arriving off a web socket is allowed to reach.
    """
    assert set(server._handlers) == FRAME_TYPES
    assert all(callable(handler) for handler in server._handlers.values())


def test_no_frame_can_ever_choose_a_method_by_name(server):
    """Dispatch by attribute name would turn a JSON string into a method call."""
    assert "getattr(self," not in SOURCE.replace(" ", "").replace("getattr(self,", "getattr(self, ")
    assert "getattr(self, " not in SOURCE
    assert "eval(" not in SOURCE
    assert "exec(" not in SOURCE


@pytest.mark.parametrize("kind", ["__init__", "stop_all", "", "SAY", "shutdown", "_on_quit"])
def test_any_frame_type_off_the_list_is_answered_with_an_error_and_logged(
    server, app, kind, recorder, assistant
):
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket([json.dumps({"type": kind})]), cookie=cookie)

    assert "error" in ws.kinds()
    assert assistant.stopped.is_set() is False
    assert any("unknown frame type" in line for line in recorder.at(logging.WARNING))


def test_a_frame_that_is_not_json_is_answered_not_fatal(server, app):
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket(["{not json at all"]), cookie=cookie)

    assert "error" in ws.kinds()


def test_a_frame_that_is_json_but_not_an_object_is_refused(server, app):
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket(["[1, 2, 3]"]), cookie=cookie)

    assert "error" in ws.kinds()


# ----------------------------------------------------------------------------------
# say, and the rest of the nine
# ----------------------------------------------------------------------------------
def test_a_typed_turn_reaches_the_assistant_under_its_own_dispatcher(server, app, assistant):
    """The desk is the keyboard. There is no guard between it and the tools."""
    cookie = admitted(server)
    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "say", "text": "  what time is it  "})]),
        cookie=cookie,
    )

    assert assistant.turns == ["what time is it"]
    assert assistant.dispatchers == [assistant.parts.dispatcher]
    assert "ack" in ws.kinds()


def test_the_desk_never_builds_a_guard_or_asks_for_a_tier(server, app, assistant):
    """Contract 24.2: full rights. A tier check appearing here would be a regression."""
    cookie = admitted(server)
    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "say", "text": "delete the log file"})]),
        cookie=cookie,
    )

    assert assistant.dispatchers == [assistant.parts.dispatcher]
    assert "RemoteGuard" not in SOURCE
    assert "remote_tier" not in SOURCE
    assert "wrap_dispatcher" not in SOURCE
    assert "allow_guarded" not in SOURCE


def test_an_empty_say_frame_is_refused_rather_than_queued(server, app, assistant, recorder):
    cookie = admitted(server)
    ws = run_socket(
        server, app, FakeSocket([json.dumps({"type": "say", "text": "   "})]), cookie=cookie
    )

    assert assistant.turns == []
    assert "error" in ws.kinds()
    assert any("empty say frame" in line for line in recorder.at(logging.WARNING))


def test_a_pasted_essay_is_capped_so_it_cannot_flood_the_model(server, app, assistant):
    cookie = admitted(server)
    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "say", "text": "x" * 50_000})]),
        cookie=cookie,
    )

    assert len(assistant.turns[0]) == MAX_SAY == 2000


def test_a_busy_assistant_is_reported_and_not_retried(server, app, assistant):
    assistant.accept = False
    cookie = admitted(server)

    ws = run_socket(
        server, app, FakeSocket([json.dumps({"type": "say", "text": "hello"})]), cookie=cookie
    )

    assert "busy" in ws.kinds()


def test_listen_stop_confirm_pause_and_resume_reach_their_hooks(server, app, assistant):
    cookie = admitted(server)
    script = [
        json.dumps({"type": "listen"}),
        json.dumps({"type": "stop"}),
        json.dumps({"type": "confirm", "granted": True}),
        json.dumps({"type": "confirm", "granted": False}),
        json.dumps({"type": "pause"}),
        json.dumps({"type": "resume"}),
    ]

    run_socket(server, app, FakeSocket(script), cookie=cookie)

    assert assistant.armed == 1
    assert assistant.aborted == 1
    assert assistant.confirmations == [True, False]
    assert assistant.paused is False


def test_a_confirm_frame_without_a_granted_key_is_read_as_a_refusal(server, app, assistant):
    """Silence is not consent: a malformed confirmation must never mean yes."""
    cookie = admitted(server)
    run_socket(server, app, FakeSocket([json.dumps({"type": "confirm"})]), cookie=cookie)

    assert assistant.confirmations == [False]


def test_a_configured_routine_runs_and_an_unknown_one_is_refused(server, app, assistant):
    cookie = admitted(server)
    script = [
        json.dumps({"type": "routine", "name": "Good night"}),
        json.dumps({"type": "routine", "name": "Format the disk"}),
    ]

    ws = run_socket(server, app, FakeSocket(script), cookie=cookie)

    assert assistant.routines == ["Good night"]
    assert "error" in ws.kinds()


def test_a_quit_frame_stops_the_assistant_off_the_socket_thread(server, app, assistant):
    """Stopping joins the very threads the frame arrived on, so it cannot run here."""
    cookie = admitted(server)
    ws = run_socket(server, app, FakeSocket([json.dumps({"type": "quit"})]), cookie=cookie)

    assert assistant.stopped.wait(timeout=5) is True
    assert "ack" in ws.kinds()


# ----------------------------------------------------------------------------------
# set: the allowlist
# ----------------------------------------------------------------------------------
def test_the_settings_allowlist_is_exactly_the_five_from_the_contract(server):
    assert set(SETTINGS) == {
        "tts.speed",
        "audio.chime_volume",
        "assistant.brief_mode",
        "wake.sensitivity",
        "ui.always_on_top",
    }


@pytest.mark.parametrize(
    "key, value",
    [
        ("brain.model", "llama3:70b"),
        ("logging.file", "C:/Windows/System32/drivers/etc/hosts"),
        ("remote.enabled", True),
        ("remote.allow_guarded", True),
        ("assistant.safety_mode", "off"),
        ("tts", {"speed": 9}),
        ("", 1),
        ("tts.speed.extra", 1),
    ],
)
def test_a_key_that_is_not_on_the_allowlist_never_reaches_the_config(
    server, app, assistant, recorder, key, value
):
    """An allowlist and not a denylist: a new config key stays unreachable by default."""
    cookie = admitted(server)
    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "set", "key": key, "value": value})]),
        cookie=cookie,
    )

    assert assistant.settings == []
    assert "error" in ws.kinds()
    assert any("which is not allowed" in line for line in recorder.at(logging.WARNING))


@pytest.mark.parametrize(
    "key, sent, stored",
    [
        ("tts.speed", 9.0, 1.6),
        ("tts.speed", -4, 0.5),
        ("tts.speed", 1.1, 1.1),
        ("audio.chime_volume", 50, 1.0),
        ("audio.chime_volume", -0.5, 0.0),
        ("wake.sensitivity", 1.0, 0.95),
        ("wake.sensitivity", 0.0, 0.05),
        ("assistant.brief_mode", True, True),
        ("assistant.brief_mode", "false", False),
        ("ui.always_on_top", 1, True),
    ],
)
def test_every_allowed_value_is_validated_and_clamped_before_it_reaches_the_config(
    server, app, assistant, key, sent, stored
):
    """The page is a slider; a slider that says 50 means "the top", not 5000 percent."""
    cookie = admitted(server)
    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "set", "key": key, "value": sent})]),
        cookie=cookie,
    )

    assert assistant.settings == [(key, stored)]


@pytest.mark.parametrize(
    "key, value",
    [
        ("tts.speed", "quickly"),
        ("tts.speed", None),
        ("tts.speed", [1.0]),
        ("tts.speed", True),
        ("tts.speed", float("nan")),
        ("tts.speed", float("inf")),
        ("audio.chime_volume", {"a": 1}),
        ("assistant.brief_mode", 7),
        ("assistant.brief_mode", "maybe"),
        ("ui.always_on_top", None),
    ],
)
def test_a_value_of_the_wrong_shape_is_refused_and_logged(
    server, app, assistant, recorder, key, value
):
    cookie = admitted(server)
    frame = json.dumps({"type": "set", "key": key, "value": value})
    # json.dumps writes NaN and Infinity, which json.loads reads back; that is the
    # exact shape a hand-written client could send, so it must be refused, not crash.
    ws = run_socket(server, app, FakeSocket([frame]), cookie=cookie)

    assert assistant.settings == []
    assert "error" in ws.kinds()
    assert any("unusable value" in line for line in recorder.at(logging.WARNING))


# ----------------------------------------------------------------------------------
# The assistant is called defensively
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hook, frame",
    [
        ("submit_desk_turn", {"type": "say", "text": "hello"}),
        ("arm_listening", {"type": "listen"}),
        ("abort_turn", {"type": "stop"}),
        ("answer_confirmation", {"type": "confirm", "granted": True}),
        ("run_routine", {"type": "routine", "name": "Good night"}),
        ("apply_setting", {"type": "set", "key": "tts.speed", "value": 1.0}),
    ],
)
def test_a_build_without_one_of_the_six_hooks_answers_with_an_error_not_a_crash(
    server, app, assistant, recorder, hook, frame
):
    """The six hooks land in another module; the window must survive their absence."""
    setattr(assistant, hook, None)
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([json.dumps(frame)]), cookie=cookie)

    assert "error" in ws.kinds()
    assert any(hook in line for line in recorder.at(logging.ERROR))


def test_a_hook_that_raises_is_logged_and_answered_not_propagated(server, app, assistant):
    def explode(text: str) -> bool:
        raise RuntimeError("the brain is on fire")

    assistant.submit_desk_turn = explode
    cookie = admitted(server)

    ws = run_socket(
        server, app, FakeSocket([json.dumps({"type": "say", "text": "hello"})]), cookie=cookie
    )

    assert ws.kinds()[0] == "hello"
    assert "busy" in ws.kinds() or "error" in ws.kinds()


# ----------------------------------------------------------------------------------
# Nothing here may stall the assistant
# ----------------------------------------------------------------------------------
def test_a_window_that_stopped_reading_drops_frames_instead_of_blocking(server):
    """The publisher is the assistant's own thread. It never waits on a web page."""
    window = _Window(FakeSocket(), queue.Queue(maxsize=4))

    started = time.monotonic()
    for index in range(5000):
        window.offer({"type": "sentence", "text": str(index)})
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert window.outbox.qsize() <= 256


def test_a_dead_socket_ends_the_window_without_raising_into_anything(server, app, bus):
    cookie = admitted(server)
    ws = DeadSocket([json.dumps({"type": "listen"})], linger=0.1)

    run_socket(server, app, ws, cookie=cookie)

    assert server.windows == 0
    assert bus._subscribers == []


def test_a_window_that_never_reads_does_not_stall_a_publisher(server, app, bus):
    """A client that connects and then goes to sleep is the classic slow-loris case."""
    cookie = admitted(server)
    ws = FakeSocket([TIMEOUT, TIMEOUT], linger=0.3)
    thread = threading.Thread(target=run_socket, args=(server, app, ws), kwargs={"cookie": cookie})
    thread.start()
    time.sleep(0.05)

    started = time.monotonic()
    for index in range(2000):
        bus.publish("sentence", text=str(index))
    elapsed = time.monotonic() - started
    thread.join(timeout=5)

    assert elapsed < 2.0
    assert thread.is_alive() is False


def test_every_window_is_accounted_for_so_the_poller_knows_when_to_stop(server, app):
    cookie = admitted(server)
    assert server.windows == 0

    run_socket(server, app, FakeSocket([]), cookie=cookie)

    assert server.windows == 0


# ----------------------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------------------
@pytest.fixture
def telemetry(monkeypatch, assistant):
    """A stand-in jarvis.core.telemetry, which another module owns."""
    module = types.ModuleType("jarvis.core.telemetry")
    readings: list[object] = []

    def snapshot(target):
        readings.append(target)
        return {"cpu": 12, "ram": 41, "gpu": {"load": 3}}

    module.snapshot = snapshot
    monkeypatch.setitem(sys.modules, "jarvis.core.telemetry", module)
    return readings


def test_status_is_published_every_couple_of_seconds_while_a_window_is_open(
    server, bus, telemetry, assistant
):
    """Telemetry is pushed, not polled by the page: one reading, every open window."""
    server.status_interval = 0.02
    server._windows = 1
    thread = threading.Thread(target=server.status_poller, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(bus.replay()) < 3:
        time.sleep(0.01)
    server._windows = 0
    server._poll_wake.set()
    thread.join(timeout=5)

    published = [event for event in bus.replay() if event.type == "status"]
    assert len(published) >= 3
    assert published[0].payload == {"cpu": 12, "ram": 41, "gpu": {"load": 3}}
    assert telemetry[0] is assistant


def test_the_poller_stops_when_the_last_window_disconnects(server, bus, telemetry):
    """nvidia-smi is expensive; nobody watching means nobody paying."""
    server.status_interval = 0.02
    server._windows = 1
    thread = threading.Thread(target=server.status_poller, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not bus.replay():
        time.sleep(0.01)

    server._windows = 0
    server._poll_wake.set()
    thread.join(timeout=5)
    assert thread.is_alive() is False

    settled = len(bus.replay())
    time.sleep(0.15)
    assert len(bus.replay()) == settled


def test_a_missing_telemetry_module_costs_the_status_strip_and_nothing_else(
    server, bus, recorder, monkeypatch
):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "jarvis.core.telemetry":
            raise ImportError("not built yet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    server._windows = 1

    server.status_poller()

    assert bus.replay() == []
    assert any("Telemetry is unavailable" in line for line in recorder.at(logging.WARNING))


def test_a_telemetry_reading_that_raises_does_not_kill_the_poller(server, bus, monkeypatch):
    module = types.ModuleType("jarvis.core.telemetry")
    calls = {"n": 0}

    def snapshot(target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("nvidia-smi hung")
        return {"cpu": 1}

    module.snapshot = snapshot
    monkeypatch.setitem(sys.modules, "jarvis.core.telemetry", module)
    server.status_interval = 0.02
    server._windows = 1
    thread = threading.Thread(target=server.status_poller, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not bus.replay():
        time.sleep(0.01)
    server._windows = 0
    server._poll_wake.set()
    thread.join(timeout=5)

    assert bus.replay()


# ----------------------------------------------------------------------------------
# The module's own manners
# ----------------------------------------------------------------------------------
def test_importing_the_module_imports_neither_flask_nor_anything_windows():
    """It must import on a bare Linux box, and open nothing until start() is called."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys, jarvis.desk.server as m; "
        "assert 'flask' not in sys.modules, sorted(sys.modules); "
        "assert m.DeskServer is not None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(root), capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr


def test_the_server_builds_a_real_desk_bus_when_that_module_exists(desk_config, assistant):
    """The stand-in above is for isolation only; in the product it is the real thing."""
    bus_module = pytest.importorskip("jarvis.desk.bus")
    made = DeskServer(desk_config, assistant)

    assert isinstance(made.bus, bus_module.DeskBus)


def test_a_window_on_the_real_bus_sees_the_real_event_shape(desk_config, assistant, recorder):
    """Everything above uses a stand-in; once is not enough for the wire format."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    bus_module = pytest.importorskip("jarvis.desk.bus")
    log = logging.getLogger("desk-test-realbus")
    log.propagate = False
    log.addHandler(recorder)
    made = DeskServer(desk_config, assistant, log, bus=bus_module.DeskBus())
    made.heartbeat = 0.05
    made.status_interval = 5.0
    made._bound_port = 8123
    app = made.create_app()
    made.bus.publish("heard", text="what is the weather")
    made.bus.publish("sentence", text="Cold, sir.")

    try:
        ws = run_socket(made, app, FakeSocket([], linger=0.1), cookie=admitted(made))
    finally:
        made.stop()
        log.removeHandler(recorder)

    events = ws.events
    assert [event["type"] for event in events[:3]] == ["hello", "heard", "sentence"]
    assert events[1]["text"] == "what is the weather"
    assert isinstance(events[2]["at"], float)


def test_a_missing_bus_module_refuses_to_start_rather_than_raising(
    desk_config, assistant, recorder, monkeypatch
):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "jarvis.desk.bus":
            raise ImportError("not built yet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    log = logging.getLogger("desk-test-nobus")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    log.addHandler(recorder)
    made = DeskServer(desk_config, assistant, log)

    try:
        assert made.start() is False
        assert any("no event bus" in line for line in recorder.at(logging.ERROR))
    finally:
        log.removeHandler(recorder)


def test_stop_is_idempotent_and_safe_before_the_server_ever_started(server):
    server.stop()
    server.stop()

    assert server.running is False


# ----------------------------------------------------------------------------------
# One real socket, end to end
# ----------------------------------------------------------------------------------
def test_a_real_loopback_request_gets_in_once_and_the_same_link_never_twice(server):
    """Everything above fakes the request; this one is a genuine browser-shaped GET."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    server._bound_port = 0
    assert server.start() is True
    # The container routes HTTP through a proxy; loopback must not go near it.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    link = server.url  # held, because reading the property again mints a new one

    try:
        with opener.open(link, timeout=10) as response:
            assert response.status == 200
            assert "HttpOnly" in response.headers.get("Set-Cookie", "")
        with pytest.raises(urllib.error.HTTPError) as caught:
            opener.open(link, timeout=10)
        assert caught.value.code == 401

        with pytest.raises(urllib.error.HTTPError) as forged:
            request = urllib.request.Request(server.safe_url, headers={"Host": "evil.example"})
            opener.open(request, timeout=10)
        assert forged.value.code == 403
    finally:
        server.stop()


# ----------------------------------------------------------------------------------
# The door under a flood
# ----------------------------------------------------------------------------------
def test_a_thousand_refused_requests_leave_at_most_a_couple_of_lines_above_debug(
    server, client, recorder
):
    """A page on the open web cannot get in, but it can knock, and the log is finite.

    ``fetch('http://127.0.0.1:<port>/', {mode:'no-cors'})`` in a loop produced 704
    records a second at ERROR — 236 KB/s, which recycles the shipped 20 MB rotation
    budget in about a minute and a half and takes the transcript and the tool-call
    audit trail CLAUDE.md requires with it. The same records went to the window's
    bus, where a 256-slot queue full of log events pushed a pending ``confirm`` off
    the back: a browser-reachable way to hide the confirmation bar while a guarded
    tool is waiting. So the detail goes to DEBUG and only a count comes out loud.
    """
    for _ in range(1000):
        assert get(client, server, "/", Origin="https://evil.example").status_code == 403

    assert len(recorder.at(logging.WARNING)) <= 2
    assert len(recorder.records) >= 1000  # the detail is still there, at DEBUG


def test_a_flood_of_uncredentialed_assets_is_counted_rather_than_written_out(
    server, client, recorder
):
    """The asset route is the second door, and it was the second flood channel."""
    for _ in range(500):
        assert get(client, server, "/static/desk.css").status_code == 401

    assert len(recorder.at(logging.WARNING)) <= 2


@pytest.mark.parametrize("door", ["no cookie", "forged origin"])
def test_a_flood_of_refused_sockets_is_counted_rather_than_written_out(
    server, app, recorder, assistant, door
):
    """And the third: an upgrade that dies on the doorstep costs one DEBUG line.

    Both of its doorsteps. The three header checks and the cookie refuse at
    different points and logged at different levels, and a flood through either
    fills the same file.
    """
    cookie = admitted(server) if door == "forged origin" else None
    origin = "https://evil.example" if door == "forged origin" else None

    for _ in range(500):
        ws = run_socket(
            server, app, FakeSocket([json.dumps({"type": "quit"})]), cookie=cookie,
            **({"origin": origin} if origin else {}),
        )
        assert ws.closed is True

    assert assistant.stopped.is_set() is False
    assert len(recorder.at(logging.WARNING)) <= 2


def test_the_one_line_that_does_come_out_says_how_many_were_refused_and_why(recorder):
    """A count is the only interesting thing about a flood; the last reason is the clue."""
    log = logging.getLogger("desk-test-refusals")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.addHandler(recorder)
    counter = _RefusalLog(log, interval=0.05)

    try:
        for _ in range(50):
            counter.note("a desk request", "the Origin 'https://evil.example' is not ours")
        time.sleep(0.06)
        counter.note("a desk request", "the Origin 'https://evil.example' is not ours")
    finally:
        log.removeHandler(recorder)

    summaries = recorder.at(logging.WARNING)
    assert len(summaries) == 2
    assert "Refused 50 desk requests" in summaries[1]
    assert "evil.example" in summaries[1]


def test_a_refusal_that_happens_once_is_still_reported_at_once(server, client, recorder):
    """Rate limiting must not turn the one refusal worth reading into silence."""
    assert get(client, server, "/", Origin="https://evil.example").status_code == 403

    assert any("Origin" in line for line in recorder.at(logging.WARNING))


# ----------------------------------------------------------------------------------
# The title bar
# ----------------------------------------------------------------------------------
def test_the_window_actions_are_exactly_the_five_a_title_bar_needs(server):
    """Navigating, evaluating and moving to another screen are not a title bar's work."""
    assert set(WINDOW_ACTIONS) == {"minimize", "maximize", "restore", "close", "resize"}
    assert "window" in server._handlers


@pytest.mark.parametrize(
    "action, called",
    [
        ("minimize", "minimize"),
        ("maximize", "maximize"),
        ("restore", "restore"),
        ("close", "hide"),
    ],
)
def test_the_title_bars_buttons_reach_the_native_window(server, app, action, called):
    """The window is frameless, so these buttons are ours and this is all they have.

    They sent a ``window`` frame that no handler accepted: Minimise and Close did
    nothing whatever, answered with a machine error, and the window could not be put
    away or shrunk by any means at all.
    """
    window = FakeWindow()
    server.attach_window(window)
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "window", "action": action})]),
        cookie=cookie,
    )

    assert window.calls == [(called,)]
    assert [event for event in ws.events if event["type"] == "ack"] == [
        {"type": "ack", "for": "window", "action": action}
    ]


def test_close_puts_the_window_in_the_tray_rather_than_ending_the_session(
    server, app, assistant
):
    """Closing to the tray is reversible from the tray; ending the session is not."""
    window = FakeWindow()
    server.attach_window(window)
    cookie = admitted(server)

    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "window", "action": "close"})]),
        cookie=cookie,
    )

    assert window.calls == [("hide",)]
    assert assistant.stopped.is_set() is False


@pytest.mark.parametrize(
    "sent, wanted",
    [
        ((1200, 800), (1200, 800)),
        ((1200.6, 800.4), (1201, 800)),
        (("1200", "800"), (1200, 800)),
        ((10, 10), (WINDOW_BOUNDS[0], WINDOW_BOUNDS[1])),
        ((99999, 99999), (WINDOW_BOUNDS[2], WINDOW_BOUNDS[3])),
    ],
)
def test_a_size_from_the_page_is_whole_and_inside_sane_bounds(server, app, sent, wanted):
    """A grip dragged off the edge of the screen means "as big as you go", not 99999."""
    window = FakeWindow()
    server.attach_window(window)
    cookie = admitted(server)
    frame = {"type": "window", "action": "resize", "width": sent[0], "height": sent[1]}

    run_socket(server, app, FakeSocket([json.dumps(frame)]), cookie=cookie)

    assert window.calls == [("resize", wanted[0], wanted[1])]


@pytest.mark.parametrize(
    "width, height",
    [
        ("wide", 800),
        (None, 800),
        (1200, None),
        (True, 800),
        (float("nan"), 800),
        (float("inf"), 800),
        ([1200], 800),
        (1200, {"height": 800}),
    ],
)
def test_a_size_that_is_not_a_pair_of_numbers_is_refused_and_logged(
    server, app, recorder, width, height
):
    """Nonsense in a size field means the frame did not come from our page."""
    window = FakeWindow()
    server.attach_window(window)
    cookie = admitted(server)
    frame = {"type": "window", "action": "resize", "width": width, "height": height}

    ws = run_socket(server, app, FakeSocket([json.dumps(frame)]), cookie=cookie)

    assert window.calls == []
    assert "error" in ws.kinds()
    assert any("unusable size" in line for line in recorder.at(logging.WARNING))


@pytest.mark.parametrize(
    "action", ["", "destroy", "evaluate_js", "load_url", "toggle", "__init__", "hide", "show"]
)
def test_a_window_action_that_is_not_on_the_list_never_reaches_the_host(
    server, app, recorder, action
):
    """An allowlist, and the page's own spelling: ``close``, never ``hide``."""
    window = FakeWindow()
    server.attach_window(window)
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "window", "action": action})]),
        cookie=cookie,
    )

    assert window.calls == []
    assert "error" in ws.kinds()
    assert any("window actions" in line for line in recorder.at(logging.WARNING))


def test_a_window_frame_with_no_window_attached_is_a_logged_no_op(server, app, recorder):
    """--no-window, a browser tab, a machine without WebView2: the buttons remain."""
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "window", "action": "minimize"})]),
        cookie=cookie,
    )

    assert "error" not in ws.kinds()
    assert any("none is attached" in record.getMessage() for record in recorder.records)


@pytest.mark.parametrize("host", ["explodes", "bare"])
def test_a_host_that_refuses_or_has_no_such_method_never_raises_into_the_page(
    server, app, host
):
    """Nothing in a face may raise into the assistant, least of all a title bar."""
    server.attach_window(FakeWindow(explode=True) if host == "explodes" else SimpleNamespace())
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "window", "action": "minimize"})]),
        cookie=cookie,
    )

    assert "ack" in ws.kinds()
    assert "error" not in ws.kinds()


# ----------------------------------------------------------------------------------
# Opening the window a second time
# ----------------------------------------------------------------------------------
def test_two_consecutive_opens_from_the_tray_both_work(server, app):
    """The ticket is single-use, so the second open was told its own link was spent.

    Closing the window to the tray and asking for it again an hour later is the
    ordinary way this app is used, and it was the one way it could not be.
    """
    first = server.url
    assert get(app.test_client(), server, path_of(first)).status_code == 200

    second = server.url

    assert second != first
    assert get(app.test_client(), server, path_of(second)).status_code == 200


def test_a_spent_ticket_is_still_refused_after_a_fresh_one_has_been_issued(server, app):
    """Re-opening the window mints a new link; it does not revive the used one."""
    stale = server.url
    assert get(app.test_client(), server, path_of(stale)).status_code == 200
    fresh = server.url

    assert get(app.test_client(), server, path_of(stale)).status_code == 401
    assert get(app.test_client(), server, path_of(fresh)).status_code == 200


def test_reading_the_url_again_does_not_churn_a_ticket_that_is_still_good(server):
    """Only a spent or timed-out ticket is replaced: the URL is not a random number."""
    assert server.url == server.url


def test_a_ticket_that_timed_out_is_replaced_rather_than_offered_again(server):
    """Two minutes after start-up the tray must still be able to open a window."""
    stale = server._ticket
    server._ticket_expires = time.monotonic() - 1.0

    assert stale not in server.url


def test_issue_ticket_clears_the_spent_flag_and_restarts_the_clock(server):
    """What the tray calls when it opens the window again."""
    old = server._ticket
    server._ticket_spent = True
    server._ticket_expires = time.monotonic() - 1.0

    fresh = server.issue_ticket()

    assert fresh != old
    assert len(fresh) >= 40  # token_urlsafe(32) is 43 characters
    assert server._ticket_spent is False
    assert server._ticket_expires > time.monotonic() + TICKET_TTL - 5


def test_a_reissued_ticket_is_never_written_to_the_log(server, recorder):
    """The new one is exactly as much of a secret as the one it replaces."""
    fresh = server.issue_ticket()

    written = "\n".join(record.getMessage() for record in recorder.records)
    assert fresh not in written


# ----------------------------------------------------------------------------------
# A routine must not block the socket
# ----------------------------------------------------------------------------------
def test_a_routine_that_waits_for_a_confirmation_can_still_be_confirmed(server, app):
    """A macro may contain a GUARDED tool, and the Confirm button is a socket frame.

    Running the macro on the reader thread meant the page showed a bar whose click
    nothing was left to read: the routine waited for a confirmation that could only
    arrive over the thread it was standing on. Both sides waited until the timeout.
    """
    assistant = WaitingAssistant()
    server.assistant = assistant
    cookie = admitted(server)
    script = [
        json.dumps({"type": "routine", "name": "Good night"}),
        lambda: assistant.asked.wait(timeout=5),
        json.dumps({"type": "confirm", "granted": True}),
    ]

    ws = run_socket(server, app, FakeSocket(script, linger=0.2), cookie=cookie)

    assert assistant.confirmations == [True]
    assert assistant.confirmed is True  # answered while it waited, not after it gave up
    assert "ack" in ws.kinds()


def test_the_page_is_acknowledged_before_the_routine_has_finished(server, app):
    """The ack says "heard", not "done": what the routine did arrives on the bus."""
    assistant = WaitingAssistant()
    server.assistant = assistant
    cookie = admitted(server)

    started = time.monotonic()
    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "routine", "name": "Good night"})], linger=0.1),
        cookie=cookie,
    )
    elapsed = time.monotonic() - started
    acks = [event for event in ws.events if event["type"] == "ack"]

    assert acks == [{"type": "ack", "for": "routine", "name": "Good night"}]
    assert elapsed < 1.0  # the macro is still waiting to be confirmed, and may wait
    assert assistant.answered.is_set() is False
    assistant.answered.set()


def test_what_the_routine_said_reaches_the_window_through_the_bus(server, app, bus, assistant):
    """The spoken line cannot ride on the ack any more, so it goes the way lines do."""
    cookie = admitted(server)

    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "routine", "name": "Good night"})]),
        cookie=cookie,
    )

    assert waited_for(lambda: assistant.routines == ["Good night"])
    assert waited_for(
        lambda: any(
            event.type == "sentence" and event.payload.get("text") == "Goodnight, sir."
            for event in bus.replay()
        )
    )


def test_a_line_the_assistant_has_already_published_is_not_printed_twice(
    server, app, bus, assistant
):
    """The macro speaks through the assistant, which narrates itself onto this bus."""
    def narrate(name: str) -> str:
        assistant.routines.append(name)
        bus.publish("sentence", text="Goodnight, sir.")
        return "Goodnight, sir."

    assistant.run_routine = narrate
    cookie = admitted(server)

    run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "routine", "name": "Good night"})]),
        cookie=cookie,
    )

    assert waited_for(lambda: assistant.routines == ["Good night"])
    time.sleep(0.1)
    spoken = [event for event in bus.replay() if event.type == "sentence"]
    assert len(spoken) == 1


def test_an_unknown_routine_is_still_refused_before_any_thread_is_started(
    server, app, assistant, recorder
):
    """Threading the run must not thread past the allowlist that guards it."""
    cookie = admitted(server)

    ws = run_socket(
        server, app,
        FakeSocket([json.dumps({"type": "routine", "name": "Format the disk"})]),
        cookie=cookie,
    )

    time.sleep(0.05)
    assert assistant.routines == []
    assert "error" in ws.kinds()


# ----------------------------------------------------------------------------------
# Bounded frames
# ----------------------------------------------------------------------------------
def test_the_socket_will_not_read_a_frame_larger_than_sixty_four_kilobytes(app):
    """Every frame the page sends is a sentence and a couple of numbers."""
    options = app.config["SOCK_SERVER_OPTIONS"]

    assert options["max_message_size"] == 65536
    assert options["ping_interval"] == 25


@pytest.mark.parametrize(
    "frame, letter",
    [
        ({"type": "z" * 5000}, "z"),
        ({"type": "set", "key": "k" * 5000, "value": 1}, "k"),
        ({"type": "window", "action": "w" * 5000}, "w"),
    ],
)
def test_a_field_off_a_frame_is_clamped_before_it_reaches_a_log_line_or_a_reply(
    server, app, recorder, frame, letter
):
    """The page chooses these strings. The log file and the window's feed do not."""
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([json.dumps(frame)]), cookie=cookie)

    written = "\n".join(record.getMessage() for record in recorder.records)
    answered = "\n".join(json.dumps(event) for event in ws.events)
    assert letter * (MAX_FIELD + 1) not in written
    assert letter * (MAX_FIELD + 1) not in answered


# ----------------------------------------------------------------------------------
# The server's own access log
# ----------------------------------------------------------------------------------
def test_the_access_log_is_silenced_so_the_ticket_never_reaches_the_console(
    server, recorder
):
    """Werkzeug prints every request line, query string and all, to stderr.

    That put the one secret this module has in front of anyone reading over the
    operator's shoulder, and gave a page that cannot get in a second way to flood —
    one our own rate limit does not cover.
    """
    pytest.importorskip("werkzeug")
    # No socket: only the handler's logging is under test, and that is all it touches.
    handler = object.__new__(server._handler_class())

    assert handler.log_request(200, 17) is None
    handler.log("info", '"GET /?t=%s HTTP/1.1" 200 -', server._ticket)

    written = "\n".join(record.getMessage() for record in recorder.records)
    assert recorder.at(logging.INFO) == []
    assert server._ticket not in written
    assert "?t=..." in written


def test_a_real_request_writes_nothing_to_werkzeugs_own_logger(server):
    """The proof that the handler is the one the running server actually uses."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    watcher = Recorder()
    werkzeug_log = logging.getLogger("werkzeug")
    werkzeug_log.addHandler(watcher)
    server._bound_port = 0
    assert server.start() is True
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(server.url, timeout=10) as response:
            assert response.status == 200
    finally:
        server.stop()
        werkzeug_log.removeHandler(watcher)

    assert watcher.records == []


# ----------------------------------------------------------------------------------
# What the switches are set to
# ----------------------------------------------------------------------------------
def test_hello_tells_the_window_what_its_switches_are_already_set_to(
    server, app, desk_config
):
    """The page read ``message.settings``, which nothing ever sent.

    Every switch and slider therefore drew a guess, and the first click wrote the
    guess into config.yaml. The same allowlist as ``set``, read in the other
    direction: the window learns about exactly the keys it may change.
    """
    desk_config.set("assistant.brief_mode", True)
    desk_config.set("tts.speed", 1.25)
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([]), cookie=cookie)

    settings = ws.events[0]["settings"]
    assert set(settings) == set(SETTINGS)
    assert settings["assistant.brief_mode"] is True
    assert settings["tts.speed"] == 1.25


def test_hello_carries_no_setting_that_the_window_may_not_change(server, app, desk_config):
    """A second face must not learn brain.model from a frame meant for a slider."""
    cookie = admitted(server)

    ws = run_socket(server, app, FakeSocket([]), cookie=cookie)

    assert "brain.model" not in ws.events[0]["settings"]
    assert desk_config.get("brain.model") not in json.dumps(ws.events[0]["settings"])


def test_the_key_allowlist_and_the_value_allowlist_are_the_same_table():
    """Two allowlists that can drift are one allowlist and one hole.

    ``Assistant.apply_setting`` guards the key with its own ``DESK_SETTINGS`` and then
    validates the value with this module's ``SETTINGS``. A key added to one and not the
    other is either a setting the window can never change or, worse, a key that passes
    the door and reaches the config unvalidated.
    """
    from jarvis.core.assistant import DESK_SETTINGS
    from jarvis.desk.server import SETTINGS

    assert set(DESK_SETTINGS) == set(SETTINGS)
