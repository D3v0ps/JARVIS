"""The assistant talking to the window, and the window never talking back into a turn.

Everything the desk needs from the loop lives here: the six hooks the server calls, the
fan that lets one narration feed two faces, and the argument about the main thread that
pywebview and tkinter would otherwise have at start-up. The desk package itself is
faked throughout - it is being written next door, and a wiring test that fails because
someone else's file is half-saved is a test that teaches nobody anything.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from jarvis.config import DEFAULTS
from jarvis.core.assistant import Assistant, _HudDispatcher, _HudFan, _Mode
from jarvis.core.state import AssistantState, StateBus
from jarvis.core.wiring import build_desk
from jarvis.ui.hud import HudModel
from jarvis.ui.tray import Tray

ROOT = Path(__file__).resolve().parent.parent


# --- doubles -------------------------------------------------------------------------
class FakeBus:
    """A DeskBus that only remembers. The real one fans out to sockets."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.closed = False

    def publish(self, kind: str, /, **payload: object) -> tuple[str, dict]:
        event = (kind, dict(payload))
        self.events.append(event)
        return event

    def close(self) -> None:
        self.closed = True

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def payloads(self, kind: str) -> list[dict]:
        return [payload for name, payload in self.events if name == kind]

    def payload(self, kind: str) -> dict:
        """The most recent event of ``kind``, or a failure that says what did arrive."""
        found = self.payloads(kind)
        if not found:
            raise AssertionError(f"nothing of type {kind!r} reached the window: {self.kinds()}")
        return found[-1]


class AngryBus:
    """A bus that has fallen over. The turn must not notice."""

    def publish(self, kind: str, /, **payload: object) -> None:
        raise RuntimeError("the socket is gone")


class FakeServer:
    """A DeskServer that binds nothing."""

    url = "http://127.0.0.1:53421/?t=ticket"

    def __init__(self, cfg=None, assistant=None, logger=None, *, starts: bool = True) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.bus = FakeBus()
        self.started = False
        self.stopped = False
        self.attached: object = "never asked"
        self._starts = starts

    def attach_window(self, window: object) -> None:
        self.attached = window

    def start(self) -> bool:
        self.started = self._starts
        return self._starts

    def stop(self) -> None:
        self.stopped = True


class FakeWindow:
    """A DeskWindow that hosts nothing, and remembers what it was asked to do."""

    def __init__(self, cfg=None, url="", *, logger=None, on_close=None,
                 needs_main_thread: bool = False, shows: bool = True,
                 hosts: str = "edge") -> None:
        self.cfg = cfg
        #: The real DeskWindow takes a string *or* something to ask, and asks afresh at
        #: every open because the ticket in the URL is single-use. The fake has to do the
        #: same, or it would pass a callable off as an address and prove nothing.
        self.url_source = url
        self.on_close = on_close
        self.needs_main_thread = needs_main_thread
        self.started = 0
        self.shown = 0
        self.ran = 0
        self.stopped = 0
        self._shows = shows
        #: Which host in the chain this machine would win with, for the probe below.
        self._hosts = hosts

    @property
    def url(self) -> str:
        source = self.url_source
        return str((source() if callable(source) else source) or "")

    def _order(self) -> tuple[str, ...]:
        return ("webview", "edge", "browser")

    def _probe(self, name: str) -> bool:
        return name == self._hosts

    def would_host(self) -> str:
        """The real DeskWindow's public answer to "which host would take this page",
        which build_desk asks before anything is built. Mirrored here, through the
        same probe, so a fake that lies about the chain cannot pass."""
        return next((name for name in self._order() if self._probe(name)), "none")

    def start(self) -> bool:
        self.started += 1
        return True

    def show(self) -> bool:
        self.shown += 1
        return self._shows

    def run(self) -> None:
        self.ran += 1

    def stop(self) -> None:
        self.stopped += 1


class FakeOverlay:
    """The ring, with the one property this argument turns on."""

    def __init__(self, *, needs_main_thread: bool = True) -> None:
        self.needs_main_thread = needs_main_thread
        self.hud = HudModel()
        self.ran = 0
        self.stopped = 0

    def run(self) -> None:
        self.ran += 1

    def stop(self) -> None:
        self.stopped += 1


class FakeSpeaker:
    """Records what would have been heard, in the shape the Speaker exposes."""

    def __init__(self) -> None:
        self.said: list[str] = []
        self.queued: list[str] = []
        self.chimes = 0
        self.stops = 0
        self.chime_volume = 0.35
        self.is_speaking = False
        # The Speaker keeps its engines private; the assistant retunes what it can see.
        self._engine = SimpleNamespace(speed=1.0)
        self._engines: dict[str, object] = {}

    def say(self, text: str, *, blocking: bool = True) -> None:
        self.said.append(text)

    def enqueue(self, sentence: str) -> None:
        self.queued.append(sentence)

    def play_chime(self, audio, sample_rate: int) -> None:
        self.chimes += 1

    def wait(self, timeout: float | None = None) -> None:
        return None

    def stop(self) -> None:
        self.stops += 1

    def close(self) -> None:
        return None


class FakeMic:
    """A microphone that hears nothing, for the loop's sake."""

    def __init__(self) -> None:
        self.flushes = 0

    def start(self) -> None:
        return None

    def read(self, timeout: float = 0.0):
        return None

    def flush(self) -> None:
        self.flushes += 1

    def stop(self) -> None:
        return None


class FakeDispatcher:
    """Records every call, and refuses nothing - that is the desk's whole point."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict):
        self.executed.append((name, dict(args)))
        return SimpleNamespace(ok=True, refused=False, summary=f"{name} done", data=None)

    def tools_payload(self) -> list[dict]:
        return []


class CardDispatcher:
    """A dispatcher whose results have a card in them - including what must not be drawn."""

    def execute(self, name: str, args: dict):
        return SimpleNamespace(
            ok=True,
            refused=False,
            summary="The Eagle, 020 7946 0991",
            data={
                "name": "The Eagle",
                "phone": "020 7946 0991",
                # Neither key is on the phone's list, and neither belongs on a page:
                # one is raw shell output, the other is where the operator keeps things.
                "stdout": "PS C:\\Users\\karim> Get-Process",
                "path": "C:\\Users\\karim\\Documents\\taxes.xlsx",
            },
        )

    def tools_payload(self) -> list[dict]:
        return []


class FakeTray:
    """The icon, with the one method anything outside it ever calls."""

    def __init__(self) -> None:
        self.states: list[AssistantState] = []

    def on_state(self, state: AssistantState) -> None:
        self.states.append(state)


class FakeBrain:
    """One sentence and done. The real brain is tested elsewhere."""

    def __init__(self, dispatcher=None) -> None:
        self.dispatcher = dispatcher
        self.seen: list[str] = []
        self.dispatchers: list[object] = []

    def turn(self, text: str, on_sentence, should_stop):
        self.seen.append(text)
        self.dispatchers.append(self.dispatcher)
        on_sentence("At once, sir.")
        return SimpleNamespace(reply="At once, sir.", error="", cancelled=False)


# --- fixtures ------------------------------------------------------------------------
@pytest.fixture
def desk_config(config):
    """The shipped config with every face off and nothing to dial out to."""
    config.set("brain.host", "http://127.0.0.1:1")
    config.set("brain.warm_on_start", False)
    config.set("assistant.startup_greeting", False)
    config.set("ui.overlay", False)
    config.set("ui.tray", False)
    config.set("ui.window", False)
    return config


@pytest.fixture
def assistant(desk_config):
    """A real Assistant with fake hardware: the methods under test are the real ones."""
    built = Assistant(desk_config, text_mode=False)
    built.parts.speaker = FakeSpeaker()
    built.parts.mic = FakeMic()
    built.parts.dispatcher = FakeDispatcher()
    built.parts.brain = FakeBrain(built.parts.dispatcher)
    yield built
    built.stop()


@pytest.fixture
def desk_package(monkeypatch):
    """``jarvis.desk`` with both halves faked, as build_desk imports them."""
    import jarvis.desk as package

    monkeypatch.setattr(package, "DeskServer", FakeServer, raising=False)
    monkeypatch.setattr(package, "DeskWindow", FakeWindow, raising=False)
    return package


@pytest.fixture
def desk_bus_module(monkeypatch):
    """A stand-in for ``jarvis.desk.bus``, so this file tests no one else's code."""
    module = types.ModuleType("jarvis.desk.bus")

    class DeskLogHandler(logging.Handler):
        def __init__(self, bus, level: int = logging.INFO) -> None:
            super().__init__(level)
            self.bus = bus

        def emit(self, record: logging.LogRecord) -> None:
            self.bus.publish(
                "log", level=record.levelname, name=record.name, text=self.format(record)
            )

    module.DeskLogHandler = DeskLogHandler
    monkeypatch.setitem(sys.modules, "jarvis.desk.bus", module)
    return module


# --- configuration -------------------------------------------------------------------
def test_the_window_keys_are_shipped_in_the_defaults_and_in_config_yaml():
    """A key in only one of the two is a setting that silently does nothing."""
    shipped = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert set(shipped["ui"]) == set(DEFAULTS["ui"])
    for key in ("window", "window_mode", "window_size", "window_pos", "window_on_top",
                "open_window_on_start"):
        assert shipped["ui"][key] == DEFAULTS["ui"][key]


def test_the_window_is_on_and_opens_itself_by_default(config):
    """The desk app is the product; having to switch it on would be an admission."""
    assert config.get("ui.window") is True
    assert config.get("ui.window_mode") == "auto"
    assert config.get("ui.open_window_on_start") is True


# --- build_desk ----------------------------------------------------------------------
def test_build_desk_gives_nothing_back_when_the_window_is_switched_off(desk_config, desk_package):
    """--no-window must not open a socket, not even a loopback one."""
    desk_config.set("ui.window", False)

    assert build_desk(desk_config, object(), logging.getLogger("test")) == (None, None)


def test_build_desk_points_the_window_at_the_servers_url(desk_config, desk_package):
    desk_config.set("ui.window", True)

    server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert server.started is True
    assert window.url == FakeServer.url, "the window must open the door the server opened"
    assert callable(window.url_source), (
        "the window must ask for the URL at every open: the ticket in it is single-use, "
        "so a link captured at build time is refused by our own front door an hour later"
    )
    assert window.started == 1


def test_build_desk_gives_up_on_the_window_when_the_server_will_not_start(
    desk_config, desk_package, monkeypatch
):
    """No page to host means no host: a window pointed at nothing is worse than none."""
    desk_config.set("ui.window", True)
    monkeypatch.setattr(
        desk_package, "DeskServer",
        lambda *a, **k: FakeServer(*a, starts=False), raising=False,
    )

    assert build_desk(desk_config, object(), logging.getLogger("test")) == (None, None)


def test_build_desk_survives_a_desk_package_that_will_not_import(
    desk_config, desk_package, monkeypatch
):
    """Flask is optional. A machine without it keeps its ring and its voice."""
    def explode(*args, **kwargs):
        raise ImportError("no module named flask")

    desk_config.set("ui.window", True)
    monkeypatch.setattr(desk_package, "DeskServer", explode, raising=False)

    assert build_desk(desk_config, object(), logging.getLogger("test")) == (None, None)


def test_build_desk_builds_the_window_it_does_not_open(desk_config, desk_package):
    """The ticket in the URL is minted once, so the tray's window must exist from the start."""
    desk_config.set("ui.window", True)
    desk_config.set("ui.open_window_on_start", False)

    server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert server is not None
    assert window is not None and window.started == 0


def test_build_desk_hands_the_server_the_window(desk_config, desk_package):
    """The page's own title bar minimises and closes, and only the server hears the page."""
    desk_config.set("ui.window", True)

    server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert server.attached is window


def test_build_desk_survives_a_server_that_takes_no_window(desk_config, desk_package,
                                                           monkeypatch):
    """attach_window is arriving in another file; start-up must not wait for it."""
    class Older(FakeServer):
        attach_window = None

    desk_config.set("ui.window", True)
    monkeypatch.setattr(desk_package, "DeskServer", Older, raising=False)

    server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert server is not None and window is not None


def test_a_webview_window_is_created_at_start_whatever_the_setting_says(
    desk_config, desk_package, monkeypatch, caplog
):
    """pywebview can only be created on the main thread, and the tray clicks far too late.

    By then thread one is inside run_forever. open_window_on_start: false on a machine
    where WebView2 wins the chain therefore meant Open JARVIS could never open
    anything at all; opening it now is much the smaller surprise, and it is logged.
    """
    desk_config.set("ui.window", True)
    desk_config.set("ui.open_window_on_start", False)
    monkeypatch.setattr(
        desk_package, "DeskWindow",
        lambda *a, **k: FakeWindow(*a, hosts="webview", **k), raising=False,
    )

    with caplog.at_level(logging.INFO, logger="test"):
        _server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert window.started == 1
    assert any("main thread" in record.message for record in caplog.records)


def test_a_window_hosted_by_edge_still_waits_in_the_tray(desk_config, desk_package):
    """Edge and the browser are separate processes and can be started from anywhere."""
    desk_config.set("ui.window", True)
    desk_config.set("ui.open_window_on_start", False)

    _server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert window.started == 0


def test_a_host_that_cannot_be_probed_is_not_assumed_to_want_the_main_thread(
    desk_config, desk_package, monkeypatch
):
    """An unanswerable question is a no; guessing yes would open a window nobody asked for."""
    class Silent(FakeWindow):
        def would_host(self) -> str:
            raise RuntimeError("the registry is not answering")

    desk_config.set("ui.window", True)
    desk_config.set("ui.open_window_on_start", False)
    monkeypatch.setattr(desk_package, "DeskWindow", Silent, raising=False)

    _server, window = build_desk(desk_config, object(), logging.getLogger("test"))

    assert window.started == 0


def test_closing_the_window_tells_the_tray_and_says_where_it_went(
    desk_config, desk_package, caplog
):
    """Closing hides into the tray rather than quitting, which looks exactly like a crash."""
    desk_config.set("ui.window", True)
    tray = FakeTray()
    assistant = SimpleNamespace(tray=tray, state=StateBus())

    _server, window = build_desk(desk_config, assistant, logging.getLogger("test"))
    with caplog.at_level(logging.INFO, logger="test"):
        window.on_close()

    assert tray.states == [AssistantState.IDLE]
    assert any("tray" in record.message.lower() for record in caplog.records)


def test_closing_the_window_before_there_is_a_tray_is_quiet(desk_config, desk_package):
    """--no-ui leaves no tray at all, and the close hook must not care."""
    desk_config.set("ui.window", True)

    _server, window = build_desk(desk_config, SimpleNamespace(tray=None, state=None),
                                 logging.getLogger("test"))

    assert window.on_close() is None


# --- the fan -------------------------------------------------------------------------
def test_a_tool_call_reaches_both_the_overlay_and_the_window():
    """The whole reason the fan exists: one narration, two faces, no second call site."""
    hud, bus = HudModel(), FakeBus()
    fan = _HudFan(hud, bus)

    event = fan.tool_started("get_time_date")
    fan.tool_finished("get_time_date", ok=True, duration_ms=12.5)

    assert event is hud.tools[0], "tool_started must return the overlay's own event"
    assert hud.tools[0].done is True and hud.tools[0].duration_ms == 12.5
    assert bus.kinds() == ["tool", "tool"]
    assert bus.events[0][1] == {"name": "get_time_date", "phase": "start"}
    assert bus.events[1][1]["phase"] == "end"
    assert bus.events[1][1]["ok"] is True


def test_a_tool_run_through_the_dispatcher_reaches_both(desk_config):
    """The dispatcher wrapper is one of the fifteen call sites; it must not know about this."""
    hud, bus = HudModel(), FakeBus()
    dispatcher = _HudDispatcher(FakeDispatcher(), _HudFan(hud, bus))

    dispatcher.execute("get_time_date", {})

    assert [event.name for event in hud.tools] == ["get_time_date"]
    assert bus.kinds() == ["tool", "tool"]


def test_a_finished_tool_carries_the_summary_and_the_data_the_card_is_made_of():
    """Every card in § 24.6 is drawn from these two fields; without them the page is blank."""
    bus = FakeBus()
    dispatcher = _HudDispatcher(CardDispatcher(), _HudFan(HudModel(), bus))

    dispatcher.execute("find_place", {})

    payload = bus.payloads("tool")[-1]
    assert payload["phase"] == "end"
    assert payload["summary"] == "The Eagle, 020 7946 0991"
    assert payload["data"]["name"] == "The Eagle"
    assert payload["data"]["phone"] == "020 7946 0991"


def test_a_key_the_phone_would_not_send_never_reaches_the_page():
    """The desk decides what JARVIS may do, not what a web page may be handed.

    Raw PowerShell output and a path into the operator's documents are exactly what a
    card renderer would happily draw, and neither has any business leaving the process.
    """
    bus = FakeBus()
    dispatcher = _HudDispatcher(CardDispatcher(), _HudFan(HudModel(), bus))

    dispatcher.execute("run_powershell", {})

    data = bus.payloads("tool")[-1]["data"]
    assert "stdout" not in data
    assert "path" not in data
    assert set(data) == {"name", "phone"}


def test_the_overlays_own_model_is_never_offered_a_card():
    """HudModel.tool_finished has four parameters and the ring has nowhere to draw one.

    Passing the card to it would be a TypeError on every guarded tool on a machine
    with no window, which is most of the ways JARVIS gets run.
    """
    hud = HudModel()
    dispatcher = _HudDispatcher(CardDispatcher(), hud)

    dispatcher.execute("find_place", {})

    assert [event.name for event in hud.tools] == ["find_place"]
    assert hud.tools[0].done is True


def test_a_tool_with_nothing_to_show_still_reaches_the_window():
    """Most tools have no card. An absent summary is an empty one, never a missing key."""
    bus = FakeBus()
    _HudFan(None, bus).tool_finished("set_volume", ok=True, duration_ms=3.0)

    payload = bus.payloads("tool")[-1]
    assert payload["summary"] == ""
    assert payload["data"] == {}


def test_the_fan_carries_what_was_heard_and_every_sentence():
    hud, bus = HudModel(), FakeBus()
    fan = _HudFan(hud, bus)

    fan.begin_turn("what is the time")
    fan.add_reply("It is nearly eleven, sir.")

    assert hud.heard == "what is the time"
    assert hud.reply == "It is nearly eleven, sir."
    assert bus.payload("heard")["text"] == "what is the time"
    assert bus.payload("sentence")["text"] == "It is nearly eleven, sir."


def test_the_latency_reaches_the_window_with_its_breakdown():
    """The window prints the whole turn; the ring only ever had room for the total."""
    hud, bus = HudModel(), FakeBus()
    fan = _HudFan(hud, bus, marks=lambda: {"text": 310.0, "first audio": 980.0})

    fan.latency_ms = 980.0

    assert hud.latency_ms == 980.0
    assert bus.payload("latency") == {"ms": 980.0, "marks": {"text": 310.0, "first audio": 980.0}}


def test_the_state_is_not_published_by_the_fan():
    """StateBus is already telling the window; two sources for one fact start disagreeing."""
    hud, bus = HudModel(), FakeBus()
    fan = _HudFan(hud, bus)

    fan.state = AssistantState.THINKING

    assert hud.state is AssistantState.THINKING
    assert fan.state is AssistantState.THINKING
    assert bus.kinds() == []


def test_the_fan_works_with_no_overlay_at_all():
    """A window and no ring is the ordinary case on a machine without the layered overlay."""
    bus = FakeBus()
    fan = _HudFan(None, bus)

    assert fan.tool_started("get_weather") is None
    fan.note = "Say confirm"
    fan.touch()

    assert fan.note == "Say confirm"
    assert bus.payload("note")["text"] == "Say confirm"


def test_a_bus_that_has_fallen_over_never_reaches_the_turn():
    """A dead window must cost a debug line, not the answer the operator was waiting for."""
    hud = HudModel()
    fan = _HudFan(hud, AngryBus())

    fan.begin_turn("are you there")
    fan.add_reply("Always, sir.")
    fan.latency_ms = 500.0

    assert hud.reply == "Always, sir."


def test_the_fan_forwards_anything_else_to_the_overlay():
    """The renderer reads the model it was given; the fan must not hide half of it."""
    hud = HudModel()
    fan = _HudFan(hud, FakeBus())
    fan.begin_turn("hello")

    assert fan.heard == "hello"
    assert fan.is_empty() is False


# --- the six hooks: a typed turn -----------------------------------------------------
def test_a_typed_turn_is_queued_with_no_guard_on_the_dispatcher(assistant):
    """The window sits at the keyboard. A guard here would be the phone's rules at the desk."""
    assert assistant.submit_desk_turn("open spotify") is True

    turn = assistant._work.get_nowait()
    assert turn.text == "open spotify"
    assert turn.audio is None
    assert turn.wrap_dispatcher is None


def test_a_typed_turn_is_spoken_out_of_the_desk_speakers(assistant):
    """Unlike the phone, which may answer only on the phone."""
    assistant.submit_desk_turn("say something")
    turn = assistant._work.get_nowait()

    assistant._handle_remote_turn(turn)

    assert assistant.parts.speaker.queued == ["At once, sir."]
    assert assistant.parts.brain.seen == ["say something"]
    assert assistant.parts.brain.dispatchers == [assistant.parts.dispatcher], (
        "the turn must run on the assistant's own dispatcher, unwrapped"
    )


def test_a_typed_turn_is_spoken_exactly_once_when_the_phone_speaks_locally(assistant):
    """remote.speak_locally already enqueues the sentence; saying it twice is a stammer."""
    assistant.cfg.set("remote.speak_locally", True)
    assistant.submit_desk_turn("say something")

    assistant._handle_remote_turn(assistant._work.get_nowait())

    assert assistant.parts.speaker.queued == ["At once, sir."]


def test_an_empty_typed_turn_is_refused(assistant):
    assert assistant.submit_desk_turn("   ") is False
    assert assistant._work.empty()


def test_a_typed_turn_is_refused_while_he_is_busy(assistant):
    """One GPU, one Whisper, one Ollama: the window is told to wait rather than fight."""
    assert assistant.submit_desk_turn("one") is True
    assert assistant.submit_desk_turn("two") is True

    assert assistant.submit_desk_turn("three") is False


# --- the six hooks: the microphone ---------------------------------------------------
def test_arming_the_microphone_takes_the_wake_words_own_path(assistant):
    """Chime, acknowledgement and follow-up window: a button that skipped them would lie."""
    assert assistant.arm_listening() is True

    assistant._route(np.zeros(320, dtype=np.float32))

    assert assistant._mode is _Mode.LISTENING
    assert assistant.state.state is AssistantState.LISTENING
    assert assistant.parts.speaker.chimes == 1
    assert assistant.parts.speaker.said == ["Sir?"]


def test_arming_touches_nothing_on_the_windows_own_thread(assistant):
    """Waking up flushes the microphone, resets the segmenter and resets the detector.

    All three belong to the capture loop and are in use at that instant, so the
    websocket reader may only leave a flag behind - the way _stop_turn is left.
    """
    flushes = assistant.parts.mic.flushes

    assert assistant.arm_listening() is True

    assert assistant._arm_request.is_set() is True
    assert assistant._mode is _Mode.IDLE
    assert assistant.parts.mic.flushes == flushes
    assert assistant.parts.speaker.chimes == 0


def test_an_arming_request_he_has_outrun_is_dropped(assistant):
    """He woke on his own between the click and the next frame; two wakes is a stammer."""
    assistant.arm_listening()
    assistant._to_listening()
    assistant.parts.speaker.chimes = 0

    assistant._route(np.zeros(320, dtype=np.float32))

    assert assistant.parts.speaker.chimes == 0
    assert assistant._mode is _Mode.LISTENING
    assert assistant._arm_request.is_set() is False


def test_arming_is_refused_while_he_is_paused(assistant):
    assistant.pause()

    assert assistant.arm_listening() is False
    assert assistant._arm_request.is_set() is False
    assert assistant._mode is _Mode.IDLE


def test_arming_is_refused_when_he_is_already_listening(assistant):
    assistant.arm_listening()
    assistant._route(np.zeros(320, dtype=np.float32))
    assistant.parts.speaker.chimes = 0

    assert assistant.arm_listening() is False
    assert assistant.parts.speaker.chimes == 0, "a second chime would be a second wake"


# --- the six hooks: stopping ---------------------------------------------------------
def test_aborting_a_turn_stops_the_mouth_and_the_thinking(assistant):
    assistant._stop_turn.clear()

    assert assistant.abort_turn() is None
    assert assistant.parts.speaker.stops == 1
    assert assistant._stop_turn.is_set() is True


def test_aborting_survives_a_speaker_that_will_not_stop(assistant):
    """Stop is the one thing that must work when everything else has gone wrong."""
    def refuse() -> None:
        raise RuntimeError("the audio device is gone")

    assistant.parts.speaker.stop = refuse

    assistant.abort_turn()

    assert assistant._stop_turn.is_set() is True


# --- the six hooks: confirmation -----------------------------------------------------
def test_a_confirmation_from_the_window_answers_the_guarded_tool(assistant):
    """The click has to arrive as a word, because the answer is read as one."""
    token, ready = assistant._open_confirmation()

    assert assistant.answer_confirmation(True) is True
    assert ready.is_set() is True

    from jarvis.tools.safety import is_cancellation, is_confirmation

    reply = assistant._close_confirmation(token, _Mode.WORKING)
    assert is_confirmation(reply) is True
    assert is_cancellation(reply) is False


def test_cancelling_from_the_window_reads_as_a_refusal(assistant):
    token, _ready = assistant._open_confirmation()

    assert assistant.answer_confirmation(False) is True

    from jarvis.tools.safety import is_cancellation, is_confirmation

    reply = assistant._close_confirmation(token, _Mode.WORKING)
    assert is_confirmation(reply) is False
    assert is_cancellation(reply) is True


def test_a_confirmation_nobody_asked_for_is_refused(assistant):
    """A click with nothing pending must not be remembered and spent on the next tool."""
    assert assistant._mode is _Mode.IDLE

    assert assistant.answer_confirmation(True) is False
    assert assistant._confirm_id == 0
    assert assistant._confirm_reply is None
    assert assistant._confirm_ready is None


def test_an_answer_meant_for_a_finished_confirmation_is_not_kept(assistant):
    """The tool timed out and stopped listening; the click that follows belongs to nobody."""
    token, _ready = assistant._open_confirmation()
    assistant._close_confirmation(token, _Mode.WORKING)

    assert assistant.answer_confirmation(True) is False
    assert assistant._settle_confirmation("confirm", source="the microphone") is False


def test_the_spoken_answer_wins_when_it_lands_first(assistant):
    """He said no and then reached for the button: the first answer is the one he meant."""
    token, _ready = assistant._open_confirmation()
    assistant._settle_confirmation("cancel", source="the microphone")

    assert assistant.answer_confirmation(True) is False
    assert assistant._close_confirmation(token, _Mode.WORKING) == "cancel"


def test_the_window_wins_when_it_lands_first(assistant):
    """And the same race the other way round, because the microphone is still open."""
    token, _ready = assistant._open_confirmation()
    assistant.answer_confirmation(False)

    assert assistant._settle_confirmation("confirm", source="the microphone") is False
    assert assistant._close_confirmation(token, _Mode.WORKING) == "cancel"


def test_a_guarded_tool_in_a_desk_turn_gives_the_ear_back(assistant, monkeypatch):
    """A desk turn is not a microphone turn, and nothing downstream puts the mode back.

    Ending every confirmation in WORKING left _route watching for barge-in for the
    rest of the session: no wake word, no follow-up window, and the window's own
    Listen button refused until JARVIS was restarted.
    """
    assistant.confirmation_timeout = 0.01
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)
    assert assistant._mode is _Mode.IDLE

    assert assistant._tool_confirm("I am about to run PowerShell, sir.") is False

    assert assistant._mode is _Mode.IDLE
    assert assistant.state.state is AssistantState.IDLE
    assert assistant.arm_listening() is True, "the Listen button must still be alive"


def test_a_guarded_tool_in_a_spoken_turn_still_ends_in_working(assistant, monkeypatch):
    """The voice path is the one this always got right, and it must go on getting it right."""
    assistant.confirmation_timeout = 0.01
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)
    assistant._mode = _Mode.WORKING

    assistant._tool_confirm("I am about to delete that file, sir.")

    assert assistant._mode is _Mode.WORKING
    assert assistant.state.state is AssistantState.THINKING


def test_a_guarded_tool_asked_mid_conversation_leaves_him_listening(assistant, monkeypatch):
    """A timer firing inside the follow-up window is the third caller, and the third mode."""
    assistant.confirmation_timeout = 0.01
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)
    assistant._to_listening()

    assistant._tool_confirm("I am about to close that, sir.")

    assert assistant._mode is _Mode.LISTENING
    assert assistant.state.state is AssistantState.LISTENING


def test_one_confirmation_cannot_grant_two_guarded_tools(assistant, monkeypatch):
    """A routine walks its calls on the socket's reader thread while the worker waits.

    With one unnamed slot between them, a single "confirm" ran both tools: the one the
    operator was answering, and one he had not been told about. Each question now has
    an id of its own and queues behind the last, so an answer can only spend itself once.
    """
    assistant.confirmation_timeout = 1.0
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)
    granted: list[bool] = []
    guard = threading.Lock()

    def ask() -> None:
        answer = assistant._tool_confirm("I am about to run PowerShell, sir.")
        with guard:
            granted.append(answer)

    askers = [threading.Thread(target=ask, name=f"guarded-{n}") for n in range(2)]
    for asker in askers:
        asker.start()
    deadline = time.monotonic() + 5.0
    while assistant._confirm_id == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    # Both tools are in flight; the second is given every chance to reach a wait of
    # its own beside the first before the one answer is given. Well inside the
    # timeout, so the question that is genuinely pending is still pending.
    time.sleep(0.15)

    assert assistant.answer_confirmation(True) is True, "one question was waiting"

    for asker in askers:
        asker.join(timeout=5.0)
        assert not asker.is_alive(), "a guarded tool waited for ever"

    assert sorted(granted) == [False, True], "the other must reach its own timeout"


def test_a_second_guarded_tool_waits_rather_than_replacing_the_first(assistant, monkeypatch):
    """Two confirmation bars at once is not a state the window can express.

    The gate is what makes the second tool queue. Without it the second call mints a
    slot over the first, and the first then waits out a question that nobody - not the
    window, not the microphone - is able to answer any more.
    """
    assistant.confirmation_timeout = 0.05
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)
    pending, most, guard = 0, 0, threading.Lock()
    really_open, really_close = assistant._open_confirmation, assistant._close_confirmation

    def opened():
        nonlocal pending, most
        with guard:
            pending += 1
            most = max(most, pending)
        return really_open()

    def closed(token, previous):
        nonlocal pending
        with guard:
            pending -= 1
        return really_close(token, previous)

    monkeypatch.setattr(assistant, "_open_confirmation", opened)
    monkeypatch.setattr(assistant, "_close_confirmation", closed)

    askers = [
        threading.Thread(target=assistant._tool_confirm, args=("I am about to restart, sir.",))
        for _ in range(3)
    ]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(timeout=5.0)
        assert not asker.is_alive(), "a guarded tool waited for ever"

    assert most == 1, "a second question was opened while the first was still pending"
    assert assistant._confirm_id == 0, "the slot is retired with the question"


# --- the six hooks: routines ---------------------------------------------------------
def test_a_routine_runs_with_the_desks_rights(assistant):
    """The phone's guard would refuse a GUARDED tool here; at the keyboard nothing does."""
    assistant.cfg.set("remote.allow_guarded", False)
    assistant.cfg.set("remote.routines", [
        {"name": "goodnight",
         "calls": [{"tool": "run_powershell", "args": {"command": "Stop-Process notepad"}}],
         "say": "Goodnight, sir."},
    ])

    said = assistant.run_routine("goodnight")

    assert said == "Goodnight, sir."
    assert assistant.parts.dispatcher.executed == [
        ("run_powershell", {"command": "Stop-Process notepad"})
    ]
    assert assistant.parts.speaker.said == ["Goodnight, sir."]


def test_an_unknown_routine_says_nothing_and_does_nothing(assistant):
    assistant.cfg.set("remote.routines", [])

    assert assistant.run_routine("goodnight") == ""
    assert assistant.parts.dispatcher.executed == []


def test_a_routine_with_no_name_is_not_a_routine(assistant):
    assert assistant.run_routine("") == ""


# --- the six hooks: settings ---------------------------------------------------------
def test_an_allowlisted_setting_is_saved_and_takes_effect_now(assistant):
    """A slider that only works after a restart is a slider nobody trusts."""
    assert assistant.apply_setting("tts.speed", 1.2) is True
    assert assistant.apply_setting("audio.chime_volume", 0.1) is True

    assert assistant.parts.speaker._engine.speed == 1.2
    assert assistant.parts.speaker.chime_volume == 0.1

    from jarvis.config import Config

    reloaded = Config.load(assistant.cfg.path)
    assert reloaded.get("tts.speed") == 1.2
    assert reloaded.get("audio.chime_volume") == 0.1


def test_the_wake_words_sensitivity_is_retuned_in_place(assistant):
    assistant.apply_setting("wake.sensitivity", 0.75)

    assert assistant.parts.wake.sensitivity == 0.75


def test_a_setting_off_the_list_is_refused_and_not_written(assistant):
    """The window may not choose the model, whatever frame it sends."""
    before = assistant.cfg.path.read_text(encoding="utf-8")

    assert assistant.apply_setting("brain.model", "something-else") is False
    assert assistant.cfg.get("brain.model") != "something-else"
    assert assistant.cfg.path.read_text(encoding="utf-8") == before


def test_a_setting_that_cannot_be_applied_is_still_saved(assistant):
    """The config is the record; this session merely misses out until the next one."""
    def refuse(value):
        raise RuntimeError("no engine here")

    assistant._set_speed = refuse

    assert assistant.apply_setting("tts.speed", 0.9) is True
    assert assistant.cfg.get("tts.speed") == 0.9


def test_a_value_the_window_would_never_send_is_refused(assistant):
    """The key allowlist says which slider; only the value table says what a slider means.

    Checking the key and waving the value through meant a frame that reached
    apply_setting by any other route could put a word, a list or NaN into config.yaml
    and JARVIS would read it back on the next start.
    """
    before = assistant.cfg.get("tts.speed")

    assert assistant.apply_setting("tts.speed", "quickly") is False
    assert assistant.apply_setting("wake.sensitivity", float("nan")) is False
    assert assistant.apply_setting("assistant.brief_mode", [1, 2]) is False

    assert assistant.cfg.get("tts.speed") == before


def test_a_slider_pushed_past_its_end_is_pinned_rather_than_refused(assistant):
    """The table clamps on purpose: 40 means "as fast as you go", not "fail"."""
    assert assistant.apply_setting("tts.speed", 40) is True

    assert assistant.cfg.get("tts.speed") == 1.6
    assert assistant.parts.speaker._engine.speed == 1.6


def test_a_config_that_could_not_be_written_is_not_reported_as_saved(assistant):
    """Config.save returns False for a read-only file, and the window is owed the truth.

    Answering the ack anyway is how an operator ends up moving the same slider three
    times and then restarting to find out none of it took.
    """
    assistant.cfg.save = lambda: False

    assert assistant.apply_setting("tts.speed", 1.1) is False


# --- the main thread -----------------------------------------------------------------
def test_the_window_takes_the_main_thread_and_the_tkinter_ring_stands_down(assistant, caplog):
    """Both want thread one. The window is the application; the ring is an ornament."""
    window = FakeWindow(needs_main_thread=True)
    overlay = FakeOverlay(needs_main_thread=True)
    assistant.window, assistant.overlay = window, overlay

    with caplog.at_level(logging.INFO, logger="jarvis.assistant"):
        assistant.run_forever()

    assert window.ran == 1
    assert overlay.ran == 0 and overlay.stopped == 1
    assert assistant.overlay is None
    assert any("ring" in record.message.lower() for record in caplog.records)


def test_the_layered_ring_and_the_window_both_live(assistant):
    """On Windows the overlay runs its own pump, so nothing has to be given up."""
    window = FakeWindow(needs_main_thread=True)
    overlay = FakeOverlay(needs_main_thread=False)
    assistant.window, assistant.overlay = window, overlay

    assistant.run_forever()

    assert window.ran == 1
    assert overlay.stopped == 0
    assert assistant.overlay is overlay


def test_the_ring_keeps_the_main_thread_when_the_window_does_not_want_it(assistant):
    """Edge and the browser are separate processes; only pywebview has a loop here."""
    window = FakeWindow(needs_main_thread=False)
    overlay = FakeOverlay(needs_main_thread=True)
    assistant.window, assistant.overlay = window, overlay

    assistant.run_forever()

    assert overlay.ran == 1
    assert window.ran == 0
    assert overlay.stopped == 0


def test_nobody_wants_the_main_thread_so_the_loop_simply_waits(assistant):
    """No window, no ring, or two that run themselves: run_forever waits on the flag."""
    window = FakeWindow(needs_main_thread=False)
    overlay = FakeOverlay(needs_main_thread=False)
    assistant.window, assistant.overlay = window, overlay

    assistant.run_forever()  # _running was never set, so this returns at once

    assert window.ran == 0 and overlay.ran == 0


# --- start(), end to end -------------------------------------------------------------
def test_start_installs_the_fan_and_subscribes_the_window(
    desk_config, desk_package, desk_bus_module, monkeypatch, caplog
):
    """The state and the log reach the drawer without a single new call site in the loop."""
    desk_config.set("ui.window", True)
    overlay = FakeOverlay(needs_main_thread=False)
    monkeypatch.setattr(
        "jarvis.core.wiring.build_face", lambda *a, **k: (overlay, None)
    )

    built = Assistant(desk_config, text_mode=False)
    built.parts.speaker = FakeSpeaker()
    built.parts.mic = FakeMic()
    try:
        # setup_logging puts the jarvis logger at INFO; nothing has run it here.
        with caplog.at_level(logging.INFO, logger="jarvis"):
            built.start()
            bus = built.desk.bus

            built.state.set(AssistantState.THINKING)
            built.log.info("Something worth reading in the drawer.")
            built._hud.add_reply("Quite so, sir.")
    finally:
        built.stop()

    assert isinstance(built._hud, _HudFan)
    assert bus.payload("state")["state"] == "thinking"
    assert any("drawer" in line["text"] for line in bus.payloads("log"))
    assert bus.payload("sentence")["text"] == "Quite so, sir."
    assert overlay.hud.reply == "Quite so, sir.", "the ring must still be narrated to"


def test_stopping_lets_go_of_the_log_and_the_state_bus(
    desk_config, desk_package, desk_bus_module, monkeypatch
):
    """A handler left behind would publish into a closed bus for the rest of the process."""
    desk_config.set("ui.window", True)
    monkeypatch.setattr("jarvis.core.wiring.build_face", lambda *a, **k: (None, None))

    built = Assistant(desk_config, text_mode=False)
    built.parts.speaker = FakeSpeaker()
    built.parts.mic = FakeMic()
    built.start()
    handler = built._desk_log_handler
    observers = built.state.observer_count
    built.stop()

    assert handler is not None
    assert handler not in logging.getLogger("jarvis").handlers
    assert built.state.observer_count < observers


def test_stopping_closes_the_window_s_bus(
    desk_config, desk_package, desk_bus_module, monkeypatch
):
    """Nothing ever closed it, so every subscriber queue stayed open past shutdown.

    The page reads an open queue as a live JARVIS and reconnects to it for the rest of
    the machine's day; closing is also what releases the replay buffer.
    """
    desk_config.set("ui.window", True)
    monkeypatch.setattr("jarvis.core.wiring.build_face", lambda *a, **k: (None, None))

    built = Assistant(desk_config, text_mode=False)
    built.parts.speaker = FakeSpeaker()
    built.parts.mic = FakeMic()
    built.start()
    bus = built.desk.bus
    built.stop()

    assert bus.closed is True
    assert built._desk_bus is None


def test_the_window_is_told_about_a_guarded_tool_and_its_answer(assistant, monkeypatch):
    """The confirmation bar and the spoken window open and close together."""
    bus = FakeBus()
    assistant._desk_bus = bus
    assistant.confirmation_timeout = 0.01
    monkeypatch.setattr(assistant.parts.segmenter, "reset", lambda: None)

    granted = assistant._tool_confirm("I am about to run PowerShell, sir.")

    assert granted is False, "nobody answered, so nothing happens"
    assert bus.payload("confirm")["announcement"] == "I am about to run PowerShell, sir."
    assert bus.payload("confirm")["seconds"] == 0.01
    assert bus.payload("confirm_done")["granted"] is False


def test_the_tray_can_open_a_window_that_was_never_shown(assistant):
    """open_window_on_start: false leaves a window built but not hosted anywhere."""
    assistant.window = FakeWindow(shows=False)

    assistant.show_window()

    assert assistant.window.shown == 1
    assert assistant.window.started == 1


def test_showing_a_window_that_does_not_exist_is_quiet(assistant):
    assistant.window = None

    assert assistant.show_window() is None


# --- the tray ------------------------------------------------------------------------
class FakePystray:
    """Just enough pystray to read a menu off the tray without a desktop."""

    class MenuItem:
        def __init__(self, text, action, **kwargs) -> None:
            self.text = text
            self.action = action
            self.kwargs = kwargs

    class Menu:
        def __init__(self, *items) -> None:
            self.items = items


def test_the_tray_offers_the_window_and_still_pauses_and_quits():
    from jarvis.core.state import StateBus

    opened: list[bool] = []
    tray = Tray(StateBus(), on_show_window=lambda: opened.append(True))

    menu = tray._build_menu(FakePystray)
    labels = [item.text for item in menu.items]
    menu.items[0].action(None, None)

    # "the console" is the word this whole feature exists to stop using.
    assert labels[0] == "Open JARVIS"
    assert labels[-1] == "Quit"
    assert len(labels) == 3
    assert menu.items[0].kwargs.get("default") is True
    assert opened == [True]


def test_the_tray_offers_no_window_when_there_is_none():
    """A menu item that cannot do anything is worse than a menu item that is not there."""
    from jarvis.core.state import StateBus

    menu = Tray(StateBus())._build_menu(FakePystray)

    assert len(menu.items) == 2
    assert menu.items[-1].text == "Quit"


# --- the command line ----------------------------------------------------------------
def _fake_assistant(monkeypatch) -> list:
    """Replace the Assistant main() imports, and hand back the configs it was built with."""
    seen: list = []

    class Fake:
        def __init__(self, cfg, *, text_mode: bool = False) -> None:
            seen.append(cfg)

        def start(self) -> None:
            return None

        def run_forever(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr("jarvis.core.assistant.Assistant", Fake)
    return seen


def test_no_window_switches_the_window_off_for_one_run(config, monkeypatch):
    from jarvis.__main__ import main

    seen = _fake_assistant(monkeypatch)

    assert main(["--config", str(config.path), "--no-window"]) == 0
    assert seen[-1].get("ui.window") is False
    assert seen[-1].get("ui.overlay") is True


def test_window_overrides_a_config_that_switched_it_off(config, monkeypatch):
    from jarvis.__main__ import main

    config.set("ui.window", False)
    config.save()
    seen = _fake_assistant(monkeypatch)

    assert main(["--config", str(config.path), "--window"]) == 0
    assert seen[-1].get("ui.window") is True


def test_no_ui_still_takes_everything_with_a_face(config, monkeypatch):
    """Including the window, and including when --window was asked for as well."""
    from jarvis.__main__ import main

    seen = _fake_assistant(monkeypatch)

    assert main(["--config", str(config.path), "--window", "--no-ui"]) == 0
    assert seen[-1].get("ui.window") is False
    assert seen[-1].get("ui.overlay") is False
    assert seen[-1].get("ui.tray") is False


def test_the_banner_is_not_printed_when_there_is_no_console(config, monkeypatch, capsys):
    """Under pythonw it would go nowhere; under a log file it would be noise."""
    import jarvis.__main__ as entry

    _fake_assistant(monkeypatch)
    monkeypatch.setattr(entry, "_install_null_streams", lambda: False)

    entry.main(["--config", str(config.path)])

    assert "J.A.R.V.I.S." not in capsys.readouterr().out


def test_the_banner_is_printed_when_there_is_one(config, monkeypatch, capsys):
    import jarvis.__main__ as entry

    _fake_assistant(monkeypatch)

    entry.main(["--config", str(config.path)])

    assert "J.A.R.V.I.S." in capsys.readouterr().out


def test_a_missing_console_gets_a_sink_before_anything_writes(monkeypatch):
    """pythonw leaves both streams as None, and a StreamHandler over None raises."""
    import jarvis.__main__ as entry

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    try:
        assert entry._install_null_streams() is False
        assert sys.stdout is not None and sys.stderr is not None

        handler = logging.StreamHandler(sys.stdout)
        handler.emit(logging.LogRecord("jarvis", logging.INFO, __file__, 1, "quiet", (), None))
        print("into the void")
    finally:
        for sink in entry._SINKS:
            sink.close()
        entry._SINKS.clear()


def test_a_console_is_left_exactly_as_it_was(capsys):
    import jarvis.__main__ as entry

    assert entry._install_null_streams() is True
    assert entry._SINKS == []
