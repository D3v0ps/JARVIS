"""What turns the phone page from a terminal into an app.

Three server-side additions: a typed turn for when you cannot talk, a report of every
tool call so the page can draw a card, and a status endpoint for the glanceable
telemetry strip. All faked, no network, no Windows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jarvis.remote.server import RemoteServer
from jarvis.remote.session import RemoteGuard, ToolReporter
from jarvis.tools.base import ToolResult

from tests.test_remote import (  # reuse the doubles rather than duplicating them
    FakeAssistant,
    FakeDispatcher,
    FakeSocket,
    _in_request,
    _token,
    pair_client,
)


# pytest does not share fixtures between test modules; these mirror test_remote.py.
@pytest.fixture
def remote_config(config, tmp_path):
    config.set(
        "remote",
        {
            "enabled": True, "host": "127.0.0.1", "port": 0, "allow_guarded": False,
            "secret_file": str(tmp_path / "remote-secret.key"), "session_ttl_hours": 24,
            "code_ttl": 900, "max_attempts": 3, "attempt_window": 300,
            "max_audio_seconds": 5, "turn_timeout": 5, "heartbeat": 5, "routines": [],
        },
    )
    return config


@pytest.fixture
def assistant():
    return FakeAssistant()


@pytest.fixture
def server(remote_config, assistant):
    return RemoteServer(remote_config, assistant)


@pytest.fixture
def client(server):
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    app = server.create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --- typed turns ------------------------------------------------------------------
def test_a_typed_say_frame_runs_a_turn_without_audio(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    frame = json.dumps({"type": "say", "text": "what time is it"})

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(FakeSocket([frame]))

    assert assistant.turns, "the say frame must reach the assistant"
    turn = assistant.turns[0]
    assert turn.text == "what time is it"
    assert turn.audio is None


def test_an_empty_say_frame_is_rejected_not_queued(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    socket = FakeSocket([json.dumps({"type": "say", "text": "   "})])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(socket)

    assert not assistant.turns
    assert "error" in socket.kinds()


def test_a_typed_turn_is_capped_so_a_paste_cannot_flood_the_model(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    frame = json.dumps({"type": "say", "text": "x" * 10_000})

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(FakeSocket([frame]))

    assert len(assistant.turns[0].text) <= 2000


# --- tool events -------------------------------------------------------------------
def test_every_tool_call_is_reported_to_the_phone():
    events: list[dict] = []
    reporter = ToolReporter(FakeDispatcher(), events.append)

    result = reporter.execute("get_time_date", {})

    assert result.ok is True
    assert events and events[0]["type"] == "tool"
    assert events[0]["name"] == "get_time_date"
    assert events[0]["ok"] is True
    assert events[0]["refused"] is False
    assert "ms" in events[0]


def test_a_refusal_by_the_guard_is_still_reported(remote_config):
    """The reporter sits outside the guard on purpose: you should see the 'no'."""
    from jarvis.tools import registry

    registry.load_all()  # the guard looks the tool up to learn its tier
    events: list[dict] = []
    guard = RemoteGuard(FakeDispatcher(), remote_config, allow_guarded=False)
    reporter = ToolReporter(guard, events.append)

    result = reporter.execute("run_powershell", {"command": "Get-Date"})

    assert result.refused is True
    assert events[0]["refused"] is True
    assert events[0]["ok"] is False


def test_only_safe_data_keys_travel_to_the_phone():
    """File paths and raw command output belong in the log, not on a phone screen."""
    class Leaky:
        def execute(self, name, args):
            return ToolResult(ok=True, summary="Done.", data={
                "phone": "+46812345678",          # fine
                "name": "Folktandvården",          # fine
                "stdout": "C:\\Users\\karre\\...", # not fine
                "path": "C:\\Users\\karre\\file",  # a file path: never
                "nested": {"secret": 1},           # not a plain value
            })
        def tools_payload(self):
            return []

    events: list[dict] = []
    ToolReporter(Leaky(), events.append).execute("find_business", {})

    data = events[0]["data"]
    assert data["phone"] == "+46812345678"
    assert data["name"] == "Folktandvården"
    assert "stdout" not in data
    assert "path" not in data
    assert "nested" not in data


def test_tool_events_reach_the_socket_in_order(server, assistant):
    """End to end through the server: the wrap installs the reporter over the guard."""
    pytest.importorskip("flask")
    app = server.create_app()

    class ToolUsingAssistant(FakeAssistant):
        def _run(self, turn):
            dispatcher = turn.wrap_dispatcher(self.parts.dispatcher)
            self.wrapped.append(dispatcher)
            dispatcher.execute("get_time_date", {})
            turn.on_sentence("It's nine, sir.")
            turn.finish("It's nine, sir.", "")

    tool_assistant = ToolUsingAssistant()
    server.assistant = tool_assistant
    socket = FakeSocket([json.dumps({"type": "say", "text": "what time is it"})])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(socket)

    kinds = socket.kinds()
    assert "tool" in kinds
    assert kinds.index("tool") < kinds.index("sentence"), "the card comes before the words"


# --- status -----------------------------------------------------------------------
def test_status_needs_a_paired_phone(client):
    response = client.get("/api/status")
    assert response.status_code == 401


def test_status_reports_state_and_timers(client, server):
    pair_client(client, server)
    server.assistant.parts.scheduler = SimpleNamespace(
        pending=lambda: [SimpleNamespace(id="t1", label="pasta", text="", kind="timer", due=1.0)]
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.get_json()
    assert body["state"] in ("idle", "listening", "thinking", "speaking", "paused")
    assert body["timers"] == [{"id": "t1", "label": "pasta", "due": 1.0, "kind": "timer"}]


def test_status_survives_a_broken_scheduler(client, server):
    """A glance at the phone must never throw because psutil or the scheduler did."""
    pair_client(client, server)
    server.assistant.parts.scheduler = SimpleNamespace(
        pending=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.get_json()["timers"] == []
