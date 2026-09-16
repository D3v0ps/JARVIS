"""What turns the phone page from a terminal into an app.

Three server-side additions: a typed turn for when you cannot talk, a report of every
tool call so the page can draw a card, and a status endpoint for the glanceable
telemetry strip. All faked, no network, no Windows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis
from jarvis.remote import server as server_module
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


# ----------------------------------------------------------------------------------
# One product, two faces
# ----------------------------------------------------------------------------------
# The user holds the phone in front of the desk window, so a token that means one
# thing there and another here is a visible seam. Nothing but a test keeps the two
# pages together: they are deliberately not factored into a shared module (§24.6),
# which is exactly why the agreement has to be asserted rather than assumed.
def phone_page() -> str:
    """The page as the server serves it, not a copy of it."""
    return (Path(server_module._STATIC_DIR) / "index.html").read_text(encoding="utf-8")


def desk_page() -> str:
    """The desk window's page, read straight off the package."""
    root = Path(jarvis.__file__).resolve().parent
    return (root / "desk" / "static" / "desk.html").read_text(encoding="utf-8")


def wash(html: str) -> str:
    """The ``body::before`` rule - the tinted layer that colours the whole room."""
    match = re.search(r"body::before\s*\{(.*?)\}", html, re.S)
    assert match, "the page no longer washes its background; this test needs rewriting"
    return match.group(1)


def test_the_phone_washes_its_background_with_the_state_colour():
    """The wash is the single biggest thing making a face feel alive, and it is free.

    The phone shipped with a fixed blue gradient painted into ``body``: pretty, but
    dead - it said the same thing while he listened, worked and spoke.
    """
    html = phone_page()
    layer = wash(html)

    assert "var(--accent)" in layer, "the wash must follow the state, not a fixed blue"
    assert "pointer-events: none" in layer, "the layer sits over the page; it may not eat taps"
    assert re.search(r"body\s*\{[^}]*background:\s*var\(--bg\);", html), \
        "the body carries the flat token now; the colour comes from the layer above it"
    assert "rgba(41,182,246,0.10)" not in html, "the old static gradient is still there"


def test_the_wash_brightens_while_he_listens_and_dims_while_he_is_paused():
    """A room light that never changes brightness is wallpaper, not a state.

    The desk gives the layer three opacities; a phone that only changed hue would
    look like a colour scheme rather than an assistant paying attention.
    """
    html = phone_page()
    brighter = re.search(
        r'body\[data-state="thinking"\]::before,\s*'
        r'body\[data-state="listening"\]::before\s*\{\s*opacity:\s*([\d.]+)', html)
    dimmer = re.search(
        r'body\[data-state="paused"\]::before\s*\{\s*opacity:\s*([\d.]+)', html)
    resting = re.search(r"opacity:\s*([\d.]+); transition", wash(html))

    assert brighter and dimmer and resting, "the three brightnesses are the state"
    assert float(brighter.group(1)) > float(resting.group(1)) > float(dimmer.group(1))


def test_both_faces_light_the_room_the_same_way():
    """Ported from the desk verbatim: same layer, same token, same reason."""
    for layer in (wash(phone_page()), wash(desk_page())):
        assert "position: fixed" in layer
        assert "var(--accent)" in layer
        assert "pointer-events: none" in layer


def test_the_phones_gpu_stat_reports_utilisation_like_the_desks_gpu_tile():
    """Two faces showed two different numbers under the same word: GPU.

    The desk has room for a tile each and labels them GPU and GPU degC; the phone has
    room for one at 390 px, and utilisation is the number that moves.
    """
    html = phone_page()
    call = re.search(
        r"drawStat\(ui\.stats\.gpu,(?P<value>.*?),\s*\{(?P<flags>.*?)\}\)", html, re.S
    )
    assert call, "the GPU stat is no longer drawn; this test needs rewriting"

    assert "util_percent" in call.group("value"), "the phone's GPU stat is utilisation"
    assert "temperature_c" not in call.group("value"), "the heat is the warning, not the value"
    assert "hot:" in call.group("flags"), "a hot GPU must still colour the stat"
    assert "temp >= 80" in html, "eighty degrees is the trigger on both faces"


def test_a_hot_gpu_is_never_reported_in_the_colour_of_good_news():
    """Both flags live on one tile here, and ``.live`` wins the cascade over ``.hot``.

    A GPU at eighty-four degrees is also busy, so without this the warning would be
    painted cyan and read as merely lively.
    """
    call = re.search(
        r"drawStat\(ui\.stats\.gpu,.*?\{(?P<flags>.*?)\}\)", phone_page(), re.S
    )

    assert re.search(r"live:\s*!hot", call.group("flags")), \
        "live must stand down while hot is set"


def test_both_faces_call_the_gpu_by_the_same_name():
    """The label is what ties the phone's one tile to the desk's pair of them."""
    assert ">GPU<" in phone_page()
    assert ">GPU<" in desk_page()


def test_the_phone_wears_the_dotted_wordmark():
    """J.A.R.V.I.S. is the mark; JARVIS is a variable name.

    Measured in Chromium at 390 px: 133 px wide against 306 px of header before the
    link, so the dotted form fits with room to spare.
    """
    html = phone_page()

    assert "<b>J.A.R.V.I.S.</b>" in html
    assert "<b>JARVIS</b>" not in html, "the plain form is the desk's old mistake"


def test_both_faces_wear_the_same_wordmark():
    """Held side by side, two spellings of the name read as two products."""
    assert "J.A.R.V.I.S." in phone_page()
    assert "J.A.R.V.I.S." in desk_page()


def test_a_live_stat_is_listening_cyan_on_the_phone():
    """``.stat.live`` and the desk's ``.tile.live`` mean one thing: this is moving.

    The phone shipped first and cyan is the token it shipped with, so the desk's
    ``.tile.live`` comes to it rather than the other way round.
    """
    assert re.search(r"\.stat\.live \.v \{ color: var\(--listening\); \}", phone_page())
    assert re.search(r"\.tile\.live \.v \{ color: var\(--listening\); \}", desk_page()), (
        "the desk must use the phone's token, or the two faces drift apart again"
    )


def test_the_phone_still_says_call_on_a_business_card():
    """A button on a phone that dials a number says Call. It shipped saying Call."""
    html = phone_page()

    assert '"Call " + d.phone' in html
    assert "Ring " not in html


def test_every_file_the_phone_asks_for_has_a_route(client):
    """The page and the manifest name their own assets; each one must be servable.

    Three PNG icons shipped referenced-but-unrouted and answered 404, so Add to Home
    Screen - step three of the instructions - produced an app with a blank square for
    an icon. This walks the references rather than naming them, so the next asset added
    to the page is caught the day it is added.
    """
    static = Path(server_module._STATIC_DIR)
    manifest = (static / "manifest.webmanifest").read_text(encoding="utf-8")

    wanted: set[str] = set()
    for text in (phone_page(), manifest):
        wanted.update(re.findall(r"[\"'(]/?([\w.-]+\.(?:png|svg|webmanifest|css|js|ico))", text))
    assert wanted, "the page references no assets at all, which cannot be right"

    ours = sorted(name for name in wanted if (static / name).is_file())
    assert ours, "none of the referenced assets ship with us; the regex has stopped working"

    missing = [name for name in ours if client.get("/" + name).status_code != 200]
    assert missing == [], f"referenced by the page but not served: {missing}"


def test_an_icon_arrives_as_a_png_and_not_as_mangled_text(client):
    """They were read with read_text once. A PNG does not survive that."""
    response = client.get("/icon-192.png")

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_the_page_is_served_with_one_charset_not_two(client):
    """Flask appends its own to a textual mimetype; passing one produced a doubled header."""
    response = client.get("/")

    assert response.headers["Content-Type"].count("charset") == 1
