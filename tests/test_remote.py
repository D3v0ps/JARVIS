"""Tests for the phone satellite: pairing, the wire format, streaming and guarding.

No network, no Windows, no hardware and no real microphone. The assistant is faked,
the WebSockets are faked objects with ``send``/``receive``/``close``, and the one test
that starts a real server binds an ephemeral port on the loopback interface.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.core.state import AssistantState, StateBus
from jarvis.remote import server as server_module
from jarvis.remote.server import MISSING_HOOK, RemoteServer
from jarvis.remote.session import (
    COOKIE_NAME,
    REMOTE_REFUSAL,
    RemoteGuard,
    RemoteTurn,
    SessionStore,
    decode_pcm16,
    load_or_create_secret,
    load_routines,
)
from jarvis.tools import registry, safety
from jarvis.tools.base import Tier, ToolResult

ROOT = Path(__file__).resolve().parent.parent

# A sentinel the fake socket turns into "receive() timed out".
TIMEOUT = object()


# ----------------------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------------------
class FakeSocket:
    """A WebSocket with a script of incoming messages and a record of outgoing ones."""

    def __init__(self, incoming: list | None = None, *, allow_sends: int | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[str] = []
        self.closed = False
        self.allow_sends = allow_sends

    def receive(self, timeout: float | None = None):
        if not self.incoming:
            raise ConnectionError("the phone went away")
        item = self.incoming.pop(0)
        return None if item is TIMEOUT else item

    def send(self, data) -> None:
        if self.closed:
            raise ConnectionError("socket is closed")
        if self.allow_sends is not None and len(self.sent) >= self.allow_sends:
            raise ConnectionError("socket broke")
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True

    @property
    def events(self) -> list[dict]:
        return [json.loads(item) for item in self.sent]

    def kinds(self, skip_pings: bool = True) -> list[str]:
        return [
            event["type"]
            for event in self.events
            if not (skip_pings and event["type"] in ("ping", "pong"))
        ]


class FakeDispatcher:
    """Records what was executed and always succeeds."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> ToolResult:
        self.executed.append((name, dict(args or {})))
        return ToolResult(ok=True, summary=f"Ran {name}.")

    def tools_payload(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in ("get_time_date", "run_powershell")
        ]


class FakeAssistant:
    """Stands in for jarvis.core.assistant.Assistant, hook and all."""

    def __init__(
        self,
        *,
        sentences: list[str] | None = None,
        transcript: str = "what is the weather",
        accept: bool = True,
        threaded: bool = False,
    ) -> None:
        self.parts = SimpleNamespace(state=StateBus(), dispatcher=FakeDispatcher())
        self.sentences = sentences if sentences is not None else ["Certainly, sir.", "It is cold."]
        self.transcript = transcript
        self.accept = accept
        self.threaded = threaded
        self.turns: list[RemoteTurn] = []
        self.wrapped: list[object] = []

    def submit_remote_turn(self, turn: RemoteTurn):
        self.turns.append(turn)
        if not self.accept:
            return False
        if self.threaded:
            threading.Thread(target=self._run, args=(turn,), daemon=True).start()
        else:
            self._run(turn)
        return True

    def _run(self, turn: RemoteTurn) -> None:
        """What the worker thread will do once the hook exists."""
        if turn.wrap_dispatcher is not None:
            self.wrapped.append(turn.wrap_dispatcher(self.parts.dispatcher))
        turn.transcribed(self.transcript)
        for sentence in self.sentences:
            turn.on_sentence(sentence)
        turn.finish(" ".join(self.sentences), "")


def _hookless(assistant: FakeAssistant) -> FakeAssistant:
    """Remove the instance hook, leaving a plain object the server must refuse."""
    assistant.submit_remote_turn = None  # type: ignore[assignment]
    return assistant


# ----------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------
@pytest.fixture
def remote_config(config, tmp_path):
    """The shipped config plus a remote section pointing at throwaway paths."""
    config.set(
        "remote",
        {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 0,
            "allow_guarded": False,
            "secret_file": str(tmp_path / "remote-secret.key"),
            "session_ttl_hours": 24,
            "code_ttl": 900,
            "max_attempts": 3,
            "attempt_window": 300,
            "max_audio_seconds": 5,
            "turn_timeout": 5,
            "heartbeat": 5,
            "routines": [
                {
                    "name": "Good night",
                    "say": "Goodnight, sir.",
                    "calls": [{"tool": "get_time_date", "args": {}}],
                },
                {"name": "Lock up", "calls": [{"tool": "lock_pc"}]},
            ],
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
    """Flask's test client, with the app built lazily like the real thing."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    app = server.create_app()
    app.config["TESTING"] = True
    return app.test_client()


def pair_client(client, server):
    """Walk the real pairing flow so the test client holds a valid cookie."""
    response = client.post(
        "/api/pair", json={"code": server.pairing_code, "device": "Test iPhone"}
    )
    assert response.status_code == 200
    return response


# ----------------------------------------------------------------------------------
# The import rule
# ----------------------------------------------------------------------------------
def test_importing_the_package_does_not_import_flask():
    """jarvis.remote must import on a box with no Flask and open nothing."""
    code = "import sys, jarvis.remote; assert 'flask' not in sys.modules, sorted(sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------
# The wire format
# ----------------------------------------------------------------------------------
def test_pcm16_round_trip_keeps_shape_and_scale():
    original = np.array([0.0, 0.5, -0.5, 0.999, -1.0], dtype=np.float32)
    payload = (np.clip(original, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    decoded = decode_pcm16(payload)

    assert decoded.dtype == np.float32
    assert decoded.shape == (5,)
    assert np.max(np.abs(decoded - original)) < 1e-3
    assert decoded.flags.writeable  # frombuffer's view is read-only; Whisper needs a copy


def test_pcm16_one_second_of_audio_is_one_second_of_samples():
    samples = (np.sin(np.linspace(0, 400, 16000)) * 0.3 * 32767).astype("<i2")
    decoded = decode_pcm16(samples.tobytes())
    assert decoded.size == 16000
    assert decoded.size / 16000 == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload, reason",
    [(b"", "empty"), (b"\x01\x02\x03", "Int16"), (b"\x00\x00" * 40, "limit")],
)
def test_pcm16_rejects_bad_frames(payload, reason):
    with pytest.raises(ValueError) as excinfo:
        decode_pcm16(payload, max_samples=32)
    assert reason in str(excinfo.value)


def test_the_pages_encoder_and_the_servers_decoder_agree():
    """Run the page's own toPcm16() and decode it here: the two halves of the wire.

    Skipped when node is not installed; the assertion is about the JavaScript in
    index.html, not about this machine.
    """
    import re
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the page's JavaScript cannot be run here")

    html = (Path(server_module._STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    match = re.search(r"function toPcm16\(samples\) \{[\s\S]*?\n\}", html)
    assert match, "the page no longer has a toPcm16(); this test needs rewriting"

    values = [0.0, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0]
    script = (
        match.group(0)
        + f"\nconst out = toPcm16(new Float32Array({values}));"
        + "\nprocess.stdout.write(Buffer.from(out).toString('hex'));"
    )
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=60, check=True
    )
    decoded = decode_pcm16(bytes.fromhex(result.stdout))

    expected = np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
    assert decoded.shape == expected.shape
    assert np.max(np.abs(decoded - expected)) < 1e-3


# ----------------------------------------------------------------------------------
# Pairing, cookies and rate limits
# ----------------------------------------------------------------------------------
def test_pairing_with_the_right_code_issues_a_verifiable_token():
    store = SessionStore(b"x" * 32)
    result = store.pair(store.pairing_code, "Karim's iPhone", "10.0.0.2")

    assert result.ok and result.status == "ok"
    device = store.verify_token(result.token)
    assert device is not None
    assert device.name == "Karim's iPhone"


def test_a_used_code_cannot_pair_a_second_device():
    store = SessionStore(b"x" * 32)
    code = store.pairing_code
    assert store.pair(code, "first", "ip").ok

    second = store.pair(code, "second", "ip")

    assert not second.ok
    assert second.status == "bad_code"
    assert store.pairing_code != code


def test_wrong_codes_are_rate_limited_per_client():
    store = SessionStore(b"x" * 32, max_attempts=3)
    wrong = "000000" if store.pairing_code != "000000" else "111111"

    statuses = [store.pair(wrong, "phone", "10.0.0.9").status for _ in range(4)]

    assert statuses == ["bad_code", "bad_code", "bad_code", "rate_limited"]
    assert store.attempts_left("10.0.0.9") == 0
    # The correct code is refused too while the lock-out lasts.
    assert store.pair(store.pairing_code, "phone", "10.0.0.9").status == "rate_limited"
    # ... but a different client is unaffected.
    assert store.pair(store.pairing_code, "phone", "10.0.0.10").ok


def test_the_rate_limit_window_expires():
    now = [1000.0]
    store = SessionStore(b"x" * 32, max_attempts=2, attempt_window=60, clock=lambda: now[0])
    wrong = "000000" if store.pairing_code != "000000" else "111111"
    store.pair(wrong, "phone", "ip")
    store.pair(wrong, "phone", "ip")
    assert store.pair(wrong, "phone", "ip").status == "rate_limited"

    now[0] += 61.0

    assert store.pair(store.pairing_code, "phone", "ip").ok


def test_a_code_that_is_too_old_is_refused_and_rotated():
    now = [1000.0]
    store = SessionStore(b"x" * 32, code_ttl=60, clock=lambda: now[0])
    code = store.pairing_code
    now[0] += 61.0

    result = store.pair(code, "phone", "ip")

    assert result.status == "expired"
    assert store.pairing_code != code


@pytest.mark.parametrize("length", [0, 3, 7])
def test_a_code_of_the_wrong_length_is_malformed(length):
    store = SessionStore(b"x" * 32)
    assert store.pair("1" * length, "phone", "ip").status == "malformed"


def test_a_tampered_or_foreign_token_is_rejected():
    store = SessionStore(b"x" * 32)
    token = store.pair(store.pairing_code, "phone", "ip").token
    head, payload, signature = token.split(".")

    assert store.verify_token(None) is None
    assert store.verify_token("") is None
    assert store.verify_token("garbage") is None
    assert store.verify_token(f"{head}.{payload}.{signature[:-2]}AA") is None
    assert store.verify_token(f"v2.{payload}.{signature}") is None
    assert SessionStore(b"y" * 32).verify_token(token) is None


def test_an_expired_token_is_rejected():
    now = [1000.0]
    store = SessionStore(b"x" * 32, ttl_hours=1, clock=lambda: now[0])
    token = store.pair(store.pairing_code, "phone", "ip").token
    assert store.verify_token(token) is not None

    now[0] += 3601.0

    assert store.verify_token(token) is None


def test_revoke_all_invalidates_every_cookie():
    store = SessionStore(b"x" * 32)
    token = store.pair(store.pairing_code, "phone", "ip").token

    store.revoke_all()

    assert store.verify_token(token) is None


def test_the_secret_survives_a_restart_and_a_broken_path(tmp_path):
    path = tmp_path / "nested" / "secret.key"
    first = load_or_create_secret(path)
    assert path.is_file()
    assert load_or_create_secret(path) == first

    path.write_text("not-hex", encoding="utf-8")
    replacement = load_or_create_secret(path)
    assert len(replacement) >= 16

    # A directory where the file should be: still returns a usable in-memory secret.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    assert len(load_or_create_secret(blocked)) >= 16


# ----------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------
def test_the_page_is_one_file_with_the_pinned_vad(client):
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "@ricky0123/vad-web@0.0.22" in body
    assert "onnxruntime-web@1.14.0" in body
    assert "manifest.webmanifest" in body
    assert "MediaRecorder" not in body  # the wire format is PCM, never a codec


def test_every_element_the_page_scripts_actually_exists(server):
    """A typo in an id is a blank screen on the phone and nothing in the logs."""
    pytest.importorskip("flask")
    import re

    html = (Path(server_module._STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    wanted = set(re.findall(r'el\("([^"]+)"\)', html))
    declared = set(re.findall(r'id="([^"]+)"', html))

    assert wanted, "the page stopped looking elements up; this test needs rewriting"
    assert wanted <= declared, f"missing from the markup: {sorted(wanted - declared)}"


def test_every_endpoint_the_page_calls_is_routed(server):
    """The page and the app agree on the URLs, or nothing works and nothing says why."""
    pytest.importorskip("flask")
    app = server.create_app()
    html = (Path(server_module._STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    routed = {rule.rule for rule in app.url_map.iter_rules()}

    for path in ("/api/session", "/api/pair", "/api/unpair", "/api/routine",
                 "/ws/audio", "/ws/state", "/manifest.webmanifest", "/icon.svg"):
        assert path in html, f"the page never calls {path}"
        assert path in routed, f"the app never serves {path}"


def test_the_manifest_makes_it_installable(client):
    response = client.get("/manifest.webmanifest")
    manifest = response.get_json()

    assert response.status_code == 200
    assert manifest["display"] == "standalone"
    assert manifest["icons"]


def test_a_missing_page_file_is_reported_not_crashed(server, monkeypatch, tmp_path):
    pytest.importorskip("flask")
    monkeypatch.setattr(server_module, "_STATIC_DIR", tmp_path / "gone")
    client = server.create_app().test_client()

    response = client.get("/")

    assert response.status_code == 500


def test_nothing_works_before_pairing(client):
    assert client.get("/api/session").get_json()["paired"] is False
    assert client.post("/api/routine", json={"name": "Good night"}).status_code == 401


def test_pairing_over_http_sets_an_httponly_cookie(client, server):
    response = pair_client(client, server)

    header = response.headers.get("Set-Cookie", "")
    assert COOKIE_NAME in header
    assert "HttpOnly" in header
    assert "SameSite=Strict" in header

    session = client.get("/api/session").get_json()
    assert session["paired"] is True
    assert session["device"] == "Test iPhone"
    assert [routine["name"] for routine in session["routines"]] == ["Good night", "Lock up"]


def test_a_wrong_code_over_http_is_refused_then_rate_limited(client, server):
    wrong = "000000" if server.pairing_code != "000000" else "111111"

    codes = [client.post("/api/pair", json={"code": wrong}).status_code for _ in range(4)]

    assert codes == [401, 401, 401, 429]
    assert client.get("/api/session").get_json()["paired"] is False


def test_unpairing_clears_the_cookie(client, server):
    pair_client(client, server)
    assert client.get("/api/session").get_json()["paired"] is True

    client.post("/api/unpair")

    assert client.get("/api/session").get_json()["paired"] is False


# ----------------------------------------------------------------------------------
# Routines
# ----------------------------------------------------------------------------------
def test_load_routines_keeps_the_good_and_drops_the_broken():
    cfg = SimpleNamespace(
        get=lambda key, default=None: [
            {"name": "Good night", "calls": [{"tool": "lock_pc", "args": {"x": 1}}], "say": "Night."},
            {"name": "No calls"},
            {"calls": [{"tool": "lock_pc"}]},
            "not a mapping",
            {"name": "Bare string call", "calls": ["get_time_date"]},
        ]
    )

    routines = load_routines(cfg)

    assert [routine.name for routine in routines] == ["Good night", "Bare string call"]
    assert routines[0].calls == (("lock_pc", {"x": 1}),)
    assert routines[1].calls == (("get_time_date", {}),)


def test_a_routine_runs_its_calls_through_the_dispatcher(client, server, assistant):
    pair_client(client, server)

    body = client.post("/api/routine", json={"name": "Good night"}).get_json()

    assert body["ok"] is True
    assert body["said"] == "Goodnight, sir."
    assert assistant.parts.dispatcher.executed == [("get_time_date", {})]


def test_an_unknown_routine_is_a_404_and_runs_nothing(client, server, assistant):
    pair_client(client, server)

    response = client.post("/api/routine", json={"name": "Open the pod bay doors"})

    assert response.status_code == 404
    assert assistant.parts.dispatcher.executed == []


def test_a_routine_without_a_dispatcher_says_so(server, assistant):
    assistant.parts.dispatcher = None
    ok, said = server.run_routine("Good night", _device(server))

    assert ok is False
    assert said.endswith("sir.")


def _device(server):
    from jarvis.remote.session import Device

    return Device(name="Test iPhone", id="abc123")


# ----------------------------------------------------------------------------------
# The audio socket
# ----------------------------------------------------------------------------------
def _pcm(seconds: float = 0.5, rate: int = 16000) -> bytes:
    samples = np.sin(np.linspace(0, 120, int(rate * seconds))) * 0.4
    return (samples * 32767).astype("<i2").tobytes()


def _in_request(server, app, *, cookie: str | None):
    """A request context carrying (or not carrying) the pairing cookie."""
    headers = {"Cookie": f"{COOKIE_NAME}={cookie}"} if cookie else {}
    return app.test_request_context("/ws/audio", headers=headers)


def _token(server) -> str:
    return server.sessions.pair(server.pairing_code, "Test iPhone", "127.0.0.1").token


def test_an_unpaired_socket_is_closed_immediately(server):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket([_pcm()])

    with _in_request(server, app, cookie=None):
        server.audio_socket(ws)

    assert ws.kinds() == ["unpaired"]
    assert ws.closed is True


def test_a_turn_streams_transcript_then_sentences_then_done(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket([_pcm(0.5)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds() == ["transcript", "sentence", "sentence", "done"]
    events = ws.events
    assert events[0]["text"] == "what is the weather"
    assert [event["text"] for event in events[1:3]] == ["Certainly, sir.", "It is cold."]
    assert events[3]["error"] == ""


def test_the_turn_reaches_the_assistant_as_float32_at_16k(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(FakeSocket([_pcm(0.5)]))

    turn = assistant.turns[0]
    assert turn.sample_rate == 16000
    assert turn.audio.dtype == np.float32
    assert turn.audio.shape == (8000,)
    assert turn.duration_s == pytest.approx(0.5)
    assert turn.device == "Test iPhone"
    assert np.max(np.abs(turn.audio)) <= 1.0


def test_the_reply_stays_on_the_phone_unless_the_config_says_otherwise(remote_config):
    pytest.importorskip("flask")
    quiet = FakeAssistant()
    server = RemoteServer(remote_config, quiet)
    with _in_request(server, server.create_app(), cookie=_token(server)):
        server.audio_socket(FakeSocket([_pcm(0.2)]))
    assert quiet.turns[0].speak_locally is False

    remote_config.set("remote.speak_locally", True)
    loud = FakeAssistant()
    server = RemoteServer(remote_config, loud)
    with _in_request(server, server.create_app(), cookie=_token(server)):
        server.audio_socket(FakeSocket([_pcm(0.2)]))
    assert loud.turns[0].speak_locally is True


def test_the_turn_hands_the_worker_a_guard_for_its_dispatcher(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(FakeSocket([_pcm(0.2)]))

    # The reporter that narrates tool calls to the phone sits outside the guard, so
    # a refusal is reported too. The guard must still be there, underneath.
    wrapped = assistant.wrapped[0]
    guard = getattr(wrapped, "_inner", wrapped)
    assert isinstance(guard, RemoteGuard)
    assert guard.allow_guarded is False


def test_sentences_stream_in_order_from_the_worker_thread(remote_config):
    pytest.importorskip("flask")
    lines = [f"Sentence {index}." for index in range(6)]
    assistant = FakeAssistant(sentences=lines, threaded=True)
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([_pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    spoken = [event["text"] for event in ws.events if event["type"] == "sentence"]
    assert spoken == lines
    assert ws.kinds()[-1] == "done"


def test_a_bad_audio_frame_is_reported_without_dropping_the_socket(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket([b"\x01\x02\x03", _pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds()[0] == "error"
    assert "transcript" in ws.kinds()  # the socket carried on and took the next turn
    assert len(assistant.turns) == 1


def test_audio_longer_than_the_limit_is_refused(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()  # max_audio_seconds is 5 in the fixture
    ws = FakeSocket([_pcm(6.0)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds() == ["error"]
    assert assistant.turns == []


def test_control_frames_are_answered(server):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket(['{"type": "hello"}', '{"type": "ping"}', "not json", '{"type": "wat"}'])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds(skip_pings=False) == ["ready", "pong", "error", "error"]


def test_a_receive_timeout_sends_a_keepalive(server):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket([TIMEOUT, TIMEOUT])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds(skip_pings=False) == ["ping", "ping"]


def test_a_busy_assistant_is_reported_not_swallowed(remote_config):
    pytest.importorskip("flask")
    assistant = FakeAssistant(accept=False)
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([_pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds() == ["busy"]


def test_a_build_without_the_hook_says_so_plainly(remote_config):
    pytest.importorskip("flask")
    assistant = _hookless(FakeAssistant())
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([_pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds() == ["error"]
    assert ws.events[0]["message"] == MISSING_HOOK


def test_a_hook_that_raises_does_not_take_the_socket_down(remote_config):
    pytest.importorskip("flask")
    assistant = FakeAssistant()
    assistant.submit_remote_turn = lambda turn: (_ for _ in ()).throw(RuntimeError("queue died"))
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([_pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds() == ["error"]


def test_a_turn_that_never_finishes_times_out(remote_config):
    pytest.importorskip("flask")
    assistant = FakeAssistant()
    assistant.submit_remote_turn = lambda turn: assistant.turns.append(turn) or True
    remote_config.set("remote.turn_timeout", 0.05)
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([_pcm(0.2)])

    with _in_request(server, app, cookie=_token(server)):
        server.audio_socket(ws)

    assert ws.kinds()[-1] == "error"
    assert "too long" in ws.events[-1]["message"]


# ----------------------------------------------------------------------------------
# The state socket
# ----------------------------------------------------------------------------------
def test_the_state_socket_sends_the_current_state_then_every_change(server, assistant):
    pytest.importorskip("flask")
    app = server.create_app()
    bus = assistant.parts.state
    ws = FakeSocket([], allow_sends=2)
    token = _token(server)

    def run() -> None:
        with app.test_request_context("/ws/state", headers={"Cookie": f"{COOKIE_NAME}={token}"}):
            server.state_socket(ws)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _wait_for(lambda: len(ws.sent) >= 1)

    bus.set(AssistantState.LISTENING)
    _wait_for(lambda: len(ws.sent) >= 2)
    bus.set(AssistantState.THINKING)  # the third send fails, ending the handler
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [event["state"] for event in ws.events] == ["idle", "listening"]
    assert bus.observer_count == 0  # the handler unsubscribed on its way out


def test_an_unpaired_state_socket_is_closed(server):
    pytest.importorskip("flask")
    app = server.create_app()
    ws = FakeSocket([])

    with app.test_request_context("/ws/state"):
        server.state_socket(ws)

    assert ws.kinds() == ["unpaired"]
    assert ws.closed is True


def test_the_state_socket_copes_with_an_assistant_that_has_no_bus(remote_config):
    pytest.importorskip("flask")
    assistant = FakeAssistant()
    assistant.parts = SimpleNamespace(dispatcher=FakeDispatcher())
    server = RemoteServer(remote_config, assistant)
    app = server.create_app()
    ws = FakeSocket([])
    token = _token(server)

    with app.test_request_context("/ws/state", headers={"Cookie": f"{COOKIE_NAME}={token}"}):
        server.state_socket(ws)

    assert ws.kinds() == ["error"]


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


# ----------------------------------------------------------------------------------
# Guarding what the phone may run
# ----------------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def tools_loaded():
    registry.load_all()


def test_a_guarded_tool_is_refused_from_the_phone(remote_config):
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config, allow_guarded=False, device="Test iPhone")

    result = guard.execute("run_powershell", {"command": "Get-Date"})

    assert result.refused is True
    assert result.ok is False
    assert result.summary == REMOTE_REFUSAL
    assert inner.executed == []  # it never reached the real dispatcher


def test_a_safe_tool_passes_straight_through(remote_config):
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config, allow_guarded=False)

    result = guard.execute("get_time_date", {})

    assert result.ok is True
    assert inner.executed == [("get_time_date", {})]


def test_allow_guarded_lets_a_guarded_tool_through(remote_config):
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config, allow_guarded=True)

    assert guard.execute("run_powershell", {"command": "Get-Date"}).ok is True
    assert inner.executed == [("run_powershell", {"command": "Get-Date"})]


def test_guarded_tools_are_hidden_from_the_model(remote_config):
    guard = RemoteGuard(FakeDispatcher(), remote_config, allow_guarded=False)

    names = [entry["function"]["name"] for entry in guard.tools_payload()]

    assert "get_time_date" in names
    assert "run_powershell" not in names


def test_an_unknown_tool_is_left_to_the_real_dispatcher(remote_config):
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config)

    guard.execute("no_such_tool", {})

    assert inner.executed == [("no_such_tool", {})]


def test_the_safety_remote_tier_hook_is_used_when_it_exists(remote_config, monkeypatch):
    """The integrator's hook decides; None means refuse, even for a SAFE tool."""
    monkeypatch.setattr(safety, "remote_tier", lambda spec, cfg: None, raising=False)
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config, allow_guarded=True)

    result = guard.execute("get_time_date", {})

    assert result.refused is True
    assert inner.executed == []


def test_the_hook_can_also_downgrade_a_tool(remote_config, monkeypatch):
    monkeypatch.setattr(safety, "remote_tier", lambda spec, cfg: Tier.SAFE, raising=False)
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config, allow_guarded=False)

    assert guard.execute("run_powershell", {"command": "Get-Date"}).ok is True


def test_a_broken_hook_falls_back_to_the_local_rules(remote_config, monkeypatch):
    def explode(spec, cfg):
        raise RuntimeError("the hook is broken")

    monkeypatch.setattr(safety, "remote_tier", explode, raising=False)
    guard = RemoteGuard(FakeDispatcher(), remote_config, allow_guarded=False)

    assert guard.execute("run_powershell", {"command": "Get-Date"}).refused is True


def test_execute_many_guards_every_call(remote_config):
    inner = FakeDispatcher()
    guard = RemoteGuard(inner, remote_config)

    results = guard.execute_many(
        [{"name": "get_time_date", "arguments": {}}, {"name": "run_powershell", "arguments": {}}]
    )

    assert [result.refused for result in results] == [False, True]


# ----------------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------------
def test_remote_is_off_unless_it_is_switched_on(config, assistant, tmp_path):
    config.set("remote", {"secret_file": str(tmp_path / "s.key")})
    server = RemoteServer(config, assistant)

    assert server.enabled is False
    assert server.start() is False
    assert server.running is False


def test_a_wildcard_bind_address_is_refused(remote_config, assistant):
    remote_config.set("remote.host", "0.0.0.0")
    server = RemoteServer(remote_config, assistant)

    assert server.start() is False
    assert server.running is False


def test_start_reports_failure_when_flask_is_missing(remote_config, assistant, monkeypatch):
    server = RemoteServer(remote_config, assistant)

    def no_flask() -> None:
        raise ImportError("No module named 'flask'")

    monkeypatch.setattr(server, "create_app", no_flask)

    assert server.start() is False


def test_start_and_stop_on_a_real_loopback_port(remote_config, assistant):
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    simple_websocket = pytest.importorskip("simple_websocket")
    from urllib.request import ProxyHandler, build_opener

    server = RemoteServer(remote_config, assistant)
    assert server.start() is True
    try:
        assert server.running is True
        assert server.url.startswith("http://127.0.0.1:")
        opener = build_opener(ProxyHandler({}))  # never go through a proxy for loopback

        with opener.open(server.url + "/", timeout=10) as response:
            assert b"JARVIS" in response.read()

        # An unpaired socket is refused by the real server, not just the fake one.
        client = simple_websocket.Client(server.url.replace("http://", "ws://") + "/ws/state")
        try:
            assert json.loads(client.receive(timeout=10))["type"] == "unpaired"
        finally:
            # The server closed first, so closing again is already an error.
            try:
                client.close()
            except Exception:
                pass
    finally:
        server.stop()
    assert server.running is False
    server.stop()  # idempotent


def test_a_whole_turn_over_a_real_websocket(remote_config, assistant):
    """The one end-to-end proof: real HTTP, real WebSocket, real PCM, real ordering."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_sock")
    simple_websocket = pytest.importorskip("simple_websocket")
    from urllib.request import ProxyHandler, Request, build_opener

    server = RemoteServer(remote_config, assistant)
    assert server.start() is True
    opener = build_opener(ProxyHandler({}))
    try:
        request = Request(
            server.url + "/api/pair",
            data=json.dumps({"code": server.pairing_code, "device": "Test iPhone"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=10) as response:
            assert json.loads(response.read())["ok"] is True
            cookie = response.headers["Set-Cookie"].split(";")[0]

        client = simple_websocket.Client(
            server.url.replace("http://", "ws://") + "/ws/audio", headers={"Cookie": cookie}
        )
        try:
            client.send(_pcm(0.4))
            events = []
            while len(events) < 4:
                message = json.loads(client.receive(timeout=10))
                if message["type"] not in ("ping", "pong"):
                    events.append(message)
        finally:
            try:
                client.close()
            except Exception:
                pass
    finally:
        server.stop()

    assert [event["type"] for event in events] == ["transcript", "sentence", "sentence", "done"]
    assert [event["text"] for event in events[1:3]] == ["Certainly, sir.", "It is cold."]
    assert assistant.turns[0].audio.shape == (6400,)


def test_the_url_reflects_the_configured_address(remote_config, assistant):
    remote_config.set("remote.host", "100.64.0.5")
    remote_config.set("remote.port", 8765)
    server = RemoteServer(remote_config, assistant)

    assert server.url == "http://100.64.0.5:8765"
    assert len(server.pairing_code) == 6 and server.pairing_code.isdigit()
