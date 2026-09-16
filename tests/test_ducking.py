"""Ducking other applications, tested without Windows, pycaw or a sound card.

``pycaw`` cannot be installed on this machine — it imports ``comtypes``, which is
Windows-only — so the whole Windows audio session API is injected through
``sys.modules`` as a handful of small fakes and ``sys.platform`` is monkeypatched to
``win32``. That is the only honest way to exercise the code that actually ships: the
alternative is testing a mock of our own design and learning nothing.

The interesting assertions are the ones about failure. A ducker that fades Spotify down
and then dies is worse than no ducker at all, so there are tests for the crash path, for
a session that disappears mid-fade, for a corrupt snapshot file and for a stop that
happens while still ducked.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

from jarvis.audio import ducking
from jarvis.audio.ducking import DEFAULT_LEVEL, Ducker, SessionVolume
from jarvis.core.state import AssistantState, StateBus

# A ramp short enough to keep the suite fast, long enough to be several steps.
TEST_RAMP_MS = 60
WAIT_S = 3.0


# --------------------------------------------------------------- the fake Windows API
class FakeVolume:
    """Stands in for ``ISimpleAudioVolume``, recording every write."""

    def __init__(self, value: float, *, fails: bool = False) -> None:
        self.value = float(value)
        self.history: list[float] = [float(value)]
        self.fails = fails

    def GetMasterVolume(self) -> float:  # noqa: N802 - the COM name
        if self.fails:
            raise OSError("the session went away")
        return self.value

    def SetMasterVolume(self, value: float, context: object) -> None:  # noqa: N802
        if self.fails:
            raise OSError("the session went away")
        assert context is None
        self.value = float(value)
        self.history.append(self.value)


class FakeProcess:
    def __init__(self, pid: int, name: str) -> None:
        self.pid = pid
        self._name = name

    def name(self) -> str:
        return self._name


class FakeSession:
    def __init__(self, pid: int, name: str, volume: float, *, fails: bool = False) -> None:
        self.Process = FakeProcess(pid, name)
        self.SimpleAudioVolume = FakeVolume(volume, fails=fails)


class SystemSoundsSession:
    """The session Windows owns itself: a volume interface but no process."""

    Process = None

    def __init__(self) -> None:
        self.SimpleAudioVolume = FakeVolume(1.0)


class FakeComtypes:
    """Counts CoInitialize/CoUninitialize so the pairing can be asserted."""

    def __init__(self) -> None:
        self.initialised = 0
        self.uninitialised = 0

    def CoInitialize(self) -> None:  # noqa: N802 - the COM name
        self.initialised += 1

    def CoUninitialize(self) -> None:  # noqa: N802
        self.uninitialised += 1


@pytest.fixture
def audio_sessions() -> list[object]:
    """Whatever the fake ``AudioUtilities.GetAllSessions()`` should return."""
    return []


@pytest.fixture
def windows(monkeypatch, audio_sessions):
    """Pretend to be Windows with a working pycaw, and hand back the fake comtypes."""
    monkeypatch.setattr(sys, "platform", "win32")

    comtypes = FakeComtypes()
    comtypes_module = types.ModuleType("comtypes")
    comtypes_module.CoInitialize = comtypes.CoInitialize
    comtypes_module.CoUninitialize = comtypes.CoUninitialize

    class AudioUtilities:
        @staticmethod
        def GetAllSessions():  # noqa: N802 - the pycaw name
            return list(audio_sessions)

    pycaw_pkg = types.ModuleType("pycaw")
    pycaw_inner = types.ModuleType("pycaw.pycaw")
    pycaw_inner.AudioUtilities = AudioUtilities
    pycaw_pkg.pycaw = pycaw_inner

    monkeypatch.setitem(sys.modules, "comtypes", comtypes_module)
    monkeypatch.setitem(sys.modules, "pycaw", pycaw_pkg)
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", pycaw_inner)
    return comtypes


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "ducking.json"


def make_ducker(bus: StateBus, store: Path, **kwargs) -> Ducker:
    kwargs.setdefault("ramp_ms", TEST_RAMP_MS)
    return Ducker(bus, store=store, **kwargs)


def wait_until(predicate, timeout: float = WAIT_S) -> bool:
    """Poll ``predicate`` until it is true. The worker thread is asynchronous."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


# ------------------------------------------------------------------ off Windows
def test_module_imports_without_pycaw_on_this_machine():
    """The import at the top of this file already proves it; assert it out loud."""
    assert "pycaw" not in sys.modules or sys.modules["pycaw"].__name__ == "pycaw"
    assert ducking.Ducker is Ducker


def test_ducking_is_a_no_op_off_windows(store):
    bus = StateBus()
    ducker = make_ducker(bus, store)
    assert ducker.available is False
    ducker.duck()
    ducker.restore()
    assert not store.exists()


def test_the_whole_lifecycle_is_harmless_off_windows(store):
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    bus.set(AssistantState.LISTENING)
    bus.set(AssistantState.IDLE)
    ducker.stop()
    assert not store.exists()


def test_missing_pycaw_on_windows_degrades_with_an_explanation(
    windows, monkeypatch, store, caplog
):
    """Windows, comtypes present, pycaw not installed: a warning, not a crash."""
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", None)  # forces an ImportError
    ducker = make_ducker(StateBus(), store)
    with caplog.at_level("WARNING"):
        assert ducker.available is False
    assert any("pycaw" in record.message for record in caplog.records)
    ducker.duck()  # and the no-op path still holds
    assert not store.exists()


def test_a_broken_com_stack_does_not_cost_the_duck(windows, audio_sessions, monkeypatch, store):
    """CoInitialize can fail on a damaged install; the session API often still works.

    COM may also simply be initialised already on this thread, which raises too. Either
    way, failing to initialise it is a debug note, not a reason to abandon the duck and
    leave the operator talking over his music.
    """
    broken = types.ModuleType("comtypes")

    def explode(*_args, **_kwargs):
        raise OSError("COM is not registered")

    broken.__getattr__ = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comtypes", broken)

    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    ducker = make_ducker(StateBus(), store)

    ducker.duck()
    assert session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    ducker.restore()
    assert session.SimpleAudioVolume.value == 1.0


def test_the_backend_is_only_resolved_once(windows, monkeypatch, store):
    """The import cost and the warning both belong to startup, not to every turn."""
    ducker = make_ducker(StateBus(), store)
    assert ducker.available is True
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", None)
    assert ducker.available is True  # cached, not re-imported on the hot path


# ------------------------------------------------------------------ the snapshot
def test_duck_snapshots_every_session_before_touching_it(windows, audio_sessions, store):
    audio_sessions.append(FakeSession(4321, "spotify.exe", 0.8))
    ducker = make_ducker(StateBus(), store)
    assert ducker.available is True

    ducker.duck()

    saved = json.loads(store.read_text(encoding="utf-8"))
    assert saved["sessions"] == [{"pid": 4321, "name": "spotify.exe", "volume": 0.8}]


def test_the_snapshot_is_on_disk_before_the_first_fader_moves(windows, audio_sessions, store):
    """A snapshot written after the fade is worthless: the crash happens in between."""
    seen: list[bool] = []

    class WatchfulVolume(FakeVolume):
        def SetMasterVolume(self, value, context):  # noqa: N802
            seen.append(store.exists())
            super().SetMasterVolume(value, context)

    session = FakeSession(4321, "spotify.exe", 1.0)
    session.SimpleAudioVolume = WatchfulVolume(1.0)
    audio_sessions.append(session)

    make_ducker(StateBus(), store).duck()

    assert seen and seen[0] is True


def test_duck_ramps_down_gradually_rather_than_jumping(windows, audio_sessions, store):
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    make_ducker(StateBus(), store).duck()

    volume = session.SimpleAudioVolume
    assert volume.value == pytest.approx(DEFAULT_LEVEL)
    # 60 ms at one write per 20 ms is three steps, plus the starting value.
    assert len(volume.history) >= 4
    assert volume.history == sorted(volume.history, reverse=True)
    assert volume.history[1] < 1.0  # the first write is already below the original
    assert volume.history[1] > DEFAULT_LEVEL  # ...but nowhere near the target yet


def test_ducking_is_relative_so_a_quiet_app_is_not_turned_up(windows, audio_sessions, store):
    session = FakeSession(4321, "spotify.exe", 0.3)
    audio_sessions.append(session)
    make_ducker(StateBus(), store).duck()
    assert session.SimpleAudioVolume.value == pytest.approx(0.3 * DEFAULT_LEVEL)


def test_jarvis_own_session_is_never_ducked(windows, audio_sessions, store):
    import os

    mine = FakeSession(os.getpid(), "python.exe", 1.0)
    theirs = FakeSession(os.getpid() + 1, "spotify.exe", 1.0)
    audio_sessions.extend([mine, theirs])

    make_ducker(StateBus(), store).duck()

    assert mine.SimpleAudioVolume.value == 1.0
    assert mine.SimpleAudioVolume.history == [1.0]
    assert theirs.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    saved = json.loads(store.read_text(encoding="utf-8"))
    assert [item["pid"] for item in saved["sessions"]] == [os.getpid() + 1]


def test_the_system_sounds_session_is_skipped(windows, audio_sessions, store):
    sounds = SystemSoundsSession()
    audio_sessions.append(sounds)
    make_ducker(StateBus(), store).duck()
    assert sounds.SimpleAudioVolume.value == 1.0
    assert not store.exists()


def test_a_session_that_dies_mid_enumeration_does_not_stop_the_others(
    windows, audio_sessions, store
):
    broken = FakeSession(1, "dying.exe", 1.0, fails=True)
    healthy = FakeSession(2, "spotify.exe", 1.0)
    audio_sessions.extend([broken, healthy])

    make_ducker(StateBus(), store).duck()

    assert healthy.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    saved = json.loads(store.read_text(encoding="utf-8"))
    assert [item["name"] for item in saved["sessions"]] == ["spotify.exe"]


def test_ducking_twice_does_not_overwrite_the_snapshot(windows, audio_sessions, store):
    """The bug this guards: re-snapshotting while ducked records 20% as "normal"."""
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    ducker = make_ducker(StateBus(), store)

    ducker.duck()
    ducker.duck()

    saved = json.loads(store.read_text(encoding="utf-8"))
    assert saved["sessions"][0]["volume"] == 1.0
    ducker.restore()
    assert session.SimpleAudioVolume.value == 1.0


def test_com_is_initialised_and_released_around_each_pass(windows, audio_sessions, store):
    audio_sessions.append(FakeSession(4321, "spotify.exe", 1.0))
    ducker = make_ducker(StateBus(), store)
    ducker.duck()
    ducker.restore()
    assert windows.initialised >= 2
    assert windows.uninitialised == windows.initialised


# ------------------------------------------------------------------ the state bus
def test_listening_ducks_and_idle_restores(windows, audio_sessions, store):
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    try:
        bus.set(AssistantState.LISTENING)
        assert wait_until(lambda: session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL))
        assert store.exists()

        bus.set(AssistantState.IDLE)
        assert wait_until(lambda: session.SimpleAudioVolume.value == 1.0)
        assert wait_until(lambda: not store.exists())
    finally:
        ducker.stop()


def test_the_music_stays_down_while_thinking_and_speaking(windows, audio_sessions, store):
    """Swelling back up between the question and the answer would be absurd."""
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    try:
        bus.set(AssistantState.LISTENING)
        assert wait_until(lambda: session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL))
        bus.set(AssistantState.THINKING)
        bus.set(AssistantState.SPEAKING)
        time.sleep(0.05)
        assert session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    finally:
        ducker.stop()
    assert session.SimpleAudioVolume.value == 1.0


def test_the_observer_never_blocks_the_thread_that_changed_state(windows, audio_sessions, store):
    """The capture loop publishes LISTENING; a 120 ms fade must not happen on it."""
    audio_sessions.append(FakeSession(4321, "spotify.exe", 1.0))
    bus = StateBus()
    ducker = make_ducker(bus, store, ramp_ms=400)
    ducker.start()
    try:
        started = time.monotonic()
        bus.set(AssistantState.LISTENING)
        assert time.monotonic() - started < 0.1
    finally:
        ducker.stop()


def test_stop_restores_even_when_idle_never_arrives(windows, audio_sessions, store):
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    bus.set(AssistantState.LISTENING)
    assert wait_until(lambda: session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL))

    ducker.stop()

    assert session.SimpleAudioVolume.value == 1.0
    assert not store.exists()


def test_stop_unsubscribes_from_the_bus(windows, audio_sessions, store):
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    assert bus.observer_count == 1
    ducker.stop()
    assert bus.observer_count == 0


def test_start_is_idempotent(windows, audio_sessions, store):
    bus = StateBus()
    ducker = make_ducker(bus, store)
    ducker.start()
    ducker.start()
    try:
        assert bus.observer_count == 1
    finally:
        ducker.stop()


# ------------------------------------------------------------------ crash recovery
def test_a_crash_leaves_a_snapshot_that_the_next_start_replays(windows, audio_sessions, store):
    """The whole point: JARVIS dies while listening and Spotify is still at 20%."""
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)

    crashed = make_ducker(StateBus(), store)
    crashed.duck()
    assert session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    del crashed  # no stop(), no restore — exactly what a crash looks like
    assert store.exists()

    bus = StateBus()
    recovered = make_ducker(bus, store)
    recovered.start()
    try:
        assert session.SimpleAudioVolume.value == 1.0
        assert not store.exists()
    finally:
        recovered.stop()


def test_recovery_only_touches_sessions_whose_pid_and_name_still_match(
    windows, audio_sessions, store
):
    """A reused pid must not have some unrelated process's volume rewritten."""
    store.write_text(
        json.dumps({"sessions": [{"pid": 4321, "name": "spotify.exe", "volume": 0.9}]}),
        encoding="utf-8",
    )
    impostor = FakeSession(4321, "chrome.exe", 0.4)
    audio_sessions.append(impostor)

    assert make_ducker(StateBus(), store).recover() == 0
    assert impostor.SimpleAudioVolume.value == 0.4


def test_recovery_survives_the_process_having_exited(windows, audio_sessions, store):
    store.write_text(
        json.dumps({"sessions": [{"pid": 4321, "name": "spotify.exe", "volume": 0.9}]}),
        encoding="utf-8",
    )
    assert make_ducker(StateBus(), store).recover() == 0
    assert not store.exists()


def test_a_corrupt_snapshot_is_ignored_not_raised(windows, audio_sessions, store, caplog):
    store.write_text("{not json at all", encoding="utf-8")
    ducker = make_ducker(StateBus(), store)
    with caplog.at_level("WARNING"):
        assert ducker.recover() == 0
    assert any("unreadable" in record.message for record in caplog.records)


def test_a_snapshot_with_junk_entries_keeps_the_usable_ones(windows, audio_sessions, store):
    store.write_text(
        json.dumps({"sessions": [
            {"pid": "nonsense"},
            {"pid": 4321, "name": "spotify.exe", "volume": 17.0},  # out of range
            {"pid": 4321, "name": "spotify.exe", "volume": 0.9},
            "not a dict",
        ]}),
        encoding="utf-8",
    )
    session = FakeSession(4321, "spotify.exe", 0.2)
    audio_sessions.append(session)
    assert make_ducker(StateBus(), store).recover() == 1
    assert session.SimpleAudioVolume.value == pytest.approx(0.9)


def test_recovery_does_nothing_and_keeps_the_file_off_windows(store):
    store.write_text(
        json.dumps({"sessions": [{"pid": 1, "name": "spotify.exe", "volume": 0.9}]}),
        encoding="utf-8",
    )
    assert make_ducker(StateBus(), store).recover() == 0
    assert store.exists(), "the snapshot must survive until a machine can act on it"


def test_a_missing_snapshot_file_is_not_an_error(windows, audio_sessions, store):
    assert make_ducker(StateBus(), store).recover() == 0


def test_an_unwritable_store_does_not_stop_the_duck(windows, audio_sessions, tmp_path, caplog):
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    # A directory where the file should be: every write fails, nothing raises.
    bad = tmp_path / "blocked.json"
    bad.mkdir()
    ducker = make_ducker(StateBus(), bad)
    with caplog.at_level("WARNING"):
        ducker.duck()
    assert session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    ducker.restore()
    assert session.SimpleAudioVolume.value == 1.0


# ------------------------------------------------------------------ small pieces
def test_level_and_ramp_are_clamped_to_something_sane(store):
    bus = StateBus()
    assert make_ducker(bus, store, level=-2.0)._level == 0.0
    assert make_ducker(bus, store, level=9.0)._level == 1.0
    assert make_ducker(bus, store, ramp_ms=-50)._ramp_ms == 0


def test_a_zero_length_ramp_still_reaches_the_target(windows, audio_sessions, store):
    session = FakeSession(4321, "spotify.exe", 1.0)
    audio_sessions.append(session)
    make_ducker(StateBus(), store, ramp_ms=0).duck()
    assert session.SimpleAudioVolume.value == pytest.approx(DEFAULT_LEVEL)
    assert len(session.SimpleAudioVolume.history) == 2  # start, then straight to target


def test_session_volume_round_trips_through_json():
    item = SessionVolume(pid=7, name="Spotify.exe", volume=0.5)
    assert SessionVolume.from_dict(item.as_dict()) == item
    assert item.key == (7, "spotify.exe")


@pytest.mark.parametrize(
    "raw",
    [None, "text", {}, {"pid": 1}, {"pid": "x", "name": "a", "volume": 0.5},
     {"pid": 1, "name": "a", "volume": -0.1}, {"pid": 1, "name": "a", "volume": 2.0}],
)
def test_unusable_snapshot_entries_are_rejected(raw):
    assert SessionVolume.from_dict(raw) is None


def test_repr_says_what_is_going_on(store):
    text = repr(make_ducker(StateBus(), store))
    assert "Ducker" in text and "ducked=False" in text


def test_context_manager_starts_and_stops(windows, audio_sessions, store):
    bus = StateBus()
    with make_ducker(bus, store) as ducker:
        assert isinstance(ducker, Ducker)
        assert bus.observer_count == 1
    assert bus.observer_count == 0
