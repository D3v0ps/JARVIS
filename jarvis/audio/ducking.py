"""Ducking every other application while JARVIS is listening.

Music over the microphone is not cosmetic: it reaches the wake word as noise, Silero as
speech and Whisper as invented words. Pulling the other audio sessions to a fifth for the
two seconds JARVIS listens fixes transcription accuracy, wake misses and false barge-in
at once. ``pycaw`` gives each process's own ``ISimpleAudioVolume``, so this is a
*per-application* fader: the master volume is untouched and JARVIS's own session is
skipped, leaving the chimes and the voice at full level while Spotify steps aside.

Two details make it safe rather than merely clever. The original volumes go to a small
JSON file **before** the fade starts and :meth:`Ducker.recover` replays it at the next
startup, so a crash mid-turn cannot leave Spotify at 20 % until the machine reboots; and
the restore runs from a ``finally`` on the worker thread, so it happens on a clean stop,
on an exception, and on a state this class did not anticipate.

``pycaw`` and ``comtypes`` are imported inside the method that needs them, so this
imports on a bare Linux box - where it is a logged no-op.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from jarvis.core.logging import get_logger
from jarvis.core.state import AssistantState, StateBus

__all__ = [
    "Ducker", "SessionVolume", "DEFAULT_LEVEL", "DEFAULT_RAMP_MS", "DEFAULT_STORE",
    "RAMP_STEP_MS", "STOP_TIMEOUT_S",
]

#: Fraction of what each application was playing at. A fifth is quiet enough for the
#: microphone and loud enough that the operator can still hear his music.
DEFAULT_LEVEL = 0.2
#: The fade, in ms: over before the first syllable, slow enough not to sound like a drop.
DEFAULT_RAMP_MS = 120
#: Where the pre-duck volumes are parked so a crash is recoverable.
DEFAULT_STORE = "logs/ducking.json"
#: One fader write per this many ms. Windows coalesces anything faster anyway.
RAMP_STEP_MS = 20.0
#: How long :meth:`Ducker.stop` waits for the worker to finish its restore.
STOP_TIMEOUT_S = 3.0

_UNCHECKED: Any = object()
_DUCK, _RESTORE = "duck", "restore"


@dataclass(frozen=True)
class SessionVolume:
    """One application's volume as it was before JARVIS touched it.

    Windows reuses process ids, so :attr:`key` pairs the pid with the process name and
    both must match before a remembered volume is written to anything.
    """

    pid: int
    name: str
    volume: float

    @property
    def key(self) -> tuple[int, str]:
        """The identity a live session has to match to be restored."""
        return (self.pid, self.name.lower())

    def as_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "name": self.name, "volume": self.volume}

    @classmethod
    def from_dict(cls, raw: Any) -> "SessionVolume | None":
        """Parse one persisted entry, returning ``None`` for anything unusable."""
        if not isinstance(raw, dict):
            return None
        try:
            pid = int(raw["pid"])
            name = str(raw["name"])
            volume = float(raw["volume"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0.0 <= volume <= 1.0:
            return None
        return cls(pid=pid, name=name, volume=volume)


@dataclass(frozen=True)
class _Backend:
    """The two Windows-only pieces this module needs, resolved once."""

    comtypes: Any
    sessions: Callable[[], Any]


class Ducker:
    """Fades other applications down while JARVIS listens, and always puts them back.

    :meth:`start` subscribes it to the state bus: it ducks on ``LISTENING`` and restores
    on ``IDLE`` and ``PAUSED``, staying down through ``THINKING`` and ``SPEAKING`` so the
    music does not swell back up between the question and the answer. Every COM call
    happens on a private worker thread; the observer only queues a word, so a 120 ms fade
    can never block the capture loop that published the change.
    """

    def __init__(
        self,
        state: StateBus,
        *,
        level: float = DEFAULT_LEVEL,
        ramp_ms: int = DEFAULT_RAMP_MS,
        store: str | Path = DEFAULT_STORE,
        logger: logging.Logger | None = None,
    ) -> None:
        self._state = state
        self._level = float(min(max(float(level), 0.0), 1.0))
        self._ramp_ms = max(0, int(ramp_ms))
        self._store = Path(store)
        self._log = logger or get_logger("audio.ducking")

        self._lock = threading.RLock()
        self._snapshot: list[SessionVolume] = []
        self._ducked = False
        self._unsubscribe: Callable[[], None] | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._backend: Any = _UNCHECKED

    # --- availability ----------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when this machine can actually duck: Windows, with pycaw importable."""
        return self._load_backend() is not None

    def _load_backend(self) -> _Backend | None:
        """Resolve pycaw/comtypes once, explaining the first time it cannot be done."""
        with self._lock:
            if self._backend is not _UNCHECKED:
                return self._backend
            self._backend = None
            if sys.platform != "win32":
                self._log.info("Ducking needs the Windows audio session API; it is "
                               "disabled on %s.", sys.platform)
                return None
            try:
                import comtypes  # noqa: PLC0415 - Windows-only dependency
                from pycaw.pycaw import AudioUtilities  # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001 - a broken COM install raises too
                self._log.warning("pycaw or comtypes is unavailable, so other "
                                  "applications will not be ducked: %s", exc)
                return None
            self._backend = _Backend(comtypes, AudioUtilities.GetAllSessions)
            return self._backend

    @contextlib.contextmanager
    def _com(self, backend: _Backend) -> Iterator[None]:
        """Initialise COM for this thread; pycaw raises without it on a worker.

        Best effort: COM already being up raises, and so does a damaged install.
        """
        started = False
        try:
            init = getattr(backend.comtypes, "CoInitialize", None)
            if callable(init):
                init()
                started = True
        except Exception as exc:  # noqa: BLE001
            self._log.debug("CoInitialize complained: %s", exc)
        try:
            yield
        finally:
            try:
                done = getattr(backend.comtypes, "CoUninitialize", None)
                if started and callable(done):
                    done()
            except Exception as exc:  # noqa: BLE001 - releasing COM never raises onwards
                self._log.debug("CoUninitialize complained: %s", exc)

    # --- session enumeration ---------------------------------------------------------
    def _live_sessions(self, backend: _Backend) -> list[tuple[Any, int, str]]:
        """Every duckable session as ``(volume interface, pid, process name)``.

        JARVIS's own process is excluded - ducking it would fade the chimes and the voice
        - and so is the system-sounds session, which owns no process to match on later.
        """
        own_pid = os.getpid()
        found: list[tuple[Any, int, str]] = []
        for session in backend.sessions():
            try:
                process = getattr(session, "Process", None)
                if process is None:
                    continue
                pid = int(process.pid)
                name = str(process.name())
            except Exception as exc:  # noqa: BLE001 - a session can die mid-enumeration
                self._log.debug("Skipping an audio session that could not be read: %s", exc)
                continue
            interface = getattr(session, "SimpleAudioVolume", None)
            if pid == own_pid or interface is None:
                self._log.debug("Skipping audio session %s (pid %d).", name, pid)
                continue
            found.append((interface, pid, name))
        return found

    def _set_volume(self, interface: Any, value: float) -> bool:
        """Write one fader, clamped. False when the session refused the write."""
        clamped = float(min(max(float(value), 0.0), 1.0))
        try:
            interface.SetMasterVolume(clamped, None)
        except Exception as exc:  # noqa: BLE001 - the process may have exited between calls
            self._log.debug("Could not set a session volume to %.3f: %s", clamped, exc)
            return False
        return True

    # --- ducking ---------------------------------------------------------------------
    def duck(self) -> None:
        """Snapshot every session's volume, persist it, then fade to ``level``.

        A no-op when already ducked: re-snapshotting would record 20 % as normal.
        """
        with self._lock:
            if self._ducked:
                return
            backend = self._load_backend()
            if backend is None:
                return
            try:
                with self._com(backend):
                    snapshot: list[SessionVolume] = []
                    targets: list[tuple[Any, float]] = []
                    for interface, pid, name in self._live_sessions(backend):
                        try:
                            current = float(interface.GetMasterVolume())
                        except Exception as exc:  # noqa: BLE001
                            self._log.debug("Could not read %s's volume: %s", name, exc)
                            continue
                        snapshot.append(SessionVolume(pid=pid, name=name, volume=current))
                        targets.append((interface, current))

                    if not snapshot:
                        self._log.debug("Nothing else is playing audio; nothing to duck.")
                        return

                    # Persist before touching a fader: the file is only useful if it is
                    # already on disk when the crash happens.
                    self._snapshot = snapshot
                    self._ducked = True
                    self._write_store(snapshot)
                    self._ramp(targets)
                    self._log.info("Ducked %d session(s) to %d%% while listening: %s",
                                   len(snapshot), round(self._level * 100),
                                   ", ".join(item.name for item in snapshot))
            except Exception:  # noqa: BLE001 - ducking must never take down a voice turn
                self._log.exception("Ducking failed; leaving the other applications alone.")

    def _ramp(self, targets: list[tuple[Any, float]]) -> None:
        """Fade each session from its own volume to ``volume * level`` over ``ramp_ms``."""
        steps = max(1, int(round(self._ramp_ms / RAMP_STEP_MS))) if self._ramp_ms > 0 else 1
        pause = (self._ramp_ms / 1000.0) / steps
        for step in range(1, steps + 1):
            fraction = step / steps
            for interface, original in targets:
                self._set_volume(interface, original * (1.0 - fraction * (1.0 - self._level)))
            if step < steps and pause > 0:
                time.sleep(pause)

    def restore(self) -> None:
        """Put every remembered session back where it was. Safe to call repeatedly."""
        with self._lock:
            snapshot = list(self._snapshot)
            self._snapshot = []
            self._ducked = False
            if not snapshot:
                # Nothing of ours to undo, but a stale file would confuse the next start.
                self._clear_store()
                return
            self._apply(snapshot, "Restored")

    def _apply(self, snapshot: list[SessionVolume], what: str) -> int:
        """Write a snapshot onto whichever of its sessions are still alive.

        On success the file is dropped; on failure it is kept for the next startup.
        """
        backend = self._load_backend()
        if backend is None:
            return 0
        wanted = {item.key: item.volume for item in snapshot}
        restored = 0
        try:
            with self._com(backend):
                for interface, pid, name in self._live_sessions(backend):
                    volume = wanted.get((pid, name.lower()))
                    if volume is not None and self._set_volume(interface, volume):
                        restored += 1
        except Exception:  # noqa: BLE001 - a failed restore must not raise onwards
            self._log.exception("%s failed; keeping %s so the next start can try again.",
                                what, self._store)
            return restored
        self._log.info("%s %d of %d session volume(s).", what, restored, len(snapshot))
        self._clear_store()
        return restored

    def recover(self) -> int:
        """Replay a snapshot left behind by a previous run. Returns sessions restored.

        The half that matters after a crash: nothing else knows Spotify is still at 20 %.
        """
        if not self.available:
            return 0
        snapshot = self._read_store()
        if not snapshot:
            return 0
        self._log.warning("Found a ducking snapshot from a previous run (%d session(s)); "
                          "a turn was interrupted while listening. Restoring the "
                          "original volumes.", len(snapshot))
        return self._apply(snapshot, "Recovered")

    # --- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        """Recover anything a previous run left ducked, then follow the state bus."""
        with self._lock:
            if self._worker is not None:
                return
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=self._run, name="jarvis-ducker",
                                            daemon=True)
            self._worker.start()
        # Synchronous on purpose: a handful of COM calls, and the operator should not get
        # a voice turn in before his music is back to normal.
        try:
            self.recover()
        except Exception:  # noqa: BLE001 - never block startup on the audio mixer
            self._log.exception("Ducking recovery failed at startup.")
        self._unsubscribe = self._state.subscribe(self._on_state)
        self._log.debug("Ducker started (available=%s, level=%.2f, ramp=%dms).",
                        self.available, self._level, self._ramp_ms)

    def stop(self) -> None:
        """Unsubscribe and restore. Always restores, even if the worker is wedged."""
        with self._lock:
            unsubscribe, worker = self._unsubscribe, self._worker
            self._unsubscribe = None
            self._worker = None
        try:
            if unsubscribe is not None:
                unsubscribe()
        except Exception:  # noqa: BLE001
            self._log.debug("Unsubscribing the ducker from the state bus failed.")
        finally:
            if worker is not None and worker.is_alive():
                self._queue.put(None)
                worker.join(timeout=STOP_TIMEOUT_S)
                if worker.is_alive():
                    self._log.warning("The ducking worker is wedged; restoring inline.")
                    self.restore()
            else:
                self.restore()

    def _on_state(self, state: AssistantState) -> None:
        """State-bus observer. Queues work; never does it on the caller's thread."""
        if state is AssistantState.LISTENING:
            action = _DUCK
        elif state in (AssistantState.IDLE, AssistantState.PAUSED):
            action = _RESTORE
        else:
            return  # THINKING and SPEAKING stay ducked
        with self._lock:
            if self._worker is not None:
                self._queue.put(action)

    def _run(self) -> None:
        """The worker loop. The ``finally`` is the guarantee the whole design rests on."""
        try:
            while True:
                action = self._queue.get()
                if action is None:
                    return
                try:
                    if action == _DUCK:
                        self.duck()
                    else:
                        self.restore()
                except Exception:  # noqa: BLE001
                    self._log.exception("The ducking worker failed on %r.", action)
        finally:
            try:
                self.restore()
            except Exception:  # noqa: BLE001
                self._log.exception("The ducking worker's final restore failed.")

    # --- the crash-recovery file -----------------------------------------------------
    def _write_store(self, snapshot: list[SessionVolume]) -> None:
        """Write the snapshot atomically. A failure here never stops the duck."""
        payload = {"saved_at": time.time(), "pid": os.getpid(),
                   "sessions": [item.as_dict() for item in snapshot]}
        tmp = self._store.with_suffix(self._store.suffix + ".tmp")
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._store)
        except OSError as exc:
            self._log.warning("Could not save the pre-duck volumes to %s (%s); a crash "
                              "now would leave the other applications quiet.",
                              self._store, exc)
            with contextlib.suppress(OSError):
                tmp.unlink()

    def _read_store(self) -> list[SessionVolume]:
        """Read the snapshot file, tolerating every way it can be broken."""
        try:
            raw = json.loads(self._store.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            self._log.warning("Ignoring an unreadable ducking snapshot at %s: %s",
                              self._store, exc)
            return []
        entries = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            self._log.warning("Ignoring a malformed ducking snapshot at %s.", self._store)
            return []
        return [item for item in map(SessionVolume.from_dict, entries) if item is not None]

    def _clear_store(self) -> None:
        with contextlib.suppress(OSError):
            self._store.unlink(missing_ok=True)

    def __enter__(self) -> "Ducker":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        with self._lock:
            return (f"<Ducker level={self._level:.2f} ramp={self._ramp_ms}ms "
                    f"ducked={self._ducked} sessions={len(self._snapshot)}>")
