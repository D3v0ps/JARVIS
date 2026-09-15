"""Timers and reminders for JARVIS.

A single daemon thread sleeps on a :class:`threading.Condition` until the next job
is due, so adding, cancelling or stopping wakes it instantly instead of polling.
Jobs are persisted to ``logs/schedule.json`` on every mutation (atomically), which
means a timer that came due while JARVIS was closed still gets announced: the
scheduler fires everything already past due once, right after :meth:`Scheduler.start`.

:func:`parse_when` turns spoken English or Swedish ("in ten minutes", "tomorrow at
08:00", "om 10 minuter", "kl 19.30") into a timezone-naive local ``datetime``.

Standard library only — this module imports on a bare Linux box.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Final

from jarvis.config import project_root
from jarvis.core.logging import get_logger

logger = get_logger("scheduler")

__all__ = ["Job", "Scheduler", "parse_when"]

#: Store format version written to ``logs/schedule.json``.
STORE_VERSION: Final[int] = 1

#: How long :meth:`Scheduler.stop` waits for the worker thread to finish.
_JOIN_TIMEOUT: Final[float] = 5.0


@dataclass
class Job:
    """One scheduled announcement.

    ``due`` is an absolute ``time.time()`` epoch so the store survives restarts.
    ``kind`` is ``"timer"`` (a countdown) or ``"reminder"`` (a wall-clock moment).
    """

    id: str
    due: float
    label: str = ""
    kind: str = "timer"
    text: str = ""

    @property
    def due_at(self) -> datetime:
        """The due moment as a local, timezone-naive ``datetime``."""
        return datetime.fromtimestamp(self.due)

    def seconds_left(self, now: float | None = None) -> float:
        """Seconds until this job fires (negative when it is already past due)."""
        return self.due - (time.time() if now is None else now)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON store."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Rebuild from a store entry, tolerating missing or oddly typed fields."""
        return cls(
            id=str(data.get("id") or ""),
            due=float(data.get("due") or 0.0),
            label=str(data.get("label") or ""),
            kind=str(data.get("kind") or "timer"),
            text=str(data.get("text") or ""),
        )


class Scheduler:
    """Single background thread firing ``on_due(job)`` when a job comes due."""

    def __init__(
        self,
        on_due: Callable[[Job], None],
        store: str | Path = "logs/schedule.json",
    ) -> None:
        self._on_due = on_due
        self._store = self._resolve_store(store)
        self._cond = threading.Condition(threading.RLock())
        self._jobs: dict[str, Job] = {}
        self._next_id: int = 1
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._load()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the worker thread. Idempotent; past-due jobs fire immediately."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, name="jarvis-scheduler", daemon=True
            )
            self._thread.start()
            logger.debug("Scheduler started with %d pending job(s)", len(self._jobs))

    def stop(self) -> None:
        """Stop the worker thread. Safe to call twice or before :meth:`start`."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning("Scheduler thread did not stop within %.1fs", _JOIN_TIMEOUT)
        logger.debug("Scheduler stopped")

    # ------------------------------------------------------------------- jobs

    def add_timer(self, minutes: float, label: str = "") -> Job:
        """Schedule a countdown ``minutes`` from now."""
        try:
            amount = float(minutes)
        except (TypeError, ValueError) as exc:
            logger.error("Invalid timer duration %r: %s", minutes, exc)
            raise ValueError(f"Invalid timer duration: {minutes!r}") from exc
        clean = (label or "").strip()
        due = time.time() + max(0.0, amount) * 60.0
        job = Job(id="", due=due, label=clean, kind="timer", text=clean or "timer")
        return self._add(job)

    def add_reminder(self, text: str, when: str) -> Job:
        """Schedule a reminder at the moment described by ``when``."""
        moment = parse_when(when)
        if moment is None:
            logger.error("Could not parse reminder time %r", when)
            raise ValueError(f"Could not understand the time: {when!r}")
        clean = (text or "").strip()
        job = Job(
            id="",
            due=moment.timestamp(),
            label=clean,
            kind="reminder",
            text=clean or "reminder",
        )
        return self._add(job)

    def cancel(self, job_id: str) -> bool:
        """Cancel by id, or by exact (case-insensitive) label when no id matches."""
        wanted = (job_id or "").strip().lower()
        if not wanted:
            return False
        with self._cond:
            match = next(
                (j for j in self._jobs.values() if j.id.lower() == wanted),
                None,
            ) or next(
                (j for j in self._jobs.values() if j.label.strip().lower() == wanted),
                None,
            )
            if match is None:
                return False
            self._jobs.pop(match.id, None)
            self._save_locked()
            self._cond.notify_all()
        logger.info("Cancelled job %s (%s)", match.id, match.label or match.kind)
        return True

    def pending(self) -> list[Job]:
        """All jobs that have not fired yet, soonest first."""
        with self._cond:
            return sorted(self._jobs.values(), key=lambda job: job.due)

    # ---------------------------------------------------------------- internals

    def _add(self, job: Job) -> Job:
        with self._cond:
            job.id = self._make_id()
            self._jobs[job.id] = job
            self._save_locked()
            self._cond.notify_all()
        logger.info(
            "Scheduled %s %s for %s (%s)",
            job.kind,
            job.id,
            job.due_at.strftime("%Y-%m-%d %H:%M:%S"),
            job.label or job.text,
        )
        return job

    def _make_id(self) -> str:
        """Short, human-speakable id ("t1", "t2"), unique for the store's lifetime."""
        while True:
            candidate = f"t{self._next_id}"
            self._next_id += 1
            if candidate not in self._jobs:
                return candidate

    def _run(self) -> None:
        while True:
            fired: list[Job] = []
            with self._cond:
                if self._stopping:
                    return
                now = time.time()
                due = sorted(
                    (job for job in self._jobs.values() if job.due <= now),
                    key=lambda job: job.due,
                )
                if due:
                    for job in due:
                        self._jobs.pop(job.id, None)
                    self._save_locked()
                    fired = due
                else:
                    nxt = min((job.due for job in self._jobs.values()), default=None)
                    timeout = None if nxt is None else max(0.0, nxt - time.time())
                    self._cond.wait(timeout)
                    continue
            for job in fired:
                self._fire(job)

    def _fire(self, job: Job) -> None:
        """Invoke the callback; a raising callback must not kill the thread."""
        if self._on_due is None:
            logger.warning("Job %s came due but no on_due callback is set", job.id)
            return
        try:
            self._on_due(job)
        except Exception:  # noqa: BLE001 - a bad callback must not stop the scheduler
            logger.exception("on_due callback failed for job %s (%s)", job.id, job.label)

    # -------------------------------------------------------------- persistence

    @staticmethod
    def _resolve_store(store: str | Path) -> Path:
        path = Path(store)
        return path if path.is_absolute() else project_root() / path

    def _load(self) -> None:
        """Read the store; a missing or corrupt file just means "no jobs"."""
        try:
            raw = self._store.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.error("Could not read schedule store %s: %s", self._store, exc)
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Corrupt schedule store %s (%s); starting empty", self._store, exc)
            return
        entries = data.get("jobs", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            logger.error("Unexpected schedule store layout in %s; starting empty", self._store)
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                job = Job.from_dict(entry)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping unreadable job entry %r: %s", entry, exc)
                continue
            if not job.id:
                job.id = self._make_id()
            self._jobs[job.id] = job
        stored_next = data.get("next_id") if isinstance(data, dict) else None
        try:
            self._next_id = max(int(stored_next or 1), self._highest_id() + 1, self._next_id)
        except (TypeError, ValueError):
            self._next_id = max(self._highest_id() + 1, self._next_id)
        logger.debug("Loaded %d job(s) from %s", len(self._jobs), self._store)

    def _highest_id(self) -> int:
        highest = 0
        for job_id in self._jobs:
            match = re.fullmatch(r"t(\d+)", job_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    def _save_locked(self) -> None:
        """Atomically write the store. Caller holds the condition's lock."""
        payload = {
            "version": STORE_VERSION,
            "next_id": self._next_id,
            "jobs": [job.to_dict() for job in sorted(self._jobs.values(), key=lambda j: j.due)],
        }
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self._store.parent),
                prefix=".schedule-",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(handle.name, self._store)
            except BaseException:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("Could not persist schedule to %s: %s", self._store, exc)
