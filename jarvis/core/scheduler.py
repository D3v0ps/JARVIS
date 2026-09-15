"""Timers and reminders for JARVIS.

A single daemon thread sleeps on a :class:`threading.Condition` until the next job is
due, so adding, cancelling or stopping wakes it instantly instead of polling. Jobs are
persisted atomically to ``logs/schedule.json`` on every mutation: a timer that came due
while JARVIS was closed still gets announced, because everything already past due fires
once right after :meth:`Scheduler.start`.

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

    ``due`` is an absolute ``time.time()`` epoch so the store survives restarts and
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
        path = Path(store)
        self._store = path if path.is_absolute() else project_root() / path
        self._cond = threading.Condition(threading.RLock())
        self._jobs: dict[str, Job] = {}
        self._next_id: int = 1
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._load()

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
        """Stop the worker thread. Safe to call twice, or before :meth:`start`."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning("Scheduler thread did not stop within %.1f s", _JOIN_TIMEOUT)
        logger.debug("Scheduler stopped")

    def add_timer(self, minutes: float, label: str = "") -> Job:
        """Schedule a countdown ``minutes`` from now."""
        try:
            amount = float(minutes)
        except (TypeError, ValueError) as exc:
            logger.error("Invalid timer duration %r: %s", minutes, exc)
            raise ValueError(f"Invalid timer duration: {minutes!r}") from exc
        clean = (label or "").strip()
        due = time.time() + max(0.0, amount) * 60.0
        return self._add(Job(id="", due=due, label=clean, kind="timer", text=clean or "timer"))

    def add_reminder(self, text: str, when: str) -> Job:
        """Schedule a reminder at the moment described by ``when``."""
        moment = parse_when(when)
        if moment is None:
            logger.error("Could not parse reminder time %r", when)
            raise ValueError(f"Could not understand the time: {when!r}")
        clean = (text or "").strip()
        job = Job(id="", due=moment.timestamp(), label=clean, kind="reminder",
                  text=clean or "reminder")
        return self._add(job)

    def cancel(self, job_id: str) -> bool:
        """Cancel by id, or by exact (case-insensitive) label when no id matches."""
        wanted = (job_id or "").strip().lower()
        if not wanted:
            return False
        with self._cond:
            match = next(
                (j for j in self._jobs.values() if j.id.lower() == wanted), None
            ) or next(
                (j for j in self._jobs.values() if j.label.strip().lower() == wanted), None
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

    def _add(self, job: Job) -> Job:
        with self._cond:
            job.id = self._make_id()
            self._jobs[job.id] = job
            self._save_locked()
            self._cond.notify_all()
        logger.info("Scheduled %s %s for %s (%s)", job.kind, job.id,
                    job.due_at.strftime("%Y-%m-%d %H:%M:%S"), job.label or job.text)
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
                due = sorted((j for j in self._jobs.values() if j.due <= now),
                             key=lambda job: job.due)
                if due:
                    for job in due:
                        self._jobs.pop(job.id, None)
                    self._save_locked()
                    fired = due
                else:
                    nxt = min((job.due for job in self._jobs.values()), default=None)
                    self._cond.wait(None if nxt is None else max(0.0, nxt - time.time()))
                    continue
            for job in fired:
                self._fire(job)

    def _fire(self, job: Job) -> None:
        """Invoke the callback; a raising callback must never kill the thread."""
        if self._on_due is None:
            logger.warning("Job %s came due but no on_due callback is set", job.id)
            return
        try:
            self._on_due(job)
        except Exception:  # noqa: BLE001 - a bad callback must not stop the scheduler
            logger.exception("on_due callback failed for job %s (%s)", job.id, job.label)

    def _load(self) -> None:
        """Read the store; a missing or corrupt file simply means "no jobs"."""
        try:
            raw = self._store.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.error("Could not read schedule store %s: %s", self._store, exc)
            return
        try:
            data = json.loads(raw)
        except ValueError as exc:
            logger.error("Corrupt schedule store %s (%s); starting empty", self._store, exc)
            return
        entries = data.get("jobs", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            logger.error("Unexpected schedule layout in %s; starting empty", self._store)
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                job = Job.from_dict(entry)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping unreadable job entry %r: %s", entry, exc)
                continue
            job.id = job.id or self._make_id()
            self._jobs[job.id] = job
        # Continue the id sequence above both the stored counter and every id seen,
        # so a restart can never hand out an id the store already used.
        stored = data.get("next_id") if isinstance(data, dict) else None
        seen = [int(m.group(1)) for m in
                (re.fullmatch(r"t(\d+)", job_id) for job_id in self._jobs) if m]
        try:
            counter = int(stored or 1)
        except (TypeError, ValueError):
            counter = 1
        self._next_id = max(counter, max(seen, default=0) + 1, self._next_id)
        logger.debug("Loaded %d job(s) from %s", len(self._jobs), self._store)

    def _save_locked(self) -> None:
        """Atomically rewrite the store. The caller holds the condition's lock."""
        payload = {
            "version": STORE_VERSION,
            "next_id": self._next_id,
            "jobs": [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.due)],
        }
        tmp_name = ""
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(self._store.parent),
                                            prefix=".schedule-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._store)
        except OSError as exc:
            logger.error("Could not persist schedule to %s: %s", self._store, exc)
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    logger.debug("Could not remove temporary store file %s", tmp_name)


#: Number words accepted in both languages, plus the fractions we need.
_WORDS: Final[dict[str, float]] = {
    "a": 1, "an": 1, "one": 1, "en": 1, "ett": 1, "two": 2, "tva": 2, "två": 2,
    "three": 3, "tre": 3, "four": 4, "fyra": 4, "five": 5, "fem": 5, "six": 6,
    "sex": 6, "seven": 7, "sju": 7, "eight": 8, "atta": 8, "åtta": 8, "nine": 9,
    "nio": 9, "ten": 10, "tio": 10, "eleven": 11, "elva": 11, "twelve": 12,
    "tolv": 12, "fifteen": 15, "femton": 15, "twenty": 20, "tjugo": 20,
    "thirty": 30, "trettio": 30, "forty": 40, "fyrtio": 40, "fifty": 50,
    "femtio": 50, "sixty": 60, "sextio": 60, "half": 0.5, "halv": 0.5,
}

#: Duration units mapped to seconds.
_UNITS: Final[dict[str, int]] = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "sek": 1, "sekund": 1, "sekunder": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "minut": 60, "minuter": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "tim": 3600, "timme": 3600, "timmar": 3600,
    "day": 86400, "days": 86400, "dag": 86400, "dagar": 86400, "dygn": 86400,
}

_WORD_ALT: Final[str] = "|".join(sorted(_WORDS, key=len, reverse=True))
_NUM: Final[str] = rf"\d+(?:[.,]\d+)?|{_WORD_ALT}"
_HOUR: Final[str] = rf"\d{{1,2}}|{_WORD_ALT}"

_REL_MARKER = re.compile(r"\b(?:in|om|efter|inom|within|about)\b")
_REL_PAIR = re.compile(rf"\b(?P<amount>{_NUM})\s*(?P<unit>[a-zåäö]+)\b")
_BARE_REL = re.compile(rf"^\s*(?:{_NUM})\s*[a-zåäö]+\s*$")
_TIME = re.compile(r"(?<![\d:.])(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?(?:\s*(?P<ap>am|pm))?(?![\d])")
_HALF_EN = re.compile(rf"\bhalf past\s+(?P<h>{_HOUR})\b")
_HALF_SV = re.compile(rf"\bhalv\s+(?P<h>{_HOUR})\b")
_QUARTER_PAST = re.compile(rf"\b(?:quarter past|kvart över)\s+(?P<h>{_HOUR})\b")
_QUARTER_TO = re.compile(rf"\b(?:quarter to|kvart i)\s+(?P<h>{_HOUR})\b")

_TOMORROW = re.compile(r"\b(?:tomorrow|imorgon|imorron|morgondagen)\b")
_OVERMORROW = re.compile(r"\b(?:overmorgon|övermorgon|day after tomorrow)\b")
_EVENING = re.compile(r"\b(?:tonight|this evening|in the evening|ikvall|ikväll|"
                      r"pa kvallen|på kvällen|i eftermiddag)\b")
_MORNING = re.compile(r"\b(?:bitti|in the morning|this morning|pa morgonen|"
                      r"på morgonen|på förmiddagen|pa formiddagen)\b")

#: Spelling unifications applied before parsing, in order.
_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    (r"\bkl\.?\b", "klockan"), (r"\bklocka\b", "klockan"),
    (r"\ba\.m\.?\b", "am"), (r"\bp\.m\.?\b", "pm"),
    (r"\bo'?clock\b", ""), (r"\bi\s+morgon\b", "imorgon"),
    (r"\bi\s+kväll\b", "ikväll"), (r"\bi\s+kvall\b", "ikväll"),
    (r"\bhalf an hour\b", "30 minutes"), (r"\bhalf a minute\b", "30 seconds"),
    (r"\b(?:en\s+)?halvtimme\b", "30 minuter"),
    (r"\ba quarter of an hour\b", "15 minutes"), (r"\bom en kvart\b", "om 15 minuter"),
)


def _normalise(text: str) -> str:
    """Lower-case, de-clutter and unify the spellings the parsers expect."""
    clean = unicodedata.normalize("NFKC", text or "").lower().strip()
    clean = clean.replace("!", " ").replace("?", " ").replace(",", " ")
    for pattern, repl in _REPLACEMENTS:
        clean = re.sub(pattern, repl, clean)
    return re.sub(r"\s+", " ", clean).strip()


def _number(token: str) -> float | None:
    """Turn "10", "1,5", "ten" or "tio" into a float, or ``None``."""
    token = token.strip()
    if token in _WORDS:
        return float(_WORDS[token])
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _parse_relative(text: str, base: datetime) -> datetime | None:
    """Handle "in 10 minutes", "om 2 timmar", "in 1 hour 30 minutes", "90 seconds"."""
    if not _REL_MARKER.search(text) and not _BARE_REL.match(text):
        return None
    total, found = 0.0, False
    for match in _REL_PAIR.finditer(text):
        seconds = _UNITS.get(match.group("unit"))
        amount = _number(match.group("amount"))
        if seconds is None or amount is None:
            continue
        total += amount * seconds
        found = True
    if not found:
        return None
    return (base + timedelta(seconds=total)).replace(microsecond=0)


def _fuzzy_clock(text: str) -> tuple[int, int] | None:
    """Handle "half past seven", "halv åtta", "kvart över sju", "kvart i åtta"."""
    for pattern, shift, minute in ((_HALF_EN, 0, 30), (_HALF_SV, -1, 30),
                                   (_QUARTER_PAST, 0, 15), (_QUARTER_TO, -1, 45)):
        match = pattern.search(text)
        if match is None:
            continue
        hour = _number(match.group("h"))
        if hour is not None:
            return int(hour) + shift, minute
    return None


def _parse_clock(text: str, base: datetime) -> datetime | None:
    """Handle "at 19:30", "19.30", "tomorrow at 08:00", "imorgon klockan 8", "ikväll 21:00"."""
    days = 2 if _OVERMORROW.search(text) else 1 if _TOMORROW.search(text) else 0
    explicit_day = days > 0
    evening, morning = bool(_EVENING.search(text)), bool(_MORNING.search(text))

    fuzzy = _fuzzy_clock(text)
    meridiem: str | None = None
    if fuzzy is not None:
        hour, minute = fuzzy
    else:
        match = _TIME.search(text)
        if match is not None:
            hour = int(match.group("h"))
            minute = int(match.group("m") or 0)
            meridiem = match.group("ap")
        elif explicit_day or evening or morning:
            # A day word with no clock time: assume the usual hour of that part of day.
            hour, minute = (8 if morning else 20 if evening else 9), 0
        else:
            return None

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem is None and evening and 1 <= hour <= 11:
        hour += 12
    elif meridiem is None and morning and hour == 12:
        hour = 0
    if hour == 24:
        hour, days = 0, days + 1
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    moment = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    moment += timedelta(days=days)
    if moment <= base and not explicit_day:
        moment += timedelta(days=1)
    return moment


def parse_when(when: str, now: datetime | None = None) -> datetime | None:
    """Parse an English or Swedish time expression into a local, naive ``datetime``.

    Understands "in 10 minutes", "in 2 hours", "at 19:30", "7:30 pm", "19.30",
    "half past seven", "tomorrow at 08:00", "tonight at 21:00", "om 10 minuter",
    "klockan 19:30", "kl 19.30", "imorgon 08:00", "imorgon klockan 8" and
    "ikväll 21:00". A bare time that already passed today means tomorrow. All
    relative maths uses ``now`` when it is supplied, so the behaviour is testable.
    Returns ``None`` when the expression genuinely cannot be understood.
    """
    base = now if now is not None else datetime.now()
    text = _normalise(when)
    if not text:
        return None
    for parser in (_parse_relative, _parse_clock):
        try:
            moment = parser(text, base)
        except (ValueError, OverflowError) as exc:
            logger.warning("Failed to parse time expression %r: %s", when, exc)
            return None
        if moment is not None:
            return moment
    logger.debug("Unparseable time expression: %r", when)
    return None
