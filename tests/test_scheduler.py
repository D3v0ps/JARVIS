"""Timers, reminders and the natural-language time parser.

Everything here runs against a store inside ``tmp_path`` — the scheduler must never
write into the repository during a test run. Every wait has a timeout and is asserted
on, so a broken scheduler fails instead of hanging the suite.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import pytest

from jarvis.core.scheduler import Job, Scheduler, parse_when

#: Frozen reference moment for the whole parse_when table: Sunday 17 May 2026, 14:30.
#: Chosen so that 19:30 is still ahead and 08:00 is already behind.
NOW = datetime(2026, 5, 17, 14, 30, 0)

#: Longest a test will ever wait for the worker thread before failing.
WAIT = 2.0


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    """A datetime in May 2026, to keep the expectation table readable."""
    return datetime(2026, 5, day, hour, minute, second)


class Collector:
    """Records fired jobs from the scheduler thread and lets a test wait for them."""

    def __init__(self, raise_on: str = "") -> None:
        self.jobs: list[Job] = []
        self._lock = threading.Lock()
        self._fired = threading.Event()
        self._raise_on = raise_on

    def __call__(self, job: Job) -> None:
        with self._lock:
            self.jobs.append(job)
        self._fired.set()
        if self._raise_on and job.label == self._raise_on:
            raise RuntimeError(f"the {job.label} callback is broken")

    @property
    def labels(self) -> list[str]:
        with self._lock:
            return [job.label for job in self.jobs]

    def wait_for(self, count: int, timeout: float = WAIT) -> bool:
        """Block until ``count`` jobs have fired; False on timeout (never hangs)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.jobs) >= count:
                    return True
            self._fired.wait(0.02)
            self._fired.clear()
        with self._lock:
            return len(self.jobs) >= count


@pytest.fixture
def store(tmp_path):
    """Absolute path to a throwaway schedule store."""
    return tmp_path / "schedule.json"


@pytest.fixture
def running(store):
    """Factory for started schedulers that are always stopped again."""
    made: list[Scheduler] = []

    def make(on_due, path=None) -> Scheduler:
        scheduler = Scheduler(on_due, path or store)
        made.append(scheduler)
        scheduler.start()
        return scheduler

    yield make
    for scheduler in made:
        scheduler.stop()


# ======================================================================================
# parse_when — the table
# ======================================================================================

PARSE_CASES = [
    # --- English, relative ---
    ("in 10 minutes", at(17, 14, 40)),
    ("in 90 seconds", at(17, 14, 31, 30)),
    ("in 2 hours", at(17, 16, 30)),
    ("in 1 minute", at(17, 14, 31)),
    ("in an hour", at(17, 15, 30)),
    ("in ten minutes", at(17, 14, 40)),
    ("in half an hour", at(17, 15, 0)),
    ("in 1 hour 30 minutes", at(17, 16, 0)),
    ("in 3 days", at(20, 14, 30)),
    ("90 seconds", at(17, 14, 31, 30)),
    # --- English, wall clock ---
    ("at 19:30", at(17, 19, 30)),
    ("19:30", at(17, 19, 30)),
    ("19.30", at(17, 19, 30)),
    ("at 7:30 pm", at(17, 19, 30)),
    ("7:30 pm", at(17, 19, 30)),
    ("at 8 am", at(18, 8, 0)),
    ("at 12 pm", at(18, 12, 0)),
    ("tomorrow at 08:00", at(18, 8, 0)),
    ("tomorrow 08:00", at(18, 8, 0)),
    ("tonight at 21:00", at(17, 21, 0)),
    ("half past seven", at(18, 7, 30)),
    ("quarter to eight", at(18, 7, 45)),
    # --- Swedish, relative ---
    ("om 10 minuter", at(17, 14, 40)),
    ("om en timme", at(17, 15, 30)),
    ("om 2 timmar", at(17, 16, 30)),
    ("om 30 sekunder", at(17, 14, 30, 30)),
    ("om tio minuter", at(17, 14, 40)),
    ("om en kvart", at(17, 14, 45)),
    # --- Swedish, wall clock ---
    ("klockan 19:30", at(17, 19, 30)),
    ("kl 19.30", at(17, 19, 30)),
    ("kl. 19:30", at(17, 19, 30)),
    ("imorgon 08:00", at(18, 8, 0)),
    ("imorgon klockan 8", at(18, 8, 0)),
    ("ikväll 21:00", at(17, 21, 0)),
    ("i kväll klockan 21", at(17, 21, 0)),
    ("ikvall 21:00", at(17, 21, 0)),  # spoken transcript without the umlaut
    ("halv åtta", at(18, 7, 30)),
    ("kvart över sju", at(18, 7, 15)),
    ("övermorgon 09:00", at(19, 9, 0)),
]


@pytest.mark.parametrize("expression,expected", PARSE_CASES, ids=[c[0] for c in PARSE_CASES])
def test_parse_when_understands_english_and_swedish(expression, expected):
    assert parse_when(expression, NOW) == expected


UNPARSEABLE = [
    "",
    "   ",
    "banana",
    "when pigs fly",
    "sometime",
    "på tisdag",
    "next week",
    "om",
    "at",
    "at 25:00",
    "tomorrow at 25:00",
    "å ä ö",
]


@pytest.mark.parametrize("expression", UNPARSEABLE, ids=[repr(c) for c in UNPARSEABLE])
def test_parse_when_returns_none_for_nonsense(expression):
    assert parse_when(expression, NOW) is None


def test_parse_when_survives_none_instead_of_a_string():
    assert parse_when(None, NOW) is None


def test_bare_time_already_past_today_means_tomorrow():
    """08:00 is behind 14:30, so the user means tomorrow morning."""
    assert parse_when("08:00", NOW) == at(18, 8, 0)
    assert parse_when("at 08:00", NOW) == at(18, 8, 0)
    assert parse_when("klockan 08:00", NOW) == at(18, 8, 0)


def test_bare_time_still_ahead_today_stays_today():
    assert parse_when("19:30", NOW) == at(17, 19, 30)


def test_bare_time_equal_to_now_rolls_over_to_tomorrow():
    """14:30 on the dot is not "in zero seconds" — it is tomorrow's 14:30."""
    assert parse_when("at 14:30", NOW) == at(18, 14, 30)


def test_explicit_tomorrow_is_not_pushed_a_second_day():
    assert parse_when("tomorrow at 19:30", NOW) == at(18, 19, 30)


def test_relative_time_never_rolls_to_tomorrow():
    """"in 10 minutes" at 23:55 crosses midnight rather than waiting a day."""
    late = datetime(2026, 5, 17, 23, 55, 0)
    assert parse_when("in 10 minutes", late) == datetime(2026, 5, 18, 0, 5, 0)


def test_parse_when_defaults_to_the_real_clock_when_now_is_omitted():
    before = datetime.now()
    moment = parse_when("in 2 hours")
    assert moment is not None
    assert timedelta(hours=1, minutes=59) <= moment - before <= timedelta(hours=2, minutes=1)


def test_parse_when_is_case_and_whitespace_insensitive():
    assert parse_when("  IN 10   Minutes ", NOW) == at(17, 14, 40)
    assert parse_when("Klockan 19:30", NOW) == at(17, 19, 30)


def test_swedish_decimal_comma_is_understood_as_a_fraction():
    assert parse_when("om 1,5 timmar", NOW) == at(17, 16, 0)


def test_english_decimal_point_is_understood_as_a_fraction():
    assert parse_when("in 1.5 hours", NOW) == at(17, 16, 0)


# ======================================================================================
# Scheduler — firing
# ======================================================================================


def test_timer_fires_exactly_once_after_its_delay(running):
    collector = Collector()
    started = time.monotonic()
    scheduler = running(collector)
    scheduler.add_timer(0.3 / 60, label="tea")

    assert collector.wait_for(1), "the 0.3 s timer never fired"
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < WAIT, f"timer fired after {elapsed:.3f} s"

    time.sleep(0.2)  # a repeat would show up here
    assert collector.labels == ["tea"]
    assert scheduler.pending() == []


def test_cancel_stops_a_timer_from_firing(running):
    collector = Collector()
    scheduler = running(collector)
    job = scheduler.add_timer(0.2 / 60, label="tea")

    assert scheduler.cancel(job.id) is True
    time.sleep(0.4)  # well past the moment it would have fired
    assert collector.labels == []
    assert scheduler.pending() == []


def test_cancel_returns_false_for_an_unknown_id(running):
    scheduler = running(Collector())
    assert scheduler.cancel("t999") is False
    assert scheduler.cancel("") is False


def test_cancel_accepts_the_spoken_label_case_insensitively(running):
    scheduler = running(Collector())
    scheduler.add_timer(5, label="Pasta")
    assert scheduler.cancel("pasta") is True
    assert scheduler.pending() == []


def test_pending_lists_jobs_soonest_first_and_shrinks_on_cancel(store):
    scheduler = Scheduler(Collector(), store)
    late = scheduler.add_timer(30, label="late")
    soon = scheduler.add_timer(1, label="soon")
    middle = scheduler.add_timer(10, label="middle")

    assert [job.label for job in scheduler.pending()] == ["soon", "middle", "late"]

    scheduler.cancel(middle.id)
    assert [job.label for job in scheduler.pending()] == ["soon", "late"]
    assert {job.id for job in scheduler.pending()} == {soon.id, late.id}


def test_a_callback_that_raises_does_not_kill_the_scheduler_thread(running):
    collector = Collector(raise_on="boom")
    scheduler = running(collector)
    scheduler.add_timer(0.05 / 60, label="boom")
    scheduler.add_timer(0.3 / 60, label="survivor")

    assert collector.wait_for(2), f"only {collector.labels} fired after the raising job"
    assert collector.labels == ["boom", "survivor"]


def test_two_threads_adding_timers_at_once_all_end_up_pending(store):
    scheduler = Scheduler(Collector(), store)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def add_ten(prefix: str) -> None:
        try:
            barrier.wait(timeout=WAIT)
            for index in range(10):
                scheduler.add_timer(30, label=f"{prefix}{index}")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=add_ten, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=WAIT)
        assert not thread.is_alive(), "adding timers deadlocked"

    assert errors == []
    jobs = scheduler.pending()
    assert len(jobs) == 20
    assert len({job.id for job in jobs}) == 20, "two threads were handed the same job id"


# ======================================================================================
# Scheduler — persistence
# ======================================================================================


def test_jobs_survive_a_restart_of_the_process(store):
    first = Scheduler(Collector(), store)
    first.add_timer(30, label="stew")
    first.add_timer(60, label="påminnelse å ä ö")

    second = Scheduler(Collector(), store)
    assert [job.label for job in second.pending()] == ["stew", "påminnelse å ä ö"]
    assert [job.kind for job in second.pending()] == ["timer", "timer"]


def test_a_job_cancelled_before_the_restart_stays_gone(store):
    first = Scheduler(Collector(), store)
    kept = first.add_timer(30, label="kept")
    dropped = first.add_timer(31, label="dropped")
    first.cancel(dropped.id)

    second = Scheduler(Collector(), store)
    assert [job.id for job in second.pending()] == [kept.id]


def test_a_job_that_was_due_while_jarvis_was_closed_fires_once_at_start(store, running):
    Scheduler(Collector(), store).add_timer(0, label="overdue")  # due already, never started

    collector = Collector()
    scheduler = running(collector, store)
    assert collector.wait_for(1), "the overdue job was never announced"

    time.sleep(0.2)
    assert collector.labels == ["overdue"]
    assert scheduler.pending() == []
    # ... and it is gone from the store, so the next restart stays quiet.
    assert Scheduler(Collector(), store).pending() == []


def test_ids_are_not_reused_after_a_restart(store):
    first = Scheduler(Collector(), store)
    first_id = first.add_timer(30, label="one").id

    second = Scheduler(Collector(), store)
    second_id = second.add_timer(30, label="two").id
    assert second_id != first_id
    assert len(second.pending()) == 2


def test_a_corrupt_store_starts_empty_instead_of_crashing(store):
    store.write_text("{ this is not json", encoding="utf-8")
    scheduler = Scheduler(Collector(), store)
    assert scheduler.pending() == []
    # It must still be usable afterwards.
    scheduler.add_timer(30, label="fresh")
    assert [job.label for job in scheduler.pending()] == ["fresh"]
    assert json.loads(store.read_text(encoding="utf-8"))["jobs"][0]["label"] == "fresh"


def test_an_empty_store_file_starts_empty(store):
    store.write_text("", encoding="utf-8")
    assert Scheduler(Collector(), store).pending() == []


def test_unreadable_job_entries_are_skipped_not_fatal(store):
    store.write_text(
        json.dumps({"version": 1, "jobs": ["nonsense", {"id": "t7", "due": 1.0, "label": "ok"}]}),
        encoding="utf-8",
    )
    jobs = Scheduler(Collector(), store).pending()
    assert [job.label for job in jobs] == ["ok"]


def test_a_missing_store_is_not_created_until_a_job_is_added(store):
    Scheduler(Collector(), store)
    assert not store.exists()


# ======================================================================================
# Scheduler — lifecycle
# ======================================================================================


def test_stop_is_safe_to_call_twice_and_returns_promptly(store):
    scheduler = Scheduler(Collector(), store)
    scheduler.add_timer(30, label="never")
    scheduler.start()

    started = time.monotonic()
    scheduler.stop()
    scheduler.stop()
    assert time.monotonic() - started < 1.0, "stop() hung"


def test_stop_before_start_is_a_no_op(store):
    scheduler = Scheduler(Collector(), store)
    scheduler.stop()
    assert scheduler.pending() == []


def test_start_is_idempotent_and_a_timer_still_fires_once(running):
    collector = Collector()
    scheduler = running(collector)
    scheduler.start()
    scheduler.start()
    scheduler.add_timer(0.2 / 60, label="tea")

    assert collector.wait_for(1), "the timer never fired"
    time.sleep(0.2)
    assert collector.labels == ["tea"], "a second worker thread double-fired the job"


def test_a_stopped_scheduler_does_not_fire(store):
    collector = Collector()
    scheduler = Scheduler(collector, store)
    scheduler.start()
    scheduler.add_timer(0.3 / 60, label="tea")
    scheduler.stop()

    time.sleep(0.5)
    assert collector.labels == []


def test_stopping_from_inside_the_callback_does_not_deadlock(store):
    done = threading.Event()
    scheduler: Scheduler

    def on_due(job: Job) -> None:
        scheduler.stop()
        done.set()

    scheduler = Scheduler(on_due, store)
    scheduler.add_timer(0.05 / 60, label="self-stop")
    scheduler.start()
    try:
        assert done.wait(WAIT), "the callback never completed — stop() deadlocked"
    finally:
        scheduler.stop()


# ======================================================================================
# Scheduler — reminders
# ======================================================================================


def test_add_reminder_schedules_the_parsed_moment(store):
    scheduler = Scheduler(Collector(), store)
    job = scheduler.add_reminder("call mother", "in 2 hours")

    assert job.kind == "reminder"
    assert job.text == "call mother"
    expected = datetime.now() + timedelta(hours=2)
    assert abs(job.due_at - expected) < timedelta(seconds=5)
    assert scheduler.pending() == [job]


def test_add_reminder_rejects_a_time_it_cannot_understand(store):
    scheduler = Scheduler(Collector(), store)
    with pytest.raises(ValueError):
        scheduler.add_reminder("call mother", "when pigs fly")
    assert scheduler.pending() == []


def test_add_timer_rejects_a_duration_that_is_not_a_number(store):
    scheduler = Scheduler(Collector(), store)
    with pytest.raises(ValueError):
        scheduler.add_timer("ten minutes")  # type: ignore[arg-type]
    assert scheduler.pending() == []


def test_a_negative_timer_is_treated_as_due_now(running):
    collector = Collector()
    scheduler = running(collector)
    scheduler.add_timer(-5, label="oops")
    assert collector.wait_for(1), "a past-due timer was never announced"
