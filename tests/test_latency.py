"""Per-turn latency: the number that decides whether JARVIS feels instant or sluggish."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from jarvis.core.latency import FIRST_AUDIO_LABEL, LatencyTracker


@pytest.fixture(autouse=True)
def restore_jarvis_logger():
    """first_audio() logs through the 'jarvis' logger; leave it exactly as we found it."""
    logger = logging.getLogger("jarvis")
    before = list(logger.handlers)
    propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = propagate


def busy_wait(seconds: float = 0.005) -> None:
    """Burn a measurable amount of monotonic time without sleeping the suite away."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def test_marks_are_reported_in_the_order_they_were_recorded():
    tracker = LatencyTracker()
    tracker.start_turn()
    tracker.mark("text")
    tracker.mark("first token")
    tracker.mark(FIRST_AUDIO_LABEL)

    assert list(tracker.marks()) == ["text", "first token", FIRST_AUDIO_LABEL]


def test_marks_grow_with_time_since_the_turn_started():
    tracker = LatencyTracker()
    tracker.start_turn()
    first = tracker.mark("text")
    busy_wait()
    second = tracker.mark("first token")

    assert 0.0 <= first < second


def test_mark_returns_the_value_it_records():
    tracker = LatencyTracker()
    tracker.start_turn()
    returned = tracker.mark("text")
    assert tracker.marks()["text"] == returned


def test_start_turn_clears_the_previous_turn():
    tracker = LatencyTracker()
    tracker.start_turn()
    tracker.mark("text")
    tracker.first_audio()

    tracker.start_turn()

    assert tracker.marks() == {}
    assert tracker.first_audio_ms is None


def test_marks_returns_a_copy_that_cannot_corrupt_the_tracker():
    tracker = LatencyTracker()
    tracker.start_turn()
    tracker.mark("text")
    snapshot = tracker.marks()
    snapshot["text"] = 999_999.0

    assert tracker.marks()["text"] != 999_999.0


def test_first_audio_is_idempotent_within_a_turn():
    """Several spoken sentences must still report one headline number."""
    tracker = LatencyTracker()
    tracker.start_turn()
    first = tracker.first_audio()
    busy_wait()
    second = tracker.first_audio()

    assert second == first
    assert tracker.first_audio_ms == first


def test_first_audio_records_a_mark_under_the_headline_label():
    tracker = LatencyTracker()
    tracker.start_turn()
    value = tracker.first_audio()

    assert tracker.marks()[FIRST_AUDIO_LABEL] == value


def test_first_audio_survives_a_logging_helper_that_raises(monkeypatch):
    import jarvis.core.latency as latency_module

    def explode(label: str, ms: float) -> None:
        raise RuntimeError("the log file went away")

    monkeypatch.setattr(latency_module, "log_latency", explode)
    tracker = LatencyTracker()
    tracker.start_turn()

    assert tracker.first_audio() >= 0.0


def test_marking_before_start_turn_is_ignored_rather_than_invented():
    """A measurement with no reference point is meaningless.

    JARVIS speaks outside a turn all the time - the startup greeting, a chime, a
    timer going off. Those used to open a turn on the spot and report a 0 ms
    latency, which was both a warning in the console and a lie in the log.
    """
    tracker = LatencyTracker()

    assert tracker.mark("text") == 0.0
    assert tracker.marks() == {}, "nothing should have been recorded"
    assert not tracker.started

    # And a real turn afterwards still measures correctly.
    tracker.start_turn()
    assert tracker.mark("text") >= 0.0
    assert "text" in tracker.marks()


def test_summary_names_every_mark_in_order():
    tracker = LatencyTracker()
    tracker.start_turn()
    tracker.mark("text")
    tracker.mark("first token")
    tracker.first_audio()

    summary = tracker.summary()

    assert summary.startswith("speech-end→text ")
    assert "→first token " in summary
    assert f"→{FIRST_AUDIO_LABEL} " in summary
    assert summary.index("text") < summary.index("first token") < summary.index(FIRST_AUDIO_LABEL)
    assert summary.count(" ms") == 3


def test_summary_is_empty_before_anything_is_measured():
    assert LatencyTracker().summary() == ""


def test_elapsed_ms_is_zero_before_the_turn_starts():
    assert LatencyTracker().elapsed_ms() == 0.0


def test_timings_use_perf_counter_so_a_frozen_wall_clock_cannot_break_them(monkeypatch):
    """A clock step (NTP, DST) must not produce a zero or negative turn."""
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)

    tracker = LatencyTracker()
    tracker.start_turn()
    busy_wait()
    elapsed = tracker.mark("text")

    assert elapsed > 0.0


def test_two_threads_reporting_first_audio_agree_on_one_value():
    tracker = LatencyTracker()
    tracker.start_turn()
    ready = threading.Barrier(2, timeout=5)
    seen: list[float] = []

    def report() -> None:
        ready.wait()
        seen.append(tracker.first_audio())

    threads = [threading.Thread(target=report) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), "first_audio() blocked forever"

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_speech_outside_a_turn_is_not_measured():
    """The startup greeting is not a turn, and reporting it as 0 ms was noise."""
    from jarvis.core.latency import LatencyTracker

    tracker = LatencyTracker()
    assert tracker.first_audio() == 0.0
    assert tracker.marks() == {}


def test_end_turn_closes_the_measurement():
    from jarvis.core.latency import LatencyTracker

    tracker = LatencyTracker()
    tracker.start_turn()
    assert tracker.first_audio() >= 0.0
    tracker.end_turn()
    assert tracker.first_audio() == 0.0, "speech after the turn must not be measured"
