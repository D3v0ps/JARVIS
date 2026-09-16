"""The fan-out that lets a window watch a turn without ever slowing one down.

The bus is the only thing standing between a minimised window whose socket nobody is
draining and the thread that is speaking, so every test here is really one question:
does the assistant keep going when the spectator misbehaves? The replay buffer gets
the same treatment, because a window opened an hour into a session must see the
conversation and must not see an hour of log lines.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import pytest

from jarvis.desk.bus import CLOSED, REPLAYABLE, DeskBus, DeskEvent, DeskLogHandler


@pytest.fixture
def bus():
    made = DeskBus()
    yield made
    made.close()


def drain(q: "queue.Queue[DeskEvent]") -> list[DeskEvent]:
    """Everything waiting for this subscriber right now."""
    events: list[DeskEvent] = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            return events


# --- publishing ----------------------------------------------------------------------
def test_publish_returns_the_event_it_built_with_a_timestamp_on_it(bus):
    before = time.time()

    event = bus.publish("sentence", text="Certainly, sir.")

    assert isinstance(event, DeskEvent)
    assert event.type == "sentence"
    assert event.payload == {"text": "Certainly, sir."}
    assert before <= event.at <= time.time()


def test_a_subscriber_is_handed_every_event_published_after_it_subscribed(bus):
    q = bus.subscribe()

    bus.publish("state", state="thinking")
    bus.publish("sentence", text="It is cold.")

    assert [(e.type, e.payload) for e in drain(q)] == [
        ("state", {"state": "thinking"}),
        ("sentence", {"text": "It is cold."}),
    ]


def test_every_open_window_gets_the_same_event_object(bus):
    """One event, one timestamp: two windows must not disagree about when it happened."""
    first, second = bus.subscribe(), bus.subscribe()

    published = bus.publish("heard", text="what is the weather")

    assert drain(first) == [published]
    assert drain(second) == [published]


def test_an_event_cannot_be_edited_by_the_window_that_receives_it(bus):
    q = bus.subscribe()
    bus.publish("note", text="The microphone is gone, sir.")

    event = drain(q)[0]

    with pytest.raises(Exception):
        event.type = "sentence"  # type: ignore[misc]


def test_a_window_that_stopped_reading_loses_its_oldest_events_not_the_publisher(bus):
    """The backlog is the window's problem. The turn must not pay for it."""
    small = DeskBus(queue_size=8)
    q = small.subscribe()

    for index in range(40):
        small.publish("sentence", text=str(index))

    received = drain(q)
    assert len(received) == 8, "the queue is bounded, so memory cannot grow"
    assert [e.payload["text"] for e in received] == [str(i) for i in range(32, 40)], (
        "what survives is the newest, because that is what a window wants to show"
    )


def test_the_dropped_counter_tells_the_window_how_much_it_missed(bus):
    small = DeskBus(queue_size=8)
    q = small.subscribe()

    for index in range(20):
        small.publish("sentence", text=str(index))

    assert small.dropped(q) == 12


def test_publish_never_raises_when_a_subscribers_queue_is_broken(bus):
    """A window may do anything with the queue it was given; the turn continues."""
    q = bus.subscribe()

    def explode(item):
        raise RuntimeError("this window is on fire")

    q.put_nowait = explode  # type: ignore[method-assign]

    event = bus.publish("state", state="speaking")

    assert event.type == "state"


def test_unsubscribing_stops_delivery_and_is_safe_to_repeat(bus):
    q = bus.subscribe()

    bus.unsubscribe(q)
    bus.unsubscribe(q)
    bus.publish("sentence", text="nobody is listening")

    assert drain(q) == []
    assert bus.subscribers == 0


def test_publishing_from_many_threads_loses_nothing_from_the_history():
    """Threads for audio, the brain and the speaker all narrate at once."""
    bus = DeskBus(history=1000)

    def spam(worker: int) -> None:
        for index in range(50):
            bus.publish("sentence", text=f"{worker}-{index}")

    threads = [threading.Thread(target=spam, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(bus.replay()) == 400


# --- the replay buffer ----------------------------------------------------------------
def test_replay_gives_a_window_opened_late_the_conversation_so_far(bus):
    bus.publish("heard", text="what is the weather")
    bus.publish("sentence", text="Cold, sir.")

    replayed = bus.replay()

    assert [e.type for e in replayed] == ["heard", "sentence"], "oldest first"


def test_replay_keeps_only_what_still_means_something_an_hour_later(bus):
    for kind in sorted(REPLAYABLE):
        bus.publish(kind)
    bus.publish("log", level="INFO", name="jarvis", text="a line")
    bus.publish("status", cpu=12)
    bus.publish("hello", version="1")

    kept = {event.type for event in bus.replay()}

    assert kept == set(REPLAYABLE)
    assert "log" not in kept and "status" not in kept


def test_replay_is_bounded_so_a_long_session_cannot_grow_forever():
    bus = DeskBus(history=10)

    for index in range(100):
        bus.publish("sentence", text=str(index))

    replayed = bus.replay()
    assert len(replayed) == 10
    assert replayed[0].payload["text"] == "90"


def test_replay_hands_back_a_list_the_caller_cannot_use_to_edit_the_history(bus):
    bus.publish("sentence", text="one")

    bus.replay().clear()

    assert len(bus.replay()) == 1


# --- closing ---------------------------------------------------------------------------
def test_close_wakes_a_subscriber_that_is_blocked_waiting_for_an_event(bus):
    """A server thread parked on q.get() must leave when the window closes, not on a timeout."""
    q = bus.subscribe()
    woken: "queue.Queue[DeskEvent]" = queue.Queue()

    def reader() -> None:
        woken.put(q.get(timeout=5))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(0.05)
    bus.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert woken.get_nowait().type == CLOSED


def test_publishing_after_close_reaches_nobody_and_does_not_raise(bus):
    q = bus.subscribe()
    bus.close()
    drain(q)

    bus.publish("sentence", text="too late")

    assert drain(q) == []
    assert bus.closed is True


def test_subscribing_to_a_closed_bus_answers_at_once_instead_of_hanging(bus):
    bus.close()

    q = bus.subscribe()

    assert q.get_nowait().type == CLOSED


def test_close_is_safe_to_call_twice(bus):
    bus.close()
    bus.close()

    assert bus.closed is True


# --- the log handler ---------------------------------------------------------------------
def test_the_log_handler_publishes_the_level_the_logger_and_the_line(bus):
    handler = DeskLogHandler(bus)
    logger = logging.getLogger("jarvis.test.desk")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    q = bus.subscribe()
    try:
        logger.warning("The microphone is gone, %s.", "sir")
    finally:
        logger.removeHandler(handler)

    event = drain(q)[0]
    assert event.type == "log"
    assert event.payload == {
        "level": "WARNING",
        "name": "jarvis.test.desk",
        "text": "The microphone is gone, sir.",
    }


def test_the_log_handler_ignores_records_below_its_level(bus):
    """The drawer tails INFO and worse; debug chatter would drown a turn in it."""
    handler = DeskLogHandler(bus, level=logging.WARNING)
    logger = logging.getLogger("jarvis.test.level")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    q = bus.subscribe()
    try:
        logger.info("chatter")
        logger.error("something real")
    finally:
        logger.removeHandler(handler)

    assert [e.payload["text"] for e in drain(q)] == ["something real"]


def test_the_log_handler_keeps_one_record_on_one_line(bus):
    """The drawer tails a log; a traceback must not become forty rows in it."""
    handler = DeskLogHandler(bus)
    q = bus.subscribe()
    record = logging.LogRecord(
        "jarvis", logging.ERROR, __file__, 1, "first\nsecond\nthird", None, None
    )

    handler.handle(record)

    assert drain(q)[0].payload["text"] == "first second third"


def test_the_log_handler_does_not_recurse_when_publishing_logs(bus):
    """Publishing can log three frames down; a handler that re-entered would never stop."""
    logger = logging.getLogger("jarvis.test.recursion")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    calls: list[str] = []

    class TalkativeBus:
        def publish(self, type: str, /, **payload) -> None:
            calls.append(type)
            logger.info("publishing said something")

    handler = DeskLogHandler(TalkativeBus())
    logger.addHandler(handler)
    try:
        logger.info("the first line")
    finally:
        logger.removeHandler(handler)

    assert calls == ["log"], "the record emitted while publishing must not re-enter"


def test_the_log_handler_never_raises_into_the_logging_call(bus):
    class BrokenBus:
        def publish(self, type: str, /, **payload) -> None:
            raise RuntimeError("the bus is gone")

    handler = DeskLogHandler(BrokenBus())
    handler.handleError = lambda record: None  # type: ignore[method-assign]
    record = logging.LogRecord("jarvis", logging.INFO, __file__, 1, "a line", None, None)

    handler.handle(record)  # must simply return


def test_a_handler_that_failed_once_still_works_afterwards(bus):
    """The re-entrancy flag has to be cleared on the way out, failure or not."""
    handler = DeskLogHandler(bus)
    handler.handleError = lambda record: None  # type: ignore[method-assign]
    q = bus.subscribe()
    broken = logging.LogRecord("jarvis", logging.INFO, __file__, 1, "%d", ("not a number",), None)

    handler.handle(broken)
    handler.handle(logging.LogRecord("jarvis", logging.INFO, __file__, 1, "fine", None, None))

    assert [e.payload["text"] for e in drain(q)] == ["fine"]
