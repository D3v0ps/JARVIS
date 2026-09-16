"""What the desk window is told, and why a turn never waits for it to listen.

A window is a spectator. It may be shut, opened an hour into a session, left
minimised for twenty minutes while its socket quietly stops being drained, or three
of them may be open at once - and none of that may cost the turn a millisecond. So
nothing here pushes at a window: the assistant publishes into the bus, each window
reads a queue of its own, and a window that has stopped reading loses its *oldest*
events rather than growing a backlog inside the assistant or blocking the thread that
is speaking.

:meth:`DeskBus.replay` is what makes a window openable late. The last few hundred
events that still mean something are kept, so a window opened after the conversation
began still shows the conversation. Log lines and telemetry are deliberately not
kept: an hour later they are noise, and they are also the two loudest event types.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CLOSED", "REPLAYABLE", "DeskBus", "DeskEvent", "DeskLogHandler"]

#: Event types worth showing to a window that opened late. ``log`` and ``status`` are
#: absent on purpose - a log line and a CPU reading from an hour ago say nothing.
REPLAYABLE: frozenset[str] = frozenset(
    {"state", "heard", "sentence", "tool", "note", "latency"}
)

#: The type of the event :meth:`DeskBus.close` hands every subscriber, so a server
#: thread parked on ``q.get()`` leaves at once instead of on its next timeout.
CLOSED = "closed"

#: How many events one window may fall behind before it starts losing the oldest.
QUEUE_LIMIT = 256


@dataclass(frozen=True)
class DeskEvent:
    """One thing that happened, timestamped once and shared by every subscriber.

    Frozen because the same object is handed to every window: nobody gets to edit
    what another window is about to read.
    """

    type: str
    payload: dict
    at: float = field(default_factory=time.time)

    def as_json(self) -> dict[str, Any]:
        """The flat frame the page expects: the payload with ``type`` and ``at`` on it."""
        return {**self.payload, "type": self.type, "at": self.at}


class DeskBus:
    """A fan-out with a memory: one lock, one ring buffer, a bounded queue per window.

    Every public method is safe from any thread and none of them raise into the
    assistant. Dropping an event is always preferable to blocking the voice.
    """

    def __init__(self, *, history: int = 200, queue_size: int = QUEUE_LIMIT) -> None:
        self._lock = threading.Lock()
        self._history: deque[DeskEvent] = deque(maxlen=max(1, int(history)))
        self._queue_size = max(8, int(queue_size))
        self._subscribers: dict[queue.Queue, int] = {}
        self._closed = False

    # --- publishing ---------------------------------------------------------------
    def publish(self, type: str, /, **payload) -> DeskEvent:
        """Hand one event to every subscriber. Returns it; never raises, whatever happens.

        The event is built before the lock so the timestamp is the moment it happened
        rather than the moment a busy window let go of the bus.
        """
        event = DeskEvent(type=str(type), payload=dict(payload), at=time.time())
        try:
            with self._lock:
                if self._closed:
                    return event
                if event.type in REPLAYABLE:
                    self._history.append(event)
                for subscriber in list(self._subscribers):
                    self._offer(subscriber, event)
        except Exception:  # noqa: BLE001 - a broken window must not break a turn
            pass
        return event

    def _offer(self, subscriber: queue.Queue, event: DeskEvent) -> None:
        """Put ``event`` on one queue, making room by discarding its oldest if need be."""
        while True:
            try:
                subscriber.put_nowait(event)
                return
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except Exception:  # noqa: BLE001 - it drained itself between the two calls
                    return
                self._subscribers[subscriber] = self._subscribers.get(subscriber, 0) + 1
            except Exception:  # noqa: BLE001 - a subscriber that mangled its own queue
                return

    # --- subscribing --------------------------------------------------------------
    def subscribe(self) -> "queue.Queue[DeskEvent]":
        """A queue of this window's own, bounded so a dead window cannot eat memory."""
        subscriber: "queue.Queue[DeskEvent]" = queue.Queue(maxsize=self._queue_size)
        with self._lock:
            if self._closed:
                subscriber.put_nowait(DeskEvent(CLOSED, {}, time.time()))
                return subscriber
            self._subscribers[subscriber] = 0
        return subscriber

    def unsubscribe(self, q: queue.Queue) -> None:
        """Forget a window. Safe to call twice, and from the reader's own thread."""
        with self._lock:
            self._subscribers.pop(q, None)

    def dropped(self, q: queue.Queue) -> int:
        """How many events this window missed while it was not reading.

        The window is told, so it can say "you missed some" rather than silently
        showing a conversation with holes in it.
        """
        with self._lock:
            return int(self._subscribers.get(q, 0))

    @property
    def subscribers(self) -> int:
        """How many windows are listening."""
        with self._lock:
            return len(self._subscribers)

    @property
    def closed(self) -> bool:
        return self._closed

    # --- history ------------------------------------------------------------------
    def replay(self) -> list[DeskEvent]:
        """The conversation so far, oldest first, for a window that has just opened."""
        with self._lock:
            return list(self._history)

    def close(self) -> None:
        """Wake every subscriber and stop accepting events. Safe to call twice."""
        farewell = DeskEvent(CLOSED, {}, time.time())
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for subscriber in list(self._subscribers):
                self._offer(subscriber, farewell)
            self._subscribers.clear()


class DeskLogHandler(logging.Handler):
    """Tails the log into the window's drawer without ever logging about doing so.

    Publishing can itself log - a library three frames down, a warning from the bus -
    and a handler that re-entered on its own record would recurse until the stack
    gave out. A thread-local flag makes that impossible, and anything else that goes
    wrong goes to :meth:`logging.Handler.handleError`, which is the convention and,
    more importantly, silent.
    """

    def __init__(self, bus: DeskBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.bus = bus
        self._busy = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._busy, "active", False):
            return
        self._busy.active = True
        try:
            text = " ".join(self.format(record).splitlines()).strip()
            self.bus.publish("log", level=record.levelname, name=record.name, text=text)
        except Exception:  # noqa: BLE001 - logging must never take a turn down
            self.handleError(record)
        finally:
            self._busy.active = False
