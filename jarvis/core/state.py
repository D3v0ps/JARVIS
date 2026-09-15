"""The assistant's current state and the thread-safe bus that publishes it.

Audio capture, the brain, the speaker and the UI all live on different threads, so the
state they share needs one owner. :class:`StateBus` is it: an ``RLock``-protected holder
that notifies observers *outside* the lock, so a slow or misbehaving observer can never
deadlock the audio loop.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any, Callable

from jarvis.core.logging import get_logger

__all__ = ["AssistantState", "StateBus"]

_logger = get_logger("state")


class AssistantState(str, Enum):
    """What JARVIS is doing right now. Also the key for the overlay's colour theme."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    PAUSED = "paused"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "AssistantState":
        """Accept an :class:`AssistantState` or its string value; raise on anything else."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown assistant state {value!r}; expected one of {[s.value for s in cls]}"
            ) from exc


class StateBus:
    """Thread-safe current-state holder with observers. The UI subscribes to it.

    Exports:
        ``state`` (property), ``set(state)``, ``subscribe(callback) -> unsubscribe``,
        ``set_text(text)``, ``text`` (property) and the convenience
        ``wait_for(state, timeout) -> bool`` used by tests and by code that needs to
        block until a transition happens.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._state: AssistantState = AssistantState.IDLE
        self._text: str = ""
        self._observers: list[Callable[[AssistantState], None]] = []

    # --- state ------------------------------------------------------------------
    @property
    def state(self) -> AssistantState:
        """The current state."""
        with self._lock:
            return self._state

    def set(self, state: AssistantState) -> None:
        """Move to ``state``. No-op when unchanged; observers are notified outside the lock."""
        target = AssistantState.coerce(state)
        with self._lock:
            if target == self._state:
                return
            previous = self._state
            self._state = target
            observers = list(self._observers)
            self._changed.notify_all()

        _logger.debug("State %s -> %s", previous.value, target.value)
        self._notify(observers, target)

    def _notify(self, observers: list[Callable[[AssistantState], None]], state: AssistantState) -> None:
        """Call every observer, isolating failures so one bad observer cannot stop the rest."""
        for callback in observers:
            try:
                callback(state)
            except Exception:
                _logger.exception(
                    "State observer %s failed for state %s",
                    getattr(callback, "__name__", repr(callback)),
                    state.value,
                )

    def subscribe(self, callback: Callable[[AssistantState], None]) -> Callable[[], None]:
        """Register ``callback(state)``; returns a callable that unsubscribes it."""
        if not callable(callback):
            raise TypeError(f"StateBus.subscribe() needs a callable, got {type(callback).__name__}")
        with self._lock:
            self._observers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._observers.remove(callback)
                except ValueError:
                    _logger.debug("State observer was already unsubscribed")

        return unsubscribe

    def wait_for(self, state: AssistantState, timeout: float | None = None) -> bool:
        """Block until the bus is in ``state``. Returns False on timeout."""
        target = AssistantState.coerce(state)
        with self._lock:
            return bool(self._changed.wait_for(lambda: self._state == target, timeout))

    # --- caption ----------------------------------------------------------------
    def set_text(self, text: str) -> None:
        """Set the optional caption shown next to the overlay ring."""
        value = "" if text is None else str(text)
        with self._lock:
            if value == self._text:
                return
            self._text = value
        _logger.debug("State caption: %s", value)

    @property
    def text(self) -> str:
        """The current caption ("" when there is none)."""
        with self._lock:
            return self._text

    # --- misc -------------------------------------------------------------------
    @property
    def observer_count(self) -> int:
        """How many observers are currently subscribed (diagnostics, tests)."""
        with self._lock:
            return len(self._observers)

    def __repr__(self) -> str:
        with self._lock:
            return f"<StateBus state={self._state.value} observers={len(self._observers)} text={self._text!r}>"
