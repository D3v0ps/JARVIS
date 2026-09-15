"""Per-turn latency measurement: end-of-speech to first spoken word.

One :class:`LatencyTracker` is created per voice turn (or reused and reset through
:meth:`LatencyTracker.start_turn`). The clock starts the moment the VAD reports end of
speech and every later milestone is stamped against it::

    tracker.start_turn()                 # end of speech
    tracker.mark("text")                 # transcription finished
    tracker.mark("first token")          # first token out of the model
    tracker.first_audio()                # first sample handed to the speakers
    tracker.summary()
    # "speech-end→text 420 ms · →first token 780 ms · →first audio 1180 ms"

Timings use :func:`time.perf_counter`, never wall clock, so an NTP step or a daylight
saving change cannot produce a negative turn. The tracker is thread-safe: the speaker
thread reports first audio while the brain thread is still marking tokens.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Optional

from jarvis.core.logging import get_logger, log_latency

__all__ = ["LatencyTracker", "FIRST_AUDIO_LABEL"]

#: Label used for the headline measurement recorded by :meth:`LatencyTracker.first_audio`.
FIRST_AUDIO_LABEL = "first audio"

_SEPARATOR = " · "  # middle dot
_ARROW = "→"
_ORIGIN = "speech-end"


class LatencyTracker:
    """One per turn. Milliseconds from end-of-speech to first spoken audio."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._log: logging.Logger = logger or get_logger("latency")
        self._lock = RLock()
        self._t0: float | None = None
        self._marks: dict[str, float] = {}
        self._first_audio_ms: float | None = None

    # --- lifecycle -------------------------------------------------------------------
    def start_turn(self) -> None:
        """Reset the tracker and stamp t0. Call this at end of speech."""
        with self._lock:
            self._t0 = time.perf_counter()
            self._marks = {}
            self._first_audio_ms = None
        self._log.debug("Latency turn started.")

    @property
    def started(self) -> bool:
        """``True`` once :meth:`start_turn` has stamped a reference point."""
        with self._lock:
            return self._t0 is not None

    # --- measurement -----------------------------------------------------------------
    def mark(self, label: str) -> float:
        """Record milliseconds since :meth:`start_turn` under ``label`` and return them."""
        now = time.perf_counter()
        clean = str(label or "").strip() or "unnamed"
        with self._lock:
            if self._t0 is None:
                # Marking without a turn is a wiring bug; recover instead of raising.
                self._log.warning(
                    "mark(%r) called before start_turn(); starting the turn now.", clean
                )
                self._t0 = now
            elapsed_ms = (now - self._t0) * 1000.0
            if clean in self._marks:
                self._log.debug(
                    "Latency mark %r recorded again (%.0f ms -> %.0f ms).",
                    clean, self._marks[clean], elapsed_ms,
                )
            self._marks[clean] = elapsed_ms
        return elapsed_ms

    def first_audio(self) -> float:
        """Record the headline "first audio" mark once per turn and return it.

        Later calls within the same turn are ignored and return the first value, so a
        speaker that plays several sentences still reports a single headline number.
        """
        with self._lock:
            if self._first_audio_ms is not None:
                return self._first_audio_ms
            elapsed_ms = self.mark(FIRST_AUDIO_LABEL)
            self._first_audio_ms = elapsed_ms
        try:
            log_latency(FIRST_AUDIO_LABEL, elapsed_ms)
        except Exception as exc:  # pragma: no cover - the log helper already guards itself
            self._log.warning("Could not log the first-audio latency: %s", exc)
        return elapsed_ms

    def marks(self) -> dict[str, float]:
        """Return a copy of every recorded mark, in insertion order."""
        with self._lock:
            return dict(self._marks)

    @property
    def first_audio_ms(self) -> float | None:
        """The headline first-audio measurement of this turn, or ``None``."""
        with self._lock:
            return self._first_audio_ms

    def elapsed_ms(self) -> float:
        """Milliseconds since :meth:`start_turn` without recording a mark."""
        with self._lock:
            if self._t0 is None:
                return 0.0
            return (time.perf_counter() - self._t0) * 1000.0

    # --- reporting -------------------------------------------------------------------
    def summary(self) -> str:
        """Render the marks in insertion order as one compact line.

        Example: ``"speech-end→text 420 ms · →first token 780 ms"``.
        Returns ``""`` when nothing was measured.
        """
        marks = self.marks()
        if not marks:
            return ""
        parts: list[str] = []
        for index, (label, ms) in enumerate(marks.items()):
            prefix = f"{_ORIGIN}{_ARROW}" if index == 0 else _ARROW
            parts.append(f"{prefix}{label} {ms:.0f} ms")
        return _SEPARATOR.join(parts)

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return self.summary() or "no latency marks"

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"LatencyTracker(marks={len(self._marks)}, started={self.started})"
