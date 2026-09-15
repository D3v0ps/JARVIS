"""Sentence queue -> synthesis thread -> player.

This is what makes JARVIS feel fast. The brain hands over one finished sentence at a
time; :meth:`Speaker.enqueue` returns immediately and a single worker thread synthesises
the next sentence *while the previous one is still playing*::

    brain thread   enqueue("Good evening, sir.")   enqueue("The weather is mild.")
    synth worker      synthesize #1 -> play #1        synthesize #2 -> play #2
    player thread                  |== audio #1 ==|                |== audio #2 ==|

The worker never waits on playback: clips go into the :class:`~jarvis.audio.player.Player`
queue, which keeps its own ordering and can be cut mid-clip for barge-in.

Exports: :class:`Speaker` (``enqueue``, ``say``, ``play_chime``, ``stop``, ``wait``,
``on_finished``, ``is_speaking``, ``close``) and :func:`detect_language`.

``on_finished(callback)`` registers a callable — invoked on the worker thread with no
arguments, returning an unsubscribe callable — for the moment the sentence queue has
drained *and* the player has gone quiet. The callback decides what state comes next, so
the assistant can go to LISTENING for a follow-up; with no callback registered the
speaker returns the state bus to IDLE itself. Callbacks do **not** fire after
:meth:`Speaker.stop`: barge-in means the caller is already driving the next state.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from jarvis.audio.chimes import DEFAULT_SAMPLE_RATE, apply_volume, error_chime
from jarvis.brain.sentences import SentenceSplitter, clean_for_speech
from jarvis.core.logging import get_logger
from jarvis.core.state import AssistantState, StateBus

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the import graph light
    from jarvis.audio.player import Player
    from jarvis.core.latency import LatencyTracker

__all__ = ["Speaker", "detect_language", "SWEDISH_FUNCTION_WORDS", "DEFAULT_CHIME_VOLUME"]

#: How long the worker waits on an empty queue before re-checking whether a burst ended.
POLL_S = 0.05
#: Bounded sentence queue: deep enough for any reply, shallow enough to expose a bug.
QUEUE_MAX = 64
#: Consecutive quiet polls before a burst is declared finished. Two polls 50 ms apart
#: cannot both land in the microscopic gap between two clips queued in the player.
IDLE_CONFIRMATIONS = 2
#: Fallback for ``audio.chime_volume`` when config.yaml cannot be read.
DEFAULT_CHIME_VOLUME = 0.35
#: Latency mark written when the player confirms that sound really left the speakers.
PLAYBACK_LABEL = "audio out"

#: Very common Swedish function words; none of them is an English word.
SWEDISH_FUNCTION_WORDS = frozenset(
    {"och", "att", "jag", "är", "inte", "det", "på", "för", "med", "kan"}
)
#: Share of letters that must be å/ä/ö before text counts as Swedish on its own.
SWEDISH_CHAR_RATIO = 0.03
_SWEDISH_LETTERS = set("åäöÅÄÖ")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """Guess the language of ``text``: ``"sv"`` for Swedish, otherwise ``"en"``.

    Deliberately cheap — it runs per sentence on the synthesis thread. Each word from
    :data:`SWEDISH_FUNCTION_WORDS` scores one point; å/ä/ö score two points when they
    are at least :data:`SWEDISH_CHAR_RATIO` of the letters and one point when they
    merely occur. Two points mean Swedish, so short or ambiguous text stays English —
    the right bias, because the configured voice then keeps talking instead of flapping.
    """
    if not text:
        return "en"
    words = _WORD_RE.findall(str(text).lower())
    if not words:
        return "en"
    score = sum(1 for word in words if word in SWEDISH_FUNCTION_WORDS)
    letters = sum(len(word) for word in words)
    special = sum(1 for char in str(text) if char in _SWEDISH_LETTERS)
    if special:
        score += 2 if letters and special / letters >= SWEDISH_CHAR_RATIO else 1
    return "sv" if score >= 2 else "en"


_config_lock = threading.Lock()
_config_cache: Any = None
_config_tried = False


def _shared_config() -> Any:
    """Load ``config.yaml`` once, lazily; returns None when it cannot be read."""
    global _config_cache, _config_tried
    with _config_lock:
        if not _config_tried:
            _config_tried = True
            try:
                from jarvis.config import Config

                _config_cache = Config.load()
            except Exception as exc:
                logging.getLogger("jarvis.tts.speaker").debug(
                    "Using built-in defaults, config.yaml is unavailable: %s", exc
                )
        return _config_cache


def _engine_name(engine: Any) -> str:
    """A readable name for an engine, whatever it exposes."""
    name = getattr(engine, "name", None)
    return name.strip() if isinstance(name, str) and name.strip() else type(engine).__name__


def _short_language(value: Any) -> str | None:
    """Normalise a language name to a two-letter code, or None when there is none."""
    return value.strip().lower()[:2] if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class _Utterance:
    """One queued sentence plus the barge-in generation it was queued in."""

    text: str
    language: str | None
    generation: int


class Speaker:
    """Sentence queue -> synth thread -> Player.

    Speech starts on sentence one while the model is still generating sentence two.

    Args:
        engine: The primary TTS engine (``jarvis.tts.engine.TTSEngine``).
        player: The audio player that owns the playback queue.
        state: The state bus, set to SPEAKING while audio is out.
        language: Forced language ("en" / "sv"), or None to detect it per sentence.
        logger: Optional logger.
        latency: Optional per-turn tracker; the first clip of a burst records the
            headline first-audio number and the summary is logged once per turn.
    """

    def __init__(
        self,
        engine: Any,
        player: "Player",
        state: StateBus,
        *,
        language: str | None = None,
        logger: logging.Logger | None = None,
        latency: "LatencyTracker | None" = None,
    ) -> None:
        self._engine = engine
        self._player = player
        self._state = state
        self._language = _short_language(language)
        self.log = logger if logger is not None else get_logger("tts.speaker")
        self._latency = latency

        cfg = _shared_config()
        volume = DEFAULT_CHIME_VOLUME if cfg is None else cfg.get("audio.chime_volume", DEFAULT_CHIME_VOLUME)
        #: Linear gain applied to chimes, from ``audio.chime_volume``.
        self.chime_volume = float(DEFAULT_CHIME_VOLUME if volume is None else volume)

        self._queue: "queue.Queue[_Utterance | None]" = queue.Queue(maxsize=QUEUE_MAX)
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._cancel = threading.Event()      # set by stop(): drop in-flight synthesis
        self._worker: threading.Thread | None = None
        self._finished: list[Callable[[], None]] = []

        # _generation is bumped by stop()/close() so stale work is dropped; _pending
        # counts sentences queued or being synthesised; _speaking marks a burst that
        # has started playing and not yet drained.
        self._generation = self._pending = self._idle_polls = 0
        self._speaking = self._closed = self._error_chime_done = self._summary_logged = False

        self._engines: dict[str, Any] = {}   # language -> engine; a switch costs nothing twice
        self._selector_failed = False

        hook = getattr(player, "on_first_audio", None)
        if callable(hook):
            try:
                hook(self._on_player_first_audio)
            except Exception as exc:
                self.log.debug("Player does not support a first-audio hook: %s", exc)

    # -- public API ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True while sentences are queued, being synthesised, or still playing."""
        with self._lock:
            return self._speaking or self._pending > 0

    def on_finished(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register ``callback()`` for the end of a burst; returns an unsubscribe callable."""
        if not callable(callback):
            raise TypeError(f"on_finished() needs a callable, got {type(callback).__name__}")
        with self._lock:
            self._finished.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._finished:
                    self._finished.remove(callback)

        return unsubscribe

    def enqueue(self, sentence: str) -> None:
        """Queue one sentence for speech and return immediately."""
        text = clean_for_speech(sentence or "")
        if not text:
            return
        with self._lock:
            if self._closed:
                self.log.debug("Speaker is closed; dropping %r.", text[:40])
                return
            self._cancel.clear()
            item = _Utterance(text, self._language, self._generation)
            self._pending += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._pending = max(0, self._pending - 1)
                self._cv.notify_all()
            self.log.warning("Speech queue is full (%d sentences); dropping %r.", QUEUE_MAX, text[:40])
            return
        self._ensure_worker()

    def say(self, text: str, *, blocking: bool = True) -> None:
        """Split ``text`` into sentences, queue them all, and optionally wait."""
        cleaned = clean_for_speech(text or "")
        if not cleaned:
            return
        splitter = SentenceSplitter()
        sentences = splitter.feed(cleaned)
        tail = splitter.flush()
        if tail:
            sentences.append(tail)
        for sentence in sentences:
            self.enqueue(sentence)
        if blocking:
            self.wait()

    def play_chime(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play an interface chime at the configured volume, bypassing synthesis.

        Works while speech is queued: the clip simply joins the player's own queue.
        """
        try:
            data = apply_volume(np.asarray(audio, dtype=np.float32), self.chime_volume)
            if data.size:
                self._player.play(data, int(sample_rate))
        except Exception as exc:
            self.log.error("Could not play a chime: %s", exc)

    def stop(self) -> None:
        """Barge-in: clear the queue, drop in-flight synthesis, cut playback. Returns at once."""
        with self._lock:
            self._generation += 1
            self._cancel.set()
            dropped = self._drain()
            self._pending = self._idle_polls = 0
            self._speaking = False
            self._cv.notify_all()
        try:
            self._player.stop()
        except Exception as exc:
            self.log.error("Player did not stop cleanly: %s", exc)
        if dropped:
            self.log.debug("Speech interrupted: %d queued sentence(s) discarded.", dropped)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every queued sentence has been spoken (or ``timeout`` expires)."""
        with self._lock:
            done = self._cv.wait_for(lambda: self._pending == 0 and not self._speaking, timeout)
            pending = self._pending
        if not done:
            self.log.debug("wait() timed out with %d sentence(s) still pending.", pending)

    def close(self) -> None:
        """Stop the worker and forget the cached engines. Safe to call twice.

        The engine, the player and the engines handed out by ``jarvis.tts.engine`` (which
        caches and shares them) belong to whoever built them, so nothing is closed here.
        """
        with self._lock:
            worker, already = self._worker, self._closed
            if already:
                return
            self._closed = True
            self._generation += 1
            self._cancel.set()
            self._drain()
            self._pending = 0
            self._speaking = False
            self._cv.notify_all()
        try:
            self._queue.put_nowait(None)  # sentinel: wakes the worker immediately
        except queue.Full:
            pass
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                self.log.warning("Speech worker did not stop within 2 seconds.")
        with self._lock:
            self._worker = None
            self._engines = {}

    # -- synthesis worker --------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Start the synthesis worker on first use."""
        with self._lock:
            if self._closed or (self._worker is not None and self._worker.is_alive()):
                return
            self._worker = threading.Thread(target=self._run, name="jarvis-tts", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        """Take sentences off the queue, synthesise them, hand the audio to the player."""
        while True:
            try:
                item = self._queue.get(timeout=POLL_S)
            except queue.Empty:
                self._poll_finish()
                continue
            if item is None:
                break
            try:
                self._handle(item)
            except Exception:  # one bad sentence must never wedge the queue
                self.log.exception("Speech worker recovered from an unexpected failure.")
            finally:
                with self._lock:
                    if item.generation == self._generation:
                        self._pending = max(0, self._pending - 1)
                    self._cv.notify_all()
        self.log.debug("Speech worker stopped.")

    def _handle(self, item: _Utterance) -> None:
        """Synthesise one sentence and queue it, unless barge-in got there first."""
        if self._stale(item.generation):
            self.log.debug("Dropping %r: speech was interrupted.", item.text[:40])
            return
        audio, rate = self._synthesize(item)
        if audio is None:
            return
        # Checked again right after synthesis: a sentence that was rendered but not yet
        # played must stay silent after stop().
        if self._stale(item.generation):
            self.log.debug("Dropping synthesised %r: speech was interrupted.", item.text[:40])
            return
        self._begin_burst()
        try:
            self._player.play(audio, rate)
        except Exception as exc:
            self.log.error("Could not queue %r for playback: %s", item.text[:40], exc)

    def _stale(self, generation: int) -> bool:
        """True when this work belongs to a burst that has since been cancelled."""
        with self._lock:
            return self._cancel.is_set() or generation != self._generation or self._closed

    def _synthesize(self, item: _Utterance) -> tuple[np.ndarray | None, int]:
        """Render one sentence. A failure is logged and chimed, never raised."""
        language = item.language or detect_language(item.text)
        engine = self._engine_for(language)
        try:
            audio_data, rate_value = engine.synthesize(item.text, language=language)
            audio = np.asarray(audio_data, dtype=np.float32)
            rate = int(rate_value)
        except Exception as exc:
            self.log.error("Speech synthesis failed for %r: %s", item.text[:60], exc)
            with self._lock:  # one error chime per burst, however many sentences fail
                chime, self._error_chime_done = not self._error_chime_done, True
            if chime:
                self.play_chime(error_chime(DEFAULT_SAMPLE_RATE), DEFAULT_SAMPLE_RATE)
            return None, 0
        if audio.size == 0 or rate <= 0:
            self.log.warning("The %s engine returned no audio for %r.", _engine_name(engine), item.text[:40])
            return None, 0
        return audio, rate

    # -- burst bookkeeping -------------------------------------------------------------

    def _begin_burst(self) -> None:
        """First audio of a burst: go to SPEAKING and stamp the latency headline."""
        with self._lock:
            if self._speaking:
                return
            self._speaking = True
            self._error_chime_done = self._summary_logged = False
            self._idle_polls = 0
        if self._latency is not None:
            try:
                self._latency.first_audio()
            except Exception as exc:
                self.log.debug("Could not record the first-audio latency: %s", exc)
        self._state.set(AssistantState.SPEAKING)

    def _poll_finish(self) -> None:
        """End the burst once the queue is empty and the player has fallen quiet."""
        with self._lock:
            if not (self._speaking and self._pending == 0):
                self._idle_polls = 0
                return
        if not self._queue.empty() or self._player_busy():
            with self._lock:
                self._idle_polls = 0
            return
        with self._lock:
            self._idle_polls += 1
            if self._idle_polls < IDLE_CONFIRMATIONS:
                return
            self._speaking = False
            self._idle_polls = 0
            callbacks = list(self._finished)
            self._cv.notify_all()
        if not callbacks:
            if self._state.state == AssistantState.SPEAKING:
                self._state.set(AssistantState.IDLE)
            return
        for callback in callbacks:
            try:
                callback()
            except Exception:
                self.log.exception("An on_finished callback failed.")

    def _player_busy(self) -> bool:
        """True while the player still has audio in hand."""
        try:
            return bool(self._player.is_playing)
        except Exception:
            return False

    def _drain(self) -> int:
        """Discard every queued sentence, returning how many were dropped."""
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return dropped
            if item is not None:
                dropped += 1

    def _on_player_first_audio(self) -> None:
        """Player callback: the real first audio. Logs the turn's summary once.

        Ignored when no burst is in progress, so an interface chime played from idle
        does not produce a second latency line for the same turn.
        """
        with self._lock:
            if self._latency is None or not self._speaking:
                return
        try:
            self._latency.first_audio()
            self._latency.mark(PLAYBACK_LABEL)
            summary = self._latency.summary()
        except Exception as exc:
            self.log.debug("Could not record the playback latency: %s", exc)
            return
        with self._lock:
            report, self._summary_logged = not self._summary_logged, True
        if report and summary:
            self.log.info("Latency: %s", summary)

    # -- engines -------------------------------------------------------------------------

    def _engine_for(self, language: str | None) -> Any:
        """Return the engine for ``language``, creating and caching it on first use."""
        if not language:
            return self._engine
        with self._lock:
            cached = self._engines.get(language)
        if cached is not None:
            return cached
        engine = self._engine
        if self._engine_language(engine) != language:
            selected = self._select_engine(language)
            if selected is not None:
                engine = selected
                self.log.info("Switched to the %s voice for %s.", _engine_name(engine), language)
        with self._lock:
            # Cached either way: a failed switch must not be retried for every sentence.
            self._engines[language] = engine
        return engine

    def _engine_language(self, engine: Any) -> str:
        """The language an engine speaks, from its own attribute or from its name."""
        value = _short_language(getattr(engine, "language", None))
        if value:
            return value
        return "sv" if "piper" in _engine_name(engine).lower() else (self._language or "en")

    def _select_engine(self, language: str) -> Any:
        """Build an engine for ``language`` through ``jarvis.tts.engine``.

        That module keeps its own engine cache, so the engine handed back is shared and
        is never closed here. A selector that cannot be imported or called is remembered
        as failed and the current voice simply keeps speaking.
        """
        with self._lock:
            if self._selector_failed:
                return None
        try:
            from jarvis.tts.engine import select_engine_for_language

            engine = select_engine_for_language(_shared_config(), language, self.log)
        except Exception as exc:
            with self._lock:
                self._selector_failed = True
            self.log.warning("Cannot switch to a %s voice, keeping the current one: %s", language, exc)
            return None
        if engine is None or engine is self._engine or not hasattr(engine, "synthesize"):
            return None
        return engine
