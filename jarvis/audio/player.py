"""Queued audio playback with instant cancellation.

:class:`Player` owns one worker thread that consumes ``(audio, sample_rate)`` items from
a queue and writes them to a lazily opened ``sounddevice.OutputStream``. Clips are written
in ~20 ms blocks and the loop checks a cancel generation between blocks, so
:meth:`Player.stop` cuts whatever is playing within roughly 50 ms — that is what makes
barge-in feel instant, and ``stop()`` itself never blocks on the worker.

``sounddevice`` is imported lazily. When it is missing (a Linux CI box, a machine with no
audio stack) the player logs once and becomes a silent sink: clips are accepted and
discarded, callbacks still fire, and the rest of the assistant keeps running.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

import numpy as np

from jarvis.audio.chimes import apply_volume
from jarvis.audio.devices import resolve_device

logger = logging.getLogger("jarvis.audio.player")

#: Playback is written in blocks of this many milliseconds, which bounds how long a
#: stop() takes to actually silence the speakers.
BLOCK_MS = 20.0

#: How long the worker waits on an empty queue before looping (and noticing close()).
_POLL_S = 0.1

_INSTALL_HINT = (
    "Audio playback needs the 'sounddevice' package (PortAudio bindings). "
    "Install it with:  pip install sounddevice"
)


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linearly resample mono ``audio`` from ``src`` Hz to ``dst`` Hz.

    Good enough for speech and chimes, and free of dependencies: ``n`` samples become
    ``round(n * dst / src)``.
    """
    data = np.asarray(audio, dtype=np.float32)
    if src <= 0 or dst <= 0:
        raise ValueError(f"Sample rates must be positive, got src={src!r}, dst={dst!r}")
    if src == dst or data.size == 0:
        return data
    n_out = int(round(data.size * float(dst) / float(src)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    if data.size == 1:
        return np.full(n_out, data[0], dtype=np.float32)
    positions = np.linspace(0.0, data.size - 1, n_out, dtype=np.float64)
    return np.interp(positions, np.arange(data.size, dtype=np.float64), data).astype(np.float32)


#: Integer sample formats and the divisor that maps them into [-1, 1].
_INT_SCALES: dict[Any, float] = {np.dtype(np.int16): 32768.0, np.dtype(np.int32): 2147483648.0}


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16/int32/float input, mono or ``(n, channels)``, to float32 mono."""
    data = np.asarray(audio)
    if data.ndim == 2:
        data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
    elif data.ndim != 1:
        raise ValueError(f"Audio must be 1-D or 2-D, got shape {data.shape}")

    scale = _INT_SCALES.get(data.dtype)
    if scale is not None:
        data = data.astype(np.float32) / scale
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    return np.ascontiguousarray(data, dtype=np.float32)


class Player:
    """Queued playback with instant :meth:`stop` for barge-in.

    Args:
        device: Output device index, a substring of its name, or None for the default.
        logger: Optional logger; a module logger is used when omitted.
        volume: Linear output gain applied to every clip (config ``audio.output_volume``).
    """

    def __init__(
        self, device: str | int | None = None, logger: logging.Logger | None = None,
        volume: float = 1.0,
    ) -> None:
        self.device = device
        self.log = logger if logger is not None else logging.getLogger("jarvis.audio.player")
        #: Linear output gain applied to every clip; may be changed at any time.
        self.volume = float(volume)

        self._queue: "queue.Queue[tuple[np.ndarray, int] | None]" = queue.Queue()
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None

        self._pending = 0          # queued clips + the one being written
        self._generation = 0       # bumped by stop()/close() to cut the current clip
        self._holding = False      # the worker has a clip in hand
        self._playing = False
        self._closing = False

        self._stream: Any | None = None
        self._stream_rate: int | None = None
        self._device_index: int | None = None
        self._silent = False       # True once sounddevice proved unavailable
        self._warned_silent = False

        self._first_audio_cb: Callable[[], None] | None = None
        self._first_audio_armed = False

    @property
    def is_playing(self) -> bool:
        """True while a clip is actually being written to the output stream."""
        with self._lock:
            return self._playing

    @property
    def available(self) -> bool:
        """False once playback has degraded to a silent sink."""
        with self._lock:
            return not self._silent

    def on_first_audio(self, callback: Callable[[], None]) -> None:
        """Register a callback fired once when audio starts after an idle period.

        This is the latency headline: the first sample of a reply reaching the speakers.
        The callback is re-armed every time the player goes idle again.
        """
        with self._lock:
            self._first_audio_cb = callback
            self._first_audio_armed = callback is not None

    def play(self, audio: np.ndarray, sample_rate: int, *, blocking: bool = False) -> None:
        """Queue a clip for playback.

        ``audio`` may be float32 or int16, mono or shaped ``(n, 1)`` / ``(n, 2)``, at any
        rate; ``blocking=True`` waits until the queue has drained before returning.
        """
        try:
            data = _to_mono_float32(audio)
        except Exception as exc:
            self.log.error("Refusing to play a clip that is not audio: %s", exc)
            return
        try:
            rate = int(sample_rate)
        except (TypeError, ValueError) as exc:
            self.log.error("Refusing to play a clip with sample rate %r: %s", sample_rate, exc)
            return
        if rate <= 0:
            self.log.error("Refusing to play a clip with sample rate %r.", sample_rate)
            return
        if data.size == 0:
            self.log.debug("Skipping an empty clip.")
            return

        data = apply_volume(data, self.volume)

        with self._lock:
            if self._closing:
                self.log.debug("Player is closed; dropping a %d-sample clip.", data.size)
                return
            self._pending += 1
        self._queue.put((data, rate))
        self._ensure_worker()

        if blocking:
            self.wait()

    def stop(self) -> None:
        """Drop everything queued and cut the current clip. Returns immediately."""
        with self._lock:
            self._generation += 1
            dropped = self._drain()
            self._pending = 1 if self._holding else 0
            self._idle.notify_all()
        if dropped:
            self.log.debug("Playback stopped: %d queued clip(s) discarded.", dropped)

    def wait(self, timeout: float | None = None) -> None:
        """Block until the queue is empty and nothing is playing (or ``timeout``)."""
        with self._lock:
            finished = self._idle.wait_for(lambda: self._pending == 0, timeout)
        if not finished:
            self.log.debug("wait() timed out with %d clip(s) still pending.", self._pending)

    def close(self) -> None:
        """Stop the worker and close the output stream. Safe to call twice."""
        with self._lock:
            worker = self._worker
            if not self._closing:
                self._closing = True
                self._generation += 1
                self._drain()
                self._pending = 0
                self._idle.notify_all()
        self._queue.put(None)  # sentinel: wakes the worker immediately

        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                self.log.warning("Playback worker did not stop within 2 seconds.")
        self._close_stream()
        with self._lock:
            self._worker = None

    def _drain(self) -> int:
        """Discard every queued clip, returning how many were dropped."""
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return dropped
            if item is not None:
                dropped += 1

    def _ensure_worker(self) -> None:
        """Start the playback worker on first use."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._closing:
                return
            self._worker = threading.Thread(target=self._run, name="jarvis-player", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        """Worker loop: take clips off the queue and write them out."""
        while True:
            try:
                item = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                with self._lock:
                    if self._closing:
                        break
                continue
            if item is None:
                break

            audio, rate = item
            with self._lock:
                generation, cancelled = self._generation, self._closing
                self._holding = True
            try:
                if not cancelled:
                    self._write_clip(audio, rate, generation)
            except Exception as exc:  # never let one bad clip kill playback
                self.log.error("Playback of a %d-sample clip failed: %s", audio.size, exc)
            self._finish_clip()

        self._close_stream()
        self.log.debug("Playback worker stopped.")

    def _finish_clip(self) -> None:
        """Mark one clip done, re-arm the first-audio callback when going idle."""
        with self._lock:
            self._playing = False
            self._holding = False
            self._pending = max(0, self._pending - 1)
            if self._pending == 0 and self._queue.empty():
                self._first_audio_armed = self._first_audio_cb is not None
            self._idle.notify_all()

    def _fire_first_audio(self) -> None:
        """Fire the first-audio callback once per play-after-idle."""
        with self._lock:
            callback = self._first_audio_cb if self._first_audio_armed else None
            self._first_audio_armed = False
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self.log.error("on_first_audio callback raised: %s", exc)

    def _write_clip(self, audio: np.ndarray, rate: int, generation: int) -> None:
        """Write one clip in small blocks, aborting as soon as stop() bumps the generation."""
        stream, stream_rate = self._ensure_stream(rate)
        if stream is None:
            # Silent sink: the clip is consumed instantly so the caller never stalls.
            self._fire_first_audio()
            self.log.debug("Discarding a %d-sample clip: no audio output available.", audio.size)
            return

        data = audio if rate == stream_rate else _resample(audio, rate, stream_rate)
        data = np.clip(data, -1.0, 1.0).astype(np.float32, copy=False)
        block = max(1, int(round(stream_rate * BLOCK_MS / 1000.0)))

        with self._lock:
            self._playing = True
        self._fire_first_audio()

        for start in range(0, data.size, block):
            with self._lock:
                if self._generation != generation or self._closing:
                    self.log.debug("Clip cut after %d of %d samples.", start, data.size)
                    return
            chunk = data[start : start + block]
            try:
                stream.write(chunk)
            except Exception as exc:
                self.log.error("Output stream write failed: %s", exc)
                self._close_stream()
                return

    def _ensure_stream(self, rate: int) -> tuple[Any | None, int]:
        """Return an open stream and its rate for ``rate``, or ``(None, rate)`` when silent.

        The stream is kept as long as the rate is unchanged and reopened when it differs;
        the effective rate comes back too, because it may not be the one requested.
        """
        with self._lock:
            if self._silent:
                return None, rate
            stream, current = self._stream, self._stream_rate
        if stream is not None and current == rate:
            return stream, int(current)

        sd = self._import_sounddevice()
        if sd is None:
            return None, rate
        if stream is not None:
            self._close_stream()

        try:
            device_index = resolve_device(self.device, kind="output")
        except ValueError as exc:
            self.log.error("Output device %r is unusable: %s", self.device, exc)
            device_index = None

        # Prefer the clip's own rate; if the device refuses it, fall back to the rate
        # that already worked and let the caller resample into it.
        candidates = [int(rate)]
        if current is not None and int(current) != int(rate):
            candidates.append(int(current))
        for candidate in candidates:
            try:
                new_stream = sd.OutputStream(
                    samplerate=candidate,
                    blocksize=max(1, int(round(candidate * BLOCK_MS / 1000.0))),
                    device=device_index,
                    channels=1,
                    dtype="float32",
                )
                new_stream.start()
            except Exception as exc:
                self.log.error(
                    "Cannot open the output stream (device=%r, %d Hz): %s",
                    device_index, candidate, exc,
                )
                continue
            with self._lock:
                self._stream, self._stream_rate = new_stream, candidate
                self._device_index = device_index
            self.log.info(
                "Output stream open: device=%s, %d Hz.",
                "default" if device_index is None else device_index, candidate,
            )
            return new_stream, candidate

        self._degrade(f"no output stream could be opened at {candidates} Hz")
        return None, rate

    def _import_sounddevice(self) -> Any | None:
        """Import ``sounddevice`` lazily; degrade to a silent sink when it is missing."""
        try:
            import sounddevice  # type: ignore[import-not-found]
        except Exception as exc:  # ImportError, or OSError when PortAudio is absent
            self._degrade(f"sounddevice is unavailable ({exc}). {_INSTALL_HINT}")
            return None
        return sounddevice

    def _degrade(self, reason: str) -> None:
        """Switch to the silent sink, logging the reason exactly once."""
        with self._lock:
            already, self._silent, self._warned_silent = self._warned_silent, True, True
        if not already:
            self.log.warning("Playback disabled, continuing without audio: %s", reason)

    def _close_stream(self) -> None:
        """Close the output stream if one is open. Never raises."""
        with self._lock:
            stream, self._stream, self._stream_rate = self._stream, None, None
        if stream is None:
            return
        for step in ("stop", "close"):
            try:
                getattr(stream, step)()
            except Exception as exc:
                self.log.warning("Error while %sing the output stream: %s", step[:-1], exc)

    def __enter__(self) -> "Player":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()