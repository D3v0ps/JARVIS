"""Microphone capture.

:class:`MicStream` wraps a ``sounddevice.InputStream`` (float32, mono, 16 kHz by
default) and pushes fixed-size frames into a bounded queue. The PortAudio callback
does the absolute minimum: convert to mono float32 and ``put_nowait``. When the
consumer falls behind, the OLDEST frame is discarded and counted — the assistant must
always hear the most recent audio, not a growing backlog.

:class:`NullMicStream` offers the identical interface but yields silence, so text mode
and machines without a microphone need no special case downstream.

``sounddevice`` is imported lazily inside :meth:`MicStream.start`, so this module
imports fine on a box with no audio stack.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Iterator

import numpy as np

from jarvis.audio.devices import resolve_device

logger = logging.getLogger("jarvis.audio.capture")

# 50 frames of 1280 samples at 16 kHz is about 4 seconds of backlog. Past that the
# audio is stale anyway, so the oldest frames are dropped.
QUEUE_MAXSIZE = 50


class MicStream:
    """A live microphone as a stream of float32 mono frames.

    Args:
        device: Device index, a substring of the device name, or None for the system
            default input.
        sample_rate: Capture rate in Hz (the models expect 16000).
        block_size: Samples per frame (1280 = 80 ms at 16 kHz, openWakeWord's chunk).
        logger: Optional logger; a module logger is used when omitted.
    """

    def __init__(
        self,
        device: str | int | None = None,
        sample_rate: int = 16000,
        block_size: int = 1280,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = device
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.log = logger if logger is not None else logging.getLogger("jarvis.audio.capture")

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._lock = threading.RLock()
        self._stream: Any | None = None
        self._running = False
        self._dropped = 0
        self._frames_read = 0
        self._overruns = 0
        self._device_index: int | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Open the input stream. Idempotent.

        Raises:
            RuntimeError: when the device cannot be opened, carrying the PortAudio
                message so the assistant can report the loss of the microphone.
        """
        with self._lock:
            if self._running:
                return

            try:
                import sounddevice as sd  # type: ignore[import-not-found]
            except Exception as exc:
                message = (
                    "Cannot open the microphone: the 'sounddevice' package is not "
                    f"available ({exc}). Install it with: pip install sounddevice"
                )
                self.log.error(message)
                raise RuntimeError(message) from exc

            try:
                self._device_index = resolve_device(self.device, kind="input")
            except ValueError as exc:
                message = f"Cannot open the microphone: {exc}"
                self.log.error(message)
                raise RuntimeError(message) from exc

            try:
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=self._device_index,
                    channels=1,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
            except Exception as exc:
                message = (
                    f"Cannot open the microphone (device={self._device_index!r}, "
                    f"{self.sample_rate} Hz): {exc}"
                )
                self.log.error(message)
                raise RuntimeError(message) from exc

            self._stream = stream
            self._running = True
            self.log.info(
                "Microphone open: device=%s, %d Hz, %d-sample frames.",
                "default" if self._device_index is None else self._device_index,
                self.sample_rate,
                self.block_size,
            )

    def stop(self) -> None:
        """Close the input stream. Safe to call twice and from another thread."""
        with self._lock:
            was_running = self._running
            self._running = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self.log.warning("Error while stopping the microphone stream: %s", exc)
            try:
                stream.close()
            except Exception as exc:
                self.log.warning("Error while closing the microphone stream: %s", exc)

        # Wake up anyone blocked in read()/frames() so they end promptly.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if was_running:
            self.log.info(
                "Microphone closed: %d frames read, %d dropped, %d PortAudio overruns.",
                self._frames_read,
                self._dropped,
                self._overruns,
            )

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------- callback

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback. Keep this tiny: no logging, no heavy allocation."""
        if status:
            self._overruns += 1
        if indata.ndim == 2:
            mono = indata[:, 0].copy() if indata.shape[1] == 1 else indata.mean(axis=1)
        else:
            mono = indata.copy()
        if mono.dtype != np.float32:
            mono = mono.astype(np.float32)

        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # Consumer fell behind: throw away the oldest frame, keep the newest.
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                self._dropped += 1

    # ---------------------------------------------------------------------- reads

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the next frame, or None when nothing arrived within ``timeout``."""
        try:
            frame = self._queue.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None
        if frame is None:  # stop() sentinel
            return None
        self._frames_read += 1
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until :meth:`stop` is called, then drain and end cleanly."""
        while True:
            if not self._running and self._queue.empty():
                return
            frame = self.read(timeout=0.1)
            if frame is None:
                continue
            yield frame

    def flush(self) -> None:
        """Drop everything buffered (called after JARVIS speaks, so he does not hear himself)."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        if dropped:
            self.log.debug("Flushed %d buffered microphone frames.", dropped)

    # ----------------------------------------------------------------- properties

    @property
    def dropped(self) -> int:
        """Frames discarded because the consumer could not keep up."""
        return self._dropped

    @property
    def running(self) -> bool:
        """True while the input stream is open."""
        return self._running

    def stats(self) -> dict[str, Any]:
        """Small snapshot for the log: frames_read, dropped, running."""
        return {
            "frames_read": self._frames_read,
            "dropped": self._dropped,
            "running": self._running,
        }


class NullMicStream:
    """A microphone that is not there.

    Exposes exactly the same interface as :class:`MicStream` and yields silent frames
    at real-time pace, so text mode — and a machine with no input device — can run the
    normal loop without any special casing.
    """

    def __init__(
        self,
        device: str | int | None = None,
        sample_rate: int = 16000,
        block_size: int = 1280,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = device
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.log = logger if logger is not None else logging.getLogger("jarvis.audio.capture")

        self._running = False
        self._frames_read = 0
        self._stop_event = threading.Event()
        self._frame_seconds = self.block_size / float(self.sample_rate or 16000)

    def start(self) -> None:
        """Pretend to open a microphone. Idempotent, never raises."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self.log.info("Silent microphone in use — no audio input is available.")

    def stop(self) -> None:
        """Stop yielding silence. Safe to call twice and from another thread."""
        self._running = False
        self._stop_event.set()

    def __enter__(self) -> "NullMicStream":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def _silence(self) -> np.ndarray:
        return np.zeros(self.block_size, dtype=np.float32)

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return a silent frame after one frame's worth of time, or None when stopped."""
        if not self._running:
            self._stop_event.wait(min(max(0.0, float(timeout)), 0.05))
            return None
        waited = self._stop_event.wait(min(self._frame_seconds, max(0.0, float(timeout))))
        if waited or not self._running:
            return None
        self._frames_read += 1
        return self._silence()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield silent frames in real time until :meth:`stop` is called."""
        while self._running:
            frame = self.read(timeout=self._frame_seconds)
            if frame is None:
                continue
            yield frame

    def flush(self) -> None:
        """Nothing is buffered; present for interface parity."""
        return None

    @property
    def dropped(self) -> int:
        """Always zero — silence is never late."""
        return 0

    @property
    def running(self) -> bool:
        """True while the silent stream is active."""
        return self._running

    def stats(self) -> dict[str, Any]:
        """Small snapshot for the log: frames_read, dropped, running."""
        return {
            "frames_read": self._frames_read,
            "dropped": 0,
            "running": self._running,
        }


__all__ = ["MicStream", "NullMicStream", "QUEUE_MAXSIZE"]
