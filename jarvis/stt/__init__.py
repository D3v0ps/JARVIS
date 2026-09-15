"""Speech-to-text side of JARVIS.

``vad``          :class:`jarvis.stt.vad.SpeechSegmenter` (Silero) and
                 :class:`jarvis.stt.vad.EnergySegmenter` (RMS fallback) — they turn a
                 stream of microphone frames into finished utterances.
``transcriber``  faster-whisper, CUDA first with a quiet CPU int8 fallback.

``torch``, ``silero_vad`` and ``faster_whisper`` are imported inside the objects that
need them, so importing this package costs nothing on a machine with neither a GPU nor
a speech stack installed.
"""

from __future__ import annotations

__all__: list[str] = []
