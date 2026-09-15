"""Speech synthesis for JARVIS.

``engine``      the :class:`~jarvis.tts.engine.TTSEngine` protocol, the silent
                :class:`~jarvis.tts.engine.NullEngine`, engine construction and the
                per-language selection rule.
``kokoro_tts``  Kokoro ONNX — the British voice used for English (24 kHz).
``piper_tts``   Piper — the Swedish voice, preferred whenever Swedish is spoken.
``sapi_tts``    Windows SAPI 5 through pyttsx3 — the fallback that needs no download.
``speaker``     the sentence queue that turns synthesis into audible speech.

Every engine-specific library (``kokoro_onnx``, ``piper``, ``pyttsx3``) is imported inside
the method that needs it, so importing this package works on a machine that has none of
them — an engine that cannot load simply reports ``available() is False`` and the next one
in the chain is tried.
"""

from jarvis.tts.engine import (
    DEFAULT_SAMPLE_RATE,
    NullEngine,
    TTSEngine,
    create_engine,
    select_engine_for_language,
)

__all__ = [
    "TTSEngine",
    "NullEngine",
    "create_engine",
    "select_engine_for_language",
    "DEFAULT_SAMPLE_RATE",
]
