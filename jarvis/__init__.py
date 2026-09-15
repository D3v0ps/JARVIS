"""JARVIS — a fully local, voice-controlled desktop assistant for Windows 11.

The package is intentionally import-cheap: importing ``jarvis`` pulls in nothing
but this docstring and the version string. Every subsystem (audio, wake word,
STT, brain, TTS, tools, UI) lives in its own submodule and is imported only when
it is actually used, so that ``import jarvis`` works on any machine — including
a bare Linux CI box without audio hardware or Windows-only libraries.

Layout::

    jarvis/config.py     configuration (dot-path access over config.yaml)
    jarvis/core/         logging, state bus, memory, scheduler, latency, assistant
    jarvis/audio/        device enumeration, capture, playback, synthesized chimes
    jarvis/wake/         wake-word detection
    jarvis/stt/          voice activity detection and transcription
    jarvis/brain/        Ollama client, conversation, turn orchestration
    jarvis/tts/          speech synthesis engines and the speaking queue
    jarvis/tools/        the tool registry, safety layer and dispatcher
    jarvis/ui/           overlay ring and system tray icon
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
