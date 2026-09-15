"""Audio subsystem for JARVIS.

Device enumeration and resolution (:mod:`jarvis.audio.devices`), microphone capture
(:mod:`jarvis.audio.capture`), playback and generated chimes live here. Every
hardware-bound library (``sounddevice`` / PortAudio) is imported lazily inside the
function that needs it, so importing this package succeeds on a machine without any
audio stack at all.
"""
