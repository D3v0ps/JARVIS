"""Core building blocks of JARVIS.

This package holds the pieces that every other subsystem leans on: logging setup and
structured event helpers, the thread-safe state bus, long-term memory, the scheduler,
the latency tracker and the assistant loop itself.

Nothing is imported here on purpose. Importing a submodule must stay cheap and free of
side effects, so callers do ``from jarvis.core.state import StateBus`` explicitly.
"""
