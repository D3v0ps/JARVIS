"""The face of JARVIS: the arc-reactor overlay ring and the system tray icon.

Both members are optional decoration. :class:`~jarvis.ui.overlay.Overlay` draws a
frameless, always-on-top tkinter ring that reports the assistant's state, and
:class:`~jarvis.ui.tray.Tray` puts a small glowing icon next to the clock. Neither
imports ``tkinter``, ``pystray`` or ``PIL`` at module import time — those arrive
lazily inside :meth:`start`, so importing this package works on a headless box and
the assistant keeps running without a face when the display or the libraries are
missing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Overlay", "Tray"]


def __getattr__(name: str) -> Any:
    """Import the UI classes on first use so ``import jarvis.ui`` stays cheap."""
    if name == "Overlay":
        from .overlay import Overlay

        return Overlay
    if name == "Tray":
        from .tray import Tray

        return Tray
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
