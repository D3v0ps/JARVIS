"""The app on the desk: one window, drawn as a web page, hosted natively.

:class:`~jarvis.desk.bus.DeskBus` fans the assistant's narration out to whatever
windows are open, :class:`~jarvis.desk.server.DeskServer` serves the page and its
sockets on loopback, and :class:`~jarvis.desk.window.DeskWindow` finds something on
this machine willing to host it. All three are optional decoration: nothing here is
imported until it is asked for, so ``import jarvis.desk`` costs nothing on a box with
no Flask, no WebView2 and no display, and the assistant keeps working with the whole
section absent.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DeskBus", "DeskEvent", "DeskLogHandler", "DeskServer", "DeskWindow"]


def __getattr__(name: str) -> Any:
    """Import the desk's pieces on first use so ``import jarvis.desk`` stays cheap."""
    if name in ("DeskBus", "DeskEvent", "DeskLogHandler"):
        from . import bus

        return getattr(bus, name)
    if name == "DeskServer":
        from .server import DeskServer

        return DeskServer
    if name == "DeskWindow":
        from .window import DeskWindow

        return DeskWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
