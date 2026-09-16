"""Remote access: the phone as a push-to-talk satellite for the desk assistant.

Importing this package never imports Flask, never opens a socket and never touches
audio hardware — :class:`~jarvis.remote.server.RemoteServer` does all of that inside
:meth:`~jarvis.remote.server.RemoteServer.start`, and only when ``remote.enabled`` is
true in ``config.yaml``.

Typical wiring::

    from jarvis.remote import RemoteServer

    remote = RemoteServer(cfg, assistant, logger)
    if remote.start():
        logger.info("Phone can connect at %s", remote.url)
    ...
    remote.stop()
"""

from __future__ import annotations

from jarvis.remote.server import DEFAULTS, MISSING_HOOK, RemoteServer
from jarvis.remote.session import (
    COOKIE_NAME,
    REMOTE_REFUSAL,
    Device,
    PairResult,
    RemoteGuard,
    RemoteTurn,
    Routine,
    SessionStore,
    decode_pcm16,
    load_routines,
)

__all__ = [
    "COOKIE_NAME",
    "DEFAULTS",
    "Device",
    "MISSING_HOOK",
    "PairResult",
    "REMOTE_REFUSAL",
    "RemoteGuard",
    "RemoteServer",
    "RemoteTurn",
    "Routine",
    "SessionStore",
    "decode_pcm16",
    "load_routines",
]
