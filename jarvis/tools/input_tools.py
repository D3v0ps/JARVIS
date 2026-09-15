"""Keyboard tool: typing text into whatever window currently has focus.

Characters are sent as Unicode key events through ``SendInput`` with
``KEYEVENTF_UNICODE``, which means the active keyboard layout is irrelevant: å, ä, ö,
em dashes and emoji-range characters all arrive intact, and nothing has to be mapped to
a scan code. Newlines and tabs are sent as real Return and Tab presses instead, because
applications react to those keys rather than to the characters.

``ctypes`` is only touched inside the functions, so this module imports on Linux, where
the tool politely refuses rather than typing into nothing.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["type_text", "MAX_TEXT_CHARS", "HEADS_UP_SECONDS"]

_log = get_logger("tools.input")

#: Longest passage JARVIS will type in one go; anything more is trimmed.
MAX_TEXT_CHARS = 2000

#: The promised pause between the spoken heads-up and the first keystroke.
HEADS_UP_SECONDS = 1.0

#: Key events sent per ``SendInput`` call, with a breath in between so slow editors keep up.
_CHUNK_EVENTS = 100

#: Pause between chunks, in seconds.
_CHUNK_PAUSE = 0.01

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_VK_RETURN = 0x0D
_VK_TAB = 0x09

#: Characters that are pressed as keys rather than injected as text.
_VIRTUAL_KEYS = {"\n": _VK_RETURN, "\t": _VK_TAB}

#: Cache for the ctypes structures, which are built once on first use.
_STRUCTURES: dict[str, Any] = {}


def _input_structures() -> tuple[Any, Any]:
    """Build (once) and return the ``INPUT`` structure and the ``user32`` handle.

    The full ``INPUT`` union — mouse, keyboard and hardware — has to be declared even
    though only the keyboard arm is used, because ``SendInput`` validates ``cbSize``
    against the size of the complete union.
    """
    cached = _STRUCTURES.get("input")
    if cached is not None:
        return cached, _STRUCTURES["user32"]

    import ctypes  # noqa: PLC0415 - Windows-only code path

    # Explicit widths rather than ctypes.wintypes: WORD/DWORD/LONG are 16/32/32 bits in
    # the Windows headers, while ctypes maps c_long to 64 bits on 64-bit Linux, which
    # would silently give the structures the wrong layout off Windows.
    word, dword, long32 = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int32
    ulong_ptr = ctypes.c_void_p  # ULONG_PTR: pointer-sized on both 32- and 64-bit

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", word), ("wScan", word), ("dwFlags", dword),
            ("time", dword), ("dwExtraInfo", ulong_ptr),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", long32), ("dy", long32), ("mouseData", dword),
            ("dwFlags", dword), ("time", dword), ("dwExtraInfo", ulong_ptr),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", dword), ("wParamL", word), ("wParamH", word)]

    class _InputUnion(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("value",)
        _fields_ = [("type", dword), ("value", _InputUnion)]

    user32 = ctypes.windll.user32
    user32.SendInput.argtypes = (ctypes.c_uint32, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = ctypes.c_uint32

    _STRUCTURES["input"] = INPUT
    _STRUCTURES["user32"] = user32
    _STRUCTURES["ctypes"] = ctypes
    return INPUT, user32


def _key_events(text: str, INPUT: Any) -> list[Any]:
    """Turn the text into a flat list of key-down / key-up ``INPUT`` events.

    The text is encoded to UTF-16 so characters outside the basic plane are sent as
    their two surrogate code units, back to back, which is what Windows expects.
    """
    events: list[Any] = []
    for character in text:
        virtual_key = _VIRTUAL_KEYS.get(character)
        if virtual_key is not None:
            for flags in (0, _KEYEVENTF_KEYUP):
                event = INPUT(type=_INPUT_KEYBOARD)
                event.ki.wVk = virtual_key
                event.ki.wScan = 0
                event.ki.dwFlags = flags
                events.append(event)
            continue
        encoded = character.encode("utf-16-le")
        for index in range(0, len(encoded), 2):
            code_unit = encoded[index] | (encoded[index + 1] << 8)
            for flags in (_KEYEVENTF_UNICODE, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP):
                event = INPUT(type=_INPUT_KEYBOARD)
                event.ki.wVk = 0
                event.ki.wScan = code_unit
                event.ki.dwFlags = flags
                events.append(event)
    return events


def _send_text(text: str) -> tuple[int, int]:
    """Type ``text`` into the focused window.

    Returns ``(events_accepted, events_attempted)`` so the caller can tell a window
    that swallowed the input apart from one that took all of it.
    """
    INPUT, user32 = _input_structures()
    ctypes = _STRUCTURES["ctypes"]
    events = _key_events(text, INPUT)
    if not events:
        return 0, 0
    size = ctypes.sizeof(INPUT)
    sent = 0
    for start in range(0, len(events), _CHUNK_EVENTS):
        chunk = events[start : start + _CHUNK_EVENTS]
        array = (INPUT * len(chunk))(*chunk)
        accepted = int(user32.SendInput(len(chunk), array, size))
        sent += accepted
        if accepted != len(chunk):
            _log.warning(
                "SendInput accepted %d of %d events; the target window may have refused input",
                accepted,
                len(chunk),
            )
            break
        if start + _CHUNK_EVENTS < len(events):
            time.sleep(_CHUNK_PAUSE)
    return sent, len(events)


def _clean(raw: Any) -> str:
    """Normalise line endings and drop control characters that cannot be typed."""
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch in ("\n", "\t") or ch.isprintable())


@tool(
    "type_text",
    description=(
        "Type a passage of text on the keyboard, into whatever window the user currently "
        "has in focus. Use it to dictate into an editor, a chat box or a form field."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Exactly the text to type; newlines are typed as Return presses.",
            }
        },
        "required": ["text"],
    },
    tier=Tier.ANNOUNCED,
    announce="I'll type that into the focused window in a moment, sir.",
)
def type_text(ctx: ToolContext, args: dict) -> ToolResult:
    """Type text into the focused window after a one-second heads-up.

    The pause is the whole point of the announcement: the user hears what is about to
    happen and has a moment to click into the right window. Input longer than
    :data:`MAX_TEXT_CHARS` is truncated rather than refused, so a runaway model cannot
    hold the keyboard hostage.
    """
    text = _clean(args.get("text"))
    if not text.strip():
        return ToolResult.fail("There was nothing for me to type, sir.")

    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
        _log.info("type_text input truncated to %d characters", MAX_TEXT_CHARS)

    if sys.platform != "win32":
        return ToolResult.fail(
            "I can only type into a window on Windows, sir.",
            detail=f"Platform is {sys.platform!r}; SendInput requires win32.",
        )

    time.sleep(HEADS_UP_SECONDS)
    try:
        sent, expected = _send_text(text)
    except Exception as exc:  # noqa: BLE001 - a failed injection must not kill the turn
        _log.error("Typing failed: %s", exc)
        return ToolResult.fail(
            "I couldn't reach the keyboard, sir.", detail=f"{type(exc).__name__}: {exc}"
        )

    data = {"characters": len(text), "events_sent": sent, "truncated": truncated}
    detail = f"Typed {len(text)} character(s) as {sent}/{expected} key events."
    if sent == 0:
        return ToolResult.fail(
            "Windows wouldn't let me type into that window, sir.", detail=detail
        )
    if truncated:
        return ToolResult(
            ok=True,
            summary="I've typed as much as I'm allowed to in one go, sir.",
            detail=detail,
            data=data,
        )
    return ToolResult(
        ok=True,
        summary="There you are, sir, typed into the focused window.",
        detail=detail,
        data=data,
    )
