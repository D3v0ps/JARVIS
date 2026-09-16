"""Clipboard tool: reading what the user copied — or merely selected on screen.

Three things matter more than the feature. *Privacy*: a password manager marks its
payload with ``ExcludeClipboardContentFromMonitorProcessing`` or
``CanIncludeInClipboardHistory``, and both are checked **before** any data is fetched,
so a marked clipboard is refused without its contents ever reaching a variable, a
``ToolResult`` or ``logs/jarvis.log``. *Contention*: ``OpenClipboard`` fails while
another process holds the clipboard, so every access retries briefly. *Size*: text is
truncated hard before it leaves this module, and ``detail`` carries counts only.

The tool is deliberately dumb: ``summarise``, ``translate`` and ``explain`` do not call
a model. They return the text in ``data`` with a one-sentence spoken summary of what was
on the clipboard, and the model does the thinking with what it already has.

``win32clipboard`` and ``ctypes`` are imported inside the functions that need them, so
this module imports cleanly on Linux, where the tool refuses politely.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from jarvis.brain.sentences import clean_for_speech
from jarvis.core.logging import get_logger, log_refusal
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.input_tools import _input_structures
from jarvis.tools.registry import tool

__all__ = [
    "clipboard", "ClipboardContent", "ClipboardBusy", "CLIPBOARD_ACTIONS", "SOURCES",
    "PRIVATE_FORMATS", "PRIVACY_SUMMARY", "MAX_TEXT_CHARS", "MAX_FILES",
    "OPEN_ATTEMPTS", "OPEN_RETRY_SECONDS", "SELECTION_WAIT_SECONDS",
]

_log = get_logger("tools.clipboard")

#: What the model may ask for. The last three return the text; they never call a model.
CLIPBOARD_ACTIONS = ("read", "summarise", "translate", "explain")
_ACTION_ALIASES = {"summarize": "summarise", "summary": "summarise", "paste": "read",
                   "get": "read", "sammanfatta": "summarise", "översätt": "translate",
                   "oversatt": "translate", "förklara": "explain", "forklara": "explain"}
#: Where to look; ``auto`` falls back to the selection when nothing has been copied.
SOURCES = ("auto", "clipboard", "selection")

MAX_TEXT_CHARS = 4000         #: hard truncation before anything reaches the model
MAX_FILES = 10                #: copied file names reported back
SPEAK_TEXT_CHARS = 180        #: below this, ``read`` simply says the text out loud
OPEN_ATTEMPTS = 6             #: tries before giving up on a clipboard someone else holds
OPEN_RETRY_SECONDS = 0.05     #: base delay between those tries, backed off linearly
SELECTION_WAIT_SECONDS = 0.6  #: how long Ctrl+C is given to change the clipboard
SELECTION_POLL_SECONDS = 0.02

#: Formats a password manager sets to keep its payload out of clipboard history.
PRIVATE_FORMATS = ("ExcludeClipboardContentFromMonitorProcessing",
                   "CanIncludeInClipboardHistory")
PRIVACY_SUMMARY = "That clipboard entry is marked private, sir, so I haven't looked at it."

#: Standard format numbers, so a fake win32clipboard does not have to define them.
CF_TEXT, CF_DIB, CF_UNICODETEXT, CF_HDROP = 1, 8, 13, 15
_VK_CONTROL, _VK_C = 0x11, 0x43
_INPUT_KEYBOARD, _KEYEVENTF_KEYUP = 1, 0x0002

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"]


class ClipboardBusy(RuntimeError):
    """Another process held the clipboard for every attempt we made."""


@dataclass
class ClipboardContent:
    """What the clipboard held: text, a file list, an image, or nothing at all."""

    kind: str = "empty"        # "text" | "files" | "image" | "empty"
    text: str = ""
    files: tuple[str, ...] = ()
    file_count: int = 0
    width: int | None = None
    height: int | None = None
    byte_size: int = 0
    characters: int = 0        # length before truncation
    truncated: bool = False
    source: str = "clipboard"  # "clipboard" | "selection"
    fmt: str = ""


def _spoken_int(value: int) -> str:
    """Render a whole number as words, so the summary reads well aloud."""
    number = int(value)
    if number < 0:
        return str(number)
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"
    for limit, group, word in ((1000, 100, "hundred"), (1_000_000, 1000, "thousand")):
        if number < limit:
            head, rest = divmod(number, group)
            spoken = f"{_spoken_int(head)} {word}"
            if rest == 0:
                return spoken
            return spoken + (" and " if rest < 100 else " ") + _spoken_int(rest)
    return str(number)


def _spoken_size(words: int) -> str:
    """A spoken word count: exact when small, rounded and hedged when not."""
    if words <= 20:
        return f"{_spoken_int(words)} word" + ("" if words == 1 else "s")
    step = 10 if words < 200 else 100
    return f"about {_spoken_int(max(step, int(round(words / step)) * step))} words"


def _snippet(text: str, limit: int = SPEAK_TEXT_CHARS) -> str:
    """A speakable one-line opening of the text, markdown and emoji removed."""
    spoken = " ".join(clean_for_speech(text).split())
    if len(spoken) > limit:
        cut = spoken.rfind(" ", 0, limit)
        spoken = spoken[: cut if cut > limit // 2 else limit].rstrip(" ,;:-") + "…"
    return spoken


# --- clipboard access -----------------------------------------------------------------
def _import_win32clipboard() -> Any | None:
    """Import ``win32clipboard`` lazily; ``None`` when pywin32 is not available."""
    try:
        import win32clipboard  # noqa: PLC0415 - Windows-only, imported on demand
    except Exception as exc:  # noqa: BLE001 - ImportError here, DLL errors on Windows
        _log.warning("win32clipboard is unavailable: %s", exc)
        return None
    return win32clipboard


@contextmanager
def _clipboard_open(w32: Any) -> Iterator[None]:
    """Open the clipboard with a short retry loop, and always close it again.

    Explorer, a browser or the clipboard history service holds the clipboard for a few
    milliseconds at a time, and ``OpenClipboard`` simply fails while it does. Retrying
    is the difference between a working tool and one that fails once a day for no
    reason the user can see.
    """
    opened, last_error = False, None
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        try:
            w32.OpenClipboard()
            opened = True
            break
        except Exception as exc:  # noqa: BLE001 - pywintypes.error, by way of pywin32
            last_error = exc
            if attempt < OPEN_ATTEMPTS:
                _log.debug("Clipboard busy (%d/%d): %s", attempt, OPEN_ATTEMPTS, exc)
                time.sleep(OPEN_RETRY_SECONDS * attempt)
    if not opened:
        raise ClipboardBusy(f"OpenClipboard failed {OPEN_ATTEMPTS} times: {last_error}")
    try:
        yield
    finally:
        try:
            w32.CloseClipboard()
        except Exception:  # noqa: BLE001 - a failed close must not mask the real error
            _log.debug("CloseClipboard failed", exc_info=True)


def _available(w32: Any, fmt: int) -> bool:
    """``IsClipboardFormatAvailable``, treating any error as "not available"."""
    try:
        return bool(w32.IsClipboardFormatAvailable(fmt))
    except Exception:  # noqa: BLE001
        return False


def _privacy_block(w32: Any) -> str | None:
    """Name the format that forbids reading this clipboard, or return ``None``.

    ``ExcludeClipboardContentFromMonitorProcessing`` means "do not look" by its mere
    presence. ``CanIncludeInClipboardHistory`` carries a DWORD where zero is an
    opt-out; an unreadable value fails closed, because a password is exactly the
    payload that would make reading it fail.
    """
    for name in PRIVATE_FORMATS:
        try:
            fmt = int(w32.RegisterClipboardFormat(name) or 0)
        except Exception:  # noqa: BLE001 - an unregistrable name cannot be on the board
            _log.debug("Could not register clipboard format %s", name, exc_info=True)
            continue
        if not fmt or not _available(w32, fmt):
            continue
        if name == "CanIncludeInClipboardHistory":
            try:
                raw = w32.GetClipboardData(fmt)
            except Exception:  # noqa: BLE001
                raw = None
            if isinstance(raw, (bytes, bytearray, memoryview)):
                raw = int.from_bytes(bytes(raw)[:4], "little") if bytes(raw) else 0
            if isinstance(raw, int) and raw:
                continue  # explicitly allowed by the owning application
        return name
    return None


def _dib_dimensions(raw: Any) -> tuple[int | None, int | None]:
    """Width and height from a ``CF_DIB`` BITMAPINFOHEADER, without decoding pixels."""
    try:
        data = bytes(raw or b"")
    except Exception:  # noqa: BLE001
        return None, None
    if len(data) < 12:
        return None, None
    width = int.from_bytes(data[4:8], "little", signed=True)
    height = int.from_bytes(data[8:12], "little", signed=True)
    return (abs(width), abs(height)) if width and height else (None, None)


def _normalise_text(raw: Any) -> str:
    """Decode whatever ``GetClipboardData`` returned and normalise line endings."""
    text = (bytes(raw).decode("utf-8", errors="replace")
            if isinstance(raw, (bytes, bytearray, memoryview)) else str(raw or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _read_content(w32: Any) -> ClipboardContent:
    """Read an already-open, already-cleared clipboard: files, then text, then image."""
    if _available(w32, CF_HDROP):
        raw = w32.GetClipboardData(CF_HDROP)
        entries = [raw] if isinstance(raw, (str, bytes)) else list(raw or ())
        paths = [str(entry) for entry in entries if str(entry).strip()]
        if paths:
            return ClipboardContent(kind="files", files=tuple(paths[:MAX_FILES]),
                                    file_count=len(paths), fmt="CF_HDROP")
    for fmt, label in ((CF_UNICODETEXT, "CF_UNICODETEXT"), (CF_TEXT, "CF_TEXT")):
        if not _available(w32, fmt):
            continue
        text = _normalise_text(w32.GetClipboardData(fmt))
        if not text.strip():
            continue
        return ClipboardContent(kind="text", text=text[:MAX_TEXT_CHARS],
                                characters=len(text),
                                truncated=len(text) > MAX_TEXT_CHARS, fmt=label)
    if _available(w32, CF_DIB):
        raw = w32.GetClipboardData(CF_DIB)
        width, height = _dib_dimensions(raw)
        size = len(bytes(raw)) if isinstance(raw, (bytes, bytearray, memoryview)) else 0
        return ClipboardContent(kind="image", width=width, height=height,
                                byte_size=size, fmt="CF_DIB")
    return ClipboardContent(kind="empty")


def _inspect(w32: Any) -> tuple[ClipboardContent, str | None]:
    """One open/close cycle: the privacy check first, content only if it passes."""
    with _clipboard_open(w32):
        blocked = _privacy_block(w32)
        if blocked:
            return ClipboardContent(kind="empty"), blocked
        return _read_content(w32), None


# --- the selection path ---------------------------------------------------------------
def _send_ctrl_c() -> tuple[int, Any]:
    """Synthesise Ctrl+C with ``type_text``'s ``INPUT``; returns (accepted, user32)."""
    import ctypes  # noqa: PLC0415 - for sizeof only; ctypes itself imports on Linux

    INPUT, user32 = _input_structures()
    events = []
    for key, flags in ((_VK_CONTROL, 0), (_VK_C, 0),
                       (_VK_C, _KEYEVENTF_KEYUP), (_VK_CONTROL, _KEYEVENTF_KEYUP)):
        event = INPUT(type=_INPUT_KEYBOARD)
        event.ki.wVk, event.ki.wScan, event.ki.dwFlags = key, 0, flags
        events.append(event)
    array = (INPUT * len(events))(*events)
    sent = int(user32.SendInput(len(events), array, ctypes.sizeof(INPUT)))
    if sent != len(events):
        _log.warning("SendInput accepted %d of %d Ctrl+C events", sent, len(events))
    return sent, user32


def _sequence_number(user32: Any) -> int | None:
    """``GetClipboardSequenceNumber``: the only reliable "did anything change" signal."""
    try:
        return int(user32.GetClipboardSequenceNumber())
    except Exception:  # noqa: BLE001
        _log.debug("GetClipboardSequenceNumber failed", exc_info=True)
        return None


def _restore_text(w32: Any, previous: str | None) -> None:
    """Put the user's own clipboard back after we borrowed it for a selection."""
    try:
        with _clipboard_open(w32):
            w32.EmptyClipboard()
            if previous:
                try:
                    w32.SetClipboardText(previous, CF_UNICODETEXT)
                except AttributeError:  # older pywin32 exposes only SetClipboardData
                    w32.SetClipboardData(CF_UNICODETEXT, previous)
    except Exception as exc:  # noqa: BLE001 - never lose the answer over a failed restore
        _log.warning("Could not restore the previous clipboard contents: %s", exc)


def _capture_selection(w32: Any,
                       previous: ClipboardContent) -> tuple[ClipboardContent | None, str]:
    """Copy what is selected on screen, read it, then put the clipboard back.

    ``content`` comes back ``None`` when the keystroke could not be sent, when the
    sequence number never moved (nothing was selected), or when the copied selection
    turned out to be marked private; the second element says which.
    """
    try:
        sent, user32 = _send_ctrl_c()
    except Exception as exc:  # noqa: BLE001 - no windll off Windows, no user32 in a service
        _log.warning("Could not synthesise Ctrl+C: %s", exc)
        return None, f"SendInput unavailable: {type(exc).__name__}: {exc}"
    if sent <= 0:
        return None, "SendInput accepted no events; the focused window refused input."
    before = _sequence_number(user32)
    deadline, changed = time.monotonic() + SELECTION_WAIT_SECONDS, False
    while time.monotonic() < deadline:
        time.sleep(SELECTION_POLL_SECONDS)
        after = _sequence_number(user32)
        if before is None or after is None or after != before:
            changed = True  # a moved counter, or no counter to trust: read and see
            break
    if not changed:
        return None, "The clipboard sequence number never moved; nothing was selected."
    try:
        content, blocked = _inspect(w32)
    finally:
        _restore_text(w32, previous.text if previous.kind == "text" else None)
    if blocked:
        return None, f"private:{blocked}"
    if content.kind == "empty":
        return None, "Ctrl+C produced nothing readable."
    content.source = "selection"
    return content, ""


# --- the spoken answer ----------------------------------------------------------------
def _describe(action: str, content: ClipboardContent) -> str:
    """One spoken sentence saying what was on the clipboard, never more."""
    place = "selection" if content.source == "selection" else "clipboard"
    if content.kind == "files":
        first = content.files[0] if content.files else ""
        name = first.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or first
        if content.file_count == 1:
            return f"The {place} holds one file, sir: {name}."
        return (f"The {place} holds {_spoken_int(content.file_count)} files, sir, "
                f"starting with {name}.")
    if content.kind == "image":
        if content.width and content.height:
            return (f"The {place} holds an image, sir, {_spoken_int(content.width)} by "
                    f"{_spoken_int(content.height)} pixels.")
        return f"The {place} holds an image, sir, not text I can read."
    size = _spoken_size(len(content.text.split()))
    if action == "read":
        if content.characters <= SPEAK_TEXT_CHARS and not content.truncated:
            return f"The {place} says: {_snippet(content.text)}"
        return f"The {place} holds {size}, sir, beginning: {_snippet(content.text, 120)}"
    verb = {"summarise": "to summarise", "translate": "to translate",
            "explain": "to explain"}[action]
    trimmed = ", trimmed to the first four thousand characters" if content.truncated else ""
    return f"I have {size} from the {place}{trimmed}, sir, {verb}."


def _payload(action: str, content: ClipboardContent) -> tuple[dict, str]:
    """``data`` for the model and ``detail`` for the log — counts only, no content."""
    data: dict[str, Any] = {"action": action, "kind": content.kind, "source": content.source}
    if content.kind == "text":
        data.update(text=content.text, characters=content.characters,
                    truncated=content.truncated)
        trimmed = f", truncated to {MAX_TEXT_CHARS}" if content.truncated else ""
        detail = (f"Read {content.characters} character(s) of {content.fmt or 'text'} "
                  f"from the {content.source}{trimmed}; the text itself is in data, "
                  "never in this log line.")
    elif content.kind == "files":
        data.update(files=list(content.files), file_count=content.file_count)
        detail = f"{content.file_count} copied path(s): " + ", ".join(content.files)
    elif content.kind == "image":
        data.update(width=content.width, height=content.height, bytes=content.byte_size)
        detail = (f"CF_DIB image, {content.width}x{content.height} pixels, "
                  f"{content.byte_size} byte(s); not decoded.")
    else:
        detail = "The clipboard held no text, files or image."
    return data, detail


@tool(
    "clipboard",
    description=(
        "Read what the user has copied to the clipboard, or the text they have selected "
        "on screen. Use it whenever the user says 'this', 'that' or 'what I just copied' "
        "and asks you to read, summarise, translate or explain it. The tool returns the "
        "text to you; you do the summarising, translating or explaining yourself."
    ),
    parameters={
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": list(CLIPBOARD_ACTIONS), "description":
                       "What the user wants done with the clipboard content."},
            "source": {"type": "string", "enum": list(SOURCES), "description":
                       "Where to look: 'clipboard' for what was copied, 'selection' to "
                       "copy what is highlighted on screen first, or 'auto' (default) to "
                       "try the clipboard and fall back to the selection."},
        },
    },
    tier=Tier.SAFE,
)
def clipboard(ctx: ToolContext, args: dict) -> ToolResult:
    """Read the clipboard (or the on-screen selection) and hand the text back."""
    raw_action = str(args.get("action") or "read").strip().lower().replace(" ", "_")
    action = _ACTION_ALIASES.get(raw_action, raw_action)
    if action not in CLIPBOARD_ACTIONS:
        return ToolResult.fail(
            "I can only read, summarise, translate or explain the clipboard, sir.",
            detail=f"Unsupported clipboard action {raw_action!r}.")
    source = str(args.get("source") or "auto").strip().lower()
    if source not in SOURCES:
        source = "auto"
    w32 = _import_win32clipboard()
    if w32 is None:
        if sys.platform != "win32":
            return ToolResult.fail("I can only reach the clipboard on Windows, sir.",
                                   detail=f"Platform is {sys.platform!r}; win32 required.")
        return ToolResult.fail("I can't reach the clipboard, sir - pywin32 isn't installed.",
                               detail="Importing win32clipboard failed; install pywin32.")
    try:
        content, blocked = _inspect(w32)
    except ClipboardBusy as exc:
        _log.warning("Clipboard stayed busy: %s", exc)
        return ToolResult.fail(
            "Something else is holding the clipboard, sir - try that again in a moment.",
            detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - a broken clipboard must not kill the turn
        _log.error("Reading the clipboard failed: %s", exc, exc_info=True)
        return ToolResult.fail("I couldn't read the clipboard, sir.",
                               detail=f"{type(exc).__name__}: {exc}")
    if blocked:
        # Nothing was fetched, so there is no content to leak into the log or the model.
        log_refusal("clipboard content is marked private",
                    f"Clipboard carries the {blocked} format; contents were not read.")
        return ToolResult.refuse(PRIVACY_SUMMARY,
                                 detail=f"Refused: clipboard marked with {blocked}.")

    wants_selection = source == "selection" or (source == "auto" and content.kind == "empty")
    if wants_selection and content.kind in ("files", "image"):
        return ToolResult.fail(
            "I'd have to overwrite what you've copied to do that, sir, so I haven't.",
            detail=f"Refusing the selection path: the clipboard holds {content.kind}.")
    if wants_selection:
        captured, reason = _capture_selection(w32, content)
        if captured is not None:
            content = captured
        elif reason.startswith("private:"):
            log_refusal("selected text is marked private", f"Selection carried {reason[8:]}.")
            return ToolResult.refuse(PRIVACY_SUMMARY, detail=f"Refused: {reason[8:]}.")
        elif content.kind == "empty":
            return ToolResult(True, "There's nothing copied and nothing selected, sir.",
                              detail=reason,
                              data={"action": action, "kind": "empty", "source": "selection"})
    if content.kind == "empty":
        return ToolResult(True, "There's nothing on the clipboard, sir.",
                          detail="The clipboard held no text, files or image.",
                          data={"action": action, "kind": "empty", "source": "clipboard"})
    data, detail = _payload(action, content)
    return ToolResult(True, _describe(action, content), detail, data)
