"""Application tools: opening programs, closing them politely, and opening web pages.

The application map lives in ``config.yaml`` under ``tools.apps`` — friendly name to
executable, URL or shell URI. Names arrive from speech, so the lookup is deliberately
forgiving: exact key, then a squashed match ("vs code" -> "vscode"), then a substring
match, then :func:`difflib.get_close_matches`.

Nothing Windows-only is imported at module level; ``os.startfile`` and ``pygetwindow``
are reached for inside the functions and their absence becomes a calm sentence.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = [
    "open_app",
    "close_app",
    "open_url",
    "resolve_app",
    "friendly_name",
    "looks_like_uri",
    "FUZZY_CUTOFF",
]

_log = get_logger("tools.apps")

#: Keeps a console window from flashing up when a helper process is started on Windows.
CREATE_NO_WINDOW = 0x08000000

#: ``difflib`` similarity a spoken name needs before it counts as the same app.
FUZZY_CUTOFF = 0.6

#: Seconds allowed for ``taskkill`` to answer.
TASKKILL_TIMEOUT = 15

#: Seconds allowed for ``cmd /c start`` — it returns as soon as the app is launched.
START_TIMEOUT = 15

#: Shell protocols that ``os.startfile`` handles directly, e.g. ``ms-settings:display``.
_URI_PREFIXES = ("ms-settings:", "ms-windows-store:", "shell:", "mailto:", "steam:", "spotify:")

_SQUASH_RE = re.compile(r"[^a-z0-9]+")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def _normalise(name: Any) -> str:
    """Lowercase, single-spaced form of a spoken application name."""
    return " ".join(str(name or "").strip().lower().split())


def _squash(name: str) -> str:
    """Letters and digits only, so "vs code" and "vscode" compare equal."""
    return _SQUASH_RE.sub("", _normalise(name))


def friendly_name(name: str) -> str:
    """The name to say out loud: "task manager" -> "Task Manager"."""
    text = " ".join(str(name or "").split())
    if not text:
        return "that application"
    return " ".join(word if word.isupper() else word.capitalize() for word in text.split())


def resolve_app(name: str, apps: dict[str, Any]) -> tuple[str, str] | None:
    """Match a spoken name against the configured applications.

    Returns ``(friendly_key, target)`` for the best match, or ``None`` when nothing
    is close enough. The four passes run from strict to forgiving: exact key, squashed
    key, substring either way (longest key wins), then difflib on both spellings.
    """
    wanted = _normalise(name)
    if not wanted or not isinstance(apps, dict) or not apps:
        return None
    table = {_normalise(key): (str(key), str(value)) for key, value in apps.items() if key}

    if wanted in table:
        return table[wanted]

    squashed = _squash(wanted)
    by_squash = {_squash(key): value for key, value in table.items()}
    if squashed and squashed in by_squash:
        return by_squash[squashed]

    candidates = [
        key for key in table
        if (key and (key in wanted or wanted in key))
        or (_squash(key) and (_squash(key) in squashed or squashed in _squash(key)))
    ]
    if candidates:
        best = max(candidates, key=len)
        return table[best]

    close = difflib.get_close_matches(wanted, list(table), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return table[close[0]]
    close = difflib.get_close_matches(squashed, list(by_squash), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return by_squash[close[0]]
    return None


def looks_like_uri(target: str) -> bool:
    """True when the mapped value should be handed to the shell rather than launched."""
    text = str(target or "").strip().lower()
    if not text:
        return False
    if text.startswith(("http://", "https://")) or "://" in text:
        return True
    return text.startswith(_URI_PREFIXES)


def _apps(ctx: ToolContext) -> dict[str, Any]:
    """The ``tools.apps`` map from the configuration, or an empty one."""
    try:
        apps = ctx.config.get("tools.apps", {}) if ctx.config is not None else {}
    except Exception as exc:  # noqa: BLE001 - a broken config must not kill the tool
        _log.warning("Could not read tools.apps: %s", exc)
        return {}
    return apps if isinstance(apps, dict) else {}


def _popen_kwargs() -> dict[str, Any]:
    """Extra ``subprocess`` keywords: hide the console window on Windows only."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def _start(target: str) -> str:
    """Launch ``target``, preferring ``os.startfile`` and falling back to ``start``.

    Returns the mechanism that worked; raises :class:`OSError` when both fail. The
    fallback goes through ``cmd /c start`` because that consults the Start menu and
    the App Paths registry, which is how a bare name like ``spotify`` resolves.
    """
    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        try:
            startfile(target)
            return "os.startfile"
        except OSError as exc:
            _log.debug("os.startfile(%r) failed: %s", target, exc)
    argv = ["cmd", "/c", "start", "", target]
    completed = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT,
        encoding="utf-8",
        errors="replace",
        **_popen_kwargs(),
    )
    if completed.returncode:
        raise OSError(
            f"'start' exited with {completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '').strip()}"
        )
    return "cmd /c start"


@tool(
    "open_app",
    description=(
        "Open an application or a configured shortcut on the user's machine by its "
        "everyday name, for example Spotify, Chrome, Steam, the settings or the calculator."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The application's everyday name, as the user said it.",
            }
        },
        "required": ["name"],
    },
    tier=Tier.SAFE,
)
def open_app(ctx: ToolContext, args: dict) -> ToolResult:
    """Launch a configured application, or try the Start menu with the raw name."""
    spoken = _normalise(args.get("name"))
    if not spoken:
        return ToolResult.fail("Which application should I open, sir?")

    apps = _apps(ctx)
    match = resolve_app(spoken, apps)
    key, target = match if match is not None else (spoken, spoken)
    label = friendly_name(key)

    try:
        how = _start(target)
    except Exception as exc:  # noqa: BLE001 - every launch failure ends in one sentence
        _log.warning("Could not open %r (target %r): %s", spoken, target, exc)
        detail = f"target={target!r} error={type(exc).__name__}: {exc}"
        if match is None:
            known = [friendly_name(str(key)) for key in list(apps)[:2]]
            hint = f", though I do have {' and '.join(known)}" if known else ""
            return ToolResult.fail(
                f"I couldn't find {label} on this machine{hint}, sir.", detail=detail
            )
        return ToolResult.fail(f"I couldn't open {label}, sir.", detail=detail)

    return ToolResult(
        ok=True,
        summary=f"Opening {label}, sir.",
        detail=f"Launched {target!r} via {how}.",
        data={"app": key, "target": target, "method": how},
    )


def _close_windows(title: str) -> int:
    """Close every visible window whose title matches, returning how many were closed."""
    import pygetwindow  # noqa: PLC0415 - Windows-only dependency

    closed = 0
    needle = _normalise(title)
    for window in pygetwindow.getAllWindows():
        window_title = _normalise(getattr(window, "title", ""))
        if not window_title or needle not in window_title:
            continue
        try:
            window.close()
            closed += 1
        except Exception as exc:  # noqa: BLE001 - one stubborn window is not fatal
            _log.debug("Could not close window %r: %s", window_title, exc)
    return closed


@tool(
    "close_app",
    description=(
        "Politely close a running application, asking it to quit so unsaved work is not "
        "lost. It is never forced."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The application's everyday name, as the user said it.",
            }
        },
        "required": ["name"],
    },
    tier=Tier.SAFE,
)
def close_app(ctx: ToolContext, args: dict) -> ToolResult:
    """Close an application gracefully: windows first, then ``taskkill`` without ``/F``."""
    spoken = _normalise(args.get("name"))
    if not spoken:
        return ToolResult.fail("Which application should I close, sir?")

    match = resolve_app(spoken, _apps(ctx))
    key, target = match if match is not None else (spoken, spoken)
    label = friendly_name(key)
    if looks_like_uri(target):
        return ToolResult.fail(
            f"{label} is a shortcut rather than a program, sir, so there's nothing to close.",
            detail=f"target={target!r} is a URL or shell URI.",
        )

    executable = Path(target).stem or target
    notes: list[str] = []

    try:
        closed = _close_windows(key)
        notes.append(f"pygetwindow closed {closed} window(s) matching {key!r}")
        if closed:
            return ToolResult(
                ok=True,
                summary=f"I've asked {label} to close, sir.",
                detail="\n".join(notes),
                data={"app": key, "windows_closed": closed, "method": "pygetwindow"},
            )
    except ImportError as exc:
        notes.append(f"pygetwindow unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - window enumeration is best effort
        notes.append(f"pygetwindow failed: {type(exc).__name__}: {exc}")

    argv = ["taskkill", "/IM", f"{executable}.exe"]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=TASKKILL_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        notes.append(f"$ {' '.join(argv)}\n{type(exc).__name__}: {exc}")
        _log.warning("Could not close %r: %s", spoken, exc)
        return ToolResult.fail(
            f"I couldn't close {label}, sir.", detail="\n".join(notes)
        )

    notes.append(
        f"$ {' '.join(argv)}\nexit code: {completed.returncode}\n"
        f"{(completed.stdout or '').strip()}\n{(completed.stderr or '').strip()}".strip()
    )
    data = {"app": key, "executable": executable, "returncode": completed.returncode}
    if completed.returncode == 0:
        return ToolResult(
            ok=True,
            summary=f"I've asked {label} to close, sir.",
            detail="\n".join(notes),
            data=data,
        )
    return ToolResult.fail(
        f"I couldn't find {label} running, sir.", detail="\n".join(notes)
    )


@tool(
    "open_url",
    description="Open a web address in the user's default browser.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The web address, with or without the https prefix.",
            }
        },
        "required": ["url"],
    },
    tier=Tier.SAFE,
)
def open_url(ctx: ToolContext, args: dict) -> ToolResult:
    """Open a http or https address, adding the scheme when the user left it out."""
    raw = " ".join(str(args.get("url") or "").split())
    if not raw:
        return ToolResult.fail("Which page should I open, sir?")

    candidate = raw
    if not _SCHEME_RE.match(candidate):
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        _log.info("Refusing to open a non-web address: %r", raw)
        return ToolResult.refuse(
            "I only open web pages, sir, and that isn't one.",
            detail=f"Rejected address {raw!r} (scheme={parsed.scheme!r}).",
        )

    try:
        opened = webbrowser.open(candidate)
    except Exception as exc:  # noqa: BLE001 - browser launch fails in many small ways
        _log.warning("Could not open %r: %s", candidate, exc)
        return ToolResult.fail(
            "I couldn't open the browser, sir.", detail=f"{type(exc).__name__}: {exc}"
        )
    site = parsed.netloc.removeprefix("www.")
    if not opened:
        return ToolResult.fail(
            f"I couldn't get a browser to open {site}, sir.",
            detail=f"webbrowser.open({candidate!r}) returned False.",
        )
    return ToolResult(
        ok=True,
        summary=f"Opening {site} in your browser, sir.",
        detail=f"Opened {candidate}",
        data={"url": candidate, "site": site},
    )
