"""Finding an application the way a person does: by typing its name into Start.

``start spotify`` only works when something called ``spotify`` is on the PATH or in
the App Paths registry, which for most applications it is not - Windows pops up
"Windows cannot find 'spotify'" and, worse, ``start`` still exits zero, so the
caller cheerfully reports success while nothing opened.

Windows already knows every application it can launch. ``Get-StartApps`` lists
them with their AppUserModelIDs, covering Store apps and desktop apps alike, and
``explorer.exe shell:AppsFolder\\<id>`` launches any of them. The Start menu's own
shortcut folders are a second source for anything that is somehow missing. This
module asks both, so JARVIS opens what the user actually has installed.

Nothing here is imported at module load on another platform: every Windows call is
inside a function and every failure degrades to "I could not find it".
"""

from __future__ import annotations

import difflib
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AppTarget", "start_apps", "shortcut_for", "resolve", "prewarm", "clear_cache"]

logger = logging.getLogger("jarvis.tools.windows_apps")

CREATE_NO_WINDOW = 0x08000000
LIST_TIMEOUT = 12.0

_cache_lock = threading.Lock()
_start_apps_cache: list[tuple[str, str]] | None = None


@dataclass
class AppTarget:
    """Something that can actually be launched, and what to call it out loud."""

    kind: str          # "appsfolder" | "shortcut" | "path" | "uri"
    target: str
    label: str

    def launch(self) -> str:
        """Start it. Returns the mechanism used; raises OSError when it will not run."""
        if self.kind == "appsfolder":
            argv = ["explorer.exe", f"shell:AppsFolder\\{self.target}"]
            # explorer.exe returns 1 even on success, so its exit code says nothing.
            subprocess.Popen(argv, creationflags=_flags())
            return "shell:AppsFolder"

        startfile = getattr(os, "startfile", None)
        if callable(startfile):
            startfile(self.target)
            return "os.startfile"
        raise OSError("os.startfile is only available on Windows")


def _flags() -> int:
    return CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _powershell(command: str, timeout: float = LIST_TIMEOUT) -> str:
    """Run a PowerShell one-liner and return stdout. Empty string on any failure."""
    if sys.platform != "win32":
        return ""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", creationflags=_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("PowerShell call failed: %s", exc)
        return ""
    if completed.returncode:
        logger.debug("PowerShell exited %s: %s", completed.returncode, completed.stderr[:200])
    return completed.stdout or ""


def start_apps(refresh: bool = False) -> list[tuple[str, str]]:
    """Every application the Start menu knows, as ``(name, AppUserModelID)``.

    Cached for the session: the call costs the better part of a second and the list
    only changes when something is installed.
    """
    global _start_apps_cache
    with _cache_lock:
        if _start_apps_cache is not None and not refresh:
            return list(_start_apps_cache)

    raw = _powershell(
        "Get-StartApps | ForEach-Object { \"$($_.Name)`t$($_.AppID)\" }"
    )
    apps: list[tuple[str, str]] = []
    for line in raw.splitlines():
        name, sep, app_id = line.partition("\t")
        if sep and name.strip() and app_id.strip():
            apps.append((name.strip(), app_id.strip()))

    with _cache_lock:
        _start_apps_cache = apps
    logger.info("Start menu knows %d application(s).", len(apps))
    return list(apps)


def clear_cache() -> None:
    """Forget the cached list, so a newly installed app can be found."""
    global _start_apps_cache
    with _cache_lock:
        _start_apps_cache = None


def prewarm() -> None:
    """Build the cache on a background thread, so the first 'open Spotify' is quick."""
    if sys.platform != "win32":
        return
    threading.Thread(target=start_apps, name="jarvis-startapps", daemon=True).start()


def _start_menu_dirs() -> list[Path]:
    roots = []
    for variable in ("APPDATA", "ProgramData"):
        base = os.environ.get(variable)
        if base:
            roots.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return [path for path in roots if path.is_dir()]


def shortcut_for(name: str) -> Path | None:
    """The best matching ``.lnk`` in either Start menu folder."""
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    candidates: list[tuple[str, Path]] = []
    for root in _start_menu_dirs():
        try:
            for path in root.rglob("*.lnk"):
                candidates.append((path.stem.lower(), path))
        except OSError as exc:  # pragma: no cover - a locked directory is not fatal
            logger.debug("Could not read %s: %s", root, exc)
    if not candidates:
        return None
    index = _best_match(wanted, [stem for stem, _ in candidates])
    return candidates[index][1] if index is not None else None


def _best_match(wanted: str, names: list[str]) -> int | None:
    """Exact, then prefix, then substring, then fuzzy - in that order of confidence."""
    lowered = [name.lower() for name in names]
    for index, name in enumerate(lowered):
        if name == wanted:
            return index
    for index, name in enumerate(lowered):
        if name.startswith(wanted):
            return index
    contains = [index for index, name in enumerate(lowered) if wanted in name]
    if contains:
        # The shortest containing name is the least surprising: "Spotify" over
        # "Spotify Web Helper".
        return min(contains, key=lambda index: len(lowered[index]))
    close = difflib.get_close_matches(wanted, lowered, n=1, cutoff=0.72)
    if close:
        return lowered.index(close[0])

    # "vs code" is not a substring of "visual studio code", but "code" is. Try the
    # longest word on its own before giving up - people abbreviate constantly.
    words = sorted((word for word in wanted.split() if len(word) > 2), key=len, reverse=True)
    for word in words:
        hits = [index for index, name in enumerate(lowered) if word in name.split()]
        if hits:
            return min(hits, key=lambda index: len(lowered[index]))
    return None


def resolve(name: str) -> AppTarget | None:
    """Find something launchable for ``name``, or None when Windows has never heard of it."""
    wanted = str(name or "").strip()
    if not wanted or sys.platform != "win32":
        return None

    apps = start_apps()
    index = _best_match(wanted.lower(), [app_name for app_name, _ in apps])
    if index is not None:
        app_name, app_id = apps[index]
        logger.debug("Resolved %r to Start app %r (%s)", wanted, app_name, app_id)
        return AppTarget(kind="appsfolder", target=app_id, label=app_name)

    shortcut = shortcut_for(wanted)
    if shortcut is not None:
        logger.debug("Resolved %r to shortcut %s", wanted, shortcut)
        return AppTarget(kind="shortcut", target=str(shortcut), label=shortcut.stem)

    return None
