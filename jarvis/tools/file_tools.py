"""File tools: finding a file by name, and moving, renaming or trashing it.

``find_file`` is a bounded search over the user's Desktop, Documents and Downloads.
``file_ops`` is the guarded one that touches the disk, and it is checked twice: the
dispatcher confirms out loud because the tool is GUARDED, and this module re-validates
the resolved path itself, so a direct call can never reach ``C:\\Windows``, a drive
root, or a path that escaped through ``..``. A delete never unlinks anything - the file
moves into ``%USERPROFILE%\\Jarvis Trash\\<date>\\``, so a mistake is recoverable.

Pure standard library, so this module imports cleanly on Linux CI.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = [
    "find_file", "file_ops", "FileHit", "FILE_ACTIONS", "MAX_DEPTH", "MAX_ENTRIES",
    "MAX_RESULTS", "MAX_WILDCARD_MATCHES", "DIRECTORY_CONFIRM_FILES", "TRASH_FOLDER_NAME",
]

_log = get_logger("tools.files")

MAX_DEPTH = 4                    #: directory levels visited below each search root
MAX_ENTRIES = 40000              #: entry budget, so a search never stalls the voice loop
MAX_RESULTS = 5                  #: hits reported back to the model
MAX_WILDCARD_MATCHES = 20        #: a wildcard matching more files than this is refused
DIRECTORY_CONFIRM_FILES = 50     #: a bigger folder needs a second, explicit confirmation
TRASH_FOLDER_NAME = "Jarvis Trash"          #: used instead of deleting anything
FILE_ACTIONS = ("move", "delete", "rename")  #: accepted values of ``action``

#: Directory names never walked: caches, repositories and machine-managed folders.
_SKIP_DIR_NAMES = {
    "appdata", "application data", "node_modules", "__pycache__", ".git", ".svn", ".hg",
    ".cache", ".venv", "venv", "site-packages", "$recycle.bin", "windows", "temp", "tmp",
    "system volume information", "program files", "program files (x86)", "programdata",
    "onedrivetemp", ".idea", ".vscode",
}
#: Path segments refused wherever they appear.
_FORBIDDEN_ANYWHERE = {
    "system32", "syswow64", "$recycle.bin", "recycler", "recycle bin", "winsxs",
    "system volume information",
}
#: Folders directly under a drive root that are refused.
_FORBIDDEN_TOP_LEVEL = {
    "windows", "winnt", "program files", "program files (x86)", "programdata", "boot",
    "$windows.~bt", "$windows.~ws", "recovery",
}
_WILDCARD_RE = re.compile(r"[*?\[]")
_SEPARATOR_RE = re.compile(r"[\\/]")
#: ``C:\...`` or a ``\\server\share`` UNC path — absolute even on a Linux test box.
_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]|^\\\\")
_NUMBER_WORDS = (
    "no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)
_TENS_WORDS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
               6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


@dataclass
class FileHit:
    """One matching file: where it is, how big it is, and when it changed."""

    path: Path
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.name


def _spoken_count(value: int) -> str:
    """Render a count as words, so the spoken sentence reads naturally."""
    number = max(0, int(value))
    if number < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS_WORDS[tens] if unit == 0 else f"{_TENS_WORDS[tens]}-{_NUMBER_WORDS[unit]}"
    return str(number)


def _user_profile() -> Path:
    """The user's home (``%USERPROFILE%`` on Windows): the single place tests patch."""
    profile = os.environ.get("USERPROFILE")
    return Path(profile) if profile else Path.home()


def _as_path(raw: str) -> Path:
    """Parse a path the user spoke or typed, understanding ``\\`` on any platform.

    The target is Windows, but CI is Linux, where a backslash is an ordinary
    character: without this, ``C:\\Windows\\System32\\x`` would be read as one long
    file name and the system-folder rules below would never see its segments.
    """
    text = str(raw).strip().strip('"').strip("'")
    if os.sep != "\\" and "\\" in text:
        text = text.replace("\\", "/")
    return Path(text).expanduser()


def _looks_absolute(raw: str, path: Path) -> bool:
    """True for a real absolute path and for a Windows drive or UNC path."""
    return path.is_absolute() or bool(_DRIVE_RE.match(str(raw).strip().strip('"').strip("'")))


def _config_dirs(ctx: ToolContext, key: str, default: list[str]) -> list[Path]:
    """Read a list of directories from the config, expanded against the user profile."""
    configured = ctx.config.get(key, default)
    if isinstance(configured, str):
        configured = [configured]
    profile, roots = _user_profile(), []
    for entry in configured or []:
        candidate = Path(str(entry)).expanduser()
        roots.append(candidate if candidate.is_absolute() else profile / candidate)
    return roots


def _search_roots(ctx: ToolContext) -> list[Path]:
    """The directories that are searched, from ``tools.file_search_dirs``."""
    roots: list[Path] = []
    for path in _config_dirs(ctx, "tools.file_search_dirs", ["Desktop", "Documents", "Downloads"]):
        if path.is_dir() and path not in roots:
            roots.append(path)
    profile = _user_profile()
    if not roots and profile.is_dir():
        roots.append(profile)  # a stripped-down profile still deserves an answer
    return roots


def _matches(name: str, needle: str) -> bool:
    """Case-insensitive substring match, plus an ``fnmatch`` glob on the file name."""
    lowered, pattern = name.lower(), needle.lower()
    if fnmatch.fnmatch(lowered, pattern):
        return True
    return not _WILDCARD_RE.search(pattern) and pattern in lowered


def _is_hidden(entry: os.DirEntry) -> bool:
    """True for dot-files and, on Windows, for hidden or system entries."""
    if entry.name.startswith(".") or entry.name.startswith("$"):
        return True
    try:
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attributes & 0x2 or attributes & 0x4)  # HIDDEN | SYSTEM


def _search(needle: str, roots: Iterable[Path]) -> tuple[list[FileHit], int, bool]:
    """Depth- and budget-limited breadth-first search -> (hits, visited, truncated).

    Symlinks are not followed and each directory is visited once, so a folder loop
    cannot turn this into an endless walk; the entry budget caps the worst case.
    """
    hits: list[FileHit] = []
    visited = 0
    queue: deque[tuple[Path, int]] = deque((root, 0) for root in roots)
    seen: set[str] = set()
    while queue:
        directory, depth = queue.popleft()
        if str(directory).lower() in seen:
            continue
        seen.add(str(directory).lower())
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > MAX_ENTRIES:
                        _log.warning("Search for %r hit the %d entry budget", needle, MAX_ENTRIES)
                        return hits, visited, True
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_dir:
                        if (depth + 1 < MAX_DEPTH and entry.name.lower() not in _SKIP_DIR_NAMES
                                and not _is_hidden(entry)):
                            queue.append((Path(entry.path), depth + 1))
                        continue
                    if not _matches(entry.name, needle) or _is_hidden(entry):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    hits.append(FileHit(Path(entry.path), int(info.st_size), float(info.st_mtime)))
        except OSError as exc:
            _log.debug("Skipping unreadable directory %s: %s", directory, exc)
    return hits, visited, False


def _newest_first(hits: Iterable[FileHit]) -> list[FileHit]:
    """Sort matches by modification time, newest first, name as the tie-breaker."""
    return sorted(hits, key=lambda hit: (-hit.mtime, hit.path.name.lower()))


def _describe(hits: list[FileHit]) -> str:
    """Numbered full paths with sizes and timestamps, for ``detail`` (never spoken)."""
    lines = []
    for index, hit in enumerate(hits, start=1):
        stamp = datetime.fromtimestamp(hit.mtime).strftime("%Y-%m-%d %H:%M")
        size = f"{hit.size} bytes" if hit.size < 1024 else f"{hit.size / 1024:.1f} KB"
        lines.append(f"{index}. {hit.path} ({size}, modified {stamp})")
    return "\n".join(lines)


def _where(path: Path) -> str:
    """A short spoken location: the containing folder, or the user folder itself."""
    parent = path.parent
    if str(parent).lower() == str(_user_profile()).lower():
        return "your user folder"
    return f"your {parent.name} folder" if parent.name else str(parent)


@tool(
    "find_file",
    description=(
        "Find a file by name or part of a name in the user's Desktop, Documents and "
        "Downloads folders. Wildcards such as '*.pdf' are supported."
    ),
    parameters={
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string", "description":
                                "The file name, part of it, or a glob such as '*.pdf'."}},
    },
    tier=Tier.SAFE,
)
def find_file(ctx: ToolContext, args: dict) -> ToolResult:
    """Search the configured folders and report the five newest matches."""
    needle = str(args.get("name") or "").strip().strip('"').strip("'")
    if not needle:
        return ToolResult.fail("I need a file name to look for, sir.")
    roots = _search_roots(ctx)
    if not roots:
        return ToolResult.fail("I have no folders to search, sir.",
                               detail=f"No search directory exists under {_user_profile()}.")
    hits, visited, truncated = _search(needle, roots)
    ranked = _newest_first(hits)[:MAX_RESULTS]
    detail = _describe(ranked) or f"No match for {needle!r} in: " + ", ".join(map(str, roots))
    if truncated:
        detail = f"{detail}\n(Search stopped after {visited} entries.)"
    data = {"query": needle, "count": len(hits), "paths": [str(hit.path) for hit in ranked]}
    if not ranked:
        return ToolResult(True, f"I found nothing matching {needle}, sir.", detail, data)
    best, others = ranked[0], len(hits) - 1
    place = _where(best.path)
    if others <= 0:
        summary = f"Found {best.name} in {place}, sir, and nothing else like it."
    elif others == 1:
        summary = f"Found {best.name} in {place}, sir, and one other match."
    else:
        summary = (f"Found {best.name} in {place}, sir, along with "
                   f"{_spoken_count(others)} other matches.")
    return ToolResult(True, summary, detail, data)


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or lives under it, compared case-insensitively."""
    parts = [part.lower() for part in path.parts]
    root_parts = [part.lower() for part in root.parts]
    return len(parts) >= len(root_parts) and parts[: len(root_parts)] == root_parts


def _safety_error(ctx: ToolContext, path: Path, raw: str) -> str | None:
    """Return why ``path`` is off limits, or ``None`` when it may be touched.

    Enforced here, not only in the dispatcher, so the rule holds for a direct call.
    Outside the user profile a path needs an explicitly allowed root
    (``tools.file_allowed_dirs``) or a folder level below the drive root.
    """
    if ".." in _as_path(raw).parts or ".." in path.parts:
        return "the path tries to climb out of its folder with two dots"
    parts = [part.strip("\\/").lower() for part in path.parts]
    if len(parts) < 2:
        return "that is a drive root, not a file"
    if any(part in _FORBIDDEN_ANYWHERE for part in parts[1:]):
        return "it sits inside a protected system folder"
    if parts[1] in _FORBIDDEN_TOP_LEVEL:
        return "it belongs to Windows itself"
    if _is_within(path, _user_profile()):
        return None
    if any(_is_within(path, root) for root in _config_dirs(ctx, "tools.file_allowed_dirs", [])):
        return None
    if len(parts) < 3:
        return "it sits directly in a drive root rather than in a folder of yours"
    return None


def _resolve_source(ctx: ToolContext, raw: str) -> tuple[Path | None, ToolResult | None]:
    """Turn ``raw`` into one path, or explain why that was not possible.

    Accepts an absolute path, a path relative to the user profile, or a bare name
    searched for the way ``find_file`` searches. Ambiguity is never resolved by
    guessing: two or more candidates is a refusal naming two of them.
    """
    candidate = _as_path(raw)
    if ".." in candidate.parts:
        return None, ToolResult.refuse(
            "I will not follow a path that climbs out of its folder with two dots, sir.",
            detail=f"Rejected source {raw!r}.")
    if _looks_absolute(raw, candidate):
        return candidate, None
    if not _WILDCARD_RE.search(raw) and (_user_profile() / candidate).exists():
        return _user_profile() / candidate, None
    roots = _search_roots(ctx)
    hits, _visited, _truncated = _search(candidate.name or raw, roots)
    if _WILDCARD_RE.search(raw) and len(hits) > MAX_WILDCARD_MATCHES:
        return None, ToolResult.refuse(
            f"That pattern matches {_spoken_count(len(hits))} files, sir, "
            "far too many for me to touch at once.",
            detail=f"{raw!r} matched {len(hits)} files; the limit is {MAX_WILDCARD_MATCHES}.")
    if not hits:
        return None, ToolResult.fail(
            f"I could not find {raw} anywhere in your usual folders, sir.",
            detail=f"No match for {raw!r} under: " + ", ".join(map(str, roots)))
    unique = {str(hit.path).lower(): hit for hit in hits}
    # An exact name beats the substring matches around it: "final.txt" means that file,
    # not "report final.txt" that merely contains it.
    exact = {key: hit for key, hit in unique.items() if hit.name.lower() == candidate.name.lower()}
    unique = exact or unique
    if len(unique) > 1:
        ranked = _newest_first(unique.values())
        return None, ToolResult.refuse(
            f"There are {_spoken_count(len(unique))} files by that name, sir - did you mean "
            f"the one in {_where(ranked[0].path)} or the one in {_where(ranked[1].path)}?",
            detail=_describe(ranked[:MAX_RESULTS]))
    return next(iter(unique.values())).path, None


def _unique_target(target: Path) -> Path:
    """``report.txt`` becomes ``report (1).txt`` when something is already there."""
    if not target.exists():
        return target
    for index in range(1, 1000):
        candidate = target.parent / f"{target.stem} ({index}){target.suffix}"
        if not candidate.exists():
            return candidate
    return target.parent / f"{target.stem} ({os.getpid()}){target.suffix}"  # pragma: no cover


def _count_files(directory: Path, limit: int) -> int:
    """Count files under ``directory``, stopping once ``limit`` is exceeded."""
    total = 0
    for _root, _dirs, files in os.walk(directory):
        total += len(files)
        if total > limit:
            break
    return total


def _delete(ctx: ToolContext, source: Path) -> ToolResult:
    """Move ``source`` into today's Jarvis Trash folder instead of unlinking it."""
    profile = _user_profile()
    if source.is_dir():
        if not _is_within(source, profile):
            return ToolResult.refuse("I will not delete a folder outside your user folder, sir.",
                                     detail=f"{source} is not under {profile}.")
        count = _count_files(source, DIRECTORY_CONFIRM_FILES)
        if count > DIRECTORY_CONFIRM_FILES:
            # The generic guarded confirmation never mentioned the size, so ask again
            # with the number spelled out before moving that many files at once.
            question = (f"That folder holds more than {_spoken_count(DIRECTORY_CONFIRM_FILES)} "
                        f"files, sir. Shall I really move {source.name} to the trash?")
            try:
                approved = bool(ctx.confirm(question))
            except Exception as exc:  # noqa: BLE001 - a broken confirm must not delete
                _log.error("Confirmation failed for %s: %s", source, exc, exc_info=True)
                approved = False
            if not approved:
                return ToolResult.refuse(
                    f"Leaving {source.name} exactly where it is, sir.",
                    detail=f"Delete of {source} ({count}+ files) was not confirmed.")
    trash = profile / TRASH_FOLDER_NAME / date.today().isoformat()
    trash.mkdir(parents=True, exist_ok=True)
    target = _unique_target(trash / source.name)
    shutil.move(str(source), str(target))
    _log.info("Trashed %s -> %s", source, target)
    return ToolResult(True, f"Moved {source.name} to the Jarvis Trash folder, sir - "
                            "recoverable if you change your mind.", f"{source} -> {target}",
                      {"action": "delete", "source": str(source), "destination": str(target)})


def _destination_dir(ctx: ToolContext, raw: str) -> tuple[Path | None, ToolResult | None]:
    """Resolve a move destination to a directory, creating it when its parent exists."""
    candidate = _as_path(raw)
    if not _looks_absolute(raw, candidate):
        candidate = _user_profile() / candidate
    reason = _safety_error(ctx, candidate, raw)
    if reason is not None:
        return None, ToolResult.refuse(f"I cannot move anything there, sir, because {reason}.",
                                       detail=f"Destination {candidate} refused: {reason}.")
    if candidate.is_file():
        return None, ToolResult.fail(
            f"{candidate.name} is a file, sir, not a folder to move things into.",
            detail=f"Destination {candidate} is an existing file.")
    if not candidate.is_dir():
        if not candidate.parent.is_dir():
            return None, ToolResult.fail(
                f"I could not find a folder called {candidate.name}, sir.",
                detail=f"Neither {candidate} nor its parent {candidate.parent} exists.")
        candidate.mkdir()
        _log.info("Created destination folder %s", candidate)
    return candidate, None


def _move(ctx: ToolContext, source: Path, destination: str) -> ToolResult:
    """Move ``source`` into the destination directory, never overwriting anything."""
    if not destination:
        return ToolResult.fail("I need a folder to move it to, sir.")
    directory, error = _destination_dir(ctx, destination)
    if error is not None or directory is None:
        return error or ToolResult.fail("I could not work out where to move that, sir.")
    if str(source.parent).lower() == str(directory).lower():
        return ToolResult(True, f"{source.name} is already in your {directory.name} folder, sir.",
                          f"{source} is already inside {directory}.",
                          {"action": "move", "source": str(source), "destination": str(source)})
    target = _unique_target(directory / source.name)
    shutil.move(str(source), str(target))
    _log.info("Moved %s -> %s", source, target)
    return ToolResult(True, f"Moved {source.name} to your {directory.name} folder, sir.",
                      f"{source} -> {target}",
                      {"action": "move", "source": str(source), "destination": str(target)})


def _rename(source: Path, destination: str) -> ToolResult:
    """Rename ``source`` in place; the new name may not contain a path."""
    new_name = destination.strip('"').strip("'").strip()
    if not new_name:
        return ToolResult.fail("I need a new name for it, sir.")
    if _SEPARATOR_RE.search(new_name) or ":" in new_name or new_name in (".", ".."):
        return ToolResult.refuse(
            "A new name cannot contain a folder path, sir - give me the name on its own.",
            detail=f"Rejected rename target {new_name!r}.")
    if not Path(new_name).suffix and source.suffix:
        new_name += source.suffix  # keep the extension the user did not say out loud
    target = source.parent / new_name
    if target.exists():
        return ToolResult.refuse(f"There is already something called {new_name} there, sir.",
                                 detail=f"Rename refused: {target} exists.")
    source.rename(target)
    _log.info("Renamed %s -> %s", source, target)
    return ToolResult(True, f"Renamed {source.name} to {new_name}, sir.", f"{source} -> {target}",
                      {"action": "rename", "source": str(source), "destination": str(target)})


@tool(
    "file_ops",
    description=(
        "Move, rename or delete one of the user's files. Deleting moves the file to a "
        "recoverable Jarvis Trash folder; it never erases anything."
    ),
    parameters={
        "type": "object",
        "required": ["action", "source"],
        "properties": {
            "action": {"type": "string", "enum": list(FILE_ACTIONS), "description":
                       "What to do with the file: move, delete or rename."},
            "source": {"type": "string", "description":
                       "The file: a full path, a path under the user folder, or just its name."},
            "destination": {"type": "string", "description":
                            "Target folder for a move, or the new file name for a rename."},
        },
    },
    tier=Tier.GUARDED,
    announce="I'm about to {action} {source}, sir.",
)
def file_ops(ctx: ToolContext, args: dict) -> ToolResult:
    """Resolve the file, re-check every safety rule, then perform the operation."""
    action = str(args.get("action") or "").strip().lower()
    raw_source = str(args.get("source") or "").strip().strip('"').strip("'")
    destination = str(args.get("destination") or "").strip()
    if action not in FILE_ACTIONS:
        return ToolResult.fail(
            "I can only move, rename or delete a file, sir.",
            detail=f"Unsupported action {action!r}; expected one of {', '.join(FILE_ACTIONS)}.")
    if not raw_source:
        return ToolResult.fail(f"I need to know which file to {action}, sir.")
    source, error = _resolve_source(ctx, raw_source)
    if error is not None or source is None:
        return error or ToolResult.fail(f"I could not find {raw_source}, sir.")
    try:
        # normpath first, so "Documents\\..\\Windows" is judged on where it really
        # points; then realpath, so a symlink cannot smuggle us out of the profile.
        resolved = Path(os.path.normpath(str(source)))
        if resolved.exists():
            resolved = Path(os.path.realpath(str(resolved)))
    except OSError as exc:  # pragma: no cover - an unreadable path
        return ToolResult.fail("I could not read that path, sir.", detail=f"{source}: {exc}")
    reason = _safety_error(ctx, resolved, raw_source)
    if reason is not None:
        _log.warning("Refused %s of %s: %s", action, resolved, reason)
        return ToolResult.refuse(f"I will not {action} that, sir, because {reason}.",
                                 detail=f"Refused {action} of {resolved}: {reason}.")
    if not resolved.exists():
        return ToolResult.fail(f"There is nothing at that path to {action}, sir.",
                               detail=f"{resolved} does not exist.")
    if resolved.is_dir() and action != "delete":
        return ToolResult.refuse(f"{resolved.name} is a folder, sir, and I only {action} files.",
                                 detail=f"Refused {action} of directory {resolved}.")
    try:
        if action == "delete":
            return _delete(ctx, resolved)
        if action == "move":
            return _move(ctx, resolved, destination)
        return _rename(resolved, destination)
    except (OSError, shutil.Error) as exc:
        _log.error("Could not %s %s: %s", action, resolved, exc, exc_info=True)
        return ToolResult.fail(f"I could not {action} {resolved.name}, sir - the system refused.",
                               detail=f"{type(exc).__name__}: {exc}")
