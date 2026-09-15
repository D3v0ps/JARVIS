"""Long-term memory for JARVIS: a tiny, tolerant JSON fact store.

The assistant remembers a handful of short facts about its user ("the user's name is
Karim", "the user prefers Celsius") and embeds them in the system prompt through
:meth:`Memory.as_prompt_block`.

Design rules:

* **Never raise on load.** A missing, empty, truncated or hand-edited ``memory.json``
  degrades to an empty store plus a WARNING — a broken file must not stop the assistant
  from booting.
* **Atomic save.** The store is written to a temporary file in the same directory and
  moved into place with :func:`os.replace`, so a crash mid-write cannot corrupt it.
* **Bounded.** At most :data:`MAX_FACTS` facts are kept (oldest dropped first) so the
  system prompt cannot grow without bound.

File format::

    {"facts": [{"text": ..., "created": ..., "topic": ...}], "version": 1}
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from jarvis.core.logging import get_logger

__all__ = ["MemoryEntry", "Memory", "MAX_FACTS", "MEMORY_VERSION", "PROMPT_HEADER"]

logger = get_logger("memory")

#: Schema version written to disk.
MEMORY_VERSION = 1

#: Hard cap on stored facts; the oldest entries are dropped once it is exceeded.
MAX_FACTS = 200

#: Header line of the block embedded in the system prompt.
PROMPT_HEADER = "Known facts about the user:"

#: Facts longer than this are truncated before being stored.
MAX_FACT_CHARS = 400

_WHITESPACE_RE = re.compile(r"\s+")

# Patterns that expose the user's name, English and Swedish.
_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:the\s+)?user(?:'s|’s|s)?\s+name\s+is\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy\s+name\s+is\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi\s*(?:'m|’m|\s+am)\s+called\s+(.+)", re.IGNORECASE),
    re.compile(r"\bcall\s+me\s+(.+)", re.IGNORECASE),
    re.compile(r"\banvändaren\s+heter\s+(.+)", re.IGNORECASE),
    re.compile(r"\banvändarens\s+namn\s+är\s+(.+)", re.IGNORECASE),
    re.compile(r"\bjag\s+heter\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmitt\s+namn\s+är\s+(.+)", re.IGNORECASE),
)

# A name token: letters (incl. Swedish), hyphen and apostrophe, nothing else.
_NAME_TOKEN_RE = re.compile(r"^[^\W\d_][\w'’-]*$", re.UNICODE)
_MAX_NAME_TOKENS = 3


def _collapse(text: str) -> str:
    """Collapse all runs of whitespace into single spaces and strip the ends."""
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _normalise(text: str) -> str:
    """Comparison key for de-duplication: whitespace-collapsed and case-folded."""
    return _collapse(text).casefold()


def _now_iso() -> str:
    """Current local time as an ISO-8601 string with seconds precision."""
    return datetime.now().isoformat(timespec="seconds")


def _capitalise_name(name: str) -> str:
    """Capitalise each word of ``name`` without destroying existing inner capitals."""
    parts: list[str] = []
    for word in name.split(" "):
        if not word:
            continue
        if any(ch.isupper() for ch in word[1:]):
            parts.append(word[0].upper() + word[1:])  # keep McDonald, O'Neill, etc.
        else:
            parts.append(word[:1].upper() + word[1:].lower())
    return " ".join(parts)


@dataclass
class MemoryEntry:
    """One remembered fact."""

    text: str
    created: str = field(default_factory=_now_iso)  # ISO-8601 local time
    topic: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-serialisable form of this entry."""
        return {"text": self.text, "created": self.created, "topic": self.topic}

    @classmethod
    def from_dict(cls, raw: Any) -> "MemoryEntry | None":
        """Build an entry from arbitrary parsed JSON, or ``None`` when unusable."""
        if isinstance(raw, str):  # tolerate a bare list of strings
            text = _collapse(raw)
            return cls(text=text) if text else None
        if not isinstance(raw, dict):
            return None
        text = _collapse(str(raw.get("text", "")))
        if not text:
            return None
        created = str(raw.get("created") or "").strip() or _now_iso()
        topic = _collapse(str(raw.get("topic") or ""))
        return cls(text=text[:MAX_FACT_CHARS], created=created, topic=topic)


class Memory:
    """A small, persistent set of facts about the user.

    All mutating operations write the store back to disk immediately, so an unclean
    shutdown never loses a fact the assistant just confirmed out loud.
    """

    def __init__(self, path: str | Path = "memory.json") -> None:
        self._path = Path(path)
        self._facts: list[MemoryEntry] = []
        self._lock = RLock()
        self.load()

    # --- properties ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """Path of the backing JSON file."""
        return self._path

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)

    def __iter__(self) -> Iterable[MemoryEntry]:
        return iter(self.facts())

    # --- persistence -----------------------------------------------------------------
    def load(self) -> None:
        """Load the store from disk; any problem degrades to an empty, logged store."""
        with self._lock:
            self._facts = []
            try:
                raw_text = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.debug("No memory file at %s yet; starting empty.", self._path)
                return
            except UnicodeDecodeError as exc:
                # A file written in a legacy Windows code page. Salvage what we can
                # rather than refusing to boot over one mis-encoded byte.
                logger.warning(
                    "Memory file %s is not valid UTF-8 (%s); reading it leniently.",
                    self._path, exc,
                )
                try:
                    raw_text = self._path.read_text(encoding="utf-8", errors="replace")
                except OSError as read_exc:
                    logger.warning("Could not read memory file %s: %s", self._path, read_exc)
                    return
            except OSError as exc:
                logger.warning("Could not read memory file %s: %s", self._path, exc)
                return

            if not raw_text.strip():
                logger.warning("Memory file %s is empty; starting with no facts.", self._path)
                return

            try:
                data: Any = json.loads(raw_text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Memory file %s is not valid JSON (%s); starting with no facts.",
                    self._path, exc,
                )
                return

            raw_facts: Any
            if isinstance(data, dict):
                raw_facts = data.get("facts", [])
            elif isinstance(data, list):  # tolerate a bare list written by an older build
                raw_facts = data
            else:
                logger.warning(
                    "Memory file %s has an unexpected shape (%s); starting with no facts.",
                    self._path, type(data).__name__,
                )
                return

            if not isinstance(raw_facts, list):
                logger.warning(
                    "Memory file %s has a non-list 'facts' field (%s); starting with no facts.",
                    self._path, type(raw_facts).__name__,
                )
                return

            skipped = 0
            entries: list[MemoryEntry] = []
            seen: set[str] = set()
            for item in raw_facts:
                entry = MemoryEntry.from_dict(item)
                if entry is None:
                    skipped += 1
                    continue
                key = _normalise(entry.text)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                entries.append(entry)

            if skipped:
                logger.warning(
                    "Ignored %d unusable entr%s while loading %s.",
                    skipped, "y" if skipped == 1 else "ies", self._path,
                )
            self._facts = entries
            self._enforce_cap()
            logger.debug("Loaded %d fact(s) from %s.", len(self._facts), self._path)

    def save(self) -> None:
        """Write the store atomically (temp file in the same directory + ``os.replace``)."""
        with self._lock:
            payload = {
                "facts": [entry.to_dict() for entry in self._facts],
                "version": MEMORY_VERSION,
            }
            target = self._path
            try:
                parent = target.parent if str(target.parent) else Path(".")
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create directory for %s: %s", target, exc)
                return

            tmp_path: str | None = None
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=str(parent)
                )
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, target)
                tmp_path = None
                logger.debug("Saved %d fact(s) to %s.", len(self._facts), target)
            except OSError as exc:
                logger.warning("Could not save memory to %s: %s", target, exc)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError as exc:  # pragma: no cover - defensive
                        logger.warning("Could not remove temp file %s: %s", tmp_path, exc)

    # --- mutation --------------------------------------------------------------------
    def remember(self, fact: str, topic: str = "") -> MemoryEntry:
        """Store ``fact``; a near-identical existing fact is replaced, not duplicated.

        "Near-identical" means the same text after collapsing whitespace, compared
        case-insensitively. Returns the stored entry (empty text when nothing was given).
        """
        clean = _collapse(fact)
        clean_topic = _collapse(topic)
        if not clean:
            logger.warning("remember() called with an empty fact; nothing stored.")
            return MemoryEntry(text="", created=_now_iso(), topic=clean_topic)
        if len(clean) > MAX_FACT_CHARS:
            logger.warning("Fact truncated to %d characters before storing.", MAX_FACT_CHARS)
            clean = clean[:MAX_FACT_CHARS]

        entry = MemoryEntry(text=clean, created=_now_iso(), topic=clean_topic)
        key = _normalise(clean)
        with self._lock:
            for index, existing in enumerate(self._facts):
                if _normalise(existing.text) == key:
                    # Keep the position, refresh the wording, timestamp and topic.
                    entry.topic = clean_topic or existing.topic
                    self._facts[index] = entry
                    logger.info("Refreshed an existing fact: %s", clean)
                    self.save()
                    return entry
            self._facts.append(entry)
            self._enforce_cap()
            logger.info("Remembered: %s", clean)
            self.save()
        return entry

    def forget(self, topic: str) -> int:
        """Remove every fact whose text or topic contains ``topic`` (case-insensitive).

        Returns the number of facts removed.
        """
        needle = _normalise(topic)
        if not needle:
            logger.warning("forget() called with an empty topic; nothing removed.")
            return 0
        with self._lock:
            kept: list[MemoryEntry] = []
            removed = 0
            for entry in self._facts:
                if needle in _normalise(entry.text) or needle in _normalise(entry.topic):
                    removed += 1
                    logger.info("Forgot: %s", entry.text)
                    continue
                kept.append(entry)
            if removed:
                self._facts = kept
                self.save()
            else:
                logger.info("Nothing stored matches '%s'.", _collapse(topic))
            return removed

    def clear(self) -> int:
        """Remove every fact. Returns how many were dropped."""
        with self._lock:
            count = len(self._facts)
            self._facts = []
            if count:
                logger.info("Cleared %d fact(s) from memory.", count)
                self.save()
            return count

    # --- reading ---------------------------------------------------------------------
    def facts(self) -> list[MemoryEntry]:
        """Return a shallow copy of the stored facts, oldest first."""
        with self._lock:
            return list(self._facts)

    def as_prompt_block(self) -> str:
        """Return a compact block for the system prompt, or ``""`` when empty."""
        entries = self.facts()
        if not entries:
            return ""
        lines = [PROMPT_HEADER]
        for entry in entries:
            if entry.topic:
                lines.append(f"- [{entry.topic}] {entry.text}")
            else:
                lines.append(f"- {entry.text}")
        return "\n".join(lines) + "\n"

    @property
    def user_name(self) -> str | None:
        """The user's name when a stored fact reveals it, capitalised; else ``None``."""
        for entry in reversed(self.facts()):
            name = self._extract_name(entry.text)
            if name:
                return name
        return None

    # --- internals -------------------------------------------------------------------
    @staticmethod
    def _extract_name(text: str) -> str | None:
        """Pull a person's name out of ``text`` using the known phrasings."""
        clean = _collapse(text)
        if not clean:
            return None
        for pattern in _NAME_PATTERNS:
            match = pattern.search(clean)
            if not match:
                continue
            tail = match.group(1).strip()
            # Cut at the first sentence/clause boundary.
            tail = re.split(r"[,.;:!?–—()]|\band\b|\boch\b", tail, maxsplit=1)[0]
            tokens: list[str] = []
            for token in tail.split():
                token = token.strip("\"'“”‘’")
                if not token or not _NAME_TOKEN_RE.match(token):
                    break
                tokens.append(token)
                if len(tokens) >= _MAX_NAME_TOKENS:
                    break
            if tokens:
                return _capitalise_name(" ".join(tokens))
        return None

    def _enforce_cap(self) -> None:
        """Drop the oldest facts until at most :data:`MAX_FACTS` remain. Caller holds the lock."""
        overflow = len(self._facts) - MAX_FACTS
        if overflow > 0:
            dropped = self._facts[:overflow]
            self._facts = self._facts[overflow:]
            logger.warning(
                "Memory cap of %d reached; dropped %d oldest fact(s), first was: %s",
                MAX_FACTS, overflow, dropped[0].text if dropped else "",
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Memory(path={str(self._path)!r}, facts={len(self)})"
