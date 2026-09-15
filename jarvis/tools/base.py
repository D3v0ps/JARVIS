"""Core dataclasses every JARVIS tool is built from.

This module is pure standard library on purpose: it is imported by the brain, the
dispatcher and every tool module, so it must import cleanly on Linux CI as well as
on the Windows target.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles at runtime
    from jarvis.config import Config
    from jarvis.core.memory import Memory
    from jarvis.core.scheduler import Scheduler
    from jarvis.core.state import StateBus

__all__ = [
    "Tier",
    "ToolResult",
    "ToolContext",
    "ToolSpec",
    "SUMMARY_MAX_CHARS",
    "ANNOUNCE_VALUE_MAX_CHARS",
]

_log = logging.getLogger("jarvis.tools.base")

#: A spoken summary longer than this is truncated; the remainder moves to ``detail``.
SUMMARY_MAX_CHARS = 400

#: Arguments interpolated into an announcement are shortened to this many characters.
ANNOUNCE_VALUE_MAX_CHARS = 120

_WHITESPACE_RE = re.compile(r"\s+")


class Tier(str, Enum):
    """How much ceremony a tool needs before it may run.

    ``SAFE``       run silently.
    ``ANNOUNCED``  say what is about to happen, then run.
    ``GUARDED``    say what is about to happen and wait for a spoken confirmation.
    """

    SAFE = "safe"
    ANNOUNCED = "announced"
    GUARDED = "guarded"


def _shorten_summary(summary: str) -> tuple[str, str]:
    """Split an over-long summary into ``(spoken_head, overflow_tail)``.

    The cut prefers the last whitespace inside the budget so the spoken part never
    ends mid-word. Returns an empty tail when the summary already fits.
    """
    text = summary.strip()
    if len(text) <= SUMMARY_MAX_CHARS:
        return text, ""
    budget = SUMMARY_MAX_CHARS - 1  # leave room for the ellipsis
    cut = text.rfind(" ", 0, budget)
    if cut < budget // 2:
        cut = budget
    head = text[:cut].rstrip(" ,;:-")
    tail = text[cut:].strip()
    return head + "…", tail


@dataclass
class ToolResult:
    """The outcome of one tool call.

    ``summary`` is the only field the model sees and speaks, so it must be one short
    sentence. Anything longer than :data:`SUMMARY_MAX_CHARS` is truncated on
    construction and the remainder is moved into ``detail`` (log only, never spoken).
    """

    ok: bool
    summary: str
    detail: str = ""
    data: dict | None = None
    refused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            self.summary = "" if self.summary is None else str(self.summary)
        if not isinstance(self.detail, str):
            self.detail = "" if self.detail is None else str(self.detail)
        self.summary = _WHITESPACE_RE.sub(" ", self.summary).strip()
        head, tail = _shorten_summary(self.summary)
        if tail:
            self.summary = head
            self.detail = f"{tail}\n\n{self.detail}".strip() if self.detail else tail

    @classmethod
    def fail(cls, summary: str, detail: str = "") -> "ToolResult":
        """A tool that tried and could not deliver."""
        return cls(ok=False, summary=summary, detail=detail)

    @classmethod
    def refuse(cls, summary: str, detail: str = "") -> "ToolResult":
        """A tool call that was blocked on purpose (safety, policy, cancellation)."""
        return cls(ok=False, summary=summary, detail=detail, refused=True)


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach into.

    Tools receive this instead of global state, which keeps them testable with a
    fake context and keeps the wiring in one place.
    """

    config: "Config"
    memory: "Memory"
    logger: logging.Logger
    speak: Callable[[str], None]
    confirm: Callable[[str], bool]
    notify: Callable[[str], None]
    scheduler: "Scheduler"
    state: "StateBus"


class _AnnounceArgs(dict):
    """``str.format_map`` mapping that tolerates missing and over-long values.

    A lookup miss is recorded in ``missing`` so the caller can decide to use the
    generic announcement rather than speak a sentence with a hole in it.
    """

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self.missing: list[str] = []

    def __missing__(self, key: str) -> str:
        self.missing.append(key)
        return ""

    def __getitem__(self, key: str) -> str:
        try:
            value = super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            self.missing.append(key)
            return ""
        return _format_value(value)


def _format_value(value: Any) -> str:
    """Render one argument for speech: single line, bounded length."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > ANNOUNCE_VALUE_MAX_CHARS:
        text = text[: ANNOUNCE_VALUE_MAX_CHARS - 1].rstrip() + "…"
    return text


@dataclass
class ToolSpec:
    """A registered tool: its schema, its safety tier and its implementation."""

    name: str
    description: str
    parameters: dict
    tier: Tier
    func: Callable[[ToolContext, dict], ToolResult]
    announce: str | None = None

    def to_ollama(self) -> dict:
        """The OpenAI-style function description Ollama expects in ``tools``.

        The parameter schema is deep-copied so a caller cannot mutate the registry.
        """
        parameters: Any = self.parameters if isinstance(self.parameters, dict) else {}
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(parameters),
            },
        }

    def render_announcement(self, args: dict) -> str:
        """Format ``announce`` with the call arguments, for speaking out loud.

        ``announce`` is a ``str.format`` template such as
        ``"About to run PowerShell: {command}"``. Values are collapsed to one line
        and shortened to :data:`ANNOUNCE_VALUE_MAX_CHARS`. Nothing raises: a missing
        or malformed template, a placeholder with no matching argument, or an
        argument that is empty all fall back to ``"About to run {name}."`` so the
        spoken announcement is never a sentence with a hole in it.
        """
        fallback = f"About to run {self.name}."
        template = self.announce
        if not template or not str(template).strip():
            return fallback
        mapping = _AnnounceArgs(args if isinstance(args, dict) else {})
        try:
            rendered = str(template).format_map(mapping)
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            _log.warning(
                "Malformed announcement template for tool %s (%s); using the fallback",
                self.name,
                exc,
            )
            return fallback
        if mapping.missing:
            _log.debug(
                "Announcement for tool %s lacks argument(s) %s; using the fallback",
                self.name,
                ", ".join(sorted(set(mapping.missing))),
            )
            return fallback
        rendered = _WHITESPACE_RE.sub(" ", rendered).strip()
        rendered = re.sub(r"\s+([,.;:!?])", r"\1", rendered)
        # An announcement that collapsed to punctuation only is useless to speak.
        if not re.search(r"\w", rendered):
            return fallback
        return rendered
