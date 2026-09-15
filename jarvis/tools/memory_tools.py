"""Memory tools: what JARVIS keeps about his user between sessions.

Three tools over :class:`jarvis.core.memory.Memory`, which already saves atomically on
every mutation — so a fact survives even if the machine dies a second later.

* ``remember`` stores one short fact.
* ``forget`` drops every fact matching a topic and reports how many went.
* ``recall`` reads back what is stored, at most three facts, in one sentence.

Pure standard library: this module imports anywhere.
"""

from __future__ import annotations

from typing import Any

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["remember", "forget", "recall", "MAX_SPOKEN_FACTS"]

_log = get_logger("tools.memory")

#: How many facts ``recall`` reads out before it starts counting the rest.
MAX_SPOKEN_FACTS = 3

#: A single fact is trimmed to this before it is spoken back.
_SPOKEN_FACT_CHARS = 120

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"]


def _spoken_number(value: int) -> str:
    """Render a small whole number as words so the count reads well aloud."""
    number = int(value)
    if number < 0 or number > 999:
        return str(number)
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"
    hundreds, rest = divmod(number, 100)
    head = f"{_UNITS[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_spoken_number(rest)}"


def _clean(value: Any) -> str:
    """Collapse an argument to a single trimmed line."""
    return " ".join(str(value or "").split())


def _trim(text: str, limit: int = _SPOKEN_FACT_CHARS) -> str:
    """Shorten one fact to something that fits in a spoken sentence."""
    clean = _clean(text)
    if len(clean) <= limit:
        return clean
    cut = clean.rfind(" ", 0, limit - 1)
    if cut < limit // 2:
        cut = limit - 1
    return clean[:cut].rstrip(" ,;:-") + "…"


def _join(items: list[str]) -> str:
    """Join spoken fragments with commas and a final "and"."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


@tool(
    "remember",
    description=(
        "Store one short fact about the user so it is available in future "
        "conversations, for example a name, a preference or a routine. Use it when "
        "the user asks to be remembered, not for passing chatter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "The fact to store, written as a short third-person statement, "
                    "for example 'the user prefers tea over coffee'."
                ),
            }
        },
        "required": ["fact"],
    },
    tier=Tier.SAFE,
)
def remember(ctx: ToolContext, args: dict) -> ToolResult:
    """Store one fact in long-term memory and confirm it in character.

    :class:`Memory` writes to disk inside ``remember()``, so nothing is lost if the
    session ends immediately afterwards. A fact that is already stored is refreshed
    rather than duplicated, and the confirmation is the same either way.
    """
    fact = _clean(args.get("fact"))
    if not fact:
        return ToolResult.fail("I need something to remember, sir.")
    try:
        entry = ctx.memory.remember(fact)
    except Exception as exc:  # noqa: BLE001 - a full disk must not kill the turn
        _log.error("Could not store the fact %r: %s", fact, exc)
        _log.debug("remember() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't write that to memory, sir.", f"{type(exc).__name__}: {exc}"
        )
    stored = getattr(entry, "text", fact) or fact
    return ToolResult(
        ok=True,
        summary="I'll remember that, sir.",
        detail=f"Stored: {stored}",
        data={"fact": stored, "count": len(ctx.memory.facts())},
    )


@tool(
    "forget",
    description=(
        "Delete every stored fact that mentions a topic, for example 'coffee' or "
        "'my birthday'. Use it when the user asks you to forget something."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The word or phrase the facts to delete contain.",
            }
        },
        "required": ["topic"],
    },
    tier=Tier.SAFE,
)
def forget(ctx: ToolContext, args: dict) -> ToolResult:
    """Drop every fact matching a topic and say how many were dropped."""
    topic = _clean(args.get("topic"))
    if not topic:
        return ToolResult.fail("I need to know what to forget, sir.")
    try:
        removed = int(ctx.memory.forget(topic))
    except Exception as exc:  # noqa: BLE001 - never let a storage fault escape
        _log.error("Could not forget %r: %s", topic, exc)
        _log.debug("forget() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't change what I have stored, sir.", f"{type(exc).__name__}: {exc}"
        )
    if removed == 0:
        return ToolResult(
            ok=True,
            summary=f"I had nothing stored about {topic}, sir.",
            detail=f"No stored fact matched '{topic}'.",
            data={"topic": topic, "removed": 0},
        )
    if removed == 1:
        summary = f"Forgotten, sir, one fact about {topic} is gone."
    else:
        summary = f"Forgotten, sir, {_spoken_number(removed)} facts about {topic} are gone."
    return ToolResult(
        ok=True,
        summary=summary,
        detail=f"Removed {removed} fact(s) matching '{topic}'.",
        data={"topic": topic, "removed": removed},
    )


@tool(
    "recall",
    description=(
        "Read back what is stored in long-term memory about the user. Give a topic "
        "to filter, or leave it out to hear the most recent facts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Optional word or phrase the facts should contain.",
            }
        },
        "required": [],
    },
    tier=Tier.SAFE,
)
def recall(ctx: ToolContext, args: dict) -> ToolResult:
    """Say what is remembered, newest first, at most three facts in one sentence.

    With a ``topic`` only matching facts are considered; matching is the same
    case-insensitive substring test :meth:`Memory.forget` uses, over both the fact
    text and its topic. Anything beyond the third fact is counted, not read out.
    """
    topic = _clean(args.get("topic"))
    try:
        entries = list(ctx.memory.facts())
    except Exception as exc:  # noqa: BLE001 - a corrupt store must not kill the turn
        _log.error("Could not read memory: %s", exc)
        _log.debug("recall() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't read my memory just now, sir.", f"{type(exc).__name__}: {exc}"
        )

    needle = topic.lower()
    if needle:
        entries = [
            entry
            for entry in entries
            if needle in (entry.text or "").lower() or needle in (entry.topic or "").lower()
        ]
    if not entries:
        summary = (
            f"I have nothing stored about {topic}, sir."
            if topic
            else "I have nothing stored about you yet, sir."
        )
        return ToolResult(
            ok=True, summary=summary, detail="Memory is empty for this query.",
            data={"topic": topic, "facts": [], "total": 0},
        )

    newest_first = list(reversed(entries))
    spoken = [_trim(entry.text) for entry in newest_first[:MAX_SPOKEN_FACTS]]
    extra = len(newest_first) - len(spoken)
    sentence = _join(spoken)
    lead = f"About {topic}, sir, I have" if topic else "I remember that"
    summary = f"{lead} {sentence}"
    if extra > 0:
        summary += f", and {_spoken_number(extra)} more"
    summary = summary.rstrip(".") + "."
    return ToolResult(
        ok=True,
        summary=summary,
        detail="\n".join(f"- {entry.text}" for entry in newest_first),
        data={
            "topic": topic,
            "facts": [entry.text for entry in newest_first],
            "total": len(newest_first),
        },
    )
