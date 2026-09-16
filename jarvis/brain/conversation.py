"""Rolling conversation history and system-prompt assembly for JARVIS.

The system message is rebuilt on every call so the clock in it is never stale. It is
made of four parts:

1. the character prompt from ``prompts/jarvis_system.md``;
2. the memory block from :meth:`jarvis.core.memory.Memory.as_prompt_block`;
3. one line with the current date, weekday and time;
4. one line pinning the reply language, but only when ``language:`` is forced in the
   configuration (``auto`` lets the model follow whatever the user speaks).

Trimming is the other job of this module, and the fiddly one. Ollama rejects a
``role: "tool"`` message whose assistant ``tool_calls`` message is missing, so the
history is only ever cut at exchange boundaries — the start of a user message — and a
final sanity pass drops any pair that still ended up broken.

Pure standard library plus the JARVIS core: this module imports cleanly on Linux.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jarvis.config import project_root
from jarvis.core.logging import get_logger
from jarvis.core.memory import Memory

__all__ = ["Conversation", "FALLBACK_SYSTEM_PROMPT", "DEFAULT_HISTORY_TURNS"]

#: Default number of user/assistant exchanges kept in the rolling window.
DEFAULT_HISTORY_TURNS = 12

#: Used when ``prompts/jarvis_system.md`` cannot be read. Same character, fewer words.
FALLBACK_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., the resident intelligence of this computer and of the man "
    "who runs it. Address the user as \"sir\". You are calm, dry, British and "
    "unshakeable. Everything you say is spoken aloud, so answer in one or two plain "
    "sentences with no lists, no markdown and no emoji. Use a tool whenever an action "
    "or a live fact is needed, never claim to have done something you did not do, and "
    "refuse anything that would damage the machine or expose credentials. If the user "
    "speaks Swedish, reply in Swedish with exactly the same manner."
)

#: Weekday and month names spelled out, so the line never depends on the system locale.
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_LANGUAGE_LINES: dict[str, str] = {
    "en": "Always answer in English, whatever language the user speaks.",
    "sv": "Svara alltid på svenska, oavsett vilket språk användaren talar.",
}

_ROLES_WITH_TEXT = ("user", "assistant")


def _now_line(now: Optional[datetime] = None) -> str:
    """Return the current-time line, e.g. ``It is Tuesday 15 September 2026, 21:47.``"""
    moment = now or datetime.now()
    return (
        f"It is {_WEEKDAYS[moment.weekday()]} {moment.day} "
        f"{_MONTHS[moment.month - 1]} {moment.year}, {moment:%H:%M}."
    )


def _has_tool_calls(message: dict) -> bool:
    """True when ``message`` is an assistant turn that asked for tools."""
    if message.get("role") != "assistant":
        return False
    calls = message.get("tool_calls")
    return isinstance(calls, list) and bool(calls)


def _normalize_tool_calls(tool_calls: Any) -> list[dict]:
    """Render tool calls in the wire shape Ollama expects when they are sent back.

    Accepts both the flat ``{"name": ..., "arguments": {...}}`` form produced by
    :class:`~jarvis.brain.ollama_client.OllamaClient` and the nested
    ``{"function": {...}}`` form the server itself uses. Anything unusable is dropped
    rather than sent, because a malformed tool call breaks the whole request.
    """
    if not isinstance(tool_calls, list):
        return []
    rendered: list[dict] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        inner = item.get("function")
        source = inner if isinstance(inner, dict) else item
        name = str(source.get("name") or "").strip()
        if not name:
            continue
        arguments = source.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else {"value": arguments}
        rendered.append({"function": {"name": name, "arguments": copy.deepcopy(arguments)}})
    return rendered


class Conversation:
    """Rolling history plus system-prompt assembly for one assistant session."""

    def __init__(
        self,
        system_prompt_path: str | Path,
        memory: Memory,
        *,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        language: str | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._log: logging.Logger = logger or get_logger("conversation")
        self._prompt_path: Path = self._resolve(system_prompt_path)
        self.memory = memory
        self.history_turns = max(1, int(history_turns or DEFAULT_HISTORY_TURNS))
        self.language: str | None = str(language).strip().lower() if language else None
        self._history: list[dict] = []
        self._prompt_text: str | None = None
        self._prompt_mtime: float | None = None
        self._missing_prompt_logged = False

    # --- system prompt ------------------------------------------------------------
    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """Anchor a relative prompt path at the project root, not the working directory."""
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (project_root() / candidate).resolve()

    @property
    def prompt_path(self) -> Path:
        """The file the character prompt is read from."""
        return self._prompt_path

    def _character_prompt(self) -> str:
        """Read the prompt file, cached by modification time; fall back when unreadable."""
        try:
            mtime = self._prompt_path.stat().st_mtime
        except OSError as exc:
            if not self._missing_prompt_logged:
                self._log.error(
                    "System prompt %s could not be read (%s); using the built-in fallback.",
                    self._prompt_path, exc,
                )
                self._missing_prompt_logged = True
            return FALLBACK_SYSTEM_PROMPT

        if self._prompt_text is not None and self._prompt_mtime == mtime:
            return self._prompt_text

        try:
            text = self._prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            self._log.error(
                "System prompt %s could not be read (%s); using the built-in fallback.",
                self._prompt_path, exc,
            )
            self._missing_prompt_logged = True
            return FALLBACK_SYSTEM_PROMPT

        if not text:
            self._log.error(
                "System prompt %s is empty; using the built-in fallback.", self._prompt_path
            )
            return FALLBACK_SYSTEM_PROMPT

        self._prompt_text = text
        self._prompt_mtime = mtime
        self._missing_prompt_logged = False
        self._log.debug("Loaded the system prompt from %s (%d characters).", self._prompt_path, len(text))
        return text

    def _memory_block(self) -> str:
        """The memory block, or ``""`` when nothing is remembered or memory is broken."""
        try:
            block = self.memory.as_prompt_block()
        except Exception as exc:  # noqa: BLE001 - memory must never break a turn
            self._log.warning("Could not read memory for the system prompt: %s", exc)
            return ""
        return str(block or "").strip()

    def _language_line(self) -> str:
        """The forced-language instruction, or ``""`` when the language is automatic."""
        if not self.language:
            return ""
        line = _LANGUAGE_LINES.get(self.language)
        if line is None:
            self._log.debug("No canned language line for %r; not forcing a language.", self.language)
            return ""
        return line

    def system_message(self) -> dict:
        """Build the system message. Deliberately free of anything that changes often.

        The clock used to live here, which meant the prompt prefix changed at every
        minute boundary and Ollama threw away its cached evaluation of two thousand
        tokens of persona and tool schemas - on every turn that happened to cross a
        minute. It now rides on the newest user message instead, where it costs
        nothing and is just as current.
        """
        parts = [self._character_prompt()]
        memory_block = self._memory_block()
        if memory_block:
            parts.append(memory_block)
        language_line = self._language_line()
        if language_line:
            parts.append(language_line)
        return {"role": "system", "content": "\n\n".join(part for part in parts if part)}

    # --- history ------------------------------------------------------------------
    def messages(self) -> list[dict]:
        """Return ``[system] + trimmed history``, ready to POST to Ollama.

        The clock is stamped onto the last user message, so everything before it is
        byte-identical from turn to turn and Ollama's prompt cache survives.
        """
        self.trim()
        history = [copy.deepcopy(item) for item in self._history]
        for item in reversed(history):
            if item.get("role") == "user":
                item["content"] = f"{item.get('content', '')}\n\n({_now_line()})"
                break
        return [self.system_message()] + history

    def add_user(self, text: str) -> None:
        """Append a user turn and close the previous exchange by trimming."""
        content = str(text or "").strip()
        if not content:
            self._log.debug("Ignoring an empty user turn.")
            return
        self._history.append({"role": "user", "content": content})
        self.trim()

    def add_assistant(self, text: str, tool_calls: list | None = None) -> None:
        """Append an assistant turn, optionally one that asked for tools.

        A turn with tool calls is never trimmed away on its own: trimming waits until
        the exchange is finished, because its ``role: "tool"`` results are still to
        come.
        """
        content = str(text or "").strip()
        rendered = _normalize_tool_calls(tool_calls)
        if not content and not rendered:
            self._log.debug("Ignoring an empty assistant turn.")
            return
        message: dict = {"role": "assistant", "content": content}
        if rendered:
            message["tool_calls"] = rendered
            self._history.append(message)
            return
        self._history.append(message)
        self.trim()

    def add_tool_result(self, name: str, content: str) -> None:
        """Append one ``role: "tool"`` result for the assistant turn just recorded."""
        tool_name = str(name or "tool").strip() or "tool"
        body = str(content or "").strip()
        # Walk back over the results already recorded for this assistant turn.
        index = len(self._history) - 1
        while index >= 0 and self._history[index].get("role") == "tool":
            index -= 1
        if index < 0 or not _has_tool_calls(self._history[index]):
            # Refuse to create the exact shape Ollama rejects: a result with no request.
            self._log.warning(
                "Dropping the result of %s: no assistant tool_calls message precedes it.", tool_name
            )
            return
        self._history.append({"role": "tool", "name": tool_name, "tool_name": tool_name, "content": body})

    def clear(self) -> None:
        """Forget the whole exchange history (the system message is rebuilt anyway)."""
        if self._history:
            self._log.debug("Cleared %d history message(s).", len(self._history))
        self._history = []

    @property
    def turns(self) -> int:
        """How many user turns are currently in the window."""
        return sum(1 for item in self._history if item.get("role") == "user")

    @property
    def history(self) -> list[dict]:
        """A copy of the current history, without the system message."""
        return [copy.deepcopy(item) for item in self._history]

    # --- trimming -----------------------------------------------------------------
    def trim(self) -> None:
        """Keep the last ``history_turns`` exchanges, cutting only at user turns.

        An exchange starts at a user message and runs up to the next one, so an
        assistant ``tool_calls`` message and its ``role: "tool"`` results are always
        kept or dropped together. A final pass removes any pair that is broken anyway,
        because a dangling tool result makes Ollama reject the whole request.
        """
        starts = [index for index, item in enumerate(self._history) if item.get("role") == "user"]
        if len(starts) > self.history_turns:
            cut = starts[-self.history_turns]
            dropped = self._history[:cut]
            self._history = self._history[cut:]
            self._log.debug(
                "Trimmed %d message(s); %d exchange(s) remain.", len(dropped), self.turns
            )
        self._history = self._sanitize(self._history)

    def _sanitize(self, messages: list[dict]) -> list[dict]:
        """Drop orphaned tool results and tool calls that never got their results."""
        kept: list[dict] = []
        index = 0
        total = len(messages)

        while index < total:
            message = messages[index]
            role = message.get("role")

            if _has_tool_calls(message):
                end = index + 1
                while end < total and messages[end].get("role") == "tool":
                    end += 1
                if end == index + 1:
                    self._log.warning("Dropping an assistant tool call that never got its results.")
                else:
                    kept.extend(messages[index:end])
                index = end
                continue

            if role == "tool":
                self._log.warning("Dropping an orphaned tool result: %s", message.get("name", "?"))
            elif role in _ROLES_WITH_TEXT:
                kept.append(message)
            else:
                self._log.debug("Dropping a history message with role %r.", role)
            index += 1

        return kept

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"Conversation(turns={self.turns}, messages={len(self._history)}, "
            f"window={self.history_turns}, language={self.language!r})"
        )
