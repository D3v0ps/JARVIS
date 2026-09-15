"""Deep thinking: one slow, careful answer instead of a fast conversational one.

Voice replies stream from a small model with reasoning switched off, because latency
is everything. When a question genuinely deserves thought, this tool hands it to the
same local daemon with reasoning switched on — and optionally to a bigger model named
by ``brain.deep_model`` — waits for the whole answer, throws the reasoning away and
speaks two sentences.

The :class:`~jarvis.brain.ollama_client.OllamaClient` is injected at wiring time with
:func:`set_deep_client`, so this module holds no import-time dependency on the brain
being built yet.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from jarvis.brain.sentences import clean_for_speech
from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["deep_think", "set_deep_client", "get_deep_client", "DEEP_TIMEOUT",
           "DEEP_SYSTEM_PROMPT"]

_log = get_logger("tools.think")

#: Seconds a deep answer may take. Reasoning models are slow and that is the point.
DEEP_TIMEOUT = 300

#: How many sentences of the answer are spoken.
MAX_SPOKEN_SENTENCES = 2

#: Characters of the answer kept for speech before it is trimmed.
_SPOKEN_ANSWER_CHARS = 360

#: The instruction that makes a reasoning model answer like a butler, not a report.
DEEP_SYSTEM_PROMPT = (
    "You are the deliberate reasoning half of JARVIS, a spoken assistant. Think the "
    "question through carefully and check your own reasoning before you commit to an "
    "answer. Then reply with the answer only, in at most two sentences of plain "
    "spoken English: no markdown, no bullet points, no headings, no code blocks, no "
    "emoji, and no description of how you reasoned. Write numbers so they read well "
    "aloud. Address the user as 'sir'."
)

#: The injected client, guarded because wiring and tool calls are on different threads.
_client: Any = None
_client_lock = threading.Lock()

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_THINK_LEFTOVER_RE = re.compile(r"<\s*/?\s*(?:think|thinking|reasoning)[^>]*>", re.IGNORECASE)


def set_deep_client(client: Any) -> None:
    """Give :func:`deep_think` the ``OllamaClient`` to reason with.

    Called once during wiring. Passing ``None`` unwires it again, which is what the
    shutdown path and the tests want.
    """
    global _client
    with _client_lock:
        _client = client
    _log.debug("Deep-think client %s", "wired" if client is not None else "cleared")


def get_deep_client() -> Any:
    """The currently wired client, or ``None``."""
    with _client_lock:
        return _client


def _deep_model(ctx: ToolContext) -> Optional[str]:
    """``brain.deep_model`` when the user configured a bigger model, else ``None``."""
    try:
        raw = ctx.config.get("brain.deep_model", None)
    except AttributeError:
        return None
    name = " ".join(str(raw or "").split())
    return name or None


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks and any stray reasoning tags, then tidy for speech."""
    spoken = clean_for_speech(text or "")
    spoken = _THINK_LEFTOVER_RE.sub(" ", spoken)
    return " ".join(spoken.split())


def _two_sentences(text: str) -> str:
    """The first two sentences, bounded so no answer ever monologues."""
    clean = _strip_reasoning(text)
    if not clean:
        return ""
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(clean) if part.strip()]
    spoken = " ".join(sentences[:MAX_SPOKEN_SENTENCES]) if sentences else clean
    if len(spoken) <= _SPOKEN_ANSWER_CHARS:
        return spoken
    cut = spoken.rfind(" ", 0, _SPOKEN_ANSWER_CHARS - 1)
    if cut < _SPOKEN_ANSWER_CHARS // 2:
        cut = _SPOKEN_ANSWER_CHARS - 1
    return spoken[:cut].rstrip(" ,;:-") + "…"


@tool(
    "deep_think",
    description=(
        "Think a hard question through with the local reasoning model and come back "
        "with a short considered answer. Use it for planning, trade-offs, tricky "
        "reasoning or maths — it is slow, so do not use it for small talk."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to reason about, stated in full.",
            }
        },
        "required": ["question"],
    },
    tier=Tier.ANNOUNCED,
    announce="Give me a moment, sir.",
)
def deep_think(ctx: ToolContext, args: dict) -> ToolResult:
    """Reason about a question properly, then answer in at most two sentences.

    One blocking call to the local daemon with reasoning enabled, using
    ``brain.deep_model`` when one is configured. The request timeout is widened to
    :data:`DEEP_TIMEOUT` for the duration of the call and put back afterwards, because
    a reasoning model happily spends minutes on a good question. The model's own
    reasoning block is stripped before anything is spoken; the full answer, reasoning
    and all, goes to the log.
    """
    question = " ".join(str(args.get("question") or "").split())
    if not question:
        return ToolResult.fail("What would you like me to think about, sir?")

    client = get_deep_client()
    if client is None:
        _log.warning("deep_think was called before a client was wired.")
        return ToolResult.fail(
            "My deeper reasoning isn't connected at the moment, sir.",
            "set_deep_client() has not been called with an OllamaClient.",
        )

    model = _deep_model(ctx)
    messages = [
        {"role": "system", "content": DEEP_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    previous_timeout = getattr(client, "timeout", None)
    try:
        if isinstance(previous_timeout, (int, float)) and previous_timeout < DEEP_TIMEOUT:
            client.timeout = DEEP_TIMEOUT
    except Exception:  # noqa: BLE001 - a read-only client is not worth failing over
        _log.debug("Could not widen the client timeout", exc_info=True)
        previous_timeout = None

    try:
        message = client.chat(messages, tools=None, model=model, think=True)
    except Exception as exc:  # noqa: BLE001 - chat() should not raise, but never trust it
        _log.error("Deep thinking failed for %r: %s", question, exc)
        _log.debug("deep_think traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't think that through just now, sir.", f"{type(exc).__name__}: {exc}"
        )
    finally:
        if previous_timeout is not None:
            try:
                client.timeout = previous_timeout
            except Exception:  # noqa: BLE001
                _log.debug("Could not restore the client timeout", exc_info=True)

    if not isinstance(message, dict):
        return ToolResult.fail(
            "I couldn't think that through just now, sir.",
            f"The client returned {type(message).__name__}, not a message.",
        )
    error = message.get("error")
    if error:
        _log.error("Deep thinking returned an error: %s", error)
        return ToolResult.fail("I couldn't think that through just now, sir.", str(error))

    raw = message.get("content")
    raw = raw if isinstance(raw, str) else ""
    answer = _two_sentences(raw)
    if not answer:
        return ToolResult.fail(
            "I thought about it and came back with nothing useful, sir.",
            f"Empty answer for {question!r}. Raw content: {raw!r}",
        )
    return ToolResult(
        ok=True,
        summary=answer,
        detail=f"question: {question}\nmodel: {model or 'default'}\nraw answer:\n{raw}",
        data={"question": question, "model": model, "answer": answer},
    )
