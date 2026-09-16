"""One voice turn: stream tokens, speak sentences, run tools in between.

:class:`Brain` is the piece that makes JARVIS answer before he has finished thinking.
Tokens arrive from :class:`~jarvis.brain.ollama_client.OllamaClient`, go through
:class:`~jarvis.brain.sentences.SentenceSplitter`, and every finished sentence is handed
straight to ``on_sentence`` — in practice the TTS queue — so speech starts on sentence
one while the model is still producing sentence two.

When the model asks for tools, the splitter is flushed (anything already said stays
said), each call goes through the dispatcher, the request and its results are appended
to the conversation, and the model gets another round. After ``max_tool_rounds`` the
loop stops and JARVIS says so honestly rather than spinning.

Nothing here touches audio hardware or Windows-only APIs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from jarvis.brain.conversation import Conversation
from jarvis.brain.ollama_client import OllamaClient
from jarvis.brain.sentences import SentenceSplitter, clean_for_speech
from jarvis.core.logging import get_logger, log_transcript
from jarvis.core.state import AssistantState

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from jarvis.core.latency import LatencyTracker
    from jarvis.tools.dispatcher import Dispatcher

__all__ = ["TurnResult", "Brain", "FIRST_TOKEN_MARK"]

#: Latency mark stamped the moment the first token of a turn arrives.
FIRST_TOKEN_MARK = "first_token"

#: Spoken when the model cannot be reached or breaks mid-answer.
MODEL_ERROR_SENTENCE = "The language model isn't responding, sir."
#: Spoken when the model keeps asking for tools past the configured limit.
TOOL_LIMIT_SENTENCE = "I've run as many checks as I usefully can, sir, so I'll stop there."
#: Spoken when a turn produced neither words nor a usable tool summary.
EMPTY_TURN_SENTENCE = "I'm afraid I have nothing useful to add, sir."


#: A local model will sometimes describe the action instead of performing it -
#: "Opening Steam, sir" with no open_app call, or "I don't have access to real-time
#: weather" while the weather tool sits right there in its list. The user is left
#: looking at a screen where nothing happened, which the system prompt explicitly
#: forbids, so the turn gets one corrective round before it is allowed to end.
ACTION_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    # English: promising, or claiming to be in the middle of it
    re.compile(r"\b(i'?ll|i will|let me|i'?m going to|i am going to|allow me to)\s+"
               r"(open|launch|start|check|look|search|set|take|lock|run|play|pause|close|"
               r"find|delete|remove|move|rename|remember|forget|type|shut)", re.I),
    re.compile(r"\b(opening|launching|starting|checking|searching|setting|taking|locking|"
               r"running|playing|closing|looking up)\b", re.I),
    # Plain past tense too: "Yes, sir, I checked the weather" is the same lie.
    re.compile(r"\b(i'?ve|i have|i just|i)\s+"
               r"(opened|launched|started|checked|set|taken|took|locked|ran|run|searched|"
               r"found|deleted|removed|moved|renamed)\b", re.I),
    # Swedish
    re.compile(r"\b(jag\s+(ska|kommer att|tänker)|låt mig)\s+"
               r"(öppna|starta|kolla|titta|söka|ställa|ta|låsa|köra|spela|stänga|leta)", re.I),
    re.compile(r"\b(öppnar|startar|kollar|söker|ställer in|tar|låser|kör|spelar|stänger)\b", re.I),
    re.compile(r"\bjag har\s+(öppnat|startat|kollat|ställt|tagit|låst|kört|sökt)", re.I),
]

#: Worse than promising: denying a capability that is in the tool list.
FALSE_INCAPACITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(don'?t|do not|cannot|can'?t|unable to)\b[^.]{0,40}"
               r"\b(access|check|open|retrieve|fetch|see|know)\b", re.I),
    re.compile(r"\bno access to\b", re.I),
    re.compile(r"\b(kan inte|har inte tillgång till|saknar tillgång)\b", re.I),
    re.compile(r"\breal[- ]time\b", re.I),
]

#: Sent for the corrective round only; never stored in the conversation.
CORRECTION = (
    "You just told the user what you would do, but you called no tool, so nothing "
    "actually happened and they are looking at a screen where nothing changed. "
    "Call the tool that performs it now. If no tool fits, say so plainly instead."
)


def claims_without_acting(reply: str) -> bool:
    """True when the reply describes an action or denies a capability it has."""
    text = str(reply or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in ACTION_CLAIM_PATTERNS) or any(
        pattern.search(text) for pattern in FALSE_INCAPACITY_PATTERNS
    )


@dataclass
class TurnResult:
    """What one voice turn produced."""

    reply: str
    tool_calls: list[str] = field(default_factory=list)
    cancelled: bool = False
    error: str = ""


class Brain:
    """One voice turn: stream -> sentences -> TTS, executing tools in between."""

    def __init__(
        self,
        client: OllamaClient,
        conversation: Conversation,
        dispatcher: "Dispatcher",
        *,
        max_tool_rounds: int = 4,
        logger: Optional[logging.Logger] = None,
        latency: Optional["LatencyTracker"] = None,
    ) -> None:
        self.client = client
        self.conversation = conversation
        self.dispatcher = dispatcher
        self.max_tool_rounds = max(0, int(max_tool_rounds))
        self.latency = latency
        self._log: logging.Logger = logger or get_logger("brain")

    # --- public API ----------------------------------------------------------------
    def turn(
        self,
        user_text: str,
        on_sentence: Callable[[str], None],
        should_stop: Callable[[], bool] = lambda: False,
    ) -> TurnResult:
        """Run one full turn and return what was said, asked for and interrupted."""
        text = str(user_text or "").strip()
        self.conversation.add_user(text)
        log_transcript("user", text)
        self._set_thinking()

        spoken: list[str] = []
        tool_names: list[str] = []
        splitter = SentenceSplitter()
        last_summary = ""
        tool_rounds = 0
        cancelled = False
        error = ""

        while True:
            if self._stopped(should_stop):
                cancelled = True
                break

            mark = len(spoken)
            round_text, pending_calls, round_error, cancelled = self._stream_round(
                splitter, on_sentence, spoken, should_stop
            )
            if round_error:
                error = round_error
                break
            if cancelled:
                self._remember(spoken[mark:], round_text)
                break

            if not pending_calls:
                self._remember(spoken[mark:], round_text)
                break

            if tool_rounds >= self.max_tool_rounds:
                self._log.warning(
                    "The model still wanted tools after %d round(s); answering without them.",
                    tool_rounds,
                )
                self._emit(TOOL_LIMIT_SENTENCE, on_sentence, spoken)
                self._remember(spoken[mark:], round_text)
                break

            tool_rounds += 1
            self.conversation.add_assistant(clean_for_speech(round_text), tool_calls=pending_calls)
            last_summary = self._run_tools(pending_calls, tool_names) or last_summary
            self._set_thinking()

        if error:
            self._emit(MODEL_ERROR_SENTENCE, on_sentence, spoken)
            reply = " ".join(spoken).strip()
            self.conversation.add_assistant(MODEL_ERROR_SENTENCE)
            log_transcript("jarvis", reply)
            return TurnResult(reply=reply, tool_calls=tool_names, cancelled=False, error=error)

        if not cancelled and not spoken:
            self._emit(self._silent_turn_sentence(last_summary), on_sentence, spoken)
            self._remember(spoken, "")

        # One corrective round when the model talked about acting but never acted.
        if (
            not cancelled
            and not error
            and not tool_names
            and self._tools_payload()
            and claims_without_acting(" ".join(spoken))
        ):
            self._log.warning(
                "The model described an action without calling a tool; asking again."
            )
            self._correct(on_sentence, spoken, tool_names, should_stop)

        reply = " ".join(spoken).strip()
        log_transcript("jarvis", reply)
        if cancelled:
            self._log.info("Turn cancelled after %d character(s) of speech.", len(reply))
        return TurnResult(reply=reply, tool_calls=tool_names, cancelled=cancelled, error="")

    def _correct(
        self,
        on_sentence: Callable[[str], None],
        spoken: list[str],
        tool_names: list[str],
        should_stop: Callable[[], bool],
    ) -> None:
        """Give the model one more chance to actually do what it said it would.

        The nudge is sent for this round only and never stored, so the conversation
        the user sees keeps no trace of the model being told off.
        """
        messages = self.conversation.messages() + [{"role": "system", "content": CORRECTION}]
        calls: list[dict] = []
        try:
            for delta in self.client.chat_stream(messages, self._tools_payload()):
                if self._stopped(should_stop):
                    return
                if delta.kind == "tool_calls" and delta.tool_calls:
                    calls.extend(delta.tool_calls)
                elif delta.kind == "error":
                    self._log.warning("The corrective round failed: %s", delta.error)
                    return
        except Exception as exc:  # noqa: BLE001 - a failed correction must not lose the turn
            self._log.warning("The corrective round failed: %s", exc)
            return

        if not calls:
            self._log.warning("The model still would not call a tool.")
            return

        self.conversation.add_assistant("", tool_calls=calls)
        summary = self._run_tools(calls, tool_names)
        self._set_thinking()

        # Report what actually happened, so the user hears the truth rather than
        # the promise.
        splitter = SentenceSplitter()
        round_text, pending, round_error, cancelled = self._stream_round(
            splitter, on_sentence, spoken, should_stop
        )
        if round_error or cancelled:
            if summary:
                self._emit(summary, on_sentence, spoken)
            return
        if not round_text.strip() and summary:
            self._emit(summary, on_sentence, spoken)
        self._remember(spoken, round_text)

    def say_directly(self, text: str, on_sentence: Callable[[str], None]) -> None:
        """Speak ``text`` without involving the model, and record it as an assistant turn."""
        # Cleaning before splitting matters: markdown glued to a full stop
        # ("**Done, sir.** Next.") would otherwise hide the sentence boundary.
        raw = clean_for_speech(str(text or ""))
        if not raw:
            return
        spoken: list[str] = []
        splitter = SentenceSplitter()
        for sentence in splitter.feed(raw + "\n"):
            self._emit(sentence, on_sentence, spoken)
        trailing = splitter.flush()
        if trailing:
            self._emit(trailing, on_sentence, spoken)
        said = " ".join(spoken).strip()
        if said:
            self.conversation.add_assistant(said)
            log_transcript("jarvis", said)

    # --- one streamed round ---------------------------------------------------------
    def _stream_round(
        self,
        splitter: SentenceSplitter,
        on_sentence: Callable[[str], None],
        spoken: list[str],
        should_stop: Callable[[], bool],
    ) -> tuple[str, list[dict], str, bool]:
        """Consume one ``chat_stream`` call.

        Returns ``(raw_text, pending_tool_calls, error, cancelled)``. Sentences are
        emitted as they complete, which is the whole point of streaming.
        """
        splitter.reset()
        chunks: list[str] = []
        pending: list[dict] = []
        error = ""
        cancelled = False
        first_token = True

        messages = self.conversation.messages()
        tools = self._tools_payload()
        stream = self.client.chat_stream(messages, tools)
        try:
            for delta in stream:
                if self._stopped(should_stop):
                    cancelled = True
                    break
                kind = getattr(delta, "kind", "")
                if kind == "token":
                    if first_token:
                        first_token = False
                        self._mark_first_token()
                    fragment = getattr(delta, "text", "") or ""
                    if not fragment:
                        continue
                    chunks.append(fragment)
                    for sentence in splitter.feed(fragment):
                        self._emit(sentence, on_sentence, spoken)
                elif kind == "tool_calls":
                    # Say whatever the model managed before asking for tools.
                    self._flush(splitter, on_sentence, spoken)
                    calls = getattr(delta, "tool_calls", None)
                    if isinstance(calls, list):
                        pending.extend(call for call in calls if isinstance(call, dict))
                elif kind == "error":
                    error = str(getattr(delta, "error", "") or "The local model failed.")
                    self._log.error("Streaming failed: %s", error)
                    break
                elif kind == "done":
                    break
        finally:
            closer = getattr(stream, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing must never break the turn
                    self._log.debug("Could not close the model stream cleanly.", exc_info=True)

        if not error:
            self._flush(splitter, on_sentence, spoken)
        else:
            splitter.reset()
        return "".join(chunks), pending, error, cancelled

    # --- tools ----------------------------------------------------------------------
    def _run_tools(self, calls: list[dict], tool_names: list[str]) -> str:
        """Execute every requested call and append one ``role: "tool"`` message each.

        Returns the summary of the last successful call, used when the model answers
        with tools only and never says a word.
        """
        last_summary = ""
        for call in calls:
            name = str(call.get("name") or "").strip() or "unknown"
            args = call.get("arguments")
            if not isinstance(args, dict):
                args = {}
            tool_names.append(name)
            try:
                result = self.dispatcher.execute(name, args)
            except Exception as exc:  # noqa: BLE001 - a broken tool must not end the turn
                self._log.error("Dispatcher raised while running %s: %s", name, exc, exc_info=True)
                self.conversation.add_tool_result(name, f"[failed] The {name} tool could not run.")
                continue
            summary = str(getattr(result, "summary", "") or "").strip()
            flag = self._status_flag(result)
            self.conversation.add_tool_result(name, f"[{flag}] {summary}".strip())
            if flag == "ok" and summary:
                last_summary = summary
            elif not last_summary and summary:
                last_summary = summary
        return last_summary

    @staticmethod
    def _status_flag(result: Any) -> str:
        """``ok`` / ``refused`` / ``failed`` — the model must know which it got."""
        if bool(getattr(result, "refused", False)):
            return "refused"
        return "ok" if bool(getattr(result, "ok", False)) else "failed"

    def _tools_payload(self) -> list[dict] | None:
        """The tool schemas for this turn, or ``None`` when the dispatcher has none."""
        getter = getattr(self.dispatcher, "tools_payload", None)
        if not callable(getter):
            return None
        try:
            payload = getter()
        except Exception as exc:  # noqa: BLE001 - never lose a turn over the tool list
            self._log.error("Could not build the tool payload: %s", exc, exc_info=True)
            return None
        return payload if isinstance(payload, list) and payload else None

    # --- speaking -------------------------------------------------------------------
    def _emit(self, raw: str, on_sentence: Callable[[str], None], spoken: list[str]) -> None:
        """Clean one sentence and hand it to the speaker, ignoring empties."""
        sentence = clean_for_speech(raw)
        if not sentence:
            return
        spoken.append(sentence)
        try:
            on_sentence(sentence)
        except Exception as exc:  # noqa: BLE001 - a failing speaker must not end the turn
            self._log.error("Could not speak a sentence: %s", exc, exc_info=True)

    def _flush(
        self, splitter: SentenceSplitter, on_sentence: Callable[[str], None], spoken: list[str]
    ) -> None:
        """Speak the trailing fragment left in the splitter, if there is one."""
        trailing = splitter.flush()
        if trailing:
            self._emit(trailing, on_sentence, spoken)

    @staticmethod
    def _silent_turn_sentence(last_summary: str) -> str:
        """A turn of pure tool calls still has to say something out loud."""
        summary = str(last_summary or "").strip()
        if not summary:
            return EMPTY_TURN_SENTENCE
        if summary[-1] not in ".!?…":
            summary += "."
        return summary

    # --- bookkeeping ----------------------------------------------------------------
    def _remember(self, round_spoken: list[str], raw_text: str) -> None:
        """Record this round's words in the history (also after an interruption)."""
        said = " ".join(round_spoken).strip() or clean_for_speech(raw_text)
        if said:
            self.conversation.add_assistant(said)

    def _stopped(self, should_stop: Callable[[], bool]) -> bool:
        """Evaluate the caller's stop flag without ever raising."""
        try:
            return bool(should_stop())
        except Exception as exc:  # noqa: BLE001 - a broken predicate means "carry on"
            self._log.error("should_stop() failed: %s", exc, exc_info=True)
            return False

    def _mark_first_token(self) -> None:
        """Stamp the first-token latency mark when a tracker was wired in."""
        if self.latency is None:
            return
        try:
            self.latency.mark(FIRST_TOKEN_MARK)
        except Exception as exc:  # noqa: BLE001 - measuring must never break speaking
            self._log.debug("Could not record the first-token latency: %s", exc)

    def _set_thinking(self) -> None:
        """Move the state bus to THINKING through the dispatcher's context, if there is one."""
        context = getattr(self.dispatcher, "ctx", None) or getattr(self.dispatcher, "context", None)
        bus = getattr(context, "state", None) if context is not None else None
        setter = getattr(bus, "set", None)
        if not callable(setter):
            return
        try:
            setter(AssistantState.THINKING)
        except Exception as exc:  # noqa: BLE001 - the UI must not be able to stop a turn
            self._log.debug("Could not set the state to thinking: %s", exc)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Brain(model={getattr(self.client, 'model', '?')!r}, max_tool_rounds={self.max_tool_rounds})"
