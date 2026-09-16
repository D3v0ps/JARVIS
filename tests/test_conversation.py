"""The rolling history: what JARVIS is told about himself, and what he is allowed to forget.

Two things here are worth a person's attention. The system message has to carry the
persona, what he remembers about his user and what time it is, and it must still be
assembled when the prompt file has gone missing. And trimming must never produce the one
shape Ollama rejects outright: a ``role: "tool"`` result whose assistant ``tool_calls``
message was dropped. A single broken pair means the next thing the user says gets a
500 back from the server instead of an answer.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import pytest

from jarvis.brain.conversation import FALLBACK_SYSTEM_PROMPT, Conversation
from jarvis.config import project_root
from jarvis.core.memory import PROMPT_HEADER

PERSONA = "You are J.A.R.V.I.S., the resident intelligence of this house. Address the user as sir."


class LogSpy:
    """A logger with its own handler, independent of the root configuration."""

    def __init__(self, name: str) -> None:
        self.records: list[logging.LogRecord] = []
        spy = self

        class Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                spy.records.append(record)

        self.logger = logging.getLogger(name)
        self.handler = Collector()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def close(self) -> None:
        self.logger.removeHandler(self.handler)

    def at_least(self, level: int) -> list[str]:
        return [record.getMessage() for record in self.records if record.levelno >= level]


@pytest.fixture
def log():
    spy = LogSpy("tests.brain.conversation")
    try:
        yield spy
    finally:
        spy.close()


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "jarvis_system.md"
    path.write_text(PERSONA, encoding="utf-8")
    return path


@pytest.fixture
def conversation(prompt_file, memory, log):
    return Conversation(prompt_file, memory, history_turns=12, logger=log.logger)


def assert_tool_pairs_intact(messages: list[dict]) -> None:
    """No tool result without its request, and no request without its results."""
    roles = [message.get("role") for message in messages]
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            back = index - 1
            while back >= 0 and messages[back].get("role") == "tool":
                back -= 1
            assert back >= 0, f"tool result at {index} starts the history: {roles}"
            assert messages[back].get("tool_calls"), (
                f"tool result at {index} has no assistant tool_calls message: {roles}"
            )
        if message.get("tool_calls"):
            assert index + 1 < len(messages), f"tool call at {index} has no results: {roles}"
            assert messages[index + 1].get("role") == "tool", (
                f"tool call at {index} is not followed by its results: {roles}"
            )


def tool_exchange(conversation: Conversation, index: int) -> None:
    """One user turn answered with a tool round and then a spoken reply."""
    conversation.add_user(f"what is the system doing, take {index}")
    conversation.add_assistant("", tool_calls=[{"name": "system_status", "arguments": {}}])
    conversation.add_tool_result("system_status", f"CPU {index} percent")
    conversation.add_assistant(f"The system is at {index} percent, sir.")


def plain_exchange(conversation: Conversation, index: int) -> None:
    conversation.add_user(f"tell me something, take {index}")
    conversation.add_assistant(f"Certainly, sir. Observation number {index}.")


# --- the system message ---------------------------------------------------------------
def test_system_message_contains_the_persona_prompt(conversation):
    message = conversation.system_message()

    assert message["role"] == "system"
    assert PERSONA in message["content"]


def test_system_message_contains_the_memory_block_when_facts_are_known(conversation, memory):
    memory.remember("the user's name is Åsa", topic="identity")

    content = conversation.system_message()["content"]

    assert PROMPT_HEADER in content
    assert "the user's name is Åsa" in content


def test_system_message_has_no_memory_block_when_nothing_is_remembered(conversation):
    assert PROMPT_HEADER not in conversation.system_message()["content"]


def test_the_model_is_told_the_current_date_and_time(conversation):
    conversation.add_user("what time is it")

    content = conversation.messages()[-1]["content"]

    match = re.search(r"It is (\w+) (\d{1,2}) (\w+) (\d{4}), (\d{2}):(\d{2})\.", content)
    assert match, content
    now = datetime.now()
    assert int(match.group(2)) == now.day
    assert int(match.group(4)) == now.year
    assert 0 <= int(match.group(5)) <= 23


def test_the_clock_is_not_in_the_system_message(conversation):
    """It used to be, and it cost a prompt cache miss at every minute boundary.

    Ollama caches its evaluation of the prompt prefix. With the clock inside the
    system message, two thousand tokens of persona and tool schemas were
    re-evaluated on every turn that happened to cross a minute.
    """
    conversation.add_user("what time is it")

    assert "It is" not in conversation.system_message()["content"]


def test_the_system_message_is_byte_identical_between_turns(conversation, monkeypatch):
    """Which is the whole point: an unchanged prefix is a cache hit."""
    moments = iter([datetime(2026, 9, 15, 8, 15, 0), datetime(2026, 9, 15, 23, 45, 0)])

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments)

    monkeypatch.setattr("jarvis.brain.conversation.datetime", FrozenDatetime)
    conversation.add_user("first")

    first = conversation.messages()
    conversation.add_user("second")
    second = conversation.messages()

    assert first[0]["content"] == second[0]["content"]
    assert "08:15" in first[-1]["content"]
    assert "23:45" in second[-1]["content"]

def test_a_missing_prompt_file_falls_back_to_the_builtin_persona(tmp_path, memory, log):
    conversation = Conversation(tmp_path / "gone.md", memory, logger=log.logger)

    content = conversation.system_message()["content"]

    assert FALLBACK_SYSTEM_PROMPT in content
    assert any("built-in fallback" in message for message in log.at_least(logging.ERROR))


def test_a_missing_prompt_file_is_only_reported_once(tmp_path, memory, log):
    conversation = Conversation(tmp_path / "gone.md", memory, logger=log.logger)

    for _ in range(5):
        conversation.system_message()

    assert len(log.at_least(logging.ERROR)) == 1


def test_an_unreadable_prompt_file_falls_back_to_the_builtin_persona(tmp_path, memory, log):
    """A directory where the prompt should be: read it and you get an OSError, not a string."""
    broken = tmp_path / "jarvis_system.md"
    broken.mkdir()
    conversation = Conversation(broken, memory, logger=log.logger)

    content = conversation.system_message()["content"]

    assert FALLBACK_SYSTEM_PROMPT in content
    assert log.at_least(logging.ERROR)


def test_an_empty_prompt_file_falls_back_to_the_builtin_persona(tmp_path, memory, log):
    empty = tmp_path / "jarvis_system.md"
    empty.write_text("   \n\n", encoding="utf-8")
    conversation = Conversation(empty, memory, logger=log.logger)

    assert FALLBACK_SYSTEM_PROMPT in conversation.system_message()["content"]


def test_a_relative_prompt_path_is_anchored_at_the_project_root(memory, log):
    conversation = Conversation("prompts/jarvis_system.md", memory, logger=log.logger)

    assert conversation.prompt_path == (project_root() / "prompts" / "jarvis_system.md").resolve()
    assert "J.A.R.V.I.S." in conversation.system_message()["content"]


def test_an_edited_prompt_file_is_picked_up_again(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, logger=log.logger)
    assert PERSONA in conversation.system_message()["content"]

    import os

    prompt_file.write_text("You are someone else entirely.", encoding="utf-8")
    os.utime(prompt_file, (0, 0))  # force a different mtime, not a same-second rewrite

    assert "You are someone else entirely." in conversation.system_message()["content"]


def test_broken_memory_does_not_break_the_system_message(prompt_file, log):
    class ExplodingMemory:
        def as_prompt_block(self):
            raise RuntimeError("memory.json is on fire")

    conversation = Conversation(prompt_file, ExplodingMemory(), logger=log.logger)

    content = conversation.system_message()["content"]

    assert PERSONA in content
    assert any("Could not read memory" in message for message in log.at_least(logging.WARNING))


# --- the language line ----------------------------------------------------------------
def test_no_language_line_when_the_language_is_automatic(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, language=None, logger=log.logger)

    content = conversation.system_message()["content"]

    assert "Always answer in English" not in content
    assert "Svara alltid på svenska" not in content


def test_english_is_pinned_when_it_is_forced(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, language="en", logger=log.logger)

    assert "Always answer in English" in conversation.system_message()["content"]


def test_swedish_is_pinned_in_swedish_when_it_is_forced(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, language="sv", logger=log.logger)

    assert "Svara alltid på svenska" in conversation.system_message()["content"]


def test_an_unsupported_language_pins_nothing(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, language="de", logger=log.logger)

    content = conversation.system_message()["content"]

    assert "Always answer in English" not in content
    assert "Svara alltid på svenska" not in content


# --- history ------------------------------------------------------------------------
def test_a_turn_is_recorded_as_user_then_assistant(conversation):
    conversation.add_user("open Spotify")
    conversation.add_assistant("Right away, sir.")

    assert conversation.history == [
        {"role": "user", "content": "open Spotify"},
        {"role": "assistant", "content": "Right away, sir."},
    ]
    assert conversation.turns == 1


def test_an_empty_user_turn_is_ignored(conversation):
    conversation.add_user("   ")
    conversation.add_user("")

    assert conversation.history == []
    assert conversation.turns == 0


def test_an_empty_assistant_turn_is_ignored(conversation):
    conversation.add_user("hello")
    conversation.add_assistant("  ")

    assert [message["role"] for message in conversation.history] == ["user"]


def test_swedish_text_survives_the_history_round_trip(conversation):
    conversation.add_user("Öppna fönstret och höj värmen, tack")
    conversation.add_assistant("Självklart, sir — det är gjort.")

    assert conversation.history[0]["content"] == "Öppna fönstret och höj värmen, tack"
    assert conversation.history[1]["content"] == "Självklart, sir — det är gjort."


def test_messages_start_with_the_system_message(conversation):
    conversation.add_user("hello")
    conversation.add_assistant("Good evening, sir.")

    messages = conversation.messages()

    assert messages[0]["role"] == "system"
    assert [message["role"] for message in messages[1:]] == ["user", "assistant"]


def test_messages_hands_out_copies_not_the_live_history(conversation):
    conversation.add_user("hello")
    messages = conversation.messages()

    messages[1]["content"] = "tampered with"

    assert conversation.history[0]["content"] == "hello"


def test_clear_forgets_the_history_but_not_the_persona(conversation):
    conversation.add_user("hello")
    conversation.add_assistant("Good evening, sir.")

    conversation.clear()

    assert conversation.history == []
    assert conversation.turns == 0
    assert PERSONA in conversation.messages()[0]["content"]


def test_flat_tool_calls_are_rendered_in_the_shape_ollama_expects(conversation):
    conversation.add_user("how is the machine")
    conversation.add_assistant("", tool_calls=[{"name": "system_status", "arguments": {"unit": "c"}}])

    assert conversation.history[1]["tool_calls"] == [
        {"function": {"name": "system_status", "arguments": {"unit": "c"}}}
    ]


def test_a_tool_call_without_a_name_is_dropped_rather_than_sent(conversation):
    conversation.add_user("how is the machine")
    conversation.add_assistant("Checking, sir.", tool_calls=[{"arguments": {}}])

    assert "tool_calls" not in conversation.history[1]


def test_a_tool_result_without_a_tool_call_is_refused(conversation, log):
    conversation.add_user("how is the machine")
    conversation.add_assistant("It is fine, sir.")

    conversation.add_tool_result("system_status", "CPU 12 percent")

    assert [message["role"] for message in conversation.history] == ["user", "assistant"]
    assert any("no assistant tool_calls" in message for message in log.at_least(logging.WARNING))


def test_several_results_attach_to_one_assistant_tool_call(conversation):
    conversation.add_user("what is going on")
    conversation.add_assistant(
        "",
        tool_calls=[{"name": "system_status", "arguments": {}}, {"name": "get_time_date", "arguments": {}}],
    )
    conversation.add_tool_result("system_status", "CPU 12 percent")
    conversation.add_tool_result("get_time_date", "It is 21:47")

    assert [message["role"] for message in conversation.history] == [
        "user", "assistant", "tool", "tool",
    ]
    assert_tool_pairs_intact(conversation.history)


# --- trimming -------------------------------------------------------------------------
def test_history_is_trimmed_to_the_configured_number_of_turns(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, history_turns=3, logger=log.logger)

    for index in range(10):
        plain_exchange(conversation, index)

    assert conversation.turns == 3
    assert conversation.history[0]["content"] == "tell me something, take 7"


def test_trimming_never_drops_the_system_message(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, history_turns=1, logger=log.logger)

    for index in range(6):
        plain_exchange(conversation, index)

    messages = conversation.messages()
    assert messages[0]["role"] == "system"
    assert PERSONA in messages[0]["content"]
    assert conversation.turns == 1


def test_a_window_of_one_turn_keeps_the_latest_exchange(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, history_turns=1, logger=log.logger)

    plain_exchange(conversation, 1)
    plain_exchange(conversation, 2)

    assert [message["content"] for message in conversation.history] == [
        "tell me something, take 2",
        "Certainly, sir. Observation number 2.",
    ]


def test_a_tool_exchange_on_the_trim_boundary_is_kept_whole(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, history_turns=2, logger=log.logger)

    tool_exchange(conversation, 1)
    plain_exchange(conversation, 2)

    assert [message["role"] for message in conversation.history] == [
        "user", "assistant", "tool", "assistant", "user", "assistant",
    ]
    assert_tool_pairs_intact(conversation.history)


def test_a_tool_exchange_is_dropped_whole_when_it_falls_out_of_the_window(prompt_file, memory, log):
    conversation = Conversation(prompt_file, memory, history_turns=1, logger=log.logger)

    tool_exchange(conversation, 1)
    plain_exchange(conversation, 2)

    assert [message["role"] for message in conversation.history] == ["user", "assistant"]
    assert_tool_pairs_intact(conversation.history)


@pytest.mark.parametrize("history_turns", [1, 2, 3, 5, 12])
def test_trimming_never_breaks_a_tool_pair_over_thirty_turns(prompt_file, memory, log, history_turns):
    """The invariant that keeps Ollama from rejecting the whole request."""
    conversation = Conversation(prompt_file, memory, history_turns=history_turns, logger=log.logger)

    for index in range(30):
        conversation.add_user(f"turn {index}")
        assert_tool_pairs_intact(conversation.history)

        if index % 3 == 0:
            conversation.add_assistant("", tool_calls=[{"name": "system_status", "arguments": {}}])
            conversation.add_tool_result("system_status", f"CPU {index} percent")
            assert_tool_pairs_intact(conversation.history)
            conversation.add_assistant(f"The system is fine, sir. ({index})")
        elif index % 3 == 1:
            conversation.add_assistant(
                "",
                tool_calls=[
                    {"name": "weather", "arguments": {"city": "Göteborg"}},
                    {"name": "get_time_date", "arguments": {}},
                ],
            )
            conversation.add_tool_result("weather", "Nine degrees and raining")
            conversation.add_tool_result("get_time_date", "It is 21:47")
            conversation.add_assistant(f"Nine degrees, sir. ({index})")
        else:
            conversation.add_assistant(f"Certainly, sir. ({index})")

        assert_tool_pairs_intact(conversation.history)
        conversation.trim()
        assert_tool_pairs_intact(conversation.history)
        assert_tool_pairs_intact(conversation.messages()[1:])
        assert conversation.turns <= history_turns


def test_an_abandoned_tool_call_is_dropped_before_the_next_request(conversation, log):
    """The user barged in before the tool answered: the call must not go back on its own."""
    conversation.add_user("how is the machine")
    conversation.add_assistant("", tool_calls=[{"name": "system_status", "arguments": {}}])

    conversation.add_user("never mind, open Spotify")

    assert [message["role"] for message in conversation.history] == ["user", "user"]
    assert_tool_pairs_intact(conversation.messages()[1:])


def test_a_tool_result_for_an_abandoned_call_is_not_recorded(conversation):
    conversation.add_user("how is the machine")
    conversation.add_assistant("", tool_calls=[{"name": "system_status", "arguments": {}}])
    conversation.add_user("never mind, open Spotify")

    conversation.add_tool_result("system_status", "CPU 12 percent")

    assert_tool_pairs_intact(conversation.history)
    assert all(message["role"] != "tool" for message in conversation.history)


def test_turns_counts_user_messages_only(conversation):
    for index in range(4):
        tool_exchange(conversation, index)

    assert conversation.turns == 4
