"""The fact store: what JARVIS remembers about his user, and how it survives bad files.

A broken ``memory.json`` must never stop the assistant from booting, so most of these
tests feed the store something it should not have been given and check that it comes
back empty with a warning instead of an exception.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from jarvis.core.memory import MAX_FACTS, MEMORY_VERSION, PROMPT_HEADER, Memory


@pytest.fixture
def memory_log():
    """Collect records from the memory logger without depending on propagation."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("jarvis.memory")
    previous_level = logger.level
    handler = Collector()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def warnings_in(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


# --- persistence ----------------------------------------------------------------------
def test_facts_survive_a_save_and_reload(tmp_path):
    store = Memory(tmp_path / "memory.json")
    store.remember("the user's name is Karim", topic="identity")
    store.remember("the user drinks tea")

    reloaded = Memory(tmp_path / "memory.json")
    assert [entry.text for entry in reloaded.facts()] == [
        "the user's name is Karim",
        "the user drinks tea",
    ]
    assert reloaded.facts()[0].topic == "identity"


def test_swedish_characters_survive_a_save_and_reload(tmp_path):
    path = tmp_path / "memory.json"
    store = Memory(path)
    store.remember("användaren bor i Göteborg och dricker kaffe på Söder", topic="hemmavid")

    assert "Göteborg".encode("utf-8") in path.read_bytes()
    reloaded = Memory(path)
    assert reloaded.facts()[0].text == "användaren bor i Göteborg och dricker kaffe på Söder"
    assert reloaded.facts()[0].topic == "hemmavid"


def test_saved_file_uses_the_documented_shape(tmp_path):
    path = tmp_path / "memory.json"
    Memory(path).remember("the user likes tea", topic="drinks")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == MEMORY_VERSION
    assert set(payload["facts"][0]) == {"text", "created", "topic"}
    assert payload["facts"][0]["text"] == "the user likes tea"


def test_a_missing_file_starts_empty_without_creating_it(tmp_path):
    path = tmp_path / "memory.json"
    store = Memory(path)
    assert store.facts() == []
    assert not path.exists()


def test_a_corrupt_file_recovers_to_empty_with_a_warning(tmp_path, memory_log):
    path = tmp_path / "memory.json"
    path.write_text('{"facts": [{"text": "half a fa', encoding="utf-8")

    store = Memory(path)

    assert store.facts() == []
    assert any(str(path) in message for message in warnings_in(memory_log))


def test_a_corrupt_file_can_still_be_written_over(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("}}} not json {{{", encoding="utf-8")
    store = Memory(path)
    store.remember("the user prefers Celsius")

    assert [entry.text for entry in Memory(path).facts()] == ["the user prefers Celsius"]


def test_an_empty_file_starts_empty_with_a_warning(tmp_path, memory_log):
    path = tmp_path / "memory.json"
    path.write_text("   \n", encoding="utf-8")

    assert Memory(path).facts() == []
    assert any(str(path) in message for message in warnings_in(memory_log))


def test_a_bare_list_of_facts_is_still_read(tmp_path):
    """Older builds wrote a plain list; those files must not be thrown away."""
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps([{"text": "the user likes tea", "created": "2024-01-01T08:00:00", "topic": ""}]),
        encoding="utf-8",
    )
    assert [entry.text for entry in Memory(path).facts()] == ["the user likes tea"]


def test_a_top_level_scalar_recovers_to_empty_with_a_warning(tmp_path, memory_log):
    path = tmp_path / "memory.json"
    path.write_text('"just a string"', encoding="utf-8")

    assert Memory(path).facts() == []
    assert any(str(path) in message for message in warnings_in(memory_log))


def test_a_facts_field_that_is_not_a_list_recovers_to_empty(tmp_path, memory_log):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"facts": {"text": "wrong shape"}, "version": 1}), encoding="utf-8")

    assert Memory(path).facts() == []
    assert any(str(path) in message for message in warnings_in(memory_log))


def test_unusable_entries_are_skipped_but_good_ones_are_kept(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "facts": [
                    {"text": "the user likes tea"},
                    {"nothing": "useful"},
                    None,
                    17,
                    {"text": "   "},
                    {"text": "the user dislikes rain"},
                ],
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    assert [entry.text for entry in Memory(path).facts()] == [
        "the user likes tea",
        "the user dislikes rain",
    ]


def test_a_non_utf8_file_is_salvaged_rather_than_dropped(tmp_path):
    """One mis-encoded byte must not cost the user everything JARVIS remembers.

    A memory.json saved by an editor in a legacy Windows code page used to raise
    UnicodeDecodeError out of Memory.load(), and therefore out of Memory(), so the
    assistant could not boot at all. It now reads the file leniently: the facts
    survive, with the undecodable characters replaced.
    """
    path = tmp_path / "memory.json"
    payload = json.dumps(
        {"facts": [{"text": "användaren bor i Göteborg", "created": "2024-01-01T08:00:00", "topic": ""}],
         "version": 1},
        ensure_ascii=False,
    )
    path.write_bytes(payload.encode("cp1252"))

    facts = Memory(path).facts()

    assert len(facts) == 1, "the fact should be salvaged, not discarded"
    assert facts[0].text.startswith("anv"), "the readable part of the text survives"


def test_a_directory_in_place_of_the_file_recovers_to_empty(tmp_path):
    store_dir = tmp_path / "memory.json"
    store_dir.mkdir()
    assert Memory(store_dir).facts() == []


# --- remember -------------------------------------------------------------------------
def test_remember_returns_the_stored_entry(memory):
    entry = memory.remember("the user likes tea", topic="drinks")
    assert entry.text == "the user likes tea"
    assert entry.topic == "drinks"
    assert entry.created


def test_remember_does_not_duplicate_the_same_fact(memory):
    memory.remember("the user likes tea")
    memory.remember("the user likes tea")
    assert len(memory.facts()) == 1


def test_remember_deduplicates_ignoring_case_and_spacing(memory):
    memory.remember("The user likes tea")
    memory.remember("the   user  likes   TEA")
    facts = memory.facts()
    assert len(facts) == 1
    assert facts[0].text == "the user likes TEA"  # the newest wording wins


def test_remember_keeps_the_position_of_a_refreshed_fact(memory):
    memory.remember("first fact")
    memory.remember("the user likes tea")
    memory.remember("last fact")
    memory.remember("The user likes TEA")
    assert [entry.text for entry in memory.facts()] == [
        "first fact",
        "The user likes TEA",
        "last fact",
    ]


def test_remember_ignores_an_empty_fact(memory):
    entry = memory.remember("   \n  ")
    assert entry.text == ""
    assert memory.facts() == []


def test_remember_collapses_whitespace_before_storing(memory):
    entry = memory.remember("  the user\n\tlikes   tea  ")
    assert entry.text == "the user likes tea"


def test_remember_drops_the_oldest_fact_once_the_cap_is_reached(tmp_path):
    store = Memory(tmp_path / "memory.json")
    for index in range(MAX_FACTS + 3):
        store.remember(f"fact number {index}")

    facts = store.facts()
    assert len(facts) == MAX_FACTS
    assert facts[0].text == "fact number 3"
    assert facts[-1].text == f"fact number {MAX_FACTS + 2}"


def test_two_threads_remembering_at_once_lose_nothing(tmp_path):
    store = Memory(tmp_path / "memory.json")
    start = threading.Barrier(2, timeout=5)

    def remember_many(prefix: str) -> None:
        start.wait()
        for index in range(25):
            store.remember(f"{prefix} fact {index}")

    threads = [threading.Thread(target=remember_many, args=(name,)) for name in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "remember() deadlocked under two threads"

    assert len(store.facts()) == 50
    assert len(Memory(tmp_path / "memory.json").facts()) == 50


# --- forget ---------------------------------------------------------------------------
def test_forget_matches_case_insensitively_on_the_fact_text(memory):
    memory.remember("the user drinks Kaffe every morning")
    assert memory.forget("KAFFE") == 1
    assert memory.facts() == []


def test_forget_matches_case_insensitively_on_the_topic(memory):
    memory.remember("the user drinks tea", topic="Drinks")
    assert memory.forget("drinks") == 1
    assert memory.facts() == []


def test_forget_matches_swedish_characters_case_insensitively(memory):
    memory.remember("användaren gillar ÅKA skidor", topic="fritid")
    assert memory.forget("åka") == 1


def test_forget_returns_the_number_of_facts_removed(memory):
    memory.remember("the user likes tea", topic="drinks")
    memory.remember("the user likes coffee", topic="drinks")
    memory.remember("the user dislikes rain", topic="weather")

    assert memory.forget("drinks") == 2
    assert [entry.text for entry in memory.facts()] == ["the user dislikes rain"]


def test_forget_returns_zero_and_keeps_everything_when_nothing_matches(memory):
    memory.remember("the user likes tea")
    assert memory.forget("bicycles") == 0
    assert len(memory.facts()) == 1


def test_forget_with_an_empty_topic_removes_nothing(memory):
    memory.remember("the user likes tea")
    assert memory.forget("   ") == 0
    assert len(memory.facts()) == 1


def test_forget_is_persisted(tmp_path):
    path = tmp_path / "memory.json"
    store = Memory(path)
    store.remember("the user likes tea")
    store.forget("tea")
    assert Memory(path).facts() == []


# --- prompt block ---------------------------------------------------------------------
def test_as_prompt_block_is_empty_when_nothing_is_remembered(memory):
    assert memory.as_prompt_block() == ""


def test_as_prompt_block_lists_facts_under_the_header(memory):
    memory.remember("the user likes tea")
    memory.remember("the user bor i Göteborg", topic="hemmavid")

    block = memory.as_prompt_block()
    assert block.startswith(PROMPT_HEADER)
    assert "- the user likes tea" in block
    assert "- [hemmavid] the user bor i Göteborg" in block
    assert block.endswith("\n")


# --- user name ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fact,expected",
    [
        ("my name is Karim", "Karim"),
        ("the user's name is Sara", "Sara"),
        ("jag heter Åsa", "Åsa"),
        ("användaren heter Björn", "Björn"),
        ("user name is karim khalil", "Karim Khalil"),
    ],
)
def test_user_name_is_read_from_the_stored_phrasing(memory, fact, expected):
    memory.remember(fact)
    assert memory.user_name == expected


def test_user_name_is_none_when_no_fact_reveals_it(memory):
    memory.remember("the user likes tea")
    memory.remember("the user dislikes rain")
    assert memory.user_name is None


def test_user_name_is_none_for_an_empty_store(memory):
    assert memory.user_name is None


def test_user_name_uses_the_most_recent_naming_fact(memory):
    memory.remember("my name is Karim")
    memory.remember("my name is Sara")
    assert memory.user_name == "Sara"


def test_user_name_stops_at_the_end_of_the_clause(memory):
    memory.remember("my name is Karim, and I live in Stockholm")
    assert memory.user_name == "Karim"


# --- misc -----------------------------------------------------------------------------
def test_clear_empties_the_store_and_reports_the_count(tmp_path):
    path = tmp_path / "memory.json"
    store = Memory(path)
    store.remember("the user likes tea")
    store.remember("the user dislikes rain")

    assert store.clear() == 2
    assert Memory(path).facts() == []


def test_facts_returns_a_copy_that_cannot_corrupt_the_store(memory):
    memory.remember("the user likes tea")
    facts = memory.facts()
    facts.clear()
    assert len(memory.facts()) == 1


def test_nothing_is_written_outside_the_given_path(tmp_path):
    path = tmp_path / "nested" / "memory.json"
    store = Memory(path)
    store.remember("the user likes tea")

    assert path.is_file()
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["memory.json"]
    assert Path(path).parent.parent == tmp_path
