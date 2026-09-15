"""Streaming sentence splitting and speech cleanup.

Everything here is fed the way the model actually streams: one token at a time. That is
the only way these tests mean anything, because the splitter has to decide whether a full
stop ends a sentence while the next character does not exist yet. A wrong early split is
audible — JARVIS says "three point" and then "five" — so the decimals, abbreviations and
times below are the real subject matter.

``clean_for_speech`` is the other half: whatever markdown, reasoning or emoji the model
produces, the speaker must be handed plain prose, with Swedish left exactly as it was.
"""

from __future__ import annotations

import pytest

from jarvis.brain.sentences import (
    HARD_MAX_CHARS,
    SOFT_MAX_CHARS,
    SentenceSplitter,
    clean_for_speech,
)


def stream(text: str, *, min_chars: int = 12, chunk: int = 1) -> tuple[list[str], str]:
    """Feed ``text`` in ``chunk``-sized tokens; return (sentences yielded, trailing fragment)."""
    splitter = SentenceSplitter(min_chars=min_chars)
    sentences: list[str] = []
    for index in range(0, len(text), chunk):
        sentences.extend(splitter.feed(text[index : index + chunk]))
    return sentences, splitter.flush()


# --- ordinary sentences ---------------------------------------------------------------
def test_a_finished_sentence_is_yielded_while_the_next_one_is_still_streaming():
    """This is what lets speech start before the model has finished thinking."""
    sentences, tail = stream("The reactor is stable, sir. Nothing to worry about.")

    assert sentences == ["The reactor is stable, sir."]
    assert tail == "Nothing to worry about."


def test_a_question_is_a_complete_sentence():
    sentences, _ = stream("Shall I open the pod bay doors, sir? I would advise against it.")

    assert sentences == ["Shall I open the pod bay doors, sir?"]


def test_an_exclamation_is_a_complete_sentence():
    sentences, _ = stream("The suit is on fire, sir! Please step away from it now.")

    assert sentences == ["The suit is on fire, sir!"]


def test_doubled_terminators_stay_with_their_sentence():
    sentences, _ = stream("Are you certain about this, sir?! I would reconsider it.")

    assert sentences == ["Are you certain about this, sir?!"]


def test_an_ellipsis_of_full_stops_does_not_split_three_times():
    sentences, _ = stream("I was thinking about it... Perhaps another approach would serve.")

    assert sentences == ["I was thinking about it..."]


def test_a_single_ellipsis_character_ends_one_sentence():
    sentences, _ = stream("It is rather late, sir… Perhaps sleep would help you.")

    assert sentences == ["It is rather late, sir…"]


def test_a_closing_quote_stays_with_its_sentence():
    sentences, _ = stream('He said "the reactor is perfectly stable." Then he left the room.')

    assert sentences == ['He said "the reactor is perfectly stable."']


def test_a_newline_ends_a_sentence_even_without_punctuation():
    sentences, tail = stream("First line of the report\nSecond line of the report\n")

    assert sentences == ["First line of the report", "Second line of the report"]
    assert tail == ""


# --- the things that must NOT split ---------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Mr. Stark is in the workshop. He asked not to be disturbed.",
            "Mr. Stark is in the workshop.",
            id="mr",
        ),
        pytest.param(
            "Dr. Banner called earlier today. He sounded calm enough.",
            "Dr. Banner called earlier today.",
            id="dr",
        ),
        pytest.param(
            "It is item No. 42 on the manifest, sir. Shall I open it?",
            "It is item No. 42 on the manifest, sir.",
            id="no",
        ),
        pytest.param(
            "Use a smaller model, e.g. the base one, for speed. That should help.",
            "Use a smaller model, e.g. the base one, for speed.",
            id="eg",
        ),
        pytest.param(
            "The drive is nearly full, i.e. under one gigabyte. Something must go.",
            "The drive is nearly full, i.e. under one gigabyte.",
            id="ie",
        ),
        pytest.param(
            "The result is approx. forty two units, sir. Nothing alarming.",
            "The result is approx. forty two units, sir.",
            id="approx",
        ),
        pytest.param(
            "t.ex. den här meningen är svensk. Nästa mening kommer här.",
            "t.ex. den här meningen är svensk.",
            id="swedish-tex",
        ),
        pytest.param(
            "Mötet är kl. 19.30 i kväll, sir. Jag påminner dig.",
            "Mötet är kl. 19.30 i kväll, sir.",
            id="swedish-kl",
        ),
        pytest.param(
            "J. Smith is waiting downstairs in the lobby. Shall I let him up?",
            "J. Smith is waiting downstairs in the lobby.",
            id="initial",
        ),
    ],
)
def test_an_abbreviation_does_not_end_a_sentence(text, expected):
    sentences, _ = stream(text)

    assert sentences == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "The temperature is 3.5 degrees outside right now. Dress warmly, sir.",
            "The temperature is 3.5 degrees outside right now.",
            id="decimal",
        ),
        pytest.param(
            "It costs 1.5 million dollars, sir. Shall I proceed?",
            "It costs 1.5 million dollars, sir.",
            id="large-decimal",
        ),
        pytest.param(
            "Your meeting starts at 19.30 this evening, sir. I will remind you.",
            "Your meeting starts at 19.30 this evening, sir.",
            id="time",
        ),
        pytest.param(
            "Version 2.0 is ready for you now, sir. Shall I install it?",
            "Version 2.0 is ready for you now, sir.",
            id="version",
        ),
        pytest.param(
            "Go to example.com for the details. Then tell me what you think.",
            "Go to example.com for the details.",
            id="domain",
        ),
    ],
)
def test_a_dot_inside_a_number_or_a_domain_does_not_end_a_sentence(text, expected):
    sentences, _ = stream(text)

    assert sentences == [expected]


def test_a_full_stop_at_the_very_end_is_held_back_until_the_next_token():
    """Until the next character arrives, "3." could still become "3.5"."""
    splitter = SentenceSplitter()

    assert splitter.feed("The temperature is 3.") == []
    assert splitter.feed("5 degrees outside, sir. ") == [
        "The temperature is 3.5 degrees outside, sir."
    ]
    assert splitter.flush() == ""


# --- short fragments ------------------------------------------------------------------
def test_a_short_answer_is_merged_into_the_sentence_that_follows_it():
    """A stray "Yes." on its own would be spoken as a separate, clipped utterance."""
    sentences, _ = stream("Yes. The reactor is perfectly stable, sir. Indeed it is fine.")

    assert sentences == ["Yes. The reactor is perfectly stable, sir."]


def test_a_list_number_never_becomes_its_own_utterance():
    sentences, tail = stream("1. The reactor is perfectly stable indeed, sir.")

    assert sentences == []
    assert tail == "1. The reactor is perfectly stable indeed, sir."


def test_a_lower_min_chars_lets_a_short_sentence_through():
    sentences, _ = stream("Yes. The reactor is perfectly stable, sir.", min_chars=3)

    assert sentences[0] == "Yes."


# --- run-on sentences -----------------------------------------------------------------
def test_a_run_on_sentence_is_cut_at_a_comma_so_speech_can_start():
    run_on = (
        "the diagnostics came back clean, the reactor is holding at ninety four percent, "
        "the workshop is locked, the coffee machine has been descaled, and the car is charged, "
        "so there is really nothing left for you to worry about this evening sir"
    )
    assert len(run_on) > SOFT_MAX_CHARS

    sentences, tail = stream(run_on)

    assert len(sentences) == 1
    assert sentences[0].endswith("the car is charged")
    assert tail == "so there is really nothing left for you to worry about this evening sir"


def test_an_unpunctuated_run_on_is_cut_at_a_space_rather_than_held_forever():
    sentences, _ = stream("word " * 90)

    assert sentences, "a model that forgot punctuation must not block all speech"
    assert len(sentences[0]) <= HARD_MAX_CHARS
    assert sentences[0].endswith("word")


def test_a_run_on_shorter_than_the_soft_limit_is_not_cut():
    sentences, tail = stream("a clause, another clause, and a third clause without an end")

    assert sentences == []
    assert tail == "a clause, another clause, and a third clause without an end"


# --- flush, reset and odd input -------------------------------------------------------
def test_flush_returns_the_trailing_fragment_and_empties_the_buffer():
    splitter = SentenceSplitter()
    splitter.feed("Half a thought with no ending")

    assert splitter.flush() == "Half a thought with no ending"
    assert splitter.flush() == ""


def test_reset_throws_the_buffer_away_for_a_barge_in():
    splitter = SentenceSplitter()
    splitter.feed("The weather in Gothenburg is")

    splitter.reset()

    assert splitter.flush() == ""


def test_feeding_nothing_yields_nothing():
    splitter = SentenceSplitter()

    assert splitter.feed("") == []
    assert splitter.feed(None) == []
    assert splitter.flush() == ""


def test_a_non_string_token_does_not_crash_the_stream():
    splitter = SentenceSplitter()

    assert splitter.feed(42) == []
    assert splitter.flush() == "42"


def test_the_token_size_does_not_change_the_result():
    text = "Öppnar Spotify nu, sir. Är det något mer du vill ha? Jag är här."

    per_character = stream(text, chunk=1)
    per_word = stream(text, chunk=5)
    in_one_go = stream(text, chunk=len(text))

    assert per_character == per_word == in_one_go


def test_swedish_sentences_split_like_english_ones():
    sentences, tail = stream("Öppnar Spotify nu, sir. Är det något mer du vill ha?")

    assert sentences == ["Öppnar Spotify nu, sir."]
    assert tail == "Är det något mer du vill ha?"


# --- clean_for_speech: markdown -------------------------------------------------------
def test_bold_markers_are_not_spoken():
    assert clean_for_speech("**Absolutely**, sir.") == "Absolutely, sir."


def test_italic_markers_are_not_spoken():
    assert clean_for_speech("*Absolutely*, sir.") == "Absolutely, sir."


def test_a_header_becomes_plain_prose():
    assert clean_for_speech("# Status report\nAll systems nominal.") == (
        "Status report All systems nominal."
    )


def test_a_fenced_code_block_is_removed_entirely():
    spoken = clean_for_speech("Here is the code:\n```python\nprint('hi')\n```\nThat is all, sir.")

    assert spoken == "Here is the code: That is all, sir."


def test_an_unterminated_code_fence_is_still_removed():
    spoken = clean_for_speech("Run this, sir:\n```powershell\nGet-Process | Sort-Object CPU")

    assert "```" not in spoken
    assert spoken == "Run this, sir:"


def test_inline_code_keeps_its_words_but_loses_the_backticks():
    assert clean_for_speech("Run `systemctl restart` now, sir.") == (
        "Run systemctl restart now, sir."
    )


def test_bullet_markers_are_dropped():
    assert clean_for_speech("- First item\n- Second item") == "First item Second item"


def test_numbered_list_markers_are_dropped():
    assert clean_for_speech("1. First item\n2. Second item") == "First item Second item"


def test_a_link_keeps_its_label_and_loses_its_target():
    spoken = clean_for_speech("See [the report](https://example.com/report) for details.")

    assert spoken == "See the report for details."
    assert "example.com" not in spoken


def test_an_image_keeps_its_alt_text():
    assert clean_for_speech("![the reactor](reactor.png) is online") == "the reactor is online"


def test_emoji_are_never_spoken():
    assert clean_for_speech("All good 🚀 sir 😀.") == "All good sir."


def test_a_table_is_reduced_to_its_cells():
    assert clean_for_speech("| a | b |\n|---|---|\n| 1 | 2 |") == "a b 1 2"


def test_a_quoted_line_loses_its_marker():
    assert clean_for_speech("> quoted line, sir") == "quoted line, sir"


def test_underscore_emphasis_is_removed_completely():
    assert clean_for_speech("_emphasis_ and __strong__ text") == "emphasis and strong text"


# --- clean_for_speech: reasoning blocks -----------------------------------------------
def test_a_complete_think_block_is_never_spoken():
    spoken = clean_for_speech(
        "<think>The user wants the time. I should check the clock.</think>"
        "It is half past three, sir."
    )

    assert spoken == "It is half past three, sir."


def test_an_unterminated_think_block_takes_the_rest_of_the_text_with_it():
    """Better silence than reading the model's private reasoning out loud."""
    spoken = clean_for_speech("<think>The user wants the time and I never stopped thinking")

    assert spoken == ""


def test_a_dangling_think_closing_tag_is_removed():
    assert clean_for_speech("leftover reasoning</think>The answer is 42, sir.") == (
        "The answer is 42, sir."
    )


def test_a_think_block_in_the_middle_is_cut_out():
    assert clean_for_speech("Before <think>hidden</think> after, sir.") == "Before after, sir."


def test_an_uppercase_think_tag_is_treated_the_same():
    assert clean_for_speech("<THINK>hidden</THINK>Visible, sir.") == "Visible, sir."


# --- clean_for_speech: symbols and unicode --------------------------------------------
def test_percent_is_spoken_as_a_word():
    assert clean_for_speech("CPU is at 24% and the disk is 41% full.") == (
        "CPU is at 24 percent and the disk is 41 percent full."
    )


def test_ampersand_is_spoken_as_and():
    assert clean_for_speech("Tony & Pepper are waiting, sir.") == (
        "Tony and Pepper are waiting, sir."
    )


def test_degrees_celsius_is_spoken_as_degrees():
    assert clean_for_speech("It is 21°C outside, sir.") == "It is 21 degrees outside, sir."


def test_a_spaced_slash_becomes_a_pause_not_a_word():
    assert clean_for_speech("Choose red / blue, sir.") == "Choose red blue, sir."


def test_swedish_letters_are_left_exactly_as_they_are():
    text = "Öppnar Spotify nu, sir — allt är som det ska. Åke och Ärlig väntar."

    assert clean_for_speech(text) == text


def test_swedish_markdown_is_cleaned_without_touching_the_letters():
    assert clean_for_speech("Detta är **mycket** viktigt, sir — å ä ö.") == (
        "Detta är mycket viktigt, sir — å ä ö."
    )


def test_an_underscore_inside_a_filename_is_left_alone():
    assert clean_for_speech("The file is called jarvis_system.md, sir.") == (
        "The file is called jarvis_system.md, sir."
    )


def test_whitespace_is_collapsed_to_single_spaces():
    assert clean_for_speech("Too    many\n\n  spaces   here") == "Too many spaces here"


@pytest.mark.parametrize("text", ["", "   \n\n  ", None])
def test_nothing_in_means_nothing_out(text):
    assert clean_for_speech(text) == ""


def test_cleaned_output_can_be_streamed_straight_back_into_the_splitter():
    """The two halves of the module are used together in the real pipeline."""
    spoken = clean_for_speech("**Done.** Spotify is open, sir. Anything else? 🎵")

    sentences, tail = stream(spoken, min_chars=3)

    assert sentences == ["Done.", "Spotify is open, sir."]
    assert tail == "Anything else?"
