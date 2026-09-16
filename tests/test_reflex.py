"""The reflex grammar: what it matches, and - more importantly - what it refuses to.

Every test here runs on a bare Linux box: the matcher is pure Python plus PyYAML and
never touches the registry, a tool or the network. The near-miss cases are the point
of the file. A template that is too greedy takes the model out of a sentence that
needed it, so anything ambiguous must come back as ``None``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.brain.reflex import Reflex, ReflexMatcher  # noqa: E402

GRAMMAR = ROOT / "prompts" / "sentences.yaml"

#: One concrete value per slot, used to turn every template back into a sentence.
SLOT_SAMPLES = {
    "app": "spotify",
    "level": "forty",
    "minutes": "ten",
    "hours": "two",
    "city": "stockholm",
    "query": "cheap flights",
    "duration": "a quarter",
}


class FakeResult:
    """Stands in for ToolResult without importing the tools package."""

    def __init__(self, summary: str = "Done.", ok: bool = True, refused: bool = False) -> None:
        self.ok = ok
        self.summary = summary
        self.refused = refused
        self.detail = ""
        self.data = None


@pytest.fixture(scope="module")
def matcher() -> ReflexMatcher:
    """The shipped grammar. Module scoped because matching never mutates it."""
    return ReflexMatcher(GRAMMAR)


@pytest.fixture
def fresh() -> ReflexMatcher:
    """A matcher with untouched reply rotation, for the rotation tests."""
    return ReflexMatcher(GRAMMAR)


@pytest.fixture
def grammar() -> dict:
    return yaml.safe_load(GRAMMAR.read_text(encoding="utf-8"))


def write_grammar(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sentences.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def test_shipped_grammar_loads(matcher: ReflexMatcher) -> None:
    assert matcher.available is True
    assert matcher.templates() > 100


def test_missing_file_degrades_quietly(tmp_path: Path) -> None:
    quiet = ReflexMatcher(tmp_path / "nope.yaml")
    assert quiet.available is False
    assert quiet.templates() == 0
    assert quiet.match("open spotify") is None


def test_broken_yaml_degrades_quietly(tmp_path: Path) -> None:
    path = write_grammar(tmp_path, "tools: [this: is: not: yaml\n")
    broken = ReflexMatcher(path)
    assert broken.available is False
    assert broken.match("open spotify") is None


def test_yaml_that_is_not_a_mapping_degrades_quietly(tmp_path: Path) -> None:
    path = write_grammar(tmp_path, "- just\n- a\n- list\n")
    assert ReflexMatcher(path).available is False


def test_grammar_without_tools_section_degrades_quietly(tmp_path: Path) -> None:
    path = write_grammar(tmp_path, "version: 1\nvalues: {}\nslots: {}\n")
    assert ReflexMatcher(path).available is False


def test_template_with_unknown_slot_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = write_grammar(tmp_path, """
version: 1
slots:
  app: {type: str, max_words: 2}
tools:
  open_app:
    replies: {en: ["Right away, sir."]}
    rules:
      - args: {name: "{app}"}
        en: ["open {nosuchslot}", "open {app}"]
""")
    partial = ReflexMatcher(path)
    assert partial.templates() == 1
    assert partial.match("open spotify").tool == "open_app"


# --------------------------------------------------------------------------------------
# Every template in the shipped file, in both languages
# --------------------------------------------------------------------------------------
def _sentences_for(template: str) -> list[str]:
    """Turn one template into the sentence(s) a user would actually say."""
    tokens = template.split()
    with_optional: list[str] = []
    without_optional: list[str] = []
    for token in tokens:
        if token.startswith("{") and token.endswith("}"):
            word = SLOT_SAMPLES[token[1:-1]]
            with_optional.append(word)
            without_optional.append(word)
        elif token.startswith("[") and token.endswith("]"):
            with_optional.append(token[1:-1])
        else:
            with_optional.append(token)
            without_optional.append(token)
    sentences = [" ".join(with_optional)]
    if without_optional != with_optional:
        sentences.append(" ".join(without_optional))
    return sentences


def test_every_template_matches_its_own_tool(matcher: ReflexMatcher, grammar: dict) -> None:
    failures: list[str] = []
    checked = 0
    for tool, spec in grammar["tools"].items():
        for rule in spec["rules"]:
            for language in ("en", "sv"):
                for template in rule.get(language) or []:
                    for sentence in _sentences_for(template):
                        checked += 1
                        reflex = matcher.match(sentence)
                        if reflex is None:
                            failures.append(f"{language} {sentence!r} matched nothing")
                        elif reflex.tool != tool:
                            failures.append(
                                f"{language} {sentence!r} matched {reflex.tool}, wanted {tool}"
                            )
    assert checked > 200
    assert failures == []


def test_every_tool_in_the_grammar_has_english_and_swedish(grammar: dict) -> None:
    for tool, spec in grammar["tools"].items():
        languages = {lang for rule in spec["rules"] for lang in ("en", "sv") if rule.get(lang)}
        assert languages == {"en", "sv"}, f"{tool} is missing a language"
        assert spec["replies"].get("en") and spec["replies"].get("sv"), f"{tool} replies"


def test_grammar_covers_the_tools_that_are_said_out_loud(grammar: dict) -> None:
    expected = {
        "open_app", "close_app", "volume", "media", "set_timer", "get_time_date",
        "system_status", "lock_pc", "screenshot", "weather", "web_search",
    }
    assert expected <= set(grammar["tools"])


# --------------------------------------------------------------------------------------
# The commands themselves
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(("utterance", "tool", "arguments"), [
    # English
    ("Open Spotify.", "open_app", {"name": "spotify"}),
    ("launch steam", "open_app", {"name": "steam"}),
    ("close discord", "close_app", {"name": "discord"}),
    ("shut down chrome", "close_app", {"name": "chrome"}),
    ("volume to forty", "volume", {"action": "set", "level": 40}),
    ("set the volume to 65", "volume", {"action": "set", "level": 65}),
    ("volume up", "volume", {"action": "up"}),
    ("turn it down", "volume", {"action": "down"}),
    ("mute the sound", "volume", {"action": "mute"}),
    ("unmute", "volume", {"action": "unmute"}),
    ("next song", "media", {"action": "next"}),
    ("pause the music", "media", {"action": "play_pause"}),
    ("previous track", "media", {"action": "previous"}),
    ("stop the music", "media", {"action": "stop"}),
    ("set a timer for ten minutes", "set_timer", {"minutes": 10}),
    ("what time is it", "get_time_date", {}),
    ("system status", "system_status", {}),
    ("lock the computer", "lock_pc", {}),
    ("take a screenshot", "screenshot", {}),
    ("what's the weather", "weather", {}),
    ("what's the weather in gothenburg", "weather", {"city": "gothenburg"}),
    ("search for cheap flights to rome", "web_search", {"query": "cheap flights to rome"}),
    # Swedish
    ("Öppna Spotify.", "open_app", {"name": "spotify"}),
    ("starta steam", "open_app", {"name": "steam"}),
    ("stäng discord", "close_app", {"name": "discord"}),
    ("stäng av spotify", "close_app", {"name": "spotify"}),
    ("volym till fyrtio", "volume", {"action": "set", "level": 40}),
    ("sätt volymen på 65", "volume", {"action": "set", "level": 65}),
    ("höj volymen", "volume", {"action": "up"}),
    ("sänk ljudet", "volume", {"action": "down"}),
    ("stäng av ljudet", "volume", {"action": "mute"}),
    ("slå på ljudet", "volume", {"action": "unmute"}),
    ("nästa låt", "media", {"action": "next"}),
    ("pausa musiken", "media", {"action": "play_pause"}),
    ("förra låten", "media", {"action": "previous"}),
    ("stoppa musiken", "media", {"action": "stop"}),
    ("sätt en timer på tio minuter", "set_timer", {"minutes": 10}),
    ("vad är klockan", "get_time_date", {}),
    ("hur mår datorn", "system_status", {}),
    ("lås datorn", "lock_pc", {}),
    ("ta en skärmdump", "screenshot", {}),
    ("vad är vädret", "weather", {}),
    ("vädret i göteborg", "weather", {"city": "göteborg"}),
    ("googla billiga flyg", "web_search", {"query": "billiga flyg"}),
])
def test_spoken_commands(matcher: ReflexMatcher, utterance: str, tool: str, arguments: dict) -> None:
    reflex = matcher.match(utterance)
    assert reflex is not None, f"{utterance!r} should have matched {tool}"
    assert reflex.tool == tool
    assert reflex.arguments == arguments
    assert reflex.template
    assert reflex.utterance


# --------------------------------------------------------------------------------------
# Near misses: everything below belongs to the model
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("utterance", [
    "open spotify and tell me the weather",      # two commands
    "öppna spotify och stäng steam",             # two commands, Swedish
    "take a screenshot and email it to me",
    "is spotify open",                           # a question about state
    "is the volume muted",
    "why is the volume so low",
    "should i close steam",
    "what did i say about spotify",
    "vad tycker du om spotify",
    "tell me a joke",
    "set a timer",                               # no duration at all
    "sätt en timer",
    "volume to eleventy",                        # not a number
    "volume to two hundred",                     # out of range
    "take a screenshot of the moon",             # trailing words
    "open the pod bay doors",                    # an article, not an app
    "stäng av datorn",                           # that is a shutdown, not an app
    "lock the front door",
    "what's the weather going to be like tomorrow afternoon in northern sweden",
    "search",                                    # nothing to search for
    "google",
    "please",
    "",
    "   ",
])
def test_near_misses_fall_through_to_the_model(matcher: ReflexMatcher, utterance: str) -> None:
    assert matcher.match(utterance) is None


def test_non_string_input_is_not_a_crash(matcher: ReflexMatcher) -> None:
    assert matcher.match(None) is None            # type: ignore[arg-type]
    assert matcher.match(12) is None              # type: ignore[arg-type]


def test_a_long_sentence_is_never_a_reflex(matcher: ReflexMatcher) -> None:
    rambling = ("open spotify because i would rather listen to something loud "
                "while i finish the whole report tonight")
    assert len(rambling.split()) > 16
    assert matcher.match(rambling) is None


# --------------------------------------------------------------------------------------
# Spoken numbers
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(("utterance", "expected"), [
    ("volume to zero", 0),
    ("volume to five", 5),
    ("volume to fifteen", 15),
    ("volume to twenty five", 25),
    ("volume to forty", 40),
    ("volume to one hundred", 100),
    ("volume to 73", 73),
    ("volym till fyrtio", 40),
    ("volym till förtio", 40),
    ("volym till tjugofem", 25),
    ("volym till tjugo fem", 25),
    ("sätt volymen på hundra", 100),
    ("volym till noll", 0),
])
def test_spoken_levels(matcher: ReflexMatcher, utterance: str, expected: int) -> None:
    reflex = matcher.match(utterance)
    assert reflex is not None and reflex.arguments["level"] == expected
    assert isinstance(reflex.arguments["level"], int)


@pytest.mark.parametrize(("utterance", "expected"), [
    ("set a timer for ten minutes", 10),
    ("set a timer for one minute", 1),
    ("set a timer for ninety minutes", 90),
    ("set a timer for a quarter of an hour", 15),
    ("set a timer for half an hour", 30),
    ("set a timer for two hours", 120),
    ("set a timer for one hour", 60),
    ("sätt en timer på tio minuter", 10),
    ("sätt en timer på en kvart", 15),
    ("sätt en timer på en halvtimme", 30),
    ("sätt en timer på två timmar", 120),
    ("sätt en timer på 45 minuter", 45),
    ("sätt en timer på tjugofem minuter", 25),
])
def test_spoken_durations(matcher: ReflexMatcher, utterance: str, expected: float) -> None:
    reflex = matcher.match(utterance)
    assert reflex is not None, utterance
    assert reflex.tool == "set_timer"
    assert reflex.arguments["minutes"] == expected
    assert isinstance(reflex.arguments["minutes"], (int, float))


# --------------------------------------------------------------------------------------
# App aliases and mishearings
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(("spoken", "expected"), [
    ("open vs code", "code"),
    ("open code", "code"),
    ("open visual studio code", "code"),
    ("open vscode", "code"),
    ("open spotify", "spotify"),
    ("open spotifi", "spotify"),          # Whisper mishears it constantly
    ("open steam", "steam"),
    ("open stream", "steam"),             # and this one too
    ("öppna utforskaren", "explorer"),
    ("öppna kalkylatorn", "calculator"),
    ("open task manager", "task manager"),
    ("open photoshop", "photoshop"),      # unknown app: pass it through untouched
])
def test_app_aliases(matcher: ReflexMatcher, spoken: str, expected: str) -> None:
    reflex = matcher.match(spoken)
    assert reflex is not None and reflex.arguments["name"] == expected


# --------------------------------------------------------------------------------------
# Wake phrase and filler
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("utterance", [
    "jarvis open spotify",
    "Jarvis, open Spotify.",
    "hey jarvis, open spotify",
    "Hey Jarvis open Spotify!",
    "ok jarvis open spotify",
    "please open spotify",
    "can you open spotify?",
    "could you please open spotify",
    "open spotify please",
    "open spotify for me",
    "open spotify now",
    "jarvis, could you open spotify for me please",
    "kan du öppna spotify",
    "snälla öppna spotify",
    "öppna spotify tack",
    "var snäll och öppna spotify",
    "hej jarvis, öppna spotify",
])
def test_wake_phrase_and_filler_are_tolerated(matcher: ReflexMatcher, utterance: str) -> None:
    reflex = matcher.match(utterance)
    assert reflex is not None, utterance
    assert (reflex.tool, reflex.arguments) == ("open_app", {"name": "spotify"})


def test_a_question_mark_does_not_stop_a_request(matcher: ReflexMatcher) -> None:
    assert matcher.match("can you set a timer for five minutes?") is not None
    assert matcher.match("could you turn the volume up?") is not None


# --------------------------------------------------------------------------------------
# Tiers stay with the dispatcher
# --------------------------------------------------------------------------------------
def test_a_guarded_tool_is_still_returned_as_a_reflex(tmp_path: Path) -> None:
    """The matcher knows nothing about tiers - the dispatcher decides to ask."""
    path = write_grammar(tmp_path, """
version: 1
slots: {}
tools:
  power:
    replies: {en: ["Certainly, sir."], sv: ["Naturligtvis, sir."]}
    rules:
      - args: {action: shutdown}
        en: ["shut the machine down"]
        sv: ["stäng ner maskinen"]
""")
    guarded = ReflexMatcher(path)
    reflex = guarded.match("shut the machine down")
    assert reflex is not None
    assert (reflex.tool, reflex.arguments) == ("power", {"action": "shutdown"})

    from jarvis.tools import registry
    from jarvis.tools.base import Tier

    registry.load_all()
    spec = registry.get("power")
    assert spec is not None and spec.tier is Tier.GUARDED


def test_strict_mode_guards_close_app_but_the_reflex_still_matches(matcher, config) -> None:
    from jarvis.tools import registry
    from jarvis.tools.base import Tier
    from jarvis.tools.safety import effective_tier

    registry.load_all()
    config.set("assistant.safety_mode", "strict")
    spec = registry.get("close_app")
    assert spec is not None
    assert effective_tier(spec, config) is Tier.GUARDED
    assert matcher.match("close spotify").tool == "close_app"


def test_grammar_agrees_with_the_registry(grammar: dict) -> None:
    """Every tool named in the YAML exists, and every argument fits its schema."""
    from jarvis.tools import registry

    registry.load_all()
    for tool, spec in grammar["tools"].items():
        registered = registry.get(tool)
        assert registered is not None, f"{tool} is not a registered tool"
        properties = registered.parameters.get("properties", {})
        required = set(registered.parameters.get("required", []))
        for rule in spec["rules"]:
            keys = set(rule.get("args") or {})
            assert keys <= set(properties), f"{tool}: {keys - set(properties)} not in schema"
            assert required <= keys, f"{tool}: missing required {required - keys}"


# --------------------------------------------------------------------------------------
# Replies
# --------------------------------------------------------------------------------------
def test_replies_rotate_in_order_not_at_random(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("open spotify")
    summary = "Opening Spotify, sir."
    expected = [reply.replace("{result}", summary) for reply in reflex.replies]
    assert len(expected) >= 2
    spoken = [fresh.reply_for(reflex, FakeResult(summary)) for _ in range(2 * len(expected))]
    assert spoken == expected * 2             # in order, and it wraps around
    assert len(set(spoken)) == len(expected)  # never the same line twice running


def test_two_matchers_rotate_identically(fresh: ReflexMatcher) -> None:
    other = ReflexMatcher(GRAMMAR)
    reflex_a, reflex_b = fresh.match("volume up"), other.match("volume up")
    result = FakeResult("Volume up, sir.")
    assert [fresh.reply_for(reflex_a, result) for _ in range(4)] == \
           [other.reply_for(reflex_b, result) for _ in range(4)]


def test_result_placeholder_is_filled_from_the_summary(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("what time is it")
    summary = "It's twenty past three on Tuesday, sir."
    spoken = [fresh.reply_for(reflex, FakeResult(summary)) for _ in range(2)]
    assert all(summary in line for line in spoken)
    assert "{result}" not in " ".join(spoken)


def test_argument_placeholder_is_filled_and_reads_aloud(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("volume to forty")
    spoken = fresh.reply_for(reflex, FakeResult("Volume set to forty percent, sir."))
    assert "40" in spoken and "{" not in spoken


def test_a_failed_tool_never_gets_a_canned_confirmation(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("open spotify")
    failure = FakeResult("I couldn't find an app called Spotify, sir.", ok=False)
    assert fresh.reply_for(reflex, failure) == failure.summary


def test_a_refused_tool_speaks_its_own_refusal(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("close spotify")
    refusal = FakeResult("Very well, sir. Cancelled.", ok=False, refused=True)
    assert fresh.reply_for(reflex, refusal) == refusal.summary


def test_a_failure_without_a_summary_still_says_something(fresh: ReflexMatcher) -> None:
    reflex = fresh.match("open spotify")
    assert fresh.reply_for(reflex, FakeResult("", ok=False)).strip() != ""


def test_a_reply_never_contains_markdown_or_a_list(matcher: ReflexMatcher, grammar: dict) -> None:
    for tool, spec in grammar["tools"].items():
        replies = list(spec["replies"].get("en", [])) + list(spec["replies"].get("sv", []))
        for rule in spec["rules"]:
            extra = rule.get("replies") or {}
            replies += list(extra.get("en", [])) + list(extra.get("sv", []))
        for reply in replies:
            assert not any(token in reply for token in ("*", "#", "`", "- ", "\n")), f"{tool}"
            assert len(reply) < 90, f"{tool}: {reply!r} is too long to speak"


def test_reply_falls_back_to_the_summary_when_there_are_no_replies() -> None:
    reflex = Reflex(tool="volume", arguments={}, utterance="mute", template="mute", replies=[])
    matcher = ReflexMatcher(GRAMMAR)
    assert matcher.reply_for(reflex, FakeResult("Muted, sir.")) == "Muted, sir."


# --------------------------------------------------------------------------------------
# Language selection
# --------------------------------------------------------------------------------------
def test_english_only_matcher_ignores_swedish() -> None:
    english = ReflexMatcher(GRAMMAR, language="en")
    assert english.match("open spotify") is not None
    assert english.match("öppna spotify") is None


def test_swedish_only_matcher_ignores_english() -> None:
    swedish = ReflexMatcher(GRAMMAR, language="sv")
    assert swedish.match("öppna spotify") is not None
    assert swedish.match("open spotify") is None


@pytest.mark.parametrize("language", [None, "auto", "", "klingon"])
def test_an_unforced_language_matches_both(language) -> None:
    both = ReflexMatcher(GRAMMAR, language=language)
    assert both.match("open spotify") is not None
    assert both.match("öppna spotify") is not None


# --------------------------------------------------------------------------------------
# Matching is stateless
# --------------------------------------------------------------------------------------
def test_matching_is_repeatable(matcher: ReflexMatcher) -> None:
    first = matcher.match("set a timer for ten minutes")
    second = matcher.match("set a timer for ten minutes")
    assert (first.tool, first.arguments, first.template) == \
           (second.tool, second.arguments, second.template)
    assert first.arguments is not second.arguments


def test_a_custom_logger_is_used(tmp_path: Path) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def _record(self, message, *args, **kwargs) -> None:
            self.lines.append(str(message) % args if args else str(message))

        info = warning = debug = error = _record

    recorder = Recorder()
    quiet = ReflexMatcher(tmp_path / "missing.yaml", logger=recorder)
    assert quiet.available is False
    assert any("missing.yaml" in line for line in recorder.lines)


def test_no_spoken_word_was_read_as_a_boolean():
    """YAML 1.1 turns bare on/off/yes/no into booleans - the "Norway problem".

    It happened here: the app slot's deny list contained the words "on" and "off",
    and PyYAML silently handed back True and False, so those two words were never
    actually denied. The failure is invisible at runtime, which is why it gets a
    test of its own rather than a comment.
    """
    import yaml

    from jarvis.brain.reflex import DEFAULT_SENTENCES_PATH

    data = yaml.safe_load(Path(DEFAULT_SENTENCES_PATH).read_text(encoding="utf-8"))

    def booleans(node, path="root"):
        if isinstance(node, bool):
            yield path
        elif isinstance(node, list):
            for index, item in enumerate(node):
                yield from booleans(item, f"{path}[{index}]")
        elif isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, bool):
                    yield f"{path}.<boolean key>"
                yield from booleans(value, f"{path}.{key}")

    found = list(booleans(data))
    assert not found, f"quote these in sentences.yaml: {found}"


def test_a_duration_that_carries_its_own_unit_is_understood():
    """"en kvart" is already a length of time; it needs no "minutes" after it."""
    matcher = ReflexMatcher()
    for utterance, minutes in [
        ("timer på en kvart", 15),
        ("set a timer for a quarter", 15),
        ("sätt en timer på en halvtimme", 30),
        ("set a timer for half an hour", 30),
        ("väck mig om en timme", 60),
    ]:
        reflex = matcher.match(utterance)
        assert reflex is not None, utterance
        assert reflex.tool == "set_timer"
        assert reflex.arguments["minutes"] == minutes
