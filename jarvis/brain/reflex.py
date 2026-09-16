"""The command grammar underneath the model.

Nine utterances in ten are imperatives - "volume to forty", "open Spotify" - and
each costs a full model turn plus a chance the model narrates instead of acting.
:class:`ReflexMatcher` compiles the templates in ``prompts/sentences.yaml`` into
plain regular expressions at start-up and answers in well under a millisecond, so
the tool reaches the *same* dispatcher about a second and a half earlier. Nothing
here looks at a tool's tier, so a GUARDED tool matched by a reflex comes back like
any other and the dispatcher still asks before it runs.

Matching is deliberately strict: a template must consume the WHOLE utterance, a
conjunction always goes to the model, and an unresolvable slot means no match.
Being too permissive takes the model out of sentences that needed it, which costs
far more than a missed fast path. Pure stdlib plus PyYAML - it imports on Linux.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from jarvis.core.logging import get_logger

__all__ = ["Reflex", "ReflexMatcher", "DEFAULT_SENTENCES_PATH"]

#: Where the grammar lives, relative to the working directory.
DEFAULT_SENTENCES_PATH = Path("prompts/sentences.yaml")

#: Longer than this and it is a sentence, not a command. Cheap early exit.
MAX_WORDS = 16

#: Spoken when a tool failed without saying why, so a reply never invents success.
FAILURE_FALLBACK = "That didn't work, sir."

#: A number slot captures up to five words ("a quarter of an hour") and is resolved
#: against the values map afterwards. Spelling every spoken number into the regex
#: would compile five times slower and be no more accurate: a phrase that is not a
#: number fails in :meth:`_Slot.resolve` and the utterance falls through anyway.
_NUMBER_PATTERN = r"\S+(?:\s+\S+){0,4}"

#: Characters kept in an utterance; punctuation and emoji are dropped. A template
#: also keeps ``{}`` for its slots and ``[]`` for its optional words.
_KEEP_ALWAYS = "+#&%"
_SPACERS = {"-", "_", "/", "\\"}

#: Two commands in one sentence belong to the model, not to a template.
_CONJUNCTION_RE = re.compile(
    r"(?:^|\s)(?:and|then|also|plus|afterwards|after that|as well as"
    r"|och|samt|sedan|sen|därefter|efteråt|efter det)(?:\s|$)"
)

#: A leading "jarvis," or "hey jarvis," is address, not part of the command.
_WAKE_RE = re.compile(r"^(?:hey|hi|ok|okay|okey|hej|hallå|yo)?\s*jarvis(?:\s+|$)")

#: Politeness carrying no meaning, stripped from the front. Longest phrase first.
_PREFIX_FILLERS = (
    "i would like you to", "kan du vara snäll och", "do me a favour and",
    "do me a favor and", "i want you to", "i need you to", "would you please",
    "could you please", "can you please", "skulle du kunna", "jag vill att du",
    "var snäll och", "är du snäll och", "go ahead and", "be a dear and",
    "could you", "would you", "will you", "can you", "vill du", "kan du",
    "please", "snälla", "kindly", "just", "bara",
)

#: The same at the end of the utterance.
_SUFFIX_FILLERS = (
    "tack så mycket", "är du snäll", "thank you", "right now", "for me",
    "please", "thanks", "snälla", "genast", "direkt", "today", "i dag",
    "idag", "sir", "tack", "now", "nu", "då", "va", "ok", "okay",
)


def _normalise(text: str, *, keep: str = "") -> str:
    """Lowercase, drop punctuation, collapse whitespace. Templates and utterances
    both go through this, so ``"what's the time"`` in the YAML and ``"What's the
    time?"`` from Whisper meet as ``whats the time``."""
    if not isinstance(text, str):
        return ""
    allowed = set(_KEEP_ALWAYS) | set(keep)
    out: list[str] = []
    for char in text.lower():
        if char.isalnum() or char in allowed:
            out.append(char)
        elif char.isspace() or char in _SPACERS:
            out.append(" ")
    return " ".join("".join(out).split())


def _strip_phrase(text: str, phrases: Iterable[str], *, prefix: bool) -> str:
    """Remove one leading or trailing filler phrase, or return the text unchanged."""
    for phrase in phrases:
        if text == phrase:
            return ""
        if prefix and text.startswith(phrase + " "):
            return text[len(phrase) + 1:]
        if not prefix and text.endswith(" " + phrase):
            return text[: -len(phrase) - 1]
    return text


def _tidy_number(value: float) -> float | int:
    """``10.0`` reads and logs better as ``10``."""
    return int(value) if float(value).is_integer() else float(value)


def _as_number(value: Any) -> float | None:
    """A bound from the YAML, or ``None`` when it is missing or nonsense."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fill(text: str, data: dict[str, Any]) -> tuple[str, bool]:
    """Substitute every ``{name}`` from ``data``; the flag says one had no value.
    A whole float drops its ``.0``, so a reply says "forty", not "forty point zero"."""
    missing = False

    def replace(found: re.Match[str]) -> str:
        nonlocal missing
        value = data.get(found.group(1))
        if value is None or (isinstance(value, str) and not value.strip()):
            missing = True
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return re.sub(r"\{(\w+)\}", replace, text), missing


@dataclass(frozen=True)
class _Slot:
    """One ``{placeholder}`` a template may contain."""

    name: str
    kind: str                      # "int" | "float" | "str"
    pattern: str                   # the regex fragment that captures it
    values: dict[str, Any] = field(default_factory=dict)
    minimum: float | None = None
    maximum: float | None = None
    deny_first_word: frozenset[str] = frozenset()

    def resolve(self, raw: str) -> Any | None:
        """Turn the captured text into an argument value; ``None`` rejects the match."""
        text = " ".join(raw.split())
        if not text:
            return None
        if self.kind in ("int", "float"):
            number = self._to_number(text)
            if number is None:
                return None
            if self.minimum is not None and number < self.minimum:
                return None
            if self.maximum is not None and number > self.maximum:
                return None
            return int(number) if self.kind == "int" else _tidy_number(number)
        # An article or a particle in front means the matcher is over-reaching:
        # "stäng av datorn" is a shutdown, not close_app("av datorn").
        if text.split()[0] in self.deny_first_word:
            return None
        mapped = self.values.get(text) or self.values.get(text.replace(" ", ""))
        return str(mapped) if mapped else text

    def _to_number(self, text: str) -> float | None:
        """Digits, a spoken number, or ``None``. ``"tjugo ett"`` folds to ``tjugoett``."""
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{1,2})?", text):
            return float(text.replace(",", "."))
        for key in (text, text.replace(" ", "")):
            if key in self.values:
                try:
                    return float(self.values[key])
                except (TypeError, ValueError):
                    return None
        return None


@dataclass(frozen=True)
class _Template:
    """One compiled sentence: its regex, its slots and the call it produces."""

    tool: str
    text: str                                # as written in the YAML, for the log
    language: str
    regex: re.Pattern[str]
    groups: tuple[tuple[str, _Slot], ...]    # (regex group name, slot)
    args: dict[str, Any]
    replies: tuple[str, ...]


@dataclass
class Reflex:
    """A matched command, ready for the dispatcher."""

    tool: str
    arguments: dict
    utterance: str      # the cleaned text that actually matched, for the log
    template: str       # which template matched, for the log
    replies: list[str]  # canned confirmations to choose between


class ReflexMatcher:
    """Compiles ``prompts/sentences.yaml`` and matches whole utterances against it."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SENTENCES_PATH,
        *,
        language: str | None = None,
        logger: Any = None,
    ) -> None:
        self._log = logger or get_logger("brain.reflex")
        self._path = Path(path)
        self._language = self._pick_language(language)
        self._templates: list[_Template] = []
        self._rotation: dict[tuple[Any, ...], int] = {}
        self._lock = threading.Lock()
        self._load()

    @property
    def available(self) -> bool:
        """True when at least one template compiled, so the fast path can be used."""
        return bool(self._templates)

    def templates(self) -> int:
        """How many templates are loaded — one line in the boot log."""
        return len(self._templates)

    def match(self, text: str) -> Reflex | None:
        """Whole-utterance match only. A conjunction, a sentence rather than a
        command, or one word the template did not consume all mean ``None``: "open
        Spotify and tell me the weather" has to reach the model."""
        if not self._templates or not isinstance(text, str):
            return None
        cleaned = self.prepare(text)
        if not cleaned or len(cleaned.split()) > MAX_WORDS:
            return None
        if _CONJUNCTION_RE.search(cleaned):
            self._log.debug("Reflex declined %r: more than one command", cleaned)
            return None
        for template in self._templates:
            matched = template.regex.fullmatch(cleaned)
            if matched is None:
                continue
            values: dict[str, Any] = {}
            for group, slot in template.groups:
                resolved = slot.resolve(matched.group(group))
                if resolved is None:
                    values = {}
                    break
                values[slot.name] = resolved
            if len(values) != len(template.groups):
                continue
            arguments = self._build_arguments(template, values)
            if arguments is None:
                continue
            self._log.info("Reflex matched %r -> %s(%s) via %r",
                           cleaned, template.tool, arguments, template.text)
            return Reflex(tool=template.tool, arguments=arguments, utterance=cleaned,
                          template=template.text, replies=list(template.replies))
        return None

    def reply_for(self, reflex: Reflex, result: Any) -> str:
        """The line to speak once the tool has run.

        A failed or refused tool speaks its own summary: a canned confirmation must
        never claim something that did not happen. Otherwise the replies rotate in
        order - not at random, so a test can pin it down - and ``{result}`` is the
        tool's summary.
        """
        summary = str(getattr(result, "summary", "") or "").strip()
        if not bool(getattr(result, "ok", True)) or bool(getattr(result, "refused", False)):
            return summary or FAILURE_FALLBACK
        replies = [str(reply) for reply in (reflex.replies or []) if str(reply).strip()]
        if not replies:
            return summary
        key = (reflex.tool, tuple(replies))
        with self._lock:
            index = self._rotation.get(key, 0)
            self._rotation[key] = (index + 1) % len(replies)
        rendered, _ = _fill(replies[index], {**dict(reflex.arguments or {}), "result": summary})
        rendered = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(rendered.split()))
        return rendered.strip(" ,;:-") or summary

    def prepare(self, text: str) -> str:
        """Normalise an utterance and strip the wake phrase and the politeness."""
        cleaned = _WAKE_RE.sub("", _normalise(text), count=1).strip()
        previous = None
        while cleaned and cleaned != previous:
            previous = cleaned
            cleaned = _strip_phrase(cleaned, _PREFIX_FILLERS, prefix=True)
            cleaned = _strip_phrase(cleaned, _SUFFIX_FILLERS, prefix=False)
        return cleaned

    def _pick_language(self, language: str | None) -> str | None:
        """``None`` means "match every language"; anything unknown means the same."""
        value = (language or "").strip().lower()
        if value.startswith("en"):
            return "en"
        if value.startswith("sv"):
            return "sv"
        if value and value not in ("auto", "any", "all"):
            self._log.debug("Unknown reflex language %r; matching every language", language)
        return None

    def _load(self) -> None:
        """Read and compile the grammar. A broken file costs the fast path, nothing more."""
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            self._log.warning("No usable reflex grammar at %s (%s); every utterance "
                              "goes to the model", self._path, exc)
            return
        if not isinstance(data, dict):
            self._log.warning("Reflex grammar %s is not a mapping; ignoring it", self._path)
            return
        try:
            self._compile_tools(data.get("tools"), self._build_slots(data))
        except Exception as exc:  # noqa: BLE001 - a bad grammar must not stop JARVIS
            self._log.warning("Reflex grammar %s could not be compiled: %s", self._path, exc)
            self._templates = []
            return
        self._log.info("Reflex grammar ready: %d template(s) from %s",
                       len(self._templates), self._path)

    def _build_slots(self, data: dict) -> dict[str, _Slot]:
        """Turn the ``slots:`` section into compiled :class:`_Slot` objects."""
        raw_values = data.get("values") if isinstance(data.get("values"), dict) else {}
        maps: dict[str, dict[str, Any]] = {}
        for name, mapping in raw_values.items():
            if isinstance(mapping, dict):
                maps[str(name)] = {_normalise(str(key)): value
                                   for key, value in mapping.items() if _normalise(str(key))}
        slots: dict[str, _Slot] = {}
        raw_slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
        for name, spec in raw_slots.items():
            if not isinstance(spec, dict):
                continue
            kind = str(spec.get("type", "str")).lower()
            kind = kind if kind in ("int", "float", "str") else "str"
            max_words = max(1, int(spec.get("max_words", 4)))
            slots[str(name)] = _Slot(
                name=str(name),
                kind=kind,
                pattern=(_NUMBER_PATTERN if kind in ("int", "float")
                         else rf"\S+(?:\s+\S+){{0,{max_words - 1}}}"),
                values=maps.get(str(spec.get("values", "")), {}),
                minimum=_as_number(spec.get("min")),
                maximum=_as_number(spec.get("max")),
                deny_first_word=frozenset(
                    _normalise(str(word)) for word in (spec.get("deny_first_word") or [])
                    if _normalise(str(word))
                ),
            )
        return slots

    def _compile_tools(self, tools: Any, slots: dict[str, _Slot]) -> None:
        """Compile every rule of every tool, in the order the file lists them."""
        if not isinstance(tools, dict):
            self._log.warning("Reflex grammar %s has no tools section", self._path)
            return
        for tool_name, spec in tools.items():
            if not isinstance(spec, dict):
                continue
            shared = spec.get("replies") if isinstance(spec.get("replies"), dict) else {}
            for rule in spec.get("rules") or []:
                if not isinstance(rule, dict):
                    continue
                args = rule.get("args") if isinstance(rule.get("args"), dict) else {}
                replies = rule.get("replies") if isinstance(rule.get("replies"), dict) else shared
                for language in ("en", "sv"):
                    if self._language is not None and language != self._language:
                        continue
                    for text in rule.get(language) or []:
                        compiled = self._compile_template(
                            str(tool_name), str(text), language, args,
                            replies.get(language) or [], slots,
                        )
                        if compiled is not None:
                            self._templates.append(compiled)

    def _compile_template(
        self, tool: str, text: str, language: str, args: dict,
        replies: Any, slots: dict[str, _Slot],
    ) -> _Template | None:
        """Build one regex. An unknown slot drops the template rather than the file."""
        pieces: list[str] = []
        groups: list[tuple[str, _Slot]] = []
        for index, token in enumerate(_normalise(text, keep="{}[]").split()):
            optional = False
            if token.startswith("{") and token.endswith("}"):
                slot = slots.get(token[1:-1])
                if slot is None:
                    self._log.warning("Template %r of %s uses unknown slot %s; skipped",
                                      text, tool, token)
                    return None
                group = f"s{len(groups)}"
                groups.append((group, slot))
                piece = f"(?P<{group}>{slot.pattern})"
            elif token.startswith("[") and token.endswith("]"):
                if not token[1:-1]:
                    continue
                optional = index > 0   # a leading optional word would double a separator
                piece = re.escape(token[1:-1])
            else:
                piece = re.escape(token)
            # Separators are mandatory whitespace, so "open {app}" cannot match
            # "opened files" by splitting a word in half.
            pieces.append(piece if not pieces
                          else rf"(?:\s+{piece})?" if optional else rf"\s+{piece}")
        if not pieces:
            return None
        try:
            regex = re.compile("".join(pieces), re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            self._log.warning("Template %r of %s is not a valid pattern: %s", text, tool, exc)
            return None
        return _Template(
            tool=tool, text=text, language=language, regex=regex, groups=tuple(groups),
            args=dict(args),
            replies=tuple(str(reply) for reply in replies if str(reply).strip()),
        )

    def _build_arguments(self, template: _Template, values: dict[str, Any]) -> dict | None:
        """Fill the rule's ``args`` from the captured slots, typed as the schema wants."""
        arguments: dict[str, Any] = {}
        for key, spec in template.args.items():
            if isinstance(spec, dict):      # {slot: hours, scale: 60} -> minutes
                captured = values.get(str(spec.get("slot", "")))
                try:
                    arguments[str(key)] = _tidy_number(
                        float(captured) * float(spec.get("scale", 1))  # type: ignore[arg-type]
                    )
                except (TypeError, ValueError):
                    return None
                continue
            if isinstance(spec, str) and "{" in spec:
                inner = spec.strip()
                if inner.startswith("{") and inner.endswith("}") and inner[1:-1] in values:
                    arguments[str(key)] = values[inner[1:-1]]   # keeps the slot's type
                    continue
                rendered, missing = _fill(inner, values)
                if missing:
                    return None
                arguments[str(key)] = rendered
                continue
            arguments[str(key)] = spec
        return arguments
