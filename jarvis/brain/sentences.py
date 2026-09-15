"""Streaming sentence splitting and text cleanup for speech.

The splitter is what makes JARVIS start talking before the model has finished
thinking: tokens go in, complete sentences come out, and the TTS queue picks them up
one at a time. It therefore has to be conservative — a false split mid-number sounds
far worse than a slightly late one.
"""

from __future__ import annotations

import logging
import re

__all__ = ["SentenceSplitter", "clean_for_speech"]

_log = logging.getLogger("jarvis.brain.sentences")

#: Characters that can end a sentence.
_TERMINATORS = ".!?…"
#: Closing punctuation allowed between the terminator and the following space.
_CLOSERS = "\"'”’)]}»"
#: Clause separators used to break up a run-on sentence.
_CLAUSE_CHARS = ",;:–—"

#: A buffer longer than this is cut at a clause boundary so speech can start.
SOFT_MAX_CHARS = 200
#: Absolute ceiling: cut at the last space rather than wait forever.
HARD_MAX_CHARS = 400

#: Lowercase tokens (trailing dot included) that do NOT end a sentence.
_ABBREVIATIONS = {
    # English
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "no.", "nos.", "vs.",
    "etc.", "e.g.", "i.e.", "fig.", "inc.", "ltd.", "co.", "approx.", "dept.", "est.",
    "min.", "max.", "a.m.", "p.m.", "u.s.", "u.k.", "ca.", "al.", "ph.d.", "ed.",
    "vol.", "pp.", "cf.", "sec.", "gen.", "capt.", "lt.", "sgt.",
    # Swedish
    "t.ex.", "bl.a.", "d.v.s.", "dvs.", "osv.", "m.m.", "m.fl.", "fr.o.m.", "t.o.m.",
    "kl.", "nr.", "obs.", "resp.", "enl.", "ang.", "f.d.", "s.k.", "ev.", "inkl.",
    "exkl.", "avd.", "milj.", "tel.",
}

_INITIAL_RE = re.compile(r"^[^\W\d_]\.$", re.UNICODE)


class SentenceSplitter:
    """Accumulate streamed tokens and yield complete sentences.

    Rules:

    * A sentence ends at ``.``, ``!``, ``?`` or ``…`` followed by whitespace, or at a
      newline. A terminator at the very end of the buffer is held back until the next
      token arrives, because we cannot yet tell ``3.`` + ``5`` from a full stop.
    * Abbreviations (``e.g.``, ``i.e.``, ``Mr.``, ``Dr.``, ``No.``, ``t.ex.``),
      initials (``J. Smith``), decimals (``3.5``), times (``19.30``) and ellipses do
      not split.
    * Fragments shorter than ``min_chars`` are held back and merged into the next
      sentence, so a stray ``1.`` never becomes its own utterance.
    * A run-on sentence longer than :data:`SOFT_MAX_CHARS` is cut at a comma or
      semicolon (and past :data:`HARD_MAX_CHARS` at a space), so the first audio is
      never delayed by a model that forgot to use full stops.
    """

    def __init__(self, min_chars: int = 12) -> None:
        self.min_chars = max(1, int(min_chars))
        self._buffer: str = ""
        self._scan: int = 0

    def feed(self, text: str) -> list[str]:
        """Add streamed text and return the sentences that are now complete."""
        if text is None:
            return []
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return []
        self._buffer += text
        return self._extract()

    def flush(self) -> str:
        """Return whatever is left in the buffer and clear it ("" when empty)."""
        remainder = self._buffer.strip()
        self._buffer = ""
        self._scan = 0
        return remainder

    def reset(self) -> None:
        """Drop everything buffered (barge-in, new turn)."""
        self._buffer = ""
        self._scan = 0

    # -- internals ---------------------------------------------------------------

    def _extract(self) -> list[str]:
        sentences: list[str] = []
        while True:
            cut = self._find_terminator()
            if cut is None:
                cut = self._find_overflow_cut()
                if cut is None:
                    break
            candidate = self._buffer[:cut].strip().rstrip(_CLAUSE_CHARS).strip()
            if len(candidate) >= self.min_chars:
                sentences.append(candidate)
                self._buffer = self._buffer[cut:].lstrip()
                self._scan = 0
            elif cut >= len(self._buffer):
                # Nothing left to scan and the fragment is too short: keep waiting.
                break
            else:
                # Too short to speak on its own: hold it and merge it forward.
                self._scan = cut
        return sentences

    def _find_terminator(self) -> int | None:
        """Index just past a real sentence end, or ``None`` if there is none yet."""
        buffer = self._buffer
        length = len(buffer)
        index = self._scan
        while index < length:
            char = buffer[index]
            if char == "\n":
                return index + 1
            if char in _TERMINATORS:
                if not self._ends_sentence(buffer, index):
                    index += 1
                    continue
                end = index + 1
                while end < length and buffer[end] in _TERMINATORS:
                    end += 1  # keep "..." and "?!" together
                while end < length and buffer[end] in _CLOSERS:
                    end += 1  # keep a closing quote with its sentence
                if end >= length:
                    return None  # wait for the next token before deciding
                if buffer[end].isspace():
                    return end
                index = end  # glued to more text, e.g. "3.5" or "example.com"
                continue
            index += 1
        return None

    @staticmethod
    def _ends_sentence(buffer: str, index: int) -> bool:
        """False when this ``.`` belongs to a number or a known abbreviation."""
        if buffer[index] != ".":
            return True
        following = buffer[index + 1 : index + 2]
        preceding = buffer[index - 1 : index]
        if following.isdigit() and preceding.isdigit():
            return False  # 3.5, 19.30
        start = index - 1
        while start >= 0 and (buffer[start].isalnum() or buffer[start] == "."):
            start -= 1
        token = buffer[start + 1 : index + 1].lower()
        if token in _ABBREVIATIONS:
            return False
        if _INITIAL_RE.match(token):
            return False  # "J. Smith"
        return True

    def _find_overflow_cut(self) -> int | None:
        """Cut position for a run-on sentence, or ``None`` while it is still short."""
        buffer = self._buffer
        if len(buffer.strip()) <= SOFT_MAX_CHARS:
            return None
        window = min(len(buffer), SOFT_MAX_CHARS)
        for index in range(window - 1, self.min_chars - 1, -1):
            if buffer[index] in _CLAUSE_CHARS:
                _log.debug("Splitting a run-on sentence at a clause boundary")
                return index + 1
        if len(buffer) <= HARD_MAX_CHARS:
            return None
        cut = buffer.rfind(" ", self.min_chars, HARD_MAX_CHARS)
        if cut <= 0:
            return None
        _log.debug("Splitting an unpunctuated run-on sentence at a space")
        return cut + 1


# --------------------------------------------------------------------------------------
# Text cleanup
# --------------------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"\A.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```[^\n]*\n.*?(?:```|\Z)|```[^\n]*\Z", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`+([^`\n]+)`+")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>", re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE_RE = re.compile(r"^\s{0,3}>+\s?")
_BULLET_RE = re.compile(r"^\s*[-*+•·]\s+")
_NUMBER_RE = re.compile(r"^\s*\d{1,3}[.)]\s+")
_RULE_RE = re.compile(r"^\s*(?:[-*_])(?:\s*[-*_]){2,}\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(\S(?:[^*]*\S)?)\*{1,3}")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_UNDERSCORE_RE = re.compile(r"(?<![\w\\])_{1,3}(\S(?:[^_]*\S)?)_{1,3}(?![\w])")
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols
    "\U00002600-\U000027bf"  # misc symbols and dingbats
    "\U00002b00-\U00002bff"  # arrows and stars
    "\U00002190-\U000021ff"  # arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U000020e3"             # keycap
    "\U0000200d"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)
_DEGREE_RE = re.compile(r"°\s*[cf]\b|℃|℉|°", re.IGNORECASE)
_SLASH_RE = re.compile(r"\s+/\s+")
_WS_RE = re.compile(r"[ \t ]+")
_NEWLINES_RE = re.compile(r"\s*\n\s*")


def _strip_think(text: str) -> str:
    """Remove reasoning blocks, including one that was never closed."""
    cleaned = _THINK_BLOCK_RE.sub(" ", text)
    if "</think" in cleaned.lower():
        cleaned = _THINK_CLOSE_RE.sub(" ", cleaned)
    if "<think" in cleaned.lower():
        cleaned = _THINK_OPEN_RE.sub(" ", cleaned)
    return cleaned


def _strip_line_markers(text: str) -> str:
    """Drop headers, quote marks, bullets, numbering, rules and table plumbing."""
    lines: list[str] = []
    for line in text.split("\n"):
        if _RULE_RE.match(line) or _TABLE_RULE_RE.match(line):
            continue
        line = _HEADER_RE.sub("", line)
        line = _QUOTE_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = _NUMBER_RE.sub("", line)
        if "|" in line:
            line = line.replace("|", " ")  # table cells become plain words
        lines.append(line)
    return "\n".join(lines)


def clean_for_speech(text: str) -> str:
    """Turn model output into something a TTS engine can read out loud.

    Removes ``<think>`` blocks (even an unterminated one), fenced code blocks,
    markdown emphasis, headers, list markers and link targets (the label is kept),
    emoji and pictographs; collapses whitespace; and expands a few symbols that
    otherwise get mangled: ``%`` -> " percent", ``&`` -> " and ", ``°C`` -> " degrees".
    A spaced slash is reduced to a space, since reading it aloud is language
    specific. Everything else is left alone on purpose — this must stay conservative
    and work for both English and Swedish.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _strip_think(text)
    cleaned = _FENCE_RE.sub(" ", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _IMAGE_RE.sub(r"\1", cleaned)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _AUTOLINK_RE.sub(r"\1", cleaned)
    cleaned = _strip_line_markers(cleaned)
    cleaned = _STRIKE_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _UNDERSCORE_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("*", "").replace("`", "")
    cleaned = _EMOJI_RE.sub(" ", cleaned)
    cleaned = _DEGREE_RE.sub(" degrees", cleaned)
    cleaned = cleaned.replace("%", " percent").replace("&", " and ")
    cleaned = _SLASH_RE.sub(" ", cleaned)
    cleaned = _NEWLINES_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()
