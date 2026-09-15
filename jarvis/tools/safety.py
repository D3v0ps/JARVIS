"""Safety layer: command blocklist, tier escalation and spoken yes/no parsing.

Everything here is pure Python and pre-compiled at import time, because
:func:`check_blocked` runs on every single tool call and must stay cheap.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable, Pattern

from jarvis.tools.base import Tier, ToolSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import Config

__all__ = [
    "BLOCKLIST_PATTERNS",
    "check_blocked",
    "effective_tier",
    "STRICT_GUARDED_TOOLS",
    "CONFIRM_WORDS",
    "CANCEL_WORDS",
    "is_confirmation",
    "is_cancellation",
]

_log = logging.getLogger("jarvis.tools.safety")

# Shared fragments. `\s+` everywhere so extra whitespace never smuggles a command past
# us, and every pattern is compiled with re.IGNORECASE.
_DEL = r"(?:del|erase|rd|rmdir|rm|ri|remove-item)"          # cmd + PowerShell aliases
_HKLM = r"(?:hklm|hkey_local_machine)"                       # both registry spellings
_DEFENDER_KEY = r"software\\+policies\\+microsoft\\+windows\s*defender"
_SERVICES_KEY = r"system\\+currentcontrolset\\+services"

#: (regex source, human-readable reason). The first match wins in :func:`check_blocked`.
BLOCKLIST_PATTERNS: list[tuple[str, str]] = [
    # --- Destroying or repartitioning a disk --------------------------------------
    # "format C:", "format.com /fs:ntfs D:" — a drive letter is required so ordinary
    # prose such as "format the text in this file" is not caught.
    (r"\bformat(?:\.com)?\s+(?:/\S+\s+)*[a-z]:", "formatting a drive"),
    # diskpart scripts can wipe partitions without further confirmation.
    (r"\bdiskpart(?:\.exe)?\b", "running diskpart"),
    # Boot configuration edits can make Windows unbootable.
    (r"\bbcdedit(?:\.exe)?\b", "editing the boot configuration"),
    (r"\bbootrec(?:\.exe)?\b", "rewriting the boot record"),

    # --- Registry attacks on security and Defender policy -------------------------
    # reg add / reg delete against HKLM service or Defender policy keys.
    (
        rf"\breg(?:\.exe)?\s+(?:add|delete)\s+[^\n]*{_HKLM}:?\\+"
        rf"(?:{_SERVICES_KEY}|{_DEFENDER_KEY})",
        "editing protected registry keys",
    ),
    # The PowerShell equivalent, e.g. Set-ItemProperty HKLM:\SOFTWARE\Policies\...
    (
        rf"\b(?:set|new|remove)-itemproperty\b[^\n]*{_HKLM}:?\\+"
        rf"(?:{_SERVICES_KEY}|{_DEFENDER_KEY})",
        "editing protected registry keys",
    ),
    # Any write to the Defender policy key, whichever tool is used.
    (rf"\b(?:new|remove)-item\b[^\n]*{_HKLM}:?\\+{_DEFENDER_KEY}", "editing Defender policy"),

    # --- Disabling Microsoft Defender ---------------------------------------------
    # Set-/Add-/Remove-MpPreference tune or switch off real-time protection.
    (r"\b(?:set|add|remove)-mppreference\b", "changing Defender settings"),
    # Stopping the Defender service via Stop-Service, sc stop or net stop.
    (
        r"\b(?:stop-service|sc(?:\.exe)?\s+(?:stop|config)|net\s+stop)\b[^\n]*\bwindefend\b",
        "stopping Defender",
    ),
    (r"\bstop-mpscan\b|\bdisable-windowsoptionalfeature\b[^\n]*defender", "disabling Defender"),

    # --- Disabling the firewall ----------------------------------------------------
    # netsh advfirewall set allprofiles state off / netsh firewall set opmode disable
    (
        r"\bnetsh\b[^\n]*\b(?:adv)?firewall\b[^\n]*\b(?:off|disable|disabled)\b",
        "turning the firewall off",
    ),
    # Set-NetFirewallProfile -Enabled False / $false
    (
        r"\bset-netfirewallprofile\b[^\n]*-enabled\s+\$?false",
        "turning the firewall off",
    ),

    # --- Credential harvesting ------------------------------------------------------
    # cmdkey lists and writes stored Windows credentials; vaultcmd reads the vault.
    (r"\bcmdkey(?:\.exe)?\b", "reading or writing stored credentials"),
    (r"\bvaultcmd(?:\.exe)?\b", "reading the credential vault"),
    # Interactive credential prompts driven by a script, and outright dumpers.
    (r"\bget-credential\b", "harvesting credentials"),
    (r"\bmimikatz\b|\bsekurlsa\b|\binvoke-mimikatz\b", "running a credential dumper"),
    (r"\b(?:procdump|rundll32)[^\n]*\blsass\b|\blsass\b[^\n]*\bdump\b", "dumping LSASS memory"),

    # --- Destroying backups and evidence --------------------------------------------
    # vssadmin delete shadows removes every restore point (classic ransomware step).
    (r"\bvssadmin(?:\.exe)?\b[^\n]*\bdelete\b[^\n]*\bshadow", "deleting shadow copies"),
    (r"\bwmic\b[^\n]*\bshadowcopy\b[^\n]*\bdelete\b", "deleting shadow copies"),
    # cipher /w overwrites free space, destroying anything recoverable.
    (r"\bcipher(?:\.exe)?\s+/w", "wiping free disk space"),
    # wevtutil cl / Clear-EventLog erase the Windows event logs.
    (r"\bwevtutil(?:\.exe)?\s+(?:cl|clear-log)\b", "wiping the event logs"),
    (r"\bclear-eventlog\b", "wiping the event logs"),

    # --- Deleting Windows itself -----------------------------------------------------
    # Any delete alias aimed at C:\Windows (which covers C:\Windows\System32).
    # The trailing separator matters: without it the \b matches before a hyphen and
    # an ordinary folder such as D:\windows-backup is refused as if it were C:\Windows.
    (
        rf"\b{_DEL}\b[^\n]*[a-z]:\\+windows(?:\\|\s|\"|'|$)",
        "deleting files inside the Windows directory",
    ),
    # ... or at System32 written relative / via %SystemRoot% / $env:windir.
    (rf"\b{_DEL}\b[^\n]*\bsystem32\b", "deleting files inside System32"),
    # ... or at a whole drive: "del C:\*", "Remove-Item D:\ -Recurse".
    (rf"\b{_DEL}\b[^\n]*\s[a-z]:\\+\*", "deleting the contents of a drive root"),
    (rf"\b{_DEL}\b[^\n]*\s[a-z]:\\+(?=[\s\"']|$)", "deleting a drive root"),
    # The POSIX classic, in case a shell tool ever runs on something else.
    (r"\brm\s+-[a-z]*r[a-z]*f?\s+/(?:\s|$)", "recursively deleting the filesystem root"),

    # --- Power state from a raw shell -------------------------------------------------
    # shutdown /s, shutdown -r, Stop-Computer: the `power` tool exists for this and
    # asks for confirmation; a bare shell command must not bypass it.
    (r"\bshutdown(?:\.exe)?\s+[/-][srghp]\b", "shutting the machine down from a shell"),
    (r"\b(?:stop|restart)-computer\b", "shutting the machine down from a shell"),
]

# Pre-compiled once at import: check_blocked() is on the hot path of every tool call.
_COMPILED: list[tuple[Pattern[str], str]] = []
for _source, _reason in BLOCKLIST_PATTERNS:
    try:
        _COMPILED.append((re.compile(_source, re.IGNORECASE), _reason))
    except re.error as exc:  # pragma: no cover - a typo here must be loud, not fatal
        _log.error("Invalid blocklist pattern %r (%s); it will not be enforced", _source, exc)


def check_blocked(text: str) -> str | None:
    """Return the reason for the first blocklist match, or ``None`` when clean."""
    if not text:
        return None
    if not isinstance(text, str):
        text = str(text)
    for pattern, reason in _COMPILED:
        if pattern.search(text):
            _log.debug("Blocklist hit (%s) on pattern %s", reason, pattern.pattern)
            return reason
    return None


#: Tools that are only "announced" normally but must be confirmed in strict mode.
STRICT_GUARDED_TOOLS = {"close_app", "type_text"}


def _cfg_value(cfg: Any, dotted: str, default: Any) -> Any:
    """Read a dotted key from a :class:`Config` or a plain nested mapping."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter) and not isinstance(cfg, Mapping):
        try:
            value = getter(dotted, default)
        except Exception as exc:  # noqa: BLE001 - a broken config must not block safety
            _log.warning("Could not read %s from the config (%s); using %r", dotted, exc, default)
            return default
        return default if value is None else value
    if isinstance(cfg, Mapping):
        node: Any = cfg
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return default if node is None else node
    _log.warning("Unsupported config object %r; using %r for %s", type(cfg).__name__, default, dotted)
    return default


def effective_tier(spec: ToolSpec, cfg: "Config") -> Tier:
    """The tier a tool actually runs at, after applying ``assistant.safety_mode``.

    In ``strict`` mode every ANNOUNCED tool is upgraded to GUARDED, and
    ``close_app`` / ``type_text`` are guarded as well. In ``normal`` mode the spec's
    own tier is used unchanged.
    """
    tier = spec.tier if isinstance(spec.tier, Tier) else Tier(str(spec.tier))
    mode = str(_cfg_value(cfg, "assistant.safety_mode", "normal")).strip().lower()
    if mode != "strict":
        return tier
    if tier is Tier.ANNOUNCED or spec.name in STRICT_GUARDED_TOOLS:
        return Tier.GUARDED
    return tier


# --------------------------------------------------------------------------------------
# Spoken confirmation / cancellation
# --------------------------------------------------------------------------------------

CONFIRM_WORDS = {
    "confirm", "yes", "yeah", "do it", "go ahead", "proceed", "affirmative",
    "kör", "ja", "gör det", "kör på", "absolut",
}
CANCEL_WORDS = {"no", "cancel", "stop", "abort", "nej", "avbryt", "stopp"}

# Extra everyday phrasings, kept separate so the contract sets above stay verbatim.
_CONFIRM_EXTRA = {
    "ok", "okay", "sure", "yep", "yup", "please do", "go on", "carry on", "correct",
    "certainly", "definitely", "confirmed", "ja tack", "javisst", "visst", "okej",
    "kör hårt", "fortsätt", "gör så", "japp", "jajamän",
}
_CANCEL_EXTRA = {
    "nope", "nah", "no thanks", "never mind", "nevermind", "forget it", "hold on",
    "wait", "negative", "dont", "do not", "dont do it", "stop it", "leave it",
    "nej tack", "glöm det", "sluta", "vänta", "låt bli", "skippa det", "inte",
}
# Words that flip a confirmation into a refusal: "no, go ahead" is a cancellation.
_NEGATIONS = {"dont", "not", "never", "inte", "aldrig", "icke", "nope", "nah"}

_ALL_CONFIRM = CONFIRM_WORDS | _CONFIRM_EXTRA
_ALL_CANCEL = CANCEL_WORDS | _CANCEL_EXTRA

#: An utterance longer than this is only a confirmation if it *starts* with one.
_SHORT_UTTERANCE_WORDS = 5

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> list[str]:
    """Lowercase, drop punctuation and return the word list ("no, don't" -> no dont)."""
    if not text:
        return []
    if not isinstance(text, str):
        text = str(text)
    lowered = unicodedata.normalize("NFC", text).lower()
    lowered = lowered.replace("'", "").replace("\u2019", "")  # don't -> dont
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", lowered).strip().split()


def _contains_phrase(words: list[str], phrase: str) -> bool:
    """True when ``phrase`` appears as a whole word sequence inside ``words``."""
    parts = phrase.split()
    if not parts or len(parts) > len(words):
        return False
    for index in range(len(words) - len(parts) + 1):
        if words[index : index + len(parts)] == parts:
            return True
    return False


def _matches_any(words: list[str], phrases: Iterable[str]) -> bool:
    return any(_contains_phrase(words, phrase) for phrase in phrases)


def _starts_with_any(words: list[str], phrases: Iterable[str]) -> bool:
    for phrase in phrases:
        parts = phrase.split()
        if parts and words[: len(parts)] == parts:
            return True
    return False


def is_cancellation(text: str) -> bool:
    """True when the user said no, stop, or negated a confirmation.

    A negated confirmation such as "no, don't" or "nej, avbryt" counts as a
    cancellation even though it contains a confirm word.
    """
    words = _normalise(text)
    if not words:
        return False
    if _matches_any(words, _ALL_CANCEL):
        return True
    # "don't do it", "gör inte det": a negation next to a confirmation is a refusal.
    if any(word in _NEGATIONS for word in words) and _matches_any(words, _ALL_CONFIRM):
        return True
    return False


def is_confirmation(text: str) -> bool:
    """True only when the user clearly said yes to the pending guarded action.

    Guard rails: a cancellation always wins, and a long sentence that merely happens
    to contain "yes" ("I said yes to that meeting yesterday, anyway ...") does not
    count — the utterance must be short (<= 5 words) or begin with a confirm word.
    """
    words = _normalise(text)
    if not words:
        return False
    if is_cancellation(text):
        return False
    if not _matches_any(words, _ALL_CONFIRM):
        return False
    if len(words) <= _SHORT_UTTERANCE_WORDS:
        return True
    return _starts_with_any(words, _ALL_CONFIRM)
