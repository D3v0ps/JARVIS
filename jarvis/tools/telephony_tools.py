"""Telephony: placing a call from the desk, and being honest about what happened.

Two ways a call can leave this machine, and the spoken summary always says which one
ran. **The Android path (the Samsung), over ADB:** ``am start -a
android.intent.action.CALL -d tel:+46...`` places the call outright, and when a build
refuses that intent for the ``shell`` uid — a ``SecurityException`` in the output, or a
non-zero exit — the attempt falls back to ``ACTION_DIAL`` plus ``input keyevent
KEYCODE_CALL``, which only puts the number on the phone's screen. The summary says
exactly that instead of claiming the line is ringing. **The Windows handoff (the
iPhone):** with no phone on the cable, ``os.startfile("tel:+46...")`` hands the number
to whatever Windows registered for ``tel:``, usually Phone Link, which dials nothing by
itself, so JARVIS says plainly that the call has to be started on the phone.

**JARVIS never speaks on the call.** Windows exposes no supported way to put audio
into a mobile call; the workarounds route a virtual cable through the Bluetooth
hands-free profile, which drops the voice to 8 kHz and feeds the speakers straight
back into the line as echo. He looks the number up, dials, and goes quiet. Do not add
audio injection here later — this paragraph is why it is missing.

Nothing Windows-only is imported at module level: ``adb`` is invoked through
:mod:`subprocess` and ``os.startfile`` is looked up with :func:`getattr`, so this
module imports cleanly on a bare Linux box.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["dial_number", "end_call", "adb_available", "attached_devices", "call_state",
           "reset_adb_cache", "normalise_number", "spoken_number", "CALL_STATES",
           "EMERGENCY_NUMBERS", "ADB_CACHE_SECONDS", "CALL_ACTION", "DIAL_ACTION"]

_log = get_logger("tools.telephony")

#: The executable and the time each kind of call may take.
ADB = "adb"
ADB_TIMEOUT = 10.0
ADB_DEVICES_TIMEOUT = 6.0

#: How long an availability answer is trusted. Short: a cable gets unplugged
#: mid-session and the very next turn has to notice.
ADB_CACHE_SECONDS = 5.0

#: Keeps a console window from flashing up when ``adb`` starts on Windows.
CREATE_NO_WINDOW = 0x08000000

#: The two Android intents, and the keys for the green and red buttons.
CALL_ACTION = "android.intent.action.CALL"
DIAL_ACTION = "android.intent.action.DIAL"
CALL_KEY = "KEYCODE_CALL"
ENDCALL_KEY = "KEYCODE_ENDCALL"

#: ``mCallState`` in ``dumpsys telephony.registry`` -> the word this module returns.
CALL_STATES: dict[int, str] = {0: "idle", 1: "ringing", 2: "offhook"}

#: Exit codes for ``adb`` calls that never reached the executable.
ADB_MISSING, ADB_TIMED_OUT, ADB_ERROR = -1, -2, -3

#: Stands in for a name so the guarded announcement still reads as a sentence.
DEFAULT_WHO = "that number"

#: Never dialled from here: JARVIS cannot speak on the call, and a silent line to an
#: emergency operator is worse than no call at all.
EMERGENCY_NUMBERS = frozenset({"112", "911", "999", "000", "110", "118", "119", "108"})

#: Country calling codes for the fallback used when ``phonenumbers`` is absent.
COUNTRY_CODES: dict[str, str] = {
    "SE": "46", "NO": "47", "DK": "45", "FI": "358", "DE": "49", "GB": "44",
    "NL": "31", "FR": "33", "ES": "34", "IT": "39", "PL": "48", "US": "1", "CA": "1",
}

_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine")
_MCALL_STATE_RE = re.compile(r"mCallState\s*=\s*(\d+)")
_LIVE_STATES = ("ringing", "offhook")

#: Output meaning "the phone refused the shell uid this intent".
_REFUSAL_MARKERS = ("securityexception", "permission denial",
                    "android.permission.call_phone", "requires permission")

#: Output meaning the phone is simply not on the cable any more.
_NO_DEVICE_MARKERS = ("device not found", "no devices/emulators found",
                      "device unauthorized", "device offline", "device still connecting")

_cache_lock = threading.Lock()
_availability: tuple[float, bool] | None = None


# --- running adb ------------------------------------------------------------------------
def _popen_kwargs() -> dict[str, Any]:
    """Extra ``subprocess`` keywords: hide the console window on Windows only."""
    return {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def _run_adb(args: list[str], timeout: float = ADB_TIMEOUT) -> tuple[int, str]:
    """Run one ``adb`` command, returning ``(exit code, stdout and stderr)``.

    Never raises: a missing executable, a timeout or an OS error come back as a
    negative sentinel with a readable line in place of the output.
    """
    executable = shutil.which(ADB)
    if not executable:
        return ADB_MISSING, "adb is not installed or not on PATH"
    try:
        completed = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", **_popen_kwargs(),
        )
    except FileNotFoundError:
        return ADB_MISSING, "adb disappeared between the PATH lookup and the call"
    except subprocess.TimeoutExpired:
        return ADB_TIMED_OUT, f"adb did not answer within {timeout:g} seconds"
    except OSError as exc:
        return ADB_ERROR, f"{type(exc).__name__}: {exc}"
    parts = [(completed.stdout or "").strip(), (completed.stderr or "").strip()]
    return int(completed.returncode), "\n".join(part for part in parts if part)


def _says(output: str, markers: tuple[str, ...]) -> bool:
    """True when any marker appears in ``output``, case-insensitively."""
    text = (output or "").lower()
    return any(marker in text for marker in markers)


def _device_gone(code: int, output: str) -> bool:
    """True when ``adb`` ran but no phone was on the other end of the cable."""
    return code in (ADB_MISSING, ADB_ERROR) or _says(output, _NO_DEVICE_MARKERS)


def reset_adb_cache() -> None:
    """Forget the cached availability answer; the next check really runs ``adb``."""
    global _availability
    with _cache_lock:
        _availability = None


def attached_devices() -> list[str]:
    """Serials of the phones ``adb devices`` reports as ready.

    Only a line whose last field is exactly ``device`` counts: an ``unauthorized``
    phone (the USB-debugging prompt is still up) or an ``offline`` one cannot be
    dialled from.
    """
    code, output = _run_adb(["devices"], ADB_DEVICES_TIMEOUT)
    if code != 0:
        _log.debug("adb devices failed (exit %s): %s", code, output)
        return []
    serials: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.startswith("*")
                or stripped.lower().startswith("list of devices")):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            continue
        if fields[-1] == "device":
            serials.append(fields[0])
        else:
            _log.debug("Ignoring phone %s in state %r", fields[0], fields[-1])
    return serials


def adb_available(*, refresh: bool = False) -> bool:
    """Is ``adb`` on PATH with at least one authorised phone attached?

    Cached for :data:`ADB_CACHE_SECONDS`: ``adb devices`` costs tens of milliseconds
    and one turn asks several times. ``refresh=True`` forces a fresh look.
    """
    global _availability
    now = time.monotonic()
    if not refresh:
        with _cache_lock:
            cached = _availability
        if cached is not None and (now - cached[0]) < ADB_CACHE_SECONDS:
            return cached[1]
    devices = attached_devices()
    with _cache_lock:
        _availability = (time.monotonic(), bool(devices))
    _log.debug("adb availability: %s (%s)", bool(devices), ", ".join(devices) or "none")
    return bool(devices)


def call_state() -> str:
    """``"idle"``, ``"ringing"``, ``"offhook"`` — or ``"unknown"`` without a phone.

    Read from ``dumpsys telephony.registry``, which prints one ``mCallState`` per
    subscription on a dual-SIM phone; the busiest state wins, so a call on the second
    SIM is never reported as idle.
    """
    if not adb_available():
        return "unknown"
    code, output = _run_adb(["shell", "dumpsys", "telephony.registry"])
    if code != 0 or not output.strip():
        _log.debug("dumpsys telephony.registry failed (exit %s): %s", code, output)
        return "unknown"
    states = {int(match) for match in _MCALL_STATE_RE.findall(output)}
    for value in (2, 1, 0):
        if value in states:
            return CALL_STATES[value]
    _log.debug("No mCallState in %d characters of dumpsys output", len(output))
    return "unknown"


# --- numbers ------------------------------------------------------------------------------
def _home_region(ctx: ToolContext) -> str:
    """``tools.home_region`` from the config, defaulting to Sweden."""
    try:
        region = str(ctx.config.get("tools.home_region", "SE") or "SE")
    except Exception:  # noqa: BLE001 - a minimal fake context in a test
        region = "SE"
    return region.strip().upper()[:2] or "SE"


def normalise_number(raw: Any, region: str = "SE") -> tuple[str, str]:
    """Normalise ``raw`` to E.164, returning ``(number, method)``.

    ``method`` is ``"phonenumbers"`` or ``"regex"``. An empty number means this was not
    a dialable number and the caller must say so rather than dial a guess. The offline
    ``phonenumbers`` library is optional, so both branches are live code.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return "", ""
    try:
        import phonenumbers  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        _log.debug("The 'phonenumbers' package is missing; using the regex fallback.")
    else:
        try:
            parsed = phonenumbers.parse(text, region)
            if phonenumbers.is_valid_number(parsed):
                formatted = phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164)
                return str(formatted).strip(), "phonenumbers"
            _log.debug("phonenumbers rejected %r for region %s", text, region)
        except Exception as exc:  # noqa: BLE001 - NumberParseException and friends
            _log.debug("phonenumbers could not parse %r: %s", text, exc)
        return "", ""
    compact = re.sub(r"[^\d+]", "", text)
    if compact.startswith("00"):
        compact = "+" + compact[2:]
    if compact.startswith("+"):
        body = re.sub(r"\D", "", compact[1:])
        return (f"+{body}", "regex") if 8 <= len(body) <= 15 else ("", "")
    body = re.sub(r"\D", "", compact)
    if not body.startswith("0"):
        # No country code and no trunk prefix: an order number, a date or a price.
        return "", ""
    code, body = COUNTRY_CODES.get(region, ""), body[1:]
    return (f"+{code}{body}", "regex") if code and 6 <= len(body) <= 12 else ("", "")


def spoken_number(number: str, region: str = "SE") -> str:
    """The number as digit words in threes, so it can be heard and written down."""
    dialled = " ".join(str(number or "").split())
    code = COUNTRY_CODES.get(region, "")
    if code and dialled.startswith(f"+{code}"):
        dialled = "0" + dialled[1 + len(code):]
    digits = re.sub(r"\D", "", dialled)
    if not digits:
        return ""
    words = [_DIGIT_WORDS[int(digit)] for digit in digits]
    grouped = ", ".join(" ".join(words[at:at + 3]) for at in range(0, len(words), 3))
    return f"plus {grouped}" if dialled.startswith("+") else grouped


def _who(args: dict) -> str:
    """The name the user gave, or ``""`` — the schema default is not a name."""
    name = " ".join(str(args.get("who") or "").split())
    return "" if name.lower() == DEFAULT_WHO else name


# --- dialling -------------------------------------------------------------------------------
def _detail(trail: list[str], **fields: Any) -> str:
    """The full, unspoken record of an attempt — this is what the log keeps."""
    return "\n".join([f"{key}: {value}" for key, value in fields.items()] + trail)


def _step(trail: list[str], args: list[str]) -> tuple[int, str]:
    """Run one ``adb`` step and record the command and its answer in ``trail``."""
    code, output = _run_adb(args)
    trail.append(f"$ adb {' '.join(args)}\nexit {code}\n{output or '(no output)'}")
    return code, output


def _dialled(summary: str, number: str, who: str, path: str, trail: list[str],
             *, connected: bool, state: str = "", ok: bool = True) -> ToolResult:
    """One dial outcome: the honest sentence plus everything the log wants."""
    data = {"number": number, "who": who, "path": path, "connected": connected,
            "transport": "windows" if path == "windows_handoff" else "adb"}
    if state:
        data["call_state"] = state
    return ToolResult(ok=ok, summary=summary,
                      detail=_detail(trail, number=number, path=path), data=data)


def _over_adb(number: str, spoken: str, who: str, trail: list[str]) -> ToolResult | None:
    """Dial over ADB. ``None`` means the phone vanished — try the Windows handoff."""
    target, called = f"tel:{number}", who or spoken
    code, output = _step(trail, ["shell", "am", "start", "-a", CALL_ACTION, "-d", target])
    if _device_gone(code, output):
        _log.warning("The phone went away mid-dial; handing the number to Windows")
        reset_adb_cache()
        return None
    if code == 0 and not _says(output, _REFUSAL_MARKERS):
        _log.info("dial_number placed %s through %s", number, CALL_ACTION)
        return _dialled(f"Calling {called} now, sir.", number, who, "call_intent",
                        trail, connected=True)

    _log.warning("%s was refused for the shell uid (exit %s); falling back to %s",
                 CALL_ACTION, code, DIAL_ACTION)
    dial_code, dial_output = _step(
        trail, ["shell", "am", "start", "-a", DIAL_ACTION, "-d", target])
    if _device_gone(dial_code, dial_output):
        reset_adb_cache()
        return None
    if dial_code != 0:
        return ToolResult.fail(
            f"I couldn't get {called} onto the phone at all, sir.",
            detail=_detail(trail, number=number, path="dial_intent_failed"))

    key_code, _ = _step(trail, ["shell", "input", "keyevent", CALL_KEY])
    state = call_state()
    trail.append(f"call state after the fallback: {state}")
    if key_code == 0 and state in _LIVE_STATES:
        # The handset itself says a call is up, so this one really is ringing.
        return _dialled(f"The phone is dialling {called} now, sir.", number, who,
                        "dial_intent", trail, connected=True, state=state)
    return _dialled(f"I've put {called} up on the phone's dialler, sir, but you'll have "
                    "to press call yourself.", number, who, "dial_intent", trail,
                    connected=False, state=state)


def _over_windows(number: str, spoken: str, who: str, trail: list[str]) -> ToolResult:
    """Hand ``tel:`` to whatever Windows registered — Phone Link, usually."""
    startfile = getattr(os, "startfile", None)
    called = who or spoken
    if startfile is None:
        trail.append("os.startfile is not available on this platform")
        return ToolResult.fail(
            "I can't place a call from this machine, sir, there's no phone attached.",
            detail=_detail(trail, number=number, platform=sys.platform, path="unsupported"))
    target = f"tel:{number}"
    try:
        startfile(target)
    except OSError as exc:
        trail.append(f"os.startfile({target!r}) raised {type(exc).__name__}: {exc}")
        return ToolResult.fail("Windows has nothing registered to place calls, sir.",
                               detail=_detail(trail, number=number, path="startfile_failed"))
    _log.info("dial_number handed %s to the Windows tel: handler", number)
    trail.append(f"os.startfile({target!r}) accepted the number")
    return _dialled(f"I've sent {called} to your phone, sir, but the call has to be "
                    "started there.", number, who, "windows_handoff", trail,
                    connected=False)


@tool(
    "dial_number",
    description=(
        "Place a telephone call to a number, through the Android phone on the cable "
        "when there is one, otherwise by handing the number to Windows for the phone "
        "to start. Look the number up first if you do not have it. JARVIS cannot "
        "speak on the call; the user does the talking."
    ),
    parameters={
        "type": "object",
        "properties": {
            "number": {"type": "string", "description":
                       "The number to call, ideally international, e.g. +46701234567."},
            "who": {"type": "string", "default": DEFAULT_WHO, "description":
                    "Who the number belongs to, for the confirmation. Optional."},
        },
        "required": ["number"],
    },
    tier=Tier.GUARDED,
    announce="I'm about to call {who} on {number}, sir. Shall I?",
)
def dial_number(ctx: ToolContext, args: dict) -> ToolResult:
    """Dial a number and report, exactly, which of the two paths actually ran.

    Only the CALL intent genuinely places a call, so the DIAL fallback is reported as
    a number waiting on the phone's screen unless the handset itself says a call is
    up. With no phone the number goes to the Windows ``tel:`` handler, which dials
    nothing by itself.
    """
    from jarvis.tools.safety import is_emergency_number

    reason = is_emergency_number(args.get("number", ""))
    if reason:
        return ToolResult.refuse(
            "I won't dial the emergency services, sir. If this is a real emergency, "
            "please call them yourself.",
            detail=f"refused: {reason}",
        )

    raw = " ".join(str(args.get("number") or "").split())
    who = _who(args)
    if not raw:
        return ToolResult.fail("I need a number before I can call anyone, sir.")

    digits = re.sub(r"\D", "", raw)
    if digits in EMERGENCY_NUMBERS:
        _log.warning("Refusing to dial the emergency number %s", digits)
        return ToolResult.refuse(
            f"I won't dial {digits} for you, sir, since I can't speak to an operator; "
            "please call it from the phone yourself.",
            detail=f"Emergency number {digits!r} refused: JARVIS cannot talk to an operator.")

    region = _home_region(ctx)
    number, method = normalise_number(raw, region)
    if not number:
        return ToolResult.fail(
            "That doesn't look like a number I can dial, sir.",
            detail=f"Could not normalise {raw!r} to E.164 for region {region}.")
    spoken = spoken_number(number, region)
    trail = [f"requested: {raw!r}", f"normalised with: {method or 'nothing'}"]

    if adb_available():
        result = _over_adb(number, spoken, who, trail)
        if result is not None:
            return result
    else:
        trail.append("no authorised phone on adb; using the Windows tel: handler")
    return _over_windows(number, spoken, who, trail)


@tool(
    "end_call",
    description=("Hang up the call in progress on the Android phone connected to this "
                 "machine. It cannot end a call running on an iPhone."),
    parameters={"type": "object", "properties": {}},
    tier=Tier.SAFE,
)
def end_call(ctx: ToolContext, args: dict) -> ToolResult:
    """Press the phone's hang-up key, then check whether the call really ended."""
    if not adb_available():
        return ToolResult.fail(
            "I can only hang up through the Android phone, sir, and it isn't connected.",
            detail="adb reported no authorised device; KEYCODE_ENDCALL was not sent.")
    trail: list[str] = []
    code, _ = _step(trail, ["shell", "input", "keyevent", ENDCALL_KEY])
    if code != 0:
        return ToolResult.fail("I couldn't reach the phone to hang up, sir.",
                               detail=_detail(trail, key=ENDCALL_KEY, exit_code=code))
    state = call_state()
    data = {"call_state": state, "transport": "adb"}
    detail = _detail(trail, call_state=state)
    if state == "idle":
        return ToolResult(ok=True, summary="The call is ended, sir.", detail=detail, data=data)
    if state in _LIVE_STATES:
        return ToolResult(ok=False, detail=detail, data=data,
                          summary="I sent the hang-up, sir, but the phone still shows a call.")
    return ToolResult(ok=True, summary="I've sent the hang-up to the phone, sir.",
                      detail=detail, data=data)
