"""Dialling, honestly: the ADB paths, the Windows handoff, and what JARVIS claims.

Everything here runs on a bare Linux box: no phone, no ``adb``, no Windows. Every
``adb`` invocation goes through a fake :func:`subprocess.run` that records the argv it
was handed and answers from canned device output, and ``os.startfile`` is injected
where the real Windows one would be.

The rule these tests exist to enforce: JARVIS never says a call is connected when all
that happened is that a number appeared on the phone's screen.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from jarvis.tools import telephony_tools as telephony
from jarvis.tools.base import Tier, ToolResult
from jarvis.tools.registry import REGISTRY

ADB_PATH = "/usr/bin/adb"
NUMBER = "+46701234567"
RAW_NUMBER = "070-123 45 67"

# --- canned adb output -----------------------------------------------------------------
DEVICES_ONE = "List of devices attached\nR58N12345\tdevice\n"
DEVICES_NONE = "List of devices attached\n\n"
DEVICES_UNAUTHORISED = "List of devices attached\nR58N12345\tunauthorized\n"
DEVICES_OFFLINE = "List of devices attached\nemulator-5554\toffline\n"
DEVICES_WITH_DAEMON = (
    "* daemon not running; starting now at tcp:5037\n"
    "* daemon started successfully\n"
    "List of devices attached\nR58N12345\tdevice\n"
)

STARTED = "Starting: Intent { act=android.intent.action.CALL dat=tel:xxx }"
SECURITY_EXCEPTION = (
    "Starting: Intent { act=android.intent.action.CALL dat=tel:xxx }\n"
    "java.lang.SecurityException: Permission Denial: starting Intent "
    "{ act=android.intent.action.CALL } requires android.permission.CALL_PHONE"
)
DEVICE_NOT_FOUND = "adb: device not found"


def dumpsys(*states: int) -> str:
    """A slice of ``dumpsys telephony.registry`` carrying one mCallState per SIM."""
    lines = ["Phone Id=0", "mSubId=1"]
    for state in states:
        lines.append(f"  mCallState={state}")
    lines.append("mCallForwarding=false")
    return "\n".join(lines)


# --- the fake phone --------------------------------------------------------------------
class FakeAdb:
    """Stands in for ``subprocess.run``, answering ``adb`` from canned output."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.devices = DEVICES_ONE
        self.call_intent: tuple[int, str, str] = (0, STARTED, "")
        self.dial_intent: tuple[int, str, str] = (0, STARTED, "")
        self.keyevent: tuple[int, str, str] = (0, "", "")
        self.dumpsys: tuple[int, str, str] = (0, dumpsys(0), "")
        self.devices_exit = 0

    # -- routing ------------------------------------------------------------------
    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        code, stdout, stderr = self._answer(argv[1:])
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    def _answer(self, args: list[str]) -> tuple[int, str, str]:
        joined = " ".join(args)
        if args[:1] == ["devices"]:
            return self.devices_exit, self.devices, ""
        if "dumpsys" in joined:
            return self.dumpsys
        if telephony.CALL_ACTION in joined:
            return self.call_intent
        if telephony.DIAL_ACTION in joined:
            return self.dial_intent
        if "keyevent" in joined:
            return self.keyevent
        raise AssertionError(f"the fake phone was asked something unexpected: {args!r}")

    # -- assertions helpers -------------------------------------------------------
    @property
    def commands(self) -> list[str]:
        """Every invocation as one string, for readable assertions."""
        return [" ".join(call) for call in self.calls]

    def ran(self, fragment: str) -> bool:
        return any(fragment in command for command in self.commands)

    def count(self, fragment: str) -> int:
        return sum(1 for command in self.commands if fragment in command)


class FakeConfig:
    """Just enough Config for the tools: ``get`` with a default."""

    def __init__(self, values: dict | None = None) -> None:
        self._values = values or {}

    def get(self, dotted: str, default=None):
        return self._values.get(dotted, default)


@pytest.fixture
def ctx(tmp_path: Path):
    """A minimal ToolContext stand-in; the tools only read the config off it."""
    return types.SimpleNamespace(
        config=FakeConfig({"tools.home_region": "SE"}),
        memory=types.SimpleNamespace(path=tmp_path / "memory.json"),
        logger=None, speak=lambda text: None, confirm=lambda text: True,
        notify=lambda text: None, scheduler=None, state=None,
    )


@pytest.fixture(autouse=True)
def fresh_cache():
    """The availability cache is module state: never let it leak between tests."""
    telephony.reset_adb_cache()
    yield
    telephony.reset_adb_cache()


@pytest.fixture
def adb(monkeypatch):
    """``adb`` on PATH with one authorised phone, and every call recorded."""
    fake = FakeAdb()
    monkeypatch.setattr(telephony.shutil, "which",
                        lambda name: ADB_PATH if name == "adb" else None)
    monkeypatch.setattr(telephony.subprocess, "run", fake)
    return fake


@pytest.fixture
def no_adb(monkeypatch):
    """No ``adb`` anywhere, and a tripwire on ``subprocess.run``."""
    def explode(*args, **kwargs):
        raise AssertionError("subprocess.run was called although adb is not installed")

    monkeypatch.setattr(telephony.shutil, "which", lambda name: None)
    monkeypatch.setattr(telephony.subprocess, "run", explode)


@pytest.fixture
def windows(monkeypatch):
    """A stand-in for ``os.startfile``, recording what Windows was handed."""
    opened: list[str] = []
    monkeypatch.setattr(telephony.os, "startfile", opened.append, raising=False)
    return opened


@pytest.fixture
def no_phonenumbers(monkeypatch):
    """Force the regex fallback even on a machine that has ``phonenumbers``."""
    monkeypatch.setitem(sys.modules, "phonenumbers", None)


CONNECTED_CLAIMS = ("calling", "is dialling", "ringing", "connected", "on the line")


def assert_no_connection_claim(summary: str) -> None:
    """A summary that only opened the dialler must not sound like a live call."""
    lowered = summary.lower()
    for claim in CONNECTED_CLAIMS:
        assert claim not in lowered, f"{summary!r} claims a call that was not placed"


# ======================================================================================
# Registration
# ======================================================================================
def test_dial_number_is_guarded_and_end_call_is_safe():
    assert REGISTRY["dial_number"].tier is Tier.GUARDED
    assert REGISTRY["end_call"].tier is Tier.SAFE


def test_the_announcement_names_who_and_the_number():
    spec = REGISTRY["dial_number"]
    announcement = spec.render_announcement({"who": "Anna", "number": NUMBER})
    assert "Anna" in announcement and NUMBER in announcement


def test_the_announcement_still_reads_as_a_sentence_without_a_name():
    """The schema default keeps the guarded confirmation from going generic."""
    spec = REGISTRY["dial_number"]
    default = spec.parameters["properties"]["who"]["default"]
    announcement = spec.render_announcement({"who": default, "number": NUMBER})
    assert NUMBER in announcement
    assert "About to run" not in announcement


def test_a_guarded_dial_asks_first_and_dials_nothing_when_refused(ctx, adb):
    from jarvis.tools.dispatcher import Dispatcher

    asked: list[str] = []

    def refuse(announcement: str) -> bool:
        asked.append(announcement)
        return False

    ctx.confirm = refuse
    result = Dispatcher(ctx).execute("dial_number", {"number": NUMBER, "who": "Anna"})

    assert len(asked) == 1 and "Anna" in asked[0] and NUMBER in asked[0]
    assert not result.ok
    assert not adb.ran("am start"), "a cancelled call still went to the phone"


# ======================================================================================
# adb_available / attached_devices
# ======================================================================================
def test_an_authorised_phone_counts_as_available(adb):
    assert telephony.adb_available() is True
    assert telephony.attached_devices() == ["R58N12345"]


def test_the_daemon_banner_is_not_mistaken_for_a_phone(adb):
    adb.devices = DEVICES_WITH_DAEMON
    assert telephony.attached_devices() == ["R58N12345"]


@pytest.mark.parametrize(
    "listing", [DEVICES_NONE, DEVICES_UNAUTHORISED, DEVICES_OFFLINE],
    ids=["nothing attached", "still unauthorised", "offline"],
)
def test_adb_without_a_usable_phone_is_not_available(adb, listing):
    adb.devices = listing
    assert telephony.adb_available() is False


def test_a_failing_adb_devices_is_not_available(adb):
    adb.devices_exit = 1
    adb.devices = "adb: failed to start daemon"
    assert telephony.adb_available() is False


def test_no_adb_on_path_is_not_available_and_starts_no_process(no_adb):
    assert telephony.adb_available() is False
    assert telephony.attached_devices() == []


def test_the_availability_answer_is_cached_briefly(adb):
    assert telephony.adb_available() is True
    assert telephony.adb_available() is True
    assert adb.count("devices") == 1, "the cache did not hold"


def test_a_refresh_and_a_reset_both_look_again(adb):
    telephony.adb_available()
    telephony.adb_available(refresh=True)
    assert adb.count("devices") == 2
    telephony.reset_adb_cache()
    telephony.adb_available()
    assert adb.count("devices") == 3


def test_the_cache_expires(adb, monkeypatch):
    monkeypatch.setattr(telephony, "ADB_CACHE_SECONDS", 0.0)
    telephony.adb_available()
    telephony.adb_available()
    assert adb.count("devices") == 2


# ======================================================================================
# call_state
# ======================================================================================
@pytest.mark.parametrize(
    "value,expected", [(0, "idle"), (1, "ringing"), (2, "offhook")],
    ids=["idle", "ringing", "offhook"],
)
def test_every_call_state_is_parsed(adb, value, expected):
    adb.dumpsys = (0, dumpsys(value), "")
    assert telephony.call_state() == expected


def test_a_call_on_the_second_sim_is_not_reported_as_idle(adb):
    adb.dumpsys = (0, dumpsys(0, 2), "")
    assert telephony.call_state() == "offhook"


def test_a_ringing_sim_beats_an_idle_one(adb):
    adb.dumpsys = (0, dumpsys(1, 0), "")
    assert telephony.call_state() == "ringing"


def test_dumpsys_without_a_call_state_is_unknown(adb):
    adb.dumpsys = (0, "Phone Id=0\nmSubId=1\n", "")
    assert telephony.call_state() == "unknown"


def test_a_failing_dumpsys_is_unknown(adb):
    adb.dumpsys = (1, "", "adb: error: closed")
    assert telephony.call_state() == "unknown"


def test_call_state_without_a_phone_is_unknown_and_asks_nothing(adb):
    adb.devices = DEVICES_NONE
    assert telephony.call_state() == "unknown"
    assert not adb.ran("dumpsys")


def test_call_state_without_adb_is_unknown(no_adb):
    assert telephony.call_state() == "unknown"


# ======================================================================================
# dial_number — the CALL path
# ======================================================================================
def test_the_call_intent_is_the_first_thing_tried(ctx, adb, caplog):
    with caplog.at_level("INFO", logger="jarvis.tools.telephony_tools"):
        result = telephony.dial_number(ctx, {"number": RAW_NUMBER, "who": "Anna"})

    assert result.ok
    assert result.data["path"] == "call_intent"
    assert result.data["connected"] is True
    assert result.summary == "Calling Anna now, sir."
    assert adb.ran(f"shell am start -a {telephony.CALL_ACTION} -d tel:{NUMBER}")
    assert not adb.ran(telephony.DIAL_ACTION), "the fallback ran although CALL worked"
    assert any(telephony.CALL_ACTION in record.getMessage() for record in caplog.records)


def test_without_a_name_the_number_is_spoken_as_digits(ctx, adb):
    result = telephony.dial_number(ctx, {"number": NUMBER})
    assert "zero seven zero" in result.summary
    assert NUMBER not in result.summary, "a raw +46 number was handed to the voice"


# ======================================================================================
# dial_number — the DIAL fallback
# ======================================================================================
@pytest.mark.parametrize(
    "call_intent",
    [
        pytest.param((0, SECURITY_EXCEPTION, ""), id="SecurityException on stdout"),
        pytest.param((1, "", SECURITY_EXCEPTION), id="SecurityException on stderr"),
        pytest.param((1, "", "Error: Activity not started"), id="a bare non-zero exit"),
    ],
)
def test_a_refused_call_intent_falls_back_to_the_dialler(ctx, adb, call_intent, caplog):
    adb.call_intent = call_intent
    with caplog.at_level("WARNING", logger="jarvis.tools.telephony_tools"):
        result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert adb.ran(f"am start -a {telephony.DIAL_ACTION} -d tel:{NUMBER}")
    assert adb.ran(f"input keyevent {telephony.CALL_KEY}")
    assert result.data["path"] == "dial_intent"
    assert any(telephony.DIAL_ACTION in record.getMessage() for record in caplog.records)


def test_the_dialler_fallback_never_claims_the_call_is_up(ctx, adb):
    adb.call_intent = (0, SECURITY_EXCEPTION, "")
    adb.dumpsys = (0, dumpsys(0), "")  # the phone is still idle afterwards

    result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert result.ok, "the number did reach the phone"
    assert result.data["connected"] is False
    assert "press call" in result.summary.lower()
    assert_no_connection_claim(result.summary)


def test_the_fallback_says_it_is_dialling_only_when_the_phone_agrees(ctx, adb):
    adb.call_intent = (0, SECURITY_EXCEPTION, "")
    adb.dumpsys = (0, dumpsys(2), "")  # the handset itself reports a call

    result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert result.data["connected"] is True
    assert result.summary == "The phone is dialling Anna now, sir."


def test_a_keyevent_that_fails_is_reported_as_a_number_on_screen(ctx, adb):
    adb.call_intent = (1, "", SECURITY_EXCEPTION)
    adb.keyevent = (1, "", "adb: error: closed")
    adb.dumpsys = (0, dumpsys(0), "")

    result = telephony.dial_number(ctx, {"number": NUMBER})

    assert result.data["connected"] is False
    assert_no_connection_claim(result.summary)


def test_when_even_the_dialler_refuses_nothing_is_claimed(ctx, adb):
    adb.call_intent = (1, "", SECURITY_EXCEPTION)
    adb.dial_intent = (1, "", "Error: Activity not started")

    result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert result.ok is False
    assert not adb.ran("keyevent"), "the green button was pressed on nothing"
    assert_no_connection_claim(result.summary)


# ======================================================================================
# dial_number — the Windows handoff (the iPhone path)
# ======================================================================================
def test_without_adb_the_number_goes_to_windows(ctx, no_adb, windows):
    result = telephony.dial_number(ctx, {"number": RAW_NUMBER, "who": "Anna"})

    assert windows == [f"tel:{NUMBER}"]
    assert result.ok
    assert result.data["path"] == "windows_handoff"
    assert result.data["connected"] is False
    assert "started" in result.summary.lower() and "phone" in result.summary.lower()
    assert_no_connection_claim(result.summary)


def test_with_adb_but_no_phone_the_number_still_goes_to_windows(ctx, adb, windows):
    adb.devices = DEVICES_NONE

    result = telephony.dial_number(ctx, {"number": NUMBER})

    assert windows == [f"tel:{NUMBER}"]
    assert not adb.ran("am start")
    assert result.data["transport"] == "windows"


def test_a_phone_unplugged_mid_dial_falls_through_to_windows(ctx, adb, windows):
    """adb said yes a moment ago; the cable came out before the intent."""
    telephony.adb_available()  # warm the cache while the phone is still there
    adb.devices = DEVICES_NONE
    adb.call_intent = (1, "", DEVICE_NOT_FOUND)

    result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert result.data["path"] == "windows_handoff"
    assert windows == [f"tel:{NUMBER}"]
    assert telephony._availability is None or telephony.adb_available() is False


def test_the_dial_intent_losing_the_phone_also_falls_through(ctx, adb, windows):
    adb.call_intent = (0, SECURITY_EXCEPTION, "")
    adb.dial_intent = (1, "", DEVICE_NOT_FOUND)

    result = telephony.dial_number(ctx, {"number": NUMBER})

    assert result.data["path"] == "windows_handoff"
    assert windows == [f"tel:{NUMBER}"]


def test_on_a_machine_without_startfile_it_says_so_plainly(ctx, no_adb, monkeypatch):
    monkeypatch.delattr(telephony.os, "startfile", raising=False)

    result = telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})

    assert result.ok is False
    assert "can't place a call" in result.summary
    assert_no_connection_claim(result.summary)


def test_a_windows_without_a_tel_handler_is_reported_honestly(ctx, no_adb, monkeypatch):
    def refuse(target: str) -> None:
        raise OSError("No application is associated with tel:")

    monkeypatch.setattr(telephony.os, "startfile", refuse, raising=False)

    result = telephony.dial_number(ctx, {"number": NUMBER})

    assert result.ok is False
    assert "nothing registered" in result.summary
    assert_no_connection_claim(result.summary)


# ======================================================================================
# dial_number — numbers it must not dial
# ======================================================================================
def test_an_empty_number_is_refused_before_anything_starts(ctx, adb):
    result = telephony.dial_number(ctx, {"number": "   "})
    assert result.ok is False
    assert adb.calls == []


def test_something_that_is_not_a_number_is_never_guessed_at(ctx, adb, no_phonenumbers):
    result = telephony.dial_number(ctx, {"number": "the dentist on the corner"})
    assert result.ok is False
    assert "dial" in result.summary
    assert not adb.ran("am start")


def test_an_emergency_number_is_refused_in_character(ctx, adb, windows):
    result = telephony.dial_number(ctx, {"number": "112"})

    assert result.refused is True
    assert result.ok is False
    assert not adb.ran("am start")
    assert windows == []


# ======================================================================================
# end_call
# ======================================================================================
def test_end_call_hangs_up_and_confirms_the_line_is_clear(ctx, adb):
    adb.dumpsys = (0, dumpsys(0), "")

    result = telephony.end_call(ctx, {})

    assert adb.ran(f"input keyevent {telephony.ENDCALL_KEY}")
    assert result.ok and result.summary == "The call is ended, sir."


def test_end_call_admits_when_the_call_is_still_up(ctx, adb):
    adb.dumpsys = (0, dumpsys(2), "")

    result = telephony.end_call(ctx, {})

    assert result.ok is False
    assert "still shows a call" in result.summary


def test_end_call_does_not_claim_more_than_it_knows(ctx, adb):
    adb.dumpsys = (1, "", "adb: error: closed")

    result = telephony.end_call(ctx, {})

    assert result.ok
    assert result.summary == "I've sent the hang-up to the phone, sir."


def test_end_call_fails_when_the_keyevent_does(ctx, adb):
    adb.keyevent = (1, "", "adb: error: closed")

    result = telephony.end_call(ctx, {})

    assert result.ok is False
    assert not adb.ran("dumpsys"), "the state was read although the hang-up failed"


def test_end_call_without_a_phone_says_so(ctx, adb):
    adb.devices = DEVICES_NONE

    result = telephony.end_call(ctx, {})

    assert result.ok is False
    assert not adb.ran("keyevent")


def test_end_call_without_adb_says_so(ctx, no_adb):
    result = telephony.end_call(ctx, {})
    assert result.ok is False
    assert "isn't connected" in result.summary


# ======================================================================================
# Numbers
# ======================================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param("070-123 45 67", "+46701234567", id="Swedish, written out"),
        pytest.param("0046701234567", "+46701234567", id="with a 00 prefix"),
        pytest.param("+46 8 123 45 67", "+4681234567", id="already international"),
        pytest.param("08-123 456", "+468123456", id="a landline"),
    ],
)
def test_numbers_are_normalised_to_e164(raw, expected, no_phonenumbers):
    number, method = telephony.normalise_number(raw, "SE")
    assert number == expected
    assert method == "regex"


@pytest.mark.parametrize(
    "raw",
    [pytest.param("", id="empty"), pytest.param("hello", id="words"),
     pytest.param("12345", id="no country code and no trunk zero"),
     pytest.param("+4670", id="far too short"), pytest.param("556677-8899", id="an org number")],
)
def test_a_number_that_does_not_add_up_comes_back_empty(raw, no_phonenumbers):
    assert telephony.normalise_number(raw, "SE") == ("", "")


def test_phonenumbers_is_used_when_it_is_installed(monkeypatch):
    """The Windows machine has the library; that branch must be exercised too."""
    seen: list[tuple[str, str]] = []

    class FakeFormat:
        E164 = 0

    fake = types.SimpleNamespace(
        PhoneNumberFormat=FakeFormat,
        parse=lambda text, region: seen.append((text, region)) or "parsed",
        is_valid_number=lambda parsed: True,
        format_number=lambda parsed, fmt: " +46701234567 ",
    )
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert telephony.normalise_number("070-123 45 67", "SE") == (NUMBER, "phonenumbers")
    assert seen == [("070-123 45 67", "SE")]


def test_a_number_phonenumbers_rejects_is_not_dialled(monkeypatch):
    fake = types.SimpleNamespace(
        PhoneNumberFormat=types.SimpleNamespace(E164=0),
        parse=lambda text, region: "parsed",
        is_valid_number=lambda parsed: False,
        format_number=lambda parsed, fmt: "+999",
    )
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert telephony.normalise_number("070-123 45 67", "SE") == ("", "")


def test_a_phonenumbers_that_raises_does_not_take_the_turn_down(monkeypatch):
    def explode(text, region):
        raise ValueError("NumberParseException")

    fake = types.SimpleNamespace(
        PhoneNumberFormat=types.SimpleNamespace(E164=0), parse=explode,
        is_valid_number=lambda parsed: True, format_number=lambda parsed, fmt: "+46",
    )
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert telephony.normalise_number("070-123 45 67", "SE") == ("", "")


def test_the_spoken_form_is_dialable_digits_not_a_large_number():
    spoken = telephony.spoken_number(NUMBER, "SE")
    assert spoken.startswith("zero seven zero")
    assert not any(character.isdigit() for character in spoken)


def test_a_foreign_number_keeps_its_country_code_when_spoken():
    spoken = telephony.spoken_number("+441234567890", "SE")
    assert spoken.startswith("plus four four")


def test_the_home_region_comes_from_the_config(ctx, adb, no_phonenumbers):
    ctx.config = FakeConfig({"tools.home_region": "no"})
    telephony.dial_number(ctx, {"number": "090-12 34 56"})
    assert adb.ran("tel:+479012345"), adb.commands


def test_a_context_without_a_usable_config_still_dials(adb, no_phonenumbers):
    broken = types.SimpleNamespace(config=None)
    result = telephony.dial_number(broken, {"number": NUMBER})
    assert isinstance(result, ToolResult) and result.ok


# ======================================================================================
# Summaries are spoken: one sentence, no markdown, no lists
# ======================================================================================
def _every_dial_summary(ctx, adb, windows) -> list[ToolResult]:
    """Run every branch once and collect what JARVIS would say."""
    results = [telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"})]
    adb.call_intent = (0, SECURITY_EXCEPTION, "")
    results.append(telephony.dial_number(ctx, {"number": NUMBER, "who": "Anna"}))
    adb.dumpsys = (0, dumpsys(2), "")
    results.append(telephony.dial_number(ctx, {"number": NUMBER}))
    adb.devices = DEVICES_NONE
    telephony.reset_adb_cache()
    results.append(telephony.dial_number(ctx, {"number": NUMBER}))
    results.append(telephony.dial_number(ctx, {"number": "nonsense"}))
    results.append(telephony.dial_number(ctx, {"number": "112"}))
    return results


def test_every_spoken_summary_is_one_plain_sentence(ctx, adb, windows):
    for result in _every_dial_summary(ctx, adb, windows):
        summary = result.summary
        assert summary and summary.endswith(("."))
        assert summary.count(".") <= 2, summary
        assert not any(mark in summary for mark in ("*", "#", "`", "\n", "- ")), summary
        assert len(summary) < 160, summary


def test_a_summary_only_claims_a_connection_when_the_data_says_so(ctx, adb, windows):
    for result in _every_dial_summary(ctx, adb, windows):
        connected = bool((result.data or {}).get("connected"))
        if not connected:
            assert_no_connection_claim(result.summary)
