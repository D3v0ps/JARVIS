"""The safety layer: what JARVIS must refuse, what he must not refuse, and yes/no.

Two failure modes matter here and they pull in opposite directions. A missed
blocklist entry lets a destructive command through; a false positive makes JARVIS
refuse ordinary work such as listing a Downloads folder. Both directions are tested
with the same weight.
"""

from __future__ import annotations

import re

import pytest

from jarvis.tools.base import Tier, ToolResult, ToolSpec
from jarvis.tools.safety import (
    BLOCKLIST_PATTERNS,
    CANCEL_WORDS,
    CONFIRM_WORDS,
    STRICT_GUARDED_TOOLS,
    check_blocked,
    effective_tier,
    is_cancellation,
    is_confirmation,
)


def spec(name: str = "run_powershell", tier: Tier = Tier.GUARDED) -> ToolSpec:
    """A minimal registered-tool stand-in for the tier tests."""
    return ToolSpec(
        name=name,
        description="A tool.",
        parameters={"type": "object", "properties": {}},
        tier=tier,
        func=lambda ctx, args: ToolResult(ok=True, summary="done"),
    )


# ======================================================================================
# Blocklist — commands that must be refused
# ======================================================================================

MUST_BLOCK = [
    # Destroying or repartitioning a disk
    "format C:",
    "format.com /fs:ntfs D:",
    "FORMAT   c:   /q",                                   # shouting, odd whitespace
    "diskpart",
    "diskpart.exe /s wipe-script.txt",
    "   DiskPart   ",
    "bcdedit /set {default} safeboot minimal",
    "bcdedit.exe /deletevalue {current} bootstatuspolicy",
    "bootrec /fixmbr",
    # Registry attacks on security policy
    r"reg add HKLM\SYSTEM\CurrentControlSet\Services\WinDefend /v Start /t REG_DWORD /d 4 /f",
    r"reg delete HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows Defender /f",
    r"Set-ItemProperty HKLM:\SOFTWARE\Policies\Microsoft\Windows Defender -Name "
    r"DisableAntiSpyware -Value 1",
    # Defender
    "Set-MpPreference -DisableRealtimeMonitoring $true",
    "Add-MpPreference -ExclusionPath C:\\",
    "set-mppreference    -disablerealtimemonitoring    $true",
    "Stop-Service WinDefend -Force",
    "Stop-Service    -Name    WinDefend",
    "sc stop WinDefend",
    "net stop windefend",
    # Firewall
    "netsh advfirewall set allprofiles state off",
    "netsh    advfirewall    set   allprofiles   state    off",
    "netsh firewall set opmode disable",
    "Set-NetFirewallProfile -Profile Domain,Public -Enabled False",
    "Set-NetFirewallProfile -Enabled $false",
    # Credentials
    "cmdkey /list",
    "cmdkey.exe /add:server01 /user:admin /pass:hunter2",
    'vaultcmd /listcreds:"Windows Credentials"',
    "Get-Credential",
    "procdump -ma lsass.exe dump.dmp",
    # Backups and evidence
    "vssadmin delete shadows /all /quiet",
    "vssadmin.exe   Delete   Shadows   /All",
    "wmic shadowcopy delete",
    "cipher /w:C",
    "cipher.exe /w:C:\\",
    "wevtutil cl System",
    "wevtutil.exe cl Security",
    "Clear-EventLog -LogName Application",
    # Deleting Windows itself
    r"Remove-Item C:\Windows\System32\drivers\etc\hosts",
    r"del C:\Windows\System32\*.dll",
    r"rm -rf C:\Windows",
    r"ri C:\Windows\explorer.exe",                        # PowerShell alias for Remove-Item
    r"rmdir /s /q C:\Windows\System32",
    r"del %SystemRoot%\System32\config",
    r"Remove-Item $env:windir\System32 -Recurse -Force",
    # Deleting a drive root
    r"del C:\*",
    r"Remove-Item D:\ -Recurse -Force",
    "rm -rf /",
    # Power state from a raw shell
    "shutdown /s /t 0",
    "shutdown -r -t 5",
    "Stop-Computer",
    "Restart-Computer -Force",
]


@pytest.mark.parametrize("command", MUST_BLOCK, ids=[c.strip()[:48] for c in MUST_BLOCK])
def test_destructive_command_is_refused(command):
    reason = check_blocked(command)
    assert reason, f"{command!r} slipped past the blocklist"
    assert isinstance(reason, str) and reason.strip()


# ======================================================================================
# Blocklist — ordinary work that must be allowed
# ======================================================================================

MUST_ALLOW = [
    "Get-ChildItem Downloads | Sort-Object Length",
    "format the text in this file",
    r"Remove-Item $env:USERPROFILE\Downloads\old.zip",
    "Get-Process",
    "netsh advfirewall show allprofiles",
    "Get-MpComputerStatus",
    "Get-Service WinDefend",
    "Get-NetFirewallProfile",
    r"Remove-Item C:\Users\karim\Desktop\notes.txt",
    r"Get-ChildItem C:\Windows -Name",
    r"Get-Content C:\Windows\System32\drivers\etc\hosts",
    "vssadmin list shadows",
    "wevtutil qe System /c:5",
    "cipher /c hemlig.txt",
    "shutdown /a",
    "Write-Output 'do not format C drive, ever'",
    "python format_report.py --disk",
    "Rename-Item gammal.txt ny.txt",
    r"Copy-Item 'C:\Users\karim\Documents\å ä ö.txt' D:\backup\\",
    "Get-Date",
]


@pytest.mark.parametrize("command", MUST_ALLOW, ids=[c[:48] for c in MUST_ALLOW])
def test_ordinary_command_is_not_refused(command):
    assert check_blocked(command) is None, f"{command!r} was refused but is harmless"


def test_empty_input_is_not_blocked():
    assert check_blocked("") is None
    assert check_blocked(None) is None  # type: ignore[arg-type]


def test_non_string_input_is_stringified_and_still_checked():
    """The dispatcher hands over name + args, which may contain non-strings."""
    assert check_blocked(["powershell", "diskpart"]) is not None  # type: ignore[arg-type]
    assert check_blocked(42) is None  # type: ignore[arg-type]


def test_blocked_arguments_are_caught_inside_a_larger_json_payload():
    payload = '{"command": "vssadmin delete shadows /all", "reason": "cleanup"}'
    assert check_blocked(payload) == "deleting shadow copies"


def test_every_blocklist_pattern_compiles_and_has_a_speakable_reason():
    """The reason is spoken back to the user, so it must be a phrase, not a regex."""
    assert BLOCKLIST_PATTERNS, "the blocklist is empty"
    for pattern, reason in BLOCKLIST_PATTERNS:
        re.compile(pattern, re.IGNORECASE)  # raises re.error on a typo
        assert isinstance(reason, str) and reason.strip()
        assert " " in reason and len(reason) <= 60, reason
        assert not set(reason) & set("\\|*+[]{}()^$"), reason


def test_the_reason_names_the_danger_not_the_regex():
    assert check_blocked("format D: /q") == "formatting a drive"
    assert check_blocked("Set-MpPreference -DisableRealtimeMonitoring $true") == (
        "changing Defender settings"
    )
    assert check_blocked("cmdkey /list") == "reading or writing stored credentials"


# ======================================================================================
# effective_tier
# ======================================================================================


def test_normal_mode_leaves_every_tier_alone(config):
    config.set("assistant.safety_mode", "normal")
    assert effective_tier(spec("get_time_date", Tier.SAFE), config) is Tier.SAFE
    assert effective_tier(spec("type_text", Tier.ANNOUNCED), config) is Tier.ANNOUNCED
    assert effective_tier(spec("close_app", Tier.ANNOUNCED), config) is Tier.ANNOUNCED
    assert effective_tier(spec("run_powershell", Tier.GUARDED), config) is Tier.GUARDED


def test_strict_mode_upgrades_announced_to_guarded(config):
    config.set("assistant.safety_mode", "strict")
    assert effective_tier(spec("deep_think", Tier.ANNOUNCED), config) is Tier.GUARDED


def test_strict_mode_guards_close_app_and_type_text(config):
    config.set("assistant.safety_mode", "strict")
    for name in ("close_app", "type_text"):
        assert effective_tier(spec(name, Tier.SAFE), config) is Tier.GUARDED, name
    assert STRICT_GUARDED_TOOLS == {"close_app", "type_text"}


def test_strict_mode_does_not_promote_an_ordinary_safe_tool(config):
    config.set("assistant.safety_mode", "strict")
    assert effective_tier(spec("get_time_date", Tier.SAFE), config) is Tier.SAFE


def test_strict_mode_keeps_guarded_tools_guarded(config):
    config.set("assistant.safety_mode", "strict")
    assert effective_tier(spec("file_ops", Tier.GUARDED), config) is Tier.GUARDED


def test_safety_mode_is_matched_case_insensitively(config):
    config.set("assistant.safety_mode", "  STRICT  ")
    assert effective_tier(spec("type_text", Tier.ANNOUNCED), config) is Tier.GUARDED


def test_an_unknown_safety_mode_falls_back_to_normal(config):
    config.set("assistant.safety_mode", "paranoid")
    assert effective_tier(spec("type_text", Tier.ANNOUNCED), config) is Tier.ANNOUNCED


def test_the_shipped_config_defaults_to_normal_mode(config):
    assert effective_tier(spec("type_text", Tier.ANNOUNCED), config) is Tier.ANNOUNCED


def test_a_plain_mapping_works_as_a_config():
    cfg = {"assistant": {"safety_mode": "strict"}}
    assert effective_tier(spec("close_app", Tier.ANNOUNCED), cfg) is Tier.GUARDED  # type: ignore[arg-type]
    assert effective_tier(spec("close_app", Tier.ANNOUNCED), {}) is Tier.ANNOUNCED  # type: ignore[arg-type]


def test_a_broken_config_does_not_take_the_safety_layer_down():
    class ExplodingConfig:
        def get(self, dotted, default=None):
            raise RuntimeError("config file went away")

    assert effective_tier(spec("type_text", Tier.ANNOUNCED), ExplodingConfig()) is Tier.ANNOUNCED  # type: ignore[arg-type]
    assert effective_tier(spec("run_powershell", Tier.GUARDED), None) is Tier.GUARDED  # type: ignore[arg-type]


def test_a_tier_stored_as_a_plain_string_is_coerced(config):
    config.set("assistant.safety_mode", "strict")
    loose = spec("type_text", "announced")  # type: ignore[arg-type]
    assert effective_tier(loose, config) is Tier.GUARDED


# ======================================================================================
# Spoken confirmation and cancellation
# ======================================================================================

CONFIRMATIONS = [
    "yes", "Yes.", "  YES  ", "yeah", "confirm", "do it", "go ahead", "proceed",
    "affirmative", "ok", "okay", "sure",
    "ja", "JA!", "kör", "kör på", "gör det", "absolut", "ja tack", "javisst", "okej",
    "yes, go ahead", "do it, sir", "gör det nu tack",
]


@pytest.mark.parametrize("text", CONFIRMATIONS, ids=[repr(t) for t in CONFIRMATIONS])
def test_spoken_yes_is_a_confirmation(text):
    assert is_confirmation(text) is True
    assert is_cancellation(text) is False


CANCELLATIONS = [
    "no", "cancel", "stop", "abort", "nope", "nah", "never mind", "forget it",
    "nej", "avbryt", "stopp", "nej tack", "glöm det", "låt bli", "vänta",
    "no, don't", "nej, avbryt", "don't do it", "stop it",
]


@pytest.mark.parametrize("text", CANCELLATIONS, ids=[repr(t) for t in CANCELLATIONS])
def test_spoken_no_is_a_cancellation(text):
    assert is_cancellation(text) is True
    assert is_confirmation(text) is False


def test_short_negated_confirmations_are_cancellations_in_both_languages():
    """"no, don't" and "nej, avbryt" are short, but they are refusals."""
    assert is_cancellation("no, don't") is True
    assert is_cancellation("nej, avbryt") is True
    assert is_confirmation("no, don't") is False
    assert is_confirmation("nej, avbryt") is False


def test_a_negation_next_to_a_confirm_word_wins():
    assert is_cancellation("no, go ahead") is True
    assert is_confirmation("no, go ahead") is False
    assert is_cancellation("don't do it") is True


def test_a_long_sentence_merely_containing_yes_is_not_a_confirmation():
    sentence = "I said yes to that meeting yesterday, anyway let us move on"
    assert is_confirmation(sentence) is False
    assert is_cancellation(sentence) is False


def test_a_long_sentence_that_starts_with_yes_is_a_confirmation():
    assert is_confirmation("yes, go ahead and run it for me please") is True


def test_empty_and_whitespace_input_is_neither():
    for text in ("", "   ", "\n\t"):
        assert is_confirmation(text) is False
        assert is_cancellation(text) is False


def test_none_input_is_neither():
    assert is_confirmation(None) is False  # type: ignore[arg-type]
    assert is_cancellation(None) is False  # type: ignore[arg-type]


def test_an_unrelated_utterance_is_neither():
    for text in ("hmm", "what time is it", "å ä ö", "maybe later"):
        assert is_confirmation(text) is False, text
        assert is_cancellation(text) is False, text


def test_a_confirm_word_inside_a_longer_word_does_not_count():
    """"Jasmine" must not be heard as the Swedish "ja"."""
    assert is_confirmation("jasmine tea") is False
    assert is_confirmation("yesterday") is False


def test_swedish_umlauts_survive_normalisation():
    assert is_confirmation("kör på") is True
    assert is_cancellation("glöm det") is True


def test_the_contract_word_lists_are_intact():
    """ARCHITECTURE.md pins these two sets verbatim."""
    assert CONFIRM_WORDS == {
        "confirm", "yes", "yeah", "do it", "go ahead", "proceed", "affirmative",
        "kör", "ja", "gör det", "kör på", "absolut",
    }
    assert CANCEL_WORDS == {"no", "cancel", "stop", "abort", "nej", "avbryt", "stopp"}


@pytest.mark.parametrize("word", sorted(CONFIRM_WORDS))
def test_every_contract_confirm_word_confirms(word):
    assert is_confirmation(word) is True


@pytest.mark.parametrize("word", sorted(CANCEL_WORDS))
def test_every_contract_cancel_word_cancels(word):
    assert is_cancellation(word) is True
    assert is_confirmation(word) is False


def test_a_folder_merely_starting_with_windows_is_not_the_windows_directory():
    assert check_blocked(r"Remove-Item D:\windows-backup\old.zip") is None


# --------------------------------------------------------------------------------------
# Emergency numbers
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "number",
    ["112", "911", "999", "000", "110", "118", "119",
     "+46112", "0046112", "00112", "1 1 2", "(112)", "1-1-2",
     "11313", "1177", "90000"],
)
def test_an_emergency_number_is_never_dialled_automatically(number):
    """A misheard word must not be able to summon an ambulance."""
    from jarvis.tools.safety import is_emergency_number

    assert is_emergency_number(number) is not None, number


@pytest.mark.parametrize(
    "number",
    ["+46812345678", "08-123 45 67", "0701234567", "112233", "+46112233445",
     "+4681234567", "020123456", "+1 202 555 0147"],
)
def test_an_ordinary_number_is_not_mistaken_for_an_emergency(number):
    """08 is Stockholm's area code. Over-blocking here means he cannot call anyone."""
    from jarvis.tools.safety import is_emergency_number

    assert is_emergency_number(number) is None, number


def test_an_empty_number_is_not_an_emergency():
    from jarvis.tools.safety import is_emergency_number

    assert is_emergency_number("") is None
    assert is_emergency_number("   ") is None


def test_the_refusal_tells_the_user_to_call_them_himself():
    """Refusing must never stand between a person and help."""
    import logging

    from jarvis.core.scheduler import Scheduler
    from jarvis.core.state import StateBus
    from jarvis.tools import registry
    from jarvis.tools.base import ToolContext
    from jarvis.tools.dispatcher import Dispatcher

    registry.load_all()
    asked: list[str] = []
    ctx = ToolContext(
        config=__import__("jarvis.config", fromlist=["Config"]).Config.load("config.yaml"),
        memory=__import__("jarvis.core.memory", fromlist=["Memory"]).Memory("/tmp/emergency.json"),
        logger=logging.getLogger("test"),
        speak=lambda s: None,
        confirm=lambda s: asked.append(s) or True,
        notify=lambda s: None,
        scheduler=Scheduler(on_due=lambda job: None),
        state=StateBus(),
    )

    result = Dispatcher(ctx).execute("dial_number", {"number": "112"})

    assert result.ok is False
    assert result.refused is True
    assert "yourself" in result.summary.lower()
    assert not asked, "it must refuse outright, not ask whether to dial 112"
