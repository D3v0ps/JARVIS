"""Logging setup and the structured event helpers.

The log file is the only forensic trail a local assistant leaves, so these tests care
about three things: it must exist where the config says, it must be readable UTF-8 with
no terminal colour codes in it, and no helper may ever throw an exception back into a
voice turn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import jarvis.core.logging as jarvis_logging
from jarvis.core.logging import (
    LOGGER_NAME,
    get_logger,
    log_latency,
    log_refusal,
    log_tool_call,
    log_transcript,
    setup_logging,
)

FLAG = jarvis_logging._CONFIGURED_FLAG


@pytest.fixture
def fresh_logger():
    """Hand each test an unconfigured 'jarvis' logger and put the old one back after."""
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    saved_flag = getattr(logger, FLAG, None)

    for handler in saved_handlers:
        logger.removeHandler(handler)
    if hasattr(logger, FLAG):
        delattr(logger, FLAG)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - cleanup best effort
                pass
        for handler in saved_handlers:
            logger.addHandler(handler)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate
        if hasattr(logger, FLAG):
            delattr(logger, FLAG)
        if saved_flag is not None:
            setattr(logger, FLAG, saved_flag)


def log_file(cfg) -> Path:
    return Path(cfg.get("logging.file"))


def read_log(cfg) -> str:
    return log_file(cfg).read_text(encoding="utf-8")


# --- setup ----------------------------------------------------------------------------
def test_setup_logging_returns_the_jarvis_logger(config, fresh_logger):
    assert setup_logging(config) is logging.getLogger(LOGGER_NAME)


def test_setup_logging_creates_the_configured_log_file(config, fresh_logger):
    setup_logging(config)
    get_logger("test").info("All systems online, sir.")

    assert "All systems online, sir." in read_log(config)


def test_setup_logging_is_idempotent(config, fresh_logger):
    logger = setup_logging(config)
    after_first = len(logger.handlers)
    setup_logging(config)
    setup_logging(config)

    assert len(logger.handlers) == after_first


def test_setup_logging_does_not_duplicate_lines_in_the_file(config, fresh_logger):
    setup_logging(config)
    setup_logging(config)
    get_logger("test").info("Only once, sir.")

    assert read_log(config).count("Only once, sir.") == 1


def test_a_second_setup_refreshes_the_level(config, fresh_logger):
    logger = setup_logging(config)
    config.set("logging.level", "DEBUG")
    setup_logging(config)

    assert logger.level == logging.DEBUG


def test_an_unknown_level_falls_back_to_info(config, fresh_logger):
    config.set("logging.level", "SHOUTING")
    assert setup_logging(config).level == logging.INFO


def test_a_relative_log_path_lands_beside_the_config_file(config, fresh_logger, tmp_path):
    config.set("logging.file", "logs/jarvis.log")
    setup_logging(config)
    get_logger("test").info("Beside the config, sir.")

    assert "Beside the config, sir." in (tmp_path / "logs" / "jarvis.log").read_text(encoding="utf-8")


def test_an_unusable_log_path_degrades_to_console_only(config, fresh_logger, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    config.set("logging.file", str(blocker / "jarvis.log"))

    logger = setup_logging(config)
    logger.info("Still talking, sir.")

    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_get_logger_returns_a_child_of_the_jarvis_logger():
    assert get_logger("memory").name == "jarvis.memory"
    assert get_logger("jarvis.memory").name == "jarvis.memory"
    assert get_logger("").name == LOGGER_NAME


# --- what ends up in the file ---------------------------------------------------------
def test_the_log_file_is_utf8_and_keeps_swedish_characters(config, fresh_logger):
    setup_logging(config)
    log_transcript("jarvis", "Det är åtta grader i Göteborg, sir.")

    raw = log_file(config).read_bytes()
    assert "Göteborg".encode("utf-8") in raw
    assert "åtta grader" in raw.decode("utf-8")


def test_no_ansi_escape_codes_reach_the_log_file(config, fresh_logger, monkeypatch):
    """The console may be colourful; the file must stay greppable."""
    monkeypatch.setattr(jarvis_logging, "_use_color", lambda cfg, stream: True)
    setup_logging(config)

    log_transcript("user", "what is the weather")
    log_transcript("jarvis", "\033[31mIt is raining, sir.\033[0m")
    log_refusal("Blocked command", "format C:")
    log_latency("first audio", 1180.0)

    text = read_log(config)
    assert "\033[" not in text
    assert "It is raining, sir." in text


def test_a_transcript_stays_on_a_single_line(config, fresh_logger):
    setup_logging(config)
    log_transcript("user", "set a timer\nfor ten minutes\n")

    lines = [line for line in read_log(config).splitlines() if "set a timer" in line]
    assert lines == [lines[0]]
    assert "for ten minutes" in lines[0]


def test_a_transcript_records_who_was_speaking(config, fresh_logger):
    setup_logging(config)
    log_transcript("user", "hello")
    log_transcript("jarvis", "Good evening, sir.")

    text = read_log(config)
    assert "USER: hello" in text
    assert "JARVIS: Good evening, sir." in text


def test_a_transcript_records_the_latency_when_given(config, fresh_logger):
    setup_logging(config)
    log_transcript("jarvis", "Good evening, sir.", latency_ms=1180.4)

    assert "[1180 ms]" in read_log(config)


def test_a_refusal_is_logged_as_a_warning(config, fresh_logger):
    setup_logging(config)
    log_refusal("That would disable Windows Defender", "Set-MpPreference -DisableRealtimeMonitoring")

    text = read_log(config)
    assert "REFUSED: That would disable Windows Defender" in text
    assert "WARNING" in text


def test_a_refusal_keeps_the_dangerous_detail_out_of_the_info_log(config, fresh_logger):
    """The spoken reason is logged; the raw command only at DEBUG level."""
    setup_logging(config)
    log_refusal("That would disable Windows Defender", "Set-MpPreference -DisableRealtimeMonitoring")

    assert "Set-MpPreference" not in read_log(config)


def test_a_tool_call_records_name_arguments_and_duration(config, fresh_logger):
    setup_logging(config)

    class Result:
        ok = True
        refused = False
        summary = "The system is at 12 percent CPU."
        detail = "cpu=12"

    log_tool_call("system_status", {"verbose": True}, Result(), duration_ms=42.0)

    text = read_log(config)
    assert "TOOL system_status" in text
    assert '"verbose": true' in text
    assert "42 ms" in text
    assert "The system is at 12 percent CPU." in text


def test_a_latency_measurement_is_logged_in_milliseconds(config, fresh_logger):
    setup_logging(config)
    log_latency("first audio", 1180.4)

    assert "LATENCY first audio: 1180 ms" in read_log(config)


# --- the helpers must never crash a voice turn ----------------------------------------
def test_helpers_survive_hostile_arguments(config, fresh_logger):
    setup_logging(config)

    class Unserialisable:
        def __repr__(self) -> str:
            return "<no json for you>"

    log_transcript("jarvis", None, latency_ms=float("inf"))
    log_transcript("", "")
    log_tool_call("weather", {"city": Unserialisable(), "tags": {"a", "b"}}, None, duration_ms=float("inf"))
    log_tool_call("weather", {object(): "unhashable key for json"}, None, duration_ms=float("nan"))
    log_refusal(None, None)
    log_latency(None, float("inf"))

    assert read_log(config).strip(), "the hostile calls should still have produced log lines"


def test_a_tool_result_that_explodes_when_inspected_does_not_propagate(config, fresh_logger):
    setup_logging(config)

    class ExplodingResult:
        @property
        def ok(self) -> bool:
            raise RuntimeError("boom")

    log_tool_call("system_status", {}, ExplodingResult(), duration_ms=1.0)  # must not raise


def test_helpers_work_before_setup_logging_was_ever_called(fresh_logger, capsys):
    log_transcript("user", "vad är klockan")
    log_latency("first audio", 900.0)

    assert fresh_logger.handlers, "a helper should install a console handler on its own"
    assert "vad är klockan" in capsys.readouterr().out


def test_setup_logging_after_a_helper_does_not_stack_handlers(config, fresh_logger):
    log_transcript("user", "hello")
    bootstrap_count = len(fresh_logger.handlers)

    logger = setup_logging(config)
    logger.info("Configured, sir.")

    assert len(logger.handlers) == 2  # console + rotating file
    assert bootstrap_count == 1
    assert read_log(config).count("Configured, sir.") == 1
