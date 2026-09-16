"""No black rectangle, ever.

The desktop icon used to open a console, run a batch file and leave the operator
looking at log lines with a ring floating over them. Three things keep that from
coming back, and all three are checked here on Linux with no Windows in sight:

* ``JARVIS.exe`` is linked for the GUI subsystem — the committed binary, not just
  the build script that produced it;
* the launcher starts ``pythonw.exe`` itself with ``CREATE_NO_WINDOW`` and has no
  route back to ``cmd.exe``;
* with no console, ``setup_logging`` installs no console handler, because a
  ``StreamHandler`` over ``None`` fails on every record it is ever given.
"""

from __future__ import annotations

import logging
import re
import struct
import sys
from pathlib import Path

import pytest

import jarvis.core.logging as jarvis_logging
from jarvis.core.logging import LOGGER_NAME, get_logger, setup_logging

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_C = ROOT / "installer" / "jarvis_launcher.c"
BUILD_SH = ROOT / "installer" / "build.sh"

FLAG = jarvis_logging._CONFIGURED_FLAG

IMAGE_SUBSYSTEM_WINDOWS_GUI = 2
IMAGE_SUBSYSTEM_WINDOWS_CUI = 3


# pytest does not share fixtures between test modules; this mirrors test_logging.py.
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


def hide_console(monkeypatch) -> None:
    """Hand the test the world pythonw.exe hands a process: no stdout, no stderr.

    Called from the body rather than a fixture on purpose - pytest reinstates its own
    capture streams between the setup phase and the call, which would put stdout back.
    """
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)


def explode(self, record):
    """A handleError that tells instead of hiding: an unemittable handler must fail loudly."""
    raise AssertionError(f"{type(self).__name__} could not emit a record")


def console_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Stream handlers that are not the log file (FileHandler subclasses StreamHandler)."""
    return [
        handler for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]


def file_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]


def strip_comments(source: str) -> str:
    """Drop /* block */ and // line comments, so prose about cmd.exe is not code."""
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


def logical_lines(script: str) -> list[str]:
    """Shell lines with backslash continuations folded back into one line each."""
    return script.replace("\\\n", " ").splitlines()


def pe_subsystem(path: Path) -> int:
    """The Subsystem field of a PE image: 2 is Windows GUI, 3 is Windows CUI."""
    blob = path.read_bytes()
    pe_offset = struct.unpack_from("<I", blob, 0x3C)[0]
    assert blob[pe_offset:pe_offset + 4] == b"PE\0\0", f"{path.name} is not a PE image"
    return struct.unpack_from("<H", blob, pe_offset + 92)[0]


# --- logging without a console ---------------------------------------------------------
def test_setup_logging_installs_no_console_handler_when_there_is_no_console(
    config, fresh_logger, monkeypatch
):
    """Under pythonw the only sane console handler is no console handler at all."""
    hide_console(monkeypatch)
    logger = setup_logging(config)

    assert console_handlers(logger) == []
    assert len(file_handlers(logger)) == 1


def test_a_log_record_with_no_console_still_reaches_the_file(config, fresh_logger, monkeypatch):
    """The file handler must carry the whole record on its own, not be skipped with the console."""
    hide_console(monkeypatch)
    setup_logging(config)
    get_logger("test").info("All systems online, sir.")

    assert "All systems online, sir." in Path(config.get("logging.file")).read_text(encoding="utf-8")


def test_logging_with_no_console_never_reaches_a_handler_that_cannot_emit(
    config, fresh_logger, monkeypatch
):
    """A StreamHandler over None fails on every record; handleError only hides it.

    Turning handleError into an exception turns that silent failure into a visible
    one, which is the whole point: no handler may be installed that cannot write.
    """
    hide_console(monkeypatch)
    monkeypatch.setattr(logging.Handler, "handleError", explode)
    setup_logging(config)
    get_logger("test").info("Diagnostics are nominal, sir.")


def test_setup_logging_says_in_the_log_that_the_file_handler_is_alone(
    config, fresh_logger, monkeypatch
):
    """Someone reading the log later needs to know the console was never an option."""
    hide_console(monkeypatch)
    config.set("logging.level", "DEBUG")
    setup_logging(config)

    assert "No console stream" in Path(config.get("logging.file")).read_text(encoding="utf-8")


def test_setup_logging_still_installs_both_handlers_when_stdout_works(config, fresh_logger):
    """Started from a terminal nothing changes: the console is still the live view."""
    logger = setup_logging(config)

    assert len(console_handlers(logger)) == 1
    assert len(file_handlers(logger)) == 1


def test_the_bootstrap_handler_is_skipped_when_there_is_no_console(fresh_logger, monkeypatch):
    """The helpers log before setup_logging runs; that path must not install one either."""
    hide_console(monkeypatch)
    jarvis_logging._fallback_console_handler()

    assert console_handlers(fresh_logger) == []


def test_a_structured_helper_does_not_raise_without_a_console(fresh_logger, monkeypatch):
    """A tool call logged before setup_logging must not take the voice turn down with it."""
    hide_console(monkeypatch)
    monkeypatch.setattr(logging.Handler, "handleError", explode)
    jarvis_logging.log_refusal("blocklisted command", "diskpart")


# --- the launcher ----------------------------------------------------------------------
def test_the_launcher_starts_pythonw_with_no_window():
    """pythonw.exe plus CREATE_NO_WINDOW is the pair that means 'no console anywhere'."""
    code = strip_comments(LAUNCHER_C.read_text(encoding="utf-8"))

    assert "pythonw.exe" in code
    assert "CREATE_NO_WINDOW" in code


def test_the_launcher_is_a_gui_subsystem_program():
    """-mwindows makes the entry point wWinMain; a leftover wmain would not link."""
    code = strip_comments(LAUNCHER_C.read_text(encoding="utf-8"))

    assert "wWinMain" in code
    assert "int wmain(" not in code


def test_the_launcher_never_goes_through_cmd_or_the_batch_file():
    """Either one puts a console on the desktop, which is the complaint being fixed."""
    code = strip_comments(LAUNCHER_C.read_text(encoding="utf-8"))

    assert "cmd.exe" not in code
    assert "start-jarvis.bat" not in code


def test_the_launcher_shows_the_log_when_the_process_exits_badly():
    """With no console, the tail of jarvis.log in a message box is the only diagnostic."""
    code = strip_comments(LAUNCHER_C.read_text(encoding="utf-8"))

    assert "logs\\\\jarvis.log" in code
    assert "ShellExecuteW" in code


# --- the build ---------------------------------------------------------------------------
def test_build_sh_links_jarvis_exe_for_the_gui_subsystem():
    lines = [line for line in logical_lines(BUILD_SH.read_text(encoding="utf-8")) if '"$ROOT/JARVIS.exe"' in line]

    assert len(lines) == 1
    assert "-mwindows" in lines[0]


def test_build_sh_leaves_the_installer_with_its_console():
    """The installer legitimately wants a console: it prints half an hour of progress."""
    lines = [
        line for line in logical_lines(BUILD_SH.read_text(encoding="utf-8"))
        if '"$ROOT/Install-JARVIS.exe"' in line
    ]

    assert len(lines) == 1
    assert "-mwindows" not in lines[0]


# --- the committed binaries ---------------------------------------------------------------
def test_the_committed_jarvis_exe_is_a_gui_binary():
    """The repository ships these executables, so the subsystem in the file is what users get."""
    assert pe_subsystem(ROOT / "JARVIS.exe") == IMAGE_SUBSYSTEM_WINDOWS_GUI


def test_the_committed_installer_exe_still_has_its_console():
    assert pe_subsystem(ROOT / "Install-JARVIS.exe") == IMAGE_SUBSYSTEM_WINDOWS_CUI
