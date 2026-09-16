"""No black rectangle, ever.

The desktop icon used to open a console, run a batch file and leave the operator
looking at log lines with a ring floating over them. Three things keep that from
coming back, and all three are checked here on Linux with no Windows in sight:

* ``JARVIS.exe`` is linked for the GUI subsystem — the committed binary, not just
  the build script that produced it;
* the launcher starts ``pythonw.exe`` itself with ``CREATE_NO_WINDOW`` and has no
  route back to ``cmd.exe``, and hands its own diagnostic the whole buffer;
* with no console, ``setup_logging`` installs no console handler - not even over the
  null streams ``__main__`` puts in place first, which look writable and are not;
* nothing JARVIS spawns while he is up may open a console of its own;
* a crash in any thread reaches the log, because the log is the only place left to
  read it.
"""

from __future__ import annotations

import ast
import logging
import re
import struct
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.__main__ as entry
import jarvis.core.logging as jarvis_logging
from jarvis.core.logging import LOGGER_NAME, NULL_STREAM_FLAG, get_logger, setup_logging

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_C = ROOT / "installer" / "jarvis_launcher.c"
BUILD_SH = ROOT / "installer" / "build.sh"

#: Every tree that can run while JARVIS is up, plus the two he never starts himself.
SOURCE_ROOTS = (ROOT / "jarvis", ROOT / "scripts", ROOT / "installer")

#: Programs a person runs from a terminal on purpose, and which therefore want the
#: console they are already standing in: the installer's icon builder and the two
#: command-line scripts. JARVIS never spawns any of them. Everything else - including
#: the preflight doctor, which JARVIS can be asked to run - must hide its window.
CONSOLE_PROGRAMS = {
    "installer/make_icon.py",
    "scripts/enable_phone.py",
    "scripts/fetch_models.py",
}

#: ``subprocess`` entry points that start a process, and therefore a console.
LAUNCHERS = {"run", "Popen", "call", "check_call", "check_output"}

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


# --- the null streams __main__ installs first --------------------------------------------
def close_extra_sinks(kept: int) -> None:
    """Close the null-device handles a test made, so no file descriptor leaks out of it."""
    while len(entry._SINKS) > kept:
        sink = entry._SINKS.pop()
        try:
            sink.close()
        except Exception:  # pragma: no cover - cleanup best effort
            pass


def test_the_null_streams_are_tagged_as_the_sinks_they_are(monkeypatch):
    """Nothing else can tell them apart: a handle on the null device writes happily."""
    hide_console(monkeypatch)
    kept = len(entry._SINKS)
    try:
        assert entry._install_null_streams() is False
        assert sys.stdout is not None and sys.stderr is not None
        assert getattr(sys.stdout, NULL_STREAM_FLAG, False) is True
        assert getattr(sys.stderr, NULL_STREAM_FLAG, False) is True
    finally:
        close_extra_sinks(kept)


def test_setup_logging_installs_no_console_handler_after_the_null_streams_are_in_place(
    config, fresh_logger, monkeypatch
):
    """This is the real pythonw order: null streams first, setup_logging second.

    ``_install_null_streams`` runs before anything else so no library trips over a
    ``None`` stream, which means ``setup_logging`` is handed a ``sys.stdout`` that
    writes without complaint. Unless it recognises that sink, it installs a console
    handler over the null device and every record is formatted, coloured and thrown
    away - the whole point of leaving the handler out, defeated.
    """
    hide_console(monkeypatch)
    kept = len(entry._SINKS)
    try:
        entry._install_null_streams()
        logger = setup_logging(config)

        assert console_handlers(logger) == []
        assert len(file_handlers(logger)) == 1
    finally:
        close_extra_sinks(kept)


def test_a_record_still_reaches_the_file_with_the_null_streams_in_place(
    config, fresh_logger, monkeypatch
):
    """The file is the only record under pythonw, so it had better hold the record."""
    hide_console(monkeypatch)
    kept = len(entry._SINKS)
    try:
        entry._install_null_streams()
        setup_logging(config)
        get_logger("test").info("Running on the reserve power, sir.")
    finally:
        close_extra_sinks(kept)

    assert "Running on the reserve power, sir." in Path(config.get("logging.file")).read_text(
        encoding="utf-8"
    )


def test_a_console_stream_that_is_not_ours_is_still_a_console(monkeypatch):
    """Only our own sinks are refused; a real terminal must keep its live view."""
    real = SimpleNamespace(write=lambda text: None)
    monkeypatch.setattr(sys, "stdout", real)

    assert jarvis_logging._console_stream() is real


def test_the_null_sink_on_stdout_does_not_promote_stderr_to_a_console(monkeypatch):
    """Both streams are ours under pythonw; neither may be mistaken for a terminal."""
    hide_console(monkeypatch)
    kept = len(entry._SINKS)
    try:
        entry._install_null_streams()

        assert jarvis_logging._console_stream() is None
    finally:
        close_extra_sinks(kept)


# --- a crash with nowhere to print it ----------------------------------------------------
def crash_hooks(logger, monkeypatch):
    """Install the hooks with monkeypatch holding both originals for the teardown."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    entry._install_crash_hooks(logger)


def test_an_unhandled_exception_in_the_main_thread_is_written_to_the_log(
    config, fresh_logger, monkeypatch
):
    """Without this the launcher's message box shows a log that never mentions the crash.

    The operator has no terminal: he gets "JARVIS stopped unexpectedly" over fifteen
    lines of perfectly ordinary logging that stop mid-sentence, and no way to tell why.
    """
    hide_console(monkeypatch)
    logger = setup_logging(config)
    crash_hooks(logger, monkeypatch)

    try:
        raise RuntimeError("the reactor housing is cracked")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())

    written = Path(config.get("logging.file")).read_text(encoding="utf-8")
    assert "the reactor housing is cracked" in written
    assert "Traceback (most recent call last)" in written


def test_an_unhandled_exception_in_a_thread_is_written_to_the_log(
    config, fresh_logger, monkeypatch
):
    """The audio, mind and desk threads all die this way, and sys.excepthook never sees it."""
    hide_console(monkeypatch)
    logger = setup_logging(config)
    crash_hooks(logger, monkeypatch)

    def die() -> None:
        raise RuntimeError("the capture loop has stopped")

    thread = threading.Thread(target=die, name="audio")
    thread.start()
    thread.join(timeout=5.0)

    written = Path(config.get("logging.file")).read_text(encoding="utf-8")
    assert "the capture loop has stopped" in written
    assert "thread audio" in written
    assert "Traceback (most recent call last)" in written


def test_the_terminal_still_gets_its_traceback_when_there_is_a_terminal(
    config, fresh_logger, monkeypatch
):
    """Logging the crash must add to the console output, not replace it."""
    printed: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: printed.append(args))
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    entry._install_crash_hooks(setup_logging(config))

    try:
        raise RuntimeError("something is on fire")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())

    assert len(printed) == 1


def test_a_keyboard_interrupt_is_not_logged_as_a_crash(config, fresh_logger, monkeypatch):
    """Ctrl+C is the operator's decision; CRITICAL in the log would be a lie."""
    hide_console(monkeypatch)
    logger = setup_logging(config)
    crash_hooks(logger, monkeypatch)

    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())

    assert "CRITICAL" not in Path(config.get("logging.file")).read_text(encoding="utf-8")


# --- whose windows these are -------------------------------------------------------------
def fake_windll(calls: list[str]) -> SimpleNamespace:
    """Just enough of ctypes.windll to record the one call we care about."""
    return SimpleNamespace(
        shell32=SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=calls.append)
    )


def test_the_app_user_model_id_is_never_attempted_off_windows(monkeypatch):
    """There is no ctypes.windll on Linux; reaching for it is a bug, not a fallback."""
    import ctypes

    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "windll", fake_windll(calls), raising=False)

    assert entry._set_app_user_model_id() is False
    assert calls == []


def test_the_app_user_model_id_is_claimed_on_windows(monkeypatch):
    """Otherwise the taskbar and Alt-Tab file every window under the Python logo."""
    import ctypes

    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", fake_windll(calls), raising=False)

    assert entry._set_app_user_model_id() is True
    assert calls == ["Jarvis.Desk"]


def test_a_shell_that_refuses_the_identity_costs_the_assistant_nothing(monkeypatch):
    """An icon is never worth a crash: the call is cosmetic and its failure is logged."""
    import ctypes

    def refuse(_app_id: str) -> None:
        raise OSError("the shell is not listening")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes, "windll",
        SimpleNamespace(shell32=SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=refuse)),
        raising=False,
    )

    assert entry._set_app_user_model_id() is False


def test_main_claims_the_identity_before_it_builds_anything_with_a_window():
    """The shell reads the identity once, when the first window appears - not after."""
    body = (ROOT / "jarvis" / "__main__.py").read_text(encoding="utf-8")
    body = body[body.index("def main("):]

    assert body.index("_set_app_user_model_id(") < body.index("Assistant(cfg")


# --- no child process may open a console -------------------------------------------------
def spawning_calls(source: str) -> list[tuple[int, str, bool]]:
    """Every process launch in ``source`` as ``(line, what, hides its window)``.

    ``**_popen_kwargs()`` counts as hiding it: that helper is the house pattern for
    "CREATE_NO_WINDOW on Windows, nothing anywhere else", because passing the flag on
    a POSIX machine is a ValueError rather than a no-op.
    """
    found: list[tuple[int, str, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name):
            continue
        if owner.id == "subprocess" and node.func.attr in LAUNCHERS:
            hidden = any(kw.arg in (None, "creationflags") for kw in node.keywords)
            found.append((node.lineno, f"subprocess.{node.func.attr}", hidden))
        elif owner.id == "os" and node.func.attr in ("system", "popen"):
            # Neither takes creationflags at all, so neither can ever be made quiet.
            found.append((node.lineno, f"os.{node.func.attr}", False))
    return found


def test_nothing_jarvis_can_spawn_flashes_a_console_window():
    """One black rectangle per spoken sentence is the complaint this feature exists to fix.

    Under pythonw the assistant has no console of his own, so any child process started
    without CREATE_NO_WINDOW gets a brand new one: it appears over whatever the operator
    is doing and vanishes again. This walks the tree rather than naming call sites, so
    the next one written is caught the day it is written.
    """
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in CONSOLE_PROGRAMS:
                continue
            for line, what, hidden in spawning_calls(path.read_text(encoding="utf-8")):
                if not hidden:
                    offenders.append(f"{relative}:{line} {what}")

    assert offenders == [], (
        "these start a process without CREATE_NO_WINDOW; pass **_popen_kwargs() or add "
        f"the file to CONSOLE_PROGRAMS if it really wants a console: {offenders}"
    )


def test_the_sweep_would_notice_a_bare_subprocess_call():
    """A test that cannot fail proves nothing; this is the negative it is meant to catch."""
    bare = spawning_calls("import subprocess\nsubprocess.run(['piper'], check=False)\n")
    hidden = spawning_calls("import subprocess\nsubprocess.run(['piper'], **_popen_kwargs())\n")

    assert bare == [(2, "subprocess.run", False)]
    assert hidden == [(2, "subprocess.run", True)]


def test_the_piper_command_line_hides_its_window_on_windows(monkeypatch):
    """Piper's fallback runs once per spoken sentence - the most visible console of all."""
    from jarvis.tts import piper_tts

    monkeypatch.setattr(sys, "platform", "win32")

    assert piper_tts._popen_kwargs() == {"creationflags": piper_tts.CREATE_NO_WINDOW}


def test_the_piper_command_line_passes_no_creationflags_off_windows(monkeypatch):
    """POSIX raises ValueError on the flag, so the Swedish voice must not send it."""
    from jarvis.tts import piper_tts

    monkeypatch.setattr(sys, "platform", "linux")

    assert piper_tts._popen_kwargs() == {}


def test_the_piper_back_end_actually_passes_the_flag_to_subprocess(monkeypatch, tmp_path):
    """The helper is only worth having if the call site uses it."""
    from jarvis.tts import piper_tts

    model = tmp_path / "sv_SE-nst-medium.onnx"
    model.write_bytes(b"not really a model")
    captured: dict[str, object] = {}

    def record(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no voice here")

    monkeypatch.setattr(piper_tts.shutil, "which", lambda _name: "/usr/bin/piper")
    monkeypatch.setattr(subprocess, "run", record)
    engine = piper_tts.PiperTTS(model_path=model)
    monkeypatch.setattr(sys, "platform", "win32")

    assert engine._synthesize_cli("Godkväll, sir.") is None
    assert captured.get("creationflags") == piper_tts.CREATE_NO_WINDOW


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


def test_the_launcher_hands_the_log_tail_the_whole_buffer():
    """jarvis_log_tail writes at most count - 1 characters, so count is the buffer length.

    UTF-8 never expands into UTF-16, but it does not shrink either: a short, all-ASCII
    log - exactly what a crash seconds after a fresh start leaves behind - needs one
    wide character per byte read. Told the buffer is one character smaller than it is,
    MultiByteToWideChar fails outright and the message box says the log could not be
    read at all, which is the one thing it must never say when the log is right there.
    """
    code = strip_comments(LAUNCHER_C.read_text(encoding="utf-8"))
    declared = re.search(r"wchar_t\s+tail\[([^\]]+)\]\s*;", code)
    passed = re.search(r"jarvis_log_tail\(\s*log_path\s*,\s*tail\s*,\s*([^)]+)\)", code)

    assert declared is not None and passed is not None
    assert " ".join(passed.group(1).split()) == " ".join(declared.group(1).split())


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
