"""The clipboard tool, with ``win32clipboard`` faked through ``sys.modules``.

No Windows, no pywin32, no keyboard: a fake clipboard module records every call so the
tests can assert on the order of operations, and the Ctrl+C path is driven through a
fake ``INPUT``/``user32`` pair that stands in for ``ctypes.windll``.

The test that matters most is :func:`test_an_excluded_clipboard_is_refused_and_never_logged`.
A password manager's payload must never reach the result, the model or
``logs/jarvis.log``, and the only way to know that stays true is to assert it.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import types

import pytest

from jarvis.tools import clipboard_tools
from jarvis.tools.base import Tier, ToolContext
from jarvis.tools.clipboard_tools import (
    CF_DIB, CF_HDROP, CF_TEXT, CF_UNICODETEXT, PRIVATE_FORMATS, ClipboardBusy, clipboard,
)

SECRET = "hunter2-correct-horse-battery-staple"


# --- fakes ----------------------------------------------------------------------------
class FakeClipboard:
    """The slice of ``win32clipboard`` this module touches, and nothing more."""

    def __init__(self, *, text: str | None = None, files: tuple[str, ...] | None = None,
                 image: bytes | None = None, private: str | None = None,
                 history_flag: object = None, busy_times: int = 0,
                 open_error: Exception | None = None) -> None:
        self.text = text
        self.files = files
        self.image = image
        self.private = private          # a PRIVATE_FORMATS name that is on the board
        self.history_flag = history_flag  # payload of CanIncludeInClipboardHistory
        self.busy_times = busy_times
        self.open_error = open_error or OSError("clipboard busy")
        self.calls: list[str] = []
        self.open_attempts = 0
        self.is_open = False
        self._registered: dict[str, int] = {}
        self.fetched: list[int] = []

    # -- registration ------------------------------------------------------------------
    def RegisterClipboardFormat(self, name: str) -> int:  # noqa: N802 - Windows API name
        return self._registered.setdefault(name, 0xC000 + len(self._registered) + 1)

    def _format_id(self, name: str) -> int:
        return self.RegisterClipboardFormat(name)

    # -- open / close ------------------------------------------------------------------
    def OpenClipboard(self) -> None:  # noqa: N802
        self.open_attempts += 1
        if self.open_attempts <= self.busy_times:
            self.calls.append("open-failed")
            raise self.open_error
        self.calls.append("open")
        self.is_open = True

    def CloseClipboard(self) -> None:  # noqa: N802
        self.calls.append("close")
        self.is_open = False

    def EmptyClipboard(self) -> None:  # noqa: N802
        assert self.is_open, "EmptyClipboard outside an open clipboard"
        self.calls.append("empty")
        self.text = self.files = self.image = None

    def SetClipboardText(self, text: str, fmt: int = CF_UNICODETEXT) -> None:  # noqa: N802
        assert self.is_open, "SetClipboardText outside an open clipboard"
        self.calls.append("set-text")
        self.text = text

    # -- reading -----------------------------------------------------------------------
    def IsClipboardFormatAvailable(self, fmt: int) -> bool:  # noqa: N802
        assert self.is_open, "IsClipboardFormatAvailable outside an open clipboard"
        if self.private and fmt == self._format_id(self.private):
            return True
        if fmt == CF_UNICODETEXT:
            return self.text is not None
        if fmt == CF_HDROP:
            return self.files is not None
        if fmt == CF_DIB:
            return self.image is not None
        return False

    def GetClipboardData(self, fmt: int):  # noqa: N802
        assert self.is_open, "GetClipboardData outside an open clipboard"
        self.fetched.append(fmt)
        if self.private and fmt == self._format_id(self.private):
            return self.history_flag
        if fmt == CF_UNICODETEXT:
            return self.text
        if fmt == CF_HDROP:
            return self.files
        if fmt == CF_DIB:
            return self.image
        raise OSError(f"format {fmt} is not on the clipboard")


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16),
                ("dwFlags", ctypes.c_uint32)]


class FakeINPUT(ctypes.Structure):
    """A real ctypes structure, so ``(INPUT * n)(...)`` and ``sizeof`` behave."""

    _fields_ = [("type", ctypes.c_uint32), ("ki", _KEYBDINPUT)]


class FakeUser32:
    """Records SendInput calls and hands out clipboard sequence numbers."""

    def __init__(self, sequence: list[int] | None = None, accept: int | None = None) -> None:
        self.sent: list[list[tuple[int, int]]] = []
        self.sequence = sequence if sequence is not None else [10, 11]
        self.accept = accept

    def SendInput(self, count: int, array, size: int) -> int:  # noqa: N802
        self.sent.append([(array[i].ki.wVk, array[i].ki.dwFlags) for i in range(count)])
        return count if self.accept is None else self.accept

    def GetClipboardSequenceNumber(self) -> int:  # noqa: N802
        return self.sequence[0] if len(self.sequence) == 1 else self.sequence.pop(0)


def dib(width: int, height: int) -> bytes:
    """A BITMAPINFOHEADER with the given dimensions and a little pixel data."""
    return (b"\x28\x00\x00\x00"
            + int(width).to_bytes(4, "little", signed=True)
            + int(height).to_bytes(4, "little", signed=True)
            + b"\x01\x00\x20\x00" + b"\x00" * 64)


# --- fixtures -------------------------------------------------------------------------
def _delegate(fake: FakeClipboard, name: str):
    """A module-level function that resolves ``fake.<name>`` at call time."""

    def call(*args, **kwargs):
        return getattr(fake, name)(*args, **kwargs)

    return call


@pytest.fixture
def board(monkeypatch):
    """Install a fake ``win32clipboard`` and return a factory for its state."""
    holder: dict[str, FakeClipboard] = {}

    def install(**kwargs) -> FakeClipboard:
        fake = FakeClipboard(**kwargs)
        module = types.ModuleType("win32clipboard")
        for name in ("RegisterClipboardFormat", "OpenClipboard", "CloseClipboard",
                     "EmptyClipboard", "SetClipboardText", "IsClipboardFormatAvailable",
                     "GetClipboardData"):
            # Looked up on every call, so a test may swap one method out afterwards.
            setattr(module, name, _delegate(fake, name))
        monkeypatch.setitem(sys.modules, "win32clipboard", module)
        holder["fake"] = fake
        return fake

    monkeypatch.setattr(clipboard_tools.time, "sleep", lambda seconds: None)
    return install


@pytest.fixture
def ctx(config, memory):
    from jarvis.core.scheduler import Scheduler
    from jarvis.core.state import StateBus

    return ToolContext(config=config, memory=memory, logger=logging.getLogger("test"),
                       speak=lambda text: None, confirm=lambda text: True,
                       notify=lambda text: None,
                       scheduler=Scheduler(on_due=lambda job: None), state=StateBus())


@pytest.fixture
def keyboard(monkeypatch):
    """Stand in for ``ctypes.windll``: a real INPUT type and a recording user32."""

    def install(user32: FakeUser32) -> FakeUser32:
        monkeypatch.setattr(clipboard_tools, "_input_structures",
                            lambda: (FakeINPUT, user32))
        return user32

    monkeypatch.setattr(clipboard_tools, "SELECTION_POLL_SECONDS", 0.0)
    monkeypatch.setattr(clipboard_tools, "SELECTION_WAIT_SECONDS", 0.05)
    return install


# --- registration ---------------------------------------------------------------------
def test_the_tool_is_registered_as_safe_with_the_four_actions():
    from jarvis.tools import registry

    registry.load_all()
    spec = registry.get("clipboard")
    assert spec is not None, "the clipboard tool never registered"
    assert spec.tier is Tier.SAFE
    schema = spec.to_ollama()["function"]["parameters"]
    assert schema["properties"]["action"]["enum"] == list(clipboard_tools.CLIPBOARD_ACTIONS)


def test_the_module_imports_without_pywin32(monkeypatch):
    """The import-time rule: nothing Windows-only at module scope."""
    monkeypatch.delitem(sys.modules, "win32clipboard", raising=False)
    assert clipboard_tools._import_win32clipboard() is None


def test_without_pywin32_the_tool_says_so_instead_of_raising(ctx, monkeypatch):
    monkeypatch.setattr(clipboard_tools, "_import_win32clipboard", lambda: None)
    result = clipboard(ctx, {"action": "read"})
    assert result.ok is False
    assert "sir" in result.summary
    assert "\n" not in result.summary


# --- text -----------------------------------------------------------------------------
def test_short_text_is_read_out_and_handed_to_the_model(board, ctx):
    board(text="Kvartalsrapporten är klar.")
    result = clipboard(ctx, {"action": "read"})

    assert result.ok is True
    assert result.summary == "The clipboard says: Kvartalsrapporten är klar."
    assert result.data["text"] == "Kvartalsrapporten är klar."
    assert result.data["kind"] == "text"
    assert result.data["source"] == "clipboard"


def test_summarise_returns_the_text_without_calling_any_model(board, ctx):
    fake = board(text="A long memo. " * 40)
    result = clipboard(ctx, {"action": "summarise"})

    assert result.ok is True
    assert result.summary.count(".") == 1, "the spoken summary must be one sentence"
    assert "summarise" in result.summary
    assert result.data["text"].startswith("A long memo.")
    # Nothing was sent anywhere: only the clipboard was touched.
    assert fake.calls == ["open", "close"]


@pytest.mark.parametrize("action", ["translate", "explain", "summarize"])
def test_every_action_hands_the_text_over(board, ctx, action):
    board(text="Detta är en text.")
    result = clipboard(ctx, {"action": action})
    assert result.ok is True
    assert result.data["text"] == "Detta är en text."


def test_a_windows_carriage_return_does_not_reach_the_model(board, ctx):
    board(text="line one\r\nline two")
    result = clipboard(ctx, {"action": "read"})
    assert "\r" not in result.data["text"]


def test_long_text_is_truncated_hard_before_it_reaches_the_model(board, ctx):
    board(text="word " * 5000)
    result = clipboard(ctx, {"action": "summarise"})

    assert len(result.data["text"]) == clipboard_tools.MAX_TEXT_CHARS
    assert result.data["truncated"] is True
    assert result.data["characters"] == 25000
    assert "trimmed" in result.summary


def test_legacy_ansi_text_is_decoded_when_there_is_no_unicode_format(board, ctx):
    fake = board()
    fake.text = None
    fake.IsClipboardFormatAvailable = lambda fmt: fmt == CF_TEXT  # noqa: N803
    fake.GetClipboardData = lambda fmt: b"ansi only"
    result = clipboard(ctx, {"action": "read"})
    assert result.data["text"] == "ansi only"


def test_whitespace_only_text_counts_as_an_empty_clipboard(board, ctx):
    board(text="   \n\t ")
    result = clipboard(ctx, {"action": "read", "source": "clipboard"})
    assert result.ok is True
    assert result.summary == "There's nothing on the clipboard, sir."


# --- files and images -------------------------------------------------------------------
def test_a_copied_file_list_is_named_not_opened(board, ctx):
    board(files=(r"C:\Users\k\Desktop\report.pdf", r"C:\Users\k\Desktop\notes.txt"))
    result = clipboard(ctx, {"action": "read"})

    assert result.summary == "The clipboard holds two files, sir, starting with report.pdf."
    assert result.data["files"][0].endswith("report.pdf")
    assert result.data["file_count"] == 2
    assert "text" not in result.data


def test_one_copied_file_is_spoken_in_the_singular(board, ctx):
    board(files=(r"C:\Users\k\Downloads\faktura.pdf",))
    result = clipboard(ctx, {"action": "read"})
    assert result.summary == "The clipboard holds one file, sir: faktura.pdf."


def test_an_image_is_described_and_never_decoded(board, ctx):
    board(image=dib(1920, -1080))
    result = clipboard(ctx, {"action": "explain"})

    assert result.ok is True
    assert result.summary == ("The clipboard holds an image, sir, one thousand nine "
                              "hundred and twenty by one thousand and eighty pixels.")
    assert "text" not in result.data
    assert result.data["width"] == 1920 and result.data["height"] == 1080


def test_an_unreadable_image_header_still_gets_an_honest_answer(board, ctx):
    board(image=b"\x28\x00")
    result = clipboard(ctx, {"action": "read"})
    assert result.summary == "The clipboard holds an image, sir, not text I can read."


def test_an_empty_clipboard_is_not_reported_as_a_failure(board, ctx):
    board()
    result = clipboard(ctx, {"action": "read", "source": "clipboard"})
    assert result.ok is True
    assert result.summary == "There's nothing on the clipboard, sir."


# --- the retry loop ---------------------------------------------------------------------
def test_a_clipboard_held_by_another_process_is_retried(board, ctx):
    fake = board(text="finally", busy_times=3)
    result = clipboard(ctx, {"action": "read"})

    assert fake.open_attempts == 4, "OpenClipboard was not retried"
    assert result.ok is True
    assert result.data["text"] == "finally"


def test_a_clipboard_that_never_opens_fails_calmly(board, ctx):
    fake = board(text="never seen", busy_times=99)
    result = clipboard(ctx, {"action": "read"})

    assert fake.open_attempts == clipboard_tools.OPEN_ATTEMPTS
    assert result.ok is False
    assert result.summary.startswith("Something else is holding the clipboard")
    assert "never seen" not in result.detail


def test_the_clipboard_is_closed_even_when_reading_explodes(board, ctx):
    fake = board(text="boom")

    def explode(fmt):
        raise OSError("COM went sideways")

    fake.GetClipboardData = explode
    result = clipboard(ctx, {"action": "read"})

    assert result.ok is False
    assert fake.calls[-1] == "close", "the clipboard was left open"
    assert fake.is_open is False


def test_open_retry_gives_up_with_clipboard_busy(board):
    fake = board(busy_times=99)

    with pytest.raises(ClipboardBusy):
        with clipboard_tools._clipboard_open(sys.modules["win32clipboard"]):
            pass
    assert fake.open_attempts == clipboard_tools.OPEN_ATTEMPTS


# --- the privacy rule -------------------------------------------------------------------
@pytest.mark.parametrize("fmt_name", list(PRIVATE_FORMATS))
def test_an_excluded_clipboard_is_refused_and_never_logged(board, ctx, caplog, fmt_name):
    """A password manager's payload must not reach the result, the model or the log."""
    fake = board(text=SECRET, private=fmt_name, history_flag=b"\x00\x00\x00\x00")

    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        result = clipboard(ctx, {"action": "read"})

    assert result.refused is True
    assert result.ok is False
    assert result.summary == clipboard_tools.PRIVACY_SUMMARY
    assert result.data is None
    blob = f"{result.summary} {result.detail} {caplog.text}"
    assert SECRET not in blob, "the secret escaped into the result or the log"
    assert CF_UNICODETEXT not in fake.fetched, "the text was fetched despite the marker"


def test_history_flag_set_to_one_is_an_explicit_permission(board, ctx):
    board(text="ordinary text", private="CanIncludeInClipboardHistory", history_flag=1)
    result = clipboard(ctx, {"action": "read"})

    assert result.ok is True
    assert result.refused is False
    assert result.data["text"] == "ordinary text"


def test_an_unreadable_history_flag_fails_closed(board, ctx):
    fake = board(text=SECRET, private="CanIncludeInClipboardHistory")

    def explode(fmt):
        if fmt == fake._format_id("CanIncludeInClipboardHistory"):
            raise OSError("cannot read the flag")
        return fake.text

    fake.GetClipboardData = explode
    result = clipboard(ctx, {"action": "read"})

    assert result.refused is True
    assert SECRET not in f"{result.summary} {result.detail}"


def test_a_format_that_cannot_be_registered_does_not_block_an_ordinary_read(board, ctx):
    fake = board(text="ordinary")
    fake.RegisterClipboardFormat = lambda name: 0
    result = clipboard(ctx, {"action": "read"})
    assert result.ok is True and result.data["text"] == "ordinary"


# --- the selection path -------------------------------------------------------------------
def test_an_empty_clipboard_falls_back_to_copying_the_selection(board, ctx, keyboard):
    fake = board(text=None)
    user32 = keyboard(FakeUser32(sequence=[10, 11]))

    def copy_on_ctrl_c(count, array, size):
        fake.text = "the selected sentence"
        return count

    user32.SendInput = copy_on_ctrl_c
    result = clipboard(ctx, {"action": "explain"})

    assert result.ok is True
    assert result.data["text"] == "the selected sentence"
    assert result.data["source"] == "selection"
    assert "selection" in result.summary


def test_the_selection_path_sends_a_real_control_c(board, ctx, keyboard):
    fake = board(text=None)
    user32 = keyboard(FakeUser32(sequence=[1, 2]))
    original_send = user32.SendInput

    def copy(count, array, size):
        accepted = original_send(count, array, size)
        fake.text = "selected"
        return accepted

    user32.SendInput = copy
    clipboard(ctx, {"action": "read"})

    assert user32.sent == [[(0x11, 0), (0x43, 0), (0x43, 0x0002), (0x11, 0x0002)]]


def test_the_previous_clipboard_is_restored_after_a_selection_read(board, ctx, keyboard):
    fake = board(text="what the user had copied earlier")
    user32 = keyboard(FakeUser32(sequence=[5, 6]))

    def copy(count, array, size):
        fake.text = "the highlighted words"
        return count

    user32.SendInput = copy
    result = clipboard(ctx, {"action": "translate", "source": "selection"})

    assert result.data["text"] == "the highlighted words"
    assert fake.text == "what the user had copied earlier", "the clipboard was not restored"
    assert "empty" in fake.calls and "set-text" in fake.calls


def test_nothing_selected_is_detected_by_the_sequence_number(board, ctx, keyboard):
    board(text=None)
    keyboard(FakeUser32(sequence=[42]))  # the counter never moves
    result = clipboard(ctx, {"action": "read"})

    assert result.ok is True
    assert result.summary == "There's nothing copied and nothing selected, sir."
    assert result.data["kind"] == "empty"


def test_a_window_that_refuses_input_is_reported_honestly(board, ctx, keyboard):
    board(text=None)
    keyboard(FakeUser32(sequence=[1, 2], accept=0))
    result = clipboard(ctx, {"action": "read"})

    assert result.summary == "There's nothing copied and nothing selected, sir."
    assert "refused input" in result.detail


def test_without_sendinput_the_selection_path_degrades(board, ctx, monkeypatch):
    board(text=None)

    def no_windll():
        raise AttributeError("module 'ctypes' has no attribute 'windll'")

    monkeypatch.setattr(clipboard_tools, "_input_structures", no_windll)
    result = clipboard(ctx, {"action": "read"})

    assert result.ok is True
    assert result.summary == "There's nothing copied and nothing selected, sir."
    assert "SendInput unavailable" in result.detail


def test_a_selection_that_is_marked_private_is_refused_too(board, ctx, keyboard, caplog):
    fake = board(text=None)
    user32 = keyboard(FakeUser32(sequence=[1, 2]))

    def copy(count, array, size):
        fake.text = SECRET
        fake.private = "ExcludeClipboardContentFromMonitorProcessing"
        return count

    user32.SendInput = copy
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        result = clipboard(ctx, {"action": "read", "source": "selection"})

    assert result.refused is True
    assert SECRET not in f"{result.summary} {result.detail} {caplog.text}"


def test_copied_files_are_not_overwritten_by_a_selection_request(board, ctx, keyboard):
    fake = board(files=(r"C:\Users\k\Desktop\report.pdf",))
    keyboard(FakeUser32(sequence=[1, 2]))
    result = clipboard(ctx, {"action": "read", "source": "selection"})

    assert result.ok is False
    assert "overwrite" in result.summary
    assert fake.files == (r"C:\Users\k\Desktop\report.pdf",), "the file list was destroyed"


def test_source_clipboard_never_touches_the_keyboard(board, ctx, monkeypatch):
    board(text=None)
    monkeypatch.setattr(clipboard_tools, "_input_structures",
                        lambda: pytest.fail("the keyboard was used for source=clipboard"))
    result = clipboard(ctx, {"action": "read", "source": "clipboard"})
    assert result.summary == "There's nothing on the clipboard, sir."


def test_an_unknown_source_falls_back_to_auto(board, ctx):
    board(text="copied")
    result = clipboard(ctx, {"action": "read", "source": "telepathy"})
    assert result.ok is True and result.data["source"] == "clipboard"


# --- arguments ----------------------------------------------------------------------------
def test_an_unknown_action_is_refused_before_the_clipboard_is_touched(board, ctx):
    fake = board(text="secret enough")
    result = clipboard(ctx, {"action": "email it to my boss"})

    assert result.ok is False
    assert fake.calls == [], "the clipboard was opened for an unsupported action"


def test_a_missing_action_defaults_to_reading(board, ctx):
    board(text="just read it")
    result = clipboard(ctx, {})
    assert result.ok is True
    assert result.data["action"] == "read"


def test_spoken_summaries_never_contain_markdown_or_newlines(board, ctx):
    board(text="# A heading\n\n- bullet one\n- bullet two\n\n**bold** text")
    result = clipboard(ctx, {"action": "read"})

    assert "\n" not in result.summary
    assert "#" not in result.summary and "*" not in result.summary
    assert "- " not in result.summary
