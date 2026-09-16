"""The four ways the desk window can be hosted, and the one way it may fail.

There is no GUI in the machine that runs this suite, no Edge, and no pywebview, which
is precisely the situation the window is written for: it must choose a host, and it
must survive not finding one. So every host here is a fake behind the module's two
lookup seams, and the questions asked of it are the ones a broken installation would
ask - is the order right, does a forced mode stay forced, does a missing host fall
through rather than raise, and does the operator's own Edge profile stay untouched.

The one behaviour worth naming: closing the window hides it. If that veto ever stops
working, the close button silently ends the window for the rest of the session and
the tray has nothing left to bring back.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from jarvis.desk import window as window_mod
from jarvis.desk.window import CHAIN, DeskWindow

URL = "http://127.0.0.1:53535/?t=ticket"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


# --- the fakes -----------------------------------------------------------------------
class FakeEvent:
    """pywebview's Event: subscribes in place with ``+=`` and hands itself back."""

    def __init__(self) -> None:
        self.handlers: list[Callable[..., Any]] = []

    def __iadd__(self, handler: Callable[..., Any]) -> "FakeEvent":
        self.handlers.append(handler)
        return self

    def fire(self, *args: Any) -> list[Any]:
        return [handler(*args) for handler in self.handlers]


class FakeEvents:
    def __init__(self) -> None:
        self.closing = FakeEvent()
        self.moved = FakeEvent()
        self.resized = FakeEvent()


class FakeWindow:
    def __init__(self, title: str, **kwargs: Any) -> None:
        self.title = title
        self.kwargs = kwargs
        self.events = FakeEvents()
        self.calls: list[str] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")

    def destroy(self) -> None:
        self.calls.append("destroy")


class FakeWebview:
    """The ``webview`` module with the GUI taken out of it."""

    def __init__(self) -> None:
        self.windows: list[FakeWindow] = []
        self.started = 0
        self.error: Exception | None = None

    def create_window(self, title: str, **kwargs: Any) -> FakeWindow:
        if self.error is not None:
            raise self.error
        made = FakeWindow(title, **kwargs)
        self.windows.append(made)
        return made

    def start(self, *args: Any, **kwargs: Any) -> None:
        self.started += 1


class FakeProcess:
    """Edge, as far as anything here can tell."""

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = list(command)
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = 0

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    def close_yourself(self) -> None:
        """What the operator clicking the X looks like from this side."""
        self.returncode = 0


class Hosts:
    """All three outside-world lookups in one place, each recording what it was asked."""

    def __init__(self) -> None:
        self.webview: FakeWebview | None = FakeWebview()
        self.webview_error: Exception | None = None
        self.webview_asked = 0
        self.edge_exe: str | None = EDGE_EXE
        self.edge_asked = 0
        self.launched: list[FakeProcess] = []
        self.opened: list[str] = []
        self.browser_answer = True

    def load_webview(self) -> Any:
        self.webview_asked += 1
        if self.webview_error is not None:
            raise self.webview_error
        return self.webview

    def find_edge(self) -> str | None:
        self.edge_asked += 1
        return self.edge_exe

    def popen(self, command: list[str], **kwargs: Any) -> FakeProcess:
        process = FakeProcess(command, **kwargs)
        self.launched.append(process)
        return process

    def open_browser(self, url: str) -> bool:
        self.opened.append(url)
        return self.browser_answer


# --- fixtures ------------------------------------------------------------------------
@pytest.fixture
def hosts(monkeypatch):
    """Every host present and faked; a test switches off the ones it wants missing."""
    made = Hosts()
    monkeypatch.setattr(window_mod, "load_webview", made.load_webview)
    monkeypatch.setattr(window_mod, "find_edge", made.find_edge)
    monkeypatch.setattr(window_mod.subprocess, "Popen", made.popen)
    monkeypatch.setattr(window_mod.webbrowser, "open", made.open_browser)
    return made


@pytest.fixture
def desk_config(config):
    config.set("ui.window_mode", "auto")
    config.set("ui.window_size", [980, 720])
    config.set("ui.window_pos", None)
    config.set("ui.window_on_top", False)
    return config


@pytest.fixture
def make_window(desk_config, hosts):
    """Builds windows and closes them afterwards, whatever the test did to them."""
    built: list[DeskWindow] = []

    def build(**kwargs: Any) -> DeskWindow:
        made = DeskWindow(
            desk_config, URL, logger=logging.getLogger("jarvis.test.desk"), **kwargs
        )
        built.append(made)
        return made

    yield build
    for made in built:
        made.stop()


def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    """Give a debounced background timer time to fire, without sleeping blindly."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- the fallback chain ---------------------------------------------------------------
def test_pywebview_is_the_first_host_tried_and_nothing_below_it_is_disturbed(
    make_window, hosts
):
    """WebView2 is the only host that gives us a window of our own design, so it wins."""
    window = make_window()

    assert window.start() is True
    assert window.backend == "webview"
    assert hosts.edge_asked == 0
    assert hosts.launched == []
    assert hosts.opened == []


def test_a_missing_pywebview_falls_through_to_edges_app_mode(make_window, hosts):
    """Most machines will not have pywebview; Edge app mode shows the same page."""
    hosts.webview = None
    window = make_window()

    assert window.start() is True
    assert window.backend == "edge"
    assert len(hosts.launched) == 1
    assert hosts.opened == []


def test_a_pywebview_that_raises_on_import_falls_through_to_edge(make_window, hosts):
    """A half-installed pywebview raises rather than returning None. Same outcome."""
    hosts.webview_error = ImportError("no module named webview")
    window = make_window()

    assert window.start() is True
    assert window.backend == "edge"


def test_an_edge_that_is_not_installed_falls_through_to_the_default_browser(
    make_window, hosts
):
    """A tab with chrome around it is still the app; no window at all is not."""
    hosts.webview = None
    hosts.edge_exe = None
    window = make_window()

    assert window.start() is True
    assert window.backend == "browser"
    assert hosts.opened == [URL]


def test_with_no_host_at_all_the_window_gives_up_quietly_and_leaves_the_ring(
    make_window, hosts
):
    """The assistant must be exactly as usable with the whole window absent."""
    hosts.webview = None
    hosts.edge_exe = None
    hosts.browser_answer = False
    window = make_window()

    assert window.start() is False
    assert window.backend == "none"
    assert window.needs_main_thread is False


def test_a_host_that_explodes_is_never_raised_into_the_assistant(make_window, hosts):
    """A GUI toolkit failing mid-construction is a fallback, not a crashed assistant."""
    assert hosts.webview is not None
    hosts.webview.error = RuntimeError("no display, and no GTK either")
    window = make_window()

    assert window.start() is True
    assert window.backend == "edge"


# --- window_mode ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", CHAIN)
def test_each_window_mode_forces_exactly_one_host(make_window, desk_config, hosts, mode):
    """Asking for a host and being given a different one is a lie about the screen."""
    desk_config.set("ui.window_mode", mode)
    window = make_window()

    assert window.start() is True
    assert window.backend == mode
    assert (hosts.webview_asked > 0) is (mode == "webview")
    assert (len(hosts.launched) > 0) is (mode == "edge")
    assert (hosts.opened == [URL]) is (mode == "browser")


def test_a_forced_host_that_is_absent_does_not_quietly_fall_through(
    make_window, desk_config, hosts
):
    """``window_mode: edge`` with no Edge means no window, not a surprise pywebview."""
    desk_config.set("ui.window_mode", "edge")
    hosts.edge_exe = None
    window = make_window()

    assert window.start() is False
    assert window.backend == "none"
    assert hosts.webview_asked == 0
    assert hosts.opened == []


def test_window_mode_off_tries_nothing_at_all(make_window, desk_config, hosts):
    """Off means off: no import, no process, no tab, and nothing to report available."""
    desk_config.set("ui.window_mode", "off")
    window = make_window()

    assert window.available() is False
    assert window.start() is False
    assert window.backend == "none"
    assert (hosts.webview_asked, hosts.edge_asked, hosts.launched, hosts.opened) == (
        0,
        0,
        [],
        [],
    )


def test_an_unknown_window_mode_is_treated_as_auto(make_window, desk_config):
    """A typo in the config must not cost the operator his window."""
    desk_config.set("ui.window_mode", "webview2")
    window = make_window()

    assert window.mode == "auto"
    assert window.start() is True
    assert window.backend == "webview"


def test_available_answers_for_the_host_that_was_demanded_without_opening_it(
    make_window, desk_config, hosts
):
    """``available()`` is asked before anything is opened, so it may open nothing."""
    desk_config.set("ui.window_mode", "webview")
    hosts.webview = None
    absent = make_window()

    assert absent.available() is False
    assert absent.backend == "none"
    assert hosts.launched == [] and hosts.opened == []

    hosts.webview = FakeWebview()
    present = make_window()

    assert present.available() is True
    assert hosts.webview.windows == [], "asking must not open a window"


# --- the main thread -------------------------------------------------------------------
def test_only_the_webview_host_needs_the_main_thread(make_window, hosts):
    """pywebview owns a message loop; a launched Edge and an opened tab own nothing."""
    webview_window = make_window()
    webview_window.start()
    assert webview_window.needs_main_thread is True

    hosts.webview = None
    edge_window = make_window()
    edge_window.start()
    assert edge_window.backend == "edge"
    assert edge_window.needs_main_thread is False


def test_run_enters_the_pywebview_loop_and_returns_at_once_for_any_other_host(
    make_window, hosts
):
    """``run()`` blocks only where there is something to block on."""
    window = make_window()
    window.start()

    window.run()

    assert hosts.webview is not None and hosts.webview.started == 1

    hosts.webview = None
    edge_window = make_window()
    edge_window.start()
    edge_window.run()  # returns, or this test never finishes


# --- the pywebview window itself --------------------------------------------------------
def test_the_window_is_frameless_draggable_and_on_the_desks_own_background(
    make_window, hosts
):
    """Windows chrome around a page with its own title bar would be two title bars."""
    window = make_window()
    window.start()

    assert hosts.webview is not None
    opened = hosts.webview.windows[0]
    assert opened.title == "J.A.R.V.I.S."
    assert opened.kwargs["url"] == URL
    assert opened.kwargs["frameless"] is True
    assert opened.kwargs["easy_drag"] is True
    assert opened.kwargs["resizable"] is True
    assert opened.kwargs["background_color"] == "#05080c"
    assert opened.kwargs["min_size"] == (640, 480)


def test_the_window_opens_where_and_how_big_the_config_remembers(
    make_window, desk_config, hosts
):
    """A window that forgets where it was is a window you reposition every morning."""
    desk_config.set("ui.window_size", [1200, 860])
    desk_config.set("ui.window_pos", [140, 60])
    desk_config.set("ui.window_on_top", True)
    window = make_window()
    window.start()

    assert hosts.webview is not None
    kwargs = hosts.webview.windows[0].kwargs
    assert (kwargs["width"], kwargs["height"]) == (1200, 860)
    assert (kwargs["x"], kwargs["y"]) == (140, 60)
    assert kwargs["on_top"] is True


def test_closing_the_window_hides_it_instead_of_ending_the_session(make_window, hosts):
    """The tray must be able to bring it back; the assistant is still listening."""
    closed: list[bool] = []
    window = make_window(on_close=lambda: closed.append(True))
    window.start()
    assert hosts.webview is not None
    opened = hosts.webview.windows[0]

    answers = opened.events.closing.fire()

    assert answers == [False], "returning False is what vetoes pywebview's close"
    assert "hide" in opened.calls
    assert window.visible is False
    assert closed == [True]


def test_stop_lets_the_window_close_for_real(make_window, hosts):
    """The same veto that saves the window from the X must not survive a shutdown."""
    window = make_window()
    window.start()
    assert hosts.webview is not None
    opened = hosts.webview.windows[0]

    window.stop()

    assert "destroy" in opened.calls
    assert opened.events.closing.fire() == [True]


def test_show_and_hide_move_the_webview_window_without_recreating_it(make_window, hosts):
    """Hiding is cheap; rebuilding a WebView2 window is not."""
    window = make_window()
    window.start()
    assert hosts.webview is not None
    opened = hosts.webview.windows[0]

    assert window.hide() is True
    assert window.toggle() is True

    assert opened.calls == ["hide", "show"]
    assert len(hosts.webview.windows) == 1


# --- remembering the geometry ------------------------------------------------------------
def test_moving_the_window_is_written_back_to_the_config(make_window, desk_config, hosts):
    """Same bargain as the ring: where you put it is where it is next time."""
    window = make_window()
    window._save_delay = 0.01
    window.start()
    assert hosts.webview is not None
    opened = hosts.webview.windows[0]

    opened.events.moved.fire(310, 128)

    assert wait_for(lambda: desk_config.get("ui.window_pos") == [310, 128])


def test_a_burst_of_resize_events_is_saved_once_with_the_final_size(
    make_window, desk_config, hosts, monkeypatch
):
    """A drag fires an event per frame; writing YAML per frame would hammer the disk."""
    saves: list[int] = []
    original = desk_config.save
    monkeypatch.setattr(desk_config, "save", lambda: (saves.append(1), original())[1])

    window = make_window()
    window._save_delay = 0.25
    window.start()
    assert hosts.webview is not None
    opened = hosts.webview.windows[0]

    for width in (1000, 1040, 1080):
        opened.events.resized.fire(width, 700)

    assert wait_for(lambda: desk_config.get("ui.window_size") == [1080, 700])
    assert len(saves) == 1


# --- Edge in app mode ----------------------------------------------------------------------
def test_edge_is_launched_in_app_mode_with_a_throwaway_profile_under_logs(
    make_window, desk_config, hosts
):
    """Borrowing the operator's own profile would hand the page his cookies and history."""
    hosts.webview = None
    desk_config.set("ui.window_size", [1010, 740])
    window = make_window()
    window.start()

    command = hosts.launched[0].command
    assert command[0] == EDGE_EXE
    assert f"--app={URL}" in command
    assert "--no-first-run" in command
    assert "--window-size=1010,740" in command

    profile = next(a for a in command if a.startswith("--user-data-dir="))
    directory = Path(profile.split("=", 1)[1])
    assert directory.parent.name == "logs"
    assert directory.is_relative_to(Path(desk_config.path).resolve().parent)
    assert directory.is_dir(), "Edge is handed a directory that exists"
    assert "User Data" not in " ".join(command)
    assert not any(a.startswith("--profile-directory") for a in command)


def test_stop_kills_the_edge_process_it_started_and_is_safe_to_call_twice(
    make_window, hosts
):
    """Shutdown must leave no orphaned browser, and must not mind being asked twice."""
    hosts.webview = None
    window = make_window()
    window.start()
    process = hosts.launched[0]

    window.stop()
    window.stop()

    assert process.terminated == 1
    assert window.backend == "none"


def test_showing_an_edge_window_the_operator_closed_starts_a_new_one(make_window, hosts):
    """Relaunching is the only honest ``show`` for a process we hold no handle inside."""
    hosts.webview = None
    window = make_window()
    window.start()
    hosts.launched[0].close_yourself()

    assert window.show() is True
    assert len(hosts.launched) == 2

    assert window.show() is True, "a live Edge window is already shown"
    assert len(hosts.launched) == 2


def test_hiding_the_edge_window_closes_it_and_toggle_brings_it_back(make_window, hosts):
    """Closing is the only hiding Edge offers, so toggle has to mean reopen."""
    hosts.webview = None
    window = make_window()
    window.start()

    assert window.hide() is True
    assert hosts.launched[0].terminated == 1

    assert window.toggle() is True
    assert len(hosts.launched) == 2


# --- the two hosts with no window to speak of --------------------------------------------
def test_show_and_hide_do_nothing_for_the_browser_and_report_that_they_did(
    make_window, hosts
):
    """A tab we opened is the user's to manage; pretending otherwise would be a lie."""
    hosts.webview = None
    hosts.edge_exe = None
    window = make_window()
    window.start()

    assert window.backend == "browser"
    assert window.show() is False
    assert window.hide() is False
    assert window.toggle() is False
    assert hosts.opened == [URL], "toggling must not open a second tab"


def test_every_control_is_a_harmless_no_op_when_there_is_no_window(make_window, hosts):
    """With no host at all, the assistant may still call all of this and carry on."""
    hosts.webview = None
    hosts.edge_exe = None
    hosts.browser_answer = False
    window = make_window()
    window.start()

    assert (window.show(), window.hide(), window.toggle()) == (False, False, False)
    window.run()
    window.stop()
