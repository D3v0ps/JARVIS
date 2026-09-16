"""Finding an application by the name a person actually says.

"start spotify" only works for something on the PATH, which most applications are
not: Windows shows "Windows cannot find 'spotify'" and start still exits zero, so
JARVIS used to announce a launch that never happened. These tests pin the matching
that replaced it, and the honesty that goes with it.
"""

from __future__ import annotations

import pytest

from jarvis.tools.windows_apps import AppTarget, _best_match, resolve, start_apps

INSTALLED = [
    "Spotify",
    "Google Chrome",
    "Visual Studio Code",
    "Steam",
    "Discord",
    "Settings",
    "Calculator",
    "Task Manager",
    "Microsoft Edge",
    "Spotify Web Helper",
]


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("spotify", "Spotify"),
        ("Spotify", "Spotify"),
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("chrome browser", "Google Chrome"),
        ("code", "Visual Studio Code"),
        ("vs code", "Visual Studio Code"),
        ("visual studio code", "Visual Studio Code"),
        ("steam", "Steam"),
        ("the calculator", "Calculator"),
        ("task manager", "Task Manager"),
        ("edge", "Microsoft Edge"),
        ("discrod", "Discord"),          # whisper mishears, and it still lands
    ],
)
def test_the_name_people_say_finds_the_app_they_mean(spoken, expected):
    index = _best_match(spoken.lower(), INSTALLED)
    assert index is not None, f"{spoken!r} matched nothing"
    assert INSTALLED[index] == expected


def test_an_exact_name_wins_over_a_longer_one_containing_it():
    """'spotify' must not open Spotify Web Helper."""
    index = _best_match("spotify", INSTALLED)
    assert INSTALLED[index] == "Spotify"


def test_the_shortest_containing_name_wins():
    apps = ["Spotify Web Helper", "Spotify Premium Installer"]
    index = _best_match("spotify", apps)
    assert apps[index] == "Spotify Web Helper"


def test_something_that_is_not_installed_matches_nothing():
    assert _best_match("a program nobody has", INSTALLED) is None


def test_resolve_is_quiet_off_windows():
    """No Start menu here, so it must return None rather than raise."""
    assert resolve("spotify") is None
    assert start_apps() == []


def test_an_apps_folder_target_launches_through_explorer(monkeypatch):
    launched: list[list[str]] = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            launched.append(argv)

    monkeypatch.setattr("jarvis.tools.windows_apps.subprocess.Popen", FakePopen)
    how = AppTarget(kind="appsfolder", target="Spotify.exe", label="Spotify").launch()

    assert how == "shell:AppsFolder"
    assert launched == [["explorer.exe", "shell:AppsFolder\\Spotify.exe"]]


def test_open_app_does_not_claim_success_when_nothing_was_found(config, memory):
    """The bug that started this: a launch that never happened, reported as done."""
    import logging

    from jarvis.core.scheduler import Scheduler
    from jarvis.core.state import StateBus
    from jarvis.tools import registry
    from jarvis.tools.base import ToolContext

    registry.load_all()
    ctx = ToolContext(
        config=config, memory=memory, logger=logging.getLogger("test"),
        speak=lambda s: None, confirm=lambda s: True, notify=lambda s: None,
        scheduler=Scheduler(on_due=lambda job: None), state=StateBus(),
    )
    result = registry.get("open_app").func(ctx, {"name": "something nobody has installed"})

    assert result.ok is False
    assert "sir" in result.summary.lower()
