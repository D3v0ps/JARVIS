"""What Check-JARVIS.bat says about the two faces that reach past the desk.

The window and the phone are the parts with no console behind them: when they do not
appear, this report is the only thing an operator has to read. So it has to be right
about which host would draw the window, and about where the phone should be pointed.
"""
from __future__ import annotations

import pytest

from jarvis.config import Config
from jarvis.preflight import check_remote, check_window


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ui:\n  window: true\nremote:\n  enabled: false\n", encoding="utf-8")
    return Config.load(path)


def named(results, name):
    found = [r for r in results if r.name == name]
    assert found, f"no {name!r} line in {[r.name for r in results]}"
    return found[0]


# --- the phone ---------------------------------------------------------------------
def test_the_phone_is_not_mentioned_until_it_is_asked_for(cfg):
    """A report full of lines about a feature that is off is a report nobody reads."""
    assert check_remote(cfg) == []


def test_it_prints_the_address_to_type_into_the_phone(cfg):
    cfg.set("remote.enabled", True)
    cfg.set("remote.url", "https://g.tail1234.ts.net")

    line = named(check_remote(cfg), "Phone address")

    assert line.ok is True
    assert line.detail == "https://g.tail1234.ts.net"


def test_a_server_with_nowhere_to_reach_it_is_a_warning(cfg):
    """remote.enabled with no door is the state that looks fine and does nothing."""
    cfg.set("remote.enabled", True)

    line = named(check_remote(cfg), "Phone address")

    assert line.ok is False
    assert line.essential is False, "he can still talk to the desk; this is not fatal"
    assert "Enable-Phone.bat" in line.fix


def test_loopback_is_correct_rather_than_something_to_fix(cfg):
    """Tailscale Serve terminates TLS in front of loopback, and a phone browser only
    opens the microphone over HTTPS. Binding the tailnet address directly is the old
    advice and would lose the microphone."""
    cfg.set("remote.enabled", True)

    line = named(check_remote(cfg), "Remote address")

    assert line.ok is True
    assert line.detail == "bound to 127.0.0.1"
    assert "127.0.0.1" in line.fix and "Tailscale address" not in line.fix


def test_a_wildcard_bind_is_called_out(cfg):
    """Something that can run PowerShell does not get to listen on every interface."""
    cfg.set("remote.enabled", True)
    cfg.set("remote.host", "0.0.0.0")

    line = named(check_remote(cfg), "Remote address")

    assert line.ok is False
    assert "every interface" in line.detail


# --- the window --------------------------------------------------------------------
def test_the_window_is_not_mentioned_when_it_is_turned_off(cfg):
    cfg.set("ui.window", False)
    assert check_window(cfg) == []


@pytest.mark.parametrize("host, ok, says", [
    ("webview", True, "frameless window of its own"),
    ("edge", True, "app mode"),
    ("browser", True, "default browser"),
    ("none", False, "the ring is all you get"),
])
def test_it_names_the_host_that_would_draw_the_window(cfg, monkeypatch, host, ok, says):
    """"No window appeared" has four different causes and four different answers."""
    from jarvis.desk import window as window_module

    monkeypatch.setattr(window_module.DeskWindow, "would_host", lambda self: host)

    line = named(check_window(cfg), "Desk window")

    assert line.ok is ok
    assert says in line.detail


def test_a_host_that_cannot_be_worked_out_is_said_so_rather_than_guessed(cfg, monkeypatch):
    """A doctor that raises is a doctor nobody can run."""
    from jarvis.desk import window as window_module

    def explode(self):
        raise RuntimeError("the registry is not answering")

    monkeypatch.setattr(window_module.DeskWindow, "would_host", explode)

    line = named(check_window(cfg), "Desk window")

    assert line.ok is False
    assert "could not be worked out" in line.detail
    assert line.essential is False
