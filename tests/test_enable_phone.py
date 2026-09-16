"""Enable-Phone.bat's script, run against a pretend Windows machine.

The real thing talks to winget and tailscale.exe; here a fake shell answers in
their place, so the decisions - install or not, sign in or not, which command
opens the door, what lands in config.yaml - are all checked without either.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import enable_phone as ep  # noqa: E402
from jarvis.config import Config  # noqa: E402

REAL_QR = ep.qr_lines  # kept before the autouse stub below replaces it

RUNNING = {"BackendState": "Running",
           "Self": {"DNSName": "desk.tail1234.ts.net.", "TailscaleIPs": ["100.64.0.7"]}}
NEEDS_LOGIN = {"BackendState": "NeedsLogin", "Self": {"DNSName": "", "TailscaleIPs": []}}


def _base(argv0: str) -> str:
    """The executable's file name, with Windows separators even on Linux."""
    return str(argv0).replace("\\", "/").rsplit("/", 1)[-1].lower()


class FakeShell(ep.Shell):
    """Answers commands from a script and remembers everything that was run."""

    def __init__(self, *, installed=True, states=None, serve_opens=True, door_open=False,
                 answers=("",)):
        super().__init__(log=self.lines.append if hasattr(self, "lines") else print)
        self.lines: list[str] = []
        self.log = self.lines.append
        self.calls: list[list[str]] = []
        self.captured: dict[str, bool] = {}
        self.timeouts: dict[str, float] = {}
        self.installed = installed
        self.states = list(states or [RUNNING])
        #: Whether `serve --bg` manages to open the door. False models the real thing
        #: when the tailnet has HTTPS certificates switched off: the command does not
        #: fail, it prints an admin-console link and blocks until somebody clicks it,
        #: so what our side sees is the timeout.
        self.serve_opens = serve_opens
        #: Whether a door is already open, which is what `serve status` reports.
        self.door_open = door_open
        self.answers = list(answers)
        self.serve_attempts = 0

    def which(self, name):
        return r"C:\Program Files\Tailscale\tailscale.exe" if self.installed and name == "tailscale" else None

    def exists(self, path):
        return False

    def ask(self, prompt):
        return self.answers.pop(0) if self.answers else ""

    def run(self, argv, *, timeout=120, capture=True):
        argv = list(argv)
        self.calls.append(argv)
        head = _base(argv[0])
        verb = argv[1] if len(argv) > 1 else ""
        # `serve --bg ...` and `serve status` are different calls with different rules
        # about capturing, so they cannot share a key.
        key = f"{head} {verb}".strip()
        if verb == "serve" and "status" not in argv:
            key += " --bg"
        self.captured[key] = capture
        self.timeouts[key] = timeout
        if head == "winget":
            self.installed = True
            return 0, ""
        if head == "tailscale.exe":
            if verb == "status":
                state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
                return 0, json.dumps(state)
            if verb == "up":
                return 0, ""
            if verb == "serve":
                if "status" in argv:
                    if not self.door_open:
                        return 0, "No serve config"
                    return 0, ("https://desk.tail1234.ts.net (tailnet only)\n"
                               "|-- / proxy http://127.0.0.1:8765")
                if "off" in argv or "reset" in argv:
                    self.door_open = False
                    return 0, ""
                self.serve_attempts += 1
                if self.serve_opens:
                    self.door_open = True
                    return 0, ""
                # What a blocked interactive flow looks like from out here: our own
                # timeout fires, exactly as Shell.run reports it.
                return 124, "tailscale.EXE took longer than 900 seconds"
        return 127, f"{head} not found"


@pytest.fixture(autouse=True)
def no_pip(monkeypatch):
    """The QR code must never reach for the network from a test."""
    monkeypatch.setattr(ep, "qr_lines", lambda url, install=True: ["    [qr]"])


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("remote:\n  enabled: false\n  host: 127.0.0.1\n  port: 8765\n  url: ''\n", encoding="utf-8")
    return Config.load(path)


def _joined(calls):
    return [" ".join(_base(c) if i == 0 else c for i, c in enumerate(call)) for call in calls]


def test_the_happy_path_opens_the_door_and_writes_config(cfg):
    shell = FakeShell()

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 0

    commands = _joined(shell.calls)
    assert "tailscale.exe serve --bg --https=443 http://127.0.0.1:8765" in commands
    assert not any(c.startswith("winget") for c in commands)
    assert not any(c == "tailscale.exe up" for c in commands), "already signed in"
    saved = Config.load(cfg.path)
    assert saved.get("remote.enabled") is True
    assert saved.get("remote.host") == "127.0.0.1"
    assert saved.get("remote.url") == "https://desk.tail1234.ts.net"
    assert any("https://desk.tail1234.ts.net" in line for line in shell.lines)
    assert any("Add to Home Screen" in line for line in shell.lines)


def test_it_installs_tailscale_when_missing(cfg):
    shell = FakeShell(installed=False)

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 0

    winget = [c for c in shell.calls if _base(c[0]) == "winget"]
    assert winget and ep.WINGET_ID in winget[0]
    assert Config.load(cfg.path).get("remote.enabled") is True


def test_it_signs_in_when_logged_out(cfg):
    shell = FakeShell(states=[NEEDS_LOGIN, RUNNING])

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 0

    assert ["C:\\Program Files\\Tailscale\\tailscale.exe", "up"] in shell.calls


def test_a_login_that_never_completes_is_reported_not_configured(cfg, monkeypatch):
    shell = FakeShell(states=[NEEDS_LOGIN])
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(ep.time, "monotonic", lambda: next(clock))

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 1

    assert Config.load(cfg.path).get("remote.enabled") is False
    assert any("not signed in" in line for line in shell.lines)


def test_a_blocked_https_flow_is_explained_instead_of_being_swallowed(cfg):
    """The failure the operator actually hit, and why it looked like nothing at all.

    `tailscale serve --https` does not fail when the tailnet has HTTPS certificates
    switched off. It prints a link to the admin console and then blocks, with no
    timeout of its own, watching for somebody to click Enable. All our side ever sees
    is its own timeout - so the message has to point at the link rather than at an
    error nobody was given.
    """
    shell = FakeShell(serve_opens=False)

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 1

    said = "\n".join(shell.lines)
    assert "press Enable" in said
    assert ep.ADMIN_DNS in said
    assert "Enable-Phone.bat again" in said
    assert Config.load(cfg.path).get("remote.enabled") is False, "no door, no promise"


def test_the_serve_output_is_never_captured(cfg):
    """This is the bug itself. Tailscale prints the link the operator must click; with
    the output piped he sees an empty window while Tailscale waits for him forever."""
    shell = FakeShell()

    ep.do_on(shell, cfg, sleep=lambda s: None)

    assert shell.captured["tailscale.exe serve --bg"] is False
    assert shell.timeouts["tailscale.exe serve --bg"] >= 600, (
        "a human has to open a browser and click something; sixty seconds is not a wait"
    )


def test_success_is_decided_by_serve_status_not_by_an_exit_code(cfg):
    """A door that is not actually open must never be written into config.yaml."""
    class Liar(FakeShell):
        def run(self, argv, *, timeout=120, capture=True):
            code, out = super().run(argv, timeout=timeout, capture=capture)
            if _base(argv[0]) == "tailscale.exe" and len(argv) > 1 and argv[1] == "serve" \
                    and "status" not in argv:
                self.door_open = False          # it said 0 and did nothing
                return 0, ""
            return code, out

    assert ep.do_on(Liar(), cfg, sleep=lambda s: None) == 1
    assert Config.load(cfg.path).get("remote.enabled") is False


def test_a_door_that_is_already_open_is_left_alone(cfg):
    """Running it twice is the documented repair path; it must not churn the config."""
    shell = FakeShell(door_open=True)

    assert ep.do_on(shell, cfg, sleep=lambda s: None) == 0

    assert shell.serve_attempts == 0
    assert any("already open" in line for line in shell.lines)
    assert Config.load(cfg.path).get("remote.url") == "https://desk.tail1234.ts.net"


def test_off_closes_the_door_and_switches_remote_off(cfg):
    ep.configure(cfg, "https://desk.tail1234.ts.net")
    shell = FakeShell()

    assert ep.do_off(shell, cfg) == 0

    assert ["C:\\Program Files\\Tailscale\\tailscale.exe", "serve", "--https=443", "off"] in shell.calls
    saved = Config.load(cfg.path)
    assert saved.get("remote.enabled") is False
    assert saved.get("remote.url") == ""


def test_status_reads_without_changing_anything(cfg):
    shell = FakeShell()

    assert ep.do_status(shell, cfg) == 0

    assert not any("serve" in c and "--bg" in c for c in shell.calls)
    assert Config.load(cfg.path).get("remote.enabled") is False
    assert any("desk.tail1234.ts.net" in line for line in shell.lines)


def test_status_json_is_read_defensively():
    class Garbage(FakeShell):
        def run(self, argv, *, timeout=120, capture=True):
            return 1, "not json at all"

    state = ep.status(Garbage(), "tailscale.exe")
    assert state.running is False
    assert state.dns_name == ""
    assert state.url == ""


def test_the_qr_code_is_optional():
    lines = REAL_QR("https://desk.tail1234.ts.net", install=False)
    # Either qrcode is installed and we get a square, or it is not and we get nothing.
    assert lines == [] or (len(lines) > 10 and all(len(line) == len(lines[0]) for line in lines))


def test_the_server_tells_the_phone_the_https_address(tmp_path):
    from jarvis.remote.server import RemoteServer
    import logging

    path = tmp_path / "config.yaml"
    path.write_text("remote:\n  enabled: true\n  host: 127.0.0.1\n  port: 8765\n"
                    "  url: https://desk.tail1234.ts.net\n", encoding="utf-8")
    server = RemoteServer(Config.load(path), None, logging.getLogger("t"))
    assert server.url == "http://127.0.0.1:8765"
    assert server.phone_url == "https://desk.tail1234.ts.net"

    path.write_text("remote:\n  enabled: true\n  host: 127.0.0.1\n  port: 8765\n", encoding="utf-8")
    server = RemoteServer(Config.load(path), None, logging.getLogger("t"))
    assert server.phone_url == server.url
