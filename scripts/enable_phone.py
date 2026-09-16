"""Let your phone reach JARVIS from anywhere, over Tailscale.

Double-click ``Enable-Phone.bat``; this is what it runs. It installs Tailscale if it
is missing, signs you in, tells JARVIS to listen on this machine only, and asks
Tailscale to put an HTTPS door in front of him. HTTPS is not decoration: a phone
browser only opens the microphone for a secure page.

    python scripts/enable_phone.py            # set it up, or repair it
    python scripts/enable_phone.py --status   # what is set up right now
    python scripts/enable_phone.py --off      # take the door down again

Nothing here is a cloud account for JARVIS. Tailscale carries the packets between
two devices you own; the words never leave your machines.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis.config import Config  # noqa: E402 - after the path fix

WINGET_ID = "tailscale.tailscale"
ADMIN_DNS = "https://login.tailscale.com/admin/dns"
LOGIN_WAIT_S = 300
PHONE_APPS = "the Tailscale app (App Store on iPhone, Play Store on Android)"


@dataclass
class Shell:
    """The few things this script asks the machine for. Tests swap it out."""

    log: Callable[[str], None] = print

    def run(self, argv: Sequence[str], *, timeout: float = 120, capture: bool = True) -> tuple[int, str]:
        """Run a command. Returns (exit code, combined output); never raises."""
        try:
            completed = subprocess.run(
                list(argv), capture_output=capture, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            return 127, f"{argv[0]} was not found"
        except subprocess.TimeoutExpired:
            return 124, f"{argv[0]} took longer than {timeout:.0f} seconds"
        except OSError as exc:
            return 1, str(exc)
        text = ((completed.stdout or "") + (completed.stderr or "")) if capture else ""
        return int(completed.returncode), text.strip()

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def exists(self, path: str) -> bool:
        return Path(path).is_file()

    def ask(self, prompt: str) -> str:
        """Wait for Enter; returns "" when there is nobody at the keyboard."""
        if not sys.stdin or not sys.stdin.isatty():
            return ""
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""


@dataclass
class TailscaleState:
    """What ``tailscale status --json`` says, reduced to the four things we need."""

    backend: str = "Unknown"
    dns_name: str = ""
    ips: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.backend == "Running"

    @property
    def url(self) -> str:
        return f"https://{self.dns_name}" if self.dns_name else ""


# --- finding and installing ------------------------------------------------------------
def candidate_paths() -> list[str]:
    """Where the Windows installer puts tailscale.exe when it is not on PATH."""
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             os.environ.get("LOCALAPPDATA", "")]
    return [os.path.join(root, "Tailscale", "tailscale.exe") for root in roots if root]


def find_tailscale(shell: Shell) -> str | None:
    found = shell.which("tailscale")
    if found:
        return found
    for path in candidate_paths():
        if shell.exists(path):
            return path
    return None


def install_tailscale(shell: Shell) -> bool:
    """Install it with winget. Windows asks for permission once; say yes."""
    shell.log("  [*] Installing Tailscale with winget - say yes to the Windows prompt...")
    code, out = shell.run([
        "winget", "install", "--id", WINGET_ID, "-e", "--silent",
        "--accept-package-agreements", "--accept-source-agreements",
    ], timeout=600, capture=False)
    if code not in (0, -1978335189):  # the second is winget's "already installed"
        shell.log(f"  [x] winget could not install Tailscale (exit {code}). {out}".rstrip())
        shell.log("      Install it yourself from https://tailscale.com/download and run this again.")
        return False
    return True


# --- talking to tailscale ----------------------------------------------------------------
def status(shell: Shell, exe: str) -> TailscaleState:
    code, out = shell.run([exe, "status", "--json"], timeout=20)
    try:
        data = json.loads(out) if out else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    self_node = data.get("Self") if isinstance(data.get("Self"), dict) else {}
    dns = str(self_node.get("DNSName") or "").rstrip(".")
    ips = [str(ip) for ip in (self_node.get("TailscaleIPs") or []) if ip]
    backend = str(data.get("BackendState") or ("Unknown" if code else "Running"))
    return TailscaleState(backend=backend, dns_name=dns, ips=ips, raw=data)


def ensure_logged_in(shell: Shell, exe: str, *, wait_s: float = LOGIN_WAIT_S,
                     sleep: Callable[[float], None] = time.sleep) -> TailscaleState:
    """Sign in if needed. ``tailscale up`` opens the browser and waits for you."""
    state = status(shell, exe)
    if state.running and state.dns_name:
        return state
    shell.log("  [*] Signing in to Tailscale - a browser window will open. Use the same")
    shell.log("      account you will use on the phone.")
    shell.run([exe, "up"], timeout=wait_s, capture=False)
    deadline = time.monotonic() + 30
    while True:
        state = status(shell, exe)
        if state.running and state.dns_name:
            return state
        if time.monotonic() >= deadline:
            return state
        sleep(2)


def serve_on(shell: Shell, exe: str, port: int) -> tuple[bool, str]:
    """Put Tailscale's HTTPS door in front of JARVIS. Survives reboots (``--bg``)."""
    code, out = shell.run([exe, "serve", "--bg", "--https=443", f"http://127.0.0.1:{port}"], timeout=60)
    if code == 0:
        return True, out
    return False, out


def serve_off(shell: Shell, exe: str) -> bool:
    code, _ = shell.run([exe, "serve", "--https=443", "off"], timeout=60)
    if code == 0:
        return True
    code, _ = shell.run([exe, "serve", "reset"], timeout=60)
    return code == 0


def needs_https_enabled(output: str) -> bool:
    """Tailscale's way of saying the tailnet has not switched HTTPS certificates on."""
    text = (output or "").lower()
    return "https" in text and any(word in text for word in ("enable", "certificate", "cert", "magicdns"))


# --- jarvis's side ----------------------------------------------------------------------
def configure(cfg: Config, url: str) -> None:
    """Listen on this machine only; Tailscale is the only door in."""
    cfg.set("remote.enabled", True)
    cfg.set("remote.host", "127.0.0.1")
    cfg.set("remote.url", url)
    cfg.save()


def unconfigure(cfg: Config) -> None:
    cfg.set("remote.enabled", False)
    cfg.set("remote.url", "")
    cfg.save()


def qr_lines(url: str, *, install: bool = True) -> list[str]:
    """The URL as a QR code drawn in half-block characters, or [] without ``qrcode``."""
    try:
        import qrcode  # noqa: PLC0415
    except ImportError:
        if not install:
            return []
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "qrcode"],
                           capture_output=True, check=False, timeout=90)
        except (OSError, subprocess.SubprocessError):
            return []
        try:
            import qrcode  # noqa: PLC0415
        except ImportError:
            return []
    code = qrcode.QRCode(border=2)
    code.add_data(url)
    code.make(fit=True)
    matrix = code.get_matrix()
    lines: list[str] = []
    for top in range(0, len(matrix), 2):
        upper = matrix[top]
        lower = matrix[top + 1] if top + 1 < len(matrix) else [False] * len(upper)
        chars = []
        for a, b in zip(upper, lower):
            chars.append("\u2588" if a and b else "\u2580" if a else "\u2584" if b else " ")
        lines.append("    " + "".join(chars))
    return lines


# --- the three verbs -------------------------------------------------------------------
def do_status(shell: Shell, cfg: Config) -> int:
    exe = find_tailscale(shell)
    shell.log(f"  Tailscale:      {exe or 'not installed'}")
    if exe:
        state = status(shell, exe)
        shell.log(f"  Signed in:      {'yes' if state.running else 'no (' + state.backend + ')'}")
        shell.log(f"  This machine:   {state.dns_name or '-'}  {' '.join(state.ips)}")
        _, out = shell.run([exe, "serve", "status"], timeout=20)
        shell.log(f"  HTTPS door:     {'up' if '127.0.0.1' in out else 'down'}")
    shell.log(f"  JARVIS remote:  {'on' if cfg.get('remote.enabled') else 'off'}"
              f"  ({cfg.get('remote.host')}:{cfg.get('remote.port')})")
    shell.log(f"  Phone opens:    {cfg.get('remote.url') or '-'}")
    return 0


def do_off(shell: Shell, cfg: Config) -> int:
    exe = find_tailscale(shell)
    if exe and serve_off(shell, exe):
        shell.log("  [*] The HTTPS door is down.")
    unconfigure(cfg)
    shell.log("  [*] JARVIS will no longer listen for the phone. Restart him for it to take.")
    return 0


def do_on(shell: Shell, cfg: Config, *, sleep: Callable[[float], None] = time.sleep) -> int:
    exe = find_tailscale(shell)
    if not exe:
        if not install_tailscale(shell):
            return 1
        exe = find_tailscale(shell)
        if not exe:
            shell.log("  [x] Tailscale installed but I cannot find tailscale.exe. Log out and in, then retry.")
            return 1
    shell.log(f"  [*] Tailscale: {exe}")

    state = ensure_logged_in(shell, exe, sleep=sleep)
    if not state.running or not state.dns_name:
        shell.log(f"  [x] Tailscale is not signed in ({state.backend}). Sign in from the tray icon")
        shell.log("      and double-click Enable-Phone.bat again.")
        return 1
    shell.log(f"  [*] This machine on your tailnet: {state.dns_name}")

    port = int(cfg.get("remote.port", 8765) or 8765)
    ok, out = serve_on(shell, exe, port)
    if not ok and needs_https_enabled(out):
        shell.log("  [!] Your tailnet has HTTPS certificates switched off. One click fixes it:")
        shell.log(f"      open {ADMIN_DNS} and press 'Enable HTTPS'.")
        shell.ask("      Press Enter here when that is done... ")
        ok, out = serve_on(shell, exe, port)
    if not ok:
        shell.log(f"  [x] Tailscale would not open the door: {out}")
        return 1
    shell.log(f"  [*] HTTPS door: {state.url}  ->  JARVIS on 127.0.0.1:{port}")

    configure(cfg, state.url)
    shell.log("  [*] config.yaml updated: remote on, listening on this machine only.")

    shell.log("")
    shell.log("  Now on the phone:")
    shell.log(f"    1. Install {PHONE_APPS} and sign in with the same account.")
    shell.log(f"    2. Open  {state.url}  in Safari or Chrome - or scan this:")
    for line in qr_lines(state.url):
        shell.log(line)
    shell.log("    3. Share -> 'Add to Home Screen'. It becomes an app with an icon.")
    shell.log("    4. Start JARVIS. The pairing code is printed in his window; type it once.")
    shell.log("")
    shell.log("  From then on the phone reaches him from anywhere, as long as this PC is awake.")
    return 0


def main(argv: Sequence[str] | None = None, shell: Shell | None = None,
         cfg: Config | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reach JARVIS from your phone, anywhere.")
    parser.add_argument("--off", action="store_true", help="take the HTTPS door down")
    parser.add_argument("--status", action="store_true", help="show what is set up")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    shell = shell or Shell()
    cfg = cfg or Config.load(args.config)
    shell.log("")
    shell.log("  J.A.R.V.I.S. - the phone")
    shell.log("  ---------------------------------------------------------------")
    if args.status:
        return do_status(shell, cfg)
    if args.off:
        return do_off(shell, cfg)
    return do_on(shell, cfg)


if __name__ == "__main__":
    sys.exit(main())
