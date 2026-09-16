"""Bring this JARVIS up to date with the published one.

Double-click ``Update-JARVIS.bat``; this is what it runs. It fetches the current
source straight from GitHub, copies in what changed, and installs anything new the
requirements ask for. Everything that is yours - ``config.yaml``, your memory, your
logs, the downloaded models and the virtual environment - is left exactly alone.

    python scripts/update_jarvis.py            # look, then update
    python scripts/update_jarvis.py --check    # only say what would change
    python scripts/update_jarvis.py --branch main

Two rules it never breaks, because this runs on the operator's only installation:
it never deletes anything, and it never overwrites a file without first copying the
old one into ``logs/update-backup-<timestamp>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
REPO = "D3v0ps/JARVIS"
DEFAULT_BRANCH = "main"
ZIP_URL = "https://github.com/{repo}/archive/refs/heads/{branch}.zip"
TIMEOUT = 120
MAX_ZIP_BYTES = 200 * 1024 * 1024        # the source is ~2 MB; this is a sanity bound

#: Proof that what we downloaded is JARVIS and not an error page in a zip's clothing.
MARKERS = ("jarvis/__init__.py", "requirements.txt", "start-jarvis.bat")

#: Never written, whatever the download contains. ``config.yaml`` is the important one:
#: the installer wrote your graphics card and your model into it, and you may well have
#: edited it since. Every key the new code adds has a default baked into jarvis/config.py,
#: so a config.yaml from an older version reads perfectly well against newer code.
PRESERVE_FILES = frozenset({
    "config.yaml", "config.local.yaml", "memory.json", "places.json",
})
#: Never descended into, for the same reason.
PRESERVE_DIRS = frozenset({
    ".git", ".venv", "venv", "logs", "models", "__pycache__", ".pytest_cache",
})


@dataclass
class Plan:
    """What an update would do, worked out before anything is written."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    new_config_keys: list[str] = field(default_factory=list)

    @property
    def touches(self) -> int:
        return len(self.added) + len(self.changed)


# --- the download ------------------------------------------------------------------------
def download(url: str, *, opener: Callable[..., object] = urllib.request.urlopen) -> bytes:
    """The source archive, as bytes. Raises RuntimeError with something readable."""
    try:
        with opener(url, timeout=TIMEOUT) as response:  # type: ignore[operator]
            payload = response.read(MAX_ZIP_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub answered {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"could not reach GitHub ({exc})") from exc
    if len(payload) > MAX_ZIP_BYTES:
        raise RuntimeError("the download is far larger than the source has any right to be")
    return payload


def open_archive(payload: bytes) -> tuple[zipfile.ZipFile, str]:
    """The archive and its single top-level folder, having checked it really is JARVIS."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("what came back was not a zip file") from exc

    roots = {name.split("/", 1)[0] for name in archive.namelist() if name.strip("/")}
    if len(roots) != 1:
        raise RuntimeError(f"expected one folder in the archive, found {len(roots)}")
    prefix = roots.pop()

    names = set(archive.namelist())
    missing = [m for m in MARKERS if f"{prefix}/{m}" not in names]
    if missing:
        raise RuntimeError("the archive does not look like JARVIS (no " + ", ".join(missing) + ")")
    return archive, prefix


def members(archive: zipfile.ZipFile, prefix: str) -> list[tuple[str, zipfile.ZipInfo]]:
    """Every real file in the archive as ``(relative path, member)``, safely.

    A zip may name a member ``../../windows/system32/...``; anything that does not stay
    inside the install is dropped and never written. That is the one attack an archive
    gets to try, and it is refused here rather than at the filesystem.
    """
    keep: list[tuple[str, zipfile.ZipInfo]] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if not name.startswith(prefix + "/"):
            continue
        relative = name[len(prefix) + 1:]
        if not relative or relative.startswith("/"):
            continue
        parts = Path(relative).parts
        if any(part in ("..", "") for part in parts) or Path(relative).is_absolute():
            continue
        if ":" in relative or "\\" in relative:     # a Windows drive or separator in a zip
            continue
        keep.append((relative, info))
    return keep


def is_preserved(relative: str) -> bool:
    """Whether this path is the operator's rather than the project's."""
    parts = Path(relative).parts
    if parts[0] in PRESERVE_DIRS or any(part in PRESERVE_DIRS for part in parts[:-1]):
        return True
    return relative in PRESERVE_FILES


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- working out what would change ---------------------------------------------------------
def build_plan(archive: zipfile.ZipFile, prefix: str, root: Path) -> Plan:
    """Compare the archive with the installation, writing nothing."""
    plan = Plan()
    for relative, info in members(archive, prefix):
        target = root / relative
        if is_preserved(relative):
            # Only the operator's own files are worth naming in the report. A .gitkeep
            # inside logs/ is preserved too, and saying so would be noise.
            if target.exists() and relative in PRESERVE_FILES:
                plan.preserved.append(relative)
            continue
        fresh = archive.read(info)
        if not target.exists():
            plan.added.append(relative)
        elif _digest(target.read_bytes()) != _digest(fresh):
            plan.changed.append(relative)
    plan.new_config_keys = config_keys_you_have_not_got(archive, prefix, root)
    for group in (plan.added, plan.changed, plan.preserved):
        group.sort()
    return plan


def config_keys_you_have_not_got(archive: zipfile.ZipFile, prefix: str, root: Path) -> list[str]:
    """Top-level-ish keys the shipped config.yaml has and yours does not.

    Reported, never written. Every one of them has a default in ``jarvis/config.py``, so
    nothing breaks without them - but an operator who wants to change one should be told
    it exists rather than having to diff two files.
    """
    yours = root / "config.yaml"
    if not yours.exists():
        return []
    try:
        import yaml  # noqa: PLC0415

        shipped = yaml.safe_load(archive.read(f"{prefix}/config.yaml")) or {}
        current = yaml.safe_load(yours.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a nicety, never a reason to stop an update
        return []

    found: list[str] = []

    def walk(new: object, old: object, path: str) -> None:
        if not isinstance(new, dict):
            return
        for key, value in new.items():
            here = f"{path}.{key}" if path else str(key)
            if not isinstance(old, dict) or key not in old:
                found.append(here)
            else:
                walk(value, old.get(key), here)

    walk(shipped, current, "")
    return found[:20]


# --- doing it -------------------------------------------------------------------------------
def apply_plan(archive: zipfile.ZipFile, prefix: str, root: Path, plan: Plan,
               backup: Path, log: Callable[[str], None]) -> list[str]:
    """Write the new files, backing up every one that is replaced. Returns what failed."""
    wanted = set(plan.added) | set(plan.changed)
    failed: list[str] = []
    for relative, info in members(archive, prefix):
        if relative not in wanted:
            continue
        target = root / relative
        try:
            if target.exists():
                kept = backup / relative
                kept.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, kept)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(info)
            # Write beside the target and move into place, so a failure halfway through
            # leaves the old file rather than half a new one.
            temporary = target.with_name(target.name + ".new")
            temporary.write_bytes(data)
            os.replace(temporary, target)
        except PermissionError:
            failed.append(relative)
            log(f"  [!] {relative} is in use; close JARVIS and run this again")
        except OSError as exc:
            failed.append(relative)
            log(f"  [!] {relative} could not be written ({exc})")
    return failed


def venv_python(root: Path) -> Path | None:
    """The interpreter inside the installation's own environment, if there is one."""
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def install_requirements(python: Path, root: Path, log: Callable[[str], None]) -> bool:
    """Install both requirements files. The optional one only ever warns."""
    ok = True
    for name, fatal in (("requirements.txt", True), ("requirements-optional.txt", False)):
        path = root / name
        if not path.is_file():
            continue
        log(f"  [*] {name}...")
        try:
            completed = subprocess.run(
                [str(python), "-m", "pip", "install", "-q", "-r", str(path)],
                capture_output=True, text=True, timeout=1800,
                encoding="utf-8", errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"  [!] pip would not run ({exc})")
            ok = ok and not fatal
            continue
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                log(f"      {line}")
            if fatal:
                log(f"  [x] {name} did not install cleanly.")
                ok = False
            else:
                log("  [!] An optional package did not install; JARVIS works without it.")
    return ok


# --- the report -------------------------------------------------------------------------------
def describe(plan: Plan, log: Callable[[str], None], *, limit: int = 12) -> None:
    def listing(label: str, names: Sequence[str]) -> None:
        if not names:
            return
        log(f"  {label} ({len(names)}):")
        for name in names[:limit]:
            log(f"      {name}")
        if len(names) > limit:
            log(f"      ... and {len(names) - limit} more")

    listing("New", plan.added)
    listing("Updated", plan.changed)
    if plan.preserved:
        log(f"  Left alone, because they are yours ({len(plan.preserved)}): "
            + ", ".join(plan.preserved))
    if plan.new_config_keys:
        log("  New settings exist that your config.yaml does not have. They all have")
        log("  sensible defaults, so nothing is broken; add them only if you want to change one:")
        for key in plan.new_config_keys:
            log(f"      {key}")


def main(argv: Sequence[str] | None = None, *, log: Callable[[str], None] = print,
         root: Path | None = None, fetch: Callable[[str], bytes] = download) -> int:
    parser = argparse.ArgumentParser(description="Bring this JARVIS up to date.")
    parser.add_argument("--check", action="store_true", help="say what would change, change nothing")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help=f"which branch (default: {DEFAULT_BRANCH})")
    parser.add_argument("--no-deps", action="store_true", help="skip the pip step")
    args = parser.parse_args(argv)

    root = root or ROOT
    log("")
    log("  J.A.R.V.I.S. - update")
    log("  ---------------------------------------------------------------")

    if not (root / "jarvis" / "__init__.py").is_file() or not (root / "requirements.txt").is_file():
        log(f"  [x] {root} does not look like a JARVIS installation.")
        return 1

    url = ZIP_URL.format(repo=REPO, branch=args.branch)
    log(f"  [*] Fetching {REPO} ({args.branch})...")
    try:
        archive, prefix = open_archive(fetch(url))
    except RuntimeError as exc:
        log(f"  [x] {exc}")
        log("      Check this machine's internet connection and try again.")
        return 1

    plan = build_plan(archive, prefix, root)
    if not plan.touches:
        log("  You are already up to date, sir.")
        return 0

    log(f"  [*] {plan.touches} file(s) differ from the published version.")
    describe(plan, log)

    if args.check:
        log("")
        log("  Nothing was changed. Run this without --check to apply it.")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = root / "logs" / f"update-backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    log(f"  [*] Replaced files are copied to logs\\{backup.name} first.")

    failed = apply_plan(archive, prefix, root, plan, backup, log)
    if failed:
        log(f"  [x] {len(failed)} file(s) could not be replaced, most likely because JARVIS")
        log("      is running. Close him (right-click the tray icon, Quit) and run this again.")
        return 1
    log(f"  [*] {plan.touches} file(s) updated.")

    if not args.no_deps:
        python = venv_python(root)
        if python is None:
            log("  [!] No .venv here; run Install-JARVIS.exe once to build one.")
        else:
            log("  [*] Installing what the new version needs...")
            if not install_requirements(python, root, log):
                log("  [x] A required package did not install. Run Install-JARVIS.exe to repair.")
                return 1

    log("")
    log("  Up to date. Double-click JARVIS to bring him back online.")
    log("  Check-JARVIS.bat will tell you what this machine can do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
