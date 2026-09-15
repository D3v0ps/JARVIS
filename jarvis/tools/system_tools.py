"""System tools: the clock, the machine's vital signs, the screen and the volume knob.

Everything Windows-only (``psutil``, ``PIL``, ``pycaw``/``comtypes``, ``ctypes.windll``)
is imported inside the function that needs it, so this module imports cleanly on a bare
Linux box. When a dependency or the platform is missing, each tool returns a calm
one-sentence failure instead of raising.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = [
    "get_time_date", "system_status", "lock_pc", "screenshot", "volume", "media",
    "MEDIA_KEYS", "VOLUME_ACTIONS", "VOLUME_STEP",
]

_log = get_logger("tools.system")

#: Keeps a console window from flashing up when a helper process starts on Windows.
CREATE_NO_WINDOW = 0x08000000
#: Seconds allowed for ``nvidia-smi`` before the GPU is left out of the answer.
NVIDIA_SMI_TIMEOUT = 3.0
#: Sampling window for the CPU percentage — without one psutil reports 0.0 on first call.
CPU_SAMPLE_SECONDS = 0.3
#: How many points ``up`` and ``down`` move the master volume, and the actions accepted.
VOLUME_STEP = 10
VOLUME_ACTIONS = ("set", "up", "down", "mute", "unmute")

#: Media key virtual-key codes, with the line JARVIS says after pressing one.
MEDIA_KEYS: dict[str, tuple[int, str]] = {
    "play_pause": (0xB3, "Play or pause it is, sir."),
    "next": (0xB0, "Skipping to the next track, sir."),
    "previous": (0xB1, "Back to the previous track, sir."),
    "stop": (0xB2, "Stopping the music, sir."),
}
_MEDIA_ALIASES = {"play": "play_pause", "pause": "play_pause", "prev": "previous", "back": "previous"}
_KEYEVENTF_KEYUP = 0x0002

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}
_UNSAFE_NAME_RE = re.compile(r"[^\w\-. ]+", re.UNICODE)


def _spoken_int(value: int) -> str:
    """Render a whole number up to 999 as words, so it reads well aloud."""
    number = int(value)
    if number < 0 or number > 999:
        return str(number)
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"
    hundreds, rest = divmod(number, 100)
    head = f"{_UNITS[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_spoken_int(rest)}"


def _spoken_amount(value: float) -> str:
    """Render a measurement as words, keeping one decimal only when it matters."""
    rounded = round(float(value), 1)
    if abs(rounded) >= 10 or rounded == int(rounded):
        return _spoken_int(round(rounded))
    whole = int(rounded)
    decimal = abs(int(round((rounded - whole) * 10)))
    return f"{_spoken_int(whole)} point {_UNITS[decimal]}" if decimal else _spoken_int(whole)


def _spoken_ordinal(day: int) -> str:
    """The day of the month as a spoken ordinal: 1 -> first, 22 -> twenty-second."""
    if day in _ORDINALS:
        return _ORDINALS[day]
    if day < 20:
        return _UNITS[day] + "th"
    tens, unit = divmod(day, 10)
    if unit == 0:
        return _TENS[tens][:-1] + "ieth"
    return f"{_TENS[tens]}-{_ORDINALS.get(unit, _UNITS[unit] + 'th')}"


def _popen_kwargs() -> dict[str, Any]:
    """Extra ``subprocess`` keywords: hide the console window on Windows only."""
    return {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def _not_windows(what: str) -> ToolResult:
    """The standard, in-character answer when a Windows-only tool runs elsewhere."""
    detail = f"Platform is {sys.platform!r}; this tool requires win32."
    return ToolResult.fail(f"I can only {what} on Windows, sir.", detail=detail)


@tool(
    "get_time_date",
    description="Tell the user the current local time, the weekday and today's date.",
    parameters={"type": "object", "properties": {}},
    tier=Tier.SAFE,
)
def get_time_date(ctx: ToolContext, args: dict) -> ToolResult:
    """Report the local clock in one sentence a voice can deliver naturally."""
    now = datetime.now()
    clock, weekday, month = now.strftime("%H:%M"), now.strftime("%A"), now.strftime("%B")
    summary = f"It's {clock} on {weekday} the {_spoken_ordinal(now.day)} of {month}, sir."
    data = {
        "time": clock, "weekday": weekday, "date": now.strftime("%Y-%m-%d"),
        "iso": now.isoformat(timespec="seconds"),
    }
    detail = now.strftime("%A %d %B %Y, %H:%M:%S")
    return ToolResult(ok=True, summary=summary, detail=detail, data=data)


def _gpu_info() -> dict[str, float] | None:
    """Utilisation, temperature and VRAM from ``nvidia-smi``, or ``None`` without a GPU."""
    query = "utilization.gpu,temperature.gpu,memory.used,memory.total"
    argv = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT,
            encoding="utf-8", errors="replace", **_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("nvidia-smi unavailable (%s); reporting without the GPU", exc)
        return None
    if completed.returncode != 0 or not (completed.stdout or "").strip():
        return None
    fields = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
    if len(fields) < 4:
        return None
    try:
        values = [float(field) for field in fields[:4]]
    except ValueError:
        _log.debug("Could not parse nvidia-smi output %r", fields)
        return None
    keys = ("util_percent", "temperature_c", "vram_used_mb", "vram_total_mb")
    return dict(zip(keys, values))


@tool(
    "system_status",
    description="Report how the machine is doing right now: processor load, memory in "
    "use, free disk space, uptime and, when there is one, the graphics card.",
    parameters={"type": "object", "properties": {}},
    tier=Tier.SAFE,
)
def system_status(ctx: ToolContext, args: dict) -> ToolResult:
    """Read the machine's vital signs and boil them down to one calm sentence."""
    try:
        import psutil  # noqa: PLC0415 - optional, Windows-target dependency
    except ImportError as exc:
        _log.warning("psutil is not installed; system_status cannot report")
        return ToolResult.fail("I can't read the system counters on this machine, sir.",
                               detail=f"psutil could not be imported: {exc}")
    try:
        cpu = float(psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS))
        memory = psutil.virtual_memory()
        drive = os.environ.get("SystemDrive", "C:") + "\\" if sys.platform == "win32" else "/"
        disk = psutil.disk_usage(drive)
        uptime_s = max(0.0, time.time() - float(psutil.boot_time()))
    except Exception as exc:  # noqa: BLE001 - a counter must never kill the turn
        _log.error("Could not read the system counters: %s", exc)
        return ToolResult.fail("I couldn't read the counters just now, sir.", detail=str(exc))

    gigabyte = 1024 ** 3
    ram_used, ram_total = memory.used / gigabyte, memory.total / gigabyte
    free_gb, total_gb = disk.free / gigabyte, disk.total / gigabyte
    hours, minutes = divmod(int(uptime_s) // 60, 60)
    gpu = _gpu_info()

    summary = (
        f"Processor at {_spoken_amount(cpu)} percent, {_spoken_amount(ram_used)} of "
        f"{_spoken_amount(ram_total)} gigabytes of memory in use"
    )
    if gpu is not None:
        summary += f", and the graphics card at {_spoken_amount(gpu['util_percent'])} percent"
    summary += ", sir."

    detail = [
        f"CPU: {cpu:.1f}%",
        f"RAM: {ram_used:.1f} / {ram_total:.1f} GB ({memory.percent:.0f}%)",
        f"Disk {drive}: {free_gb:.1f} GB free of {total_gb:.1f} GB",
        f"Uptime: {hours} h {minutes} min",
    ]
    data: dict[str, Any] = {
        "cpu_percent": round(cpu, 1), "ram_used_gb": round(ram_used, 1),
        "ram_total_gb": round(ram_total, 1), "disk_free_gb": round(free_gb, 1),
        "uptime_s": round(uptime_s),
    }
    if gpu is None:
        detail.append("GPU: not detected (no nvidia-smi or no NVIDIA card)")
    else:
        detail.append(
            f"GPU: {gpu['util_percent']:.0f}% utilisation, {gpu['temperature_c']:.0f} C, "
            f"{gpu['vram_used_mb']:.0f} / {gpu['vram_total_mb']:.0f} MB VRAM"
        )
        data.update(
            gpu_percent=round(gpu["util_percent"]),
            gpu_temperature_c=round(gpu["temperature_c"]),
            vram_used_mb=round(gpu["vram_used_mb"]),
            vram_total_mb=round(gpu["vram_total_mb"]),
        )
    return ToolResult(ok=True, summary=summary, detail="\n".join(detail), data=data)


@tool(
    "lock_pc",
    description="Lock the Windows session, exactly as pressing Windows and L would.",
    parameters={"type": "object", "properties": {}},
    tier=Tier.SAFE,
)
def lock_pc(ctx: ToolContext, args: dict) -> ToolResult:
    """Lock the workstation through ``rundll32.exe user32.dll,LockWorkStation``."""
    if sys.platform != "win32":
        return _not_windows("lock the session")
    argv = ["rundll32.exe", "user32.dll,LockWorkStation"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=10, **_popen_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        _log.error("Could not lock the workstation: %s", exc)
        return ToolResult.fail("I couldn't lock the machine, sir.", detail=str(exc))
    if done.returncode:
        detail = f"$ {' '.join(argv)}\nexit code: {done.returncode}\n{done.stderr}"
        return ToolResult.fail("Windows wouldn't lock the session, sir.", detail=detail)
    return ToolResult(ok=True, summary="Locking the machine, sir.", detail=" ".join(argv))


def _screenshot_dir(ctx: ToolContext) -> Path:
    """``tools.screenshot_dir`` when set, otherwise ``%USERPROFILE%\\Pictures\\Jarvis``."""
    configured = ctx.config.get("tools.screenshot_dir") if ctx.config is not None else None
    if configured:
        return Path(str(configured)).expanduser()
    profile = os.environ.get("USERPROFILE")
    return (Path(profile) if profile else Path.home()) / "Pictures" / "Jarvis"


def _screenshot_name(raw: Any) -> str:
    """Turn a requested name into a safe PNG filename, or invent a timestamped one."""
    text = " ".join(str(raw or "").split())
    if text:
        if text.lower().endswith(".png"):
            text = text[:-4]
        # Sanitise before anything else: a name like "holiday/1" must not become "1".
        stem = _UNSAFE_NAME_RE.sub("_", text).strip(" ._")
        if stem:
            return f"{stem[:60]}.png"
    return datetime.now().strftime("jarvis-%Y-%m-%d-%H%M%S.png")


@tool(
    "screenshot",
    description="Capture the whole screen, across every monitor, and save it as a PNG "
    "image in the user's screenshot folder.",
    parameters={"type": "object", "properties": {
        "name": {"type": "string",
                 "description": "Optional file name for the image; a timestamp is used when omitted."},
    }},
    tier=Tier.SAFE,
)
def screenshot(ctx: ToolContext, args: dict) -> ToolResult:
    """Grab every screen with Pillow and report the file name, not the whole path."""
    filename = _screenshot_name(args.get("name"))
    try:
        from PIL import ImageGrab  # noqa: PLC0415 - optional, Windows-target dependency
    except ImportError as exc:
        _log.warning("Pillow is not installed; screenshot is unavailable")
        return ToolResult.fail("I can't take screenshots on this machine, sir.",
                               detail=f"PIL.ImageGrab could not be imported: {exc}")
    try:
        directory = _screenshot_dir(ctx)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        ImageGrab.grab(all_screens=True).save(target, "PNG")
    except Exception as exc:  # noqa: BLE001 - screen capture fails in many small ways
        _log.error("Screenshot failed: %s", exc)
        return ToolResult.fail("I couldn't capture the screen, sir.", detail=f"{exc!r}")
    return ToolResult(
        ok=True,
        summary=f"Screen captured and saved as {filename}, sir.",
        detail=f"Saved to {target}",
        data={"path": str(target), "filename": filename},
    )


def _volume_endpoint() -> Any:
    """The pycaw master-volume interface for the default speakers.

    COM is initialised by the caller; this only resolves the endpoint.
    """
    from ctypes import POINTER, cast  # noqa: PLC0415 - Windows-only code path
    from comtypes import CLSCTX_ALL  # noqa: PLC0415
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # noqa: PLC0415

    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _apply_volume(endpoint: Any, action: str, level: Any) -> tuple[int, bool]:
    """Perform one volume action and return the resulting ``(percent, muted)``."""
    current = int(round(endpoint.GetMasterVolumeLevelScalar() * 100))
    if action in ("set", "up", "down"):
        target = (
            max(0, min(100, int(round(float(level)))))
            if action == "set"
            else max(0, min(100, current + (VOLUME_STEP if action == "up" else -VOLUME_STEP)))
        )
        endpoint.SetMasterVolumeLevelScalar(target / 100.0, None)
        if action != "down":
            endpoint.SetMute(0, None)  # setting or raising the volume implies unmuting
    else:
        endpoint.SetMute(1 if action == "mute" else 0, None)
    return int(round(endpoint.GetMasterVolumeLevelScalar() * 100)), bool(endpoint.GetMute())


@tool(
    "volume",
    description="Change the system output volume: set it to a percentage, nudge it up "
    "or down by ten points, or mute and unmute the speakers.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "enum": list(VOLUME_ACTIONS),
                   "description": "What to do: set, up, down, mute or unmute."},
        "level": {"type": "integer",
                  "description": "Target volume from 0 to 100; only used with the set action."},
    }, "required": ["action"]},
    tier=Tier.SAFE,
)
def volume(ctx: ToolContext, args: dict) -> ToolResult:
    """Drive the Windows master volume through pycaw.

    Tools run on a worker thread, so COM has to be initialised here and released again
    afterwards; otherwise pycaw raises "CoInitialize has not been called".
    """
    action = str(args.get("action") or "").strip().lower().replace("-", "_")
    level = args.get("level")
    if action not in VOLUME_ACTIONS:
        return ToolResult.fail("I can set the volume, nudge it up or down, or mute it, sir.",
                               detail=f"Unsupported volume action {action!r}.")
    if action == "set":
        try:
            level = int(round(float(level)))
        except (TypeError, ValueError):
            return ToolResult.fail("I need a level between zero and a hundred, sir.",
                                   detail=f"Unusable level {level!r}.")
    if sys.platform != "win32":
        return _not_windows("change the volume")
    try:
        import comtypes  # noqa: PLC0415 - Windows-only dependency
    except ImportError as exc:
        _log.warning("comtypes/pycaw is not installed; volume control unavailable")
        return ToolResult.fail("I can't reach the volume control on this machine, sir.",
                               detail=f"comtypes could not be imported: {exc}")

    comtypes.CoInitialize()
    try:
        percent, muted = _apply_volume(_volume_endpoint(), action, level)
    except Exception as exc:  # noqa: BLE001 - COM errors are many and uninteresting
        _log.error("Volume %s failed: %s", action, exc)
        return ToolResult.fail("I couldn't change the volume, sir.", detail=f"{exc!r}")
    finally:
        try:
            comtypes.CoUninitialize()
        except Exception as exc:  # noqa: BLE001 - releasing COM must never raise onwards
            _log.debug("CoUninitialize complained: %s", exc)

    if muted:
        summary = "The sound is muted, sir."
    elif action == "unmute":
        summary = f"Sound is back on at {_spoken_int(percent)} percent, sir."
    else:
        summary = f"Volume is at {_spoken_int(percent)} percent, sir."
    return ToolResult(
        ok=True, summary=summary,
        detail=f"action={action} level={level} -> {percent}% muted={muted}",
        data={"percent": percent, "muted": muted, "action": action},
    )


@tool(
    "media",
    description="Control whatever is playing music or video by pressing a media key: "
    "play or pause, next track, previous track, or stop.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "enum": list(MEDIA_KEYS),
                   "description": "Which media key to press: play_pause, next, previous or stop."},
    }, "required": ["action"]},
    tier=Tier.SAFE,
)
def media(ctx: ToolContext, args: dict) -> ToolResult:
    """Send one media key so the active player reacts, whichever player it is."""
    raw = str(args.get("action") or "").strip().lower().replace("-", "_").replace(" ", "_")
    action = _MEDIA_ALIASES.get(raw, raw)
    entry = MEDIA_KEYS.get(action)
    if entry is None:
        return ToolResult.fail("I can play or pause, skip forward or back, or stop the music, sir.",
                               detail=f"Unsupported media action {raw!r}.")
    if sys.platform != "win32":
        return _not_windows("press the media keys")
    key_code, spoken = entry
    try:
        import ctypes  # noqa: PLC0415 - windll is Windows-only

        user32 = ctypes.windll.user32
        user32.keybd_event(key_code, 0, 0, 0)
        user32.keybd_event(key_code, 0, _KEYEVENTF_KEYUP, 0)
    except Exception as exc:  # noqa: BLE001 - a missing user32 must not raise onwards
        _log.error("Media key %s failed: %s", action, exc)
        return ToolResult.fail("I couldn't reach the media keys, sir.", detail=f"{exc!r}")
    return ToolResult(
        ok=True, summary=spoken,
        detail=f"Sent virtual key 0x{key_code:02X} for {action}.",
        data={"action": action, "key": key_code},
    )
