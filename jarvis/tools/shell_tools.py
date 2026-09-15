"""Shell and power tools: the two sharpest things JARVIS is allowed to do.

Both tools are GUARDED, so the dispatcher speaks the announcement and waits for a
spoken confirmation before either of them runs. :func:`run_powershell` additionally
re-runs the blocklist on the exact command string it is about to execute — defence in
depth, in case the tool is ever called without going through the dispatcher.

Nothing Windows-only is imported at module level: the module imports and its helpers
are testable on Linux, where the executables simply do not exist and the tools return
a calm sentence instead of a traceback.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from jarvis.core.logging import get_logger, log_refusal
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool
from jarvis.tools.safety import check_blocked

__all__ = ["run_powershell", "power", "POWER_ACTIONS", "CREATE_NO_WINDOW"]

_log = get_logger("tools.shell")

#: ``subprocess`` creation flag that keeps a console window from flashing up on
#: Windows. Defined here as a plain int so nothing Windows-only has to be imported.
CREATE_NO_WINDOW = 0x08000000

#: Default when ``tools.powershell_timeout`` is missing or unusable.
DEFAULT_POWERSHELL_TIMEOUT = 30

#: Seconds allowed for the (near-instant) power commands themselves.
POWER_TIMEOUT = 15

#: How long a single spoken output line may be before it is trimmed.
_SPOKEN_LINE_CHARS = 180

_POWERSHELL_FLAGS = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]

#: action -> (argv, spoken confirmation)
POWER_ACTIONS: dict[str, tuple[list[str], str]] = {
    "sleep": (
        ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
        "Going to sleep now, sir.",
    ),
    "restart": (
        ["shutdown", "/r", "/t", "5"],
        "Restarting in five seconds, sir.",
    ),
    "shutdown": (
        ["shutdown", "/s", "/t", "5"],
        "Shutting down in five seconds, sir.",
    ),
}

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _spoken_number(value: int) -> str:
    """Render a small whole number as words, so the summary reads well aloud.

    Anything above nine hundred and ninety-nine stays as digits — a voice line that
    long is no clearer spelled out.
    """
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
    return head if rest == 0 else f"{head} and {_spoken_number(rest)}"


def _timeout(ctx: ToolContext) -> int:
    """``tools.powershell_timeout`` from the config, with a sane floor."""
    try:
        raw: Any = ctx.config.get("tools.powershell_timeout", DEFAULT_POWERSHELL_TIMEOUT)
        seconds = int(float(raw))
    except (AttributeError, TypeError, ValueError):
        _log.debug("Unusable tools.powershell_timeout; using the default", exc_info=True)
        return DEFAULT_POWERSHELL_TIMEOUT
    return max(1, seconds)


def _popen_kwargs() -> dict[str, Any]:
    """Extra ``subprocess`` keywords: hide the console window on Windows only."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def _meaningful_lines(text: str) -> list[str]:
    """Non-empty, whitespace-trimmed lines of a command's output."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _trim_for_speech(line: str) -> str:
    """Shorten one output line to something a voice can deliver in a breath."""
    clean = " ".join(str(line or "").split())
    if len(clean) <= _SPOKEN_LINE_CHARS:
        return clean
    cut = clean.rfind(" ", 0, _SPOKEN_LINE_CHARS - 1)
    if cut < _SPOKEN_LINE_CHARS // 2:
        cut = _SPOKEN_LINE_CHARS - 1
    return clean[:cut].rstrip(" ,;:-") + "…"


def _detail(command: str, stdout: str, stderr: str, returncode: int | None) -> str:
    """The full, unspoken record of a run — this is what ends up in the log."""
    parts = [f"$ {command}", f"exit code: {returncode}"]
    parts.append(f"stdout:\n{stdout}" if stdout else "stdout: (empty)")
    parts.append(f"stderr:\n{stderr}" if stderr else "stderr: (empty)")
    return "\n".join(parts)


@tool(
    "run_powershell",
    description=(
        "Run a PowerShell command on the user's Windows machine and report what it "
        "printed. Use it for system questions and small administrative jobs that no "
        "other tool covers. Destructive or security-weakening commands are refused."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The PowerShell command to run, exactly as typed at a prompt.",
            }
        },
        "required": ["command"],
    },
    tier=Tier.GUARDED,
    announce="I'm about to run this in PowerShell: {command}.",
)
def run_powershell(ctx: ToolContext, args: dict) -> ToolResult:
    """Run one PowerShell command and summarise its output in a single sentence.

    The command is executed with ``-NoProfile -NonInteractive -ExecutionPolicy Bypass``
    so nothing in the user's profile interferes and nothing can sit waiting for input.
    The blocklist is re-checked here on the exact command string: the dispatcher
    already did it, but this tool must be safe on its own as well.

    The spoken summary is deliberately thin — no output, the single line that was
    printed, or a count of lines — while the complete stdout and stderr always go into
    ``detail`` for the log.
    """
    command = str(args.get("command") or "").strip()
    if not command:
        return ToolResult.fail("I need a command to run, sir.")

    # Defence in depth: safe even when called without the dispatcher.
    reason = check_blocked(command)
    if reason is not None:
        detail = f"Blocked PowerShell command: {command!r} ({reason})."
        log_refusal(reason, detail)
        return ToolResult.refuse(
            "I'm afraid that's beyond what I'm willing to do, sir.", detail=detail
        )

    timeout = _timeout(ctx)
    try:
        completed = subprocess.run(
            ["powershell.exe", *_POWERSHELL_FLAGS, command],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs(),
        )
    except subprocess.TimeoutExpired:
        _log.warning("PowerShell command timed out after %s s: %s", timeout, command)
        return ToolResult.fail(
            f"That command was still running after {_spoken_number(timeout)} seconds, "
            "sir, so I stopped it.",
            detail=f"$ {command}\nTimed out after {timeout} seconds.",
        )
    except FileNotFoundError:
        _log.error("powershell.exe is not available on this system")
        return ToolResult.fail(
            "I can only run PowerShell on Windows, sir.",
            detail=f"powershell.exe was not found while running: {command}",
        )
    except OSError as exc:
        _log.error("Could not start PowerShell: %s", exc)
        return ToolResult.fail(
            "I couldn't start PowerShell, sir.", detail=f"$ {command}\n{exc}"
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    returncode = completed.returncode
    detail = _detail(command, stdout, stderr, returncode)
    lines = _meaningful_lines(stdout)
    data = {"returncode": returncode, "lines": len(lines)}

    if returncode:
        error_lines = _meaningful_lines(stderr)
        if error_lines:
            summary = f"PowerShell objected, sir: {_trim_for_speech(error_lines[0])}"
        else:
            summary = (
                f"That command ended with exit code {_spoken_number(returncode)}, sir."
            )
        return ToolResult(ok=False, summary=summary, detail=detail, data=data)

    if not lines:
        summary = "Done, sir - no output."
    elif len(lines) == 1:
        summary = _trim_for_speech(lines[0])
    else:
        summary = (
            f"Done, sir - {_spoken_number(len(lines))} lines of output; "
            "I've put them in the log."
        )
    return ToolResult(ok=True, summary=summary, detail=detail, data=data)


@tool(
    "power",
    description=(
        "Put the machine to sleep, restart it, or shut it down. Restart and shutdown "
        "are scheduled five seconds out so they can still be waved off."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sleep", "restart", "shutdown"],
                "description": "What to do with the machine: sleep, restart or shutdown.",
            }
        },
        "required": ["action"],
    },
    tier=Tier.GUARDED,
    announce="I'm about to {action} the machine, sir.",
)
def power(ctx: ToolContext, args: dict) -> ToolResult:
    """Change the machine's power state after a confirmed, announced request.

    ``sleep`` calls ``rundll32.exe powrprof.dll,SetSuspendState 0,1,0``. Note that
    Windows ignores the "do not hibernate" argument when hibernation is enabled, so on
    a machine with hibernation on this hibernates rather than sleeps — which is
    harmless, but the user may notice a slower wake. ``restart`` and ``shutdown`` use
    ``shutdown /r /t 5`` and ``shutdown /s /t 5``, leaving a five-second window in
    which ``shutdown /a`` would abort them.
    """
    action = str(args.get("action") or "").strip().lower()
    entry = POWER_ACTIONS.get(action)
    if entry is None:
        return ToolResult.fail(
            "I can only put the machine to sleep, restart it, or shut it down, sir.",
            detail=f"Unsupported power action {action!r}; expected one of "
            f"{', '.join(sorted(POWER_ACTIONS))}.",
        )
    argv, spoken = entry

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=POWER_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs(),
        )
    except subprocess.TimeoutExpired:
        _log.warning("Power command %s did not return in time", action)
        return ToolResult.fail(
            f"The {action} command did not come back, sir, so I stopped waiting.",
            detail=f"$ {' '.join(argv)}\nTimed out after {POWER_TIMEOUT} seconds.",
        )
    except FileNotFoundError:
        _log.error("Power command %s is not available on this system", action)
        return ToolResult.fail(
            "I can only change the power state on Windows, sir.",
            detail=f"{argv[0]} was not found while trying to {action} the machine.",
        )
    except OSError as exc:
        _log.error("Could not run the %s command: %s", action, exc)
        return ToolResult.fail(
            f"I couldn't {action} the machine, sir.",
            detail=f"$ {' '.join(argv)}\n{exc}",
        )

    detail = _detail(" ".join(argv), completed.stdout or "", completed.stderr or "",
                     completed.returncode)
    data = {"action": action, "returncode": completed.returncode}
    if completed.returncode:
        error_lines = _meaningful_lines(completed.stderr or "")
        summary = (
            f"Windows refused that, sir: {_trim_for_speech(error_lines[0])}"
            if error_lines
            else f"The {action} command failed with exit code "
            f"{_spoken_number(completed.returncode)}, sir."
        )
        return ToolResult(ok=False, summary=summary, detail=detail, data=data)
    return ToolResult(ok=True, summary=spoken, detail=detail, data=data)
