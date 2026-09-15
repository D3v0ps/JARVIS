"""Environment doctor.

Answers one question honestly: will JARVIS actually work on this machine, and if
not, what is missing and how is it fixed? ``install.ps1`` runs this at the end of
an installation, and the user can run it any time:

    python -m jarvis --preflight
    python -m jarvis --preflight --fix
    python -m jarvis --preflight --fix --set-model qwen3:14b --set-whisper medium

Exit codes: 0 everything essential passed, 1 something essential is missing,
2 the arguments made no sense. An absent GPU or Swedish voice is a warning, never
a failure - JARVIS runs on a CPU and speaks through Windows' own voice if he has
to, and the installer must be allowed to finish and say so.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jarvis.config import Config, project_root

# --------------------------------------------------------------------------- #
#  Result type
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    """One line of the report."""

    name: str
    ok: bool
    detail: str
    fix: str = ""
    #: Essential checks decide the exit code. Optional ones only warn.
    essential: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        return "MISSING" if self.essential else "warn"


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #


def _run(command: list[str], timeout: float = 8.0) -> tuple[int, str]:
    """Run a command and return (exit code, combined output). Never raises."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            creationflags=_no_window(),
        )
        return completed.returncode, ((completed.stdout or "") + (completed.stderr or "")).strip()
    except FileNotFoundError:
        return 127, f"{command[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, f"{command[0]} timed out"
    except OSError as exc:  # pragma: no cover - platform specific
        return 1, str(exc)


def _no_window() -> int:
    """CREATE_NO_WINDOW on Windows so probing does not flash console windows."""
    return 0x08000000 if sys.platform == "win32" else 0


def _has_module(name: str) -> bool:
    """True when the module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _http_json(url: str, timeout: float = 3.0) -> Any | None:
    """GET a small JSON document. Returns None on any failure."""
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - a doctor never dies of its own diagnosis
        return None


# --------------------------------------------------------------------------- #
#  Hardware detection
# --------------------------------------------------------------------------- #


def detect_vram() -> tuple[float | None, str | None]:
    """VRAM in GB and the GPU's name, from nvidia-smi. (None, None) without one."""
    if not shutil.which("nvidia-smi"):
        return None, None
    code, output = _run(
        ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"]
    )
    if code != 0 or not output:
        return None, None
    first = output.splitlines()[0]
    parts = [piece.strip() for piece in first.split(",", 1)]
    if len(parts) != 2:
        return None, None
    try:
        megabytes = float(parts[0])
    except ValueError:
        return None, None
    return round(megabytes / 1024, 1), parts[1]


def pick_whisper_model(vram_gb: float | None) -> str:
    """Whisper size that leaves room for the language model on the same card."""
    if vram_gb is None:
        return "small"
    if vram_gb >= 10:
        return "medium"
    if vram_gb >= 6:
        return "small"
    return "base"


# --------------------------------------------------------------------------- #
#  The checks
# --------------------------------------------------------------------------- #

#: (import name, pip name, what it is for, essential)
_PACKAGES: tuple[tuple[str, str, str, bool], ...] = (
    ("numpy", "numpy", "audio maths", True),
    ("yaml", "PyYAML", "configuration", True),
    ("requests", "requests", "talking to Ollama", True),
    ("sounddevice", "sounddevice", "microphone and speakers", True),
    ("openwakeword", "openwakeword", "the wake word", True),
    ("faster_whisper", "faster-whisper", "speech recognition", True),
    ("onnxruntime", "onnxruntime", "running the wake word model", True),
    ("psutil", "psutil", "system_status", False),
    ("kokoro_onnx", "kokoro-onnx", "the British voice", False),
    ("silero_vad", "silero-vad", "end-of-speech detection", False),
    ("ddgs", "ddgs", "web_search", False),
    ("PIL", "pillow", "screenshots and the tray icon", False),
    ("pystray", "pystray", "the tray icon", False),
    ("tkinter", "python3-tk / reinstall Python with tcl-tk", "the overlay", False),
    ("pyttsx3", "pyttsx3", "the fallback voice", False),
)


def check_python() -> CheckResult:
    version = sys.version_info
    text = f"{platform.python_version()} at {sys.executable}"
    if version < (3, 10):
        return CheckResult(
            "Python", False, text, "Install Python 3.13 and rebuild the virtual environment."
        )
    return CheckResult("Python", True, text)


def check_venv() -> CheckResult:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = "running inside .venv" if in_venv else "running against the system Python"
    return CheckResult(
        "Virtual environment",
        in_venv,
        detail,
        "Run start-jarvis.bat, which builds .venv and installs everything into it.",
        essential=False,
    )


def check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_essential: list[str] = []
    missing_optional: list[str] = []

    for module, package, purpose, essential in _PACKAGES:
        if _has_module(module):
            continue
        (missing_essential if essential else missing_optional).append(
            f"{module} for {purpose} -> {package}"
        )

    if missing_essential:
        results.append(
            CheckResult(
                "Required packages",
                False,
                "missing: " + ", ".join(missing_essential),
                "pip install -r requirements.txt",
            )
        )
    else:
        results.append(CheckResult("Required packages", True, "all present"))

    if missing_optional:
        results.append(
            CheckResult(
                "Optional packages",
                False,
                "missing: " + ", ".join(missing_optional),
                "pip install -r requirements.txt",
                essential=False,
            )
        )
    return results


def check_gpu(vram_gb: float | None, gpu_name: str | None) -> CheckResult:
    if vram_gb is None:
        return CheckResult(
            "Graphics card",
            False,
            "no NVIDIA GPU detected - speech recognition will run on the CPU",
            "Nothing to do. CPU int8 is slower but perfectly usable.",
            essential=False,
        )
    return CheckResult("Graphics card", True, f"{gpu_name} with {vram_gb} GB of VRAM")


def check_ollama(cfg: Config) -> list[CheckResult]:
    results: list[CheckResult] = []
    host = str(cfg.get("brain.host", "http://127.0.0.1:11434")).rstrip("/")
    wanted = str(cfg.get("brain.model", "qwen3:8b"))

    binary = shutil.which("ollama")
    if binary:
        code, output = _run(["ollama", "--version"])
        version = output.splitlines()[0] if code == 0 and output else "installed"
        results.append(CheckResult("Ollama", True, version))
    else:
        results.append(
            CheckResult(
                "Ollama",
                False,
                "not on PATH",
                "winget install Ollama.Ollama",
            )
        )

    tags = _http_json(f"{host}/api/tags")
    if tags is None:
        results.append(
            CheckResult(
                "Ollama service",
                False,
                f"nothing answered on {host}",
                "Run 'ollama serve' in a terminal, or restart the machine.",
            )
        )
        # Without a daemon there is nothing to say about the model.
        results.append(
            CheckResult(
                "Language model",
                False,
                f"cannot check for {wanted} while the service is down",
                f"ollama pull {wanted}",
            )
        )
        return results

    results.append(CheckResult("Ollama service", True, f"answering on {host}"))

    names = [str(model.get("name", "")) for model in (tags.get("models") or [])]
    base = wanted.split(":", 1)[0]
    if any(name == wanted or name.startswith(base + ":") for name in names):
        matched = next(n for n in names if n == wanted or n.startswith(base + ":"))
        results.append(CheckResult("Language model", True, matched))
    else:
        have = ", ".join(names) if names else "none"
        results.append(
            CheckResult(
                "Language model",
                False,
                f"{wanted} is not pulled (have: {have})",
                f"ollama pull {wanted}",
            )
        )
    return results


def check_voice(cfg: Config) -> list[CheckResult]:
    results: list[CheckResult] = []
    engine = str(cfg.get("tts.engine", "kokoro")).lower()

    model = cfg.resolve_path("tts.kokoro_model")
    voices = cfg.resolve_path("tts.kokoro_voices")
    if model.is_file() and voices.is_file():
        size_mb = (model.stat().st_size + voices.stat().st_size) / (1024 * 1024)
        results.append(CheckResult("Kokoro voice", True, f"{model.name} + {voices.name} ({size_mb:.0f} MB)"))
    else:
        absent = [p.name for p in (model, voices) if not p.is_file()]
        results.append(
            CheckResult(
                "Kokoro voice",
                False,
                "missing: " + ", ".join(absent) + (" - falling back to the Windows voice" if engine == "kokoro" else ""),
                "python scripts/fetch_models.py",
                essential=False,
            )
        )

    piper = cfg.resolve_path("tts.piper_model")
    results.append(
        CheckResult(
            "Swedish voice",
            piper.is_file(),
            piper.name if piper.is_file() else "not installed (only needed for spoken Swedish)",
            "python scripts/fetch_models.py --swedish",
            essential=False,
        )
    )
    return results


def check_wake_models() -> CheckResult:
    """openWakeWord keeps its models inside its own package directory."""
    if not _has_module("openwakeword"):
        return CheckResult(
            "Wake word model",
            False,
            "openwakeword is not installed",
            "pip install -r requirements.txt",
        )
    try:
        spec = importlib.util.find_spec("openwakeword")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("no package path")
        package_dir = Path(list(spec.submodule_search_locations)[0])
    except Exception:  # noqa: BLE001
        return CheckResult(
            "Wake word model", False, "could not locate the openwakeword package",
            "pip install --force-reinstall openwakeword",
        )

    candidates = list((package_dir / "resources" / "models").glob("hey_jarvis*"))
    if candidates:
        return CheckResult("Wake word model", True, f"hey_jarvis ({len(candidates)} file(s))")
    return CheckResult(
        "Wake word model",
        False,
        "hey_jarvis has not been downloaded yet",
        "python scripts/fetch_models.py",
    )


def check_audio_devices() -> list[CheckResult]:
    if not _has_module("sounddevice"):
        return [
            CheckResult("Audio devices", False, "sounddevice is not installed", "pip install -r requirements.txt")
        ]
    try:
        import sounddevice  # noqa: PLC0415 - deliberately lazy

        devices = sounddevice.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
    except Exception as exc:  # noqa: BLE001 - PortAudio raises all sorts
        return [
            CheckResult(
                "Audio devices",
                False,
                f"PortAudio would not start: {exc}",
                "Check that Windows can see your microphone in Sound settings.",
            )
        ]

    results = [
        CheckResult(
            "Microphone",
            bool(inputs),
            f"{len(inputs)} input device(s); default: {_default_device_name(0)}" if inputs else "none found",
            "Plug in a microphone and allow apps to use it in Windows privacy settings.",
        ),
        CheckResult(
            "Speakers",
            bool(outputs),
            f"{len(outputs)} output device(s); default: {_default_device_name(1)}" if outputs else "none found",
            "Check the playback device in Windows sound settings.",
        ),
    ]
    return results


def _default_device_name(index: int) -> str:
    try:
        import sounddevice

        default = sounddevice.default.device[index]
        if default is None or default < 0:
            return "system default"
        return str(sounddevice.query_devices(default)["name"])
    except Exception:  # noqa: BLE001
        return "unknown"


def check_files(cfg: Config) -> list[CheckResult]:
    root = project_root()
    results: list[CheckResult] = []

    prompt = root / "prompts" / "jarvis_system.md"
    results.append(
        CheckResult(
            "System prompt",
            prompt.is_file(),
            str(prompt.relative_to(root)) if prompt.is_file() else "prompts/jarvis_system.md is missing",
            "Restore the file from the repository - JARVIS has no personality without it.",
        )
    )

    log_path = cfg.resolve_path("logging.file")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        probe = log_path.parent / ".preflight-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
        detail = f"{log_path.parent} is writable"
    except OSError as exc:
        writable = False
        detail = f"cannot write to {log_path.parent}: {exc}"
    results.append(
        CheckResult(
            "Log directory",
            writable,
            detail,
            "Move JARVIS somewhere your user can write, such as C:\\JARVIS.",
        )
    )
    return results


# --------------------------------------------------------------------------- #
#  Running everything
# --------------------------------------------------------------------------- #


def run_checks(cfg: Config, *, fix: bool = False) -> list[CheckResult]:
    """Run every check. ``fix`` writes the detected hardware into config.yaml."""
    vram_gb, gpu_name = detect_vram()

    results: list[CheckResult] = [check_python(), check_venv()]
    results.extend(check_packages())
    results.append(check_gpu(vram_gb, gpu_name))
    results.extend(check_ollama(cfg))
    results.append(check_wake_models())
    results.extend(check_voice(cfg))
    results.extend(check_audio_devices())
    results.extend(check_files(cfg))

    if fix:
        for change in apply_fixes(cfg):
            results.append(CheckResult("Configuration", True, change, essential=False))
    return results


def apply_fixes(
    cfg: Config, *, model: str | None = None, whisper: str | None = None
) -> list[str]:
    """Write what we detected into config.yaml. Returns human-readable changes.

    Never raises: a read-only config file is a warning, because the installer has
    to be able to finish and tell the user what it could not do.
    """
    changes: list[str] = []
    vram_gb, gpu_name = detect_vram()

    if vram_gb is not None and cfg.get("system.vram_gb") != vram_gb:
        cfg.set("system.vram_gb", vram_gb)
        changes.append(f"system.vram_gb = {vram_gb}")
    if gpu_name and cfg.get("system.gpu_name") != gpu_name:
        cfg.set("system.gpu_name", gpu_name)
        changes.append(f"system.gpu_name = {gpu_name}")

    if model:
        cfg.set("brain.model", model)
        changes.append(f"brain.model = {model}")

    chosen_whisper = whisper or (pick_whisper_model(vram_gb) if cfg.get("stt.model") == "auto" else None)
    if chosen_whisper and cfg.get("stt.model") != chosen_whisper:
        cfg.set("stt.model", chosen_whisper)
        changes.append(f"stt.model = {chosen_whisper}")

    if not changes:
        return []

    try:
        saved = cfg.save()
        if saved is False:
            return [f"could not write {cfg.path} - the settings were not saved"]
    except Exception as exc:  # noqa: BLE001 - never take the installer down
        return [f"could not write {cfg.path}: {exc}"]
    return changes


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:  # noqa: BLE001
            return False
    return True


def print_report(results: Iterable[CheckResult]) -> None:
    results = list(results)
    color = _colors_enabled()

    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    width = max((len(r.name) for r in results), default=10)
    print()
    print(paint("  JARVIS preflight", _BOLD))
    print(paint("  " + "-" * (width + 50), _DIM))

    for result in results:
        if result.ok:
            mark, code = "ok  ", _GREEN
        elif result.essential:
            mark, code = "FAIL", _RED
        else:
            mark, code = "warn", _YELLOW
        print(f"  {paint(mark, code)}  {result.name.ljust(width)}  {result.detail}")

    problems = [r for r in results if not r.ok and r.fix]
    if problems:
        print()
        print(paint("  What to do about it", _BOLD))
        seen: set[str] = set()
        for result in problems:
            if result.fix in seen:
                continue
            seen.add(result.fix)
            print(f"    {result.name}: {result.fix}")

    essential_failures = [r for r in results if not r.ok and r.essential]
    print()
    if essential_failures:
        print(paint(f"  {len(essential_failures)} essential check(s) failed.", _RED))
    else:
        warnings = [r for r in results if not r.ok]
        if warnings:
            print(paint(f"  Ready, with {len(warnings)} thing(s) worth knowing about.", _YELLOW))
        else:
            print(paint("  Everything checks out. Say \"Hey Jarvis\".", _GREEN))
    print()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis --preflight",
        description="Check whether JARVIS can run on this machine.",
        add_help=True,
    )
    parser.add_argument("--fix", action="store_true", help="write detected values into config.yaml")
    parser.add_argument("--set-model", dest="set_model", default=None, help="write brain.model")
    parser.add_argument("--set-whisper", dest="set_whisper", default=None, help="write stt.model")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--config", default=None, help="path to config.yaml")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a bad argument; keep that, but never let --help
        # look like a failure to the installer.
        return int(exc.code or 0)

    try:
        cfg = Config.load(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read the configuration: {exc}", file=sys.stderr)
        return 1

    # --set-model / --set-whisper imply --fix.
    should_fix = args.fix or bool(args.set_model) or bool(args.set_whisper)
    changes: list[str] = []
    if should_fix:
        changes = apply_fixes(cfg, model=args.set_model, whisper=args.set_whisper)
        # Re-read so the checks below see what we just wrote.
        try:
            cfg = Config.load(args.config)
        except Exception:  # noqa: BLE001
            pass

    results = run_checks(cfg, fix=False)
    for change in changes:
        results.append(CheckResult("Configuration", True, change, essential=False))

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not any(not r.ok and r.essential for r in results),
                    "checks": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "essential": r.essential,
                            "detail": r.detail,
                            "fix": r.fix,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        print_report(results)

    return 1 if any(not r.ok and r.essential for r in results) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
