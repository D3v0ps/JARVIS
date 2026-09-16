"""Logging setup and structured event helpers for JARVIS.

The whole assistant logs through a single ``jarvis`` logger tree:

* a ``RotatingFileHandler`` (UTF-8, path/size/backups taken from ``config.yaml``) so the
  Swedish characters JARVIS speaks (å, ä, ö) survive in ``logs/jarvis.log``;
* a colored console handler that never crashes on a terminal which cannot encode a
  character (it falls back to ``errors="replace"``), and which is left out entirely
  when the process has no console — under ``pythonw.exe`` the file is the only record.

The structured helpers (:func:`log_transcript`, :func:`log_tool_call`, :func:`log_refusal`,
:func:`log_latency`) work even when :func:`setup_logging` was never called and never raise
into the caller — a broken log line must never take down a voice turn.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from jarvis.config import Config
    from jarvis.tools.base import ToolResult

__all__ = [
    "setup_logging", "get_logger", "log_transcript", "log_tool_call",
    "log_refusal", "log_latency", "LOGGER_NAME",
]

LOGGER_NAME = "jarvis"

# Defaults mirror the `logging:` section of the shipped config.yaml.
DEFAULT_LOG_FILE = "logs/jarvis.log"
DEFAULT_MAX_BYTES = 5_242_880
DEFAULT_BACKUPS = 3
DEFAULT_LEVEL = "INFO"

_CONFIGURED_FLAG = "_jarvis_configured"
_FALLBACK_FLAG = "_jarvis_fallback_handler"
_ARGS_PREVIEW_CHARS = 500
_setup_lock = RLock()

# --- ANSI ---------------------------------------------------------------------------
RESET = "\033[0m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD_RED = "\033[1;31m"

LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: DIM, logging.INFO: "", logging.WARNING: YELLOW,
    logging.ERROR: RED, logging.CRITICAL: BOLD_RED,
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"


# --- handlers and formatters ---------------------------------------------------------
class _PlainFormatter(logging.Formatter):
    """File formatter: strips any ANSI escape that slipped into a message."""

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_RE.sub("", super().format(record))


class _ColorFormatter(logging.Formatter):
    """Console formatter. Honours a per-record ``jarvis_color`` attribute."""

    def __init__(self, fmt: str, datefmt: str | None = None, *, use_color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return _ANSI_RE.sub("", text)
        color = getattr(record, "jarvis_color", None)
        if color is None:
            color = LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{text}{RESET}" if color else text


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that degrades instead of raising on un-encodable characters."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + self.terminator
            stream = self.stream
            try:
                stream.write(message)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))
            self.flush()
        except RecursionError:  # pragma: no cover - defensive, per stdlib convention
            raise
        except Exception:  # pragma: no cover - logging must never kill the caller
            self.handleError(record)


# --- small helpers -------------------------------------------------------------------
def _cfg_get(cfg: "Config | Mapping[str, Any] | None", dotted: str, default: Any) -> Any:
    """Read a dotted config key from a Config, a plain dict, or nothing at all."""
    if cfg is None:
        return default
    try:
        if isinstance(cfg, Mapping):
            node: Any = cfg
            for part in dotted.split("."):
                if not isinstance(node, Mapping) or part not in node:
                    return default
                node = node[part]
            return default if node is None else node
        getter = getattr(cfg, "get", None)
        if callable(getter):
            value = getter(dotted, default)
            return default if value is None else value
    except Exception:
        logging.getLogger(LOGGER_NAME).debug("Could not read config key %r; using default", dotted, exc_info=True)
    return default


def _project_root(cfg: "Config | Mapping[str, Any] | None") -> Path:
    """Directory the relative log path is resolved against."""
    try:
        cfg_path = getattr(cfg, "path", None)
        if cfg_path:
            return Path(cfg_path).resolve().parent
    except Exception:
        logging.getLogger(LOGGER_NAME).debug("Config has no usable path attribute", exc_info=True)
    try:
        from jarvis.config import project_root  # local import: avoids an import cycle

        return Path(project_root())
    except Exception:
        # jarvis/core/logging.py -> jarvis/core -> jarvis -> project root
        return Path(__file__).resolve().parents[2]


def _coerce_level(value: Any) -> int:
    """Turn 'DEBUG' / 10 / garbage into a usable logging level."""
    if isinstance(value, bool):
        return logging.INFO
    if isinstance(value, int):
        return value
    name = str(value or "").strip().upper()
    level = logging.getLevelName(name)
    if isinstance(level, int):
        return level
    logging.getLogger(LOGGER_NAME).warning("Unknown logging level %r; falling back to INFO", value)
    return logging.INFO


def _enable_windows_ansi() -> bool:
    """Turn on virtual-terminal processing so ANSI colors work in cmd.exe/PowerShell."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes  # local import: Windows console API only

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        enable_vt = 0x0004
        ok = False
        for std_handle in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(std_handle)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            if kernel32.SetConsoleMode(handle, mode.value | enable_vt):
                ok = True
        return ok
    except Exception:
        logging.getLogger(LOGGER_NAME).debug("Could not enable ANSI colors on this console", exc_info=True)
        return False


def _console_stream() -> Any:
    """stdout, reconfigured to replace characters it cannot encode — or ``None``.

    Started from ``pythonw.exe`` a process has no standard streams at all: both
    ``sys.stdout`` and ``sys.stderr`` are ``None``. ``print()`` quietly does nothing in
    that case, but a ``StreamHandler`` wrapped around ``None`` raises on every single
    record, which would turn a silent start into a storm of handler errors. Returning
    ``None`` lets the callers leave the console handler out altogether.
    """
    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is None or not callable(getattr(stream, "write", None)):
        return None
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(errors="replace")
        except Exception:
            logging.getLogger(LOGGER_NAME).debug("Console stream does not support reconfigure()", exc_info=True)
    return stream


def _use_color(cfg: "Config | Mapping[str, Any] | None", stream: Any) -> bool:
    if stream is None or not bool(_cfg_get(cfg, "logging.color", True)):
        return False
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    return _enable_windows_ansi()


def _json_preview(args: Any, limit: int = _ARGS_PREVIEW_CHARS) -> str:
    """Compact JSON rendering of tool arguments, truncated for the log line."""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        text = repr(args)
    if len(text) > limit:
        return f"{text[:limit]}… (+{len(text) - limit} chars)"
    return text


def _fallback_console_handler() -> None:
    """Give the ``jarvis`` logger a console handler when setup_logging never ran."""
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, _CONFIGURED_FLAG, False) or logger.handlers:
        return
    with _setup_lock:
        if getattr(logger, _CONFIGURED_FLAG, False) or logger.handlers:
            return
        stream = _console_stream()
        if stream is None:
            # No console to bootstrap onto (pythonw). A NullHandler keeps this path
            # from running again on every event and is dropped by setup_logging.
            handler: logging.Handler = logging.NullHandler()
        else:
            handler = _SafeStreamHandler(stream)
            handler.setFormatter(
                _ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATEFMT, use_color=_use_color(None, stream))
            )
        handler.setLevel(logging.INFO)
        setattr(handler, _FALLBACK_FLAG, True)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _event_logger(name: str) -> logging.Logger:
    _fallback_console_handler()
    return get_logger(name)


def _report_helper_failure(helper: str, error: BaseException) -> None:
    """Last-ditch reporting: a logging helper failed, say so without raising."""
    try:
        logging.getLogger(f"{LOGGER_NAME}.logging").error("Logging helper %s failed: %s", helper, error, exc_info=True)
    except Exception:
        try:
            print(f"[jarvis] logging helper {helper} failed: {error}", file=sys.stderr)
        except Exception:
            pass


# --- public API ----------------------------------------------------------------------
def setup_logging(cfg: "Config") -> logging.Logger:
    """Configure and return the root ``jarvis`` logger.

    Adds a UTF-8 ``RotatingFileHandler`` (``logging.file``/``max_bytes``/``backups``,
    resolved against the project root, directory created when missing) plus a colored
    console handler when there is a console to write to. Idempotent: the logger is marked
    with an attribute, so a second call only refreshes the level instead of duplicating
    handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    level = _coerce_level(_cfg_get(cfg, "logging.level", DEFAULT_LEVEL))

    with _setup_lock:
        if getattr(logger, _CONFIGURED_FLAG, False):
            logger.setLevel(level)
            return logger

        # Drop the bootstrap handler installed by the structured helpers, if any.
        for handler in list(logger.handlers):
            if getattr(handler, _FALLBACK_FLAG, False):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    logger.debug("Could not close the bootstrap console handler", exc_info=True)

        logger.setLevel(level)
        logger.propagate = False

        # Console handler first, so file problems can still be reported. Under
        # pythonw there is no console at all and the file is the whole record.
        stream = _console_stream()
        if stream is not None:
            console = _SafeStreamHandler(stream)
            console.setFormatter(_ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATEFMT, use_color=_use_color(cfg, stream)))
            console.setLevel(level)
            logger.addHandler(console)

        raw_path = str(_cfg_get(cfg, "logging.file", DEFAULT_LOG_FILE))
        log_path = Path(raw_path).expanduser()
        if not log_path.is_absolute():
            log_path = _project_root(cfg) / log_path
        try:
            max_bytes = int(_cfg_get(cfg, "logging.max_bytes", DEFAULT_MAX_BYTES))
            backups = int(_cfg_get(cfg, "logging.backups", DEFAULT_BACKUPS))
        except (TypeError, ValueError):
            logger.warning("Invalid logging.max_bytes/backups in config; using defaults", exc_info=True)
            max_bytes, backups = DEFAULT_MAX_BYTES, DEFAULT_BACKUPS

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max(0, max_bytes),
                backupCount=max(0, backups),
                encoding="utf-8",  # Swedish characters must survive cp1252 consoles
                delay=False,
            )
            file_handler.setFormatter(_PlainFormatter(FILE_FORMAT))
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.error("Could not open log file %s: %s — console logging only", log_path, exc, exc_info=True)

        setattr(logger, _CONFIGURED_FLAG, True)
        if stream is None:
            # Said after the file handler exists, so there is somewhere to say it.
            logger.debug("No console stream available; the log file is the only record")
        logger.debug("Logging configured (level=%s, file=%s)", logging.getLevelName(level), log_path)
        return logger


def get_logger(name: str) -> logging.Logger:
    """Return the ``jarvis.<name>`` child logger."""
    clean = (name or "").strip().strip(".")
    if not clean or clean == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if clean.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(clean)
    return logging.getLogger(f"{LOGGER_NAME}.{clean}")


def log_transcript(who: str, text: str, *, latency_ms: float | None = None) -> None:
    """Log one side of the conversation. ``who`` is ``"user"`` or ``"jarvis"``."""
    try:
        speaker = (who or "?").strip()
        key = speaker.lower()
        if key == "user":
            label, color = "USER", CYAN
        elif key == "jarvis":
            label, color = "JARVIS", GREEN
        else:
            label, color = speaker.upper() or "?", BLUE
        body = " ".join(str(text or "").split())
        suffix = f" [{float(latency_ms):.0f} ms]" if latency_ms is not None else ""
        _event_logger("transcript").info("%s: %s%s", label, body, suffix, extra={"jarvis_color": color})
    except Exception as exc:  # pragma: no cover - defensive
        _report_helper_failure("log_transcript", exc)


def log_tool_call(name: str, args: dict, result: "ToolResult", *, duration_ms: float) -> None:
    """Log a tool invocation: arguments, outcome, one-sentence summary and duration.

    ``result`` is duck-typed (``ok`` / ``summary`` / ``detail`` / ``refused``) so this
    module never has to import :mod:`jarvis.tools.base` and create an import cycle.
    """
    try:
        logger = _event_logger("tools")
        ok = bool(getattr(result, "ok", False))
        refused = bool(getattr(result, "refused", False))
        summary = str(getattr(result, "summary", "") or "")
        detail = str(getattr(result, "detail", "") or "")
        status = "refused" if refused else ("ok" if ok else "failed")
        color = RED if (refused or not ok) else YELLOW
        logger.info(
            "TOOL %s(%s) -> %s in %.0f ms: %s", name, _json_preview(args), status,
            float(duration_ms), summary, extra={"jarvis_color": color},
        )
        if detail:
            logger.debug("TOOL %s detail: %s", name, detail)
    except Exception as exc:  # pragma: no cover - defensive
        _report_helper_failure("log_tool_call", exc)


def log_refusal(reason: str, detail: str) -> None:
    """Log a refused action (blocklist hit, cancelled confirmation, policy)."""
    try:
        logger = _event_logger("safety")
        logger.warning("REFUSED: %s", str(reason or "").strip(), extra={"jarvis_color": RED})
        if detail:
            logger.debug("REFUSED detail: %s", detail)
    except Exception as exc:  # pragma: no cover - defensive
        _report_helper_failure("log_refusal", exc)


def log_latency(label: str, ms: float) -> None:
    """Log one latency measurement in milliseconds."""
    try:
        _event_logger("latency").info(
            "LATENCY %s: %.0f ms", str(label or "?"), float(ms), extra={"jarvis_color": DIM}
        )
    except Exception as exc:  # pragma: no cover - defensive
        _report_helper_failure("log_latency", exc)
