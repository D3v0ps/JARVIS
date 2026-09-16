"""One reading of the machine, cheap enough that two faces can poll it every second.

The phone's status strip and the desk window's telemetry card are looking at the same
computer, so they must not each grow their own idea of what it is doing - and they
must not each shell out to ``nvidia-smi``, which is by far the slowest thing here. One
function, one module-level cache, and the callers simply ask again.

Everything in the reading is optional. ``psutil`` may not be installed, there may be
no NVIDIA card, there may be no assistant at all; every one of those means the key is
simply absent. A missing number is honest, a zero would be a lie, and an exception on
the way to a status bar would be absurd.
"""

from __future__ import annotations

import threading
import time
from typing import Any

__all__ = ["snapshot", "reset_cache"]

#: The machine reading, shared by every caller. There is only one machine, so the
#: cache is keyed by nothing at all.
_cache: tuple[float, dict[str, Any]] | None = None
_cache_lock = threading.Lock()

_GIB = 1024 ** 3
#: The soonest few jobs; a status bar with twenty countdowns on it is a list, not a
#: glance.
_MAX_TIMERS = 5


def reset_cache() -> None:
    """Forget the cached reading, so the next :func:`snapshot` truly reads the machine."""
    global _cache
    with _cache_lock:
        _cache = None


def snapshot(assistant: Any = None, *, cache_seconds: float = 2.5) -> dict[str, Any]:
    """How the machine and the assistant are doing, right now.

    Keys: ``state`` always; ``cpu``, ``ram``, ``ram_used_gb``, ``ram_total_gb``,
    ``disk_free_gb`` and ``uptime_s`` when ``psutil`` is installed; ``gpu`` when
    ``nvidia-smi`` answers; ``timers`` and ``model`` when ``assistant`` can say.

    The machine half is cached for ``cache_seconds``; the assistant half is free to
    read and is therefore always current, which is what a state light needs.
    """
    reading = _machine(cache_seconds)
    reading["state"] = _state(assistant)
    if assistant is None:
        return reading
    timers = _timers(assistant)
    if timers is not None:
        reading["timers"] = timers
    model = _model(assistant)
    if model:
        reading["model"] = model
    return reading


# --- the machine --------------------------------------------------------------------
def _machine(cache_seconds: float) -> dict[str, Any]:
    """The counters, re-read at most every ``cache_seconds``."""
    global _cache
    now = time.monotonic()
    with _cache_lock:
        cached = _cache
        if cached is not None and cache_seconds > 0 and now - cached[0] < cache_seconds:
            return _copy(cached[1])

    reading: dict[str, Any] = {}
    _read_counters(reading)
    _read_gpu(reading)

    with _cache_lock:
        _cache = (now, reading)
    return _copy(reading)


def _copy(reading: dict[str, Any]) -> dict[str, Any]:
    """A caller may add its own keys to the answer; the cache must not notice."""
    clone = dict(reading)
    gpu = clone.get("gpu")
    if isinstance(gpu, dict):
        clone["gpu"] = dict(gpu)
    return clone


def _read_counters(reading: dict[str, Any]) -> None:
    """CPU, memory, disk and uptime, each independently optional."""
    try:
        import psutil  # noqa: PLC0415 - optional, Windows-target dependency
    except Exception:  # noqa: BLE001 - no psutil, no counters, no complaint
        return
    try:
        reading["cpu"] = round(float(psutil.cpu_percent(interval=None)))
    except Exception:  # noqa: BLE001
        pass
    try:
        memory = psutil.virtual_memory()
        reading["ram"] = round(float(memory.percent))
        reading["ram_used_gb"] = round(memory.used / _GIB, 1)
        reading["ram_total_gb"] = round(memory.total / _GIB, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        reading["disk_free_gb"] = round(psutil.disk_usage(_disk_root()).free / _GIB, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        reading["uptime_s"] = int(max(0.0, time.time() - float(psutil.boot_time())))
    except Exception:  # noqa: BLE001
        pass


def _disk_root() -> str:
    """The drive the operator means by "free space": the one Windows booted from."""
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if sys.platform == "win32":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


def _read_gpu(reading: dict[str, Any]) -> None:
    """Utilisation, temperature and VRAM, through the one ``nvidia-smi`` reader we have."""
    try:
        from jarvis.tools.system_tools import _gpu_info  # noqa: PLC0415

        gpu = _gpu_info()
    except Exception:  # noqa: BLE001 - no card, no driver, no key
        return
    if not gpu:
        return
    try:
        reading["gpu"] = {key: round(float(value)) for key, value in gpu.items()}
    except Exception:  # noqa: BLE001 - nvidia-smi said something unexpected
        pass


# --- the assistant ------------------------------------------------------------------
def _state(assistant: Any) -> str:
    """What JARVIS is doing; ``idle`` when there is nobody to ask."""
    parts = getattr(assistant, "parts", None)
    bus = getattr(parts, "state", None) or getattr(assistant, "state", None)
    try:
        return str(bus.state) if bus is not None else "idle"
    except Exception:  # noqa: BLE001
        return "idle"


def _timers(assistant: Any) -> list[dict[str, Any]] | None:
    """The soonest few timers and reminders, or ``None`` when there is no scheduler."""
    parts = getattr(assistant, "parts", None)
    scheduler = getattr(parts, "scheduler", None) or getattr(assistant, "scheduler", None)
    if scheduler is None:
        return None
    try:
        jobs = sorted(scheduler.pending(), key=lambda job: job.due)[:_MAX_TIMERS]
        return [
            {
                "id": job.id,
                "label": job.label or job.text or job.kind,
                "due": job.due,
                "kind": job.kind,
            }
            for job in jobs
        ]
    except Exception:  # noqa: BLE001 - a broken scheduler is not a broken status bar
        return []


def _model(assistant: Any) -> str:
    """The model Ollama is actually holding, which is not always the configured one.

    Wiring falls back to whatever is pulled when ``brain.model`` is not, so the live
    client is the only honest source; when there is no client we say nothing.
    """
    parts = getattr(assistant, "parts", None)
    for owner in (parts, getattr(parts, "brain", None), assistant):
        try:
            name = str(getattr(getattr(owner, "client", None), "model", "") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if name:
            return name
    return ""
