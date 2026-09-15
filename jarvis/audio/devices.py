"""Audio device enumeration and resolution.

``sounddevice`` (PortAudio) is imported lazily inside every function here, so this
module imports cleanly on a machine with no audio stack. When the library is missing
:func:`list_devices` simply returns an empty list and :func:`print_devices` explains
how to install it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jarvis.audio.devices")

INSTALL_HINT = (
    "Audio device enumeration needs the 'sounddevice' package (PortAudio bindings). "
    "Install it with:  pip install sounddevice"
)

_KINDS = ("input", "output")

# Remember whether we already complained about the missing library, so a polling
# caller does not fill the log with the same warning.
_warned_missing = False


@dataclass
class DeviceInfo:
    """One PortAudio device as JARVIS cares about it."""

    index: int
    name: str
    max_input: int
    max_output: int
    default_samplerate: float

    def supports(self, kind: str) -> bool:
        """True when the device has at least one channel of the requested ``kind``."""
        if kind == "input":
            return self.max_input > 0
        if kind == "output":
            return self.max_output > 0
        raise ValueError(f"Unknown device kind {kind!r}; expected 'input' or 'output'.")

    def describe(self) -> str:
        """One human-readable line, used by ``--list-devices``."""
        return (
            f"[{self.index:>2}] {self.name}  "
            f"(in: {self.max_input}, out: {self.max_output}, "
            f"{self.default_samplerate:.0f} Hz)"
        )


def _import_sounddevice() -> Any | None:
    """Import ``sounddevice`` lazily, returning None when it is unavailable."""
    global _warned_missing
    try:
        import sounddevice  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, or an OSError when PortAudio is absent
        if not _warned_missing:
            _warned_missing = True
            logger.warning("sounddevice is unavailable (%s). %s", exc, INSTALL_HINT)
        return None
    return sounddevice


def list_devices() -> list[DeviceInfo]:
    """Return every audio device PortAudio can see, or [] when it cannot be queried."""
    sd = _import_sounddevice()
    if sd is None:
        return []
    try:
        raw = sd.query_devices()
    except Exception as exc:
        logger.error("Could not query audio devices: %s", exc)
        return []

    devices: list[DeviceInfo] = []
    for index, entry in enumerate(raw):
        try:
            devices.append(
                DeviceInfo(
                    index=int(entry.get("index", index)),
                    name=str(entry.get("name", f"device {index}")).strip(),
                    max_input=int(entry.get("max_input_channels", 0) or 0),
                    max_output=int(entry.get("max_output_channels", 0) or 0),
                    default_samplerate=float(entry.get("default_samplerate", 0.0) or 0.0),
                )
            )
        except Exception as exc:
            logger.warning("Skipping unreadable audio device at index %d: %s", index, exc)
    return devices


def _default_indices() -> tuple[int | None, int | None]:
    """Return (default input index, default output index); (None, None) when unknown."""
    sd = _import_sounddevice()
    if sd is None:
        return (None, None)
    try:
        default = sd.default.device
        values = tuple(default) if isinstance(default, (list, tuple)) else (default, default)
        in_index = int(values[0]) if values[0] is not None and int(values[0]) >= 0 else None
        out_index = int(values[1]) if values[1] is not None and int(values[1]) >= 0 else None
        return (in_index, out_index)
    except Exception as exc:
        logger.debug("Could not read the default audio devices: %s", exc)
        return (None, None)


def print_devices() -> None:
    """Print the device table used by ``python -m jarvis --list-devices``."""
    devices = list_devices()
    if not devices:
        print("No audio devices found.")
        print(INSTALL_HINT)
        print("If sounddevice is installed, check that a sound card is present and enabled.")
        return

    default_in, default_out = _default_indices()
    print("Audio devices:")
    for device in devices:
        marks = []
        if device.index == default_in:
            marks.append("default input")
        if device.index == default_out:
            marks.append("default output")
        suffix = f"  <- {', '.join(marks)}" if marks else ""
        print(f"  {device.describe()}{suffix}")
    print("")
    print("Set audio.input_device / audio.output_device in config.yaml to an index or")
    print("to any part of a device name (case-insensitive).")


def _candidates(devices: list[DeviceInfo], kind: str) -> list[DeviceInfo]:
    """The devices that actually have channels of the requested kind."""
    return [device for device in devices if device.supports(kind)]


def _format_candidates(candidates: list[DeviceInfo]) -> str:
    if not candidates:
        return "  (none)"
    return "\n".join(f"  {device.describe()}" for device in candidates)


def resolve_device(spec: str | int | None, *, kind: str) -> int | None:
    """Turn a config value into a PortAudio device index.

    ``None`` means "use the system default" and is returned unchanged. An ``int`` is
    validated against the device count. A ``str`` is matched case-insensitively against
    the names of devices that have channels of ``kind``: an exact name match wins,
    otherwise a substring match is used.

    Raises:
        ValueError: when ``kind`` is not "input"/"output", when an index is out of
            range, or when a name matches no device or several devices.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown device kind {kind!r}; expected 'input' or 'output'.")
    if spec is None:
        return None

    devices = list_devices()
    candidates = _candidates(devices, kind)

    if isinstance(spec, bool):  # bool is an int subclass — almost certainly a mistake
        raise ValueError(
            f"Invalid {kind} device {spec!r}: expected an index, a device name or null."
        )

    if isinstance(spec, int):
        if not devices:
            logger.warning(
                "Cannot validate %s device index %d: no device list available. %s",
                kind,
                spec,
                INSTALL_HINT,
            )
            return spec
        count = max(device.index for device in devices) + 1
        if spec < 0 or spec >= count:
            raise ValueError(
                f"{kind.capitalize()} device index {spec} is out of range "
                f"(0..{count - 1}). Available {kind} devices:\n"
                f"{_format_candidates(candidates)}"
            )
        match = next((device for device in devices if device.index == spec), None)
        if match is not None and not match.supports(kind):
            raise ValueError(
                f"Device {spec} ({match.name}) has no {kind} channels. "
                f"Available {kind} devices:\n{_format_candidates(candidates)}"
            )
        return spec

    if isinstance(spec, str):
        needle = spec.strip().lower()
        if not needle:
            return None
        if not devices:
            raise ValueError(
                f"Cannot resolve the {kind} device {spec!r}: no audio devices are "
                f"visible. {INSTALL_HINT}"
            )

        exact = [device for device in candidates if device.name.strip().lower() == needle]
        partial = [device for device in candidates if needle in device.name.lower()]
        matches = exact or partial

        if len(matches) == 1:
            chosen = matches[0]
            logger.debug("Resolved %s device %r to [%d] %s", kind, spec, chosen.index, chosen.name)
            return chosen.index
        if len(matches) > 1:
            raise ValueError(
                f"The {kind} device name {spec!r} is ambiguous; it matches "
                f"{len(matches)} devices:\n{_format_candidates(matches)}\n"
                "Use a longer substring or the numeric index."
            )
        raise ValueError(
            f"No {kind} device matches {spec!r}. Available {kind} devices:\n"
            f"{_format_candidates(candidates)}"
        )

    raise ValueError(
        f"Invalid {kind} device {spec!r} of type {type(spec).__name__}; "
        "expected an index (int), a device name (str) or null."
    )
