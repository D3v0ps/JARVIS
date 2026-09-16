"""Per-connection state for the phone: pairing, rate limits, wire decoding, guarding.

Everything in this module is pure standard library plus numpy, so it imports on a bare
Linux box with no Flask installed. The Flask layer lives in
:mod:`jarvis.remote.server` and leans on the primitives here.

Three separate concerns live together because they are all "one phone, one connection":

* :class:`SessionStore` — the one-time pairing code, the signed cookie it turns into,
  and the rate limit that stops someone guessing the code.
* :func:`decode_pcm16` — the wire format. The page sends Int16 little-endian mono at
  16 kHz, which is exactly what :meth:`jarvis.stt.transcriber.Transcriber.transcribe`
  wants once it is scaled back to float.
* :class:`RemoteGuard` — a dispatcher wrapper that refuses GUARDED tools when the turn
  came from the phone, because a spoken confirmation is worthless when whoever holds
  the phone also holds the microphone.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jarvis.core.logging import get_logger, log_refusal
from jarvis.tools import registry, safety
from jarvis.tools.base import Tier, ToolResult

__all__ = [
    "COOKIE_NAME",
    "REMOTE_REFUSAL",
    "Device",
    "PairResult",
    "RemoteGuard",
    "RemoteTurn",
    "Routine",
    "SessionStore",
    "decode_pcm16",
    "load_or_create_secret",
    "load_routines",
]

_log = get_logger("remote.session")

#: Name of the signed, HttpOnly cookie handed out after a successful pairing.
COOKIE_NAME = "jarvis_remote"

#: Spoken when a tool is refused purely because the request came from the phone.
REMOTE_REFUSAL = "I won't do that from the phone, sir."

#: Token layout version, so an old cookie is rejected rather than misread.
_TOKEN_VERSION = "v1"

#: Digits in a pairing code. Six is the most a person will retype without resentment;
#: the rate limit, not the length, is what makes guessing hopeless.
DEFAULT_CODE_LENGTH = 6


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def load_or_create_secret(path: str | Path, logger: logging.Logger | None = None) -> bytes:
    """Return the cookie-signing secret at ``path``, creating it the first time.

    The secret is persisted so a restart does not un-pair the phone. If the file
    cannot be read or written — a read-only install directory, a locked file — a
    fresh in-memory secret is returned and the fact is logged: remote access still
    works, it simply asks to be paired again after every restart.
    """
    log = logger or _log
    target = Path(path)
    try:
        if target.is_file():
            raw = target.read_text(encoding="utf-8").strip()
            secret = bytes.fromhex(raw) if raw else b""
            if len(secret) >= 16:
                return secret
            log.warning("Remote secret in %s is too short; generating a new one.", target)
    except (OSError, ValueError) as exc:
        log.warning("Could not read the remote secret from %s (%s); generating one.", target, exc)

    secret = secrets.token_bytes(32)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(secret.hex(), encoding="utf-8")
        try:  # Best effort: on Windows this is a no-op, which is fine.
            os.chmod(target, 0o600)
        except OSError:
            pass
    except OSError as exc:
        log.warning(
            "Could not save the remote secret to %s (%s); paired devices will have to "
            "pair again after a restart.",
            target,
            exc,
        )
    return secret


# ----------------------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Device:
    """A paired phone, as recovered from its cookie."""

    name: str
    id: str
    issued_at: float = 0.0
    expires_at: float = 0.0

    def __str__(self) -> str:
        return self.name or self.id or "an unnamed device"


@dataclass(frozen=True)
class PairResult:
    """The outcome of one pairing attempt. ``status`` is what the page branches on."""

    ok: bool
    status: str  # ok | bad_code | expired | rate_limited | malformed
    token: str = ""
    device: Device | None = None
    retry_after: float = 0.0

    @property
    def message(self) -> str:
        """A short line for the phone's screen (displayed, never spoken)."""
        return {
            "ok": "Paired.",
            "bad_code": "That code is not right.",
            "expired": "That code has expired; a new one is on the console.",
            "rate_limited": "Too many attempts. Wait a moment and try again.",
            "malformed": "A pairing code is six digits.",
        }.get(self.status, "Pairing failed.")


class SessionStore:
    """Pairing codes, signed session tokens and the rate limit that guards them.

    Thread-safety is by GIL-atomic operations on small structures plus the fact that
    every mutating method is short; there is one store per :class:`RemoteServer`.
    """

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_hours: float = 720.0,
        max_attempts: int = 5,
        attempt_window: float = 300.0,
        code_ttl: float = 600.0,
        code_length: int = DEFAULT_CODE_LENGTH,
        clock: Callable[[], float] = time.time,
        logger: logging.Logger | None = None,
    ) -> None:
        if not secret:
            raise ValueError("SessionStore needs a non-empty signing secret")
        self._secret = bytes(secret)
        self.ttl_seconds = max(60.0, float(ttl_hours) * 3600.0)
        self.max_attempts = max(1, int(max_attempts))
        self.attempt_window = max(1.0, float(attempt_window))
        self.code_ttl = max(30.0, float(code_ttl))
        self.code_length = max(4, int(code_length))
        self._clock = clock
        self._log = logger or _log
        self._attempts: dict[str, list[float]] = {}
        #: Bumped by :meth:`revoke_all`, which invalidates every cookie already issued.
        self._generation = 1
        self._code = ""
        self._code_created = 0.0
        self.rotate_code()

    # --- the code ------------------------------------------------------------------
    @property
    def pairing_code(self) -> str:
        """The current pairing code. Printed to the console, never logged to file."""
        return self._code

    @property
    def code_expired(self) -> bool:
        return (self._clock() - self._code_created) > self.code_ttl

    def rotate_code(self) -> str:
        """Generate a fresh pairing code and return it."""
        upper = 10**self.code_length
        self._code = str(secrets.randbelow(upper)).zfill(self.code_length)
        self._code_created = self._clock()
        return self._code

    # --- attempts ------------------------------------------------------------------
    def _recent_attempts(self, client_id: str) -> list[float]:
        now = self._clock()
        recent = [t for t in self._attempts.get(client_id, []) if now - t < self.attempt_window]
        if recent:
            self._attempts[client_id] = recent
        else:
            self._attempts.pop(client_id, None)
        return recent

    def attempts_left(self, client_id: str) -> int:
        """How many wrong codes this client may still send before being locked out."""
        return max(0, self.max_attempts - len(self._recent_attempts(str(client_id))))

    def _record_failure(self, client_id: str) -> None:
        self._attempts.setdefault(str(client_id), []).append(self._clock())

    # --- pairing -------------------------------------------------------------------
    def pair(self, code: str, device_name: str, client_id: str = "") -> PairResult:
        """Check ``code`` and, when it matches, mint a signed session token.

        A correct code is consumed: a new one is generated immediately, so the same
        code can never pair a second device. Wrong codes are counted per client and
        refused outright once :attr:`max_attempts` have been used inside the window.
        """
        client = str(client_id or "unknown")
        offered = "".join(ch for ch in str(code or "") if ch.isdigit())

        if len(self._recent_attempts(client)) >= self.max_attempts:
            oldest = min(self._recent_attempts(client))
            retry_after = max(1.0, self.attempt_window - (self._clock() - oldest))
            self._log.warning("Pairing attempt from %s refused: rate limited.", client)
            return PairResult(False, "rate_limited", retry_after=retry_after)

        if len(offered) != self.code_length:
            self._record_failure(client)
            return PairResult(False, "malformed")

        # Constant time: a timing side channel on a six-digit code is not theoretical.
        if not hmac.compare_digest(offered, self._code):
            self._record_failure(client)
            self._log.warning(
                "Wrong pairing code from %s (%d attempt(s) left).", client, self.attempts_left(client)
            )
            return PairResult(False, "bad_code")

        if self.code_expired:
            self._record_failure(client)
            self.rotate_code()
            self._log.warning("Pairing code from %s was correct but expired; rotated.", client)
            return PairResult(False, "expired")

        self._attempts.pop(client, None)
        now = self._clock()
        device = Device(
            name=_clean_device_name(device_name),
            id=secrets.token_hex(8),
            issued_at=now,
            expires_at=now + self.ttl_seconds,
        )
        token = self._sign_device(device)
        self.rotate_code()
        self._log.info("Paired with %s from %s.", device, client)
        return PairResult(True, "ok", token=token, device=device)

    # --- tokens --------------------------------------------------------------------
    def _sign_device(self, device: Device) -> str:
        payload = json.dumps(
            {
                "n": device.name,
                "i": device.id,
                "t": int(device.issued_at),
                "e": int(device.expires_at),
                "g": self._generation,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{_TOKEN_VERSION}.{_b64e(payload)}.{_b64e(signature)}"

    def verify_token(self, token: str | None) -> Device | None:
        """Return the :class:`Device` a token belongs to, or ``None`` if it is not ours.

        Rejects, silently and identically: a missing token, a wrong version, a bad
        signature, a token from before the last :meth:`revoke_all`, and an expired one.
        """
        if not token or not isinstance(token, str):
            return None
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _TOKEN_VERSION:
            return None
        try:
            payload = _b64d(parts[1])
            signature = _b64d(parts[2])
        except (ValueError, TypeError):
            return None
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            data = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict) or int(data.get("g", 0)) != self._generation:
            return None
        expires = float(data.get("e", 0))
        if expires and self._clock() > expires:
            return None
        return Device(
            name=str(data.get("n", "")),
            id=str(data.get("i", "")),
            issued_at=float(data.get("t", 0)),
            expires_at=expires,
        )

    def revoke_all(self) -> None:
        """Invalidate every cookie already handed out, without changing the secret."""
        self._generation += 1
        self._log.info("Every paired device was revoked.")


def _clean_device_name(name: Any) -> str:
    """One short line of ASCII-ish text: this ends up in the log next to every turn."""
    text = " ".join(str(name or "").split())
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:40] or "an unnamed phone"


# ----------------------------------------------------------------------------------
# The wire format
# ----------------------------------------------------------------------------------
def decode_pcm16(payload: bytes, *, max_samples: int | None = None) -> np.ndarray:
    """Turn one WebSocket binary frame into the float32 mono array Whisper expects.

    The page sends what ``@ricky0123/vad-web`` handed it — a Float32Array of 16 kHz
    mono — converted sample by sample to little-endian Int16. Undoing that is one
    ``frombuffer`` and one division, with no codec and no ffmpeg anywhere.

    Raises :class:`ValueError` for an empty frame, an odd byte count (which means the
    sender is not producing Int16 at all) and anything longer than ``max_samples``.
    """
    if payload is None or len(payload) == 0:
        raise ValueError("empty audio frame")
    if len(payload) % 2:
        raise ValueError(f"{len(payload)} bytes is not a whole number of Int16 samples")
    if max_samples is not None and len(payload) // 2 > int(max_samples):
        raise ValueError(
            f"{len(payload) // 2} samples is longer than the {int(max_samples)}-sample limit"
        )
    # astype() copies, so the result is writable even though frombuffer's view is not.
    return np.frombuffer(bytes(payload), dtype="<i2").astype(np.float32) / 32768.0


# ----------------------------------------------------------------------------------
# One remote turn
# ----------------------------------------------------------------------------------
@dataclass
class RemoteTurn:
    """One utterance from the phone, queued for the assistant's worker thread.

    This is the object the server hands to ``Assistant.submit_remote_turn``. It
    carries its own callbacks so the worker thread never has to know what a WebSocket
    is: it transcribes, calls ``Brain.turn(text, on_sentence=turn.on_sentence)`` and
    reports back through :meth:`finish`.
    """

    audio: np.ndarray | None
    sample_rate: int = 16000
    device: str = "phone"
    #: A typed turn: no audio, nothing to transcribe. For the bus, or a quiet room.
    text: str = ""
    on_sentence: Callable[[str], None] = lambda sentence: None
    on_transcript: Callable[[str], None] | None = None
    on_done: Callable[[str, str], None] | None = None
    #: Each tool the brain ran during this turn, so the phone can draw a card for it.
    on_tool: Callable[[dict], None] | None = None
    #: Wraps the brain's dispatcher for the duration of this turn (see RemoteGuard).
    wrap_dispatcher: Callable[[Any], Any] | None = None
    #: False keeps the reply on the phone only; the desk stays quiet.
    speak_locally: bool = False
    created_at: float = field(default_factory=time.monotonic)

    @property
    def duration_s(self) -> float:
        rate = float(self.sample_rate or 16000)
        return float(getattr(self.audio, "size", 0) or 0) / rate if rate > 0 else 0.0

    def send(self, sentence: str) -> None:
        """Stream one finished sentence to the phone. Never raises into the worker."""
        _safely(self.on_sentence, sentence)

    def tool_event(self, event: dict) -> None:
        """Report one tool call. Never raises into the worker."""
        if self.on_tool is None:
            return
        try:
            self.on_tool(event)
        except Exception:  # noqa: BLE001
            _log.debug("Remote tool callback failed.", exc_info=True)

    def transcribed(self, text: str) -> None:
        """Report the recognised text. Never raises into the worker thread."""
        _safely(self.on_transcript, text)

    def finish(self, reply: str = "", error: str = "") -> None:
        """Report the end of the turn — exactly once, success or failure."""
        callback = self.on_done
        if callback is None:
            return
        try:
            callback(reply, error)
        except Exception:  # noqa: BLE001 - the worker thread must survive a dead socket
            _log.debug("Remote turn completion callback failed.", exc_info=True)


def _safely(callback: Callable[[str], None] | None, text: str) -> None:
    if callback is None:
        return
    try:
        callback(text)
    except Exception:  # noqa: BLE001
        _log.debug("Remote turn callback failed.", exc_info=True)


# ----------------------------------------------------------------------------------
# Telling the phone what happened
# ----------------------------------------------------------------------------------
class ToolReporter:
    """Dispatcher wrapper that reports every call, so the phone can draw a card.

    Sits *outside* the guard: a refusal is reported too, which is exactly what the
    user needs to see when something was not allowed from the phone.
    """

    #: Only these keys of a result's ``data`` travel to the phone. Everything else
    #: - file paths, raw PowerShell output - stays in the log where it belongs.
    SAFE_DATA_KEYS = frozenset({
        # places and calls
        "name", "phone", "spoken_phone", "website", "address", "opening_hours", "places",
        # timers and reminders (``due`` is an epoch the phone counts down from)
        "minutes", "label", "due", "when", "text", "id", "kind", "jobs", "cancelled",
        # the clock
        "time", "weekday", "date",
        # weather
        "city", "country", "temperature_c", "apparent_c", "condition", "high_c", "low_c",
        "wind_ms", "humidity_pct",
        # the machine
        "cpu_percent", "ram_used_gb", "ram_total_gb", "disk_free_gb", "uptime_s",
        "gpu_percent", "gpu_temperature_c", "vram_used_mb", "vram_total_mb",
        # apps, search and the web
        "app", "count", "results", "query", "source", "title", "url",
    })

    def __init__(self, inner: Any, report: Callable[[dict], None]) -> None:
        self._inner = inner
        self._report = report

    def execute(self, name: str, args: dict) -> Any:
        started = time.monotonic()
        result = self._inner.execute(name, args)
        data = getattr(result, "data", None)
        clean = None
        if isinstance(data, dict):
            clean = {k: v for k, v in data.items()
                     if k in self.SAFE_DATA_KEYS and isinstance(v, (str, int, float, bool, list))}
        try:
            self._report({
                "type": "tool",
                "name": str(name),
                "ok": bool(getattr(result, "ok", False)),
                "refused": bool(getattr(result, "refused", False)),
                "summary": str(getattr(result, "summary", "") or ""),
                "data": clean,
                "ms": round((time.monotonic() - started) * 1000),
            })
        except Exception:  # noqa: BLE001 - reporting must never break the turn
            _log.debug("Tool report failed.", exc_info=True)
        return result

    def execute_many(self, calls: Any) -> list[Any]:
        results = []
        for call in calls or []:
            name = getattr(call, "name", None)
            args = getattr(call, "arguments", None)
            if name is None and isinstance(call, dict):
                name, args = call.get("name", ""), call.get("arguments", {})
            results.append(self.execute(str(name or ""), args or {}))
        return results

    def tools_payload(self) -> list[dict]:
        return self._inner.tools_payload()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


# ----------------------------------------------------------------------------------
# Guarding what the phone may run
# ----------------------------------------------------------------------------------
class RemoteGuard:
    """Dispatcher wrapper that refuses GUARDED tools for a turn from the phone.

    The tier decision is delegated to ``jarvis.tools.safety.remote_tier(spec, cfg)``
    when that function exists — the hook the integrator is expected to add. Its
    contract, as called here:

        ``remote_tier(spec: ToolSpec, cfg: Config) -> Tier | None``
        — the tier this tool runs at when the caller is remote, or ``None`` to refuse
        it outright.

    Until it exists, :func:`jarvis.tools.safety.effective_tier` is used and anything
    GUARDED is refused unless ``remote.allow_guarded`` is true. Either way the
    blocklist is untouched: it still runs inside the real dispatcher.
    """

    def __init__(
        self,
        inner: Any,
        cfg: Any,
        *,
        allow_guarded: bool = False,
        device: str = "phone",
        logger: logging.Logger | None = None,
    ) -> None:
        self._inner = inner
        self.cfg = cfg
        self.allow_guarded = bool(allow_guarded)
        self.device = device or "phone"
        self._log = logger or _log

    # --- tiers ---------------------------------------------------------------------
    def tier_for(self, spec: Any) -> Tier | None:
        """The tier ``spec`` runs at remotely; ``None`` means refuse it."""
        hook = getattr(safety, "remote_tier", None)
        if callable(hook):
            try:
                return hook(spec, self.cfg)
            except Exception as exc:  # noqa: BLE001 - a broken hook must not open the gate
                self._log.warning("safety.remote_tier failed (%s); falling back.", exc)
        try:
            return safety.effective_tier(spec, self.cfg)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("effective_tier failed for %s (%s); refusing.", spec, exc)
            return None

    def _refused(self, spec: Any) -> bool:
        tier = self.tier_for(spec)
        if tier is None:
            return True
        return tier is Tier.GUARDED and not self.allow_guarded

    # --- dispatcher surface ---------------------------------------------------------
    def execute(self, name: str, args: dict) -> Any:
        spec = registry.get(name.strip() if isinstance(name, str) else str(name))
        if spec is not None and self._refused(spec):
            log_refusal(
                f"{name} is not available over the phone",
                f"Device {self.device!r} asked for {name} with {args!r}.",
            )
            return ToolResult.refuse(
                REMOTE_REFUSAL,
                detail=f"{name} needs a spoken confirmation at the desk; "
                "set remote.allow_guarded to change that.",
            )
        return self._inner.execute(name, args)

    def execute_many(self, calls: Any) -> list[Any]:
        """Mirror of :meth:`execute` for dispatchers that batch calls."""
        results = []
        for call in calls or []:
            name = getattr(call, "name", None)
            args = getattr(call, "arguments", None)
            if name is None and isinstance(call, dict):
                name = call.get("name", "")
                args = call.get("arguments", {})
            results.append(self.execute(str(name or ""), args or {}))
        return results

    def tools_payload(self) -> list[dict]:
        """The tool list for the model, with everything it may not run removed.

        Hiding a refused tool is better than refusing it after the fact: the model
        never proposes it, so the phone never hears "I won't do that" for something
        it could not have known about.
        """
        payload = self._inner.tools_payload()
        if self.allow_guarded:
            return payload
        allowed = []
        for entry in payload or []:
            name = str((entry.get("function") or {}).get("name", "")) if isinstance(entry, dict) else ""
            spec = registry.get(name)
            if spec is not None and self._refused(spec):
                continue
            allowed.append(entry)
        return allowed

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


# ----------------------------------------------------------------------------------
# Routines: a fixed allowlist of macros the phone may run
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Routine:
    """A named macro from ``config.yaml``: a fixed sequence of tool calls."""

    name: str
    calls: tuple[tuple[str, dict], ...] = ()
    say: str = ""

    def as_json(self) -> dict:
        """What the phone's routines strip needs to draw a button."""
        return {"name": self.name, "tools": [name for name, _ in self.calls]}


def load_routines(cfg: Any) -> list[Routine]:
    """Read ``remote.routines`` from the config, ignoring anything malformed.

    A routine that cannot be parsed is logged and dropped rather than raising: a typo
    in the config must not stop the phone from working for everything else.
    """
    raw = cfg.get("remote.routines", []) if hasattr(cfg, "get") else []
    routines: list[Routine] = []
    for index, entry in enumerate(raw or []):
        if not isinstance(entry, dict):
            _log.warning("remote.routines[%d] is not a mapping; ignored.", index)
            continue
        name = " ".join(str(entry.get("name", "")).split())[:40]
        calls: list[tuple[str, dict]] = []
        for call in entry.get("calls", []) or []:
            if isinstance(call, str):
                calls.append((call, {}))
                continue
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool", "")).strip()
            args = call.get("args", {})
            if tool_name:
                calls.append((tool_name, args if isinstance(args, dict) else {}))
        if not name or not calls:
            _log.warning("remote.routines[%d] has no name or no calls; ignored.", index)
            continue
        routines.append(
            Routine(name=name, calls=tuple(calls), say=" ".join(str(entry.get("say", "")).split()))
        )
    return routines
