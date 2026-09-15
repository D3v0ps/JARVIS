"""Streaming HTTP client for a local Ollama daemon.

This is the only module that talks to the language model. It is deliberately thin:
one POST to ``/api/chat`` with ``stream=true`` and an NDJSON response that is turned
into :class:`ChatDelta` values as the bytes arrive. Nothing is buffered — the first
content fragment is yielded the moment it lands, which is what lets the sentence
splitter hand a first sentence to the speech engine long before the model is done.

Robustness is the point of this module. A refused connection, a read timeout, a 500,
an HTML error page or a truncated JSON line all become exactly one
``ChatDelta(kind="error", ...)`` followed by the end of the iterator. No exception ever
escapes :meth:`OllamaClient.chat_stream`, because the caller is in the middle of a voice
turn and a traceback there is a silent assistant.

Older Ollama builds reject the ``think`` field with a 400. The first time that happens
the field is dropped, the request is retried once, and the fact is remembered for the
rest of the process so no later turn pays for the retry.

Only the standard library plus ``requests`` is imported here; the module loads fine on a
machine with no model, no GPU and no Ollama installed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator, Optional

import requests

from jarvis.core.logging import get_logger

__all__ = ["ChatDelta", "OllamaClient", "DEFAULT_HOST", "DEFAULT_MODEL"]

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"

#: Timeout for the cheap "is the daemon up" probe.
TAGS_TIMEOUT = 2.0
#: Connection timeouts stay short even when the read timeout is minutes long.
CONNECT_TIMEOUT = 5.0
#: How much of an error body is worth logging.
_ERROR_BODY_CHARS = 400

# Remembered for the lifetime of the process: an Ollama build that rejects `think`.
_think_lock = Lock()
_think_unsupported = False


def _think_is_unsupported() -> bool:
    with _think_lock:
        return _think_unsupported


def _mark_think_unsupported(log: logging.Logger) -> None:
    """Remember (once, process-wide) that this daemon does not accept ``think``."""
    global _think_unsupported
    with _think_lock:
        already = _think_unsupported
        _think_unsupported = True
    if not already:
        log.info("Ollama rejected the 'think' field; dropping it for the rest of this session")


@dataclass
class ChatDelta:
    """One event out of a streaming chat call."""

    kind: str  # "token" | "tool_calls" | "done" | "error"
    text: str = ""
    tool_calls: list[dict] | None = None  # [{"name": str, "arguments": dict}]
    message: dict | None = None  # the assembled assistant message on "done"
    error: str = ""


@dataclass
class _Assembled:
    """Accumulator for the assistant message rebuilt from the stream."""

    content: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)

    def message(self) -> dict:
        return {
            "role": "assistant",
            "content": "".join(self.content),
            "tool_calls": list(self.tool_calls),
        }


def _parse_arguments(raw: Any, log: logging.Logger) -> dict:
    """Coerce tool-call arguments into a dict.

    Ollama sends a JSON object; some builds and OpenAI-compatible proxies send the
    same object as a JSON string. Anything else degrades to an empty dict rather
    than breaking the turn.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            log.warning("Tool-call arguments were not valid JSON, ignoring them")
            return {}
        if isinstance(parsed, dict):
            return parsed
        log.warning("Tool-call arguments decoded to %s, not an object", type(parsed).__name__)
        return {}
    if raw is None:
        return {}
    log.warning("Tool-call arguments had unexpected type %s", type(raw).__name__)
    return {}


def _normalize_tool_calls(raw: Any, log: logging.Logger) -> list[dict]:
    """Turn whatever the server sent into ``[{"name": str, "arguments": dict}]``."""
    if not isinstance(raw, list):
        return []
    calls: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Ollama: {"function": {"name": ..., "arguments": {...}}}; be tolerant of a flat shape.
        inner = item.get("function")
        source = inner if isinstance(inner, dict) else item
        name = str(source.get("name") or "").strip()
        if not name:
            continue
        calls.append({"name": name, "arguments": _parse_arguments(source.get("arguments"), log)})
    return calls


class OllamaClient:
    """Streaming and blocking chat against a local Ollama daemon."""

    def __init__(
        self,
        host: str,
        model: str,
        *,
        keep_alive: int | str = -1,
        think: bool = False,
        num_ctx: int = 8192,
        temperature: float = 0.6,
        timeout: int = 120,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.keep_alive = keep_alive
        self.think = bool(think)
        self.num_ctx = int(num_ctx)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self._log: logging.Logger = logger or get_logger("ollama")
        self._session = requests.Session()

    # --- payload ---------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.host}/{path.lstrip('/')}"

    def _payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str | None,
        think: bool | None,
        stream: bool,
        extra_options: dict | None = None,
    ) -> dict:
        options: dict[str, Any] = {"num_ctx": self.num_ctx, "temperature": self.temperature}
        if extra_options:
            options.update(extra_options)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages or []),
            "stream": bool(stream),
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if tools:
            payload["tools"] = list(tools)
        if not _think_is_unsupported():
            payload["think"] = self.think if think is None else bool(think)
        return payload

    # --- availability ----------------------------------------------------------------
    def _tags(self, timeout: float = TAGS_TIMEOUT) -> list[str] | None:
        """Model names installed on the daemon, or ``None`` when it cannot be reached."""
        try:
            response = self._session.get(self._url("/api/tags"), timeout=timeout)
            if response.status_code != 200:
                self._log.debug("GET /api/tags returned HTTP %s", response.status_code)
                return None
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            self._log.debug("Ollama is not reachable at %s: %s", self.host, exc)
            return None
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if name:
                    names.append(str(name))
            elif isinstance(entry, str):
                names.append(entry)
        return names

    def available(self) -> bool:
        """True when the daemon answers ``/api/tags``. Never raises."""
        return self._tags() is not None

    @staticmethod
    def _matches(installed: str, wanted: str) -> bool:
        """``qwen3`` matches ``qwen3:8b``, and a registry prefix is ignored."""
        left = installed.strip().lower()
        right = wanted.strip().lower()
        if not left or not right:
            return False
        if left == right:
            return True
        left_base = left.rsplit("/", 1)[-1]
        right_base = right.rsplit("/", 1)[-1]
        if left_base == right_base:
            return True
        return left_base.split(":", 1)[0] == right_base.split(":", 1)[0]

    def has_model(self, name: str | None = None) -> bool:
        """True when ``name`` (default: the configured model) is installed. Never raises."""
        wanted = (name or self.model or "").strip()
        if not wanted:
            return False
        installed = self._tags()
        if not installed:
            return False
        return any(self._matches(candidate, wanted) for candidate in installed)

    def warm(self) -> None:
        """Load the weights with a one-token chat so the first real answer is fast."""
        started = time.perf_counter()
        payload = self._payload(
            [{"role": "user", "content": "hi"}],
            None,
            model=None,
            think=False,
            stream=False,
            extra_options={"num_predict": 1},
        )
        try:
            response = self._session.post(
                self._url("/api/chat"), json=payload, timeout=(CONNECT_TIMEOUT, self.timeout)
            )
            if response.status_code == 400 and self._body_mentions_think(response) and "think" in payload:
                _mark_think_unsupported(self._log)
                payload.pop("think", None)
                response = self._session.post(
                    self._url("/api/chat"), json=payload, timeout=(CONNECT_TIMEOUT, self.timeout)
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            if response.status_code != 200:
                self._log.warning(
                    "Warm-up of %s failed with HTTP %s after %.0f ms",
                    payload["model"], response.status_code, elapsed,
                )
                return
            self._log.info("Model %s warm in %.0f ms", payload["model"], elapsed)
        except Exception as exc:  # noqa: BLE001 - warming is best effort
            elapsed = (time.perf_counter() - started) * 1000.0
            self._log.warning("Could not warm %s after %.0f ms: %s", payload["model"], elapsed, exc)

    # --- streaming chat ----------------------------------------------------------------
    @staticmethod
    def _body_mentions_think(response: "requests.Response") -> bool:
        try:
            return "think" in (response.text or "").lower()
        except Exception:  # noqa: BLE001 - a body we cannot read is not about think
            return False

    def _error_text(self, response: "requests.Response") -> str:
        try:
            body = (response.text or "").strip()
        except Exception:  # noqa: BLE001
            body = ""
        return body[:_ERROR_BODY_CHARS]

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        think: bool | None = None,
    ) -> Iterator[ChatDelta]:
        """Stream one assistant turn as :class:`ChatDelta` events.

        Yields a ``token`` delta per content fragment, a ``tool_calls`` delta when the
        model asks for tools, and finally one ``done`` delta carrying the assembled
        assistant message. Any failure yields exactly one ``error`` delta and stops.
        """
        payload = self._payload(messages, tools, model=model, think=think, stream=True)
        response: "requests.Response | None" = None
        try:
            response, error = self._open_stream(payload)
            if response is None:
                yield ChatDelta(kind="error", error=error)
                return
            yield from self._read_stream(response)
        except Exception as exc:  # noqa: BLE001 - nothing may escape this generator
            self._log.error("Unexpected failure while streaming from Ollama: %s", exc)
            yield ChatDelta(kind="error", error="Something went wrong while talking to the local model.")
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # noqa: BLE001 - closing must not mask the real outcome
                    pass

    def _open_stream(self, payload: dict) -> tuple["requests.Response | None", str]:
        """POST the request, retrying once without ``think`` on a 400 that mentions it."""
        for attempt in (0, 1):
            try:
                response = self._session.post(
                    self._url("/api/chat"),
                    json=payload,
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, self.timeout),
                )
            except requests.exceptions.Timeout:
                self._log.error("Ollama timed out after %s s", self.timeout)
                return None, "The local model did not answer in time."
            except requests.exceptions.ConnectionError as exc:
                self._log.error("Cannot reach Ollama at %s: %s", self.host, exc)
                return None, f"I cannot reach the local model at {self.host}."
            except Exception as exc:  # noqa: BLE001
                self._log.error("Request to Ollama failed: %s", exc)
                return None, "I could not send that to the local model."

            if response.status_code == 200:
                return response, ""

            status = response.status_code
            body = self._error_text(response)
            retry_think = (
                attempt == 0
                and status == 400
                and "think" in payload
                and ("think" in body.lower() or not body)
            )
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
            if retry_think:
                _mark_think_unsupported(self._log)
                payload.pop("think", None)
                continue
            self._log.error("Ollama returned HTTP %s: %s", status, body)
            return None, f"The local model returned an error, status {status}."
        return None, "The local model refused the request."

    def _read_stream(self, response: "requests.Response") -> Iterator[ChatDelta]:
        """Turn NDJSON lines into deltas. Stops after the first malformed line."""
        assembled = _Assembled()
        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    self._log.error("Malformed line from Ollama: %s", line[:_ERROR_BODY_CHARS])
                    yield ChatDelta(kind="error", error="The local model sent something I could not read.")
                    return
                if not isinstance(obj, dict):
                    self._log.error("Unexpected stream payload of type %s", type(obj).__name__)
                    yield ChatDelta(kind="error", error="The local model sent something I could not read.")
                    return
                server_error = obj.get("error")
                if server_error:
                    self._log.error("Ollama reported an error: %s", server_error)
                    yield ChatDelta(kind="error", error=str(server_error)[:_ERROR_BODY_CHARS])
                    return

                message = obj.get("message")
                if isinstance(message, dict):
                    fragment = message.get("content")
                    if isinstance(fragment, str) and fragment:
                        assembled.content.append(fragment)
                        yield ChatDelta(kind="token", text=fragment)
                    calls = _normalize_tool_calls(message.get("tool_calls"), self._log)
                    if calls:
                        assembled.tool_calls.extend(calls)
                        yield ChatDelta(kind="tool_calls", tool_calls=calls)

                if obj.get("done"):
                    yield ChatDelta(kind="done", message=assembled.message())
                    return
        except requests.exceptions.Timeout:
            self._log.error("Ollama stopped sending after %s s", self.timeout)
            yield ChatDelta(kind="error", error="The local model stopped answering half way through.")
            return
        except requests.exceptions.RequestException as exc:
            self._log.error("Stream from Ollama broke: %s", exc)
            yield ChatDelta(kind="error", error="The connection to the local model broke.")
            return
        # The socket closed without a done line: hand back what we collected.
        self._log.debug("Ollama stream ended without a done flag")
        yield ChatDelta(kind="done", message=assembled.message())

    # --- blocking chat ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        think: bool | None = None,
    ) -> dict:
        """One blocking chat call, returning the assistant message dict.

        Used by ``deep_think``, which wants a whole answer rather than a stream. On
        failure the same shape comes back with empty ``content`` plus an ``error`` key,
        so the caller never has to catch anything.
        """
        payload = self._payload(messages, tools, model=model, think=think, stream=False)
        for attempt in (0, 1):
            try:
                response = self._session.post(
                    self._url("/api/chat"), json=payload, timeout=(CONNECT_TIMEOUT, self.timeout)
                )
            except requests.exceptions.Timeout:
                self._log.error("Ollama timed out after %s s", self.timeout)
                return self._failed_message("The local model did not answer in time.")
            except requests.exceptions.ConnectionError as exc:
                self._log.error("Cannot reach Ollama at %s: %s", self.host, exc)
                return self._failed_message(f"I cannot reach the local model at {self.host}.")
            except Exception as exc:  # noqa: BLE001
                self._log.error("Request to Ollama failed: %s", exc)
                return self._failed_message("I could not send that to the local model.")

            if response.status_code != 200:
                status = response.status_code
                body = self._error_text(response)
                retry_think = (
                    attempt == 0
                    and status == 400
                    and "think" in payload
                    and ("think" in body.lower() or not body)
                )
                response.close()
                if retry_think:
                    _mark_think_unsupported(self._log)
                    payload.pop("think", None)
                    continue
                self._log.error("Ollama returned HTTP %s: %s", status, body)
                return self._failed_message(f"The local model returned an error, status {status}.")

            try:
                data = response.json()
            except ValueError:
                self._log.error("Ollama answered with something that is not JSON")
                return self._failed_message("The local model sent something I could not read.")
            finally:
                response.close()

            if not isinstance(data, dict):
                return self._failed_message("The local model sent something I could not read.")
            if data.get("error"):
                self._log.error("Ollama reported an error: %s", data["error"])
                return self._failed_message(str(data["error"])[:_ERROR_BODY_CHARS])
            message = data.get("message")
            if not isinstance(message, dict):
                return self._failed_message("The local model returned an empty answer.")
            content = message.get("content")
            return {
                "role": str(message.get("role") or "assistant"),
                "content": content if isinstance(content, str) else "",
                "tool_calls": _normalize_tool_calls(message.get("tool_calls"), self._log),
            }
        return self._failed_message("The local model refused the request.")

    @staticmethod
    def _failed_message(error: str) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": [], "error": error}

    def close(self) -> None:
        """Release the pooled HTTP connections. Safe to call more than once."""
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass
