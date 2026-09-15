"""Streaming HTTP client for a local Ollama daemon.

This is the only module that talks to the language model: one POST to ``/api/chat`` with
``stream=true`` and an NDJSON response turned into :class:`ChatDelta` values as the bytes
arrive. Nothing is buffered, so the first content fragment reaches the sentence splitter —
and the speech engine — long before the model has finished its answer.

Robustness is the point here. A refused connection, a read timeout, a 500, an HTML error
page or a truncated JSON line all become exactly one ``ChatDelta(kind="error", ...)``
followed by the end of the iterator — no exception ever escapes :meth:`OllamaClient.chat_stream`,
because a traceback in the middle of a voice turn is a silently dead assistant. Older Ollama
builds reject the ``think`` field with a 400; that is retried once without it and remembered
for the rest of the process. Only the standard library plus ``requests`` is imported, so the
module loads on a machine with no model, no GPU and no Ollama installed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
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


def _parse_arguments(raw: Any, log: logging.Logger) -> dict:
    """Coerce tool-call arguments into a dict.

    Ollama sends a JSON object; some builds and OpenAI-compatible proxies send that same
    object as a JSON string. Anything else degrades to an empty dict instead of breaking
    the turn.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            log.warning("Tool-call arguments were not valid JSON, ignoring them")
            return {}
        if isinstance(parsed, dict):
            return parsed
    log.warning("Tool-call arguments were not an object, ignoring them")
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
        with _think_lock:
            send_think = not _think_unsupported
        if send_think:
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
            name = entry.get("name") or entry.get("model") if isinstance(entry, dict) else entry
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def available(self) -> bool:
        """True when the daemon answers ``/api/tags``. Never raises."""
        return self._tags() is not None

    @staticmethod
    def _matches(installed: str, wanted: str) -> bool:
        """Does an installed tag satisfy a wanted name?

        ``qwen3:8b`` is matched by ``qwen3:8b`` and by the bare family name ``qwen3``; a
        registry prefix such as ``library/`` is ignored. Two different tags of the same
        family (``qwen3:8b`` against ``qwen3:14b``) do **not** match.
        """
        left = installed.strip().lower().rsplit("/", 1)[-1]
        right = wanted.strip().lower().rsplit("/", 1)[-1]
        if not left or not right:
            return False
        if left == right:
            return True
        left_family, _, left_tag = left.partition(":")
        right_family, _, right_tag = right.partition(":")
        if left_family != right_family:
            return False
        # One side named the family only, or asked for the implicit "latest" tag.
        return not left_tag or not right_tag or "latest" in (left_tag, right_tag)

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
        response, error = self._post(payload, stream=False)
        elapsed = (time.perf_counter() - started) * 1000.0
        if response is None:
            self._log.warning("Could not warm %s after %.0f ms: %s", payload["model"], elapsed, error)
            return
        response.close()
        self._log.info("Model %s warm in %.0f ms", payload["model"], elapsed)

    # --- requests ----------------------------------------------------------------------
    def _error_text(self, response: "requests.Response") -> str:
        """The first part of an error body, never raising on an unreadable one."""
        try:
            body = (response.text or "").strip()
        except Exception:  # noqa: BLE001
            body = ""
        return body[:_ERROR_BODY_CHARS]

    def _post(self, payload: dict, *, stream: bool) -> tuple["requests.Response | None", str]:
        """POST ``/api/chat``; on HTTP 200 the response, else ``(None, one-sentence error)``.

        A 400 that mentions ``think`` is retried once without that field.
        """
        for attempt in (0, 1):
            try:
                response = self._session.post(
                    self._url("/api/chat"),
                    json=payload,
                    stream=stream,
                    timeout=(CONNECT_TIMEOUT, self.timeout),
                )
            except requests.exceptions.Timeout:
                self._log.error("Ollama timed out after %s s", self.timeout)
                return None, "The local model did not answer in time."
            except requests.exceptions.ConnectionError as exc:
                self._log.error("Cannot reach Ollama at %s: %s", self.host, exc)
                return None, f"I cannot reach the local model at {self.host}."
            except Exception as exc:  # noqa: BLE001 - no request failure may escape
                self._log.error("Request to Ollama failed: %s", exc)
                return None, "I could not send that to the local model."

            if response.status_code == 200:
                return response, ""

            status = response.status_code
            body = self._error_text(response)
            retry_without_think = (
                attempt == 0 and status == 400 and "think" in payload and "think" in body.lower()
            )
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
            if retry_without_think:
                _mark_think_unsupported(self._log)
                payload.pop("think", None)
                continue
            self._log.error("Ollama returned HTTP %s: %s", status, body)
            return None, f"The local model returned an error, status {status}."
        return None, "The local model refused the request."

    # --- streaming chat ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        think: bool | None = None,
    ) -> Iterator[ChatDelta]:
        """Stream one assistant turn as :class:`ChatDelta` events.

        A ``token`` delta per content fragment as it arrives, a ``tool_calls`` delta when
        the model asks for tools, then one ``done`` delta with the assembled assistant
        message. Any failure yields exactly one ``error`` delta and stops.
        """
        payload = self._payload(messages, tools, model=model, think=think, stream=True)
        response: "requests.Response | None" = None
        try:
            response, error = self._post(payload, stream=True)
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

    def _read_stream(self, response: "requests.Response") -> Iterator[ChatDelta]:
        """Turn NDJSON lines into deltas. Stops after the first malformed line."""
        content: list[str] = []
        collected: list[dict] = []

        def assistant_message() -> dict:
            return {"role": "assistant", "content": "".join(content), "tool_calls": list(collected)}
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
                    obj = None
                if not isinstance(obj, dict):
                    self._log.error("Malformed line from Ollama: %s", line[:_ERROR_BODY_CHARS])
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
                        content.append(fragment)
                        yield ChatDelta(kind="token", text=fragment)
                    calls = _normalize_tool_calls(message.get("tool_calls"), self._log)
                    if calls:
                        collected.extend(calls)
                        yield ChatDelta(kind="tool_calls", tool_calls=calls)

                if obj.get("done"):
                    yield ChatDelta(kind="done", message=assistant_message())
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
        yield ChatDelta(kind="done", message=assistant_message())

    # --- blocking chat --------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        think: bool | None = None,
    ) -> dict:
        """One blocking chat call, returning the assistant message dict.

        Used by ``deep_think``, which wants a whole answer rather than a stream. Returns
        ``{"role", "content", "tool_calls"}``; on failure the same shape with empty
        ``content`` plus an ``error`` key, so the caller never has to catch anything.
        """
        payload = self._payload(messages, tools, model=model, think=think, stream=False)
        response, error = self._post(payload, stream=False)
        if response is None:
            return self._failed_message(error)
        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - a bad body is an error, not a crash
            self._log.error("Could not read the answer from Ollama: %s", exc)
            return self._failed_message("The local model sent something I could not read.")
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass

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

    @staticmethod
    def _failed_message(error: str) -> dict:
        """The shape :meth:`chat` returns when the request did not work out."""
        return {"role": "assistant", "content": "", "tool_calls": [], "error": error}

    def close(self) -> None:
        """Release the pooled HTTP connections. Safe to call more than once."""
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass
