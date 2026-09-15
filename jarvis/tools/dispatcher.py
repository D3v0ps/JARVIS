"""The one place where a tool is actually allowed to run.

Every tool call goes through :meth:`Dispatcher.execute` in a fixed order: unknown
tool -> a calm failure; argument coercion (the model's loose JSON becomes the
schema's arguments); the blocklist, unconditionally, for every tier including SAFE;
the effective tier (GUARDED confirms, ANNOUNCED announces); timed execution from
which no exception escapes; and :func:`log_tool_call`, always, for refusals and
failures too. Pure standard library, so it imports on Linux CI as well as Windows.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Iterable, Sequence

from jarvis.core.logging import get_logger, log_refusal, log_tool_call
from jarvis.tools import registry
from jarvis.tools.base import Tier, ToolContext, ToolResult, ToolSpec
from jarvis.tools.safety import check_blocked, effective_tier

__all__ = ["Dispatcher", "REFUSAL_SUMMARY", "CANCELLED_SUMMARY"]

#: Spoken when the blocklist stops a call. Deliberately the same for every pattern:
#: JARVIS declines, he does not explain how to get around him.
REFUSAL_SUMMARY = "I'm afraid that's beyond what I'm willing to do, sir."

#: Spoken when the user answers "no" to a guarded confirmation.
CANCELLED_SUMMARY = "Very well, sir. Cancelled."

#: How deep the blocklist walks into nested arguments when collecting strings.
_MAX_ARG_DEPTH = 6

_TRUE_WORDS = {"true", "yes", "y", "on", "1", "ja", "sant"}
_FALSE_WORDS = {"false", "no", "n", "off", "0", "nej", "falskt"}


def _convert(value: Any, target: str) -> Any:
    """Convert one argument to the JSON-Schema type ``target``.

    Raises :class:`ValueError` or :class:`TypeError` when the value cannot be made to
    fit, which the caller turns into "pass it through unchanged".
    """
    if value is None or not target:
        return value
    if target == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    if target in ("integer", "number"):
        if isinstance(value, (bool, int, float)):
            number = float(value)
        else:  # "5", "5.0", " 7 ", "3,5" — all things a small model really sends
            number = float(str(value).strip().replace(",", ".").replace(" ", ""))
        return int(round(number)) if target == "integer" else number
    if target == "boolean":
        if isinstance(value, (bool, int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if target == "array":
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if not isinstance(value, str):
            return [value]
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        return [part.strip() for part in text.split(",") if part.strip()]
    if target == "object":
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{value!r} is not an object")
    return value


class Dispatcher:
    """Runs registered tools for one :class:`ToolContext`.

    One dispatcher is built during wiring and shared for the whole session; it holds
    no per-call state, so the audio thread and text mode can both use it.
    """

    def __init__(self, ctx: ToolContext, logger: logging.Logger | None = None) -> None:
        self.ctx = ctx
        self.logger = logger or get_logger("tools.dispatcher")

    # -- public API ---------------------------------------------------------------------
    def execute(self, name: str, args: dict) -> ToolResult:
        """Execute one tool call and return its :class:`ToolResult`.

        Never raises, with one exception: a :class:`KeyboardInterrupt` from inside a
        tool is re-raised, because Ctrl+C belongs to the user and must still stop the
        assistant. Everything else — an unknown tool, unusable arguments, a blocked
        command, a refused confirmation, a crash — comes back as a result whose
        ``summary`` is one sentence JARVIS can say out loud.
        """
        started = time.perf_counter()
        tool_name = name.strip() if isinstance(name, str) else str(name)

        # 1. Unknown tool ----------------------------------------------------------------
        spec = registry.get(tool_name)
        if spec is None:
            result = ToolResult.fail(
                f"I don't have a tool called {tool_name or 'that'}, sir.",
                detail=f"Unknown tool {tool_name!r}. Known tools: "
                f"{', '.join(sorted(registry.REGISTRY)) or 'none'}.",
            )
            self.logger.warning("Model asked for unknown tool %r", tool_name)
            self._log(tool_name, self._loggable(args), result, started)
            return result

        # 2. Argument coercion -----------------------------------------------------------
        coerced, missing = self._coerce_args(spec, args)
        if missing:
            result = ToolResult.fail(
                f"I need to know {self._spoken_list(missing)}, sir.",
                detail=f"Tool {spec.name} is missing required argument(s): "
                f"{', '.join(missing)}. Received: {self._loggable(args)!r}",
            )
            self._log(spec.name, coerced, result, started)
            return result

        # 3. Blocklist — unconditional, every tier, SAFE tools included -------------------
        reason = self._blocked_reason(spec.name, coerced)
        if reason is not None:
            detail = f"Blocked tool call {spec.name}({self._loggable(coerced)!r}): {reason}."
            log_refusal(reason, detail)
            result = ToolResult.refuse(REFUSAL_SUMMARY, detail=detail)
            self._log(spec.name, coerced, result, started)
            return result

        # 4. Tier: confirm or announce ----------------------------------------------------
        tier = self._tier(spec)
        announcement = spec.render_announcement(coerced)
        if tier is Tier.GUARDED:
            if not self._confirm(announcement):
                result = ToolResult(ok=False, summary=CANCELLED_SUMMARY)
                self.logger.info("Guarded tool %s was not confirmed; cancelled", spec.name)
                self._log(spec.name, coerced, result, started)
                return result
        elif tier is Tier.ANNOUNCED:
            self._speak(announcement)

        # 5. Execute, and 6. log -----------------------------------------------------------
        result = self._invoke(spec, coerced)
        self._log(spec.name, coerced, result, started)
        return result

    def execute_many(self, calls: Iterable[Any]) -> list[ToolResult]:
        """Execute several tool calls and return their results in the same order.

        ``calls`` is whatever the model handed back: ``{"name", "arguments"}``
        mappings, the raw Ollama ``{"function": {...}}`` shape, or ``(name, args)``
        pairs. Calls are independent — one failure does not stop the rest — and a call
        whose shape makes no sense becomes a failed result rather than an exception,
        so the returned list always lines up one-to-one with the input.
        """
        results: list[ToolResult] = []
        for call in list(calls or []):
            name, args = self._unpack_call(call)
            if not name:
                self.logger.warning("Ignoring malformed tool call %r", call)
                results.append(ToolResult.fail(
                    "That tool call arrived without a name, sir.",
                    detail=f"Malformed tool call: {call!r}",
                ))
                continue
            results.append(self.execute(name, args))
        return results

    def tools_payload(self) -> list[dict]:
        """The ``tools`` array for ``POST /api/chat``, straight from the registry.

        Delegating keeps one source of truth — whatever is registered is what the
        model is offered — and each schema is deep-copied on the way out, so a caller
        cannot mutate the registry by editing the payload.
        """
        return registry.ollama_tools()

    # -- steps ---------------------------------------------------------------------------
    def _tier(self, spec: ToolSpec) -> Tier:
        """The tier this call runs at, honouring ``assistant.safety_mode``."""
        try:
            return effective_tier(spec, self.ctx.config)
        except Exception:  # noqa: BLE001 - a broken config must never lower the guard
            self.logger.warning("Could not resolve the tier for %s; using the spec's own",
                                spec.name, exc_info=True)
            return spec.tier if isinstance(spec.tier, Tier) else Tier(str(spec.tier))

    def _confirm(self, announcement: str) -> bool:
        """Ask the user out loud; anything going wrong counts as "no"."""
        try:
            return bool(self.ctx.confirm(announcement))
        except Exception:  # noqa: BLE001 - no confirmation means no action
            self.logger.error("Confirmation failed for %r; treating it as a cancellation",
                              announcement, exc_info=True)
            return False

    def _speak(self, announcement: str) -> None:
        """Announce an ANNOUNCED tool; a mute assistant still runs the tool."""
        try:
            self.ctx.speak(announcement)
        except Exception:  # noqa: BLE001 - losing a voice line must not lose the tool
            self.logger.error("Could not speak %r", announcement, exc_info=True)

    def _invoke(self, spec: ToolSpec, args: dict) -> ToolResult:
        """Call the tool function, turning anything it throws into a failure."""
        try:
            raw = spec.func(self.ctx, args)
        except KeyboardInterrupt:
            self.logger.warning("Tool %s interrupted by the user", spec.name)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad tool must not end the turn
            detail = traceback.format_exc()
            self.logger.error("Tool %s raised %s: %s", spec.name, type(exc).__name__, exc)
            self.logger.debug("Traceback for tool %s", spec.name, exc_info=True)
            return ToolResult.fail(
                f"I ran into a problem with {spec.name.replace('_', ' ')}, sir.",
                detail=detail,
            )
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, str) and raw.strip():
            self.logger.debug("Tool %s returned a bare string; wrapping it", spec.name)
            return ToolResult(ok=True, summary=raw.strip())
        self.logger.error("Tool %s returned %r instead of a ToolResult", spec.name, raw)
        return ToolResult.fail(
            f"I couldn't make sense of what {spec.name.replace('_', ' ')} gave back, sir.",
            detail=f"Expected a ToolResult, got {type(raw).__name__}: {raw!r}",
        )

    def _log(self, name: str, args: Any, result: ToolResult, started: float) -> None:
        """Write the structured tool-call line; logging never breaks a turn."""
        duration_ms = (time.perf_counter() - started) * 1000.0
        try:
            log_tool_call(name, self._loggable(args), result, duration_ms=duration_ms)
        except Exception:  # noqa: BLE001 - defensive; log_tool_call already guards itself
            self.logger.debug("log_tool_call failed for %s", name, exc_info=True)

    # -- blocklist -------------------------------------------------------------------------
    def _blocked_reason(self, name: str, args: dict) -> str | None:
        """Run the blocklist over the tool name and every string in its arguments."""
        candidates: list[str] = [name] if name else []
        self._collect_strings(args, candidates, 0)
        for text in candidates:
            try:
                reason = check_blocked(text)
            except Exception:  # noqa: BLE001 - a broken pattern must not open the gate
                self.logger.error("Blocklist check failed on %r", text, exc_info=True)
                return "an argument that could not be safety-checked"
            if reason:
                return reason
        return None

    def _collect_strings(self, value: Any, into: list[str], depth: int) -> None:
        """Depth-limited walk gathering every string hiding in the arguments."""
        if depth > _MAX_ARG_DEPTH:
            return
        if isinstance(value, str):
            if value.strip():
                into.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.strip():
                    into.append(key)
                self._collect_strings(item, into, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                self._collect_strings(item, into, depth + 1)

    # -- argument coercion -------------------------------------------------------------------
    def _coerce_args(self, spec: ToolSpec, args: Any) -> tuple[dict, list[str]]:
        """Turn whatever the model sent into arguments matching ``spec.parameters``.

        Handles the three things small models get wrong: a JSON string instead of an
        object, a bare value instead of an object, and numbers or booleans sent as
        strings. Unknown keys are dropped, schema defaults are filled in, and the
        names of any still-missing required arguments are returned so the caller can
        say so out loud instead of hitting a ``TypeError``.
        """
        schema = spec.parameters if isinstance(spec.parameters, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [key for key in schema.get("required", []) if isinstance(key, str)]

        coerced: dict[str, Any] = {}
        for key, value in self._as_mapping(args, properties).items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                self.logger.debug("Dropping unknown argument %r for tool %s", key, spec.name)
                continue
            coerced[key] = self._coerce_value(spec.name, key, value, prop)

        for key, prop in properties.items():
            if key not in coerced and isinstance(prop, dict) and "default" in prop:
                coerced[key] = prop["default"]

        missing = [
            key for key in required
            if coerced.get(key) is None
            or (isinstance(coerced[key], str) and not coerced[key].strip())
        ]
        return coerced, missing

    def _as_mapping(self, args: Any, properties: dict) -> dict:
        """Coax ``args`` into a plain dict of argument name -> value."""
        if isinstance(args, dict):
            return {str(key): value for key, value in args.items()}
        if args is None:
            return {}
        if isinstance(args, (bytes, bytearray)):
            args = args.decode("utf-8", errors="replace")
        if isinstance(args, str):
            text = args.strip()
            if not text:
                return {}
            try:  # the classic: '{"command": "Get-Date"}' sent as a string
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return self._single_value(text, properties)
            return ({str(key): value for key, value in parsed.items()}
                    if isinstance(parsed, dict) else self._as_mapping(parsed, properties))
        if isinstance(args, (list, tuple)):
            if len(args) == 1:
                return self._single_value(args[0], properties)
            # Positional arguments: map them onto the declared property order.
            return dict(zip(properties, args))
        return self._single_value(args, properties)

    def _single_value(self, value: Any, properties: dict) -> dict:
        """A bare value only makes sense for a single-argument tool."""
        names = list(properties)
        if len(names) == 1:
            return {names[0]: value}
        self.logger.debug("Ignoring bare argument %r: the tool takes %d", value, len(names))
        return {}

    def _coerce_value(self, tool_name: str, key: str, value: Any, prop: dict) -> Any:
        """Best-effort conversion of one argument to the type its schema declares."""
        declared = prop.get("type")
        if isinstance(declared, (list, tuple)):  # e.g. ["string", "null"]
            declared = next((item for item in declared
                             if isinstance(item, str) and item != "null"), "")
        target = declared if isinstance(declared, str) else ""
        try:
            value = _convert(value, target)
        except (TypeError, ValueError):
            self.logger.debug("Could not coerce %s.%s=%r to %s; passing it through",
                              tool_name, key, value, target)
            return value
        options = prop.get("enum")
        if isinstance(options, (list, tuple)) and isinstance(value, str):
            text = value.strip()
            return next((option for option in options
                         if isinstance(option, str) and option.lower() == text.lower()), text)
        return value

    # -- small helpers -----------------------------------------------------------------------
    @staticmethod
    def _spoken_list(names: Sequence[str]) -> str:
        """``["source", "destination"]`` -> ``"the source and the destination"``."""
        friendly = [f"the {str(name).replace('_', ' ').strip()}"
                    for name in names if str(name).strip()]
        if not friendly:
            return "a little more"
        if len(friendly) == 1:
            return friendly[0]
        return ", ".join(friendly[:-1]) + " and " + friendly[-1]

    @staticmethod
    def _loggable(args: Any) -> dict:
        """Arguments in a shape the structured log helper can serialise."""
        return args if isinstance(args, dict) else {"arguments": args}

    @staticmethod
    def _unpack_call(call: Any) -> tuple[str, Any]:
        """Pull ``(name, arguments)`` out of the shapes a tool call arrives in."""
        if isinstance(call, dict):
            function = call.get("function")
            if isinstance(function, dict):
                call = function
            name = call.get("name") or call.get("tool") or call.get("tool_name") or ""
            args: Any = None
            for key in ("arguments", "args", "parameters", "input"):
                if call.get(key) is not None:
                    args = call[key]
                    break
            return (str(name).strip(), {} if args is None else args)
        if isinstance(call, (list, tuple)) and len(call) == 2:
            return (str(call[0]).strip(), call[1])
        return ("", {})
