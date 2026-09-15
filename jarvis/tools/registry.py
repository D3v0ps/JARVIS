"""The global tool registry.

Tool modules decorate their functions with :func:`tool`; importing the module is
what registers them. :func:`load_all` discovers and imports every
``jarvis.tools.*_tools`` module so the assistant can build its Ollama payload.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Callable

from jarvis.tools.base import Tier, ToolContext, ToolResult, ToolSpec

__all__ = [
    "REGISTRY",
    "tool",
    "get",
    "all_specs",
    "ollama_tools",
    "load_all",
    "loaded_modules",
    "failed_modules",
]

_log = logging.getLogger("jarvis.tools.registry")

#: name -> spec, populated at import time by the ``@tool`` decorator.
REGISTRY: dict[str, ToolSpec] = {}

#: Module suffix that marks a file as a tool module, e.g. ``system_tools``.
_MODULE_SUFFIX = "_tools"

_loaded_modules: set[str] = set()
_failed_modules: dict[str, str] = {}


def _validate_parameters(name: str, parameters: Any) -> dict:
    """Make sure ``parameters`` is a JSON-Schema object the model can fill in."""
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Tool {name!r}: parameters must be a JSON-Schema dict, got "
            f"{type(parameters).__name__}"
        )
    schema_type = parameters.get("type")
    if schema_type != "object":
        raise ValueError(
            f"Tool {name!r}: parameters['type'] must be 'object', got {schema_type!r}"
        )
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(
            f"Tool {name!r}: parameters['properties'] must be a dict, got "
            f"{type(properties).__name__}"
        )
    required = parameters.get("required", [])
    if not isinstance(required, (list, tuple)):
        raise ValueError(f"Tool {name!r}: parameters['required'] must be a list")
    unknown = [key for key in required if key not in properties]
    if unknown:
        raise ValueError(
            f"Tool {name!r}: required names {unknown} are not in properties"
        )
    return parameters


def tool(
    name: str,
    description: str,
    parameters: dict,
    tier: Tier = Tier.SAFE,
    announce: str | None = None,
) -> Callable[[Callable[[ToolContext, dict], ToolResult]], Callable[[ToolContext, dict], ToolResult]]:
    """Register a tool implementation and return it unchanged.

    The decorated function keeps its plain ``(ctx, args) -> ToolResult`` signature so
    it stays directly callable from tests. Registration happens at import time and a
    duplicate name is a programming error, so it raises instead of silently winning.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool name must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Tool {name!r}: description must be a non-empty string")
    schema = _validate_parameters(name, parameters)
    tier_value = tier if isinstance(tier, Tier) else Tier(str(tier))

    def decorator(
        func: Callable[[ToolContext, dict], ToolResult],
    ) -> Callable[[ToolContext, dict], ToolResult]:
        if not callable(func):
            raise TypeError(f"Tool {name!r}: the decorated object is not callable")
        existing = REGISTRY.get(name)
        if existing is not None:
            raise ValueError(
                f"Duplicate tool name {name!r}: already registered by "
                f"{getattr(existing.func, '__module__', '?')}."
                f"{getattr(existing.func, '__qualname__', '?')}"
            )
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description.strip(),
            parameters=schema,
            tier=tier_value,
            func=func,
            announce=announce,
        )
        _log.debug("Registered tool %s (tier=%s)", name, tier_value.value)
        return func

    return decorator


def get(name: str) -> ToolSpec | None:
    """Look a tool up by name; ``None`` when the model invented one."""
    if not isinstance(name, str):
        return None
    return REGISTRY.get(name.strip())


def all_specs() -> list[ToolSpec]:
    """Every registered spec, sorted by name for a stable prompt and stable tests."""
    return sorted(REGISTRY.values(), key=lambda spec: spec.name)


def ollama_tools() -> list[dict]:
    """The full ``tools`` payload for ``POST /api/chat``."""
    return [spec.to_ollama() for spec in all_specs()]


def load_all() -> None:
    """Import every ``jarvis.tools.*_tools`` module exactly once.

    Modules are discovered with :mod:`pkgutil` rather than a hardcoded list, so a new
    tool module only has to exist. A module that fails to import (a missing optional
    dependency, a syntax error in an experimental tool) is logged and skipped: one
    broken optional tool must never take the assistant down. Calling this twice is a
    no-op.
    """
    package = importlib.import_module(__package__ or "jarvis.tools")
    search_paths = list(getattr(package, "__path__", []))
    if not search_paths:
        _log.error("Tool package %s has no __path__; no tools loaded", package.__name__)
        return
    for module_info in sorted(
        pkgutil.iter_modules(search_paths), key=lambda info: info.name
    ):
        short_name = module_info.name
        if not short_name.endswith(_MODULE_SUFFIX) or short_name.startswith("_"):
            continue
        dotted = f"{package.__name__}.{short_name}"
        if dotted in _loaded_modules:
            continue
        if dotted in _failed_modules:
            _log.debug("Skipping tool module %s, it failed to import earlier", dotted)
            continue
        before = len(REGISTRY)
        try:
            importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001 - one bad tool must not kill JARVIS
            _failed_modules[dotted] = f"{type(exc).__name__}: {exc}"
            _log.warning("Tool module %s could not be imported: %s", dotted, exc)
            _log.debug("Import traceback for %s", dotted, exc_info=True)
            continue
        _loaded_modules.add(dotted)
        _log.debug(
            "Loaded tool module %s (%d tool(s))", dotted, len(REGISTRY) - before
        )
    _log.info(
        "Tool registry ready: %d tool(s) from %d module(s), %d module(s) skipped",
        len(REGISTRY),
        len(_loaded_modules),
        len(_failed_modules),
    )


def loaded_modules() -> list[str]:
    """Dotted names of the tool modules imported by :func:`load_all`."""
    return sorted(_loaded_modules)


def failed_modules() -> dict[str, str]:
    """Dotted name -> error string for tool modules that refused to import."""
    return dict(_failed_modules)
