"""The tool registry: registration, the Ollama payload and module discovery.

The registry is global module state, so every test here snapshots it and puts it
back afterwards — the suite must pass in any order and twice in a row.

Note: no ``*_tools`` module ships in ``jarvis/tools/`` yet, so the invariant tests
inject a probe tool module through the package's ``__path__``. They iterate over the
whole registry, so they will cover the real tools the moment those land.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

import jarvis.tools as tools_package
from jarvis.tools import registry
from jarvis.tools.base import Tier, ToolContext, ToolResult, ToolSpec

OBJECT_SCHEMA = {"type": "object", "properties": {}}


@pytest.fixture(autouse=True)
def clean_registry():
    """Restore the global registry and the import bookkeeping after every test."""
    saved_specs = dict(registry.REGISTRY)
    saved_loaded = set(registry._loaded_modules)
    saved_failed = dict(registry._failed_modules)
    saved_modules = set(sys.modules)
    # Start from a blank registry. Without this, a test that registers a probe tool
    # under a real tool's name (lock_pc, weather, ...) collides with whatever another
    # test file happened to load first, and the failure looks like a registry bug
    # rather than what it is - test pollution.
    registry.REGISTRY.clear()
    registry._loaded_modules.clear()
    registry._failed_modules.clear()
    try:
        yield
    finally:
        registry.REGISTRY.clear()
        registry.REGISTRY.update(saved_specs)
        registry._loaded_modules.clear()
        registry._loaded_modules.update(saved_loaded)
        registry._failed_modules.clear()
        registry._failed_modules.update(saved_failed)
        for name in set(sys.modules) - saved_modules:
            if name.startswith("jarvis.tools."):
                del sys.modules[name]


@pytest.fixture
def tool_dir(tmp_path, monkeypatch):
    """A throwaway directory that ``load_all()`` will search for tool modules."""
    package_dir = tmp_path / "probe_tools"
    package_dir.mkdir()
    monkeypatch.setattr(
        tools_package, "__path__", [*tools_package.__path__, str(package_dir)]
    )

    def write(filename: str, source: str) -> None:
        (package_dir / filename).write_text(source, encoding="utf-8")
        importlib.invalidate_caches()

    write.path = package_dir  # type: ignore[attr-defined]
    return write


def a_tool(name: str = "probe", summary: str = "Probe reporting.") -> str:
    """Source for a tool module that registers one working tool."""
    return (
        "from jarvis.tools.registry import tool\n"
        "from jarvis.tools.base import Tier, ToolResult\n"
        f"@tool({name!r}, 'Probe the system.', {{'type': 'object', 'properties': {{}}}},"
        " Tier.SAFE)\n"
        f"def _run(ctx, args):\n    return ToolResult(ok=True, summary={summary!r})\n"
    )


def noop(ctx: ToolContext, args: dict) -> ToolResult:
    return ToolResult(ok=True, summary="Done, sir.")


# ======================================================================================
# The @tool decorator
# ======================================================================================


def test_tool_registers_a_spec_the_dispatcher_can_find():
    registry.tool("get_time_date", "Tell the time.", OBJECT_SCHEMA, Tier.SAFE)(noop)

    spec = registry.get("get_time_date")
    assert isinstance(spec, ToolSpec)
    assert spec.name == "get_time_date"
    assert spec.description == "Tell the time."
    assert spec.tier is Tier.SAFE
    assert spec.func is noop


def test_tool_returns_the_function_unchanged_so_it_stays_callable():
    decorated = registry.tool("probe", "Probe.", OBJECT_SCHEMA)(noop)
    assert decorated is noop


def test_a_duplicate_tool_name_is_rejected():
    registry.tool("probe", "The first one.", OBJECT_SCHEMA)(noop)

    with pytest.raises(ValueError, match="Duplicate tool name"):
        registry.tool("probe", "The second one.", OBJECT_SCHEMA)(noop)

    assert registry.get("probe").description == "The first one.", "the impostor won"
    assert len(registry.all_specs()) == 1


MALFORMED_SCHEMAS = [
    pytest.param("not a dict at all", id="a string instead of a schema"),
    pytest.param(None, id="None"),
    pytest.param([], id="a list"),
    pytest.param({}, id="no type"),
    pytest.param({"type": "array", "items": {}}, id="type is not object"),
    pytest.param({"type": "object"}, id="no properties"),
    pytest.param({"type": "object", "properties": []}, id="properties is a list"),
    pytest.param(
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"},
        id="required is a bare string",
    ),
    pytest.param(
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]},
        id="required names an unknown property",
    ),
]


@pytest.mark.parametrize("parameters", MALFORMED_SCHEMAS)
def test_a_malformed_parameter_schema_is_rejected(parameters):
    with pytest.raises(ValueError):
        registry.tool("probe", "Probe.", parameters)(noop)
    assert registry.get("probe") is None, "a tool with a broken schema got registered"


def test_a_malformed_schema_is_rejected_before_the_function_is_decorated():
    """The module fails at import time, not on the first call from the model."""
    with pytest.raises(ValueError):
        registry.tool("probe", "Probe.", {"type": "array", "properties": {}})


def test_a_valid_schema_with_required_arguments_is_accepted():
    schema = {
        "type": "object",
        "properties": {"minutes": {"type": "number"}, "label": {"type": "string"}},
        "required": ["minutes"],
    }
    registry.tool("set_timer", "Start a timer.", schema, Tier.SAFE)(noop)
    assert registry.get("set_timer").parameters["required"] == ["minutes"]


@pytest.mark.parametrize("name", ["", "   ", None, 7])
def test_a_tool_needs_a_non_empty_string_name(name):
    with pytest.raises(ValueError, match="non-empty string"):
        registry.tool(name, "Probe.", OBJECT_SCHEMA)(noop)


@pytest.mark.parametrize("description", ["", "   ", None])
def test_a_tool_needs_a_non_empty_description(description):
    with pytest.raises(ValueError, match="description"):
        registry.tool("probe", description, OBJECT_SCHEMA)(noop)


def test_a_non_callable_cannot_be_registered_as_a_tool():
    with pytest.raises(TypeError):
        registry.tool("probe", "Probe.", OBJECT_SCHEMA)("not a function")


def test_a_tier_given_as_a_string_is_coerced():
    registry.tool("run_powershell", "Run PowerShell.", OBJECT_SCHEMA, "guarded")(noop)
    assert registry.get("run_powershell").tier is Tier.GUARDED


def test_an_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        registry.tool("probe", "Probe.", OBJECT_SCHEMA, "mostly harmless")


def test_get_returns_none_for_a_tool_the_model_invented():
    assert registry.get("teleport") is None
    assert registry.get("") is None
    assert registry.get(None) is None  # type: ignore[arg-type]
    assert registry.get(123) is None  # type: ignore[arg-type]


def test_get_tolerates_whitespace_around_the_name():
    registry.tool("probe", "Probe.", OBJECT_SCHEMA)(noop)
    assert registry.get("  probe  ") is not None


def test_all_specs_is_sorted_by_name_for_a_stable_prompt():
    for name in ("weather", "get_time_date", "screenshot"):
        registry.tool(name, f"Tool {name}.", OBJECT_SCHEMA)(noop)
    assert [spec.name for spec in registry.all_specs()] == [
        "get_time_date",
        "screenshot",
        "weather",
    ]


# ======================================================================================
# ollama_tools()
# ======================================================================================


def register_a_realistic_set() -> None:
    registry.tool("get_time_date", "Tell the current time and date.", OBJECT_SCHEMA)(noop)
    registry.tool(
        "set_timer",
        "Start a countdown timer.",
        {
            "type": "object",
            "properties": {
                "minutes": {"type": "number", "description": "How long, in minutes."},
                "label": {"type": "string", "description": "What the timer is för."},
            },
            "required": ["minutes"],
        },
        Tier.SAFE,
    )(noop)
    registry.tool(
        "run_powershell",
        "Run a PowerShell command.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        Tier.GUARDED,
        announce="About to run PowerShell: {command}",
    )(noop)


def test_ollama_tools_has_the_exact_shape_ollama_expects():
    register_a_realistic_set()

    payload = registry.ollama_tools()
    assert len(payload) == len(registry.REGISTRY)
    for entry in payload:
        assert set(entry) == {"type", "function"}
        assert entry["type"] == "function"
        function = entry["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert isinstance(function["name"], str) and function["name"]
        assert isinstance(function["description"], str) and function["description"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        for prop in parameters["properties"].values():
            assert isinstance(prop, dict) and "type" in prop
        for name in parameters.get("required", []):
            assert name in parameters["properties"]


def test_ollama_tools_carries_no_tier_or_announcement_to_the_model():
    register_a_realistic_set()
    dumped = json.dumps(registry.ollama_tools())
    assert "guarded" not in dumped
    assert "About to run PowerShell" not in dumped


def test_the_ollama_payload_is_json_serialisable_including_umlauts():
    register_a_realistic_set()
    dumped = json.dumps(registry.ollama_tools(), ensure_ascii=False)
    assert "för" in dumped


def test_mutating_the_payload_does_not_corrupt_the_registry():
    register_a_realistic_set()
    payload = registry.ollama_tools()
    payload[0]["function"]["parameters"]["properties"]["injected"] = {"type": "string"}

    assert "injected" not in registry.get("get_time_date").parameters["properties"]
    assert "injected" not in json.dumps(registry.ollama_tools())


def test_ollama_tools_is_empty_when_nothing_is_registered():
    registry.REGISTRY.clear()
    assert registry.ollama_tools() == []


def test_a_tool_with_no_arguments_still_gets_an_object_schema():
    registry.tool("lock_pc", "Lock the workstation.", OBJECT_SCHEMA)(noop)
    parameters = registry.ollama_tools()[0]["function"]["parameters"]
    assert parameters == {"type": "object", "properties": {}}


# ======================================================================================
# load_all()
# ======================================================================================


def test_load_all_imports_tool_modules_and_registers_their_tools(tool_dir):
    tool_dir("probe_one_tools.py", a_tool("probe_one"))
    tool_dir("probe_two_tools.py", a_tool("probe_two"))

    registry.load_all()

    assert registry.get("probe_one") is not None
    assert registry.get("probe_two") is not None
    assert "jarvis.tools.probe_one_tools" in registry.loaded_modules()


def test_load_all_ignores_modules_that_are_not_tool_modules(tool_dir):
    tool_dir("helper.py", "raise RuntimeError('this must never be imported')\n")
    tool_dir("_private_tools.py", "raise RuntimeError('this must never be imported')\n")
    tool_dir("probe_tools.py", a_tool())

    registry.load_all()

    assert registry.get("probe") is not None
    assert "jarvis.tools.helper" not in registry.loaded_modules()
    assert registry.failed_modules() == {}


def test_load_all_is_idempotent(tool_dir):
    tool_dir("probe_tools.py", a_tool())

    registry.load_all()
    first_names = sorted(registry.REGISTRY)
    first_modules = registry.loaded_modules()

    registry.load_all()
    registry.load_all()

    assert sorted(registry.REGISTRY) == first_names
    assert registry.loaded_modules() == first_modules


def test_a_module_with_a_missing_library_is_skipped_not_fatal(tool_dir):
    tool_dir("broken_tools.py", "import definitely_not_installed_library\n")
    tool_dir("working_tools.py", a_tool("still_here"))

    registry.load_all()  # must not raise

    assert registry.get("still_here") is not None, "one bad module took the others down"
    failed = registry.failed_modules()
    assert "jarvis.tools.broken_tools" in failed
    assert "ModuleNotFoundError" in failed["jarvis.tools.broken_tools"]


def test_a_module_that_explodes_at_import_is_skipped(tool_dir):
    tool_dir("angry_tools.py", "raise RuntimeError('no COM on Linux')\n")
    tool_dir("working_tools.py", a_tool("still_here"))

    registry.load_all()

    assert registry.get("still_here") is not None
    assert "RuntimeError" in registry.failed_modules()["jarvis.tools.angry_tools"]


def test_a_module_registering_a_duplicate_name_is_skipped_not_fatal(tool_dir):
    registry.tool("probe", "The original.", OBJECT_SCHEMA)(noop)
    tool_dir("clashing_tools.py", a_tool("probe"))

    registry.load_all()

    assert registry.get("probe").description == "The original."
    assert "jarvis.tools.clashing_tools" in registry.failed_modules()


def test_a_failed_module_is_not_retried_on_the_next_call(tool_dir, caplog):
    """A broken optional tool must not take the assistant down, nor be retried.

    The real tool modules load alongside it, so the assertion is about the broken
    one specifically: it lands in failed_modules, never in loaded_modules, and the
    second load_all() does not try it again (one warning, not two).
    """
    tool_dir("broken_tools.py", "import definitely_not_installed_library\n")

    with caplog.at_level("WARNING", logger="jarvis.tools.registry"):
        registry.load_all()
        registry.load_all()

    assert "jarvis.tools.broken_tools" in registry.failed_modules()
    assert "jarvis.tools.broken_tools" not in registry.loaded_modules()

    attempts = [r for r in caplog.records if "broken_tools" in r.getMessage()]
    assert len(attempts) == 1, f"the broken module was imported {len(attempts)} times"


def test_load_all_survives_a_package_without_a_search_path(monkeypatch):
    monkeypatch.setattr(tools_package, "__path__", [])
    before = set(registry.loaded_modules())
    registry.load_all()  # logs and returns rather than raising
    assert set(registry.loaded_modules()) == before, "nothing is discoverable without a path"


# ======================================================================================
# Invariants every registered tool must satisfy
# ======================================================================================


def test_every_registered_tool_has_a_description_and_a_valid_tier(tool_dir):
    tool_dir("probe_tools.py", a_tool())
    registry.load_all()

    specs = registry.all_specs()
    assert specs, "nothing registered — this test would be vacuous"
    for spec in specs:
        assert isinstance(spec.description, str) and spec.description.strip(), spec.name
        assert spec.description == spec.description.strip()
        assert isinstance(spec.tier, Tier), f"{spec.name} has tier {spec.tier!r}"
        assert spec.tier in (Tier.SAFE, Tier.ANNOUNCED, Tier.GUARDED)
        assert callable(spec.func), spec.name


def test_every_registered_tool_has_a_usable_json_schema(tool_dir):
    tool_dir("probe_tools.py", a_tool())
    registry.load_all()

    specs = registry.all_specs()
    assert specs, "nothing registered — this test would be vacuous"
    for spec in specs:
        assert spec.parameters["type"] == "object", spec.name
        assert isinstance(spec.parameters["properties"], dict), spec.name
        json.dumps(spec.to_ollama())  # raises TypeError on anything unserialisable


def test_a_guarded_tool_announcement_never_leaves_a_hole_in_the_sentence():
    registry.tool(
        "run_powershell",
        "Run a PowerShell command.",
        {"type": "object", "properties": {"command": {"type": "string"}}},
        Tier.GUARDED,
        announce="About to run PowerShell: {command}",
    )(noop)
    spec = registry.get("run_powershell")

    assert spec.render_announcement({"command": "Get-Process"}) == (
        "About to run PowerShell: Get-Process"
    )
    assert spec.render_announcement({}) == "About to run run_powershell."
    assert spec.render_announcement({"command": "   "}) == "About to run run_powershell."
