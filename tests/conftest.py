"""Shared test fixtures.

The whole suite runs on any machine: no microphone, no GPU, no network, no
Ollama. Anything that would touch hardware is stubbed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def config(tmp_path):
    """A Config backed by a throwaway copy of the shipped config.yaml."""
    from jarvis.config import Config

    target = tmp_path / "config.yaml"
    target.write_text((ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = Config.load(target)
    cfg.set("logging.file", str(tmp_path / "jarvis.log"))
    return cfg


@pytest.fixture
def memory(tmp_path):
    from jarvis.core.memory import Memory

    store = Memory(tmp_path / "memory.json")
    store.load()
    return store


class RecordingResult:
    """Stands in for ToolResult without importing the tools package."""

    def __init__(self, summary: str, ok: bool = True, refused: bool = False) -> None:
        self.ok = ok
        self.summary = summary
        self.detail = ""
        self.refused = refused
        self.data = None


@pytest.fixture
def fake_dispatcher():
    """A dispatcher that records calls and always succeeds."""

    class FakeDispatcher:
        def __init__(self) -> None:
            self.executed: list[tuple[str, dict]] = []
            self.summary = "The system is at 12 percent CPU and 41 degrees."

        def execute(self, name: str, args: dict) -> RecordingResult:
            self.executed.append((name, args))
            return RecordingResult(self.summary)

        def tools_payload(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "system_status",
                        "description": "Report CPU, RAM and GPU.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

    return FakeDispatcher()
