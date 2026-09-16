"""When the model talks about acting instead of acting.

A local model will sometimes answer "Right away, sir. Opening Steam." and call
nothing at all, leaving the user looking at a screen where nothing happened. The
system prompt forbids it, but a prompt is not a guarantee, so the turn gets one
corrective round before it is allowed to end.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis.brain.brain import Brain, claims_without_acting
from jarvis.brain.conversation import Conversation
from jarvis.brain.ollama_client import OllamaClient

PROMPT = "prompts/jarvis_system.md"


# --- the detector ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "Right away, sir. Opening Steam.",
        "I will check the weather for you, sir.",
        "I've opened Spotify, sir.",
        "Let me delete files older than a month from your Downloads folder.",
        "Yes, sir, I checked the weather forecast for tomorrow.",
        "I'm sorry, sir, but I don't have access to real-time weather data.",
        "Jag öppnar Spotify, sir.",
        "Jag ska kolla vädret åt dig.",
        "Jag har startat Steam.",
    ],
)
def test_an_unkept_promise_is_recognised(reply):
    assert claims_without_acting(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        "Spotify is open, sir.",
        "The system is at 12 percent, sir. Nothing to trouble you.",
        "It's 21:47 on Tuesday, sir.",
        "Very well, sir. Cancelled.",
        "I'm afraid that's beyond what I'm willing to do, sir.",
        "Which city, sir?",
        "Timer set for ten minutes, sir.",
        "Done, sir. Five files, the largest a two gigabyte video.",
        "Klockan är kvart i tio, sir.",
        "Good evening, sir. All systems online.",
    ],
)
def test_an_honest_report_is_left_alone(reply):
    """A false positive costs a wasted round trip on every ordinary answer."""
    assert claims_without_acting(reply) is False


# --- the corrective round ---------------------------------------------------------------
def _ndjson(*objects: dict) -> bytes:
    return b"".join(json.dumps(obj).encode() + b"\n" for obj in objects)


@pytest.fixture
def lazy_ollama():
    """Round 1 promises and calls nothing; round 2 (the correction) calls the tool."""
    state = {"requests": [], "round": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A002
            pass

        def do_GET(self):  # noqa: N802
            body = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            state["requests"].append(payload)
            state["round"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            if state["round"] == 1:
                self.wfile.write(_ndjson(
                    {"message": {"role": "assistant", "content": "Right away, sir. Opening Steam."},
                     "done": False},
                    {"message": {"role": "assistant", "content": ""}, "done": True},
                ))
            elif state["round"] == 2:
                self.wfile.write(_ndjson(
                    {"message": {"role": "assistant", "content": "",
                                 "tool_calls": [{"function": {"name": "open_app",
                                                              "arguments": {"name": "steam"}}}]},
                     "done": False},
                    {"message": {"role": "assistant", "content": ""}, "done": True},
                ))
            else:
                self.wfile.write(_ndjson(
                    {"message": {"role": "assistant", "content": "Steam is open, sir."},
                     "done": False},
                    {"message": {"role": "assistant", "content": ""}, "done": True},
                ))
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    thread.join(timeout=2)


def test_a_promise_without_a_tool_call_triggers_one_corrective_round(
    lazy_ollama, memory, fake_dispatcher
):
    fake_dispatcher.tools_payload = lambda: [
        {"type": "function",
         "function": {"name": "open_app", "description": "Open an application.",
                      "parameters": {"type": "object",
                                     "properties": {"name": {"type": "string"}},
                                     "required": ["name"]}}}
    ]
    client = OllamaClient(lazy_ollama["url"], "qwen3:8b")
    brain = Brain(client, Conversation(PROMPT, memory), fake_dispatcher)

    spoken: list[str] = []
    result = brain.turn("open steam", spoken.append)

    assert fake_dispatcher.executed == [("open_app", {"name": "steam"})], (
        "the action must actually happen, not just be described"
    )
    assert result.tool_calls == ["open_app"]
    assert len(lazy_ollama["requests"]) == 3, "one ordinary round, one correction, one report"


def test_the_correction_is_never_stored_in_the_conversation(
    lazy_ollama, memory, fake_dispatcher
):
    """The user should never see JARVIS being told off."""
    fake_dispatcher.tools_payload = lambda: [
        {"type": "function",
         "function": {"name": "open_app", "description": "Open an application.",
                      "parameters": {"type": "object", "properties": {}, "required": []}}}
    ]
    client = OllamaClient(lazy_ollama["url"], "qwen3:8b")
    conversation = Conversation(PROMPT, memory)
    Brain(client, conversation, fake_dispatcher).turn("open steam", lambda s: None)

    stored = " ".join(str(m.get("content") or "") for m in conversation.messages())
    assert "called no tool" not in stored
    assert "nothing actually happened" not in stored


@pytest.fixture
def honest_ollama():
    """Answers plainly, with no promise to act."""
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A002
            pass

        def do_GET(self):  # noqa: N802
            body = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            state["requests"].append(json.loads(self.rfile.read(length) or b"{}"))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(_ndjson(
                {"message": {"role": "assistant", "content": "I'm here, sir."}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ))
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    thread.join(timeout=2)


def test_an_honest_answer_costs_no_extra_round(memory, fake_dispatcher, honest_ollama):
    """The safety net must not fire on ordinary conversation."""
    client = OllamaClient(honest_ollama["url"], "qwen3:8b")
    brain = Brain(client, Conversation(PROMPT, memory), fake_dispatcher)

    brain.turn("are you there", lambda s: None)

    assert len(honest_ollama["requests"]) == 1
