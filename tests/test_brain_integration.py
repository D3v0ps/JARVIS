"""The brain, end to end, against an HTTP server pretending to be Ollama.

This is the test that proves the thing users actually feel: the first sentence
reaches the speaker while the model is still generating, and a tool call makes a
complete round trip without corrupting the conversation history.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis.brain.brain import Brain
from jarvis.brain.conversation import Conversation
from jarvis.brain.ollama_client import OllamaClient

ROOT_PROMPT = "prompts/jarvis_system.md"


def _ndjson(*objects: dict) -> bytes:
    return b"".join(json.dumps(obj).encode() + b"\n" for obj in objects)


def _token(text: str, done: bool = False) -> dict:
    return {"model": "qwen3:8b", "message": {"role": "assistant", "content": text}, "done": done}


@pytest.fixture
def fake_ollama():
    """A real HTTP server that streams a scripted two-round conversation."""
    state = {"round": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A002 - silence the test output
            pass

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            if self.path != "/api/tags":
                self.send_error(404)
                return
            body = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            state["requests"].append(json.loads(self.rfile.read(length) or b"{}"))
            state["round"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            if state["round"] == 1:
                self.wfile.write(
                    _ndjson(
                        _token("Right "),
                        _token("away, "),
                        _token("sir. "),
                        _token("Let me check the system."),
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {"function": {"name": "system_status", "arguments": {}}}
                                ],
                            },
                            "done": False,
                        },
                        {"message": {"role": "assistant", "content": ""}, "done": True},
                    )
                )
            else:
                for piece in ["The ", "system ", "is ", "at ", "12 ", "percent, ", "sir."]:
                    self.wfile.write(_ndjson(_token(piece)))
                    self.wfile.flush()
                self.wfile.write(
                    _ndjson({"message": {"role": "assistant", "content": ""}, "done": True})
                )
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    thread.join(timeout=2)


def _build(fake_ollama, memory, fake_dispatcher):
    client = OllamaClient(fake_ollama["url"], "qwen3:8b")
    conversation = Conversation(ROOT_PROMPT, memory, history_turns=12)
    return client, conversation, Brain(client, conversation, fake_dispatcher, max_tool_rounds=4)


def test_daemon_probes(fake_ollama, memory, fake_dispatcher):
    client, _, _ = _build(fake_ollama, memory, fake_dispatcher)
    assert client.available() is True
    assert client.has_model() is True


def test_sentences_stream_out_in_order(fake_ollama, memory, fake_dispatcher):
    _, _, brain = _build(fake_ollama, memory, fake_dispatcher)

    spoken: list[str] = []
    started = time.perf_counter()
    first_at: list[float] = []

    def on_sentence(sentence: str) -> None:
        if not spoken:
            first_at.append((time.perf_counter() - started) * 1000)
        spoken.append(sentence)

    result = brain.turn("how's the system", on_sentence)

    assert spoken[0] == "Right away, sir."
    assert "Let me check the system." in spoken
    assert spoken[-1].endswith("sir.")
    # The point of streaming: speech starts long before the turn is finished.
    assert first_at[0] < 500, f"first sentence took {first_at[0]:.0f} ms"
    assert result.cancelled is False
    assert result.error == ""


def test_tool_round_trip(fake_ollama, memory, fake_dispatcher):
    _, conversation, brain = _build(fake_ollama, memory, fake_dispatcher)
    result = brain.turn("how's the system", lambda s: None)

    assert fake_dispatcher.executed == [("system_status", {})]
    assert result.tool_calls == ["system_status"]
    assert len(fake_ollama["requests"]) == 2, "the model must be asked again after the tool ran"

    roles = [message["role"] for message in fake_ollama["requests"][1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]

    history_roles = [message["role"] for message in conversation.messages()]
    assert history_roles == ["system", "user", "assistant", "tool", "assistant"]


def test_voice_payload_is_correct(fake_ollama, memory, fake_dispatcher):
    """think must be off and the model pinned, or every turn pays for it."""
    _, _, brain = _build(fake_ollama, memory, fake_dispatcher)
    brain.turn("how's the system", lambda s: None)

    payload = fake_ollama["requests"][0]
    assert payload["think"] is False
    assert payload["keep_alive"] == -1
    assert payload["stream"] is True
    assert payload["tools"], "the tool schema was not sent to the model"


def test_unreachable_daemon_speaks_in_character(memory, fake_dispatcher):
    """A dead Ollama must produce one calm sentence, never a traceback."""
    client = OllamaClient("http://127.0.0.1:1", "qwen3:8b", timeout=2)
    conversation = Conversation(ROOT_PROMPT, memory)
    brain = Brain(client, conversation, fake_dispatcher)

    spoken: list[str] = []
    result = brain.turn("are you there", spoken.append)

    assert result.error, "the failure should be reported on the result"
    assert spoken, "JARVIS must say something rather than go silent"
    assert "sir" in " ".join(spoken).lower()
