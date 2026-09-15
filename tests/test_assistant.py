"""The whole assistant, wired for real, against a fake Ollama.

Everything here is the production path except the model server and the audio
hardware: real wiring, real registry, real dispatcher, real tools. This is the
test that would have caught "it imports fine but the turn never completes".
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis.core.assistant import Assistant


def _ndjson(*objects: dict) -> bytes:
    return b"".join(json.dumps(obj).encode() + b"\n" for obj in objects)


def _token(text: str, done: bool = False) -> dict:
    return {"message": {"role": "assistant", "content": text}, "done": done}


@pytest.fixture
def fake_ollama():
    """Answers a plain question directly and a time question with a tool call."""
    state: dict = {"requests": []}

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
            last = payload["messages"][-1]

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            if payload.get("options", {}).get("num_predict") == 1:
                self.wfile.write(_ndjson(_token("ok", True)))
            elif last.get("role") == "tool":
                self.wfile.write(
                    _ndjson(_token("It is late, sir."),
                            {"message": {"role": "assistant", "content": ""}, "done": True})
                )
            elif "time" in (last.get("content") or ""):
                self.wfile.write(
                    _ndjson(
                        _token("One moment, sir."),
                        {"message": {"role": "assistant", "content": "",
                                     "tool_calls": [{"function": {"name": "get_time_date",
                                                                  "arguments": {}}}]},
                         "done": False},
                        {"message": {"role": "assistant", "content": ""}, "done": True},
                    )
                )
            else:
                self.wfile.write(
                    _ndjson(_token("I'm here, sir."),
                            {"message": {"role": "assistant", "content": ""}, "done": True})
                )
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def assistant(config, fake_ollama, tmp_path):
    config.set("brain.host", fake_ollama["url"])
    config.set("brain.warm_on_start", False)
    config.set("ui.overlay", False)
    config.set("ui.tray", False)
    built = Assistant(config, text_mode=True)
    yield built
    built.stop()


def test_everything_wires_together(assistant):
    assert len(assistant.parts.dispatcher.tools_payload()) >= 20
    assert assistant.parts.client.available() is True
    assert assistant.parts.brain is not None


def test_a_plain_question_is_answered(assistant):
    assistant.start()
    assert "sir" in assistant.handle_text("are you there").lower()


def test_a_tool_call_completes_the_turn(assistant):
    assistant.start()
    reply = assistant.handle_text("what time is it")

    assert reply, "the turn must produce something to say"
    roles = [message["role"] for message in assistant.parts.conversation.messages()]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_a_guarded_tool_does_nothing_when_the_answer_is_no(assistant, monkeypatch):
    announced: list[str] = []
    monkeypatch.setattr(assistant, "_confirm_by_typing",
                        lambda text: announced.append(text) or False)

    result = assistant.parts.dispatcher.execute("run_powershell", {"command": "Get-ChildItem"})

    assert announced, "a guarded tool must say what it is about to do"
    assert "PowerShell" in announced[0]
    assert result.ok is False
    assert "cancelled" in result.summary.lower()


def test_the_blocklist_refuses_in_character(assistant):
    result = assistant.parts.dispatcher.execute("run_powershell", {"command": "format C: /q"})

    assert result.refused is True
    assert result.ok is False
    assert "sir" in result.summary.lower()


def test_a_safe_tool_runs_without_asking(assistant, monkeypatch):
    monkeypatch.setattr(assistant, "_confirm_by_typing",
                        lambda text: pytest.fail("a safe tool must never ask"))

    result = assistant.parts.dispatcher.execute("get_time_date", {})

    assert result.ok is True
    assert result.summary


def test_a_missing_voice_is_reported_not_fatal(assistant):
    """No Kokoro model on this box: he should still run, and say so."""
    assistant.start()
    assert assistant.handle_text("are you there")
    assert any("silent" in problem.lower() or "voice" in problem.lower()
               for problem in assistant.parts.problems)


def test_stop_is_idempotent(assistant):
    assistant.start()
    assistant.stop()
    assistant.stop()
