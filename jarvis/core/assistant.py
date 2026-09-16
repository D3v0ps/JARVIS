"""The loop: wake, listen, think, act, speak, and go quiet again.

One thread reads the microphone and nothing else, because it must never block -
not while whisper transcribes, not while the model writes, not while JARVIS
talks. Everything slow happens on a worker thread, which is also what makes
barge-in possible: the ear is still listening while the mouth is moving.

    IDLE        the wake word model sees every frame
    LISTENING   the segmenter collects an utterance
    WORKING     the worker transcribes, thinks, acts and speaks; the ear watches
                for you talking over him
    CONFIRMING  a guarded tool is waiting for you to say "confirm"
"""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum
from typing import Any, Callable

import numpy as np

from jarvis.audio import chimes
from jarvis.config import Config
from jarvis.core import wiring
from jarvis.core.logging import get_logger, log_transcript
from jarvis.core.state import AssistantState
from jarvis.tools.safety import is_cancellation, is_confirmation

__all__ = ["Assistant"]

CHIME_RATE = 24000
FRAME_TIMEOUT = 0.5


class _Mode(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    WORKING = "working"
    CONFIRMING = "confirming"


class Assistant:
    """Everything, wired together and running."""

    def __init__(self, cfg: Config, *, text_mode: bool = False) -> None:
        self.cfg = cfg
        self.text_mode = text_mode
        self.log = get_logger("assistant")

        self._mode = _Mode.IDLE
        self._running = threading.Event()
        self._paused = threading.Event()
        self._stop_turn = threading.Event()
        self._listen_until = 0.0
        self._audio_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._work: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)

        self._confirm_reply: str | None = None
        self._confirm_ready = threading.Event()
        self._barge_frames = 0

        self.conversation_timeout = float(cfg.get("assistant.conversation_timeout", 20) or 20)
        self.confirmation_timeout = float(cfg.get("assistant.confirmation_timeout", 10) or 10)
        self.barge_in = bool(cfg.get("audio.barge_in", True))
        self.barge_in_ms = float(cfg.get("audio.barge_in_speech_ms", 400) or 400)
        self.sample_rate = int(cfg.get("audio.sample_rate", 16000))
        self.block_size = int(cfg.get("audio.block_size", 1280))

        self.parts = wiring.build(
            cfg,
            on_due=self._on_job_due,
            speak=self._tool_speak,
            confirm=self._tool_confirm,
            notify=self._notify,
            text_mode=text_mode,
            logger=self.log,
        )
        self.state = self.parts.state
        self.reflex = self._build_reflex()
        self.ducker = self._build_ducker()
        self.remote: Any = None
        self.overlay: Any = None
        self.tray: Any = None
        self._hud: Any = None

    def _build_reflex(self) -> Any:
        """The grammar that answers plain commands without waking the model."""
        if not self.cfg.get("assistant.reflex", True):
            return None
        try:
            from jarvis.brain.reflex import ReflexMatcher

            matcher = ReflexMatcher(
                self.cfg.resolve_path("prompts/sentences.yaml"),
                language=self.cfg.get("language"),
                logger=get_logger("brain.reflex"),
            )
            if matcher.available:
                self.log.info("Reflex grammar ready: %d template(s).", matcher.templates())
                return matcher
            self.log.info("No reflex grammar; every utterance goes to the model.")
        except Exception as exc:  # noqa: BLE001 - the fast path is optional, always
            self.log.warning("Could not load the reflex grammar: %s", exc)
        return None

    def _build_ducker(self) -> Any:
        """Quietens everything else while he is listening."""
        if self.text_mode or not self.cfg.get("audio.ducking.enabled", True):
            return None
        try:
            from jarvis.audio.ducking import Ducker

            return Ducker(
                self.state,
                level=float(self.cfg.get("audio.ducking.level", 0.2)),
                ramp_ms=int(self.cfg.get("audio.ducking.ramp_ms", 120)),
                logger=get_logger("audio.ducking"),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.debug("Audio ducking unavailable: %s", exc)
        return None

    # --- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        """Bring everything online: face, scheduler, model warm-up, greeting."""
        if self._running.is_set():
            return
        self._running.set()
        self.parts.scheduler.start()

        if not self.text_mode:
            self.overlay, self.tray = wiring.build_face(
                self.cfg, self.state,
                on_quit=self.stop, on_toggle_pause=self.toggle_pause, logger=self.log,
            )
            self._hud = getattr(self.overlay, "hud", None)
            self.parts.dispatcher = _HudDispatcher(self.parts.dispatcher, self._hud)
            self.parts.brain.dispatcher = self.parts.dispatcher

        if self.ducker is not None:
            # A synchronous crash-recovery pass first: a kill mid-turn must not leave
            # Spotify at twenty percent forever.
            self.ducker.start()

        self._warm_up()
        self._greet()
        self._start_remote()

        if not self.text_mode:
            self._audio_thread = threading.Thread(
                target=self._audio_loop, name="jarvis-ears", daemon=True
            )
            self._worker = threading.Thread(
                target=self._work_loop, name="jarvis-mind", daemon=True
            )
            self._audio_thread.start()
            self._worker.start()

    def run_forever(self) -> None:
        """Block until stopped. The overlay may need the main thread; this honours that."""
        if self.overlay is not None and getattr(self.overlay, "needs_main_thread", True):
            run = getattr(self.overlay, "run", None)
            if callable(run):
                run()
                return
        while self._running.is_set():
            time.sleep(0.2)

    def run_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="jarvis-main", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self.log.info("Shutting down.")
        self._running.clear()
        self._stop_turn.set()
        self._work.put(None)
        for component, method in (
            (self.remote, "stop"), (self.ducker, "stop"),
            (self.parts.speaker, "stop"), (self.parts.speaker, "close"),
            (self.parts.player, "close"), (self.parts.mic, "stop"),
            (self.parts.scheduler, "stop"), (self.parts.client, "close"),
            (self.overlay, "stop"), (self.tray, "stop"),
        ):
            if component is None:
                continue
            try:
                getattr(component, method)()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.debug("%s.%s failed: %s", type(component).__name__, method, exc)

    def pause(self) -> None:
        self._paused.set()
        self.parts.speaker.stop()
        self.state.set(AssistantState.PAUSED)
        self.log.info("Paused.")

    def resume(self) -> None:
        self._paused.clear()
        self._to_idle(chime=False)
        self.log.info("Resumed.")

    def toggle_pause(self) -> None:
        self.resume() if self._paused.is_set() else self.pause()

    # --- startup ----------------------------------------------------------------------
    def _warm_up(self) -> None:
        if not self.cfg.get("brain.warm_on_start", True):
            return
        if not self.parts.client.available():
            self.parts.note(
                "Ollama is not answering; JARVIS can hear you but has nothing to think with."
            )
            return
        if not self.parts.client.has_model():
            self.parts.note(f"The model {self.cfg.get('brain.model')} has not been pulled yet.")
            return
        threading.Thread(target=self.parts.client.warm, name="jarvis-warm", daemon=True).start()
        if not self.text_mode:
            threading.Thread(target=self._preload_stt, name="jarvis-stt", daemon=True).start()

    def _preload_stt(self) -> None:
        try:
            self.parts.transcriber.load()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Speech recognition could not be loaded: %s", exc)

    def _greet(self) -> None:
        if not self.cfg.get("assistant.startup_greeting", True) or self.text_mode:
            return
        hour = time.localtime().tm_hour
        part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        greeting = f"Good {part}, sir. All systems online."
        if self.parts.problems:
            greeting = f"Good {part}, sir. I'm online, though not everything is where it should be."
        self.parts.speaker.say(greeting, blocking=False)

    # --- the ear ----------------------------------------------------------------------
    def _audio_loop(self) -> None:
        mic = self.parts.mic
        try:
            mic.start()
        except Exception as exc:  # noqa: BLE001
            self.parts.note(f"I've lost the microphone, sir. ({exc})")
            return

        self.state.set(AssistantState.IDLE)
        while self._running.is_set():
            frame = mic.read(timeout=FRAME_TIMEOUT)
            if frame is None:
                self._expire_conversation()
                continue
            if self._paused.is_set():
                continue
            try:
                self._route(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the loop
                self.log.exception("Audio routing failed: %s", exc)
            self._expire_conversation()

    def _route(self, frame: np.ndarray) -> None:
        self._report_level(frame)

        if self._mode is _Mode.IDLE:
            wake = self.parts.wake
            if wake is not None and wake.process(frame):
                self._on_wake()
            return

        if self._mode is _Mode.WORKING:
            self._watch_for_barge_in(frame)
            return

        utterance = self.parts.segmenter.push(frame)
        if utterance is None:
            return

        if self._mode is _Mode.CONFIRMING:
            self._deliver_confirmation(utterance)
            return

        self.parts.latency.start_turn()
        self._mode = _Mode.WORKING
        self.state.set(AssistantState.THINKING)
        try:
            self._work.put_nowait(utterance)
        except queue.Full:
            self.log.warning("Still busy with the previous utterance; dropping this one.")
            self._to_listening()

    def _on_wake(self) -> None:
        self.log.info("Wake word detected (score %.2f).", getattr(self.parts.wake, "last_score", 0))
        self.parts.speaker.play_chime(chimes.wake_chime(CHIME_RATE), CHIME_RATE)
        if self.cfg.get("assistant.wake_greeting", True):
            phrase = str(self.cfg.get("assistant.acknowledge_phrase", "Sir?"))
            if phrase:
                self.parts.speaker.say(phrase, blocking=False)
        self._to_listening()

    def _watch_for_barge_in(self, frame: np.ndarray) -> None:
        """Cut him off when you start talking over him."""
        if not self.barge_in or not self.parts.speaker.is_speaking:
            self._barge_frames = 0
            return
        level = float(np.sqrt(np.mean(np.square(frame))) if frame.size else 0.0)
        frame_ms = 1000.0 * frame.size / self.sample_rate
        needed = max(1, int(self.barge_in_ms / max(frame_ms, 1.0)))
        # A deliberately high bar: the speakers bleed into the microphone, and cutting
        # him off because he heard himself would be worse than not cutting him off.
        if level > 0.055:
            self._barge_frames += 1
            if self._barge_frames >= needed:
                self.log.info("Barge-in: stopping mid-sentence.")
                self._barge_frames = 0
                self._stop_turn.set()
                self.parts.speaker.stop()
                self._to_listening()
        else:
            self._barge_frames = 0

    def _report_level(self, frame: np.ndarray) -> None:
        """Feed the overlay the live microphone level, so the ring answers the room."""
        overlay = self.overlay
        if overlay is None or not hasattr(overlay, "set_amplitude"):
            return
        try:
            level = float(np.sqrt(np.mean(np.square(frame))) if frame.size else 0.0)
            overlay.set_amplitude(min(1.0, level * 12.0))
        except Exception:  # noqa: BLE001
            pass

    # --- the mind ---------------------------------------------------------------------
    def _work_loop(self) -> None:
        while self._running.is_set():
            item = self._work.get()
            if item is None:
                return
            from_phone = not isinstance(item, np.ndarray)
            try:
                if from_phone:
                    self._handle_remote_turn(item)
                else:
                    self._handle_utterance(item)
            except Exception as exc:  # noqa: BLE001
                self.log.exception("The turn failed: %s", exc)
                if not from_phone:
                    self._say_in_character("Something went wrong on my end, sir.")
            finally:
                # A turn that came from the phone must not leave the desk microphone
                # armed and waiting for a follow-up nobody is going to speak.
                if not from_phone:
                    self._to_listening()

    def _handle_utterance(self, utterance: np.ndarray) -> None:
        transcript = self.parts.transcriber.transcribe(utterance, self.sample_rate)
        self.parts.latency.mark("text")
        text = (transcript.text or "").strip()
        if not text:
            self.log.debug("Nothing intelligible in %.1f s of audio.", transcript.duration_s)
            return

        # Brain.turn logs the user line itself; logging it here too printed it twice.
        if self._hud is not None:
            self._hud.begin_turn(text)
            self._hud.state = AssistantState.THINKING

        if self._answer_by_reflex(text):
            return

        self._stop_turn.clear()
        result = self.parts.brain.turn(
            text, on_sentence=self._on_sentence, should_stop=self._stop_turn.is_set
        )
        if not result.cancelled:
            self.parts.speaker.wait(timeout=60)
        if self._hud is not None:
            self._hud.latency_ms = self.parts.latency.marks().get("first audio")
            self._hud.touch()
        self.log.info("Turn complete: %s", self.parts.latency.summary())
        self.parts.latency.end_turn()

    def _answer_by_reflex(self, text: str) -> bool:
        """Answer a plain command from the grammar, skipping the model entirely.

        Nine utterances in ten are imperatives. Going through the model costs a second
        and a half and gives it a chance to describe the action instead of taking it;
        the grammar dispatches through the *same* dispatcher, so tiers and the
        blocklist still hold, and a guarded tool still asks.
        """
        matcher = self.reflex
        if matcher is None or self._mode is _Mode.CONFIRMING:
            return False
        try:
            reflex = matcher.match(text)
        except Exception as exc:  # noqa: BLE001 - never lose a turn to the fast path
            self.log.warning("The reflex grammar failed on %r: %s", text, exc)
            return False
        if reflex is None:
            return False

        log_transcript("user", text)
        self.log.info("Reflex: %s -> %s(%s)", reflex.template, reflex.tool, reflex.arguments)
        if self._hud is not None:
            self._hud.tool_started(reflex.tool)

        started = time.perf_counter()
        result = self.parts.dispatcher.execute(reflex.tool, dict(reflex.arguments))
        duration_ms = (time.perf_counter() - started) * 1000.0
        if self._hud is not None:
            self._hud.tool_finished(
                reflex.tool, ok=bool(getattr(result, "ok", False)),
                duration_ms=duration_ms, refused=bool(getattr(result, "refused", False)),
            )

        self._speak_reflex_result(reflex, result)

        if self._hud is not None:
            self._hud.latency_ms = self.parts.latency.marks().get("first audio")
            self._hud.touch()
        self.log.info("Reflex turn complete in %.0f ms: %s", duration_ms,
                      self.parts.latency.summary() or "no model involved")
        self.parts.latency.end_turn()
        return True

    def _speak_reflex_result(self, reflex: Any, result: Any) -> None:
        """Say the confirmation - or, in brief mode, simply play a tone.

        A tone instead of a sentence is not only faster to hear: it removes the whole
        text-to-speech step from a turn that was already model-free.
        """
        ok = bool(getattr(result, "ok", False))
        refused = bool(getattr(result, "refused", False))

        if self.cfg.get("assistant.brief_mode", False) and not self.text_mode:
            spec = None
            try:
                from jarvis.tools import registry

                spec = registry.get(reflex.tool)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None and not getattr(spec, "speak_result", True):
                chime = chimes.done() if ok else chimes.error_chime()
                self.parts.speaker.play_chime(chime, CHIME_RATE)
                log_transcript("jarvis", "(tone)")
                return

        try:
            reply = self.reflex.reply_for(reflex, result)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("reply_for failed: %s", exc)
            reply = getattr(result, "summary", "") or ""
        if reply:
            self._say_in_character(reply)
            log_transcript("jarvis", reply)
        if not ok and not refused:
            self.parts.speaker.play_chime(chimes.error_chime(), CHIME_RATE)

    # --- the phone ---------------------------------------------------------------------
    def _start_remote(self) -> None:
        """Bring the phone server up, if it was asked for."""
        if self.text_mode or not self.cfg.get("remote.enabled", False):
            return
        try:
            from jarvis.remote import RemoteServer

            self.remote = RemoteServer(self.cfg, self, get_logger("remote"))
            if self.remote.start():
                self.log.info("Remote access is on at %s", self.remote.url)
            else:
                self.remote = None
        except Exception as exc:  # noqa: BLE001 - never let the phone stop the desk
            self.log.warning("Remote access could not start: %s", exc)
            self.remote = None

    def submit_remote_turn(self, turn: Any) -> bool:
        """Queue one turn that came from the phone. False means busy, try again.

        It goes onto the same queue as the microphone rather than around it: there is
        one GPU, one Whisper and one Ollama, and two turns at once would fight.
        """
        try:
            self._work.put_nowait(turn)
            return True
        except queue.Full:
            self.log.info("Busy with a turn already; the phone will have to wait.")
            return False

    def _handle_remote_turn(self, turn: Any) -> None:
        """Transcribe and answer a turn that arrived from the phone."""
        text = (getattr(turn, "text", "") or "").strip()
        audio = getattr(turn, "audio", None)
        if not text and audio is not None:
            self.parts.latency.start_turn()
            transcript = self.parts.transcriber.transcribe(audio, self.sample_rate)
            self.parts.latency.mark("text")
            text = (transcript.text or "").strip()
            if text:
                # Show the phone what was heard, before the answer starts.
                report = getattr(turn, "transcribed", None)
                if callable(report):
                    report(text)
        if not text:
            turn.finish("", error="I couldn't make that out, sir.")
            return

        device = getattr(turn, "device", "the phone")
        self.log.info("Remote turn from %s: %r", device, text)
        if self._hud is not None:
            self._hud.begin_turn(text)

        spoken: list[str] = []

        def on_sentence(sentence: str) -> None:
            spoken.append(sentence)
            turn.send(sentence)
            if self._hud is not None:
                self._hud.add_reply(sentence)
            if self.cfg.get("remote.speak_locally", False):
                self.parts.speaker.enqueue(sentence)

        # The phone's guard wraps the dispatcher for exactly this turn. Without it a
        # remote caller gets the desk's rules, and type_text would send blind
        # keystrokes into whatever window is focused here.
        brain = self.parts.brain
        original_dispatcher = brain.dispatcher
        wrap = getattr(turn, "wrap_dispatcher", None)
        if callable(wrap):
            try:
                brain.dispatcher = wrap(original_dispatcher)
            except Exception as exc:  # noqa: BLE001 - no guard means no turn
                self.log.error("Could not apply the remote guard: %s", exc)
                turn.finish("", error="I can't take that from the phone right now, sir.")
                return

        self._stop_turn.clear()
        try:
            result = brain.turn(text, on_sentence=on_sentence,
                                should_stop=self._stop_turn.is_set)
            turn.finish(" ".join(spoken).strip() or result.reply, error=result.error)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("The remote turn failed: %s", exc)
            turn.finish("", error="Something went wrong on my end, sir.")
        finally:
            brain.dispatcher = original_dispatcher
            self.parts.latency.end_turn()

    def _on_sentence(self, sentence: str) -> None:
        self.parts.speaker.enqueue(sentence)
        if self._hud is not None:
            self._hud.add_reply(sentence)

    # --- guarded confirmation ----------------------------------------------------------
    def _tool_confirm(self, announcement: str) -> bool:
        """Say what is about to happen, then wait to be told to go ahead."""
        self._say_in_character(announcement)
        if self.text_mode:
            return self._confirm_by_typing(announcement)

        if self._hud is not None:
            self._hud.note = "Say confirm"
            self._hud.touch()

        self._confirm_reply = None
        self._confirm_ready.clear()
        self.parts.speaker.wait(timeout=20)
        self.parts.mic.flush()
        self.parts.segmenter.reset()
        self._mode = _Mode.CONFIRMING
        self.state.set(AssistantState.LISTENING)

        # A tone every two seconds, so the window you have to answer in is audible
        # as well as visible. Five repeats cover the ten seconds exactly.
        waiting = threading.Event()

        def pulse() -> None:
            period = getattr(chimes, "AWAITING_PERIOD_MS", 2000) / 1000.0
            while not waiting.wait(period):
                try:
                    self.parts.speaker.play_chime(chimes.awaiting(CHIME_RATE), CHIME_RATE)
                except Exception:  # noqa: BLE001 - a missing tone is not a failure
                    return

        pulser = threading.Thread(target=pulse, name="jarvis-awaiting", daemon=True)
        pulser.start()

        granted = False
        if self._confirm_ready.wait(timeout=self.confirmation_timeout):
            reply = (self._confirm_reply or "").strip()
            log_transcript("user", reply)
            granted = is_confirmation(reply) and not is_cancellation(reply)

        waiting.set()
        self._mode = _Mode.WORKING
        self.state.set(AssistantState.THINKING)
        if self._hud is not None:
            self._hud.note = "" if granted else "cancelled"
            self._hud.touch()
        if not granted:
            self._say_in_character("Very well, sir. Cancelled.")
        return granted

    def _confirm_by_typing(self, announcement: str) -> bool:
        try:
            answer = input(f"  {announcement}\n  confirm? ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return is_confirmation(answer) and not is_cancellation(answer)

    def _deliver_confirmation(self, utterance: np.ndarray) -> None:
        """Transcribed on the ear's thread: the mind is busy waiting for this answer."""
        try:
            transcript = self.parts.transcriber.transcribe(utterance, self.sample_rate)
            self._confirm_reply = (transcript.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Could not transcribe the confirmation: %s", exc)
            self._confirm_reply = ""
        finally:
            self._confirm_ready.set()

    # --- speaking ----------------------------------------------------------------------
    def _tool_speak(self, text: str) -> None:
        self._say_in_character(text)

    def _say_in_character(self, text: str) -> None:
        if not text:
            return
        if self.text_mode:
            print(f"  jarvis > {text}")
            return
        self.parts.speaker.say(text, blocking=False)
        if self._hud is not None:
            self._hud.add_reply(text)

    def _notify(self, text: str) -> None:
        """Speak regardless of state - a timer coming due from idle still gets said."""
        was_idle = self._mode is _Mode.IDLE
        self._say_in_character(text)
        if was_idle and not self.text_mode:
            self.parts.speaker.wait(timeout=30)
            self._to_idle(chime=False)

    def _on_job_due(self, job: Any) -> None:
        text = getattr(job, "text", "") or getattr(job, "label", "") or "Your timer is up"
        self.log.info("Scheduler fired: %s", text)
        self._notify(f"{text}, sir.")

    # --- mode transitions ---------------------------------------------------------------
    def _to_listening(self) -> None:
        self.parts.mic.flush()
        self.parts.segmenter.reset()
        self._barge_frames = 0
        self._listen_until = time.monotonic() + self.conversation_timeout
        self._mode = _Mode.LISTENING
        self.state.set(AssistantState.LISTENING)

    def _to_idle(self, *, chime: bool = True) -> None:
        if chime and not self.text_mode:
            self.parts.speaker.play_chime(chimes.sleep_chime(CHIME_RATE), CHIME_RATE)
        self.parts.segmenter.reset()
        self._mode = _Mode.IDLE
        self._listen_until = 0.0
        if self.parts.wake is not None:
            self.parts.wake.reset()
        self.state.set(AssistantState.PAUSED if self._paused.is_set() else AssistantState.IDLE)

    def _expire_conversation(self) -> None:
        """The follow-up window closing is what sends him back to sleep."""
        if self._mode is not _Mode.LISTENING or not self._listen_until:
            return
        if time.monotonic() < self._listen_until:
            return
        if self.parts.speaker.is_speaking:
            self._listen_until = time.monotonic() + 2.0
            return
        self.log.debug("Conversation window closed.")
        self._to_idle()

    # --- text mode -----------------------------------------------------------------------
    def handle_text(self, text: str) -> str:
        """One turn with no audio anywhere. Same brain, same tools."""
        said: list[str] = []

        def collect(sentence: str) -> None:
            said.append(sentence)

        self._stop_turn.clear()
        self.parts.latency.start_turn()
        log_transcript("user", text)
        result = self.parts.brain.turn(text, on_sentence=collect, should_stop=lambda: False)
        return " ".join(said).strip() or result.reply


class _HudDispatcher:
    """Wraps the dispatcher so the overlay can show a tool while it is running."""

    def __init__(self, inner: Any, hud: Any) -> None:
        self._inner = inner
        self._hud = hud

    def execute(self, name: str, args: dict) -> Any:
        if self._hud is not None:
            self._hud.tool_started(name)
        started = time.perf_counter()
        result = self._inner.execute(name, args)
        if self._hud is not None:
            self._hud.tool_finished(
                name,
                ok=bool(getattr(result, "ok", False)),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                refused=bool(getattr(result, "refused", False)),
            )
        return result

    def execute_many(self, calls: Any) -> Any:
        return [self.execute(getattr(c, "name", c.get("name", "")), getattr(c, "arguments", c.get("arguments", {}))) for c in calls]

    def tools_payload(self) -> list[dict]:
        return self._inner.tools_payload()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
