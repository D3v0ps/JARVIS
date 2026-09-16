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

import logging
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

#: The fan has no assistant to borrow a logger from, and its failures are debug noise.
_fan_log = get_logger("desk.fan")

#: The only keys a face may write. The desk server validates the shape of a value;
#: this decides whether the key may be touched at all, so a second face - or a bug in
#: the first - still cannot reach brain.model from a web page.
DESK_SETTINGS: frozenset[str] = frozenset({
    "tts.speed",
    "audio.chime_volume",
    "assistant.brief_mode",
    "wake.sensitivity",
    "ui.always_on_top",
})

def safe_card_data(data: Any) -> dict:
    """Only the keys a face is already trusted with, and only if they render.

    The desk has full rights over what JARVIS may *do*; that is not a reason to post a
    tool's raw output into a web page, where the card renderer would draw it. The
    allowlist is the phone's own, imported rather than restated, because a second copy
    is a second thing to remember when a tool grows a field. No list, no card data.
    """
    if not isinstance(data, dict):
        return {}
    try:
        from jarvis.remote.session import ToolReporter

        allowed = ToolReporter.SAFE_DATA_KEYS
    except Exception:  # noqa: BLE001 - without the list, nothing is known to be safe
        _fan_log.debug("The safe-key list is unavailable; the card goes out empty.")
        return {}
    return {
        key: value
        for key, value in data.items()
        if key in allowed and isinstance(value, (str, int, float, bool, list))
    }


#: A button press has no words, and the confirmation path downstream reads words.
#: These two are what a click is written down as, and both survive ``is_confirmation``.
CONFIRM_WORD = "confirm"
CANCEL_WORD = "cancel"


class _Mode(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    WORKING = "working"
    CONFIRMING = "confirming"


#: What the ring should say for each mode. A guarded confirmation interrupts whatever
#: the loop was doing and has to hand both the mode *and* the state back afterwards,
#: and the two disagreeing is how the ring ends up thinking he is still busy.
_STATE_FOR_MODE: dict[_Mode, AssistantState] = {
    _Mode.IDLE: AssistantState.IDLE,
    _Mode.LISTENING: AssistantState.LISTENING,
    _Mode.WORKING: AssistantState.THINKING,
    _Mode.CONFIRMING: AssistantState.THINKING,
}


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

        # Only one guarded tool may ask at a time. A routine frame runs its calls on
        # the socket's reader thread and can reach a GUARDED tool while the worker
        # thread is already waiting for an answer; two confirmation bars at once is
        # not a state the window can express, so the second one queues behind the first.
        self._confirm_gate = threading.Lock()
        # The spoken answer and the window's button race for the same slot; the lock
        # is what makes "first one wins" true rather than nearly true.
        self._confirm_lock = threading.Lock()
        #: Every confirmation is minted with an id of its own, so an answer that
        #: arrives late cannot be applied to whichever tool happens to be asking now.
        self._confirm_seq = 0
        self._confirm_id = 0
        self._confirm_reply: str | None = None
        self._confirm_ready: threading.Event | None = None
        #: Set from the window's thread, read by the ear's. Waking up touches the
        #: microphone, the segmenter and the wake word, all of which belong to the
        #: capture loop, so the crossing is a flag - exactly as _stop_turn is.
        self._arm_request = threading.Event()
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
        self.desk: Any = None
        self.window: Any = None
        self._hud: Any = None
        self._desk_bus: Any = None
        self._desk_log_handler: logging.Handler | None = None
        self._desk_unsubscribe: Callable[[], None] | None = None

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
                on_quit=self.stop, on_toggle_pause=self.toggle_pause,
                on_show_window=self.show_window, logger=self.log,
            )
            self.desk, self.window = wiring.build_desk(self.cfg, self, self.log)
            self._hud = self._build_hud(getattr(self.overlay, "hud", None))
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
        """Block until stopped, giving the main thread to whichever face must have it.

        Only one of them can: pywebview's loop and tkinter's both insist on thread
        one. The window wins that argument - it is the application, the ring is an
        ornament - and on Windows the argument does not happen at all, because the
        layered overlay runs its own message pump on a thread of its own.
        """
        window, overlay = self.window, self.overlay
        if window is not None and getattr(window, "needs_main_thread", False):
            if overlay is not None and getattr(overlay, "needs_main_thread", True):
                self.log.info(
                    "The desk window needs the main thread, so the fallback ring "
                    "stands down for this session, sir."
                )
                self._drop_overlay()
            if self._run_on_main(window):
                return
        elif overlay is not None and getattr(overlay, "needs_main_thread", True):
            if self._run_on_main(overlay):
                return
        while self._running.is_set():
            time.sleep(0.2)

    def _run_on_main(self, face: Any) -> bool:
        """Hand the main thread to ``face``. False when it has nothing to block in."""
        run = getattr(face, "run", None)
        if not callable(run):
            return False
        run()
        return True

    def _drop_overlay(self) -> None:
        """Let the ring go: it cannot have the thread the window is about to take."""
        overlay, self.overlay = self.overlay, None
        if overlay is None:
            return
        try:
            overlay.stop()
        except Exception as exc:  # noqa: BLE001 - it was on its way out anyway
            self.log.debug("The overlay did not stop cleanly: %s", exc)

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
        self._unwire_desk()
        for component, method in (
            (self.remote, "stop"), (self.ducker, "stop"),
            (self.parts.speaker, "stop"), (self.parts.speaker, "close"),
            (self.parts.player, "close"), (self.parts.mic, "stop"),
            (self.parts.scheduler, "stop"), (self.parts.client, "close"),
            (self.window, "stop"), (self.desk, "stop"),
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

        # The window asked for the microphone from its own thread; wake up here, where
        # the mic, the segmenter and the detector are all ours to touch.
        if self._arm_request.is_set():
            self._arm_request.clear()
            if self._mode is _Mode.IDLE:
                self._on_wake()
                return
            self.log.debug("The armed microphone arrived while %s; ignored.", self._mode.value)

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

    # --- the window ---------------------------------------------------------------------
    def submit_desk_turn(self, text: str) -> bool:
        """One typed turn from the window: full rights, spoken out of these speakers.

        It travels as a :class:`~jarvis.remote.session.RemoteTurn` because that is
        already how anything which is not the microphone gets a turn - but with no
        ``wrap_dispatcher``, which is precisely what "the desk" means. The window sits
        at the keyboard, and the keyboard was never the phone. False means busy.
        """
        said = " ".join(str(text or "").split())
        if not said:
            self.log.warning("The window sent an empty turn.")
            return False
        try:
            from jarvis.remote.session import RemoteTurn
        except Exception as exc:  # noqa: BLE001 - no remote package, no typed turn
            self.log.error("A desk turn needs jarvis.remote.session: %s", exc)
            return False

        def speak(sentence: str) -> None:
            # A remote turn is spoken here only when remote.speak_locally says so; a
            # desk turn is always spoken here, and must be spoken exactly once.
            if not self.cfg.get("remote.speak_locally", False):
                self.parts.speaker.enqueue(sentence)

        def done(reply: str, error: str) -> None:
            if error:
                self._say_in_character(error)
            self.log.info("Desk turn complete: %s", (error or reply or "")[:80])

        turn = RemoteTurn(
            audio=None,
            text=said,
            device="the desk",
            on_sentence=speak,
            on_done=done,
            speak_locally=True,
        )
        self.log.info("Desk turn (typed): %r", said[:80])
        return self.submit_remote_turn(turn)

    def arm_listening(self) -> bool:
        """Open the microphone as though the wake word had fired. False when he cannot.

        The same call the ear makes, rather than a second copy of it: the chime, the
        acknowledgement and the follow-up window all belong to waking up, and a button
        that only did some of that would be a different thing wearing the same name.

        It is *asked for* rather than done here, because this runs on the websocket
        reader thread and waking up flushes the microphone, resets the segmenter and
        resets the wake word - three objects the capture loop is using at that instant.
        The next frame turns the flag into the wake the ear would have made itself.
        """
        if self.text_mode:
            return False
        if self._paused.is_set():
            self.log.info("The window asked me to listen, but I am paused, sir.")
            return False
        if self._mode is not _Mode.IDLE:
            self.log.info("The window asked me to listen while %s.", self._mode.value)
            return False
        self.log.info("The window armed the microphone.")
        self._arm_request.set()
        return True

    def abort_turn(self) -> None:
        """Stop talking and abandon the turn - the button that barge-in gives you by voice."""
        try:
            self.parts.speaker.stop()
        except Exception as exc:  # noqa: BLE001 - a dead speaker is not a failed stop
            self.log.debug("The speaker would not stop: %s", exc)
        self._stop_turn.set()
        self.log.info("The turn was stopped from the window.")

    def answer_confirmation(self, granted: bool) -> bool:
        """Answer a waiting GUARDED tool with a click. False when nothing is waiting.

        The mode and the pending id are read under the same lock the answer is written
        with: between a check taken outside it and a write made inside it, the tool
        that was asking can have timed out and a second one taken its place.
        """
        word = CONFIRM_WORD if granted else CANCEL_WORD
        with self._confirm_lock:
            if self._mode is not _Mode.CONFIRMING or not self._confirm_id:
                self.log.info("The window answered a confirmation nobody was waiting for.")
                return False
            if not self._settle_locked(word, source="the window"):
                return False
        self.log.info("The window %s the confirmation.", "granted" if granted else "refused")
        return True

    def run_routine(self, name: str) -> str:
        """Run a named macro from ``remote.routines``; returns the line it came back with.

        The phone's runner already finds the macro, walks its calls and composes that
        line. The only thing that differs at the desk is the rights, so the loop is
        borrowed and the tier guard dropped rather than the whole of it written twice.
        """
        wanted = " ".join(str(name or "").split())
        if not wanted:
            return ""
        try:
            from jarvis.remote.server import RemoteServer

            class _DeskRoutines(RemoteServer):
                """The phone's macro runner with the desk's rights: no tier guard."""

                def guard_for(self, device: Any) -> Callable[[Any], Any]:
                    return lambda inner: inner

            ok, said = _DeskRoutines(self.cfg, self, self.log).run_routine(wanted, "the desk")
        except Exception as exc:  # noqa: BLE001 - a macro is never worth the session
            self.log.exception("The routine %r failed: %s", wanted, exc)
            return ""
        if said:
            self._say_in_character(said)
        self.log.info("Routine %r finished %s.", wanted, "well" if ok else "badly")
        return said

    def apply_setting(self, key: str, value: Any) -> bool:
        """Change one allowlisted key, write it to config.yaml, and apply it now if cheap.

        The key and the value go through the same door. A face allowed to set
        ``tts.speed`` is not thereby allowed to set it to ``"quickly"``, to ``NaN`` or
        to forty, and the table that knows the difference is the desk server's -
        imported rather than copied, because two allowlists drift the day one is edited.
        """
        name = str(key or "").strip()
        if name not in DESK_SETTINGS:
            self.log.warning("Refusing to set %r from a face: it is not on the list.", name)
            return False
        try:
            from jarvis.desk.server import SETTINGS
        except Exception as exc:  # noqa: BLE001 - no desk package means no window either
            self.log.warning(
                "Refusing to set %s: the desk's value table is unavailable (%s).", name, exc
            )
            return False
        rule = SETTINGS.get(name)
        if rule is None:
            self.log.warning("Refusing to set %s: the desk has no shape for it.", name)
            return False
        try:
            checked = rule(value)
        except (TypeError, ValueError) as exc:
            self.log.warning("Refusing %r for %s: %s", value, name, exc)
            return False
        try:
            self.cfg.set(name, checked)
            saved = self.cfg.save()
        except Exception as exc:  # noqa: BLE001 - a read-only config file is not a crash
            self.log.warning("Could not save %s: %s", name, exc)
            return False
        if saved is False:
            # Config.save reports a file it could not write by returning False. Telling
            # the window a slider was saved when it was not is the kind of lie that ends
            # with the operator moving it three times and restarting.
            self.log.warning("%s could not be written to config.yaml.", name)
            return False
        self._apply_setting_now(name, checked)
        self.log.info("%s is now %r.", name, checked)
        return True

    def show_window(self) -> None:
        """The tray's way back to the window: raise it, or open it for the first time."""
        window = self.window
        if window is None:
            self.log.info("There is no desk window this session, sir.")
            return
        try:
            if not window.show():
                window.start()
        except Exception as exc:  # noqa: BLE001 - the tray thread must outlive this
            self.log.warning("The desk window would not open: %s", exc)

    def _apply_setting_now(self, key: str, value: Any) -> None:
        """Make a saved setting true for this session wherever that costs nothing.

        ``assistant.brief_mode`` is read from the config every time it matters and so
        needs nothing here; ``ui.always_on_top`` is decided when the overlay is built
        and honestly waits for the next start.
        """
        try:
            if key == "tts.speed":
                self._set_speed(float(value))
            elif key == "audio.chime_volume":
                self.parts.speaker.chime_volume = float(value)
            elif key == "wake.sensitivity" and self.parts.wake is not None:
                self.parts.wake.sensitivity = float(value)
        except Exception as exc:  # noqa: BLE001 - it is saved; this session can miss out
            self.log.debug("%s could not be applied to this session: %s", key, exc)

    def _set_speed(self, speed: float) -> None:
        """Retune the voice mid-session, by a setter if the Speaker ever grows one.

        It keeps its engines to itself and has no such setter today, so until it does
        this reaches for the engines it can see rather than making the operator
        restart JARVIS to hear a slider move.
        """
        speaker = self.parts.speaker
        setter = getattr(speaker, "set_speed", None)
        if callable(setter):
            setter(speed)
            return
        cached = getattr(speaker, "_engines", None)
        engines = [getattr(speaker, "_engine", None)]
        engines += list(cached.values()) if isinstance(cached, dict) else []
        for engine in engines:
            if engine is not None and hasattr(engine, "speed"):
                engine.speed = speed

    def _build_hud(self, hud: Any) -> Any:
        """What the fifteen narration call sites write to for the rest of the session.

        With no window that is the overlay's own model, exactly as before. With one it
        is the fan, so neither the overlay nor the loop ever learns that a second face
        is listening.
        """
        bus = getattr(self.desk, "bus", None)
        if bus is None:
            return hud
        self._desk_bus = bus
        self._wire_desk_bus(bus)
        return _HudFan(hud, bus, marks=self.parts.latency.marks)

    def _wire_desk_bus(self, bus: Any) -> None:
        """The two streams the narration does not carry: the state, and the log tail."""
        try:
            self._desk_unsubscribe = self.state.subscribe(
                lambda state: bus.publish("state", state=str(state))
            )
        except Exception as exc:  # noqa: BLE001
            self.log.debug("The window cannot follow the state: %s", exc)
        try:
            from jarvis.desk.bus import DeskLogHandler

            handler = DeskLogHandler(bus)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("jarvis").addHandler(handler)
            self._desk_log_handler = handler
        except Exception as exc:  # noqa: BLE001 - a drawer with nothing in it will do
            self.log.debug("The window's log drawer is unavailable: %s", exc)

    def _unwire_desk(self) -> None:
        """Detach from the log and the state bus before the window's bus is closed."""
        handler, self._desk_log_handler = self._desk_log_handler, None
        if handler is not None:
            try:
                logging.getLogger("jarvis").removeHandler(handler)
                handler.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.debug("The log handler would not detach: %s", exc)
        unsubscribe, self._desk_unsubscribe = self._desk_unsubscribe, None
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception as exc:  # noqa: BLE001
                self.log.debug("The state bus had already let the window go: %s", exc)
        # Closed last, and only once nothing can still publish into it: a bus left open
        # keeps every subscriber's queue alive, and the page reads that as a live
        # server and reconnects to it for the rest of the machine's day.
        bus, self._desk_bus = self._desk_bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.debug("The window's bus would not close: %s", exc)

    def _desk_publish(self, kind: str, **payload: Any) -> None:
        """Tell the window something the HUD interface has no word for."""
        bus = self._desk_bus
        if bus is None:
            return
        try:
            bus.publish(kind, **payload)
        except Exception as exc:  # noqa: BLE001 - the window is never worth a turn
            self.log.debug("Publishing %s to the window failed: %s", kind, exc)

    # --- guarded confirmation ----------------------------------------------------------
    def _tool_confirm(self, announcement: str) -> bool:
        """Say what is about to happen, then wait to be told to go ahead.

        Serialised: a desk routine walks its tool calls on the websocket's reader
        thread, so a second GUARDED tool can reach here while the worker thread is
        still waiting for an answer to the first. Sharing one slot meant one spoken
        "confirm" granted both of them, and the window cannot draw two confirmation
        bars in any case, so the second question waits for the first to be answered.
        """
        with self._confirm_gate:
            return self._confirm_once(announcement)

    def _confirm_once(self, announcement: str) -> bool:
        """One question, asked and answered, with the gate already held."""
        self._say_in_character(announcement)
        if self.text_mode:
            return self._confirm_by_typing(announcement)

        if self._hud is not None:
            self._hud.note = "Say confirm"
            self._hud.touch()

        self.parts.speaker.wait(timeout=20)
        self.parts.mic.flush()
        self.parts.segmenter.reset()
        # Whatever the loop was doing before the question, it goes back to afterwards.
        # A microphone turn is already WORKING, but a typed desk turn, a routine or a
        # timer firing may be IDLE or LISTENING, and hard-coding WORKING on the way
        # out left the ear watching for barge-in and the wake word unread for ever.
        previous = self._mode
        token, ready = self._open_confirmation()
        self.state.set(AssistantState.LISTENING)
        # Published only once the mode is set, so the bar the window draws and the
        # button on it become live at the same moment.
        self._desk_publish(
            "confirm", announcement=announcement, seconds=self.confirmation_timeout
        )

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

        answered = ready.wait(timeout=self.confirmation_timeout)
        waiting.set()
        reply = self._close_confirmation(token, previous)
        granted = False
        if answered:
            log_transcript("user", reply)
            granted = is_confirmation(reply) and not is_cancellation(reply)

        self.state.set(_STATE_FOR_MODE.get(previous, AssistantState.THINKING))
        self._desk_publish("confirm_done", granted=granted)
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
            reply = (transcript.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Could not transcribe the confirmation: %s", exc)
            reply = ""
        self._settle_confirmation(reply, source="the microphone")

    def _open_confirmation(self) -> tuple[int, threading.Event]:
        """Mint a confirmation of its own and go into CONFIRMING with it.

        The id is what keeps two guarded tools apart. An answer is written into the
        slot that is open at the moment it lands, and the tool that asked reads back
        only the answer that was given to *its* id - so a "confirm" shouted at one
        question can never be the thing that runs another.
        """
        with self._confirm_lock:
            self._confirm_seq += 1
            self._confirm_id = self._confirm_seq
            self._confirm_reply = None
            self._confirm_ready = threading.Event()
            self._mode = _Mode.CONFIRMING
            return self._confirm_id, self._confirm_ready

    def _close_confirmation(self, token: int, previous: _Mode) -> str:
        """Retire ``token``'s slot, restore the mode it interrupted, return its answer.

        Empty when nobody answered, and empty as well when the slot has already moved
        on - which cannot happen while the gate is held, and is the honest answer if
        it ever does.
        """
        with self._confirm_lock:
            if self._confirm_id != token:
                return ""
            reply = self._confirm_reply
            self._confirm_id = 0
            self._confirm_reply = None
            self._confirm_ready = None
            self._mode = previous
        return (reply or "").strip()

    def _settle_confirmation(self, reply: str, *, source: str) -> bool:
        """Write down the first answer to arrive. False means somebody else was first.

        The spoken word and the window's button are two threads racing for one slot,
        and a race that both win would run a guarded tool on the strength of an answer
        the operator had already changed his mind about.
        """
        with self._confirm_lock:
            return self._settle_locked(reply, source=source)

    def _settle_locked(self, reply: str, *, source: str) -> bool:
        """:meth:`_settle_confirmation` with ``_confirm_lock`` already held."""
        pending, ready = self._confirm_id, self._confirm_ready
        if not pending or ready is None:
            self.log.debug("A confirmation from %s arrived with nothing pending.", source)
            return False
        if ready.is_set():
            self.log.debug("A confirmation from %s arrived second; ignored.", source)
            return False
        self._confirm_reply = reply
        ready.set()
        return True

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


class _HudFan:
    """Looks exactly like HudModel; writes to the overlay and to the DeskBus.

    The loop narrates itself at fifteen call sites, and the window wants every one of
    them. A second call beside each would be fifteen chances to add one and forget
    the other, so this satisfies the same interface and the loop goes on talking to a
    single object. Either half may be missing - no overlay, no window, or neither -
    and nothing here raises back into the turn that was only trying to say what it
    was doing.
    """

    def __init__(
        self,
        hud: Any = None,
        bus: Any = None,
        *,
        marks: Callable[[], dict] | None = None,
    ) -> None:
        self._hud = hud
        self._bus = bus
        #: Reads the latency tracker's marks, so the window gets the breakdown and not
        #: just the total. Optional: the overlay only ever needed the total.
        self._marks = marks
        self._note = ""
        self._latency_ms: float | None = None
        self._state: AssistantState = AssistantState.IDLE

    # --- the attributes the loop assigns to -------------------------------------------
    @property
    def state(self) -> AssistantState:
        return self._state

    @state.setter
    def state(self, value: AssistantState) -> None:
        # Not published: the window hears about states from StateBus, and two sources
        # for one fact is how they start disagreeing.
        self._state = value
        if self._hud is not None:
            self._hud.state = value

    @property
    def note(self) -> str:
        return self._note

    @note.setter
    def note(self, value: str) -> None:
        self._note = value
        if self._hud is not None:
            self._hud.note = value
        self._publish("note", text=str(value or ""))

    @property
    def latency_ms(self) -> float | None:
        return self._latency_ms

    @latency_ms.setter
    def latency_ms(self, value: float | None) -> None:
        self._latency_ms = value
        if self._hud is not None:
            self._hud.latency_ms = value
        if value is None:
            return
        marks: dict = {}
        if self._marks is not None:
            try:
                marks = dict(self._marks() or {})
            except Exception:  # noqa: BLE001 - a missing breakdown is not a failure
                marks = {}
        self._publish("latency", ms=float(value), marks=marks)

    # --- the methods the loop calls ----------------------------------------------------
    def begin_turn(self, heard: str) -> None:
        self._note = ""
        self._latency_ms = None
        if self._hud is not None:
            self._hud.begin_turn(heard)
        self._publish("heard", text=str(heard or ""))

    def add_reply(self, sentence: str) -> None:
        if self._hud is not None:
            self._hud.add_reply(sentence)
        self._publish("sentence", text=str(sentence or ""))

    def tool_started(self, name: str) -> Any:
        """Start a tool. Returns the overlay's own ToolEvent, or None when it has none."""
        event = self._hud.tool_started(name) if self._hud is not None else None
        self._publish("tool", name=str(name), phase="start")
        return event

    def tool_finished(
        self,
        name: str,
        *,
        ok: bool,
        duration_ms: float,
        refused: bool = False,
        summary: str = "",
        data: Any = None,
    ) -> None:
        """The overlay's signature, plus the two fields a card is made of.

        Both are keyword-only and both default, because ``HudModel`` has neither and
        the same call site writes to whichever of the two this session built. The ring
        never sees them - it has no room for a card - so they are not forwarded.
        """
        if self._hud is not None:
            self._hud.tool_finished(name, ok=ok, duration_ms=duration_ms, refused=refused)
        self._publish(
            "tool", name=str(name), phase="end", ok=bool(ok), refused=bool(refused),
            ms=float(duration_ms), summary=str(summary or ""), data=safe_card_data(data),
        )

    def touch(self) -> None:
        if self._hud is not None:
            self._hud.touch()

    # --- the halves --------------------------------------------------------------------
    def _publish(self, kind: str, **payload: Any) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            bus.publish(kind, **payload)
        except Exception:  # noqa: BLE001 - a face is never worth the turn it describes
            _fan_log.debug("The desk bus refused a %s event.", kind, exc_info=True)

    def __getattr__(self, item: str) -> Any:
        """Anything else the renderer reads - heard, reply, tools - belongs to the HUD."""
        hud = self.__dict__.get("_hud")
        if hud is None:
            raise AttributeError(item)
        return getattr(hud, item)


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
            # What the tool actually found is the whole of a card, and this is the only
            # call site that still holds the result. Offered to the fan alone: HudModel's
            # signature is the overlay's, and widening it for a face it cannot draw would
            # be the wrong file to change.
            extra: dict[str, Any] = {}
            if isinstance(self._hud, _HudFan):
                extra = {
                    "summary": str(getattr(result, "summary", "") or ""),
                    "data": getattr(result, "data", None),
                }
            self._hud.tool_finished(
                name,
                ok=bool(getattr(result, "ok", False)),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                refused=bool(getattr(result, "refused", False)),
                **extra,
            )
        return result

    def execute_many(self, calls: Any) -> Any:
        return [self.execute(getattr(c, "name", c.get("name", "")), getattr(c, "arguments", c.get("arguments", {}))) for c in calls]

    def tools_payload(self) -> list[dict]:
        return self._inner.tools_payload()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
