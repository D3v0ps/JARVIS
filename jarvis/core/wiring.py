"""Building JARVIS: every component, constructed once, in the right order.

Kept apart from the loop in :mod:`jarvis.core.assistant` so that neither file
becomes the kind of module nobody wants to open. The rule here is that a missing
piece is never fatal: no microphone, no GPU, no voice, no Ollama - each of those
degrades to something that still runs and says so, because an assistant that
refuses to start tells you nothing about what is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.audio.player import Player
from jarvis.brain.conversation import Conversation
from jarvis.brain.ollama_client import OllamaClient
from jarvis.config import Config, resolve_language
from jarvis.core.latency import LatencyTracker
from jarvis.core.logging import get_logger
from jarvis.core.memory import Memory
from jarvis.core.scheduler import Scheduler
from jarvis.core.state import StateBus

__all__ = ["Components", "build", "build_desk", "build_face"]


@dataclass
class Components:
    """Everything the loop needs, already wired together."""

    cfg: Config
    log: logging.Logger
    state: StateBus
    memory: Memory
    latency: LatencyTracker
    scheduler: Scheduler
    player: Player
    speaker: Any
    mic: Any
    wake: Any
    segmenter: Any
    transcriber: Any
    client: OllamaClient
    conversation: Conversation
    dispatcher: Any
    brain: Any
    language: str | None = None
    problems: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        """Record something the user should be told about, once."""
        if message not in self.problems:
            self.problems.append(message)
            self.log.warning(message)


def build(
    cfg: Config,
    *,
    on_due: Callable[[Any], None],
    speak: Callable[[str], None],
    confirm: Callable[[str], bool],
    notify: Callable[[str], None],
    text_mode: bool = False,
    logger: logging.Logger | None = None,
) -> Components:
    """Construct every part of the assistant. Never raises for a missing component."""
    log = logger or get_logger("assistant")
    state = StateBus()
    latency = LatencyTracker(log)
    language = resolve_language(cfg)

    memory = Memory(cfg.resolve_path("memory.json") if cfg.get("memory_path") else "memory.json")
    memory.load()

    scheduler = Scheduler(on_due=on_due)

    # --- voice out -------------------------------------------------------------------
    player = Player(
        cfg.get("audio.output_device"),
        logger=get_logger("audio.player"),
        volume=float(cfg.get("audio.output_volume", 1.0) or 1.0),
    )
    from jarvis.tts.engine import create_engine

    engine = create_engine(cfg, get_logger("tts"))
    from jarvis.tts.speaker import Speaker

    speaker = Speaker(engine, player, state, language=language,
                      logger=get_logger("tts.speaker"), latency=latency)

    components_problems: list[str] = []
    if getattr(engine, "name", "") == "null":
        components_problems.append("No speech engine could be loaded; JARVIS will be silent.")

    # --- ears ------------------------------------------------------------------------
    mic = _build_mic(cfg, log, text_mode, components_problems)
    wake = _build_wake(cfg, log, text_mode, components_problems)
    segmenter = _build_segmenter(cfg, log)
    transcriber = _build_transcriber(cfg, log)

    # --- brain -----------------------------------------------------------------------
    client = OllamaClient(
        str(cfg.get("brain.host", "http://127.0.0.1:11434")),
        str(cfg.get("brain.model", "qwen3:8b")),
        keep_alive=cfg.get("brain.keep_alive", -1),
        think=bool(cfg.get("brain.think", False)),
        num_ctx=int(cfg.get("brain.num_ctx", 8192)),
        temperature=float(cfg.get("brain.temperature", 0.6)),
        timeout=int(cfg.get("brain.request_timeout", 120)),
        logger=get_logger("brain.client"),
    )
    conversation = Conversation(
        cfg.resolve_path("prompts/jarvis_system.md"),
        memory,
        history_turns=int(cfg.get("brain.history_turns", 12)),
        language=language,
        logger=get_logger("brain.conversation"),
    )

    # --- hands -----------------------------------------------------------------------
    from jarvis.tools import registry
    from jarvis.tools.base import ToolContext
    from jarvis.tools.dispatcher import Dispatcher

    registry.load_all()

    # Asking Windows for its Start menu takes the better part of a second; do it now
    # so the first "open Spotify" does not pay for it.
    try:
        from jarvis.tools import windows_apps

        windows_apps.prewarm()
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not pre-warm the application list: %s", exc)

    ctx = ToolContext(
        config=cfg, memory=memory, logger=get_logger("tools"),
        speak=speak, confirm=confirm, notify=notify,
        scheduler=scheduler, state=state,
    )
    dispatcher = Dispatcher(ctx, logger=get_logger("tools.dispatcher"))

    try:  # deep_think needs a client of its own; wiring is the only place that has one
        from jarvis.tools.think_tools import set_deep_client

        set_deep_client(client)
    except Exception as exc:  # noqa: BLE001 - an optional tool must not break startup
        log.debug("deep_think is unavailable: %s", exc)

    # The configured model may not be the one that is actually pulled.
    resolve_model(client, cfg, log)

    from jarvis.brain.brain import Brain

    brain = Brain(
        client, conversation, dispatcher,
        max_tool_rounds=int(cfg.get("assistant.max_tool_rounds", 4)),
        logger=get_logger("brain"), latency=latency,
    )

    built = Components(
        cfg=cfg, log=log, state=state, memory=memory, latency=latency, scheduler=scheduler,
        player=player, speaker=speaker, mic=mic, wake=wake, segmenter=segmenter,
        transcriber=transcriber, client=client, conversation=conversation,
        dispatcher=dispatcher, brain=brain, language=language,
    )
    for problem in components_problems:
        built.note(problem)
    log.info(
        "Wired: %d tool(s), voice=%s, wake=%s, vad=%s",
        len(registry.all_specs()), getattr(engine, "name", "?"),
        "on" if getattr(wake, "available", False) else "off",
        "silero" if getattr(segmenter, "available", False) else "energy",
    )
    return built


# --- individual pieces ----------------------------------------------------------------
def _build_mic(cfg: Config, log: logging.Logger, text_mode: bool, problems: list[str]) -> Any:
    from jarvis.audio.capture import MicStream, NullMicStream

    if text_mode:
        return NullMicStream()
    try:
        return MicStream(
            device=cfg.get("audio.input_device"),
            sample_rate=int(cfg.get("audio.sample_rate", 16000)),
            block_size=int(cfg.get("audio.block_size", 1280)),
            logger=get_logger("audio.capture"),
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"The microphone could not be opened ({exc}); voice input is off.")
        log.warning("Falling back to a silent microphone: %s", exc)
        return NullMicStream()


def _build_wake(cfg: Config, log: logging.Logger, text_mode: bool, problems: list[str]) -> Any:
    from jarvis.wake.detector import WakeWord

    if text_mode or not cfg.get("wake.enabled", True):
        return None
    detector = WakeWord(
        model=str(cfg.get("wake.model", "hey_jarvis")),
        sensitivity=float(cfg.get("wake.sensitivity", 0.5)),
        framework=str(cfg.get("wake.framework", "onnx")),
        cooldown=float(cfg.get("wake.cooldown", 2.0)),
        logger=get_logger("wake"),
    )
    if not detector.available:
        problems.append(
            "The wake word model is not available; say nothing and he will not wake. "
            "Run scripts/fetch_models.py."
        )
    return detector


def _build_segmenter(cfg: Config, log: logging.Logger) -> Any:
    # Silero loads through torch.jit.load, which torch now warns about on every start.
    # It is torch talking to silero, not anything the user can act on, so keep it out
    # of a console that is supposed to read calmly.
    import warnings

    warnings.filterwarnings(
        "ignore", message=r".*torch\.jit\.load.*", category=FutureWarning
    )
    from jarvis.stt.vad import EnergySegmenter, SpeechSegmenter

    kwargs = dict(
        threshold=float(cfg.get("vad.threshold", 0.5)),
        silence_ms=int(cfg.get("vad.silence_ms", 700)),
        min_speech_ms=int(cfg.get("vad.min_speech_ms", 250)),
        max_utterance_s=float(cfg.get("vad.max_utterance_s", 15)),
        pre_roll_ms=int(cfg.get("vad.pre_roll_ms", 300)),
        sample_rate=int(cfg.get("audio.sample_rate", 16000)),
        logger=get_logger("stt.vad"),
    )
    silero = SpeechSegmenter(**kwargs)
    if silero.available:
        return silero
    log.info("Silero VAD is unavailable; using the energy segmenter instead.")
    return EnergySegmenter(**kwargs)


def _build_transcriber(cfg: Config, log: logging.Logger) -> Any:
    from jarvis.stt.transcriber import Transcriber

    return Transcriber(
        model=str(cfg.get("stt.model", "auto")),
        device=str(cfg.get("stt.device", "auto")),
        compute_type=str(cfg.get("stt.compute_type", "auto")),
        beam_size=int(cfg.get("stt.beam_size", 1)),
        language=resolve_language(cfg),
        vram_gb=cfg.get("system.vram_gb"),
        logger=get_logger("stt"),
    )


def build_face(
    cfg: Config,
    state: StateBus,
    *,
    on_quit: Callable[[], None],
    on_toggle_pause: Callable[[], None],
    on_show_window: Callable[[], None] | None = None,
    logger: logging.Logger,
) -> tuple[Any, Any]:
    """Return ``(overlay, tray)``, either of which may be None.

    The layered overlay is tried first because it is the one that looks right; the
    tkinter ring is the fallback for anything that is not Windows.
    """
    overlay = None
    if cfg.get("ui.overlay", True):
        try:
            from jarvis.ui.layered import LayeredOverlay

            candidate = LayeredOverlay(
                cfg, state, on_quit=on_quit, on_toggle_pause=on_toggle_pause, logger=logger
            )
            overlay = candidate if candidate.start() else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("The layered overlay could not start: %s", exc)
        if overlay is None:
            try:
                from jarvis.ui.overlay import Overlay

                overlay = Overlay(
                    cfg, state, on_quit=on_quit, on_toggle_pause=on_toggle_pause, logger=logger
                )
                overlay.start()
                if not overlay.is_alive():
                    overlay = None
            except Exception as exc:  # noqa: BLE001
                logger.info("No overlay this session: %s", exc)
                overlay = None

    tray = None
    if cfg.get("ui.tray", True):
        try:
            from jarvis.ui.tray import Tray

            tray = Tray(
                state,
                on_quit=on_quit,
                on_toggle_pause=on_toggle_pause,
                on_show_window=on_show_window,
                logger=logger,
            )
            tray.start()
        except Exception as exc:  # noqa: BLE001
            logger.info("No tray icon this session: %s", exc)
            tray = None

    return overlay, tray


def build_desk(
    cfg: Config, assistant: Any, logger: logging.Logger
) -> tuple[Any, Any]:
    """Return ``(server, window)``, either of which may be None.

    In the spirit of :func:`build_face`, and for the same reason: the desk's window is
    the nicest face JARVIS has and the least essential thing he owns. Flask may not be
    installed, the port may be taken, WebView2 may be missing - each of those ends in
    ``None`` and a line in the log, and the assistant carries on with the ring, or
    with nothing at all.

    The window is built even when it is not opened at start, because the ticket in the
    server's URL is minted per open, because its ticket is single-use; the tray's
    ``Open JARVIS`` opens the
    window that was made here rather than inventing a second door.
    """
    if not cfg.get("ui.window", True):
        return None, None

    server = None
    try:
        from jarvis.desk import DeskServer

        candidate = DeskServer(cfg, assistant, get_logger("desk"))
        server = candidate if candidate.start() else None
    except Exception as exc:  # noqa: BLE001 - ImportError included: the desk is optional
        logger.info("No desk window this session: %s", exc)
        server = None
    if server is None:
        return None, None

    window = None
    try:
        from jarvis.desk import DeskWindow

        window = DeskWindow(
            # A callable, never the string: the ticket in the URL is single-use and
            # expires, so a link captured here is worthless by the time the tray opens
            # the window an hour later. Every open asks the server for a fresh one.
            cfg, lambda: server.url,
            logger=get_logger("desk.window"),
            on_close=lambda: _window_closed(assistant, logger),
        )
        if _should_open_now(cfg, window, logger) and not window.start():
            logger.info("The desk window is served, but nothing here would host it.")
    except Exception as exc:  # noqa: BLE001
        logger.info("The desk window could not be opened: %s", exc)
        window = None
    if window is not None:
        _attach_window(server, window, logger)
    return server, window


def _attach_window(server: Any, window: Any, logger: logging.Logger) -> None:
    """Let the server reach the window, for the frames that show and hide it.

    The page's own title bar has a minimise and a close, and both of those are the
    window's business rather than the assistant's; the server is the only thing the
    page can talk to, so it is the thing that has to hold the reference.
    """
    attach = getattr(server, "attach_window", None)
    if not callable(attach):
        logger.debug("This DeskServer takes no window; the page's title bar will be inert.")
        return
    try:
        attach(window)
    except Exception as exc:  # noqa: BLE001 - a face is never worth the start-up
        logger.debug("The desk server would not take the window: %s", exc)


def _window_closed(assistant: Any, logger: logging.Logger) -> None:
    """The operator closed the window. It hid into the tray rather than quitting.

    Anyone who has just clicked the X has no way of knowing that, so the tray's menu is
    redrawn and one line goes to the log. This is also the hook the tray grows into the
    day it wants to say anything more than "Open JARVIS".
    """
    logger.info("The window has gone to the tray, sir; I am still listening.")
    tray = getattr(assistant, "tray", None)
    state = getattr(assistant, "state", None)
    if tray is None or state is None:
        return
    try:
        tray.on_state(state.state)
    except Exception as exc:  # noqa: BLE001 - the tray is decoration, always
        logger.debug("The tray would not take the news: %s", exc)


def _should_open_now(cfg: Config, window: Any, logger: logging.Logger) -> bool:
    """Whether to host the page at start-up. ``open_window_on_start`` decides, mostly.

    The exception is WebView2. pywebview can only create a window on the main thread,
    and by the time the tray's ``Open JARVIS`` is clicked the main thread is
    already inside ``run_forever``. With ``open_window_on_start: false`` on a machine
    where pywebview wins the chain, the tray's menu item would therefore never be able
    to open anything at all - so the window is created now and the operator is told
    why, which is the smaller of the two surprises.
    """
    if cfg.get("ui.open_window_on_start", True):
        return True
    if _winning_host(window) != "webview":
        return False
    logger.info(
        "WebView2 hosts the window here and can only be created on the main thread, "
        "so it opens now despite ui.open_window_on_start; set ui.window_mode to edge "
        "to have it wait in the tray instead."
    )
    return True


def _winning_host(window: Any) -> str:
    """Which host would take the page, asked without opening anything. ``none`` if unsure."""
    try:
        order = window._order()
        probe = window._probe
        return next((name for name in order if probe(name)), "none")
    except Exception:  # noqa: BLE001 - an unanswerable question is a no
        return "none"


# --- choosing a model that is actually there -------------------------------------------
#: Rough VRAM cost of a q4-quantised model: parameters * 0.6 GB, plus overhead.
_GB_PER_BILLION = 0.6
_MODEL_OVERHEAD_GB = 1.6


def _parameter_size(tag: str) -> float:
    """Billions of parameters from a model tag: qwen3:14b -> 14, llama3.1:8b-q4 -> 8."""
    import re

    suffix = tag.split(":", 1)[1] if ":" in tag else tag
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", suffix.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def choose_installed_model(
    wanted: str, installed: list[str], vram_gb: float | None = None
) -> str | None:
    """Pick the best model that is actually pulled when ``wanted`` is not.

    Prefers the same family - someone who pulled qwen3:14b wants qwen3 - and within
    it the largest that still fits the card. Falls back to anything installed rather
    than leaving the assistant with nothing to think with.
    """
    if not installed:
        return None
    if wanted in installed:
        return wanted

    def fits(tag: str) -> bool:
        if vram_gb is None:
            return True  # the user pulled it deliberately; do not second-guess them
        size = _parameter_size(tag)
        if size <= 0:
            return True
        return (size * _GB_PER_BILLION + _MODEL_OVERHEAD_GB) <= float(vram_gb) + 0.5

    base = wanted.split(":", 1)[0].lower()
    family = [tag for tag in installed if tag.split(":", 1)[0].lower() == base]
    for candidates in (family, installed):
        usable = [tag for tag in candidates if fits(tag)]
        pool = usable or candidates
        if pool:
            return max(pool, key=_parameter_size)
    return None


def resolve_model(client: OllamaClient, cfg: Config, log: logging.Logger) -> str | None:
    """Point the client at a model that exists, and say so.

    The configured model may simply not be pulled - config.yaml still saying qwen3:8b
    on a machine where the installer pulled qwen3:14b is exactly the case this exists
    for. Nothing is written to disk; the choice applies to this run and is logged.
    """
    wanted = str(cfg.get("brain.model", "qwen3:8b"))
    if not client.available():
        return None
    if client.has_model(wanted):
        return wanted

    installed = client.models()
    choice = choose_installed_model(wanted, installed, cfg.get("system.vram_gb"))
    if not choice:
        log.warning(
            "%s is not pulled and no other model is installed. Run: ollama pull %s",
            wanted, wanted,
        )
        return None
    if choice != wanted:
        client.model = choice
        log.warning(
            "%s is not pulled; using %s instead. Set brain.model to %s in config.yaml "
            "to make that permanent.",
            wanted, choice, choice,
        )
    return choice
