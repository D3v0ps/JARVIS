"""Command line entry point.

    python -m jarvis                      the whole thing: voice, window, ring, tray
    python -m jarvis --no-window          the ring and the tray, no desk window
    python -m jarvis --no-ui              console only
    python -m jarvis --text               type instead of talk, same brain and tools
    python -m jarvis --say "Good evening" speak one line and exit
    python -m jarvis --list-devices       audio devices, for config.yaml
    python -m jarvis --preflight [--fix]  environment doctor
    python -m jarvis --config other.yaml  alternative configuration

Everything heavy is imported lazily inside the branch that needs it, so
``--preflight`` still works on a machine where the audio stack is broken or
half-installed - which is exactly when you need it most.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Any

#: Kept alive for the life of the process: a closed sink is worse than no sink.
_SINKS: list[Any] = []

BANNER = r"""
   _____ ___    ____  _    __ ___ _____
  |_   _|   |  |  _ \| |  / /|_ _/  ___|     J.A.R.V.I.S.
    | | | | |  | |_) | | / /  | | \ \        Just A Rather Very Intelligent System
   _| |_| | |  |  _ <| |/ /   | |  \ \       Everything runs on this machine.
  |_____|___|  |_| \_\___/   |___|\___/
"""


def _install_null_streams() -> bool:
    """Give the process somewhere to write when Windows gave it nowhere.

    Started from ``pythonw.exe`` there is no console: ``sys.stdout`` and ``sys.stderr``
    are ``None``. ``print()`` copes with that by doing nothing, but a ``StreamHandler``
    over ``None`` raises on every record, and so does any library that assumes a
    stream exists. Two handles on the null device cost nothing and make the question
    go away before the first import that might ask it.

    Each sink is tagged with ``NULL_STREAM_FLAG`` so that :func:`setup_logging`, which
    runs later, can tell it apart from a real console. Untagged, a handle on the null
    device looks like a perfectly good stream and earns a console handler that writes
    every record to nowhere - which is precisely the sink the logging change exists to
    avoid.

    Returns whether there was a console to print a banner to.
    """
    from jarvis.core.logging import NULL_STREAM_FLAG

    console = sys.stdout is not None
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            sink = open(os.devnull, "w", encoding="utf-8")
            try:
                setattr(sink, NULL_STREAM_FLAG, True)
            except (AttributeError, TypeError):  # pragma: no cover - CPython allows it
                pass
            _SINKS.append(sink)
            setattr(sys, name, sink)
    return console


def _install_crash_hooks(logger: Any) -> None:
    """Send every unhandled exception to the log, because there is nowhere else.

    Under ``pythonw`` a traceback printed to ``sys.stderr`` goes to the null device, so
    the launcher's message box shows the last fifteen lines of a log that never mentions
    the crash: "JARVIS exited" and a record that stops mid-sentence. Both hooks are
    needed - the audio, mind and desk threads die through
    :func:`threading.excepthook`, not through :data:`sys.excepthook`.
    """
    previous_hook = sys.excepthook

    def fall_through(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        """Let the terminal have its traceback too, when there is a terminal."""
        try:
            previous_hook(exc_type, exc, tb)
        except Exception:  # noqa: BLE001 - the hook of last resort may not raise
            pass

    def on_main_thread(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            fall_through(exc_type, exc, tb)  # Ctrl+C is a decision, not a fault
            return
        try:
            logger.critical("Unhandled exception in the main thread", exc_info=(exc_type, exc, tb))
        finally:
            fall_through(exc_type, exc, tb)

    def on_other_thread(args: Any) -> None:
        if args.exc_type is None or issubclass(args.exc_type, SystemExit):
            return
        thread = getattr(args, "thread", None)
        logger.critical(
            "Unhandled exception in thread %s", getattr(thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = on_main_thread
    threading.excepthook = on_other_thread


def _set_app_user_model_id(app_id: str = "Jarvis.Desk", logger: Any = None) -> bool:
    """Claim an identity in the shell before the first window exists.

    Without this, every window the process opens is filed under the interpreter: the
    taskbar and Alt-Tab show the Python logo and the name "python", which rather
    undermines an assistant with his own icon. Windows-only, and a failure is a
    cosmetic one - it is logged and the assistant carries on.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # local import: the shell API exists on Windows only

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception:  # noqa: BLE001 - an icon is never worth a crash
        if logger is not None:
            logger.debug("Could not set the AppUserModelID %r", app_id, exc_info=True)
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis",
        description="J.A.R.V.I.S. - a fully local voice assistant.",
    )
    parser.add_argument("--config", default=None, metavar="PATH", help="path to config.yaml")
    parser.add_argument("--no-ui", action="store_true",
                        help="no window, no overlay and no tray icon")
    parser.add_argument("--window", dest="window", action="store_true", default=None,
                        help="open the desk window, whatever config.yaml says")
    parser.add_argument("--no-window", dest="window", action="store_false",
                        help="no desk window; the ring and the tray remain")
    parser.add_argument("--text", action="store_true", help="terminal chat instead of voice")
    parser.add_argument("--say", metavar="TEXT", default=None, help="speak one line and exit")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--preflight", action="store_true", help="check this machine and exit")
    parser.add_argument("--overlay-test", dest="overlay_test", action="store_true",
                        help="show the overlay on its own for ten seconds and exit")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    # Accepted here so the installer can pass them through in one command line.
    parser.add_argument("--fix", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--set-model", dest="set_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set-whisper", dest="set_whisper", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    console = _install_null_streams()
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.version:
        from jarvis import __version__

        print(f"JARVIS {__version__}")
        return 0

    # --- the doctor: no audio, no model, no assistant required ---------------
    if args.preflight or args.fix or args.set_model or args.set_whisper:
        from jarvis.preflight import main as preflight_main

        forwarded: list[str] = []
        if args.config:
            forwarded += ["--config", args.config]
        if args.fix:
            forwarded.append("--fix")
        if args.set_model:
            forwarded += ["--set-model", args.set_model]
        if args.set_whisper:
            forwarded += ["--set-whisper", args.set_whisper]
        if args.json:
            forwarded.append("--json")
        return preflight_main(forwarded)

    if args.overlay_test:
        from jarvis.config import Config as _Config
        from jarvis.core.logging import setup_logging as _setup

        try:
            cfg = _Config.load(args.config)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read the configuration: {exc}", file=sys.stderr)
            return 1
        logger = _setup(cfg)
        _install_crash_hooks(logger)
        _set_app_user_model_id(logger=logger)
        return _overlay_test(cfg, logger)

    if args.list_devices:
        from jarvis.audio.devices import print_devices

        print_devices()
        return 0

    # --- everything past here needs the configuration ------------------------
    from jarvis.config import Config

    try:
        cfg = Config.load(args.config)
    except Exception as exc:  # noqa: BLE001 - a bad config file must not traceback
        print(f"Could not read the configuration: {exc}", file=sys.stderr)
        return 1

    from jarvis.core.logging import setup_logging

    logger = setup_logging(cfg)
    _install_crash_hooks(logger)
    # Before the overlay, the tray or the desk window: the shell reads it once.
    _set_app_user_model_id(logger=logger)

    if args.say is not None:
        return _say_once(cfg, args.say, logger)

    if args.window is not None:
        cfg.set("ui.window", bool(args.window))
    if args.no_ui:
        # The blunt instrument, and it stays blunt: everything with a face goes.
        cfg.set("ui.overlay", False)
        cfg.set("ui.tray", False)
        cfg.set("ui.window", False)

    if console:
        print(BANNER)

    from jarvis.core.assistant import Assistant

    assistant = Assistant(cfg, text_mode=args.text)
    try:
        if args.text:
            return _text_loop(assistant)
        assistant.start()
        assistant.run_forever()
    except KeyboardInterrupt:
        print()
        logger.info("Interrupted from the keyboard.")
    finally:
        assistant.stop()
    return 0


def _overlay_test(cfg, logger) -> int:
    """Cycle the overlay through every state so you can see it in ten seconds.

    Worth its own flag: the overlay is the one part that cannot be proven by a test,
    only by looking at it.
    """
    import time

    from jarvis.core.state import AssistantState, StateBus
    from jarvis.core.wiring import build_face

    state = StateBus()
    overlay, tray = build_face(
        cfg, state, on_quit=lambda: None, on_toggle_pause=lambda: None, logger=logger
    )
    if overlay is None:
        print("  No overlay could be started on this machine. The log says why.")
        return 1

    backend = type(overlay).__name__
    print(f"  {backend} is up. Cycling through the states - watch the corner of your screen.")

    hud = getattr(overlay, "hud", None)
    script = [
        (AssistantState.IDLE, 2.0, None, None),
        (AssistantState.LISTENING, 2.5, "what's the time and how's the system", None),
        (AssistantState.THINKING, 2.5, None, "system_status"),
        (AssistantState.SPEAKING, 3.0,
         None, "It's just gone eleven, sir, and the system is barely awake at four percent."),
        (AssistantState.IDLE, 1.5, None, None),
    ]
    try:
        for assistant_state, seconds, heard, extra in script:
            state.set(assistant_state)
            if hud is not None:
                if heard:
                    hud.begin_turn(heard)
                if assistant_state is AssistantState.THINKING and extra:
                    hud.tool_started(extra)
                if assistant_state is AssistantState.SPEAKING and extra:
                    hud.tool_finished("system_status", ok=True, duration_ms=118)
                    hud.add_reply(extra)
                    hud.latency_ms = 1180
            print(f"    {assistant_state.value}")
            deadline = time.time() + seconds
            while time.time() < deadline:
                if hasattr(overlay, "set_amplitude") and assistant_state in (
                    AssistantState.LISTENING, AssistantState.SPEAKING
                ):
                    overlay.set_amplitude(0.3 + 0.5 * abs(time.time() % 1 - 0.5))
                time.sleep(0.05)
        print("  Done. If that looked right, he is ready.")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for component in (overlay, tray):
            if component is not None:
                try:
                    component.stop()
                except Exception:  # noqa: BLE001
                    pass


def _say_once(cfg, text: str, logger) -> int:
    """TTS smoke test: does this machine actually have a voice?"""
    from jarvis.audio.player import Player
    from jarvis.tts.engine import create_engine

    engine = create_engine(cfg, logger)
    player = Player(cfg.get("audio.output_device"), logger=logger,
                    volume=float(cfg.get("audio.output_volume", 1.0)))
    try:
        audio, rate = engine.synthesize(text)
        print(f"  voice: {engine.name}  ({len(audio) / rate:.1f} s of audio)")
        player.play(audio, rate, blocking=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  could not speak: {exc}", file=sys.stderr)
        return 1
    finally:
        player.close()
        engine.close()


def _text_loop(assistant) -> int:
    """Same brain, same tools, no microphone. Handy for testing on any machine."""
    print(BANNER)
    print("  Text mode. Empty line or Ctrl+C to leave.\n")
    assistant.start()
    while True:
        try:
            line = input("  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        reply = assistant.handle_text(line)
        print(f"  jarvis > {reply}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
