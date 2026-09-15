"""Command line entry point.

    python -m jarvis                      the whole thing: voice, overlay, tray
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
import sys

BANNER = r"""
   _____ ___    ____  _    __ ___ _____
  |_   _|   |  |  _ \| |  / /|_ _/  ___|     J.A.R.V.I.S.
    | | | | |  | |_) | | / /  | | \ \        Just A Rather Very Intelligent System
   _| |_| | |  |  _ <| |/ /   | |  \ \       Everything runs on this machine.
  |_____|___|  |_| \_\___/   |___|\___/
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis",
        description="J.A.R.V.I.S. - a fully local voice assistant.",
    )
    parser.add_argument("--config", default=None, metavar="PATH", help="path to config.yaml")
    parser.add_argument("--no-ui", action="store_true", help="no overlay and no tray icon")
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
        return _overlay_test(cfg, _setup(cfg))

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

    if args.say is not None:
        return _say_once(cfg, args.say, logger)

    if args.no_ui:
        cfg.set("ui.overlay", False)
        cfg.set("ui.tray", False)

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
