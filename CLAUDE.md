# JARVIS — project rules

You are building and maintaining JARVIS: a fully local, voice-controlled desktop assistant for Windows 11, in the spirit of J.A.R.V.I.S. from Iron Man. The user is the operator. Talk to him in Swedish; write code, comments, commit messages and docs in English. The full mission and build plan live in `JARVIS-BUILD-BRIEF.md` — read it at the start of every session.

## Environment (verified — do not re-ask)
- Windows 11 (build 26200), native PowerShell. No WSL, no Docker.
- Python 3.13 on PATH as `python`. Never call `python3` — on this machine it resolves to a Windows Store stub.
- GPU: NVIDIA GeForce RTX 5060 Ti (8 or 16 GB VRAM — check `nvidia-smi` once, record the number in `config.yaml`). 32 GB DDR5 RAM.
- Microsoft C++ Build Tools (Desktop development with C++) are installed; native wheels can compile if unavoidable.
- Ollama may or may not be installed. Verify with `ollama --version`; if missing, `winget install Ollama.Ollama`. Model: `qwen3:8b` — pull it if absent (~5 GB).
- Audio I/O: `sounddevice` (bundled PortAudio wheels). Never use PyAudio.

## Hard rules
- Everything runs locally. No cloud LLM APIs, no accounts, no API keys. Web search and weather use keyless endpoints only (DuckDuckGo via `ddgs`, Open-Meteo).
- Tool safety tiers: SAFE tools run immediately. GUARDED tools (PowerShell, file delete/move/rename, sleep/restart/shutdown, anything touching settings, registry or network) require Jarvis to say what he is about to do and receive a spoken confirmation first.
- Hard blocklist regardless of tier: format/diskpart, bcdedit, security registry keys, Set-MpPreference, disabling the firewall, credential or vault access, deleting under System32. Refuse in character and log it.
- Log every transcript, tool call (with arguments and outcome) and refusal to `logs/jarvis.log`.
- Anything that gets spoken is 1–2 sentences unless the user asks for more. No markdown, lists or emojis in spoken text.
- Every build phase ends with something runnable and voice-testable. Run it — don't just write it — and tell the user exactly what to say to test it.
- Ask the user only when a real decision is needed. Otherwise pick the sensible default, state it in one line, and continue.
- Never download copyrighted audio (film sound effects, voices). Chimes are synthesized with numpy.

## Stack (do not swap without a stated reason)
- Wake word: openWakeWord, ONNX backend, bundled `hey_jarvis` model
- VAD: Silero VAD
- STT: faster-whisper (`small` on 8 GB VRAM / `medium` on 16 GB; CPU int8 fallback)
- Brain: Ollama `qwen3:8b` via `/api/chat` with `tools`, `think: false`, `keep_alive: -1`, streaming
- TTS: Kokoro via `kokoro-onnx`, voice `bm_george` (British); fallback pyttsx3; optional Piper `sv_SE-nst-medium` when `language: sv`
- UI: frameless always-on-top tkinter overlay + pystray tray icon
- Config: `config.yaml`; memory: `memory.json`; entry point: `start-jarvis.bat`

## Conventions
- Package layout: `jarvis/{audio,wake,stt,brain,tts,tools,ui,core}`, `prompts/`, `logs/`, `tests/`
- Type hints, small modules, one responsibility each. Threads for audio; never block the capture loop.
- Measure latency (end of speech → first audio) and print it in the console log on every turn.
