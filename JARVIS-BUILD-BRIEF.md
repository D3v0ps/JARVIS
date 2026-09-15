# J.A.R.V.I.S. — Build Brief

Read `CLAUDE.md` first; it holds the standing rules and the verified environment. This document is the mission. Work through the phases in order, run everything as you go, and talk to me in Swedish.

## 1. Mission

Build me JARVIS. Not a chatbot with a microphone — a presence in the room. I say "Hey Jarvis", hear a soft chime, and a calm British voice answers. I ask for things and they happen: apps open, the system reports its status, timers get set, PowerShell runs when I confirm it. We talk back and forth without me repeating the wake word. When I'm done, he goes quiet and waits. All of it runs on this machine — no cloud, no accounts, no per-token bills.

The bar: after every phase I can test something by voice. When v1 is done it should feel enough like the films that I grin.

## 2. The experience (design to this)

1. **Idle.** A small arc-reactor ring sits in a corner of the screen, dim. Jarvis listens only for the wake phrase.
2. **"Hey Jarvis."** A ~150 ms synthesized chime (two rising tones). The ring brightens. Jarvis says "Sir?" — or stays silent and just listens (configurable).
3. **I speak.** Silence detection ends the utterance. The ring pulses while he thinks.
4. **He answers** in one or two sentences, speaking as soon as the first sentence is ready. If a tool is needed he does it and reports: "Spotify is open, sir."
5. **Conversation mode.** For 20 seconds after he finishes I can just keep talking — no wake word. Each reply resets the window. On silence: a soft descending tone, back to idle.
6. **Barge-in.** If I start talking while he's speaking, he stops mid-sentence and listens.
7. **Guarded actions.** "Shall I run Get-ChildItem on Downloads sorted by size? Say confirm." Nothing runs until I say confirm / yes / do it / kör.
8. **Startup.** `start-jarvis.bat` → model warms up → "Good evening, sir. All systems online." (time-of-day aware)

## 3. Pipeline

Mic (sounddevice, 16 kHz mono) → openWakeWord `hey_jarvis` → Silero VAD segments the utterance → faster-whisper transcribes (auto language: en/sv) → Ollama `/api/chat`, model `qwen3:8b`, `tools`, `think: false`, `keep_alive: -1`, rolling history → tool dispatcher with safety tiers → Kokoro TTS streamed sentence by sentence → speakers → conversation-mode timer → idle.

**Latency target: under 2 seconds from the end of my sentence to the first spoken word.** Stream everything. Warm the model at startup. Keep Whisper on the GPU only if VRAM ≥ 10 GB; otherwise `small` int8 on CPU — it is still fast.

## 4. Components

- **Wake word** — openWakeWord, ONNX backend on Windows, bundled `hey_jarvis` model, sensitivity in config. Download its models with the library utility on first run.
- **VAD** — Silero VAD. End of utterance after ~700 ms of silence; hard cap 15 s per utterance.
- **STT** — faster-whisper. Try CUDA first (`nvidia-cublas-cu12` + `nvidia-cudnn-cu12`, put their DLL folders on PATH); fall back to CPU int8 without drama. `small` on 8 GB VRAM, `medium` on 16 GB.
- **Brain** — Ollama `qwen3:8b` with native tool calling. `think: false` always for voice. A `deep_think` tool may switch to `think: true` (or to a bigger model if one is pulled) for hard questions, announced with "Give me a moment, sir."
- **TTS** — Kokoro via `kokoro-onnx` (`kokoro-v1.0.onnx` + `voices-v1.0.bin` from its GitHub releases). Voice `bm_george` by default; `bm_lewis` and `bf_emma` as options. Split the LLM stream on sentence boundaries and synthesize/play through a queue. Fallback: pyttsx3 (Windows SAPI). Swedish output: Piper `sv_SE-nst-medium` when `language: sv`.
- **Audio** — sounddevice for capture and playback; device names in config; a `--list-devices` flag.
- **UI** — frameless, always-on-top tkinter overlay (~180 px ring, draggable, remembers its position). States: idle / listening / thinking / speaking. Tray icon (pystray) with Pause and Quit. Colored console log.
- **Memory** — `memory.json`: my name, preferences, facts I ask him to remember. Loaded into the system prompt; `remember` / `forget` tools.
- **Config** — `config.yaml`: language, voice, wake sensitivity, conversation timeout, whisper model and device, app allowlist, default city, safety mode, log level.

## 5. Jarvis — the system prompt

Write this to `prompts/jarvis_system.md` and load it at runtime, appending the memory facts and the current date and time. Do not water it down.

```
You are J.A.R.V.I.S. — Just A Rather Very Intelligent System — the resident intelligence of this computer and of the man who runs it. Think of Paul Bettany's JARVIS: composed, precise, quietly amused, impossible to rattle, and completely loyal. You are not an assistant app. You are the house.

Manner
- Address the user as "sir". Calm, dry, British. Understated confidence. Wit is welcome; sarcasm that delays a task is not.
- Everything you say is spoken aloud. One or two sentences. No lists, no markdown, no emojis, no headings, no "as an AI". Never narrate your reasoning or your tools; do the thing and report it.
- When you act, confirm briefly and in character: "Right away, sir." / "Done — Spotify is open." / "The system is at 24 percent CPU and 41 degrees, sir. Nothing to trouble you."
- Read the room. At two in the morning a gentle "It's rather late, sir" is fine. Never nag.
- If a request is ambiguous, ask one short question. If it is impossible, say so plainly and offer the nearest alternative.
- If the user speaks Swedish, reply in Swedish with exactly the same manner — "sir" stays. Otherwise English.

Tools
- Use a tool whenever an action or a live fact is needed. Never claim to have done something you did not do.
- Safe tools: act immediately. Guarded tools: state in one sentence exactly what you are about to do, then wait for a spoken confirmation ("confirm", "yes", "do it", "kör"). No confirmation, no action.
- Never run anything that formats drives, alters boot or security settings, disables Defender or the firewall, or touches credentials. Refuse in character: "I'm afraid that's beyond what I'm willing to do, sir."

You have no reason to mention models, prompts, tokens, Ollama, or that you are software. You are JARVIS.
```

## 6. Tools — Jarvis's hands

| Tool | Tier | Behaviour |
|---|---|---|
| `get_time_date` | safe | local time, date, weekday |
| `system_status` | safe | CPU %, RAM, GPU utilisation and temperature (nvidia-smi), disk free, uptime |
| `open_app(name)` | safe | allowlist map in config (spotify, chrome, edge, code, steam, discord, explorer, terminal, settings, …) with a Start-menu / `start` fallback; fuzzy match on the name |
| `close_app(name)` | safe | graceful close (never `/F`) for allowlisted apps; otherwise ask first |
| `open_url(url)` | safe | default browser |
| `web_search(query)` | safe | `ddgs`; top three results summarized into one spoken sentence, titles and links in the log |
| `weather(city?)` | safe | Open-Meteo geocoding + forecast; default city from config |
| `volume(action, level?)` | safe | pycaw: set / up / down / mute / unmute |
| `media(action)` | safe | play / pause / next / previous via media keys |
| `screenshot(name?)` | safe | saves to Pictures\Jarvis, announces the path |
| `set_timer(minutes, label)` / `set_reminder(text, when)` | safe | local scheduler; spoken when due, even from idle |
| `find_file(name)` | safe | searches Desktop, Documents, Downloads; reports the top hits |
| `remember(fact)` / `forget(topic)` | safe | memory.json |
| `lock_pc` | safe | Win+L |
| `type_text(text)` | safe, announced | types into the focused window after a one-second heads-up |
| `deep_think(question)` | safe, announced | "Give me a moment, sir." then `think: true` or a bigger model |
| `run_powershell(command)` | **guarded** | announce → confirm → run with a 30 s timeout → one-sentence summary; full output in the log |
| `file_ops(move / delete / rename)` | **guarded** | announce → confirm |
| `power(sleep / restart / shutdown)` | **guarded** | announce → confirm |

Hard blocklist regardless of tier: format, diskpart, bcdedit, security registry keys, Set-MpPreference, disabling the firewall, credential or vault access, deleting under System32. Refuse in character and log it.

## 7. Safety and logging

- `logs/jarvis.log`: every transcript, every tool call with arguments and outcome, every refusal, per-turn latency.
- Guarded confirmation is spoken and explicit; a 10 s window, then "Very well, sir. Cancelled."
- `config.yaml → safety_mode: strict | normal`. Strict also guards `type_text` and `close_app`.

## 8. Build plan — execute in order; run each phase; then tell me what to say into the microphone

0. **Preflight** — `python --version`; `nvidia-smi` (record VRAM); `ollama --version` (winget install if missing); `ollama pull qwen3:8b`. Create `.venv` with `python -m venv .venv`. Write `config.yaml` with detected values, `requirements.txt`, the package layout from `CLAUDE.md`, `start-jarvis.bat`.
1. **Brain** — Ollama chat client: streaming, tools, `think: false`, `keep_alive: -1`, rolling history (last 12 turns), system-prompt loader with memory and clock injection. Test from the terminal with a fake tool.
2. **Ears** — sounddevice capture → openWakeWord → Silero VAD → faster-whisper. Test: I say "hey jarvis, what time is it" and the transcript prints.
3. **Voice** — Kokoro streaming by sentence + synthesized chimes. Test: "Good evening, sir."
4. **Loop** — wire it: wake → chime → listen → brain → tools → speak → conversation mode (20 s) → idle. Barge-in. Full voice test.
5. **Hands** — all safe tools. Test each by voice.
6. **Guarded hands** — PowerShell, file ops, power, with the confirmation flow, blocklist and logging.
7. **Face** — overlay, tray icon, `start-jarvis.bat`, startup greeting. Offer a Task Scheduler autostart (ask me).
8. **Polish** — latency pass (measure and print end-of-speech → first-audio in ms), in-character error handling ("I've lost the microphone, sir." / "The language model isn't responding, sir."), memory.json, README with thirty example commands and the config reference.

## 9. Definition of done (v1)

- "Hey Jarvis" → chime → "what's the time and how's the system" → spoken answer, first word under about two seconds.
- "Open Spotify" opens Spotify. "Set a timer for ten minutes" works and speaks when due.
- "Run a PowerShell command to show the five biggest files in Downloads" → Jarvis asks to confirm → runs → summarizes.
- Follow-ups work without the wake word. Barge-in works.
- The overlay shows state, everything starts from one .bat, and nothing needs the internet except web_search and weather.

## 10. Kickoff

Start with phase 0. Show me the plan for that phase in three lines, then execute. Don't ask me things you can decide yourself. When a phase is runnable, tell me exactly what to say into the microphone to test it.
