# J.A.R.V.I.S.

A fully local, voice-controlled desktop assistant for Windows 11 — wake word, speech
recognition, a language model, tools with real hands on the machine, and a calm British
voice. Nothing leaves the computer except two keyless lookups (web search and weather).
No cloud LLM, no accounts, no API keys, no per-token bills.

```
"Hey Jarvis."                    → chime, the ring brightens
"What's the time and how's the system?"
                                 → "It's 21:47, sir. The system is at 12 percent
                                    with 41 degrees on the GPU. Nothing to trouble you."
"Open Spotify."                  → Spotify opens. "Spotify is open, sir."
"Run a PowerShell command to show the five biggest files in Downloads."
                                 → "I'm about to run this in PowerShell: ... Say confirm."
"Confirm."                       → runs, then one sentence of summary.
```

---

## 1. What it is made of

| Layer | Component | Notes |
|---|---|---|
| Wake word | **openWakeWord** (`hey_jarvis`, ONNX) | runs continuously on the CPU, ~1 % of a core |
| Segmentation | **Silero VAD** | ends the utterance after 700 ms of silence, 15 s cap |
| Speech to text | **faster-whisper** | CUDA when there is VRAM for it, otherwise CPU int8 |
| Brain | **Ollama** running `qwen3:8b` | native tool calling, streaming, `think: false`, resident |
| Speech | **Kokoro** (`kokoro-onnx`), voice `bm_george` | Piper for Swedish, Windows SAPI as a fallback |
| Face | tkinter arc-reactor overlay + pystray tray icon | frameless, always on top, draggable |
| Memory | `memory.json` | facts you ask him to remember, injected into the system prompt |

Everything is wired together in `jarvis/core/assistant.py`:

```
mic → wake word → VAD → Whisper → Ollama (+tools) → Kokoro → speakers
                                       ↓
                            tool dispatcher with safety tiers
```

---

## 2. Install

Requirements: Windows 11, Python 3.13 on `PATH`, an NVIDIA GPU is nice but not required.

```powershell
git clone <this repo> JARVIS
cd JARVIS

# 1. Ollama and the model (about 5 GB)
winget install Ollama.Ollama
ollama pull qwen3:8b

# 2. Everything else — creates .venv, installs dependencies, runs the preflight checks
.\start-jarvis.bat
```

The first launch also needs the voice model:

```powershell
.venv\Scripts\python scripts\fetch_models.py            # Kokoro voice + wake word models
.venv\Scripts\python scripts\fetch_models.py --swedish  # optional Swedish voice
```

Check the machine at any time:

```powershell
.venv\Scripts\python -m jarvis --preflight       # report
.venv\Scripts\python -m jarvis --preflight --fix # also write the detected values into config.yaml
```

### Start it your way

| Command | What it does |
|---|---|
| `start-jarvis.bat` | the whole thing: voice, overlay, tray |
| `python -m jarvis` | same, from an activated venv |
| `python -m jarvis --no-ui` | console only, no overlay |
| `python -m jarvis --text` | type instead of talk — same brain, same tools |
| `python -m jarvis --say "Good evening, sir."` | TTS smoke test |
| `python -m jarvis --list-devices` | audio device names for `config.yaml` |
| `python -m jarvis --preflight` | environment doctor |
| `python -m jarvis --config other.yaml` | alternative configuration |

Autostart at logon (optional):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1
powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1
```

---

## 3. Talking to him

Say **"Hey Jarvis"**, wait for the chime, then speak. After he answers you have
**20 seconds** to keep talking without the wake word — each reply resets the window.
Talk over him and he stops mid-sentence and listens.

### Thirty things to say

**The house**
1. "Hey Jarvis." — *he answers "Sir?"*
2. "What time is it?"
3. "What's the date today?"
4. "How's the system?"
5. "How hot is the GPU?"
6. "How much disk space is left?"
7. "How long has this machine been up?"

**Apps and windows**
8. "Open Spotify."
9. "Open Chrome."
10. "Open VS Code."
11. "Close Spotify."
12. "Open YouTube."
13. "Open github dot com."

**Sound and media**
14. "Volume to forty percent."
15. "Turn it up."
16. "Mute."
17. "Pause the music."
18. "Next track."

**The world**
19. "What's the weather?"
20. "What's the weather in Gothenburg?"
21. "Search the web for the latest on the Artemis programme."

**Time**
22. "Set a timer for ten minutes."
23. "Set a timer for three minutes for the pasta."
24. "Remind me to call Erik at half past seven."
25. "What timers are running?"

**Files and memory**
26. "Find a file called invoice."
27. "Remember that my sister's birthday is the fourth of May."
28. "What do you remember about me?"

**The serious end**
29. "Run a PowerShell command to show the five biggest files in Downloads." → *"…say confirm."* → **"Confirm."**
30. "Lock the computer." / "Take a screenshot." / "Type out my email address."

Swedish works too — "Hey Jarvis" is still the wake phrase, but say *"Hur mår systemet?"*
and he answers in Swedish with the same manner.

---

## 4. Tools

| Tool | Tier | What it does |
|---|---|---|
| `get_time_date` | safe | time, date, weekday |
| `system_status` | safe | CPU, RAM, disk, uptime, GPU load and temperature |
| `open_app` / `close_app` | safe | the allowlist in `config.yaml`, fuzzy matched; close is never forced |
| `open_url` | safe | default browser |
| `web_search` | safe | DuckDuckGo via `ddgs`, keyless, summarised into one sentence |
| `weather` | safe | Open-Meteo, keyless, metric |
| `volume` / `media` | safe | pycaw and the media keys |
| `screenshot` | safe | saved to `Pictures\Jarvis` |
| `set_timer` / `set_reminder` / `list_timers` / `cancel_timer` | safe | spoken when due, even from idle |
| `find_file` | safe | Desktop, Documents, Downloads |
| `remember` / `forget` / `recall` | safe | `memory.json` |
| `lock_pc` | safe | Win+L |
| `type_text` | announced | types into the focused window after a one-second heads-up |
| `deep_think` | announced | "Give me a moment, sir." — then thinks properly |
| `run_powershell` | **guarded** | announced, confirmed out loud, 30 s timeout |
| `file_ops` | **guarded** | move / rename / delete — delete goes to `Jarvis Trash`, never straight out |
| `power` | **guarded** | sleep / restart / shutdown |

**Guarded** means he says exactly what he is about to do and waits for you to say
*confirm / yes / do it / kör*. Ten seconds of silence and it is cancelled.

**Always refused**, whatever you say and whatever tier: formatting drives, `diskpart`,
`bcdedit`, security and Defender registry keys, `Set-MpPreference`, turning off the
firewall, credential and vault access, wiping event logs, deleting under `System32`.
Refusals are logged.

`safety_mode: strict` in the config additionally guards `type_text` and `close_app`.

---

## 5. Configuration reference (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `language` | `auto` | `auto`, `en` or `sv`. Auto follows what you speak. |
| `assistant.wake_greeting` | `true` | say "Sir?" when woken |
| `assistant.acknowledge_phrase` | `"Sir?"` | what he says on waking |
| `assistant.startup_greeting` | `true` | time-of-day greeting at launch |
| `assistant.conversation_timeout` | `20` | seconds of follow-up without the wake word |
| `assistant.confirmation_timeout` | `10` | seconds to say "confirm" |
| `assistant.safety_mode` | `normal` | `normal` or `strict` |
| `assistant.max_tool_rounds` | `4` | tool loops per turn |
| `audio.input_device` / `output_device` | `null` | device name substring or index; `--list-devices` shows them |
| `audio.sample_rate` | `16000` | capture rate — the models expect 16 kHz |
| `audio.block_size` | `1280` | 80 ms frames |
| `audio.chime_volume` | `0.35` | how loud the chimes are |
| `audio.barge_in` | `true` | interrupt him by speaking |
| `audio.barge_in_speech_ms` | `400` | sustained speech needed to cut him off |
| `wake.sensitivity` | `0.5` | raise if the TV wakes him, lower if he misses you |
| `wake.cooldown` | `2.0` | seconds before the wake word can fire again |
| `vad.silence_ms` | `700` | end of utterance |
| `vad.max_utterance_s` | `15` | hard cap |
| `stt.model` | `auto` | `auto` picks by VRAM: ≥10 GB `medium`, ≥6 GB `small`, else `base` |
| `stt.device` / `compute_type` | `auto` | CUDA with a silent CPU int8 fallback |
| `brain.model` | `qwen3:8b` | any Ollama model with tool support |
| `brain.deep_model` | `null` | optional bigger model for `deep_think` |
| `brain.keep_alive` | `-1` | keep the weights resident forever |
| `brain.history_turns` | `12` | rolling conversation window |
| `tts.engine` | `kokoro` | `kokoro`, `piper` or `sapi` |
| `tts.voice` | `bm_george` | also `bm_lewis`, `bf_emma` |
| `tts.speed` | `1.0` | 0.8 is statelier, 1.2 is brisker |
| `ui.overlay` / `ui.tray` | `true` | the ring and the tray icon |
| `ui.position` | `null` | remembered when you drag the ring |
| `tools.default_city` | `Stockholm` | for `weather` with no city |
| `tools.powershell_timeout` | `30` | seconds |
| `tools.apps` | see file | the `open_app` allowlist — add your own |
| `logging.level` | `INFO` | `DEBUG` shows every frame decision |

---

## 6. Latency

The target is **under two seconds** from the end of your sentence to his first spoken word.
Every turn prints its own breakdown:

```
speech-end→text 380 ms · →first token 720 ms · →first audio 1140 ms
```

What makes it fast: the model is warmed at startup and pinned with `keep_alive: -1`,
the reply is streamed and split into sentences so speech starts on sentence one, and
Kokoro synthesises sentence two while sentence one is still playing.

If it feels slow: check `--preflight` for the CUDA verdict, drop `stt.model` to `base`,
and make sure nothing else is holding the GPU.

---

## 7. When something goes wrong

| Symptom | Cause and cure |
|---|---|
| "The language model isn't responding, sir." | Ollama is not running — `ollama serve`, then `ollama list` should show `qwen3:8b` |
| "I've lost the microphone, sir." | another app grabbed the device; check `--list-devices` and set `audio.input_device` |
| He never wakes | lower `wake.sensitivity` to 0.35; make sure the mic is the default device |
| He wakes at the television | raise `wake.sensitivity` to 0.6–0.7 |
| He interrupts himself | your speakers are bleeding into the mic — raise `audio.barge_in_speech_ms` or use headphones |
| No voice at all | `python -m jarvis --say "test"`; if silent, run `scripts\fetch_models.py` |
| Transcription is nonsense | `stt.model: small` at minimum; check the mic level in Windows sound settings |
| CUDA errors at startup | harmless — he falls back to CPU int8 and logs it once |

Everything — transcripts, tool calls with arguments, refusals and per-turn latency —
lands in `logs/jarvis.log`.

---

## 8. Development

```bash
python -m pytest tests/ -q     # the whole suite runs without a microphone, GPU or network
```

Layout:

```
jarvis/
  audio/    capture, playback, synthesized chimes
  wake/     openWakeWord
  stt/      Silero VAD + faster-whisper
  brain/    Ollama client, conversation, the turn loop, sentence splitting
  tts/      Kokoro / Piper / SAPI engines and the speaking queue
  tools/    registry, safety tiers, dispatcher, and the tools themselves
  ui/       overlay and tray
  core/     config, logging, state, memory, scheduler, latency, the assistant loop
prompts/    the system prompt
docs/       ARCHITECTURE.md — the module contract every part is built against
```

`docs/ARCHITECTURE.md` is the authoritative description of every module's API.
