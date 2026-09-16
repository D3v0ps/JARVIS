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

## 1b. The face

A small arc reactor sits in the corner of the screen. It is not a status light —
you can watch him work:

```
 ◉   LISTENING                                        
     › run a powershell command to show the five biggest files in downloads

 ◉   WORKING                                          
     › confirm
     ▸ run_powershell

 ◉   SPEAKING                                   1.42 s
     ✓ run_powershell  412 ms
     Done, sir. The largest is a 2.4 gigabyte video.
```

Motion carries the meaning, so you can read his state from across the room
without reading a word:

| State | What the ring does |
|---|---|
| idle | breathes slowly, dim, and the panel disappears entirely |
| listening | the eight coils follow your actual microphone level |
| working | a comet sweeps around the ring |
| speaking | waves leave the core, in time with his voice |
| paused | grey and still |

The panel shows what he heard, which tool is running, what it cost in
milliseconds, what he is saying as it streams, and the turn's total latency.
A guarded action puts **Say confirm** in amber at the bottom. Everything fades
out after nine seconds, leaving just the ring.

It is drawn with `UpdateLayeredWindow` and a premultiplied bitmap — a genuine
per-pixel alpha channel, which is what makes the glow soft and the edges clean.
Drag it anywhere; it remembers. Right-click for pause, quit, and a click-through
toggle if you would rather it never caught a click. On anything that is not
Windows it falls back to a plain tkinter ring.

See it on its own, before committing to a full session:

```powershell
.venv\Scripts\python -m jarvis --overlay-test
```

---

## 2. Install — double-click, that is the whole procedure

You do not need to know what PowerShell, pip or a virtual environment is. You need a
mouse and an internet connection.

> ### 1. Download this folder and unzip it, all of it.
> ### 2. Double-click **`Install-JARVIS.exe`**.
> ### 3. Say yes to the Windows prompt, then wait.
> ### 4. Double-click **JARVIS** on your desktop and say *"Hey Jarvis."*

That is it. The installer does the rest:

| It checks | And if it is missing |
|---|---|
| Windows version, RAM, free disk | warns you before it starts downloading 15 GB |
| Python 3.13 | installs it |
| Your graphics card | reads the VRAM and picks the model that fits — see below |
| Ollama | installs it and starts the service |
| The language model | pulls it (several GB — this is the slow part) |
| The Python environment | creates `.venv` and installs every dependency |
| The voice | downloads the Kokoro British voice and the wake-word model |
| `config.yaml` | writes in what it found, so nothing is left to guess |
| Shortcuts | puts **JARVIS** on your desktop and in the Start menu |
| Autostart | asks whether he should come online when you log in |

Run it twice and nothing breaks — every step checks whether it is already done, so it
also works as a repair tool. Everything it does is written to `logs\install.log`.

**Which model you get** is decided by your graphics card, because the model and Whisper
have to share the VRAM:

| Your VRAM | Model | Speech recognition |
|---|---|---|
| 16 GB or more | `qwen3:14b` — the sharpest that still leaves room | Whisper `medium` |
| 8–15 GB | `qwen3:8b` — the sweet spot for a sub-two-second answer | Whisper `small` |
| 4–7 GB | `qwen3:4b` | Whisper `base` |
| No NVIDIA card | `qwen3:4b` on the CPU — slower, but it works | Whisper `base` |

Want to override it: `Install-JARVIS.exe -Model qwen3:14b`.

### If Windows says "Windows protected your PC"

It will, the first time. The installer is not signed with a commercial certificate
(those cost money and this is your own software). Click **More info** → **Run anyway**.
If you would rather not, right-click `install.ps1` → **Run with PowerShell** does exactly
the same thing.

### The manual path, if you prefer it

```powershell
winget install Python.Python.3.13
winget install Ollama.Ollama
ollama pull qwen3:8b
.\start-jarvis.bat                                      # builds the venv and installs deps
.venv\Scripts\python scripts\fetch_models.py            # the voice
.venv\Scripts\python scripts\fetch_models.py --swedish  # optional Swedish voice
```

Check the machine at any time:

```powershell
.venv\Scripts\python -m jarvis --preflight       # report
.venv\Scripts\python -m jarvis --preflight --fix # also write the detected values into config.yaml
```

### Installer options

| Flag | Effect |
|---|---|
| `-Silent` | no questions, take every default |
| `-Swedish` | also download the Piper Swedish voice |
| `-Autostart` | register the logon task without asking |
| `-NoLaunch` | do not start JARVIS when it finishes |
| `-Model qwen3:14b` | force a specific model |
| `-SkipModels` | skip the voice download |

### Start it your way

Everything you need is a double-click. No terminal, ever.

| Double-click | What it does |
|---|---|
| **JARVIS.exe** | brings him online — this is the one you want |
| **Install-JARVIS.exe** | installs or repairs everything |
| **Check-JARVIS.bat** | reports what is installed and what is missing |
| **Test-Overlay.bat** | shows the arc reactor cycling through every state |
| **Test-Voice.bat** | speaks one line, so you know the voice works |
| **Get-Voice.bat** | downloads the British Kokoro voice and the wake-word models |

| Command | What it does |
|---|---|
| `start-jarvis.bat` | the same as JARVIS.exe, without the icon |
| `python -m jarvis` | from an activated venv |
| `python -m jarvis --no-ui` | console only, no overlay |
| `python -m jarvis --text` | type instead of talk — same brain, same tools |
| `python -m jarvis --say "Good evening, sir."` | TTS smoke test |
| `python -m jarvis --list-devices` | audio device names for `config.yaml` |
| `python -m jarvis --overlay-test` | cycle the overlay through every state for ten seconds |
| `python -m jarvis --preflight` | environment doctor |
| `python -m jarvis --config other.yaml` | alternative configuration |

To uninstall: delete the folder, then `winget uninstall Ollama.Ollama` if you want the
model gone too. JARVIS writes nothing outside his own folder except the two shortcuts and,
if you asked for it, the logon task (`scripts\uninstall-autostart.ps1` removes that).

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
| `find_file` | safe | Desktop, Documents, Downloads — and *inside* your PDFs and Word files, through the Windows index |
| `clipboard` | safe | reads what you copied, or merely selected on screen |
| `find_business` / `recall_business` | safe | a real business's phone number, from OpenStreetMap and the web |
| `dial_number` | **guarded** | dials on your Android over ADB, or hands an iPhone the number |
| `end_call` | safe | hangs up |
| `remember` / `forget` / `recall` | safe | `memory.json` |
| `lock_pc` | safe | Win+L |
| `type_text` | announced | types into the focused window after a one-second heads-up |
| `deep_think` | announced | "Give me a moment, sir." — then thinks properly |
| `run_powershell` | **guarded** | announced, confirmed out loud, 30 s timeout |
| `file_ops` | **guarded** | move / rename / delete — delete goes to `Jarvis Trash`, never straight out |
| `power` | **guarded** | sleep / restart / shutdown |

**Guarded** means he says exactly what he is about to do and waits for you to say
*confirm / yes / do it / kör*. Ten seconds of silence and it is cancelled.

**Never dialled**, whatever you say: emergency numbers. A misheard word must not be
able to summon an ambulance, so 112, 911, 999 and their relatives are refused before
the confirmation is even asked — and the refusal tells you to call them yourself,
because "that's beyond what I'm willing to do" is a useless thing to hear in an
emergency.

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
