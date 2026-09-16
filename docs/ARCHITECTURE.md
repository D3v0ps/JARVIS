# JARVIS — module contracts (authoritative)

Every module is written against THIS file. Do not invent alternative signatures, do not
rename anything, do not add a module that is not listed. If something here looks wrong,
implement it as written anyway and note the concern in your return value.

Project root: `/home/user/JARVIS`. Package: `jarvis/`. Target: **Windows 11, Python 3.13**.
Development/CI happens on Linux, so:

* **Never import a Windows-only module at module import time.** `pycaw`, `comtypes`,
  `pyttsx3`, `winreg`, `ctypes.windll`, `pygetwindow`, `sounddevice`, `openwakeword`,
  `faster_whisper`, `kokoro_onnx`, `torch`, `pystray`, `tkinter` — all of these are imported
  **inside the function or method that needs them**, or inside a `try/except ImportError`
  at first use. `import jarvis.<anything>` must succeed on a bare Linux box with only
  numpy + PyYAML + requests installed. This is enforced by a test.
* Guard platform behaviour with `sys.platform == "win32"` and degrade with a clear,
  in-character message rather than a traceback.
* Type hints everywhere. One responsibility per module. No module over ~400 lines.
* All code, comments, docstrings and log messages in **English**.

---

## 1. `jarvis/config.py`

```python
DEFAULT_CONFIG_PATH = Path("config.yaml")

class Config:
    """Dot-path access over the parsed config.yaml, with defaults baked in."""
    def __init__(self, data: dict, path: Path | None = None) -> None
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config"     # merges file over DEFAULTS
    def get(self, dotted: str, default: Any = None) -> Any        # e.g. cfg.get("brain.model")
    def set(self, dotted: str, value: Any) -> None                # in-memory
    def save(self) -> None                                        # writes back to self.path (YAML)
    def section(self, name: str) -> dict                          # e.g. cfg.section("stt")
    @property
    def path(self) -> Path
    @property
    def data(self) -> dict
```

`DEFAULTS` is a module-level dict mirroring the shipped `config.yaml` exactly, so a missing
key never raises. `load()` deep-merges the file on top of `DEFAULTS`. Unknown keys are kept.

Helpers in the same module:
```python
def resolve_language(cfg: Config) -> str | None   # "en" | "sv" | None when language: auto
def project_root() -> Path                        # directory containing config.yaml / the package parent
```

## 2. `jarvis/core/logging.py`

```python
def setup_logging(cfg: Config) -> logging.Logger
    """Root 'jarvis' logger: rotating file handler (cfg logging.file/max_bytes/backups)
    plus a colored console handler (ANSI; enable VT mode on Windows via colorama-free
    ctypes call, degrade silently). Idempotent — calling twice does not duplicate handlers."""

def get_logger(name: str) -> logging.Logger        # logging.getLogger(f"jarvis.{name}")

# Structured event helpers — every one of these also writes a normal log line.
def log_transcript(who: str, text: str, *, latency_ms: float | None = None) -> None
    # who in {"user", "jarvis"}
def log_tool_call(name: str, args: dict, result: "ToolResult", *, duration_ms: float) -> None
def log_refusal(reason: str, detail: str) -> None
def log_latency(label: str, ms: float) -> None
```

Console colors: user transcripts cyan, JARVIS replies green, tool calls yellow,
refusals/errors red, latency dim. Never print secrets. Never crash the caller.

## 3. `jarvis/core/state.py`

```python
class AssistantState(str, Enum):
    IDLE = "idle"; LISTENING = "listening"; THINKING = "thinking"
    SPEAKING = "speaking"; PAUSED = "paused"

class StateBus:
    """Thread-safe current-state holder with observers. UI subscribes to it."""
    def __init__(self) -> None
    @property
    def state(self) -> AssistantState
    def set(self, state: AssistantState) -> None          # no-op if unchanged; notifies observers
    def subscribe(self, callback: Callable[[AssistantState], None]) -> Callable[[], None]  # returns unsubscribe
    def set_text(self, text: str) -> None                 # optional caption for the overlay
    @property
    def text(self) -> str
```
Observer exceptions are caught and logged, never propagated.

## 4. `jarvis/core/memory.py`

```python
@dataclass
class MemoryEntry:
    text: str
    created: str            # ISO-8601 local time
    topic: str = ""

class Memory:
    def __init__(self, path: str | Path = "memory.json") -> None
    def load(self) -> None            # tolerant: missing/corrupt file -> empty, logged
    def save(self) -> None            # atomic write (tmp + os.replace)
    def remember(self, fact: str, topic: str = "") -> MemoryEntry
    def forget(self, topic: str) -> int        # case-insensitive substring match, returns count removed
    def facts(self) -> list[MemoryEntry]
    def as_prompt_block(self) -> str  # "" when empty, else "Known facts about the user:\n- ...\n"
    @property
    def user_name(self) -> str | None          # from a fact like "the user's name is X"
```
File format: `{"facts": [{"text": ..., "created": ..., "topic": ...}], "version": 1}`.

## 5. `jarvis/core/scheduler.py`

```python
@dataclass
class Job:
    id: str; due: float                  # time.time() epoch
    label: str; kind: str                # "timer" | "reminder"
    text: str

class Scheduler:
    """Single background thread, fires callback(job) when a job is due. Survives restart
    by persisting to `logs/schedule.json` (jobs already past due fire once at startup)."""
    def __init__(self, on_due: Callable[[Job], None], store: str | Path = "logs/schedule.json") -> None
    def start(self) -> None
    def stop(self) -> None
    def add_timer(self, minutes: float, label: str = "") -> Job
    def add_reminder(self, text: str, when: str) -> Job     # when: natural-ish, parsed by parse_when()
    def cancel(self, job_id: str) -> bool
    def pending(self) -> list[Job]

def parse_when(when: str, now: datetime | None = None) -> datetime | None
    """Understands: 'in 10 minutes', 'in 2 hours', 'at 19:30', '19:30', 'tomorrow at 08:00',
    'om 10 minuter', 'klockan 19:30', 'imorgon 08:00'. Returns None if unparseable."""
```

## 6. `jarvis/core/latency.py`

```python
class LatencyTracker:
    """One per turn. Milliseconds from end-of-speech to first spoken audio."""
    def __init__(self, logger=None) -> None
    def mark(self, label: str) -> float        # records and returns ms since start
    def start_turn(self) -> None               # call at end of speech
    def first_audio(self) -> float             # ms since start_turn, logged as the headline number
    def summary(self) -> str                   # "speech-end→text 420 ms · →first token 780 ms · →first audio 1180 ms"
    def marks(self) -> dict[str, float]
```

## 7. `jarvis/audio/`

### `devices.py`
```python
@dataclass
class DeviceInfo: index: int; name: str; max_input: int; max_output: int; default_samplerate: float
def list_devices() -> list[DeviceInfo]
def print_devices() -> None                    # used by `--list-devices`
def resolve_device(spec: str | int | None, *, kind: str) -> int | None   # kind: "input"|"output"
    # None -> None (system default). int -> validated. str -> case-insensitive substring match.
```

### `capture.py`
```python
class MicStream:
    """sounddevice.InputStream, float32 mono at cfg audio.sample_rate, pushed into a bounded
    queue. Never blocks the PortAudio callback: on overflow drop the oldest frame and count it."""
    def __init__(self, device=None, sample_rate=16000, block_size=1280, logger=None) -> None
    def start(self) -> None
    def stop(self) -> None
    def __enter__/__exit__
    def read(self, timeout: float = 1.0) -> np.ndarray | None   # (block_size,) float32 in [-1,1]
    def frames(self) -> Iterator[np.ndarray]                    # yields until stop()
    def flush(self) -> None                                     # drop everything buffered
    @property
    def dropped(self) -> int
    @property
    def running(self) -> bool
```

### `player.py`
```python
class Player:
    """Queued playback with instant stop() for barge-in. One worker thread, sounddevice
    OutputStream opened per clip (simplest reliable path on Windows/WASAPI)."""
    def __init__(self, device=None, logger=None, volume: float = 1.0) -> None
    def play(self, audio: np.ndarray, sample_rate: int, *, blocking: bool = False) -> None
    def stop(self) -> None            # clears the queue and cuts the current clip within ~50 ms
    def wait(self, timeout: float | None = None) -> None
    @property
    def is_playing(self) -> bool
    def on_first_audio(self, callback: Callable[[], None]) -> None  # fired once per play-after-idle
    def close(self) -> None
```
Audio arrays are float32 mono; resample with a small linear resampler if the clip's rate
differs from the stream's (`_resample(audio, src, dst)` helper is fine).

### `chimes.py`
Pure numpy, no files, no downloads (copyright rule).
```python
def wake_chime(sample_rate: int = 24000) -> np.ndarray      # ~150 ms, two RISING tones (A5→E6), soft attack/decay
def sleep_chime(sample_rate: int = 24000) -> np.ndarray     # ~200 ms, two DESCENDING tones
def confirm_chime(sample_rate: int = 24000) -> np.ndarray   # short single blip
def error_chime(sample_rate: int = 24000) -> np.ndarray     # low double buzz
def tone(freq, ms, sample_rate, *, volume=0.3, fade_ms=8) -> np.ndarray
```
All return float32 in [-1, 1], Hann-ish fades so nothing clicks.

## 8. `jarvis/wake/detector.py`
```python
class WakeWord:
    """openWakeWord with the bundled 'hey_jarvis' model, ONNX backend."""
    def __init__(self, model: str = "hey_jarvis", sensitivity: float = 0.5,
                 framework: str = "onnx", cooldown: float = 2.0, logger=None) -> None
    def ensure_models(self) -> None    # openwakeword.utils.download_models() on first run, idempotent
    def process(self, frame: np.ndarray) -> bool
        """frame: float32 mono 16 kHz. Returns True exactly once per detection (cooldown applied).
        openWakeWord wants int16 — convert internally."""
    def reset(self) -> None
    @property
    def last_score(self) -> float
    @property
    def available(self) -> bool        # False if the library/model could not be loaded
```

## 9. `jarvis/stt/`

### `vad.py`
```python
class SpeechSegmenter:
    """Silero VAD over 16 kHz float32 frames. Feed frames, get a finished utterance."""
    def __init__(self, threshold=0.5, silence_ms=700, min_speech_ms=250,
                 max_utterance_s=15, pre_roll_ms=300, sample_rate=16000, logger=None) -> None
    def reset(self) -> None
    def push(self, frame: np.ndarray) -> np.ndarray | None
        """Returns the full utterance (float32, pre-roll included) when the end of speech is
        reached or the hard cap is hit; otherwise None."""
    @property
    def speaking(self) -> bool
    @property
    def available(self) -> bool     # False -> caller falls back to EnergySegmenter
    @property
    def last_prob(self) -> float

class EnergySegmenter:
    """RMS-threshold fallback with the same push()/reset()/speaking interface, used when
    silero-vad is missing or fails to load. Calibrates noise floor from the first ~500 ms."""
```
Silero is loaded lazily: try `from silero_vad import load_silero_vad` first, then
`torch.hub.load('snakers4/silero-vad')`, else mark unavailable. Silero wants 512-sample
chunks at 16 kHz — buffer internally, do not assume the caller's block size.

### `transcriber.py`
```python
@dataclass
class Transcript:
    text: str; language: str; duration_s: float; latency_ms: float

class Transcriber:
    """faster-whisper. CUDA first, CPU int8 fallback without drama."""
    def __init__(self, model: str = "auto", device: str = "auto", compute_type: str = "auto",
                 beam_size: int = 1, language: str | None = None, vram_gb: float | None = None,
                 logger=None) -> None
    def load(self) -> None          # blocking; safe to call twice
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript
    @property
    def ready(self) -> bool
    @property
    def device_in_use(self) -> str

def pick_model(vram_gb: float | None) -> str      # >=10 -> "medium", >=6 -> "small", else "base"
def add_cuda_dll_paths(logger=None) -> None
    """Windows: os.add_dll_directory() for nvidia/cublas/bin and nvidia/cudnn/bin inside the
    active venv's site-packages, so faster-whisper finds the CUDA libs. No-op elsewhere."""
```

## 10. `jarvis/brain/`

### `sentences.py`
```python
class SentenceSplitter:
    """Accumulates streamed tokens and yields complete sentences for the TTS queue."""
    def __init__(self, min_chars: int = 12) -> None
    def feed(self, text: str) -> list[str]     # returns 0..n finished sentences
    def flush(self) -> str                     # the trailing fragment ("" if none)
    def reset(self) -> None

def clean_for_speech(text: str) -> str
    """Strip markdown (**, *, #, `, lists, links -> label), emojis, <think> blocks,
    collapse whitespace, expand a few symbols (%, &, /) for natural speech."""
```

### `ollama_client.py`
```python
@dataclass
class ChatDelta:
    kind: str                      # "token" | "tool_calls" | "done" | "error"
    text: str = ""
    tool_calls: list[dict] | None = None      # [{"name": str, "arguments": dict}]
    message: dict | None = None               # the assembled assistant message on "done"
    error: str = ""

class OllamaClient:
    def __init__(self, host: str, model: str, *, keep_alive: int | str = -1, think: bool = False,
                 num_ctx: int = 8192, temperature: float = 0.6, timeout: int = 120, logger=None)
    def available(self) -> bool                  # GET /api/tags, short timeout, never raises
    def has_model(self, name: str | None = None) -> bool
    def warm(self) -> None                       # tiny /api/chat to load weights, keep_alive applied
    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None, *,
                    model: str | None = None, think: bool | None = None) -> Iterator[ChatDelta]
        """POST /api/chat with stream=True, NDJSON lines. Yields token deltas as they arrive,
        then a tool_calls delta if the model asked for tools, then done. Network errors become
        a single kind='error' delta — never an exception."""
```
Payload shape: `{"model":…, "messages":…, "tools":…, "stream":true, "think":false,
"keep_alive":-1, "options":{"num_ctx":…, "temperature":…}}`. `think` is only sent when the
server reports support — send it always but tolerate a 400 by retrying once without it.

### `conversation.py`
```python
class Conversation:
    """Rolling history + system prompt assembly."""
    def __init__(self, system_prompt_path: str | Path, memory: Memory, *, history_turns: int = 12,
                 language: str | None = None, logger=None) -> None
    def system_message(self) -> dict     # role system: prompt + memory block + current date/time +
                                         # a language line when cfg language is forced
    def messages(self) -> list[dict]     # [system] + trimmed history
    def add_user(self, text: str) -> None
    def add_assistant(self, text: str, tool_calls: list | None = None) -> None
    def add_tool_result(self, name: str, content: str) -> None   # role "tool"
    def trim(self) -> None
    def clear(self) -> None
    @property
    def turns(self) -> int
```
Trimming counts user+assistant pairs, never drops the system message, and never leaves a
dangling tool result without its assistant tool_calls message.

### `brain.py`
```python
@dataclass
class TurnResult:
    reply: str; tool_calls: list[str]; cancelled: bool = False; error: str = ""

class Brain:
    """One voice turn: stream -> sentences -> TTS, executing tools in between."""
    def __init__(self, client: OllamaClient, conversation: Conversation, dispatcher: "Dispatcher",
                 *, max_tool_rounds: int = 4, logger=None, latency: LatencyTracker | None = None)
    def turn(self, user_text: str, on_sentence: Callable[[str], None],
             should_stop: Callable[[], bool] = lambda: False) -> TurnResult
        """Loop: chat_stream -> feed SentenceSplitter -> on_sentence(sentence) for each finished
        sentence (that is what makes speech start before the model is done). If the model returns
        tool_calls: execute each through the dispatcher, append results, go round again
        (max_tool_rounds). should_stop() true -> abandon the stream, cancelled=True."""
    def say_directly(self, text: str, on_sentence) -> None   # bypass the model (greetings, errors)
```

## 11. `jarvis/tools/`

### `base.py`
```python
class Tier(str, Enum):
    SAFE = "safe"; ANNOUNCED = "announced"; GUARDED = "guarded"

@dataclass
class ToolResult:
    ok: bool
    summary: str                 # ONE sentence, this is what the model sees/speaks
    detail: str = ""             # full output -> log only, never spoken
    data: dict | None = None
    refused: bool = False
    @classmethod
    def fail(cls, summary: str, detail: str = "") -> "ToolResult"
    @classmethod
    def refuse(cls, summary: str, detail: str = "") -> "ToolResult"

@dataclass
class ToolContext:
    config: Config
    memory: Memory
    logger: logging.Logger
    speak: Callable[[str], None]              # speak a line right now (blocking until queued)
    confirm: Callable[[str], bool]            # announce + wait for a spoken confirmation
    notify: Callable[[str], None]             # speak even from idle (timers firing)
    scheduler: "Scheduler"
    state: StateBus

@dataclass
class ToolSpec:
    name: str; description: str; parameters: dict; tier: Tier
    func: Callable[[ToolContext, dict], ToolResult]
    announce: str | None = None      # format string, e.g. "About to run PowerShell: {command}"
    def to_ollama(self) -> dict       # {"type":"function","function":{...}}
```

### `registry.py`
```python
REGISTRY: dict[str, ToolSpec]
def tool(name: str, description: str, parameters: dict, tier: Tier = Tier.SAFE,
         announce: str | None = None): ...        # decorator, registers on import
def get(name: str) -> ToolSpec | None
def all_specs() -> list[ToolSpec]
def ollama_tools() -> list[dict]
def load_all() -> None      # imports every jarvis.tools.*_tools module exactly once
```

### `safety.py`
```python
BLOCKLIST_PATTERNS: list[tuple[str, str]]     # (regex, human reason)
    # format/diskpart/bcdedit, reg add|delete on HKLM\SYSTEM\CurrentControlSet\Services\
    # or \SOFTWARE\Policies\Microsoft\Windows Defender, Set-MpPreference, Add-MpPreference,
    # netsh advfirewall set ... off, Set-NetFirewallProfile -Enabled False, cmdkey, vaultcmd,
    # Get-Credential dumps, rm/del/Remove-Item under System32 or C:\Windows,
    # Stop-Service WinDefend, shutdown from a raw shell, vssadmin delete shadows, cipher /w
def check_blocked(text: str) -> str | None        # returns the reason when blocked
def effective_tier(spec: ToolSpec, cfg: Config) -> Tier   # strict mode upgrades ANNOUNCED->GUARDED
                                                          # and close_app/type_text -> GUARDED
CONFIRM_WORDS = {"confirm", "yes", "yeah", "do it", "go ahead", "proceed", "affirmative",
                 "kör", "ja", "gör det", "kör på", "absolut"}
CANCEL_WORDS  = {"no", "cancel", "stop", "abort", "nej", "avbryt", "stopp"}
def is_confirmation(text: str) -> bool
def is_cancellation(text: str) -> bool
```

### `dispatcher.py`
```python
class Dispatcher:
    def __init__(self, ctx: ToolContext, logger=None) -> None
    def execute(self, name: str, args: dict) -> ToolResult
        """1. unknown tool -> ToolResult.fail. 2. blocklist check over name+args -> refuse,
        log_refusal, in-character summary. 3. effective tier: GUARDED -> ctx.confirm(announcement)
        and abort with 'Very well, sir. Cancelled.' on no; ANNOUNCED -> ctx.speak(announcement).
        4. run with timing, catch every exception -> ToolResult.fail. 5. log_tool_call."""
    def tools_payload(self) -> list[dict]
```

### tool modules (each `@tool`-decorated, all returning ToolResult with a ONE-sentence summary)
* `system_tools.py` — `get_time_date`, `system_status` (psutil CPU/RAM/disk/uptime +
  nvidia-smi GPU util/temp via subprocess, absent GPU handled), `lock_pc`
  (`rundll32.exe user32.dll,LockWorkStation`), `screenshot(name?)` (Pillow ImageGrab ->
  Pictures\Jarvis), `volume(action, level?)` (pycaw; action set|up|down|mute|unmute),
  `media(action)` (play_pause|next|previous|stop via keybd_event VK_MEDIA_* through ctypes).
* `app_tools.py` — `open_app(name)` (fuzzy match against cfg tools.apps with difflib, then
  `os.startfile` / `start` fallback; URLs open in the browser), `close_app(name)` (graceful:
  pygetwindow close, else `taskkill /IM x.exe` **without** /F), `open_url(url)` (webbrowser,
  https:// prefixed when missing).
* `web_tools.py` — `web_search(query)` (`ddgs`, top `tools.search_results`, ONE spoken
  sentence, titles+links in detail), `weather(city?)` (Open-Meteo geocoding + current +
  today's min/max, keyless, °C, m/s; default city from config).
* `file_tools.py` — `find_file(name?, contains?, modified_since?)` — `name` is no longer
  required, and `contains` queries the Windows Search index, which has already read
  inside PDFs and .docx. (walk Desktop/Documents/Downloads, depth-limited,
  top 5 hits, newest first), `file_ops(action, source, destination?)` GUARDED
  (move|delete|rename; delete goes to a `Jarvis Trash` folder rather than unlinking; refuses
  paths under C:\Windows or outside the user profile).
* `memory_tools.py` — `remember(fact)`, `forget(topic)`.
* `timer_tools.py` — `set_timer(minutes, label?)`, `set_reminder(text, when)`, plus
  `list_timers` and `cancel_timer(label_or_id)`.
* `input_tools.py` — `type_text(text)` ANNOUNCED: one-second heads-up then types into the
  focused window (ctypes SendInput/keybd_event with unicode scancodes; no third-party dep).
* `think_tools.py` — `deep_think(question)` ANNOUNCED: says "Give me a moment, sir.", then a
  non-streaming call with `think: true` (or `brain.deep_model` when set), returns a
  two-sentence answer. Needs a reference to the OllamaClient — accept it via
  `set_deep_client(client)` module function called from wiring.
* `shell_tools.py` — `run_powershell(command)` GUARDED (`powershell -NoProfile
  -NonInteractive -ExecutionPolicy Bypass -Command`, timeout from config, summary = first
  meaningful line or "N lines of output", full stdout/stderr to the log),
  `power(action)` GUARDED (sleep|restart|shutdown; sleep via
  `rundll32.exe powrprof.dll,SetSuspendState 0,1,0`, restart/shutdown via `shutdown /r /t 5`
  / `/s /t 5`).

## 12. `jarvis/tts/`

### `engine.py`
```python
class TTSEngine(Protocol):
    sample_rate: int
    name: str
    def synthesize(self, text: str, *, language: str | None = None) -> tuple[np.ndarray, int]
    def available(self) -> bool
    def close(self) -> None

def create_engine(cfg: Config, logger=None) -> TTSEngine
    """Primary from cfg.tts.engine; on failure fall back through cfg.tts.fallback and finally
    to NullEngine (logs, returns silence) so the assistant never dies for want of a voice."""
```
Implementations: `kokoro_tts.KokoroTTS` (kokoro-onnx, voice from config, speed, 24 kHz),
`piper_tts.PiperTTS` (Swedish, model path from config), `sapi_tts.SapiTTS` (pyttsx3 ->
temp wav -> numpy; Windows only), `engine.NullEngine`.

### `speaker.py`
```python
class Speaker:
    """Sentence queue -> synth thread -> Player. Speech starts on sentence 1 while the model
    is still generating sentence 2."""
    def __init__(self, engine: TTSEngine, player: Player, state: StateBus, *,
                 language: str | None = None, logger=None, latency: LatencyTracker | None = None)
    def enqueue(self, sentence: str) -> None
    def say(self, text: str, *, blocking: bool = True) -> None    # split + enqueue + optionally wait
    def play_chime(self, audio: np.ndarray, sample_rate: int) -> None
    def stop(self) -> None            # barge-in: clear queue, stop player, drop in-flight synth
    def wait(self, timeout: float | None = None) -> None
    @property
    def is_speaking(self) -> bool
    def close(self) -> None
```
Sets `StateBus` to SPEAKING while audio is out and reports the first-audio moment to the
LatencyTracker.

## 13. `jarvis/ui/`

### `overlay.py`
```python
class Overlay:
    """Frameless always-on-top tkinter arc-reactor ring. MUST run on the main thread;
    other threads call `on_state()` which marshals via root.after()."""
    def __init__(self, cfg: Config, state: StateBus, *, on_quit=None, on_toggle_pause=None, logger=None)
    def start(self) -> None      # builds the window and subscribes to the StateBus
    def run(self) -> None        # tk mainloop (blocks — main thread only)
    def stop(self) -> None
    def is_alive(self) -> bool
```
180 px transparent-ish window, draggable (position saved to config on release), animated
ring: dim breathing when idle, bright pulsing when listening, rotating arc when thinking,
waveform-ish pulse when speaking. Right-click closes. Uses `-topmost`, `overrideredirect`,
`-transparentcolor` on Windows, degrading gracefully when unsupported.

### `tray.py`
```python
class Tray:
    """pystray icon with Pause/Resume and Quit. Runs in its own thread."""
    def __init__(self, state: StateBus, *, on_quit=None, on_toggle_pause=None, logger=None)
    def start(self) -> None
    def stop(self) -> None
```
Icon drawn with Pillow (a small glowing ring), colored by state.

## 14. `jarvis/core/assistant.py` — the loop

```python
class Assistant:
    def __init__(self, cfg: Config, *, text_mode: bool = False) -> None
        """Builds everything: logging, memory, state bus, scheduler, mic, player, chimes,
        wake word, segmenter, transcriber, TTS engine, speaker, ollama client, conversation,
        tool registry + dispatcher, latency tracker."""
    def start(self) -> None          # warm the model, greet, start the audio thread
    def run_forever(self) -> None    # blocks in the audio loop (used when the overlay is off)
    def run_in_thread(self) -> threading.Thread
    def stop(self) -> None
    def pause(self) -> None
    def resume(self) -> None
    def handle_text(self, text: str) -> str     # text-mode turn, returns the reply (no audio)
```

Loop shape:
```
IDLE:      read frame -> wake.process(frame)?  -> chime + optional "Sir?" -> LISTENING
LISTENING: segmenter.push(frame) -> utterance -> THINKING
THINKING:  transcribe -> (empty? back to idle) -> brain.turn(..., on_sentence=speaker.enqueue)
SPEAKING:  handled by Speaker; barge-in watcher runs the mic through the segmenter and calls
           speaker.stop() on sustained speech
AFTER:     conversation window of cfg assistant.conversation_timeout seconds: any utterance is
           treated as a follow-up with no wake word; each reply resets the window; on timeout
           play the sleep chime -> IDLE
```
Guarded confirmation (`ctx.confirm`) reuses the same capture path: speak the announcement,
then listen for up to `assistant.confirmation_timeout` seconds; transcribe; `is_confirmation`
-> True; anything else / timeout -> speak "Very well, sir. Cancelled." and return False.
Timers firing from idle go through `notify()` which speaks regardless of state.
Every component is optional: if the mic, wake word, STT or TTS cannot start, log it, say it
in character if speech is possible, and keep the rest alive (text mode still works).

## 15. `jarvis/__main__.py`
```
python -m jarvis                 # full voice assistant
python -m jarvis --text          # terminal chat against the same brain + tools (no audio)
python -m jarvis --list-devices  # audio devices
python -m jarvis --preflight     # environment doctor (jarvis/preflight.py)
python -m jarvis --say "text"    # TTS smoke test
python -m jarvis --no-ui         # console only
python -m jarvis --config path   # alternative config file
```
Preflight sub-flags, used by `install.ps1` so the installer never has to edit YAML itself:
```
python -m jarvis --preflight --fix                  # write detected values into config.yaml
python -m jarvis --preflight --fix --set-model qwen3:14b --set-whisper medium
```
`--set-model` writes `brain.model`, `--set-whisper` writes `stt.model`; both imply `--fix`.
Unknown flags must not crash the installer - print the usage and exit 2.

## 16. `jarvis/preflight.py`
```python
@dataclass
class CheckResult: name: str; ok: bool; detail: str; fix: str = ""
def run_checks(cfg: Config, *, fix: bool = False) -> list[CheckResult]
def detect_vram() -> tuple[float | None, str | None]     # nvidia-smi --query-gpu=memory.total,name
def apply_fixes(cfg: Config, *, model: str | None = None, whisper: str | None = None) -> list[str]
    """Write system.vram_gb, system.gpu_name and, when given, brain.model and stt.model
    into config.yaml. Returns a list of human-readable changes. Never raises - a read-only
    config file becomes a warning, because the installer must still finish."""
def main(argv=None) -> int
```
`main()` accepts `--fix`, `--set-model NAME`, `--set-whisper SIZE` and `--json`.
Exit code 0 when everything essential passed, 1 when something essential is missing
(no Python deps, no Ollama daemon, no model), 2 on bad arguments. A missing *optional*
piece (no GPU, no Swedish voice) is a warning, not a failure - the installer treats a
non-zero exit as "tell the user" and carries on.
Checks: Python version, venv active, required packages importable, ollama binary + daemon +
`qwen3:8b` present, GPU/VRAM (writes `system.vram_gb` + `system.gpu_name` into config.yaml),
STT model choice, Kokoro model files present (points at `scripts/fetch_models.py`),
openWakeWord models present, audio input/output devices, logs/ writable.
Prints a tidy table and returns a non-zero exit code when something essential is missing.

## 17. Tests (`tests/`)
pytest, no hardware, no network, Linux-friendly. `test_imports.py` (every module imports),
`test_config.py`, `test_memory.py`, `test_scheduler.py` (incl. `parse_when`), `test_sentences.py`,
`test_safety.py` (blocklist + confirmation words), `test_registry.py` (every tool has a valid
JSON schema and a tier), `test_dispatcher.py` (fake ctx: guarded asks, cancel path, blocked
path, exception path), `test_ollama_client.py` (streaming parsed from a fake NDJSON HTTP
server on 127.0.0.1), `test_conversation.py` (trimming keeps tool pairs intact),
`test_chimes.py` (shape/range/no clipping), `test_vad_energy.py`, `test_brain_turn.py`
(fake client + fake dispatcher: sentences stream out in order, tool round-trip works).

---

# Part two — the features added after the first real week of use

Same rules as above: exact signatures, no Windows-only imports at module load, every
module testable on a bare Linux box, nothing over ~400 lines.

## 18. `jarvis/brain/reflex.py` — the command grammar underneath the model

Nine utterances in ten are imperatives: "volume to forty", "set a ten minute timer",
"open Spotify". Each costs a full model turn today, and each is a chance for the model
to narrate instead of act. A template match dispatches those in ~200 ms through the
*same* dispatcher, so tiers and the blocklist still hold.

```python
@dataclass
class Reflex:
    tool: str                    # the registered tool name
    arguments: dict              # already coerced to the schema's types
    utterance: str               # what was matched
    template: str                # which template matched, for the log
    replies: list[str]           # canned confirmations to choose between

class ReflexMatcher:
    def __init__(self, path: str | Path = "prompts/sentences.yaml", *,
                 language: str | None = None, logger=None) -> None
    def match(self, text: str) -> Reflex | None
        """Whole-utterance match only. Returns None for anything with a conjunction,
        a question, or trailing words the template did not consume - "open Spotify and
        tell me the weather" must fall through to the model."""
    def reply_for(self, reflex: Reflex, result) -> str
        """One of `replies`, varied between calls so the fast path does not sound
        like a recording. Uses the ToolResult's summary when the template says {result}."""
    @property
    def available(self) -> bool
    def templates(self) -> int
```

`prompts/sentences.yaml` holds, per tool, a list of templates in English and Swedish
with `{named}` slots, plus `replies`. Slots may declare `type: int|float|str` and a
`values:` map for spoken numbers and app aliases. Matching is pure Python regex built
from the templates at load - no new dependency - and is case- and punctuation-insensitive.

Wired in `core/assistant.py::_handle_utterance`, between transcription and
`brain.turn()`, and ONLY for a fresh utterance: never in `_Mode.CONFIRMING`, never
inside a tool round. A GUARDED tool matched by a reflex still goes through
`Dispatcher.execute`, so it still asks. Config: `assistant.reflex: true`.

## 19. `jarvis/tools/places_tools.py` — finding a real business's number

```python
@tool("find_business", tier=SAFE)   # name, near?
@tool("recall_business", tier=SAFE) # what was looked up before
```
Order: Nominatim `search?format=jsonv2&extratags=1` (extratags carries `phone`,
`contact:phone`, `opening_hours`), then Overpass for a category near a point, then the
existing `ddgs` search plus a `requests.get` on the business's own site with a phone
regex. Roughly 40 % of Swedish dentists and hairdressers carry a phone tag in OSM, so
the fallback carries most lookups - say so in the log, not out loud.

Both OSM endpoints need a descriptive `User-Agent` and hard throttling: Nominatim is
1 request/second, Overpass gives anonymous callers two slots. A module-level rate
limiter enforces it. Never scrape hitta.se or eniro.se - better data, against terms.

Numbers are normalised to E.164 with `phonenumbers` (offline), region from
`tools.home_region` (default `SE`). Results cache to `places.json` beside `memory.json`
so a barber is looked up once, ever.

## 20. `jarvis/tools/telephony_tools.py` — dialling, honestly

```python
@tool("dial_number", tier=GUARDED)   # number, who?
@tool("end_call", tier=SAFE)
def adb_available() -> bool
def call_state() -> str              # "idle" | "ringing" | "offhook" | "unknown"
```
Android over ADB is the real path: `adb shell am start -a android.intent.action.CALL
-d tel:+46...`. Some builds refuse it for the `shell` uid, so fall back to
`ACTION_DIAL` plus `input keyevent KEYCODE_CALL`, and log which fired. `call_state()`
reads `dumpsys telephony.registry`. No device, or an iPhone: fall back to
`os.startfile("tel:+46...")`, which hands the number to whatever Windows has
registered - usually Phone Link - and say plainly that the call must be started on the
phone. Never claim a call was placed that was not.

While a call is up, `Assistant` pauses: no wake word, no speaking. That is the
etiquette, and it is enforced in the loop rather than in the prompt.

**JARVIS never speaks on the call.** There is no Windows API to put audio into one,
and the workarounds degrade the voice to 8 kHz and feed the speakers back into the
line. He looks the number up, drafts what to say onto the HUD, dials, and goes quiet.

## 21. Brief mode, ducking and the boot sweep

* `tools/base.py`: `ToolSpec.speak_result: bool = True`. False means the dispatcher
  plays an earcon and the turn ends - which removes an entire model round trip from
  the latency budget. Config `assistant.brief_mode` turns it on globally.
* `audio/chimes.py` gains `ack()`, `done()`, `awaiting()`, sharing the wake chime's
  A5→E6 motif. Binary only - succeeded, refused, waiting. Never encode *which* tool ran
  in a tone; abstract earcons are measurably the worst way to tell things apart.
* `audio/ducking.py`:
  ```python
  class Ducker:
      def __init__(self, state: StateBus, *, level: float = 0.2, ramp_ms: int = 120,
                   store: str | Path = "logs/ducking.json", logger=None)
      def start(self) -> None      # subscribes to the state bus
      def stop(self) -> None       # restores, always
      def duck(self) -> None
      def restore(self) -> None
      @property
      def available(self) -> bool
  ```
  `pycaw` is already a dependency. Snapshot every session's `SimpleAudioVolume`, ramp
  to `level` on LISTENING, restore on IDLE in a `finally` **and** from the persisted
  snapshot at startup, so a crash never leaves Spotify at 20 % forever. Exclude
  JARVIS's own PID. This is not decoration: it fixes transcription accuracy, wake
  misses and false barge-in over music in one move.
* `ui/reactor.py`: `render(..., boot_t: float | None = None)` animates an iris-open,
  inner radius and coil count 0→8 over ~1.2 s. Hard-capped at three seconds, and the
  wake-word thread starts first. No continuous hum - it is fan noise by week two.

## 22. `jarvis/tools/clipboard_tools.py` and search inside files

```python
@tool("clipboard", tier=SAFE)   # action: read | summarise | translate | explain
```
`win32clipboard` for `CF_UNICODETEXT`, `CF_HDROP` and `CF_DIB`. For text merely
selected on screen, synthesise Ctrl+C with the `SendInput` code already in
`input_tools.py`, read, then restore the previous contents; detect "nothing was
selected" by comparing `GetClipboardSequenceNumber` before and after. Wrap
`OpenClipboard` in a short retry loop, and honour
`ExcludeClipboardContentFromMonitorProcessing` and `CanIncludeInClipboardHistory` or a
password manager's payload will one day end up in `logs/jarvis.log`. Truncate hard.

`file_tools.find_file` gains `contains` and `modified_since` rather than a new tool.
Query the index Windows already maintains through `pywin32`:
`ADODB.Connection` with `Provider=Search.CollatorDSO`, `SELECT TOP 10
System.ItemPathDisplay FROM SYSTEMINDEX WHERE CONTAINS(...) AND SCOPE='file:...'`. It
reads inside PDFs and .docx because the index already parsed them. Keep the `fnmatch`
walk as the fallback, and say "nothing indexed matches" rather than "that file does not
exist" when the index is rebuilding.

## 23. `jarvis/remote/` — the phone as a push-to-talk satellite

```
remote/__init__.py
remote/server.py     RemoteServer(cfg, assistant, logger)  .start() .stop() .url
remote/session.py    per-connection state, pairing, rate limits
remote/static/index.html   the whole PWA: one file, no build step
```
Flask + flask-sock + waitress - threaded WSGI, matching this codebase's thread model.

The wire format is the decisive detail: **not `MediaRecorder`** (iOS gives AAC, Android
gives webm/opus, and the server would need a decoder). The page uses
`@ricky0123/vad-web`, whose `onSpeechEnd` hands back a **Float32Array of 16 kHz mono** -
byte for byte what `Transcriber.transcribe(audio, sample_rate=16000)` already takes.
Same Silero model as the desk.

A remote turn goes onto the assistant's existing `_work` queue, never around it: one
GPU, one Whisper, one Ollama. `Brain.turn`'s `on_sentence` pushes each finished
sentence down the socket so the phone starts speaking on sentence one, exactly like the
desk. A second socket carries `StateBus` changes so the page shows the same states.

Security, because this thing runs PowerShell:
* `remote.enabled: false` by default. An installer must never open a socket.
* Bind to loopback and let **Tailscale Serve** be the only door
  (`scripts/enable_phone.py`, run by `Enable-Phone.bat`: `tailscale serve --bg
  --https=443 http://127.0.0.1:8765`). Serve terminates TLS with a certificate for
  `<machine>.<tailnet>.ts.net`, reachable only from the tailnet - and HTTPS is what
  makes a phone browser hand over the microphone at all. The address is recorded in
  `remote.url` so the console banner and the phone agree. Binding the Tailscale IP
  directly is the HTTP fallback; never `0.0.0.0`.
* `safety.remote_tier()`: GUARDED tools are **refused** over the phone unless
  `remote.allow_guarded` is true - spoken confirmation is a weak control when the
  attacker holds the microphone. The blocklist is unchanged.
* A one-time pairing code printed to the console and shown on the ring, then a signed
  HttpOnly cookie. Every remote turn logged with the device name.
* A routines strip: named macros from `config.yaml`, each a sequence of existing tool
  calls, run through the dispatcher. A fixed allowlist is the security-correct way to
  expose real power remotely.

iOS specifics that must be handled: create and `resume()` the AudioContext inside the
push-to-talk tap handler or output is silent; offer Add to Home Screen; AudioWorklet is
fine from iOS 14.1. Always-on wake word in the browser is not attempted - iOS suspends
the tab and it drains the battery. Push-to-talk is the design, not a compromise.

---

## 24. `jarvis/desk/` — the app on the desk

Everything up to here gave the desk a *ring*. The window behind the ring was still a
console: the banner, the log lines, the pairing code, the tracebacks. That is a
terminal with a nice ornament floating over it, and a terminal is not an app.

This section is the app: **one window, drawn as a web page, hosted natively, with no
console anywhere.** The page is the same design language as the phone - the same
reactor geometry, the same cards, the same palette - because the phone and the desk
should be the same product seen from two places.

```
jarvis/desk/__init__.py          lazy exports, like jarvis/ui
jarvis/desk/bus.py               DeskBus - the event fan-out and its replay buffer
jarvis/desk/server.py            DeskServer - loopback Flask + flask-sock
jarvis/desk/window.py            DeskWindow - the native host and its fallbacks
jarvis/desk/static/desk.html     the app itself, one file, no build step
jarvis/core/telemetry.py         snapshot() - one reading of the machine
```

### 24.1 Why a web page in a native window

tkinter cannot draw this: no anti-aliasing, no blur, no transitions, no flexbox, and a
`Text` widget that looks like 1996. The arc reactor already had to leave tkinter for a
per-pixel-alpha bitmap (§ on `layered.py`); a whole application window would have to
leave it too, and re-implementing card layout, scrolling and countdown animation over a
Pillow blitter is a month of work to land somewhere worse than a stylesheet.

**Edge WebView2 is already on every Windows 11 machine.** `pywebview` binds to it in a
few hundred kilobytes and gives a frameless, resizable, always-on-top-capable window
that renders the page we have already designed and tested. When it is missing, Edge's
own app mode (`msedge --app=<url>`) shows the same page in a window with no browser
chrome. When even that fails, the ring alone remains and JARVIS behaves exactly as he
does today. Three hosts, one page, and the assistant never depends on any of them.

This is a stated stack addition, not a swap: tkinter keeps the fallback ring, the
layered overlay keeps the reactor, `pywebview` is optional and lazily imported, and
`python -m jarvis --no-window` turns the whole section off.

### 24.2 The transport, and why there is a socket at all

`pywebview` can bridge Python and JavaScript directly (`js_api`) with no socket, which
is tempting for a local app. The Edge fallback cannot: a browser needs a URL. Rather
than write the page twice, the desk runs **one loopback server** and both hosts point at
it. It also means the window can be closed and reopened, or opened for the first time an
hour into a session, and still see the conversation so far.

```python
class DeskServer:
    def __init__(self, cfg, assistant, logger) -> None
    @property
    def bus(self) -> DeskBus
    @property
    def url(self) -> str            # http://127.0.0.1:<port>/?t=<ticket>
    @property
    def running(self) -> bool
    def start(self) -> bool         # False, never an exception, when it cannot
    def stop(self) -> None
    def create_app(self) -> Any     # Flask; imported lazily, for the tests
```

Rules the implementation must keep:

* **Loopback only, ephemeral port.** Bind `127.0.0.1` with port `0` and read back what
  the OS gave. `0.0.0.0` is refused the way `RemoteServer` refuses it.
* **A single-use ticket, then a cookie.** The URL carries `?t=<32 urlsafe bytes>`,
  generated per run and never written to disk. `GET /` exchanges it for a session
  cookie (`HttpOnly`, `SameSite=Strict`, no `Secure` - this is `http://127.0.0.1`) and
  **burns it**; a second use is refused. The ticket only exists to survive one command
  line (Edge's) and one process list, so it expires after 120 s as well.
* **Every request must come from a loopback peer** and carry `Host: 127.0.0.1:<port>`.
  A request whose `Origin` is set and is not that host is refused - that is the DNS
  rebinding defence, and it costs four lines.
* **Full rights.** This is the desk, not the phone: no `RemoteGuard`, no
  `remote_tier()`. GUARDED tools take the normal announce-and-confirm path; the window
  just gives you a Confirm button as well as a microphone. The blocklist is unchanged
  and applies exactly as it does at the keyboard.
* The desk server is **independent of `remote.enabled`**. Turning the phone off must
  not take the desk's window with it, and turning the window off must not close the
  phone's door.

### 24.3 `DeskBus` - what the window is told

```python
@dataclass(frozen=True)
class DeskEvent:
    type: str
    payload: dict
    at: float

class DeskBus:
    def __init__(self, *, history: int = 200) -> None
    def publish(self, type: str, /, **payload) -> DeskEvent
    def subscribe(self) -> "queue.Queue[DeskEvent]"   # its own queue, unbounded-but-capped
    def unsubscribe(self, q) -> None
    def replay(self) -> list[DeskEvent]               # the ring buffer, oldest first
    def close(self) -> None
```

Thread-safe, never raises into a caller, drops the oldest event for a subscriber that
has stopped draining rather than growing without limit.

Server to page:

| `type` | payload |
|---|---|
| `hello` | `version, model, whisper, gpu, phone_url, paused, routines[], tools[]` |
| `state` | `state` — idle / listening / thinking / speaking / paused |
| `heard` | `text` |
| `sentence` | `text` |
| `tool` | `name, phase("start"\|"end"), ok, refused, summary, data, ms` |
| `confirm` | `announcement, seconds` — a GUARDED tool is waiting |
| `confirm_done` | `granted` |
| `latency` | `ms, marks{}` |
| `note` | `text` — a problem the operator should see |
| `log` | `level, name, text` |
| `status` | the `telemetry.snapshot()` dict, pushed every 2 s while a window is open |

Page to server (`/ws/desk`, JSON frames):

| `type` | effect |
|---|---|
| `say` | one typed turn, full rights, through the same queue as the microphone |
| `listen` | arm the microphone as if the wake word had fired |
| `stop` | stop speaking / abandon the turn |
| `confirm` | `granted: bool` answers a pending GUARDED confirmation |
| `pause` / `resume` | the same toggle the tray has |
| `routine` | run a named macro from `remote.routines` |
| `set` | one key from a small allowlist: `tts.speed`, `audio.chime_volume`, `assistant.brief_mode`, `wake.sensitivity`, `ui.always_on_top` |
| `quit` | `assistant.stop()` |

Anything else is answered with an `error` frame and logged. No frame may reach `eval`,
the dispatcher or the config by name that is not on these lists.

The server never reaches into the assistant's internals for any of this. `Assistant`
grows exactly six methods, each of which returns rather than raises:

```python
def submit_desk_turn(self, text: str) -> bool     # a typed turn, spoken aloud, full rights
def arm_listening(self) -> bool                   # as if the wake word had fired
def abort_turn(self) -> None                      # stop speaking, abandon the turn
def answer_confirmation(self, granted: bool) -> bool   # resolve a pending GUARDED confirm
def run_routine(self, name: str) -> str           # a named macro; returns what was said
def apply_setting(self, key: str, value: Any) -> bool  # one allowlisted config key
```

### 24.4 Where the events come from

The assistant already narrates itself to the overlay's `HudModel` at fifteen call
sites (`begin_turn`, `add_reply`, `tool_started`, `tool_finished`, `note`, `touch`,
`latency_ms`). Rather than add a second set of calls beside every one of them, the desk
inserts a **fan** that satisfies the same interface and forwards to both:

```python
class _HudFan:          # jarvis/core/assistant.py, private
    """Looks exactly like HudModel; writes to the overlay and to the DeskBus."""
```

`Assistant.start()` builds it when a desk window is up, so `self._hud` keeps working
whether the overlay exists, the window exists, both or neither. `StateBus` is subscribed
directly. Log lines arrive through a `logging.Handler` that publishes to the bus at
`INFO` and above, with a formatter that keeps one line per record.

### 24.5 `DeskWindow` - the host and its three fallbacks

```python
class DeskWindow:
    def __init__(self, cfg, url, *, logger, on_close=None) -> None
    @property
    def backend(self) -> str          # "webview" | "edge" | "browser" | "none"
    @property
    def needs_main_thread(self) -> bool   # True only for "webview"
    def start(self) -> bool
    def run(self) -> None             # blocks; the pywebview loop, else returns at once
    def show(self) -> None
    def hide(self) -> None
    def toggle(self) -> None
    def stop(self) -> None
```

Order: `pywebview` (frameless, `easy_drag`, remembered size and position, closing hides
to the tray instead of quitting) → `msedge --app=<url>` with a dedicated
`--user-data-dir` under `logs/` so it never touches the user's profile → the default
browser → `none`, which logs one line and leaves the ring to do its job.

**The main thread** belongs to whoever must have it. `LayeredOverlay` already declares
`needs_main_thread = False` and runs its own message pump, so on Windows the reactor and
a pywebview window coexist. The tkinter `Overlay` does need it: when the window takes the
main thread and only the tkinter ring is available, the ring is dropped with one logged
line - the window is the bigger loss.

### 24.6 What the page shows

The layout is a single column of live evidence, not a chat log:

* a title bar of our own (drag region, minimise, close-to-tray) - no Windows chrome;
* the reactor, large, in the desk's states, and it is also the push-to-talk button;
* **the working line**: heard → thinking → each tool as it fires → the answer, with the
  end-of-speech-to-first-audio latency printed when the turn lands;
* **cards**, the same renderer the phone uses (`jarvis/ui/web/cards.js`): a business with
  its number, a timer that counts down, the weather, a refusal in red;
* **telemetry**: CPU, RAM, GPU load and temperature, VRAM, the model that is resident;
* **timers and reminders**, counting down, cancellable;
* **a confirmation bar** that appears for a GUARDED tool with Confirm and Cancel, next to
  the spoken window - whichever answers first wins;
* **a composer** for typed turns, and the routine buttons;
* **a log drawer**, collapsed by default, tailing `logs/jarvis.log` through the bus.

`desk.html` is one self-contained file, like the phone's `index.html`: same reactor
geometry, same card vocabulary, same palette, no build step and no shared bundle. The
two pages are deliberately not factored into a common module yet - the phone page ships
and works, and coupling it to a second consumer to save a few hundred lines would risk
a face that is already in the user's hand. When a third face appears, extract then.

### 24.7 No console, ever

`JARVIS.exe` is rebuilt for the **GUI subsystem** (`-mwindows`) and starts
`.venv\Scripts\pythonw.exe -m jarvis` directly with `CREATE_NO_WINDOW` - no `cmd.exe`,
no `start-jarvis.bat`, no black rectangle at any point. When the process exits non-zero
it shows a message box with the last lines of `logs\jarvis.log` and a button that opens
the file. `start-jarvis.bat` stays exactly as it is for anyone who wants the console.

Under `pythonw` there is no `sys.stdout`: `print()` becomes a silent no-op (CPython
returns early when `sys.stdout is None`), but a `StreamHandler` over `None` raises on
every record. `setup_logging` therefore **skips the console handler when there is no
console stream**, and `jarvis/__main__.py` installs a null stream before anything else
so third-party code cannot trip over it either.

### 24.8 Configuration

```yaml
ui:
  window: true            # the desk app window
  window_mode: auto       # auto | webview | edge | browser | off
  window_size: [980, 720]
  window_pos: null        # [x, y], remembered when you move it
  window_on_top: false
  open_window_on_start: true
```

`--window` / `--no-window` override it for one run; `--no-ui` still turns off
everything with a face.
