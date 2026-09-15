# models/

Model weights live here. They are **not** committed — `.gitignore` keeps them out of
the repository because they are hundreds of megabytes.

Fetch them with:

```powershell
.venv\Scripts\python scripts\fetch_models.py           # Kokoro voice + wake word models
.venv\Scripts\python scripts\fetch_models.py --swedish # also the Piper Swedish voice
```

| File | Purpose | Source |
|---|---|---|
| `kokoro-v1.0.onnx` | Kokoro TTS model (the British voice) | kokoro-onnx GitHub releases |
| `voices-v1.0.bin` | Kokoro voice bank (`bm_george`, `bm_lewis`, `bf_emma`, …) | kokoro-onnx GitHub releases |
| `sv_SE-nst-medium.onnx(.json)` | Piper Swedish voice, used when `language: sv` | rhasspy/piper-voices |

The wake word model (`hey_jarvis`) is downloaded into openWakeWord's own package
directory by its bundled utility, not into this folder.

Whisper models are downloaded automatically by faster-whisper into the Hugging Face
cache (`%USERPROFILE%\.cache\huggingface`) the first time JARVIS transcribes anything.
