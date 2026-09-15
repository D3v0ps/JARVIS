"""Download the model files JARVIS needs that cannot ship in the repository.

Everything here is free, keyless and downloaded straight from the projects that
publish it. No audio is ever taken from copyrighted material - the chimes are
synthesized in ``jarvis/audio/chimes.py``.

    python scripts/fetch_models.py            # Kokoro voice + wake word models
    python scripts/fetch_models.py --swedish  # also the Piper Swedish voice
    python scripts/fetch_models.py --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/nst/medium"

# (url, destination, approximate size for the progress line)
KOKORO_FILES = [
    (f"{KOKORO_BASE}/kokoro-v1.0.onnx", MODELS / "kokoro-v1.0.onnx", 310_000_000),
    (f"{KOKORO_BASE}/voices-v1.0.bin", MODELS / "voices-v1.0.bin", 27_000_000),
]
PIPER_FILES = [
    (f"{PIPER_BASE}/sv_SE-nst-medium.onnx", MODELS / "sv_SE-nst-medium.onnx", 63_000_000),
    (f"{PIPER_BASE}/sv_SE-nst-medium.onnx.json", MODELS / "sv_SE-nst-medium.onnx.json", 5_000),
]


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, expected: int = 0) -> bool:
    """Stream a file to ``dest`` with a progress line. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [ok]   {dest.name} already present ({_human(dest.stat().st_size)})")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get]  {dest.name}  (~{_human(expected)})")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "jarvis-setup/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or expected or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r         {pct:5.1f}%  {_human(done)} / {_human(total)}", end="")
            print()
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"\n  [fail] {dest.name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def fetch_wakeword_models() -> bool:
    """openWakeWord ships its own downloader; the 'hey_jarvis' model comes with it."""
    print("\nWake word models (openWakeWord)")
    try:
        import openwakeword.utils as oww_utils
    except ImportError:
        print("  [skip] openwakeword is not installed yet - run it after pip install -r requirements.txt")
        return False
    try:
        oww_utils.download_models()
        print("  [ok]   openWakeWord models downloaded (including hey_jarvis)")
        return True
    except Exception as exc:  # noqa: BLE001 - the library raises a variety of network errors
        print(f"  [fail] {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download JARVIS model files")
    parser.add_argument("--swedish", action="store_true", help="also fetch the Piper Swedish voice")
    parser.add_argument("--all", action="store_true", help="fetch everything")
    parser.add_argument("--skip-wake", action="store_true", help="do not download wake word models")
    args = parser.parse_args(argv)

    MODELS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODELS).free
    if free < 1_500_000_000:
        print(f"Only {_human(free)} free on this drive - the models need roughly 1 GB.")

    ok = True
    print("Kokoro voice (English, bm_george and friends)")
    for url, dest, size in KOKORO_FILES:
        ok &= download(url, dest, size)

    if args.swedish or args.all:
        print("\nPiper Swedish voice (sv_SE-nst-medium)")
        for url, dest, size in PIPER_FILES:
            ok &= download(url, dest, size)

    if not args.skip_wake:
        ok &= fetch_wakeword_models()

    print("\nDone." if ok else "\nFinished with errors - see the lines marked [fail] above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
