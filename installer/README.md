# installer/

The two things a person double-clicks.

| File | What it is |
|---|---|
| `install_jarvis.c` | `Install-JARVIS.exe` — finds `install.ps1` next to itself and runs it. Manifested `requireAdministrator`, because installing Python and Ollama through winget needs an elevated token. |
| `jarvis_launcher.c` | `JARVIS.exe` — the everyday icon. Runs `start-jarvis.bat`; if `.venv` is missing it offers to run the installer instead of failing with something cryptic. Manifested `asInvoker` — JARVIS never needs administrator rights to run. |
| `launcher_common.h` | shared wide-character path and process helpers, so folder names with spaces, å, ä or ö behave |
| `install.manifest` / `launcher.manifest` | UAC level, long-path awareness, UTF-8 code page, Windows 10/11 compatibility |
| `install_res.rc` / `jarvis_res.rc` | icon, manifest and version information compiled into each binary |
| `make_icon.py` | draws the arc reactor into `assets/jarvis.ico` at seven sizes, with no image library |
| `build.sh` | cross-compiles both executables from Linux with mingw-w64 |

The actual installation logic lives in `../install.ps1`. These two programs exist purely
so that nobody has to open a terminal, know what an execution policy is, or type anything.

Both binaries are committed at the repository root so the repository works the moment it
is unzipped. Rebuild them only if you change something here:

```bash
sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
./installer/build.sh
```

They are unsigned, so SmartScreen will warn the first time — *More info* → *Run anyway*.
Signing needs a paid code-signing certificate; `install.ps1` can always be run directly
instead (right-click → Run with PowerShell).
