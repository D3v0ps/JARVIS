# installer/

The two things a person double-clicks.

| File | What it is |
|---|---|
| `install_jarvis.c` | `Install-JARVIS.exe` — finds `install.ps1` next to itself and runs it. Manifested `requireAdministrator`, because installing Python and Ollama through winget needs an elevated token. |
| `jarvis_launcher.c` | `JARVIS.exe` — the everyday icon. A GUI-subsystem program that starts `.venv\Scripts\pythonw.exe -m jarvis` directly, so no console ever appears; if `.venv` is missing it offers to run the installer instead of failing with something cryptic. Manifested `asInvoker` — JARVIS never needs administrator rights to run. |
| `launcher_common.h` | shared wide-character path and process helpers, so folder names with spaces, å, ä or ö behave |
| `install.manifest` / `launcher.manifest` | UAC level, long-path awareness, UTF-8 code page, Windows 10/11 compatibility |
| `install_res.rc` / `jarvis_res.rc` | icon, manifest and version information compiled into each binary |
| `make_icon.py` | draws the arc reactor into `assets/jarvis.ico` at seven sizes, with no image library |
| `build.sh` | cross-compiles both executables from Linux with mingw-w64 |

The actual installation logic lives in `../install.ps1`. These two programs exist purely
so that nobody has to open a terminal, know what an execution policy is, or type anything.

Both binaries are committed at the repository root so the repository works the moment it
is unzipped; rebuild them only if you change something here (see *Rebuilding* below).

They are unsigned, so SmartScreen will warn the first time — *More info* → *Run anyway*.
Signing needs a paid code-signing certificate; `install.ps1` can always be run directly
instead (right-click → Run with PowerShell).

## What JARVIS.exe does

The desktop icon used to run `cmd.exe /c start-jarvis.bat` and wait, which left a black
console full of log lines sitting behind the arc reactor. It no longer does anything of
the sort:

1. It works out its own folder and looks for `.venv\Scripts\python.exe`. If that is
   missing, JARVIS is not installed: it offers to run `Install-JARVIS.exe` (elevated, via
   `runas`) and stops there, because the installer offers to start him at the end.
2. Otherwise it builds `"<folder>\.venv\Scripts\pythonw.exe" -m jarvis <arguments>` and
   runs it with `CreateProcessW` and `CREATE_NO_WINDOW`, working directory set to the
   folder. Arguments given to `JARVIS.exe` are passed straight through, so
   `JARVIS.exe --no-window` works. `python.exe` is used only if `pythonw.exe` is somehow
   absent from the virtual environment, and even then with `CREATE_NO_WINDOW`.
3. It waits. On a clean exit it simply disappears. On a non-zero exit it shows a message
   box with the last fifteen lines of `logs\jarvis.log` and offers to open the file —
   with no console, that box is the only diagnostic a user has. If the log cannot be read
   it says so plainly rather than vanishing.

`start-jarvis.bat` is untouched and is still the right thing to double-click when you
*want* the console: first run, dependency trouble, or watching a traceback go past.

## Rebuilding

```bash
sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
./installer/build.sh
```

`build.sh` links `JARVIS.exe` with `-mwindows` (GUI subsystem — no console is allocated,
and the entry point is `wWinMain`, not `wmain`) and `Install-JARVIS.exe` without it: the
installer prints half an hour of progress and wants its console. Check that the two came
out right with:

```bash
x86_64-w64-mingw32-objdump -p JARVIS.exe         | grep -i subsystem   # Windows GUI
x86_64-w64-mingw32-objdump -p Install-JARVIS.exe | grep -i subsystem   # Windows CUI
```

`tests/test_no_console.py` asserts the same thing against the committed binaries, so a
launcher rebuilt for the wrong subsystem fails the suite rather than the user.
