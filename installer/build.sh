#!/usr/bin/env bash
# Build the two Windows launchers. They are committed to the repository already,
# so you only need this if you change the C sources, the manifests or the icon.
#
# On Debian/Ubuntu:  sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
# Then:              ./installer/build.sh
#
# On Windows with MSVC, the equivalent is:
#   rc install_res.rc && cl /O2 install_jarvis.c install_res.res user32.lib shell32.lib
set -euo pipefail

CC=${CC:-x86_64-w64-mingw32-gcc}
WINDRES=${WINDRES:-x86_64-w64-mingw32-windres}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

command -v "$CC" >/dev/null || { echo "error: $CC not found - install gcc-mingw-w64-x86-64"; exit 1; }
command -v "$WINDRES" >/dev/null || { echo "error: $WINDRES not found - install binutils-mingw-w64-x86-64"; exit 1; }

cd "$HERE"

echo "  regenerating the icon..."
python3 make_icon.py

echo "  compiling resources..."
"$WINDRES" install_res.rc -O coff -o install_res.o
"$WINDRES" jarvis_res.rc  -O coff -o jarvis_res.o

echo "  linking Install-JARVIS.exe..."
"$CC" -municode -O2 -s -Wall -Wextra \
      -o "$ROOT/Install-JARVIS.exe" install_jarvis.c install_res.o -lshell32 -luser32

# -mwindows on this one only: JARVIS.exe is a GUI-subsystem program so that
# double-clicking it never flashes a console. Install-JARVIS.exe keeps its
# console on purpose - the install is a half-hour of output worth watching.
echo "  linking JARVIS.exe (GUI subsystem)..."
"$CC" -municode -mwindows -O2 -s -Wall -Wextra \
      -o "$ROOT/JARVIS.exe" jarvis_launcher.c jarvis_res.o -lshell32 -luser32

rm -f install_res.o jarvis_res.o
ls -la "$ROOT"/*.exe
echo "  done."
