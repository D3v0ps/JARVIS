/* Install-JARVIS.exe
 *
 * Double-click this and JARVIS installs himself: Python, Ollama, the right
 * qwen3 model for the graphics card in this machine, the virtual environment,
 * every dependency, the voice and wake-word models, and the shortcuts.
 *
 * All this executable does is find install.ps1 next to itself and run it with
 * PowerShell. It exists so that nobody has to know what PowerShell is.
 *
 * Built on Linux with:
 *   x86_64-w64-mingw32-gcc -municode -O2 -s install_jarvis.c install_res.o -o Install-JARVIS.exe
 */

#include "launcher_common.h"

#define MAX_LONG_PATH 4096

int wmain(int argc, wchar_t **argv)
{
    wchar_t dir[MAX_LONG_PATH];
    wchar_t script[MAX_LONG_PATH];
    wchar_t powershell[MAX_PATH];
    wchar_t cmdline[MAX_LONG_PATH * 2];

    if (!jarvis_exe_dir(dir, MAX_LONG_PATH)) {
        jarvis_message(L"JARVIS", L"I could not work out where I am on this disk.", MB_ICONERROR);
        return 1;
    }

    if (!jarvis_join(script, MAX_LONG_PATH, dir, L"install.ps1")) {
        jarvis_message(L"JARVIS", L"The path to this folder is too long for me to work with.", MB_ICONERROR);
        return 1;
    }

    if (!jarvis_file_exists(script)) {
        jarvis_message(L"JARVIS",
                       L"install.ps1 is missing.\n\n"
                       L"Install-JARVIS.exe has to sit in the JARVIS folder, "
                       L"next to install.ps1 and requirements.txt.\n\n"
                       L"If you unpacked a zip file, make sure you unpacked all of it.",
                       MB_ICONERROR);
        return 1;
    }

    if (!jarvis_system_exe(L"WindowsPowerShell\\v1.0\\powershell.exe", powershell, MAX_PATH)
        || !jarvis_file_exists(powershell)) {
        jarvis_message(L"JARVIS", L"Windows PowerShell is missing from this machine, which is unusual.",
                       MB_ICONERROR);
        return 1;
    }

    SetConsoleTitleW(L"J.A.R.V.I.S. - installing");

    /* powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<dir>\install.ps1" <extra args> */
    cmdline[0] = L'\0';
    BOOL built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"")
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, powershell)
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\" -NoProfile -ExecutionPolicy Bypass -File \"")
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, script)
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");

    /* Pass anything the user dropped on the icon straight through, so
     * -Swedish, -Silent or -Autostart still work for those who want them. */
    for (int i = 1; i < argc && built; ++i) {
        built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L" \"")
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, argv[i])
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");
    }

    if (!built) {
        jarvis_message(L"JARVIS", L"The path to this folder is too long for me to work with.", MB_ICONERROR);
        return 1;
    }

    int code = jarvis_run_and_wait(cmdline, dir, 0);

    if (code == -1) {
        jarvis_message(L"JARVIS", L"Windows would not let me start PowerShell.", MB_ICONERROR);
        return 1;
    }
    return code;
}
