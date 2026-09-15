/* JARVIS.exe
 *
 * The everyday icon: double-click it and JARVIS comes online. If he has not
 * been installed yet, it offers to run the installer instead of failing with
 * something cryptic.
 *
 * Built on Linux with:
 *   x86_64-w64-mingw32-gcc -municode -O2 -s jarvis_launcher.c jarvis_res.o -o JARVIS.exe
 */

#include "launcher_common.h"

#define MAX_LONG_PATH 4096

int wmain(int argc, wchar_t **argv)
{
    wchar_t dir[MAX_LONG_PATH];
    wchar_t venv_python[MAX_LONG_PATH];
    wchar_t launcher[MAX_LONG_PATH];
    wchar_t installer[MAX_LONG_PATH];
    wchar_t cmd_exe[MAX_PATH];
    wchar_t cmdline[MAX_LONG_PATH * 2];

    if (!jarvis_exe_dir(dir, MAX_LONG_PATH)) {
        jarvis_message(L"JARVIS", L"I could not work out where I am on this disk.", MB_ICONERROR);
        return 1;
    }

    if (!jarvis_join(venv_python, MAX_LONG_PATH, dir, L".venv\\Scripts\\python.exe")
        || !jarvis_join(launcher, MAX_LONG_PATH, dir, L"start-jarvis.bat")
        || !jarvis_join(installer, MAX_LONG_PATH, dir, L"Install-JARVIS.exe")) {
        jarvis_message(L"JARVIS", L"The path to this folder is too long for me to work with.", MB_ICONERROR);
        return 1;
    }

    /* Not installed yet - offer the installer rather than a wall of errors. */
    if (!jarvis_file_exists(venv_python)) {
        if (jarvis_file_exists(installer)
            && jarvis_ask(L"JARVIS",
                          L"JARVIS is not installed on this machine yet.\n\n"
                          L"Shall I run the installer now?\n\n"
                          L"It fetches Python, Ollama, the language model and the voice. "
                          L"It takes a while the first time and needs an internet connection.")) {
            SHELLEXECUTEINFOW info;
            ZeroMemory(&info, sizeof(info));
            info.cbSize = sizeof(info);
            info.fMask = SEE_MASK_NOCLOSEPROCESS;
            info.lpVerb = L"runas";              /* the installer asks for elevation */
            info.lpFile = installer;
            info.lpDirectory = dir;
            info.nShow = SW_SHOWNORMAL;
            if (!ShellExecuteExW(&info)) {
                jarvis_message(L"JARVIS", L"The installer would not start.", MB_ICONERROR);
                return 1;
            }
            if (info.hProcess) {
                WaitForSingleObject(info.hProcess, INFINITE);
                CloseHandle(info.hProcess);
            }
            /* The installer offers to start him at the end, so stop here. */
            return 0;
        }
        jarvis_message(L"JARVIS",
                       L"JARVIS is not installed yet.\n\n"
                       L"Double-click Install-JARVIS.exe in this folder first.",
                       MB_ICONINFORMATION);
        return 1;
    }

    if (!jarvis_file_exists(launcher)) {
        jarvis_message(L"JARVIS",
                       L"start-jarvis.bat is missing from this folder.\n\n"
                       L"Unpack the whole JARVIS folder and try again.",
                       MB_ICONERROR);
        return 1;
    }

    if (!jarvis_system_exe(L"cmd.exe", cmd_exe, MAX_PATH)) {
        jarvis_message(L"JARVIS", L"I could not find cmd.exe, which is unusual.", MB_ICONERROR);
        return 1;
    }

    SetConsoleTitleW(L"J.A.R.V.I.S.");

    /* cmd.exe /c ""<dir>\start-jarvis.bat"" - the doubled quotes are cmd's rule
     * for a quoted program path followed by quoted arguments. */
    cmdline[0] = L'\0';
    BOOL built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"")
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, cmd_exe)
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\" /c \"\"")
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, launcher)
              && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");

    for (int i = 1; i < argc && built; ++i) {
        built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L" \"")
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, argv[i])
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");
    }
    built = built && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");

    if (!built) {
        jarvis_message(L"JARVIS", L"The path to this folder is too long for me to work with.", MB_ICONERROR);
        return 1;
    }

    int code = jarvis_run_and_wait(cmdline, dir, 0);
    if (code == -1) {
        jarvis_message(L"JARVIS", L"Windows would not let me start the launcher.", MB_ICONERROR);
        return 1;
    }
    return code;
}
