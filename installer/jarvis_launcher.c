/* JARVIS.exe
 *
 * The everyday icon: double-click it and JARVIS comes online. If he has not
 * been installed yet, it offers to run the installer instead of failing with
 * something cryptic.
 *
 * This program is linked for the GUI subsystem and starts pythonw.exe directly
 * with CREATE_NO_WINDOW, because the point of an icon on the desktop is that
 * nothing about it looks like a terminal - no interpreter window, no batch
 * file, no black rectangle behind the ring, not even for the half second it
 * takes Python to start. start-jarvis.bat still exists for anyone who wants
 * the console; this is the path for everyone who does not.
 *
 * Without a console there is also nowhere for a crash to be read, so a
 * non-zero exit is reported in the only place left: a message box holding the
 * tail of logs\jarvis.log, with the offer to open the whole file.
 *
 * Built on Linux with:
 *   x86_64-w64-mingw32-gcc -municode -mwindows -O2 -s jarvis_launcher.c jarvis_res.o -o JARVIS.exe
 */

#include "launcher_common.h"

#define MAX_LONG_PATH 4096
#define TAIL_BYTES 8192
#define TAIL_LINES 15
#define MESSAGE_CHARS (TAIL_BYTES + 512)

/* The last TAIL_LINES lines of a UTF-8 log file, as wide text. FALSE when the
 * file cannot be read at all - which is itself something to say out loud. */
static BOOL jarvis_log_tail(const wchar_t *path, wchar_t *out, int count)
{
    HANDLE file;
    LARGE_INTEGER size;
    LARGE_INTEGER offset;
    DWORD want;
    DWORD got = 0;
    int start;
    int end;
    int lines = 0;
    int wide;
    char buffer[TAIL_BYTES + 1];

    out[0] = L'\0';
    file = CreateFileW(path, GENERIC_READ,
                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        return FALSE;
    }
    if (!GetFileSizeEx(file, &size)) {
        CloseHandle(file);
        return FALSE;
    }
    want = (size.QuadPart > (LONGLONG)TAIL_BYTES) ? (DWORD)TAIL_BYTES : (DWORD)size.QuadPart;
    offset.QuadPart = size.QuadPart - (LONGLONG)want;
    if (!SetFilePointerEx(file, offset, NULL, FILE_BEGIN)
        || (want > 0 && !ReadFile(file, buffer, want, &got, NULL))) {
        CloseHandle(file);
        return FALSE;
    }
    CloseHandle(file);
    if (got > (DWORD)TAIL_BYTES) {
        got = (DWORD)TAIL_BYTES;
    }

    end = (int)got;
    while (end > 0 && (buffer[end - 1] == '\n' || buffer[end - 1] == '\r')) {
        end--;
    }

    start = end;
    while (start > 0) {
        if (buffer[start - 1] == '\n') {
            lines++;
            if (lines >= TAIL_LINES) {
                break;              /* start already sits just past that newline */
            }
        }
        start--;
    }
    if (lines < TAIL_LINES && offset.QuadPart > 0) {
        /* The read began in the middle of a line; drop the fragment. */
        while (start < end && buffer[start] != '\n') {
            start++;
        }
        if (start < end) {
            start++;
        }
    }

    if (end <= start) {
        return TRUE;                /* readable, but empty - the caller decides */
    }
    wide = MultiByteToWideChar(CP_UTF8, 0, buffer + start, end - start, out, count - 1);
    if (wide <= 0) {
        return FALSE;
    }
    out[wide] = L'\0';
    return TRUE;
}

/* The only diagnostic left to a user with no console. */
static void jarvis_report_failure(const wchar_t *dir, int code)
{
    wchar_t log_path[MAX_LONG_PATH];
    wchar_t tail[TAIL_BYTES + 1];
    wchar_t text[MESSAGE_CHARS];
    wchar_t headline[128];
    BOOL built;

    wsprintfW(headline, L"JARVIS stopped unexpectedly (exit code %d).\n\n", code);

    if (!jarvis_join(log_path, MAX_LONG_PATH, dir, L"logs\\jarvis.log")
        || !jarvis_log_tail(log_path, tail, TAIL_BYTES)
        || tail[0] == L'\0') {
        text[0] = L'\0';
        if (jarvis_append(text, MESSAGE_CHARS, headline)) {
            jarvis_append(text, MESSAGE_CHARS,
                          L"His log could not be read, sir, so I am afraid I cannot "
                          L"tell you why.");
        }
        jarvis_message(L"JARVIS", text, MB_ICONERROR);
        return;
    }

    text[0] = L'\0';
    built = jarvis_append(text, MESSAGE_CHARS, headline)
         && jarvis_append(text, MESSAGE_CHARS, L"The last lines of his log:\n\n")
         && jarvis_append(text, MESSAGE_CHARS, tail)
         && jarvis_append(text, MESSAGE_CHARS, L"\n\nShall I open the full log?");
    if (!built) {
        jarvis_message(L"JARVIS", headline, MB_ICONERROR);
        return;
    }
    if (jarvis_ask(L"JARVIS", text)) {
        ShellExecuteW(NULL, L"open", log_path, NULL, dir, SW_SHOWNORMAL);
    }
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command_line, int show)
{
    wchar_t dir[MAX_LONG_PATH];
    wchar_t venv_python[MAX_LONG_PATH];
    wchar_t venv_pythonw[MAX_LONG_PATH];
    wchar_t installer[MAX_LONG_PATH];
    wchar_t cmdline[MAX_LONG_PATH * 2];
    const wchar_t *interpreter;
    wchar_t **argv;
    int argc = 0;
    int code;
    int i;
    BOOL built;

    (void)instance;
    (void)previous;
    (void)command_line;
    (void)show;

    if (!jarvis_exe_dir(dir, MAX_LONG_PATH)) {
        jarvis_message(L"JARVIS", L"I could not work out where I am on this disk.", MB_ICONERROR);
        return 1;
    }

    if (!jarvis_join(venv_python, MAX_LONG_PATH, dir, L".venv\\Scripts\\python.exe")
        || !jarvis_join(venv_pythonw, MAX_LONG_PATH, dir, L".venv\\Scripts\\pythonw.exe")
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

    /* pythonw.exe is the one with no console attached. python.exe is only the
     * fallback for a virtual environment that somehow lacks it, and even then
     * CREATE_NO_WINDOW keeps the window off the desktop. */
    interpreter = jarvis_file_exists(venv_pythonw) ? venv_pythonw : venv_python;

    /* "<dir>\.venv\Scripts\pythonw.exe" -m jarvis "<arg>"... */
    argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    cmdline[0] = L'\0';
    built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"")
         && jarvis_append(cmdline, MAX_LONG_PATH * 2, interpreter)
         && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\" -m jarvis");

    for (i = 1; argv != NULL && i < argc && built; ++i) {
        built = jarvis_append(cmdline, MAX_LONG_PATH * 2, L" \"")
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, argv[i])
             && jarvis_append(cmdline, MAX_LONG_PATH * 2, L"\"");
    }
    if (argv != NULL) {
        LocalFree(argv);
    }

    if (!built) {
        jarvis_message(L"JARVIS", L"The path to this folder is too long for me to work with.", MB_ICONERROR);
        return 1;
    }

    code = jarvis_run_and_wait(cmdline, dir, CREATE_NO_WINDOW);
    if (code == -1) {
        jarvis_message(L"JARVIS", L"Windows would not let me start JARVIS.", MB_ICONERROR);
        return 1;
    }
    if (code != 0) {
        jarvis_report_failure(dir, code);
    }
    return code;
}
