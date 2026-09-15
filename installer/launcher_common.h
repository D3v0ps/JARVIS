/* Shared helpers for the two JARVIS launchers.
 *
 * Both are tiny native Windows programs whose only job is to let someone start
 * JARVIS by double-clicking an icon - no terminal, no PowerShell, no arguments.
 * Everything is wide-character so paths with spaces, accents or a Swedish user
 * name work exactly like any other path.
 */
#ifndef JARVIS_LAUNCHER_COMMON_H
#define JARVIS_LAUNCHER_COMMON_H

#include <windows.h>
#include <shellapi.h>

/* Bounded wide-string append. Returns FALSE when the result would not fit. */
static inline BOOL jarvis_append(wchar_t *dest, size_t count, const wchar_t *src)
{
    size_t used = 0;
    while (used < count && dest[used] != L'\0') {
        used++;
    }
    if (used >= count) {
        return FALSE;
    }
    size_t i = 0;
    while (src[i] != L'\0' && used + i + 1 < count) {
        dest[used + i] = src[i];
        i++;
    }
    dest[used + i] = L'\0';
    return src[i] == L'\0';
}

/* Directory this executable lives in, without a trailing backslash. */
static inline BOOL jarvis_exe_dir(wchar_t *out, DWORD count)
{
    DWORD len = GetModuleFileNameW(NULL, out, count);
    if (len == 0 || len >= count) {
        return FALSE;
    }
    for (DWORD i = len; i > 0; --i) {
        if (out[i - 1] == L'\\') {
            out[i - 1] = L'\0';
            return TRUE;
        }
    }
    return FALSE;
}

/* dir + "\" + leaf  ->  out */
static inline BOOL jarvis_join(wchar_t *out, size_t count, const wchar_t *dir, const wchar_t *leaf)
{
    out[0] = L'\0';
    return jarvis_append(out, count, dir)
        && jarvis_append(out, count, L"\\")
        && jarvis_append(out, count, leaf);
}

static inline BOOL jarvis_file_exists(const wchar_t *path)
{
    DWORD attrs = GetFileAttributesW(path);
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

/* Absolute path to a System32 executable, so PATH cannot be used to slip us a
 * different powershell.exe or cmd.exe. */
static inline BOOL jarvis_system_exe(const wchar_t *name, wchar_t *out, UINT count)
{
    UINT len = GetSystemDirectoryW(out, count);
    if (len == 0 || len >= count) {
        return FALSE;
    }
    return jarvis_append(out, count, L"\\") && jarvis_append(out, count, name);
}

/* Run a command line, wait for it, and return its exit code (-1 if it would not
 * start). `cmdline` is modified in place by CreateProcessW, so it must be a
 * writable buffer, never a string literal. */
static inline int jarvis_run_and_wait(wchar_t *cmdline, const wchar_t *working_dir, DWORD flags)
{
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE, flags, NULL, working_dir, &si, &pi)) {
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}

static inline void jarvis_message(const wchar_t *title, const wchar_t *text, UINT icon)
{
    MessageBoxW(NULL, text, title, icon | MB_OK | MB_TOPMOST);
}

static inline int jarvis_ask(const wchar_t *title, const wchar_t *text)
{
    return MessageBoxW(NULL, text, title, MB_YESNO | MB_ICONQUESTION | MB_TOPMOST) == IDYES;
}

#endif /* JARVIS_LAUNCHER_COMMON_H */
