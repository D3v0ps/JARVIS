"""ctypes structures for the layered overlay.

Kept apart from :mod:`jarvis.ui.layered` so the definitions stay readable, and so
the module can be imported and inspected on any platform - the structures are
just memory layouts, and defining them costs nothing on Linux.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

__all__ = [
    "POINT", "SIZE", "RECT", "BLENDFUNCTION", "BITMAPINFOHEADER", "BITMAPINFO",
    "MSG", "WNDCLASSEX", "WNDPROC",
]

# ``ctypes.wintypes`` is only populated on Windows; on other platforms fall back to
# equivalent C types so the module still imports and can be unit-tested.
_LONG = getattr(wintypes, "LONG", ctypes.c_long)
_DWORD = getattr(wintypes, "DWORD", ctypes.c_uint32)
_WORD = getattr(wintypes, "WORD", ctypes.c_uint16)
_BYTE = getattr(wintypes, "BYTE", ctypes.c_ubyte)
_UINT = getattr(wintypes, "UINT", ctypes.c_uint)
_HWND = getattr(wintypes, "HWND", ctypes.c_void_p)
_HANDLE = getattr(wintypes, "HANDLE", ctypes.c_void_p)
_LPCWSTR = getattr(wintypes, "LPCWSTR", ctypes.c_wchar_p)

if hasattr(ctypes, "WINFUNCTYPE"):
    WNDPROC = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, _HWND, _UINT, ctypes.c_size_t, ctypes.c_ssize_t
    )
else:  # pragma: no cover - Linux, for import only
    WNDPROC = ctypes.CFUNCTYPE(
        ctypes.c_ssize_t, _HWND, _UINT, ctypes.c_size_t, ctypes.c_ssize_t
    )


class POINT(ctypes.Structure):
    _fields_ = [("x", _LONG), ("y", _LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", _LONG), ("cy", _LONG)]


class RECT(ctypes.Structure):
    _fields_ = [("left", _LONG), ("top", _LONG), ("right", _LONG), ("bottom", _LONG)]


class BLENDFUNCTION(ctypes.Structure):
    """Tells the compositor the colour channels are already multiplied by alpha."""

    _fields_ = [
        ("BlendOp", _BYTE),
        ("BlendFlags", _BYTE),
        ("SourceConstantAlpha", _BYTE),
        ("AlphaFormat", _BYTE),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", _DWORD),
        ("biWidth", _LONG),
        ("biHeight", _LONG),          # negative means a top-down bitmap
        ("biPlanes", _WORD),
        ("biBitCount", _WORD),
        ("biCompression", _DWORD),
        ("biSizeImage", _DWORD),
        ("biXPelsPerMeter", _LONG),
        ("biYPelsPerMeter", _LONG),
        ("biClrUsed", _DWORD),
        ("biClrImportant", _DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", _DWORD * 3)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hWnd", _HWND),
        ("message", _UINT),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t),
        ("time", _DWORD),
        ("pt", POINT),
    ]


class WNDCLASSEX(ctypes.Structure):
    _fields_ = [
        ("cbSize", _UINT),
        ("style", _UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", _HANDLE),
        ("hIcon", _HANDLE),
        ("hCursor", _HANDLE),
        ("hbrBackground", _HANDLE),
        ("lpszMenuName", _LPCWSTR),
        ("lpszClassName", _LPCWSTR),
        ("hIconSm", _HANDLE),
    ]


# --- function prototypes --------------------------------------------------------------
# Without these, ctypes marshals every argument as a C int. A module handle or a window
# handle is a 64-bit pointer, so the first one that exceeds 32 bits raises
# "OverflowError: int too long to convert" - which is exactly how the layered overlay
# used to fail at CreateWindowExW's hInstance and fall back to the tkinter ring.

_HDC = _HANDLE
_HMENU = _HANDLE
_HBITMAP = _HANDLE
_HGDIOBJ = _HANDLE
_HINSTANCE = _HANDLE
_HCURSOR = _HANDLE
_BOOL = ctypes.c_int
_ATOM = _WORD
_COLORREF = _DWORD
_LPVOID = ctypes.c_void_p

#: Windows' own typedefs for the pointer-sized message parameters.
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t
LRESULT = ctypes.c_ssize_t
LONG_PTR = ctypes.c_ssize_t


def bind() -> tuple:
    """Declare every signature we use and return ``(user32, gdi32, kernel32)``.

    Idempotent: calling it twice simply re-declares the same prototypes. Raises
    ``OSError`` off Windows, where these libraries do not exist.
    """
    if not hasattr(ctypes, "WinDLL"):  # pragma: no cover - non-Windows
        raise OSError("The Win32 API is only available on Windows")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.GetModuleHandleW.argtypes = [_LPCWSTR]
    kernel32.GetModuleHandleW.restype = _HINSTANCE

    user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEX)]
    user32.RegisterClassExW.restype = _ATOM

    user32.CreateWindowExW.argtypes = [
        _DWORD, _LPCWSTR, _LPCWSTR, _DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _HWND, _HMENU, _HINSTANCE, _LPVOID,
    ]
    user32.CreateWindowExW.restype = _HWND

    user32.DestroyWindow.argtypes = [_HWND]
    user32.DestroyWindow.restype = _BOOL
    user32.ShowWindow.argtypes = [_HWND, ctypes.c_int]
    user32.ShowWindow.restype = _BOOL
    user32.SetWindowPos.argtypes = [
        _HWND, _HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _UINT
    ]
    user32.SetWindowPos.restype = _BOOL

    user32.GetDC.argtypes = [_HWND]
    user32.GetDC.restype = _HDC
    user32.ReleaseDC.argtypes = [_HWND, _HDC]
    user32.ReleaseDC.restype = ctypes.c_int

    user32.UpdateLayeredWindow.argtypes = [
        _HWND, _HDC, ctypes.POINTER(POINT), ctypes.POINTER(SIZE),
        _HDC, ctypes.POINTER(POINT), _COLORREF,
        ctypes.POINTER(BLENDFUNCTION), _DWORD,
    ]
    user32.UpdateLayeredWindow.restype = _BOOL

    user32.DefWindowProcW.argtypes = [_HWND, _UINT, WPARAM, LPARAM]
    user32.DefWindowProcW.restype = LRESULT

    user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), _HWND, _UINT, _UINT, _UINT]
    user32.PeekMessageW.restype = _BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.TranslateMessage.restype = _BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.PostMessageW.argtypes = [_HWND, _UINT, WPARAM, LPARAM]
    user32.PostMessageW.restype = _BOOL

    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.GetCursorPos.restype = _BOOL
    user32.SetCapture.argtypes = [_HWND]
    user32.SetCapture.restype = _HWND
    user32.ReleaseCapture.argtypes = []
    user32.ReleaseCapture.restype = _BOOL

    user32.CreatePopupMenu.argtypes = []
    user32.CreatePopupMenu.restype = _HMENU
    user32.AppendMenuW.argtypes = [_HMENU, _UINT, ctypes.c_size_t, _LPCWSTR]
    user32.AppendMenuW.restype = _BOOL
    user32.TrackPopupMenu.argtypes = [
        _HMENU, _UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, _HWND, ctypes.c_void_p
    ]
    user32.TrackPopupMenu.restype = _BOOL
    user32.DestroyMenu.argtypes = [_HMENU]
    user32.DestroyMenu.restype = _BOOL
    user32.SetForegroundWindow.argtypes = [_HWND]
    user32.SetForegroundWindow.restype = _BOOL

    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.LoadCursorW.argtypes = [_HINSTANCE, _LPCWSTR]
    user32.LoadCursorW.restype = _HCURSOR
    user32.SetConsoleTitleW = kernel32.SetConsoleTitleW  # lives in kernel32, not user32

    # Present from Windows 10 1703; absent on older builds, hence the guard.
    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        user32.SetProcessDpiAwarenessContext.argtypes = [_HANDLE]
        user32.SetProcessDpiAwarenessContext.restype = _BOOL

    # The Ptr variants only exist on 64-bit; fall back to the 32-bit names.
    for getter, setter in (("GetWindowLongPtrW", "SetWindowLongPtrW"),
                           ("GetWindowLongW", "SetWindowLongW")):
        if hasattr(user32, getter):
            getattr(user32, getter).argtypes = [_HWND, ctypes.c_int]
            getattr(user32, getter).restype = LONG_PTR
            getattr(user32, setter).argtypes = [_HWND, ctypes.c_int, LONG_PTR]
            getattr(user32, setter).restype = LONG_PTR

    gdi32.CreateCompatibleDC.argtypes = [_HDC]
    gdi32.CreateCompatibleDC.restype = _HDC
    gdi32.CreateDIBSection.argtypes = [
        _HDC, ctypes.POINTER(BITMAPINFO), _UINT,
        ctypes.POINTER(ctypes.c_void_p), _HANDLE, _DWORD,
    ]
    gdi32.CreateDIBSection.restype = _HBITMAP
    gdi32.SelectObject.argtypes = [_HDC, _HGDIOBJ]
    gdi32.SelectObject.restype = _HGDIOBJ
    gdi32.DeleteObject.argtypes = [_HGDIOBJ]
    gdi32.DeleteObject.restype = _BOOL
    gdi32.DeleteDC.argtypes = [_HDC]
    gdi32.DeleteDC.restype = _BOOL

    kernel32.SetConsoleTitleW.argtypes = [_LPCWSTR]
    kernel32.SetConsoleTitleW.restype = _BOOL

    return user32, gdi32, kernel32


def cursor_resource(value: int) -> "ctypes.c_wchar_p":
    """MAKEINTRESOURCE: a small integer masquerading as a string pointer.

    ``LoadCursorW(None, IDC_ARROW)`` takes a resource id where a string is declared,
    so the integer has to be cast rather than wrapped - ``c_wchar_p(32512)`` is a
    different thing entirely and fails.
    """
    return ctypes.cast(ctypes.c_void_p(value), ctypes.c_wchar_p)
