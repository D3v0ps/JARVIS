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
        ctypes.c_longlong, _HWND, _UINT, ctypes.c_size_t, ctypes.c_ssize_t
    )
else:  # pragma: no cover - Linux, for import only
    WNDPROC = ctypes.CFUNCTYPE(
        ctypes.c_longlong, _HWND, _UINT, ctypes.c_size_t, ctypes.c_ssize_t
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
