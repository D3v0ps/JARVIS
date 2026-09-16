"""Update-JARVIS.bat's script, run against a pretend installation and a pretend GitHub.

This is the one piece of JARVIS that writes over the operator's only copy of himself,
so the tests are mostly about what it refuses to do: never delete, never overwrite
something of his, never trust a path out of an archive.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import update_jarvis as up  # noqa: E402

PREFIX = "JARVIS-main"
SHIPPED = {
    "jarvis/__init__.py": '__version__ = "1.1.0"\n',
    "requirements.txt": "numpy\n",
    "start-jarvis.bat": "@echo off\n",
    "jarvis/desk/window.py": "# the new window\n",
    "config.yaml": "ui:\n  window: true\n  window_mode: auto\nbrain:\n  model: qwen3:8b\n",
}


def archive_bytes(files: dict[str, str] | None = None, prefix: str = PREFIX) -> bytes:
    """A GitHub source archive, as GitHub actually shapes one."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in (SHIPPED if files is None else files).items():
            zf.writestr(f"{prefix}/{name}", text)
    return buffer.getvalue()


@pytest.fixture
def install(tmp_path):
    """An installation from yesterday: older source, and files that are the operator's."""
    root = tmp_path / "JARVIS"
    (root / "jarvis" / "desk").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / "jarvis" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "start-jarvis.bat").write_text("@echo off\n", encoding="utf-8")
    (root / "config.yaml").write_text(
        "ui:\n  overlay: true\n"
        "brain:\n  model: qwen3:14b   # mine, chosen for my card\n", encoding="utf-8")
    (root / "memory.json").write_text('{"facts": [{"text": "his name is Karim"}]}', encoding="utf-8")
    (root / "logs" / "jarvis.log").write_text("yesterday\n", encoding="utf-8")
    return root


def run(install, payload=None, argv=(), **kwargs):
    lines: list[str] = []
    code = up.main(
        list(argv), log=lines.append, root=install,
        fetch=lambda url: archive_bytes() if payload is None else payload, **kwargs
    )
    return code, "\n".join(lines)


# --- what it refuses ------------------------------------------------------------------
def test_it_refuses_a_folder_that_is_not_a_jarvis(tmp_path):
    """Pointed at the wrong folder it would otherwise scatter a copy of JARVIS across it."""
    code, out = run(tmp_path / "somewhere-else")
    assert code == 1
    assert "does not look like a JARVIS installation" in out


def test_an_error_page_in_a_zips_clothing_is_refused(install):
    code, out = run(install, payload=b"<html>404</html>")
    assert code == 1
    assert "not a zip file" in out
    assert (install / "jarvis" / "__init__.py").read_text() == '__version__ = "1.0.0"\n'


def test_an_archive_that_is_not_jarvis_is_refused(install):
    """Somebody else's repository, or a half-built one, must not be unpacked over him."""
    code, out = run(install, payload=archive_bytes({"README.md": "hello\n"}))
    assert code == 1
    assert "does not look like JARVIS" in out


def test_an_archive_with_two_roots_is_refused(install):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in SHIPPED.items():
            zf.writestr(f"A/{name}", text)
        zf.writestr("B/stray.txt", "x")
    code, out = run(install, payload=buffer.getvalue())
    assert code == 1
    assert "one folder" in out


@pytest.mark.parametrize("escape", [
    "../escaped.txt",
    "../../escaped.txt",
    "jarvis/../../escaped.txt",
])
def test_a_member_that_climbs_out_of_the_folder_is_never_written(install, escape):
    """The one attack an archive gets to try. Refused here, not at the filesystem."""
    payload = archive_bytes({**SHIPPED, escape: "owned\n"})
    code, _ = run(install, payload=payload, argv=["--no-deps"])

    assert code == 0
    assert not (install.parent / "escaped.txt").exists()
    assert not (install.parent.parent / "escaped.txt").exists()
    assert not (install / "escaped.txt").exists()


def test_an_absolute_member_is_never_written(install, tmp_path):
    payload = archive_bytes({**SHIPPED, "C:/windows/system32/evil.dll": "owned\n"})
    code, _ = run(install, payload=payload, argv=["--no-deps"])
    assert code == 0
    assert not list(install.rglob("evil.dll"))


# --- what it leaves alone -------------------------------------------------------------
def test_the_operators_own_config_is_never_overwritten(install):
    """His graphics card and his model are in there, and very likely his own edits.

    Every key the new version adds has a default in jarvis/config.py, so an older
    config.yaml reads perfectly well against newer code - which is exactly why this
    file can be left alone instead of merged.
    """
    before = (install / "config.yaml").read_text(encoding="utf-8")

    code, out = run(install, argv=["--no-deps"])

    assert code == 0
    assert (install / "config.yaml").read_text(encoding="utf-8") == before
    assert "qwen3:14b" in (install / "config.yaml").read_text(encoding="utf-8")
    assert "config.yaml" in out


def test_memory_and_logs_and_the_environment_are_untouched(install):
    code, _ = run(install, argv=["--no-deps"])
    assert code == 0
    assert "Karim" in (install / "memory.json").read_text(encoding="utf-8")
    assert (install / "logs" / "jarvis.log").read_text(encoding="utf-8") == "yesterday\n"
    assert (install / ".venv" / "Scripts").is_dir()


def test_nothing_is_ever_deleted(install):
    """A file this version no longer ships stays. Deleting is how an updater eats a home."""
    stray = install / "jarvis" / "my_own_tool.py"
    stray.write_text("# mine\n", encoding="utf-8")

    code, _ = run(install, argv=["--no-deps"])

    assert code == 0
    assert stray.read_text(encoding="utf-8") == "# mine\n"


# --- what it does ---------------------------------------------------------------------
def test_it_brings_in_what_changed_and_what_is_new(install):
    code, out = run(install, argv=["--no-deps"])

    assert code == 0
    assert (install / "jarvis" / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.1.0"\n'
    assert (install / "jarvis" / "desk" / "window.py").read_text(encoding="utf-8") == "# the new window\n"
    assert "Up to date" in out


def test_every_replaced_file_is_backed_up_first(install):
    code, _ = run(install, argv=["--no-deps"])

    assert code == 0
    backups = list((install / "logs").glob("update-backup-*"))
    assert len(backups) == 1
    kept = backups[0] / "jarvis" / "__init__.py"
    assert kept.read_text(encoding="utf-8") == '__version__ = "1.0.0"\n', "the old version"
    assert not (backups[0] / "jarvis" / "desk" / "window.py").exists(), "a new file has no old one"


def test_a_half_written_file_is_never_left_behind(install):
    code, _ = run(install, argv=["--no-deps"])
    assert code == 0
    assert not list(install.rglob("*.new"))


def test_an_installation_that_is_current_changes_nothing(install):
    run(install, argv=["--no-deps"])
    for stale in (install / "logs").glob("update-backup-*"):
        for path in stale.rglob("*"):
            if path.is_file():
                path.unlink()

    code, out = run(install, argv=["--no-deps"])

    assert code == 0
    assert "already up to date" in out


def test_check_writes_absolutely_nothing(install):
    before = {p: p.read_bytes() for p in install.rglob("*") if p.is_file()}

    code, out = run(install, argv=["--check"])

    assert code == 0
    assert "Nothing was changed" in out
    after = {p: p.read_bytes() for p in install.rglob("*") if p.is_file()}
    assert after == before


def test_settings_you_have_not_got_are_reported_and_not_written(install):
    """Telling him a setting exists is help; editing his config.yaml behind his back is not."""
    code, out = run(install, argv=["--check"])

    assert code == 0
    assert "ui.window" in out, "the new key inside a section he already has"
    assert "ui.window_mode" in out
    assert "window" not in (install / "config.yaml").read_text(encoding="utf-8")


def test_a_file_that_is_in_use_is_named_rather_than_swallowed(install, monkeypatch):
    """On Windows the running JARVIS.exe cannot be replaced, and silence would be cruel."""
    real = up.os.replace

    def refuse(src, dst):
        if str(dst).endswith("__init__.py"):
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(up.os, "replace", refuse)
    code, out = run(install, argv=["--no-deps"])

    assert code == 1
    assert "is in use" in out
    assert "close JARVIS" in out.lower() or "Close him" in out


# --- the pieces -------------------------------------------------------------------------
@pytest.mark.parametrize("path, preserved", [
    ("config.yaml", True),
    ("memory.json", True),
    ("places.json", True),
    ("logs/jarvis.log", True),
    (".venv/Scripts/python.exe", True),
    ("models/kokoro-v1.0.onnx", True),
    ("jarvis/__init__.py", False),
    ("jarvis/desk/server.py", False),
    ("README.md", False),
])
def test_what_counts_as_the_operators_own(path, preserved):
    assert up.is_preserved(path) is preserved


def test_a_download_far_too_large_is_refused():
    """The source is about two megabytes; anything near the cap is not it."""
    class Huge:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return b"x" * n

    with pytest.raises(RuntimeError, match="larger"):
        up.download("https://example.invalid", opener=lambda url, timeout=0: Huge())


def test_github_saying_no_is_reported_in_words(install):

    def boom(url):
        raise RuntimeError("GitHub answered 404 for " + url)

    lines: list[str] = []
    code = up.main(["--check"], log=lines.append, root=install, fetch=boom)
    assert code == 1
    assert "404" in "\n".join(lines)
    assert "internet connection" in "\n".join(lines)
