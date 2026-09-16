"""``find_file`` searching inside files, with ADODB and the Windows index faked.

There is no Windows Search service here, so ``win32com.client`` is a module built in
``sys.modules`` whose ``Dispatch`` hands back a recording fake connection. The three
cases that matter are a hit, no hits at all, and a provider that is simply not there -
the last one must fall back to the file-name walk and say "nothing indexed matches"
rather than claiming the file does not exist.
"""

from __future__ import annotations

import logging
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis.tools import file_tools
from jarvis.tools.base import ToolContext
from jarvis.tools.file_tools import NOTHING_INDEXED, find_file

ROW_FIELDS = file_tools.INDEX_FIELDS


# --- fake ADODB -------------------------------------------------------------------------
class FakeField:
    def __init__(self, value: object) -> None:
        self.Value = value


class FakeFields:
    """``record.Fields.Item("System.ItemPathDisplay")``, and nothing else."""

    def __init__(self, row: dict) -> None:
        self._row = row

    def Item(self, name: str) -> FakeField:  # noqa: N802 - COM API name
        if name not in self._row:
            raise KeyError(f"the provider does not expose {name}")
        return FakeField(self._row[name])


class FakeRecordset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.index = 0
        self.closed = False

    @property
    def EOF(self) -> bool:  # noqa: N802
        return self.index >= len(self._rows)

    @property
    def Fields(self) -> FakeFields:  # noqa: N802
        return FakeFields(self._rows[self.index])

    def MoveNext(self) -> None:  # noqa: N802
        self.index += 1


class FakeConnection:
    """Records the connection string and the SQL, then replays canned rows."""

    def __init__(self, rows: list[dict], *, open_error: Exception | None = None,
                 execute_error: Exception | None = None, as_tuple: bool = True) -> None:
        self.rows = rows
        self.open_error = open_error
        self.execute_error = execute_error
        self.as_tuple = as_tuple
        self.opened_with: str | None = None
        self.sql: str | None = None
        self.closed = False

    def Open(self, connection_string: str) -> None:  # noqa: N802
        if self.open_error is not None:
            raise self.open_error
        self.opened_with = connection_string

    def Execute(self, sql: str):  # noqa: N802
        if self.execute_error is not None:
            raise self.execute_error
        self.sql = sql
        record = FakeRecordset(self.rows)
        return (record, len(self.rows)) if self.as_tuple else record

    def Close(self) -> None:  # noqa: N802
        self.closed = True


def row(path: str, modified: datetime | str | None = None, size: int = 2048) -> dict:
    return {ROW_FIELDS[0]: path,
            ROW_FIELDS[1]: datetime.now() if modified is None else modified,
            ROW_FIELDS[2]: size}


@pytest.fixture
def index(monkeypatch):
    """Install a fake ``win32com.client``; returns the connection it will hand out."""

    def install(rows=(), **kwargs) -> FakeConnection:
        connection = FakeConnection(list(rows), **kwargs)
        client = types.ModuleType("win32com.client")
        client.Dispatch = lambda name: connection  # type: ignore[attr-defined]
        package = types.ModuleType("win32com")
        package.client = client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "win32com", package)
        monkeypatch.setitem(sys.modules, "win32com.client", client)
        return connection

    return install


@pytest.fixture
def no_index(monkeypatch):
    """No pywin32 at all: importing ``win32com.client`` raises, as it does on Linux."""
    monkeypatch.delitem(sys.modules, "win32com", raising=False)
    monkeypatch.delitem(sys.modules, "win32com.client", raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("win32com"):
            raise ImportError("No module named 'win32com'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guard)


@pytest.fixture
def profile(tmp_path, monkeypatch, config):
    """A user profile with Desktop/Documents/Downloads and a couple of real files."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("Desktop", "Documents", "Downloads"):
        (tmp_path / name).mkdir()
    (tmp_path / "Documents" / "kvartalsrapport.txt").write_text("nothing", encoding="utf-8")
    (tmp_path / "Desktop" / "notes.md").write_text("nothing", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(config, memory):
    from jarvis.core.scheduler import Scheduler
    from jarvis.core.state import StateBus

    return ToolContext(config=config, memory=memory, logger=logging.getLogger("test"),
                       speak=lambda text: None, confirm=lambda text: True,
                       notify=lambda text: None,
                       scheduler=Scheduler(on_due=lambda job: None), state=StateBus())


# --- a hit ---------------------------------------------------------------------------------
def test_text_inside_a_pdf_is_found_through_the_index(profile, ctx, index):
    connection = index([row(str(profile / "Documents" / "avtal.pdf"))])

    result = find_file(ctx, {"contains": "uppsägningstid"})

    assert result.ok is True
    assert result.summary.startswith("Found avtal.pdf in your Documents folder, sir, "
                                     "with uppsägningstid inside")
    assert result.data["source"] == "index"
    assert result.data["paths"] == [str(profile / "Documents" / "avtal.pdf")]
    assert connection.opened_with == file_tools.INDEX_CONNECTION_STRING
    assert connection.closed is True


def test_the_query_asks_systemindex_the_way_windows_expects(profile, ctx, index):
    connection = index([row(str(profile / "Documents" / "avtal.pdf"))])

    find_file(ctx, {"contains": "faktura 12345", "name": "avtal"})

    sql = connection.sql
    assert sql.startswith(f"SELECT TOP {file_tools.INDEX_RESULTS} ")
    assert "FROM SYSTEMINDEX" in sql
    assert "CONTAINS('\"faktura 12345\"')" in sql
    assert f"SCOPE='file:{str(profile / 'Documents').replace(chr(92), '/')}'" in sql
    assert "System.FileName LIKE '%avtal%'" in sql
    assert "ORDER BY System.DateModified DESC" in sql


def test_several_hits_are_counted_in_the_spoken_sentence(profile, ctx, index):
    now = datetime.now()
    index([row(str(profile / "Documents" / "a.pdf"), now),
           row(str(profile / "Documents" / "b.docx"), now - timedelta(hours=1)),
           row(str(profile / "Desktop" / "c.txt"), now - timedelta(hours=2))])

    result = find_file(ctx, {"contains": "budget"})

    assert result.summary.endswith("along with two other matches.")
    assert result.data["count"] == 3


def test_a_row_whose_date_column_is_missing_is_still_returned(profile, ctx, index):
    index([{ROW_FIELDS[0]: str(profile / "Desktop" / "odd.txt")}])

    result = find_file(ctx, {"contains": "anything"})

    assert result.ok is True
    assert result.data["paths"] == [str(profile / "Desktop" / "odd.txt")]


def test_a_provider_that_returns_a_bare_recordset_is_handled(profile, ctx, index):
    index([row(str(profile / "Documents" / "avtal.pdf"))], as_tuple=False)

    result = find_file(ctx, {"contains": "avtal"})

    assert result.ok is True
    assert result.data["source"] == "index"


def test_an_iso_date_from_the_provider_is_understood(profile, ctx, index):
    index([row(str(profile / "Documents" / "avtal.pdf"), "2026-04-01 09:30:00")])

    result = find_file(ctx, {"contains": "avtal"})

    assert result.ok is True
    assert "avtal.pdf" in result.detail


# --- no hits ---------------------------------------------------------------------------------
def test_an_index_that_knows_of_nothing_says_so_without_inventing_a_file(profile, ctx, index):
    index([])

    result = find_file(ctx, {"contains": "a phrase nobody ever wrote"})

    assert result.ok is True
    assert result.summary == ("Nothing I have indexed contains a phrase nobody ever "
                              "wrote, sir.")
    assert result.data["paths"] == []
    assert result.data["source"] == "index"


# --- the index is not there --------------------------------------------------------------------
def test_without_pywin32_the_search_falls_back_to_the_walk(profile, ctx, no_index):
    result = find_file(ctx, {"contains": "kvartalsrapport"})

    assert result.ok is True
    assert result.data["source"] == "walk"
    # The name walk still found the file, and the sentence does not claim more than that.
    assert result.summary.startswith("I couldn't search inside your files, sir, but "
                                     "kvartalsrapport.txt")
    assert "Windows Search index was not available" in result.detail


def test_a_dead_index_and_no_name_match_says_nothing_indexed_matches(profile, ctx, no_index):
    result = find_file(ctx, {"contains": "something nobody has written down"})

    assert result.ok is True
    assert result.summary == NOTHING_INDEXED
    assert "does not exist" not in result.summary
    assert result.data["source"] == "walk"


@pytest.mark.parametrize("failure", [
    pytest.param({"open_error": OSError("the Search service is not running")},
                 id="the service is stopped"),
    pytest.param({"execute_error": OSError("the catalogue is being rebuilt")},
                 id="the catalogue is rebuilding"),
])
def test_an_index_that_refuses_the_query_falls_back_instead_of_raising(
    profile, ctx, index, failure
):
    connection = index([], **failure)

    result = find_file(ctx, {"contains": "kvartalsrapport"})

    assert result.ok is True
    assert result.data["source"] == "walk"
    assert connection.closed is True


def test_the_index_helper_returns_none_rather_than_an_empty_list_when_it_fails(
    profile, ctx, index
):
    """``None`` means "could not ask"; ``[]`` means "asked, nothing there"."""
    index([], open_error=OSError("no provider"))
    assert file_tools._index_search("x", [profile], None) is None

    index([])
    assert file_tools._index_search("x", [profile], None) == []


# --- modified_since -----------------------------------------------------------------------------
def test_modified_since_filters_the_plain_name_walk(profile, ctx):
    old = profile / "Downloads" / "ancient.txt"
    old.write_text("old", encoding="utf-8")
    import os

    stale = (datetime.now() - timedelta(days=40)).timestamp()
    os.utime(old, (stale, stale))
    fresh = profile / "Downloads" / "brand-new.txt"
    fresh.write_text("new", encoding="utf-8")

    result = find_file(ctx, {"name": "*.txt", "modified_since": "today"})

    assert str(fresh) in result.data["paths"]
    assert str(old) not in result.data["paths"], "a 40-day-old file survived 'today'"
    assert "ancient.txt" not in result.detail


def test_modified_since_narrows_the_index_query_too(profile, ctx, index):
    connection = index([row(str(profile / "Documents" / "recent.pdf"))])

    find_file(ctx, {"contains": "budget", "modified_since": "3 days"})

    assert "System.DateModified > '" in connection.sql


def test_an_index_row_older_than_the_cutoff_is_dropped(profile, ctx, index):
    index([row(str(profile / "Documents" / "old.pdf"),
               datetime.now() - timedelta(days=400))])

    result = find_file(ctx, {"contains": "budget", "modified_since": "today"})

    assert result.data["paths"] == []


@pytest.mark.parametrize("phrase", ["today", "yesterday", "last week", "3 days",
                                    "in the last 2 hours", "2026-04-01", "idag"])
def test_the_date_phrases_a_person_actually_says_are_understood(phrase):
    cutoff, understood = file_tools._parse_since(phrase)
    assert understood is True
    assert isinstance(cutoff, float)
    assert cutoff <= datetime.now().timestamp()


def test_an_empty_modified_since_means_no_filter():
    assert file_tools._parse_since("") == (None, True)


def test_a_date_range_that_makes_no_sense_is_admitted_not_ignored(profile, ctx):
    result = find_file(ctx, {"name": "notes", "modified_since": "when the herring run"})

    assert result.ok is False
    assert result.summary.startswith("I couldn't work out what time you meant")


# --- arguments and safety ------------------------------------------------------------------------
def test_find_file_still_works_with_only_a_name(profile, ctx):
    result = find_file(ctx, {"name": "notes"})

    assert result.ok is True
    assert result.summary.startswith("Found notes.md in your Desktop folder, sir")
    assert result.data["source"] == "walk"


def test_neither_a_name_nor_text_is_a_polite_failure(profile, ctx):
    result = find_file(ctx, {})

    assert result.ok is False
    assert result.summary == "I need a file name or some text to look for, sir."


def test_the_schema_offers_both_new_parameters():
    from jarvis.tools import registry

    registry.load_all()
    schema = registry.get("find_file").to_ollama()["function"]["parameters"]
    assert set(schema["properties"]) == {"name", "contains", "modified_since"}
    assert schema.get("required", []) == []


def test_a_quote_in_the_search_text_cannot_break_out_of_the_sql(profile, ctx, index):
    connection = index([])

    find_file(ctx, {"contains": "x' OR SCOPE='file:C:/Windows"})

    assert "OR SCOPE='file:C:/Windows'" not in connection.sql
    assert connection.sql.count("CONTAINS('\"") == 1
    assert "'" not in connection.sql.split("CONTAINS('\"")[1].split("\"')")[0]


def test_summaries_stay_one_spoken_sentence(profile, ctx, index):
    index([row(str(profile / "Documents" / "avtal.pdf"))])

    for args in ({"contains": "avtal"}, {"name": "notes"}, {"contains": "nothing here"}):
        summary = find_file(ctx, args).summary
        assert "\n" not in summary
        assert "*" not in summary and "- " not in summary
        assert summary.endswith(".")
