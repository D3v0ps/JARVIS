"""Finding a business's number: the OSM chain, the web fallback, and honesty.

Everything here runs on a bare Linux box: no network, no Windows, no hardware. Every
HTTP call goes through a fake ``requests.get`` that serves canned Nominatim, Overpass
and web-page payloads, and the module-level rate limiters are reset and their sleeps
recorded rather than slept.

The rule these tests exist to enforce: JARVIS never speaks a number he did not
actually retrieve.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import requests

from jarvis.tools import places_tools as places
from jarvis.tools.base import Tier
from jarvis.tools.registry import REGISTRY

# --- canned payloads -----------------------------------------------------------------
NOMINATIM_WITH_PHONE = [
    {
        "display_name": "Tandlakare Soder, Gotgatan 1, Stockholm",
        "name": "Tandlakare Soder",
        "lat": "59.3159",
        "lon": "18.0722",
        "extratags": {
            "phone": "08-123 45 67",
            "website": "https://tandlakaresoder.se",
            "opening_hours": "Mo-Fr 08:00-17:00",
        },
    }
]

NOMINATIM_NO_PHONE = [
    {
        "display_name": "Tandlakare Soder, Gotgatan 1, Stockholm",
        "name": "Tandlakare Soder",
        "lat": "59.3159",
        "lon": "18.0722",
        "extratags": {"website": "https://tandlakaresoder.se"},
    }
]

NOMINATIM_BARE = [
    {
        "display_name": "Tandlakare Soder, Gotgatan 1, Stockholm",
        "name": "Tandlakare Soder",
        "lat": "59.3159",
        "lon": "18.0722",
        "extratags": {},
    }
]

OVERPASS_WITH_PHONE = {
    "elements": [
        {
            "type": "node",
            "lat": 59.3161,
            "lon": 18.0725,
            "tags": {"name": "Tandlakare Soder", "contact:phone": "+46 8 123 45 67",
                     "opening_hours": "Mo-Fr 08:00-17:00"},
        }
    ]
}

OVERPASS_EMPTY = {"elements": []}

WEBSITE_HTML = """
<html><body><h1>Tandlakare Soder</h1>
<p>Ring oss: <a href="tel:+46 8 123 45 67">08-123 45 67</a></p>
</body></html>
"""


# --- fake HTTP -----------------------------------------------------------------------
class FakeResponse:
    """The slice of ``requests.Response`` this module touches."""

    def __init__(self, payload=None, text: str = "", status: int = 200) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON in this response")
        return self._payload


class FakeHTTP:
    """Routes GETs by URL substring and records every call made."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, object]] = []
        self.calls: list[dict] = []

    def add(self, needle: str, response: object) -> "FakeHTTP":
        self.routes.append((needle, response))
        return self

    def urls(self) -> list[str]:
        return [call["url"] for call in self.calls]

    def __call__(self, url, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {},
                           "timeout": timeout})
        for needle, response in self.routes:
            if needle in url:
                if isinstance(response, Exception):
                    raise response
                return response
        return FakeResponse(payload=[], text="")


class FakeConfig:
    """Just enough Config for the tools: ``get`` with a default."""

    def __init__(self, values: dict | None = None) -> None:
        self.values = values or {}

    def get(self, dotted: str, default=None):
        return self.values.get(dotted, default)

    def resolve_path(self, value):
        return Path(value)


@pytest.fixture(autouse=True)
def quiet_limiters(monkeypatch):
    """Reset the module-level limiters and record sleeps instead of serving them."""
    places.NOMINATIM_LIMITER._last = 0.0
    places.OVERPASS_LIMITER._last = 0.0
    slept: list[float] = []
    monkeypatch.setattr(places.time, "sleep", slept.append)
    yield slept
    places.NOMINATIM_LIMITER._last = 0.0
    places.OVERPASS_LIMITER._last = 0.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may reach the real internet, even by accident."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args!r}")

    monkeypatch.setattr(places.requests, "get", refuse)
    monkeypatch.setattr(places, "_web_rows", lambda query: [])


@pytest.fixture
def ctx(tmp_path):
    """A minimal ToolContext stand-in whose memory.json lives in tmp_path."""
    return types.SimpleNamespace(
        config=FakeConfig({"tools.default_city": "Stockholm", "tools.home_region": "SE"}),
        memory=types.SimpleNamespace(path=tmp_path / "memory.json"),
        logger=None, speak=lambda text: None, confirm=lambda text: True,
        notify=lambda text: None, scheduler=None, state=None,
    )


def http(monkeypatch, fake: FakeHTTP) -> FakeHTTP:
    """Install a FakeHTTP as ``requests.get`` for the module under test."""
    monkeypatch.setattr(places.requests, "get", fake)
    return fake


def cache_file(ctx) -> Path:
    return Path(ctx.memory.path).parent / places.CACHE_FILENAME


# ======================================================================================
# Registration and importability
# ======================================================================================
def test_the_module_imports_with_no_windows_and_no_optional_packages():
    """The development box has neither phonenumbers nor ddgs; the import must hold."""
    assert places.find_business.__module__ == "jarvis.tools.places_tools"
    for forbidden in ("pycaw", "comtypes", "win32clipboard", "sounddevice", "flask"):
        assert forbidden not in sys.modules or forbidden == ""


def test_both_tools_are_registered_as_safe():
    for name in ("find_business", "recall_business"):
        spec = REGISTRY.get(name)
        assert spec is not None, f"{name} never registered"
        assert spec.tier is Tier.SAFE
        assert spec.parameters["type"] == "object"


# ======================================================================================
# Phone normalisation
# ======================================================================================
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("08-123 45 67", "+4681234567"),
        ("+46 8 123 45 67", "+4681234567"),
        ("0701234567", "+46701234567"),
        ("070-123 45 67", "+46701234567"),
        ("0046 8 123 45 67", "+4681234567"),
        ("08 123 45 67", "+4681234567"),
    ],
)
def test_swedish_formats_normalise_to_the_same_e164(raw, expected):
    number, method = places.normalise_phone(raw, "SE")
    assert number == expected
    assert method in {"phonenumbers", "regex"}


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "hello", "12", "org.nr 556677-8899", "priset ar 1 495 kronor"],
)
def test_nonsense_never_becomes_a_number(raw):
    assert places.normalise_phone(raw, "SE") == ("", "")


def test_a_foreign_region_uses_its_own_country_code():
    assert places.normalise_phone("0123 456 7890", "GB")[0] == "+441234567890"


def test_phonenumbers_is_used_when_it_is_installed(monkeypatch):
    """The library wins over the regex, and its verdict is reported in ``method``."""
    calls: list[tuple[str, str]] = []

    class FakeFormat:
        E164 = 0

    fake = types.SimpleNamespace(
        PhoneNumberFormat=FakeFormat,
        parse=lambda text, region: calls.append((text, region)) or "parsed",
        is_valid_number=lambda parsed: True,
        format_number=lambda parsed, fmt: "+46812345678",
    )
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert places.normalise_phone("08-1234 5678", "SE") == ("+46812345678", "phonenumbers")
    assert calls == [("08-1234 5678", "SE")]


def test_a_number_phonenumbers_rejects_is_not_invented_by_the_regex(monkeypatch):
    fake = types.SimpleNamespace(
        PhoneNumberFormat=types.SimpleNamespace(E164=0),
        parse=lambda text, region: "parsed",
        is_valid_number=lambda parsed: False,
        format_number=lambda parsed, fmt: "+46000000000",
    )
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert places.normalise_phone("08-123 45 67", "SE") == ("", "")


def test_a_broken_phonenumbers_falls_back_to_no_number_rather_than_raising(monkeypatch):
    def explode(text, region):
        raise ValueError("NumberParseException")

    fake = types.SimpleNamespace(PhoneNumberFormat=types.SimpleNamespace(E164=0),
                                 parse=explode, is_valid_number=lambda p: True,
                                 format_number=lambda p, f: "+46000000000")
    monkeypatch.setitem(sys.modules, "phonenumbers", fake)

    assert places.normalise_phone("08-123 45 67", "SE") == ("", "")


def test_the_number_is_spoken_as_dialable_digits():
    spoken = places.spoken_phone("+4681234567", "SE")
    assert spoken == "zero eight one, two three four, five six seven"
    assert not any(char.isdigit() for char in spoken)


def test_a_foreign_number_keeps_its_plus_when_spoken():
    assert places.spoken_phone("+441234567890", "SE").startswith("plus four four")


# ======================================================================================
# The rate limiter
# ======================================================================================
def test_the_rate_limiter_actually_delays_the_second_call(monkeypatch):
    clock = {"now": 1000.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(places.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(places.time, "sleep", fake_sleep)

    limiter = places._RateLimiter(1.0, "test")
    assert limiter.wait() == 0.0, "the first call must not be delayed"
    assert slept == []

    delay = limiter.wait()
    assert delay == pytest.approx(1.0)
    assert slept == [pytest.approx(1.0)]

    clock["now"] += 5.0  # long enough that the third call is free again
    assert limiter.wait() == 0.0


def test_two_nominatim_lookups_in_a_row_are_throttled(ctx, monkeypatch, quiet_limiters):
    """Two searches inside one lookup must cost a real second of waiting."""
    fake = http(monkeypatch, FakeHTTP()
                .add("nominatim", FakeResponse(payload=[]))
                .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))

    places.find_business(ctx, {"name": "dentist", "near": "Stockholm"})

    assert len([url for url in fake.urls() if "nominatim" in url]) == 2
    assert quiet_limiters and quiet_limiters[0] > 0.0, "the second call was not throttled"


def test_every_osm_request_identifies_the_project(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP().add("nominatim",
                                            FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    places.find_business(ctx, {"name": "Tandlakare Soder"})

    agent = fake.calls[0]["headers"]["User-Agent"]
    assert "JARVIS" in agent and "http" in agent, "the terms require an identifying UA"


# ======================================================================================
# find_business: the happy path
# ======================================================================================
def test_a_phone_tag_in_nominatim_answers_without_touching_anything_else(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP().add("nominatim",
                                            FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert result.ok
    assert result.data["phone"] == "+4681234567"
    assert result.data["source"] == "nominatim"
    assert "zero eight one, two three four, five six seven" in result.summary
    assert fake.urls() == [places.NOMINATIM_URL], "no fallback should have been needed"


def test_the_spoken_summary_is_one_plain_sentence(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP().add("nominatim", FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    summary = places.find_business(ctx, {"name": "Tandlakare Soder"}).summary

    assert summary.endswith("sir.")
    assert summary.count(".") == 1
    assert not any(token in summary for token in ("*", "#", "\n", "- ", "http"))


def test_opening_hours_are_spoken_only_when_they_were_asked_for(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP().add("nominatim", FakeResponse(payload=NOMINATIM_WITH_PHONE)))
    quiet = places.find_business(ctx, {"name": "Tandlakare Soder"})
    assert "open" not in quiet.summary

    # Second call comes from the cache, which carries the hours just the same.
    asked = places.find_business(ctx, {"name": "Tandlakare Soder", "hours": True})
    assert "Monday to Friday 8 to 17" in asked.summary


def test_the_search_is_anchored_on_the_home_city_when_the_user_did_not_say(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP().add("nominatim",
                                            FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert fake.calls[0]["params"]["q"] == "Tandlakare Soder Stockholm"
    assert fake.calls[0]["params"]["extratags"] == 1


def test_an_empty_name_is_refused_before_any_request_is_made(ctx):
    result = places.find_business(ctx, {"name": "   "})
    assert not result.ok
    assert "name" in result.summary


# ======================================================================================
# find_business: the fallbacks
# ======================================================================================
def test_overpass_answers_when_the_geocoded_hit_has_no_phone_tag(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP()
                .add("nominatim", FakeResponse(payload=NOMINATIM_BARE))
                .add("overpass", FakeResponse(payload=OVERPASS_WITH_PHONE)))

    result = places.find_business(ctx, {"name": "dentist", "near": "Stockholm"})

    assert result.ok
    assert result.data["phone"] == "+4681234567"
    assert result.data["source"] == "overpass"
    overpass_query = [call for call in fake.calls if "overpass" in call["url"]][0]
    assert 'amenity"="dentist' in overpass_query["params"]["data"], "category not used"
    assert "around:" in overpass_query["params"]["data"]


def test_the_business_own_website_carries_the_number_when_osm_does_not(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP()
                .add("nominatim", FakeResponse(payload=NOMINATIM_NO_PHONE))
                .add("overpass", FakeResponse(payload=OVERPASS_EMPTY))
                .add("tandlakaresoder.se", FakeResponse(text=WEBSITE_HTML)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert result.ok
    assert result.data["phone"] == "+4681234567"
    assert "tandlakaresoder.se" in result.data["source"]
    assert any("tandlakaresoder.se" in url for url in fake.urls())


def test_the_web_search_carries_it_when_osm_knows_nothing(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload=[]))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))
    monkeypatch.setattr(places, "_web_rows", lambda query: [
        {"title": "Frisor Sodermalm", "url": "https://frisorsodermalm.se",
         "body": "Boka tid pa 08-987 65 43"},
    ])

    result = places.find_business(ctx, {"name": "Frisor Sodermalm"})

    assert result.ok
    assert result.data["phone"] == "+468987 6543".replace(" ", "")
    assert "web snippet" in result.data["source"]
    assert "zero eight nine, eight seven six, five four three" in result.summary


def test_the_forbidden_directories_are_never_read(ctx, monkeypatch):
    """hitta.se and eniro.se have the number and forbid this; we skip them entirely."""
    fake = http(monkeypatch, FakeHTTP()
                .add("nominatim", FakeResponse(payload=[]))
                .add("overpass", FakeResponse(payload=OVERPASS_EMPTY))
                .add("frisorsodermalm.se", FakeResponse(text=WEBSITE_HTML)))
    monkeypatch.setattr(places, "_web_rows", lambda query: [
        {"title": "Frisor - hitta.se", "url": "https://www.hitta.se/frisor",
         "body": "Telefon 08-987 65 43"},
        {"title": "Frisor Sodermalm", "url": "https://frisorsodermalm.se", "body": ""},
    ])

    result = places.find_business(ctx, {"name": "Frisor Sodermalm"})

    assert result.ok
    assert result.data["phone"] == "+4681234567", "the number came from the allowed site"
    assert not any("hitta.se" in url or "eniro.se" in url for url in fake.urls())


def test_a_missing_search_package_is_not_an_excuse_to_guess(ctx, monkeypatch):
    """No ddgs installed is the normal state here; the answer must still be honest."""
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload=[]))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))
    monkeypatch.setattr(places, "_web_rows", lambda query: [])

    result = places.find_business(ctx, {"name": "Frisor Sodermalm"})

    assert not result.ok
    assert "found no listing" in result.summary
    assert not any(digit in result.summary for digit in places._DIGITS)


# ======================================================================================
# find_business: the failure paths
# ======================================================================================
def test_a_hit_with_no_number_anywhere_is_not_reported_as_success(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload=NOMINATIM_BARE))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert "no telephone number" in result.summary
    assert result.data["phone"] == ""
    assert result.data["spoken_phone"] == ""


def test_a_network_error_says_so_instead_of_inventing_a_number(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", requests.ConnectionError("no route to host"))
         .add("overpass", requests.ConnectionError("no route to host")))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert result.summary == places.NETWORK_FAILURE
    assert "ConnectionError" in result.detail


def test_a_timeout_on_the_way_out_is_still_a_calm_sentence(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP().add("nominatim", requests.Timeout("timed out")))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert result.summary == places.NETWORK_FAILURE


def test_garbage_json_from_nominatim_is_survived(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload={"error": "rate limited"}))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert "not a list" in result.detail


def test_an_http_error_page_does_not_become_a_phone_number(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload=NOMINATIM_NO_PHONE))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY))
         .add("tandlakaresoder.se", FakeResponse(text="down", status=503)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert result.data["phone"] == ""


def test_the_detail_says_when_the_regex_did_the_normalising(ctx, monkeypatch):
    """phonenumbers is not installed here, and the log must admit it."""
    monkeypatch.setitem(sys.modules, "phonenumbers", None)
    http(monkeypatch, FakeHTTP().add("nominatim", FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    result = places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert "normalised with: a regex" in result.detail


# ======================================================================================
# The cache
# ======================================================================================
def test_a_looked_up_business_is_answered_from_the_cache_next_time(ctx, monkeypatch):
    fake = http(monkeypatch, FakeHTTP().add("nominatim",
                                            FakeResponse(payload=NOMINATIM_WITH_PHONE)))
    first = places.find_business(ctx, {"name": "Tandlakare Soder"})
    assert first.ok and len(fake.calls) == 1

    second = places.find_business(ctx, {"name": "tandlakare  soder"})

    assert second.ok
    assert second.data["phone"] == first.data["phone"]
    assert second.data["cached"] is True
    assert len(fake.calls) == 1, "the cached lookup went to the network anyway"


def test_the_cache_file_lands_beside_memory_json_and_is_valid_json(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP().add("nominatim", FakeResponse(payload=NOMINATIM_WITH_PHONE)))

    places.find_business(ctx, {"name": "Tandlakare Soder"})

    path = cache_file(ctx)
    assert path.exists()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["version"] == places.CACHE_VERSION
    assert stored["places"][0]["phone"] == "+4681234567"
    assert stored["places"][0]["query"] == "Tandlakare Soder"
    leftovers = [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], "the atomic write left a temp file behind"


def test_a_business_without_a_number_is_not_cached(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP()
         .add("nominatim", FakeResponse(payload=NOMINATIM_BARE))
         .add("overpass", FakeResponse(payload=OVERPASS_EMPTY)))

    places.find_business(ctx, {"name": "Tandlakare Soder"})

    assert not cache_file(ctx).exists()


def test_the_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "places.json"
    cache = places.PlaceCache(path)
    cache.put(places.Place(name="Frisor Sodermalm", phone="+46701234567",
                           source="nominatim", opening_hours="Mo-Fr 10:00-18:00"))

    reopened = places.PlaceCache(path)
    found = reopened.find("frisor sodermalm")

    assert found is not None
    assert found.phone == "+46701234567"
    assert found.opening_hours == "Mo-Fr 10:00-18:00"
    assert reopened.find("a place nobody looked up") is None


def test_storing_the_same_business_twice_replaces_it(tmp_path):
    path = tmp_path / "places.json"
    cache = places.PlaceCache(path)
    cache.put(places.Place(name="Frisor", phone="+46701111111"))
    cache.put(places.Place(name="frisor", phone="+46702222222"))

    assert len(places.PlaceCache(path).all()) == 1
    assert places.PlaceCache(path).find("Frisor").phone == "+46702222222"


@pytest.mark.parametrize(
    "content",
    [pytest.param("", id="empty"),
     pytest.param("{ not json", id="truncated"),
     pytest.param('{"places": "a string"}', id="wrong shape"),
     pytest.param('[{"no_name": 1}]', id="unusable rows")],
)
def test_a_damaged_cache_degrades_to_empty_instead_of_raising(tmp_path, content):
    path = tmp_path / "places.json"
    path.write_text(content, encoding="utf-8")

    cache = places.PlaceCache(path)

    assert cache.all() == []
    assert cache.find("anything") is None


def test_a_cache_that_cannot_be_written_is_logged_not_raised(tmp_path):
    path = tmp_path / "missing-dir" / "places.json"
    path.parent.mkdir()
    path.parent.chmod(0o500)
    try:
        cache = places.PlaceCache(path)
        cache.put(places.Place(name="Frisor", phone="+46701234567"))
    finally:
        path.parent.chmod(0o700)
    assert cache.all()[0].phone == "+46701234567", "the in-memory entry survived"


# ======================================================================================
# recall_business
# ======================================================================================
def test_recall_reads_the_cache_without_any_network(ctx, monkeypatch):
    http(monkeypatch, FakeHTTP().add("nominatim", FakeResponse(payload=NOMINATIM_WITH_PHONE)))
    places.find_business(ctx, {"name": "Tandlakare Soder"})
    monkeypatch.setattr(places.requests, "get",
                        lambda *a, **k: pytest.fail("recall must not use the network"))

    result = places.recall_business(ctx, {"name": "Tandlakare"})

    assert result.ok
    assert "zero eight one, two three four, five six seven" in result.summary


def test_recall_with_nothing_stored_says_so(ctx):
    result = places.recall_business(ctx, {})
    assert result.ok
    assert "haven't looked up any businesses" in result.summary
    assert result.data["places"] == []


def test_recall_lists_what_is_stored_when_no_name_is_given(ctx, monkeypatch):
    cache = places.PlaceCache(cache_file(ctx))
    cache.put(places.Place(name="Frisor Sodermalm", phone="+46701111111",
                           looked_up="2026-01-01T10:00:00"))
    cache.put(places.Place(name="Tandlakare Soder", phone="+46702222222",
                           looked_up="2026-02-01T10:00:00"))

    result = places.recall_business(ctx, {})

    assert result.ok
    assert result.summary == ("I have numbers for Tandlakare Soder and "
                              "Frisor Sodermalm, sir.")
    assert result.data["count"] == 2


def test_recall_of_an_unknown_business_admits_it(ctx):
    places.PlaceCache(cache_file(ctx)).put(
        places.Place(name="Frisor Sodermalm", phone="+46701111111"))

    result = places.recall_business(ctx, {"name": "Tandlakare Soder"})

    assert not result.ok
    assert "nothing stored" in result.summary


# ======================================================================================
# Small pieces
# ======================================================================================
@pytest.mark.parametrize(
    "spoken, tag",
    [("dentist", "amenity=dentist"), ("Tandlakare Soder", "amenity=dentist"),
     ("a barber near me", "shop=hairdresser"), ("apotek", "amenity=pharmacy"),
     ("Restaurang Prinsen", "amenity=restaurant"), ("Ikea", "")],
)
def test_categories_map_to_osm_tags_in_both_languages(spoken, tag):
    assert places._category_for(spoken) == tag


def test_opening_hours_are_rewritten_for_speech():
    spoken = places._speakable_hours("Mo-Fr 08:00-17:00; Sa 10:00-14:00")
    assert spoken == "Monday to Friday 8 to 17, Saturday 10 to 14"
    assert places._speakable_hours("") == ""


def test_a_tel_link_beats_a_stray_number_in_the_page():
    html = '<p>Org 556677-8899</p><a href="tel:+46812345 67">call</a>'
    assert places._phone_from_text(html, "SE") == "+4681234567"


def test_a_page_with_no_number_yields_nothing():
    assert places._phone_from_text("<p>Welcome to our website</p>", "SE") == ""


def test_the_cache_path_follows_memory_json(ctx):
    assert places._cache_path(ctx) == Path(ctx.memory.path).parent / "places.json"


def test_the_cache_path_survives_a_context_without_a_memory(monkeypatch):
    bare = types.SimpleNamespace(memory=None, config=FakeConfig())
    assert places._cache_path(bare).name == "places.json"


def test_the_home_region_defaults_to_sweden():
    bare = types.SimpleNamespace(config=FakeConfig())
    assert places._home_region(bare) == "SE"
    assert places._home_region(types.SimpleNamespace(config=None)) == "SE"


def test_a_configured_home_region_is_honoured(monkeypatch):
    other = types.SimpleNamespace(config=FakeConfig({"tools.home_region": "no"}),
                                  memory=None)
    assert places._home_region(other) == "NO"
