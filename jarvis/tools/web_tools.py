"""Web tools: a keyless DuckDuckGo search and a keyless Open-Meteo forecast.

Neither tool needs an account or an API key. ``requests`` is a hard dependency of
JARVIS so it is imported at module level; the search package is optional and is
imported lazily in :func:`_load_ddgs`, so this module imports on a bare Linux box.

Both tools answer in one spoken sentence. Titles, links, raw snippets and the full
forecast go into ``detail``, which is logged and never spoken.
"""

from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from jarvis.brain.sentences import clean_for_speech
from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["web_search", "weather", "WEATHER_CODES", "describe_weather_code",
           "GEOCODING_URL", "FORECAST_URL"]

_log = get_logger("tools.web")

#: Open-Meteo endpoints; both are free and need no key.
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Seconds for one HTTP round trip, and the bounds on ``tools.search_results``.
HTTP_TIMEOUT = 10
DEFAULT_SEARCH_RESULTS = 3
MIN_SEARCH_RESULTS = 1
MAX_SEARCH_RESULTS = 10

#: The spoken part of a search answer is trimmed to about this many characters.
_SPOKEN_CLAUSE_CHARS = 190

#: What JARVIS says when the network is simply not there.
NETWORK_FAILURE = "I couldn't reach the web just now, sir."
WEATHER_FAILURE = "I couldn't reach the weather service just now, sir."

#: WMO present-weather codes (ww, 0-99) as phrases that fit "It's ... in <place>".
WEATHER_CODES: dict[int, str] = {
    0: 'clear', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast', 4: 'smoky', 5: 'hazy',
    6: 'dusty', 7: 'dusty and windy', 8: 'dusty with whirls of sand',
    9: 'blowing up a duststorm', 10: 'misty', 11: 'patchily foggy', 12: 'foggy in patches',
    13: 'flickering with distant lightning', 14: 'raining without it reaching the ground',
    15: 'raining in the distance', 16: 'raining close by', 17: 'thundery without rain',
    18: 'squally', 19: 'threatening a funnel cloud', 20: 'clearing after drizzle',
    21: 'clearing after rain', 22: 'clearing after snow', 23: 'clearing after sleet',
    24: 'clearing after freezing rain', 25: 'clearing after rain showers',
    26: 'clearing after snow showers', 27: 'clearing after hail', 28: 'clearing after fog',
    29: 'clearing after a thunderstorm', 30: 'easing out of a duststorm',
    31: 'stuck in a duststorm', 32: 'building into a duststorm',
    33: 'easing out of a heavy duststorm', 34: 'in a heavy duststorm',
    35: 'building into a heavy duststorm', 36: 'drifting with light snow',
    37: 'drifting with heavy snow', 38: 'blowing snow about', 39: 'blowing heavy snow about',
    40: 'foggy in the distance', 41: 'foggy in patches', 42: 'thinning out of fog',
    43: 'thinning out of dense fog', 44: 'foggy', 45: 'foggy',
    46: 'thinning out of freezing fog', 47: 'densely foggy', 48: 'foggy and freezing',
    49: 'densely foggy and freezing', 50: 'drizzling on and off', 51: 'drizzling lightly',
    52: 'drizzling', 53: 'drizzling steadily', 54: 'drizzling heavily on and off',
    55: 'drizzling heavily', 56: 'drizzling and freezing', 57: 'drizzling and freezing hard',
    58: 'drizzling with light rain', 59: 'drizzling with rain', 60: 'raining on and off',
    61: 'raining lightly', 62: 'raining', 63: 'raining steadily',
    64: 'raining heavily on and off', 65: 'raining heavily', 66: 'raining and freezing',
    67: 'raining and freezing hard', 68: 'sleeting lightly', 69: 'sleeting',
    70: 'snowing on and off', 71: 'snowing lightly', 72: 'snowing', 73: 'snowing steadily',
    74: 'snowing heavily on and off', 75: 'snowing heavily', 76: 'sparkling with diamond dust',
    77: 'snowing grains of snow', 78: 'snowing ice crystals', 79: 'raining ice pellets',
    80: 'showery with light rain', 81: 'showery with rain', 82: 'showery with violent rain',
    83: 'showery with light sleet', 84: 'showery with heavy sleet',
    85: 'showery with light snow', 86: 'showery with heavy snow',
    87: 'showery with light soft hail', 88: 'showery with heavy soft hail',
    89: 'showery with light hail', 90: 'showery with heavy hail',
    91: 'raining lightly after a thunderstorm', 92: 'raining heavily after a thunderstorm',
    93: 'snowing lightly after a thunderstorm', 94: 'snowing heavily after a thunderstorm',
    95: 'thundery', 96: 'thundery with light hail', 97: 'thundery and heavy',
    98: 'thundery with a duststorm', 99: 'thundery with heavy hail',
}

#: Domain labels that are never the brand part of a host name.
_GENERIC_LABELS = {"www", "m", "mobile", "amp", "co", "com", "org", "net", "ac",
                   "gov", "edu", "gob", "go", "or", "ne"}

#: Keys DuckDuckGo wrappers have used over the years for the same three fields.
_TITLE_KEYS = ("title", "heading", "name")
_URL_KEYS = ("href", "url", "link")
_BODY_KEYS = ("body", "snippet", "description", "abstract", "excerpt")

_VOWELS = set("aeiouy")
_LEADING_DATE_RE = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}\s*(?:\.{3}|…|—|-)\s*")
_LEADING_ELLIPSIS_RE = re.compile(r"^\s*(?:\.{3}|…)\s*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

#: Trailing characters removed from a spoken clause, which continues into "according to".
_CLAUSE_TRIM = " .,;:!?-"

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"]


def _spoken_number(value: float) -> str:
    """Render a temperature as words, so the sentence reads well aloud."""
    number = int(round(float(value)))
    if number < 0:
        return f"minus {_spoken_number(-number)}"
    if number > 999:
        return str(number)
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"
    hundreds, rest = divmod(number, 100)
    head = f"{_UNITS[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_spoken_number(rest)}"


def describe_weather_code(code: Any) -> str:
    """The plain-English phrase for a WMO weather code, never raising."""
    try:
        return WEATHER_CODES.get(int(code), "hard to pin down")
    except (TypeError, ValueError):
        return "hard to pin down"


def _load_ddgs() -> Callable[..., Any] | None:
    """Import the DuckDuckGo client lazily, new package name first.

    Returns the ``DDGS`` class, or ``None`` when neither package is installed —
    the normal state on the Linux development box.
    """
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
        return DDGS
    except ImportError:
        _log.debug("The 'ddgs' package is missing; trying 'duckduckgo_search'.")
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        return DDGS
    except ImportError:
        _log.warning("No DuckDuckGo search package is installed; web search is off.")
        return None


def _result_count(ctx: ToolContext) -> int:
    """How many results to ask for, from ``tools.search_results``."""
    try:
        wanted = int(float(ctx.config.get("tools.search_results", DEFAULT_SEARCH_RESULTS)))
    except (AttributeError, TypeError, ValueError):
        _log.debug("Unusable tools.search_results; using the default", exc_info=True)
        wanted = DEFAULT_SEARCH_RESULTS
    return max(MIN_SEARCH_RESULTS, min(MAX_SEARCH_RESULTS, wanted))


def _field(row: dict, keys: tuple[str, ...]) -> str:
    """First non-empty string among ``keys`` in a result row."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _normalise_results(raw: Any) -> list[dict[str, str]]:
    """Turn whatever the search package returned into ``{title, url, body}`` rows."""
    try:
        items = list(raw) if raw is not None else []
    except TypeError:
        _log.warning("The search package returned something I cannot iterate over.")
        return []
    rows = [{"title": _field(item, _TITLE_KEYS), "url": _field(item, _URL_KEYS),
             "body": _field(item, _BODY_KEYS)} for item in items if isinstance(item, dict)]
    return [row for row in rows if any(row.values())]


def _brand(labels: list[str]) -> str:
    """The brand label of a host name, skipping ``www``, ``co.uk`` and friends."""
    while labels and labels[0] in _GENERIC_LABELS:
        labels.pop(0)
    if len(labels) < 2:
        return labels[0] if labels else ""
    if labels[-2] not in _GENERIC_LABELS:
        return labels[-2]
    return labels[-3] if len(labels) >= 3 else labels[0]


def _source_name(url: str, title: str) -> str:
    """A short, speakable name for where a result came from."""
    try:
        host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    except ValueError:
        host = ""
    brand = _brand([part for part in host.lower().split(".") if part])
    if not brand:
        # No usable host: fall back to the publisher suffix of the title, if any.
        parts = re.split(r"\s+[-–|]\s+", title)
        return (parts[-1].strip() if len(parts) > 1 else "") or "the web"
    if not (set(brand) & _VOWELS) and len(brand) <= 4:
        return brand.upper()
    return brand[:1].upper() + brand[1:]


def _first_clause(text: str, limit: int = _SPOKEN_CLAUSE_CHARS) -> str:
    """The most useful opening clause of a snippet, short enough to speak."""
    clean = _LEADING_ELLIPSIS_RE.sub("", clean_for_speech(text or ""))
    clean = _LEADING_DATE_RE.sub("", clean).strip()
    if not clean:
        return ""
    candidate = _SENTENCE_END_RE.split(clean)[0].strip() or clean
    if len(candidate) <= limit:
        return candidate.rstrip(_CLAUSE_TRIM)
    window = candidate[:limit]
    for separator in (";", ",", " — ", " - "):
        cut = window.rfind(separator)
        if cut > limit // 2:
            return window[:cut].rstrip(_CLAUSE_TRIM)
    cut = window.rfind(" ")
    return window[: cut if cut >= limit // 2 else limit].rstrip(_CLAUSE_TRIM) + "…"


def _search_detail(query: str, rows: list[dict[str, str]]) -> str:
    """The full, unspoken record of a search — titles, links and snippets."""
    lines = [f"query: {query}"]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. {row['title'] or '(untitled)'}")
        if row["url"]:
            lines.append(f"   {row['url']}")
        if row["body"]:
            lines.append(f"   {row['body']}")
    return "\n".join(lines)


@tool(
    "web_search",
    description=(
        "Search the web with DuckDuckGo and answer from the best result. Use it for "
        "facts you do not know, current events, prices and opening hours."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, phrased as a search query.",
            }
        },
        "required": ["query"],
    },
    tier=Tier.SAFE,
)
def web_search(ctx: ToolContext, args: dict) -> ToolResult:
    """Search the web and answer in one sentence that credits its source.

    The summary leads with the best snippet, trimmed to a clause, because a list of
    links is useless out loud. Titles, links and snippets go to ``detail``.
    """
    query = " ".join(str(args.get("query") or "").split())
    if not query:
        return ToolResult.fail("I need something to search for, sir.")

    ddgs_class = _load_ddgs()
    if ddgs_class is None:
        return ToolResult.fail(
            "I can't search the web, sir, the search package isn't installed.",
            "Neither 'ddgs' nor 'duckduckgo_search' could be imported.",
        )

    count = _result_count(ctx)
    try:
        raw = ddgs_class().text(
            query, max_results=count, region="wt-wt", safesearch="moderate"
        )
        rows = _normalise_results(raw)[:count]
    except Exception as exc:  # noqa: BLE001 - any network or parser fault sounds alike
        _log.warning("Web search for %r failed: %s", query, exc)
        _log.debug("Search traceback", exc_info=True)
        return ToolResult.fail(NETWORK_FAILURE, f"{type(exc).__name__}: {exc}")

    if not rows:
        return ToolResult(
            ok=True,
            summary=f"I found nothing on the web about {query}, sir.",
            detail=_search_detail(query, rows),
            data={"query": query, "results": []},
        )

    best = rows[0]
    source = _source_name(best["url"], best["title"])
    clause = _first_clause(best["body"]) or _first_clause(best["title"])
    summary = (
        f"{clause}, according to {source}, sir."
        if clause
        else f"The closest match I found is on {source}, sir."
    )
    return ToolResult(
        ok=True,
        summary=summary,
        detail=_search_detail(query, rows),
        data={"query": query, "results": rows, "source": source},
    )


def _get_json(url: str, params: dict[str, Any]) -> tuple[dict | None, str]:
    """GET a JSON document, returning ``(data, error)`` and never raising."""
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - transport and decoding faults alike
        _log.warning("Request to %s failed: %s", url, exc)
        _log.debug("Request traceback", exc_info=True)
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, f"{url} returned {type(data).__name__}, not an object"
    return data, ""


def _number(value: Any) -> float | None:
    """Coerce a forecast field to a float, or ``None`` when it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(values: Any) -> Any:
    """First element of a daily array, tolerating a missing or empty array."""
    return values[0] if isinstance(values, (list, tuple)) and values else None


def _weather_detail(place: str, current: dict, daily: dict) -> str:
    """The full forecast record for the log, in metric units."""
    fields = [("temperature_2m", "temperature (C)"), ("apparent_temperature", "feels like (C)"),
              ("relative_humidity_2m", "humidity (%)"), ("precipitation", "precipitation (mm)"),
              ("weather_code", "WMO code"), ("wind_speed_10m", "wind (m/s)")]
    daily_fields = [("temperature_2m_max", "today's high (C)"),
                    ("temperature_2m_min", "today's low (C)"),
                    ("precipitation_probability_max", "rain chance today (%)")]
    lines = [f"place: {place}"]
    lines += [f"{label}: {current.get(key)}" for key, label in fields]
    lines += [f"{label}: {_first(daily.get(key))}" for key, label in daily_fields]
    return "\n".join(lines)


@tool(
    "weather",
    description=(
        "Report the current weather and today's high and low for a city, in Celsius. "
        "Leave the city out to use the user's home city."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, for example Stockholm. Optional.",
            }
        },
        "required": [],
    },
    tier=Tier.SAFE,
)
def weather(ctx: ToolContext, args: dict) -> ToolResult:
    """Answer the weather in one sentence: condition, temperature and context.

    Two keyless Open-Meteo calls — geocoding, then the forecast. Metric throughout.
    """
    city = " ".join(str(args.get("city") or "").split())
    if not city:
        city = " ".join(str(ctx.config.get("tools.default_city", "Stockholm") or "").split())
    if not city:
        return ToolResult.fail("I need a city before I can check the weather, sir.")

    geo, error = _get_json(
        GEOCODING_URL, {"name": city, "count": 1, "language": "en", "format": "json"}
    )
    if geo is None:
        return ToolResult.fail(WEATHER_FAILURE, error)
    places = geo.get("results")
    place = places[0] if isinstance(places, list) and places else None
    place = place if isinstance(place, dict) else None
    latitude = _number(place.get("latitude")) if place else None
    longitude = _number(place.get("longitude")) if place else None
    if place is None or latitude is None or longitude is None:
        return ToolResult.fail(
            f"I couldn't find a place called {city}, sir.",
            f"Geocoding returned no usable match for {city!r}.",
        )
    name = str(place.get("name") or city)
    country = str(place.get("country") or "")

    forecast, error = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                        "precipitation,weather_code,wind_speed_10m"),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "wind_speed_unit": "ms",
        },
    )
    if forecast is None:
        return ToolResult.fail(WEATHER_FAILURE, error)

    current = forecast.get("current") if isinstance(forecast.get("current"), dict) else {}
    daily = forecast.get("daily") if isinstance(forecast.get("daily"), dict) else {}
    where = f"{name}, {country}".strip(", ")
    temperature = _number(current.get("temperature_2m"))
    if temperature is None:
        return ToolResult.fail(
            f"The weather service gave me nothing usable for {name}, sir.",
            _weather_detail(where, current, daily),
        )

    condition = describe_weather_code(current.get("weather_code"))
    feels_like = _number(current.get("apparent_temperature"))
    high = _number(_first(daily.get("temperature_2m_max")))
    low = _number(_first(daily.get("temperature_2m_min")))
    head = f"It's {condition} in {name} at {_spoken_number(temperature)} degrees"
    if feels_like is not None and abs(feels_like - temperature) >= 2:
        summary = f"{head}, feeling like {_spoken_number(feels_like)}, sir."
    elif high is not None and low is not None:
        summary = (f"{head}, with a high of {_spoken_number(high)} and a low of "
                   f"{_spoken_number(low)} today, sir.")
    else:
        summary = f"{head}, sir."
    return ToolResult(
        ok=True,
        summary=summary,
        detail=_weather_detail(where, current, daily),
        data={"city": name, "country": country, "temperature_c": temperature,
              "apparent_c": feels_like, "high_c": high, "low_c": low,
              "weather_code": current.get("weather_code"), "condition": condition,
              "wind_ms": _number(current.get("wind_speed_10m")),
              "humidity_pct": _number(current.get("relative_humidity_2m"))},
    )
