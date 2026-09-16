"""Finding a real business's telephone number, without inventing one.

"Call the dentist" only works if the number is real, so nothing here is guessed: every
number spoken came out of a response JARVIS actually received, and when there was none
he says so.

Order of attack: Nominatim (``extratags`` carries ``phone``, ``contact:phone``,
``website``, ``opening_hours``), then Overpass for a category around that point, then
the existing ``ddgs`` search plus a plain ``requests.get`` on the business's own site,
scanned for ``tel:`` links and phone-shaped text. Most Swedish dentists and barbers
carry no phone tag in OSM, so the last step carries most lookups - that goes in the
log, not out loud. Both OSM services are throttled by a module-level rate limiter
(Nominatim allows one request per second, Overpass gives anonymous callers two slots)
and sent a descriptive ``User-Agent``, as their terms require. ``hitta.se`` and
``eniro.se`` have better data and forbid this, so they are never touched.

Numbers are normalised to E.164 with :mod:`phonenumbers` when it is installed (offline)
and with a conservative regex when it is not. Hits cache to ``places.json`` beside
``memory.json``, so a barber is looked up once, ever. Only ``requests`` is imported at
module level, so this module imports on a bare Linux box.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from jarvis.core.logging import get_logger
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["find_business", "recall_business", "Place", "PlaceCache", "normalise_phone",
           "spoken_phone", "NOMINATIM_URL", "OVERPASS_URL", "USER_AGENT", "CACHE_FILENAME"]

_log = get_logger("tools.places")

#: Keyless endpoints, used strictly within their terms of use.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Both services require a descriptive, identifying User-Agent.
USER_AGENT = ("JARVIS-local-voice-assistant/1.0 "
              "(offline personal assistant; https://github.com/jarvis-local/jarvis)")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "sv,en;q=0.8"}

HTTP_TIMEOUT = 10
OVERPASS_TIMEOUT = 25
NOMINATIM_MIN_INTERVAL = 1.0
OVERPASS_MIN_INTERVAL = 2.0

#: Cache beside memory.json, and its bounds.
CACHE_FILENAME = "places.json"
CACHE_VERSION = 1
MAX_CACHED_PLACES = 500

#: Overpass radius in metres, web results read, and pages actually fetched.
OVERPASS_RADIUS_M = 3000
WEB_RESULTS = 4
MAX_PAGE_FETCHES = 2
MAX_PAGE_CHARS = 300_000

NETWORK_FAILURE = "I couldn't reach the map service just now, sir."
_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

#: Country calling codes for the fallback used when ``phonenumbers`` is absent.
COUNTRY_CODES: dict[str, str] = {
    "SE": "46", "NO": "47", "DK": "45", "FI": "358", "IS": "354", "EE": "372",
    "DE": "49", "GB": "44", "IE": "353", "NL": "31", "BE": "32", "FR": "33",
    "ES": "34", "PT": "351", "IT": "39", "PL": "48", "AT": "43", "CH": "41",
    "US": "1", "CA": "1",
}

#: Spoken category -> the OSM tag Overpass matches on, English and Swedish.
CATEGORY_TAGS: dict[str, str] = {
    word: tag
    for tag, words in {
        "amenity=dentist": ("dentist", "dentists", "tandlakare", "tandläkare"),
        "shop=hairdresser": ("hairdresser", "barber", "frisor", "frisör"),
        "amenity=pharmacy": ("pharmacy", "apotek"),
        "amenity=doctors": ("doctor", "doctors", "lakare", "läkare"),
        "amenity=clinic": ("clinic", "vardcentral", "vårdcentral"),
        "amenity=restaurant": ("restaurant", "restaurang"),
        "amenity=veterinary": ("vet", "veterinary", "veterinar", "veterinär"),
        "shop=optician": ("optician", "optiker"),
        "shop=car_repair": ("garage", "bilverkstad"),
    }.items()
    for word in words
}

_DAY_WORDS = {"mo": "Monday", "tu": "Tuesday", "we": "Wednesday", "th": "Thursday",
              "fr": "Friday", "sa": "Saturday", "su": "Sunday", "ph": "public holidays"}
_PHONE_KEYS = ("phone", "contact:phone", "contact:mobile", "phone:mobile", "telephone")
_WEBSITE_KEYS = ("website", "contact:website", "url")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{6,20}\d")
_TEL_HREF_RE = re.compile(r"""href\s*=\s*["']\s*tel:([^"']{4,40})""", re.IGNORECASE)
_MARKUP_RE = re.compile(r"<[^>]+>")


def _clean(value: Any) -> str:
    """Collapse any argument to one trimmed line."""
    return " ".join(str(value or "").split())


def _as_bool(value: Any) -> bool:
    """Coerce a model-supplied flag to a bool without raising."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "ja"}
    return bool(value)


def _now_iso() -> str:
    """Local time, seconds precision."""
    return datetime.now().isoformat(timespec="seconds")


def _key(name: str) -> str:
    """Cache key for a business name: case- and punctuation-insensitive."""
    return _clean(re.sub(r"[^\w\s]", " ", _clean(name).casefold()))


def _spoken_list(names: list[str]) -> str:
    """Join names the way a person says them."""
    if len(names) < 2:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


class _RateLimiter:
    """Blocks until ``min_interval`` has passed since the last call through it.

    The instances are module-level, so every tool call in every thread shares one clock
    per service. This is not politeness: exceeding one request a second gets JARVIS
    banned from Nominatim.
    """

    def __init__(self, min_interval: float, name: str = "") -> None:
        self.min_interval = max(0.0, float(min_interval))
        self.name = name
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> float:
        """Sleep as long as needed; returns how long that was, in seconds."""
        with self._lock:
            now = time.monotonic()
            delay = (self._last + self.min_interval) - now if self._last else 0.0
            if delay > 0:
                _log.debug("Throttling %s for %.2f s", self.name or "requests", delay)
                time.sleep(delay)
                now = time.monotonic()
            self._last = now
            return max(0.0, delay)


NOMINATIM_LIMITER = _RateLimiter(NOMINATIM_MIN_INTERVAL, "nominatim")
OVERPASS_LIMITER = _RateLimiter(OVERPASS_MIN_INTERVAL, "overpass")


def _get_json(url: str, params: dict[str, Any], limiter: _RateLimiter,
              timeout: int = HTTP_TIMEOUT) -> tuple[Any, str]:
    """GET a JSON document through ``limiter``; returns ``(payload, error)``."""
    limiter.wait()
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json(), ""
    except Exception as exc:  # noqa: BLE001 - transport and decoding faults alike
        _log.warning("Request to %s failed: %s", url, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _load_phonenumbers() -> Any:
    """The offline ``phonenumbers`` library, or ``None`` when it is not installed."""
    try:
        import phonenumbers  # type: ignore[import-not-found]
        return phonenumbers
    except ImportError:
        _log.debug("The 'phonenumbers' package is missing; using the regex fallback.")
        return None


def _regex_e164(text: str, region: str) -> str:
    """Best-effort E.164 without ``phonenumbers``; empty when it does not add up."""
    compact = re.sub(r"[^\d+]", "", text)
    if compact.startswith("00"):
        compact = "+" + compact[2:]
    if compact.startswith("+"):
        body = re.sub(r"\D", "", compact[1:])
        return f"+{body}" if 8 <= len(body) <= 15 else ""
    body = re.sub(r"\D", "", compact)
    if body.startswith("0"):  # national trunk prefix
        body = body[1:]
    code = COUNTRY_CODES.get((region or "SE").upper(), "")
    return f"+{code}{body}" if code and 6 <= len(body) <= 12 else ""


def normalise_phone(raw: Any, region: str = "SE") -> tuple[str, str]:
    """Normalise ``raw`` to E.164, returning ``(number, method)``.

    ``method`` is ``"phonenumbers"``, ``"regex"`` or ``""``. An empty number means the
    input was not a usable telephone number, and the caller must report that nothing
    was found rather than speak the raw text back.
    """
    text = _clean(raw)
    if not text:
        return "", ""
    module = _load_phonenumbers()
    if module is not None:
        try:
            parsed = module.parse(text, (region or "SE").upper())
            if module.is_valid_number(parsed):
                fmt = module.PhoneNumberFormat.E164
                return _clean(module.format_number(parsed, fmt)), "phonenumbers"
            _log.debug("phonenumbers rejected %r for region %s", text, region)
        except Exception as exc:  # noqa: BLE001 - NumberParseException and friends
            _log.debug("phonenumbers could not parse %r: %s", text, exc)
        return "", ""
    number = _regex_e164(text, region)
    return (number, "regex") if number else ("", "")


def spoken_phone(number: str, region: str = "SE") -> str:
    """The number as separated digit words, dialled the way it is dialled at home."""
    dialled = _clean(number)
    code = COUNTRY_CODES.get((region or "SE").upper(), "")
    if code and dialled.startswith(f"+{code}"):
        dialled = "0" + dialled[1 + len(code):]
    digits = re.sub(r"\D", "", dialled)
    if not digits:
        return ""
    words = [_DIGITS[int(digit)] for digit in digits]
    spoken = ", ".join(" ".join(words[i:i + 3]) for i in range(0, len(words), 3))
    return f"plus {spoken}" if dialled.startswith("+") else spoken


def _speakable_hours(raw: str) -> str:
    """Turn OSM ``opening_hours`` into something that survives being read aloud."""
    text = _clean(raw)
    if not text:
        return ""
    text = re.sub(r"\b(Mo|Tu|We|Th|Fr|Sa|Su|PH)\b",
                  lambda match: _DAY_WORDS[match.group(1).lower()], text, flags=re.I)
    text = text.replace("24/7", "around the clock").replace("off", "closed")
    text = re.sub(r"\s*-\s*", " to ", text.replace(";", ", "))
    return _clean(re.sub(r"\b0?(\d{1,2}):00\b", r"\1", text))


@dataclass
class Place:
    """One looked-up business, as cached and as handed to the telephony tools."""

    name: str
    phone: str = ""
    website: str = ""
    address: str = ""
    opening_hours: str = ""
    source: str = ""
    query: str = ""
    near: str = ""
    looked_up: str = field(default_factory=_now_iso)
    lat: float | None = None
    lon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "Place | None":
        """Build a Place from parsed JSON, or ``None`` when the row is unusable."""
        if not isinstance(raw, dict) or not _clean(raw.get("name")):
            return None
        text = {key: _clean(raw.get(key)) for key in
                ("name", "phone", "website", "address", "opening_hours", "source",
                 "query", "near")}
        return cls(**text, looked_up=_clean(raw.get("looked_up")) or _now_iso(),
                   lat=_float(raw.get("lat")), lon=_float(raw.get("lon")))


def _float(value: Any) -> float | None:
    """A coordinate as a float, or ``None`` when it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PlaceCache:
    """``places.json``: a tolerant store, so a barber is looked up once, ever.

    Loading never raises - a truncated or hand-edited file degrades to an empty cache
    and a warning. Saving is atomic (temp file in the same directory plus
    ``os.replace``), exactly as :mod:`jarvis.core.memory` does it.
    """

    def __init__(self, path: str | Path = CACHE_FILENAME) -> None:
        self.path = Path(path)
        self._places: list[Place] = []
        self.load()

    def load(self) -> None:
        """Read the file; any problem leaves an empty cache and a logged warning."""
        self._places = []
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeDecodeError) as exc:
            _log.warning("Could not read %s: %s", self.path, exc)
            return
        try:
            data = json.loads(raw_text) if raw_text.strip() else {}
        except (json.JSONDecodeError, ValueError) as exc:
            _log.warning("%s is not valid JSON (%s); starting empty.", self.path, exc)
            return
        rows = data.get("places") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            if rows is not None:
                _log.warning("%s has no usable list of places; starting empty.", self.path)
            return
        seen: set[str] = set()
        for row in rows:
            place = Place.from_dict(row)
            if place is None or _key(place.name) in seen:
                continue
            seen.add(_key(place.name))
            self._places.append(place)

    def save(self) -> None:
        """Write the cache atomically; failures are logged, never raised."""
        payload = {"places": [p.to_dict() for p in self._places[-MAX_CACHED_PLACES:]],
                   "version": CACHE_VERSION}
        tmp_path: str | None = None
        try:
            parent = self.path.parent if str(self.path.parent) else Path(".")
            parent.mkdir(parents=True, exist_ok=True)
            handle_fd, tmp_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=str(parent))
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            tmp_path = None
        except OSError as exc:
            _log.warning("Could not save %s: %s", self.path, exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:  # pragma: no cover - defensive
                    _log.debug("Could not remove the temp file %s", tmp_path)

    def find(self, name: str) -> Place | None:
        """The cached place matching ``name`` exactly, else as a substring."""
        needle = _key(name)
        if not needle:
            return None
        exact = [p for p in self._places if _key(p.name) == needle]
        loose = [p for p in self._places
                 if _key(p.name) and (needle in _key(p.name) or _key(p.name) in needle)]
        found = exact or loose
        return found[-1] if found else None

    def put(self, place: Place) -> None:
        """Store ``place``, replacing any earlier entry with the same name."""
        self._places = [row for row in self._places if _key(row.name) != _key(place.name)]
        self._places.append(place)
        self.save()

    def all(self) -> list[Place]:
        """Every cached place, oldest first."""
        return list(self._places)


def _cache_path(ctx: ToolContext) -> Path:
    """``places.json`` beside ``memory.json``, wherever that turned out to be."""
    stored = getattr(getattr(ctx, "memory", None), "path", None)
    if isinstance(stored, (str, Path)) and str(stored):
        return Path(stored).expanduser().parent / CACHE_FILENAME
    try:
        return ctx.config.resolve_path(CACHE_FILENAME)
    except Exception:  # noqa: BLE001 - a minimal fake context, or a broken config
        return Path(CACHE_FILENAME)


def _tag(tags: dict[str, Any], keys: tuple[str, ...]) -> str:
    """First non-empty value among ``keys``."""
    for key in keys:
        value = _clean(tags.get(key))
        if value:
            return value
    return ""


def _place_from_row(row: dict[str, Any], region: str, source: str) -> Place | None:
    """Turn one Nominatim or Overpass element into a :class:`Place`."""
    tags = row.get("extratags") or row.get("tags") or {}
    tags = tags if isinstance(tags, dict) else {}
    display = _clean(row.get("display_name"))
    name = _clean(row.get("name")) or _clean(tags.get("name")) or display.split(",")[0]
    if not name:
        return None
    centre = row.get("center") if isinstance(row.get("center"), dict) else row
    phone, _method = normalise_phone(_tag(tags, _PHONE_KEYS), region)
    return Place(name=name, phone=phone, website=_tag(tags, _WEBSITE_KEYS),
                 address=display, opening_hours=_clean(tags.get("opening_hours")),
                 source=source, lat=_float(centre.get("lat")),
                 lon=_float(centre.get("lon")))


def _nominatim_search(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """Search Nominatim; returns ``(rows, error)`` and never raises."""
    payload, error = _get_json(
        NOMINATIM_URL,
        {"format": "jsonv2", "extratags": 1, "addressdetails": 1, "limit": limit,
         "q": query},
        NOMINATIM_LIMITER)
    if payload is None:
        return [], error
    if not isinstance(payload, list):
        return [], f"Nominatim returned {type(payload).__name__}, not a list"
    return [row for row in payload if isinstance(row, dict)], ""


def _category_for(name: str) -> str:
    """The OSM tag for a spoken category such as "dentist", or ``""``."""
    for word in re.split(r"\W+", _clean(name).casefold()):
        if word in CATEGORY_TAGS:
            return CATEGORY_TAGS[word]
    return ""


def _overpass_search(name: str, lat: float, lon: float) -> tuple[list[dict], str]:
    """Ask Overpass for phone-carrying POIs of this category around a point."""
    key, _, value = _category_for(name).partition("=")
    selector = f'["{key}"="{value}"]' if value else f'["name"~"{re.escape(_clean(name))}",i]'
    around = f"around:{OVERPASS_RADIUS_M},{lat:.6f},{lon:.6f}"
    query = (f"[out:json][timeout:{OVERPASS_TIMEOUT}];("
             f'node({around}){selector}["phone"];'
             f'node({around}){selector}["contact:phone"];'
             f'way({around}){selector}["phone"];);out tags center 10;')
    payload, error = _get_json(OVERPASS_URL, {"data": query}, OVERPASS_LIMITER,
                               timeout=OVERPASS_TIMEOUT)
    if payload is None:
        return [], error
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        return [], "Overpass returned no element list"
    return [row for row in elements if isinstance(row, dict)], ""


def _get_text(url: str) -> str:
    """Fetch a page as text, bounded and never raising."""
    target = url if "//" in url else f"https://{url}"
    try:
        response = requests.get(target, headers=HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return str(response.text or "")[:MAX_PAGE_CHARS]
    except Exception as exc:  # noqa: BLE001 - a business website may be anything
        _log.debug("Could not read %s: %s", target, exc)
        return ""


def _phone_from_text(text: str, region: str) -> str:
    """The first genuine-looking phone number in a page or a snippet."""
    for candidate in _TEL_HREF_RE.findall(text):
        number, _method = normalise_phone(candidate, region)
        if number:
            return number
    for match in _PHONE_RE.finditer(_MARKUP_RE.sub(" ", text)):
        number, _method = normalise_phone(match.group(0), region)
        if number:
            return number
    return ""


def _web_rows(query: str) -> list[dict[str, str]]:
    """Top web results for ``query``, through the existing ddgs wrapper."""
    from jarvis.tools import web_tools  # local: the search package is optional

    ddgs_class = web_tools._load_ddgs()
    if ddgs_class is None:
        _log.info("No DuckDuckGo package installed; the web fallback is unavailable.")
        return []
    try:
        raw = ddgs_class().text(query, max_results=WEB_RESULTS, region="wt-wt",
                                safesearch="moderate")
        return web_tools._normalise_results(raw)[:WEB_RESULTS]
    except Exception as exc:  # noqa: BLE001 - network or parser fault
        _log.warning("Web fallback search for %r failed: %s", query, exc)
        return []


def _phone_from_web(name: str, near: str, website: str, region: str,
                    trail: list[str]) -> tuple[str, str]:
    """Last resort: the business's own site, then the search results themselves."""
    if website:
        number = _phone_from_text(_get_text(website), region)
        trail.append(f"website {website}: {'phone found' if number else 'no phone'}")
        if number:
            return number, f"website {website}"
    query = _clean(f"{name} {near} telefon")
    rows = _web_rows(query)
    trail.append(f"web search {query!r}: {len(rows)} result(s)")
    fetched = 0
    for row in rows:
        number = _phone_from_text(f"{row.get('title', '')} {row.get('body', '')}", region)
        if number:
            return number, _clean(f"web snippet {row.get('url', '')}")
        url = _clean(row.get("url"))
        if url and fetched < MAX_PAGE_FETCHES:
            fetched += 1
            number = _phone_from_text(_get_text(url), region)
            if number:
                return number, f"website {url}"
    return "", ""


def _lookup(name: str, near: str, region: str,
            trail: list[str]) -> tuple[Place | None, bool]:
    """Run the whole chain; returns ``(best place or None, network_trouble)``."""
    query = _clean(f"{name} {near}") or name
    rows, error = _nominatim_search(query)
    trouble = bool(error)
    trail.append(f"nominatim {query!r}: {len(rows)} result(s)"
                 + (f" error={error}" if error else ""))
    best: Place | None = None
    for row in rows:
        place = _place_from_row(row, region, "nominatim")
        if place is None:
            continue
        if place.phone:
            trail.append(f"phone from the OSM tags of {place.name}")
            return place, False
        best = best or place

    point: tuple[float, float] | None = None
    if best is not None and best.lat is not None and best.lon is not None:
        point = (best.lat, best.lon)
    elif near:
        anchors, anchor_error = _nominatim_search(near, limit=1)
        trouble = trouble or bool(anchor_error)
        anchor = _place_from_row(anchors[0], region, "nominatim") if anchors else None
        if anchor is not None and anchor.lat is not None and anchor.lon is not None:
            point = (anchor.lat, anchor.lon)
    if point is not None:
        elements, overpass_error = _overpass_search(name, point[0], point[1])
        trouble = trouble or bool(overpass_error)
        trail.append(f"overpass around {point[0]:.4f},{point[1]:.4f}: "
                     f"{len(elements)} element(s)"
                     + (f" error={overpass_error}" if overpass_error else ""))
        for element in elements:
            place = _place_from_row(element, region, "overpass")
            if place is not None and place.phone:
                return place, False
            best = best or place

    number, source = _phone_from_web(name, near, best.website if best else "",
                                     region, trail)
    if number:
        _log.info("Phone for %r came from the web fallback (%s), not from OSM.",
                  name, source)
        place = best or Place(name=name)
        place.phone, place.source = number, source or "web"
        return place, False
    return best, trouble and best is None


def _home_region(ctx: ToolContext) -> str:
    """``tools.home_region`` from the config, defaulting to Sweden."""
    try:
        region = _clean(ctx.config.get("tools.home_region", "SE")) or "SE"
    except Exception:  # noqa: BLE001 - a minimal fake context in a test
        region = "SE"
    return region.upper()[:2]


def _default_near(ctx: ToolContext) -> str:
    """Where to look when the user did not say: the configured home city."""
    try:
        return _clean(ctx.config.get("tools.default_city", ""))
    except Exception:  # noqa: BLE001
        return ""


def _result(place: Place, region: str, include_hours: bool, detail: str,
            cached: bool = False) -> ToolResult:
    """One sentence with the name and the number, said as dialable digits."""
    spoken = spoken_phone(place.phone, region)
    hours = _speakable_hours(place.opening_hours)
    summary = f"The number for {place.name} is {spoken}, sir."
    if include_hours and hours:
        summary = (f"The number for {place.name} is {spoken}, and they are open "
                   f"{hours}, sir.")
    method = "phonenumbers" if _load_phonenumbers() is not None else "a regex"
    lines = [f"name: {place.name}", f"phone: {place.phone}",
             f"source: {place.source or 'cache'}", f"normalised with: {method}",
             f"cached: {cached}", f"address: {place.address}",
             f"website: {place.website}", f"opening_hours: {place.opening_hours}"]
    return ToolResult(ok=True, summary=summary,
                      detail="\n".join(lines + ([detail] if detail else [])),
                      data={**place.to_dict(), "spoken_phone": spoken, "cached": cached})


@tool(
    "find_business",
    description=("Find a real business's telephone number - a dentist, a hairdresser, "
                 "a restaurant - from OpenStreetMap and the open web. Use it before "
                 "dialling. It never invents a number: it says when it found none."),
    parameters={"type": "object", "required": ["name"], "properties": {
        "name": {"type": "string", "description":
                 "The business, or the kind of business, e.g. 'Tandlakare Soder' or 'dentist'."},
        "near": {"type": "string", "description":
                 "Town, district or address to search near. Optional."},
        "hours": {"type": "boolean", "description":
                  "Say the opening hours too. Only when the user asked for them."}}},
    tier=Tier.SAFE,
)
def find_business(ctx: ToolContext, args: dict) -> ToolResult:
    """Look a business up and answer with its number, or plainly with nothing."""
    name = _clean(args.get("name"))
    if not name:
        return ToolResult.fail("I need a name before I can look up a number, sir.")
    near = _clean(args.get("near")) or _default_near(ctx)
    include_hours = _as_bool(args.get("hours"))
    region = _home_region(ctx)

    cache = PlaceCache(_cache_path(ctx))
    cached = cache.find(name)
    if cached is not None and cached.phone:
        _log.info("Answering %r from %s, looked up %s.", name, cache.path, cached.looked_up)
        return _result(cached, region, include_hours, "answered from the cache", True)

    trail: list[str] = []
    place, trouble = _lookup(name, near, region, trail)
    detail = "\n".join(trail)
    if place is None:
        if trouble:
            return ToolResult.fail(NETWORK_FAILURE, detail)
        return ToolResult.fail(f"I found no listing for {name}, sir.", detail)
    if not place.phone:
        return ToolResult(ok=False, detail=detail,
                          summary=f"I found {place.name} but no telephone number for "
                                  "them, sir.",
                          data={**place.to_dict(), "spoken_phone": "", "cached": False})
    place.query, place.near, place.looked_up = name, near, _now_iso()
    cache.put(place)
    return _result(place, region, include_hours, detail)


@tool(
    "recall_business",
    description=("Read back a business JARVIS has already looked up, from the local "
                 "cache, without touching the network. Leave the name out to hear what "
                 "is stored."),
    parameters={"type": "object", "required": [], "properties": {
        "name": {"type": "string", "description": "The business to recall. Optional."}}},
    tier=Tier.SAFE,
)
def recall_business(ctx: ToolContext, args: dict) -> ToolResult:
    """Answer from ``places.json`` alone - this tool never looks anything up."""
    name = _clean(args.get("name"))
    cache = PlaceCache(_cache_path(ctx))
    places = cache.all()
    if not places:
        return ToolResult(ok=True, summary="I haven't looked up any businesses yet, sir.",
                          detail=f"cache: {cache.path}", data={"places": []})
    if name:
        place = cache.find(name)
        if place is None or not place.phone:
            return ToolResult.fail(f"I have nothing stored for {name}, sir.",
                                   f"cache: {cache.path}, {len(places)} entries")
        return _result(place, _home_region(ctx), False, f"cache: {cache.path}", True)
    recent = sorted(places, key=lambda row: row.looked_up, reverse=True)[:3]
    return ToolResult(
        ok=True,
        summary=f"I have numbers for {_spoken_list([row.name for row in recent])}, sir.",
        detail=f"cache: {cache.path}\n"
               + "\n".join(f"{row.name}: {row.phone}" for row in places),
        data={"places": [row.to_dict() for row in recent], "count": len(places)})
