"""The desk window's page, read as text.

None of this runs the page: there is no browser in the test suite and there is not
going to be one. What can be checked without one is everything that makes the page
*fail silently* - a frame the contract sends that nothing draws, a verb the window
is supposed to have that nothing sends, a colour that has drifted away from the
phone's, an element id that was renamed in the markup and not in the script, or a
script tag that would leave the machine to fetch something.

The last of those matters most. The window has to open on a desk with the network
unplugged, in a WebView2 that may have no proxy and no DNS, and a page that waits
on a CDN is a page that shows a blank rectangle instead of JARVIS.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "jarvis" / "desk" / "static" / "desk.html"
PHONE = ROOT / "jarvis" / "remote" / "static" / "index.html"

HTML = PAGE.read_text(encoding="utf-8")
SCRIPT = HTML.split("<script>", 1)[1].split("</script>", 1)[0]

#: Server to page, from contract 24.3. Every one of these must be drawn.
INBOUND = (
    "hello", "state", "heard", "sentence", "tool", "confirm", "confirm_done",
    "latency", "note", "log", "status",
)

#: Page to server, from contract 24.3, plus the window frame the title bar needs.
OUTBOUND = (
    "say", "listen", "stop", "confirm", "pause", "resume", "routine", "set",
    "quit", "window",
)

#: The arc reactor, to the last decimal the phone draws it with.
GEOMETRY = (
    'r="29.05"',                        # the coil band
    'stroke-width="11.90"',             # its thickness, which makes the eight coils
    'stroke-dasharray="18.25 4.56"',    # coil, gap, coil
    'stroke-dasharray="20.53 182.53"',  # the comet that sweeps it while he works
    'r="37.2"',                         # the outer rim
    'r="19.1"',                         # the inner ring
    'r="14.2"',                         # the bloom
    'r="7.5"',                          # the white-hot core
)

#: Colours, not sizes: a phone ring is 152px and a desk ring is not.
PALETTE = (
    "bg", "panel", "panel-2", "edge", "ink", "dim",
    "idle", "listening", "thinking", "speaking", "paused", "bad", "accent",
)


def _palette(html: str) -> dict[str, str]:
    """The custom properties declared on ``:root``, as they are written."""
    block = html.split(":root {", 1)[1].split("}", 1)[0]
    return {name: value.strip() for name, value in re.findall(r"--([a-z0-9-]+):\s*([^;]+);", block)}


# --- self-containment -----------------------------------------------------------------
def test_the_page_asks_nothing_at_all_of_the_network():
    """A window that waits on a CDN is a blank rectangle on a desk that is offline."""
    tags = re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", HTML, re.I)
    offenders = [tag for tag in tags if "http://" in tag or "https://" in tag]

    assert not offenders, f"the page fetches something: {offenders}"
    assert not re.search(r"<script\b[^>]*\bsrc=", HTML, re.I), "the page loads an external script"
    assert "@import" not in HTML, "a stylesheet import can reach off the machine too"
    assert "url(http" not in HTML, "a CSS url() can reach off the machine too"


def test_the_page_sits_where_the_desk_server_looks_for_it():
    """The server serves one path; a page filed anywhere else is a 404 and a dead window."""
    from jarvis.desk import server as server_module

    assert (Path(server_module._STATIC_DIR) / "desk.html").resolve() == PAGE.resolve()


def test_the_page_never_evaluates_what_it_is_sent():
    """Frames arrive from a socket. None of them may ever become code or markup."""
    for weapon in ("eval(", "new Function(", "innerHTML", "outerHTML", "document.write"):
        assert weapon not in SCRIPT, f"the page uses {weapon}"


def test_the_page_never_navigates_itself_away():
    """There is no back button on a frameless window: one stray link and the app is gone."""
    assert 'target="_blank"' not in HTML
    assert "location.href" not in SCRIPT
    assert "window.open(" not in SCRIPT


# --- the wire -------------------------------------------------------------------------
def test_every_frame_the_desk_sends_the_window_is_drawn():
    """An undrawn frame is a feature that works everywhere except where you can see it."""
    for kind in INBOUND:
        assert f'case "{kind}":' in SCRIPT, f"nothing draws the {kind} frame"


def test_every_verb_the_window_is_given_is_actually_sent():
    """The contract's page-to-server table is the window's whole vocabulary."""
    sent = set(re.findall(r'type:\s*"([a-z_]+)"', SCRIPT))

    for kind in OUTBOUND:
        assert kind in sent, f"nothing ever sends the {kind} frame"


def test_the_window_sends_nothing_the_contract_does_not_list():
    """A frame the server has no handler for is an error frame and a puzzled operator."""
    sent = set(re.findall(r'type:\s*"([a-z_]+)"', SCRIPT))

    assert sent <= set(OUTBOUND), f"unknown frames: {sorted(sent - set(OUTBOUND))}"


def test_the_window_talks_to_the_desk_socket_and_no_other():
    """The phone's sockets are the phone's; this page has exactly one door."""
    paths = set(re.findall(r'wsUrl\("([^"]+)"\)', SCRIPT))

    assert paths == {"/ws/desk"}
    assert "/ws/audio" not in SCRIPT and "/ws/state" not in SCRIPT


def test_a_dropped_link_is_retried_with_a_growing_delay():
    """A window left open overnight must find its way back without being clicked."""
    assert "onclose" in SCRIPT and "retry(" in SCRIPT
    assert re.search(r"setTimeout\(connect,\s*Math\.min\(", SCRIPT), "the retry is not backed off"


def test_a_reconnected_window_starts_from_an_empty_feed():
    """hello is followed by the replay; without the reset every line would appear twice."""
    hello = SCRIPT.split("function onHello(", 1)[1].split("\n}", 1)[0]

    assert "ui.feed.textContent" in hello, "onHello does not clear the conversation"


# --- one product, two windows ---------------------------------------------------------
def test_the_reactor_is_the_phones_reactor_to_the_last_decimal():
    """Two faces of one assistant. A coil that is 29.05 here and 29 there is two products."""
    phone = PHONE.read_text(encoding="utf-8")

    for attribute in GEOMETRY:
        assert attribute in phone, f"the phone no longer draws {attribute}; this test is stale"
        assert attribute in HTML, f"the desk's reactor has drifted: {attribute}"


def test_the_palette_is_the_phones_palette():
    """The state colours are the product's vocabulary; they may not fork per window."""
    desk, phone = _palette(HTML), _palette(PHONE.read_text(encoding="utf-8"))

    for token in PALETTE:
        assert token in desk, f"the desk page lost --{token}"
        assert desk[token] == phone[token], f"--{token} has drifted from the phone"


def test_every_state_has_both_a_caption_and_a_ring():
    """A state the stylesheet has never heard of is a reactor frozen mid-animation."""
    for state in ("idle", "listening", "thinking", "speaking", "paused"):
        assert f'body[data-state="{state}"]' in HTML, f"{state} has no ring style"
        assert f"{state}:" in SCRIPT.split("const CAPTIONS", 1)[1][:260], f"{state} has no caption"


# --- the window itself ----------------------------------------------------------------
def test_every_element_the_script_reaches_for_actually_exists():
    """A renamed id is a blank panel on the desk and not one line in the log."""
    wanted = set(re.findall(r'el\("([^"]+)"\)', SCRIPT))
    declared = set(re.findall(r'id="([^"]+)"', HTML))

    assert wanted, "the page stopped looking elements up; this test needs rewriting"
    assert wanted <= declared, f"missing from the markup: {sorted(wanted - declared)}"


def test_the_title_bar_drags_the_window_but_its_buttons_do_not():
    """pywebview drags by anything inside the region, so the buttons must be outside it."""
    bar = HTML.split('<header id="bar">', 1)[1].split("</header>", 1)[0]
    region = bar.split('class="drag pywebview-drag-region"', 1)[1].split("</div>", 1)[0]

    assert "pywebview-drag-region" in bar, "nothing tells pywebview where to drag from"
    assert "<button" not in region, "a window button sits inside the drag region"
    assert bar.count("<button") == 2, "the title bar is minimise and close, nothing else"


def test_closing_the_window_hides_it_rather_than_stopping_him():
    """Close is close-to-tray: it sends a window frame, and quit is a different button."""
    assert '{ type: "window", action: "close" }' in SCRIPT
    assert '{ type: "window", action: "minimize" }' in SCRIPT
    assert "tray" in HTML.split('id="close"', 1)[1].split(">", 1)[0].lower(), "no tooltip says so"


def test_the_reactor_is_also_the_push_to_talk_button():
    """The one control the operator reaches for first is the one he is looking at."""
    assert 'ui.talk.addEventListener("click", arm)' in SCRIPT
    assert '{ type: "listen" }' in SCRIPT


def test_space_arms_the_reactor_but_only_when_nothing_is_being_typed():
    """A spacebar in the composer is a space. Swallowing it would be unforgivable."""
    handler = SCRIPT.split('document.addEventListener("keydown"', 1)[1]

    assert "input|textarea|select" in handler, "the handler does not know what a field is"
    assert handler.index("if (typing) return;") < handler.index('event.code === "Space"')


def test_escape_stops_him_from_anywhere_including_the_composer():
    """The stop key has to work with the cursor wherever the operator left it."""
    handler = SCRIPT.split('document.addEventListener("keydown"', 1)[1]
    escape = handler.split('event.key === "Escape"', 1)[1].split("return;", 1)[0]

    assert '{ type: "stop" }' in escape


def test_the_confirmation_bar_starts_hidden_and_offers_both_answers():
    """A GUARDED tool is approved by mouse or by voice; whichever answers first wins."""
    assert re.search(r'<section id="confirm"[^>]*\bhidden\b', HTML), "the bar starts on screen"
    assert "answerConfirm(true)" in SCRIPT and "answerConfirm(false)" in SCRIPT
    assert '{ type: "confirm", granted: granted }' in SCRIPT
    # Cancel is the quiet one and Confirm is the bright one, not the other way round.
    assert 'id="confirmNo" class="btn calm"' in HTML
    assert 'id="confirmYes" class="btn go"' in HTML


def test_the_bar_only_leaves_when_the_assistant_says_it_may():
    """Hiding it on our own click would lose a confirmation the microphone then answered."""
    answer = SCRIPT.split("function answerConfirm(", 1)[1].split("\n}", 1)[0]

    assert "hidden" not in answer, "the page hides the bar itself instead of waiting"
    assert "ui.confirm.hidden = true;" in SCRIPT.split("function hideConfirm(", 1)[1]


def test_the_log_drawer_is_collapsed_until_it_is_asked_for():
    """The log is evidence, not furniture: a window that opens on a wall of it is a console."""
    assert re.search(r'<div id="drawer"[^>]*\bhidden\b', HTML)


def test_every_countdown_on_screen_ticks_from_one_interval():
    """Twenty timers must not mean twenty timers; the phone learned this first."""
    assert SCRIPT.count("setInterval(") == 1, "something started a second clock"
    assert SCRIPT.count('querySelectorAll("[data-due]")') == 1


def test_a_tool_is_drawn_when_it_starts_and_finished_in_place():
    """The desk's one advantage over the phone: a slow search visibly works, not pauses."""
    handler = SCRIPT.split("function onTool(", 1)[1].split("\n}", 1)[0]

    assert 'ev.phase === "start"' in handler
    assert "app.running.set" in handler and "app.running.delete" in handler


def test_the_cards_are_the_phones_cards():
    """A business, a timer, the weather, a search, an app, the clock - the same vocabulary."""
    phone = PHONE.read_text(encoding="utf-8")
    for name in ("find_business", "set_timer", "set_reminder", "list_timers", "weather",
                 "system_status", "web_search", "open_url", "open_app", "get_time_date"):
        assert f'case "{name}":' in phone, f"the phone stopped drawing {name}; this test is stale"
        assert f'case "{name}":' in SCRIPT, f"the desk draws no card for {name}"


def test_a_refusal_is_told_apart_from_a_failure():
    """Red for "I will not", amber for "I could not". They are not the same answer."""
    assert 'ev.refused ? "refused" : ev.ok ? "done" : "failed"' in SCRIPT
    assert ".card.refused" in HTML and ".card.failed" in HTML


def test_nothing_the_window_says_out_loud_is_written_for_a_machine():
    """The window is a face of a butler: it says sir, and it never says exception."""
    for word in ("Traceback", "undefined", "ERROR:", "null"):
        assert f">{word}" not in HTML
    assert "sir" in HTML
