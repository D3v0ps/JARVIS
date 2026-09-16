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

import inspect
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
def _fetched(tag: str) -> list[str]:
    """What a tag would actually go and get: its ``src`` or ``href``, as written.

    Read as URLs rather than searched as text, because the page carries an inline
    SVG favicon and every SVG declares ``xmlns="http://www.w3.org/2000/svg"`` - a
    string that looks like a fetch, is never fetched, and is not even a URL to
    anything. What matters is the scheme of the value the browser would resolve.
    """
    return re.findall(r'\b(?:src|href)\s*=\s*"([^"]*)"', tag)


def test_the_page_asks_nothing_at_all_of_the_network():
    """A window that waits on a CDN is a blank rectangle on a desk that is offline."""
    tags = re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", HTML, re.I)
    offenders = [url for tag in tags for url in _fetched(tag)
                 if url.startswith(("http://", "https://", "//"))]

    assert not offenders, f"the page fetches something: {offenders}"
    for tag in tags:
        for url in _fetched(tag):
            assert url.startswith("data:"), f"{tag} fetches {url!r} instead of carrying it"
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
    assert bar.count("<button") == 3, "the title bar is minimise, maximise and close"


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


def test_the_safe_answer_takes_the_keyboard_focus():
    """This bar appears for the tools that delete files and shut the machine down.

    A startled operator hits Enter or Space, so whatever holds the focus is what
    they will choose by accident. It has to be Cancel.
    """
    body = SCRIPT.split("function showConfirm(", 1)[1].split("\nfunction ", 1)[0]
    assert "ui.confirmNo.focus()" in body
    assert "ui.confirmYes.focus()" not in body


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


# --- what the reviewers found on screen -----------------------------------------------
def test_the_newest_line_is_scrolled_by_whatever_is_actually_overflowing():
    """Which box scrolls depends on how wide the window is, and that is the bug.

    At a desk width ``#feed`` is the element with the overflow. Narrow the window
    past the one-column breakpoint and ``#feed`` is simply as tall as its contents
    while ``<main>`` is the box that scrolls, so ``feed.scrollTop = feed.scrollHeight``
    moves nothing whatsoever: his answer lands below the fold and stays there for the
    rest of the session, which is the single worst thing a window of evidence can do.
    """
    assert "function scrollFeed(" in SCRIPT, "there is no helper"
    assert "box.parentElement" in SCRIPT, "the helper never looks past the feed"
    assert "ui.feed.scrollTop" not in SCRIPT, "something still scrolls the feed by hand"

    for caller in ("function post(", "function appendReply(", "function fillCard(",
                   "function drawWork("):
        body = SCRIPT.split(caller, 1)[1].split("\n}", 1)[0]
        assert "scrollFeed()" in body, f"{caller.split()[1]} never scrolls what it drew into view"


def test_the_bar_and_the_drawer_put_back_what_they_took_from_the_feed():
    """Both appear *above* the feed and take 90 to 180 pixels of it as they do.

    Whatever was last said is then behind the composer, which is exactly when the
    operator is being asked to approve deleting forty-one files.
    """
    for caller in ("function showConfirm(", "function hideConfirm("):
        body = SCRIPT.split(caller, 1)[1].split("\n}", 1)[0]
        assert "scrollFeed()" in body, f"{caller.split()[1]} leaves the feed where it was"

    drawer = SCRIPT.split('ui.drawerToggle.addEventListener("click"', 1)[1].split("\n});", 1)[0]
    assert "scrollFeed()" in drawer


def test_the_feed_grows_up_from_the_composer_without_stranding_its_own_overflow():
    """One exchange in a 450px column belongs at the bottom of it, not the top.

    ``justify-content: flex-end`` is the obvious way to write that and is a trap:
    overflow in a flex-end column goes off the *top* of the scroll box, where
    ``scrollTop`` cannot reach it. Measured in this engine, forty lines gave
    ``scrollHeight === clientHeight`` and a first line at -1624px - a conversation
    that is not merely scrolled away but gone. An auto margin on the first line
    resolves to zero as soon as the free space is negative, and both behaviours hold.
    """
    feed = HTML.split("  #feed {", 1)[1].split("}", 1)[0]

    assert "justify-content: flex-end" not in feed, "flex-end strands everything above the fold"
    assert "#feed > :first-child { margin-top: auto; }" in HTML


def test_the_right_hand_column_fills_its_own_height():
    """Four panels stacked at the top of a 700px column leave 250px of nothing.

    Empty space inside a bordered panel reads as a list with room in it. The same
    space with no panel round it reads as a page that failed to finish drawing.
    """
    assert ".instruments .fill {" in HTML
    assert "flex: 1 1 auto" in HTML.split(".instruments .fill {", 1)[1].split("}", 1)[0]
    assert '<section class="panel fill">' in HTML, "no panel is the one that grows"


def test_the_working_line_keeps_what_was_heard_and_fades_what_it_cannot_fit():
    """Scrolling this line to the bottom took the heard step off the top of it.

    What is left is a sliced row of tool pills describing a question nobody can
    read. The line is three rows tall, it is never scrolled for you, and a turn too
    long for three rows says so with a fade rather than by hiding its own subject.
    """
    work = HTML.split("  #work {", 1)[1].split("}", 1)[0]

    assert "max-height: 116px" in work, "the cap is not three rows of steps"
    assert "ui.work.scrollTop" not in SCRIPT, "the line still scrolls its first step away"
    assert "#work.more {" in HTML, "there is no fade for the rows that do not fit"
    assert 'ui.work.classList.toggle("more"' in SCRIPT, "the fade is never turned on"


def test_ordinary_narration_is_not_painted_as_a_warning():
    """"Dialling, sir." is not a problem, and the phone draws that same line grey.

    A note line that is always amber teaches the operator to ignore the one line
    that will ever matter, so the colour is kept for the note frame and for a verb
    that did not happen.
    """
    note = HTML.split("  .note {", 1)[1].split("}", 1)[0]

    assert "color: var(--dim)" in note, "ordinary narration is still in the warning colour"
    assert ".note.bad { color: var(--thinking); }" in HTML

    body = SCRIPT.split("function notice(", 1)[1].split("\n}", 1)[0]
    assert 'classList.toggle("bad"' in body

    dispatch = SCRIPT.split("function onDeskMessage(", 1)[1].split("\n}", 1)[0]
    for frame in ("note", "error", "busy"):
        line = dispatch.split(f'case "{frame}":', 1)[1].split("break;", 1)[0]
        assert "true" in line, f"the {frame} frame is drawn as ordinary narration"


def test_the_scrollbar_thumb_can_actually_be_seen():
    """Content that scrolled off looks like content that was never drawn otherwise.

    At 0.16 alpha on this background the thumb is invisible, and a window whose
    whole job is to show evidence may not hide the fact that there is more of it.
    """
    assert "background: rgba(41,182,246,0.38)" in HTML, "the thumb is still invisible"
    assert "background: rgba(41,182,246,0.55)" in HTML, "the thumb does not answer the pointer"

    instruments = HTML.split("\n  .instruments {", 1)[1].split("}", 1)[0]
    assert instruments.count("#000 12px") == 2, "the column has no top mask to match the feed's"


def test_a_short_window_shrinks_the_reactor_before_anything_else():
    """The native window now goes down to 620px tall, and the ring is 196 of them.

    Something has to give, and it is the most decorative thing on the page rather
    than the conversation, the working line or the composer.
    """
    assert "@media (max-height: 700px)" in HTML
    assert "@media (max-height: 560px)" in HTML

    short = HTML.split("@media (max-height: 700px)", 1)[1].split("}", 1)[0]
    assert "--ring: 140px" in short

    shorter = HTML.split("@media (max-height: 560px)", 1)[1].split("\n  }", 1)[0]
    assert "--ring: 108px" in shorter
    assert ".caption" in shorter, "the caption keeps a height the window no longer has"


def test_a_frameless_window_is_given_a_corner_to_resize_by():
    """pywebview draws no chrome at all, so the operating system offers no border.

    Without this the window is stuck at whatever size it was last given, and the
    page is the only thing left that can offer the handle.
    """
    assert '<div id="grip"' in HTML
    opening = HTML.split('<div id="grip"', 1)[1].split(">", 1)[0]
    assert "nodrag" in opening, "the grip sits in the drag region and moves the window instead"

    grip = HTML.split('<div id="grip"', 1)[1].split("</div>", 1)[0]
    assert grip.count("<path") == 3, "a resize corner is three diagonal hairlines"

    style = HTML.split("  #grip {", 1)[1].split("}", 1)[0]
    assert "cursor: nwse-resize" in style

    bar = HTML.split('<header id="bar">', 1)[1].split("</header>", 1)[0]
    assert 'id="grip"' not in bar, "the grip is in the title bar"


def test_the_grip_is_throttled_and_sends_what_it_was_dragged_to():
    """A pointer moves far more often than a window can usefully be resized.

    The names matter as much as the numbers: this frame is read in another file, and
    ``w``/``h`` would be two missing fields clamped to the smallest window there is.
    """
    from jarvis.desk import server as server_module

    assert "resize" in server_module.WINDOW_ACTIONS, "the server no longer resizes; this is stale"

    body = SCRIPT.split("function gripSend(", 1)[1].split("\n}", 1)[0]
    assert 'action: "resize"' in body
    assert "width:" in body and "height:" in body
    assert "window.outerWidth" in SCRIPT, "the size is not measured from the window it resizes"
    assert "requestAnimationFrame(gripSend)" in SCRIPT, "one frame per pointer move"

    reader = inspect.getsource(server_module.DeskServer._on_window)
    assert '"width"' in reader and '"height"' in reader, "the two sides disagree on the field names"


def test_the_title_bar_can_maximise_and_restore():
    """Every other window on the machine does this, by button and by double-click."""
    assert '{ type: "window", action: next ? "maximize" : "restore" }' in SCRIPT
    assert 'ui.maxi.addEventListener("click", toggleMaximize)' in SCRIPT
    assert 'ui.drag.addEventListener("dblclick", toggleMaximize)' in SCRIPT

    bar = HTML.split('<header id="bar">', 1)[1].split("</header>", 1)[0]
    assert bar.index('id="min"') < bar.index('id="maxi"') < bar.index('id="close"')


def test_an_acknowledgement_that_failed_is_said_out_loud():
    """Pressing the reactor mid-turn is answered with ok:false and nothing else.

    Nothing happening is indistinguishable from a window that has quietly stopped
    listening, so the one case where it did not work has to be spoken.
    """
    assert 'case "ack":' in SCRIPT

    body = SCRIPT.split("function onAck(", 1)[1].split("\n}", 1)[0]
    assert "message.ok !== false" in body, "a successful ack is not silent"
    assert "true" in body, "a refusal is drawn as ordinary narration"

    lines = SCRIPT.split("const ACK_REFUSED = {", 1)[1].split("};", 1)[0]
    assert "I am in the middle of something, sir." in lines
    assert "There is nothing waiting, sir." in lines


def test_a_countdown_with_nothing_to_count_says_so():
    """A timer whose due time never arrived rendered "NaN:NaN" on the card."""
    body = SCRIPT.split("function countdown(", 1)[1].split("\n}", 1)[0]

    assert "isFinite" in body
    assert '"--:--"' in body


def test_a_timer_card_with_no_end_falls_back_to_what_he_said():
    """The due time is the one field this card exists to show.

    Without it the renderer drew a dash and told ``fillCard`` it had drawn a card,
    which suppressed the sentence that was the whole of the answer.
    """
    branch = SCRIPT.split('case "set_timer": {', 1)[1].split("\n    }", 1)[0]
    guard, _, rest = branch.partition("return false;")

    assert "isFinite" in guard, "the branch never checks for a due time"
    assert "appendChild" not in guard, "the card is half drawn before it gives up"
    assert "appendChild" in rest, "this test is looking at the wrong branch"


def test_the_small_things_that_sat_a_pixel_out():
    """A butler is judged on exactly this sort of thing."""
    # HALT was 5px off the reactor's axis whenever the note line was empty.
    assert ".note:empty { display: none; }" in HTML
    # The countdown wrapped and took the Cancel button down with it at 880px.
    left = HTML.split("  #confirmLeft {", 1)[1].split("}", 1)[0]
    assert "white-space: nowrap" in left and "flex: 0 0 auto" in left
    # The feed is selectable, and an arrow cursor over a paragraph says it is not.
    assert ".line, .card, .lg { cursor: text; }" in HTML


def test_the_log_and_the_filter_stay_on_the_facts_row_at_any_width():
    """Wrapped onto a second line they move the composer up and the drawer off.

    The facts are furniture and may be cut short; these two are controls.
    """
    facts = HTML.split('<div class="facts" id="facts">', 1)[1].split("</div>", 1)[0]
    assert '<span class="controls-right">' in facts
    assert facts.index("controls-right") < facts.index('id="onlyProblems"')
    assert facts.index("controls-right") < facts.index('id="drawerToggle"')

    style = HTML.split("  .facts .controls-right {", 1)[1].split("}", 1)[0]
    assert "margin-left: auto" in style

    row = HTML.split("  .facts {", 1)[1].split("}", 1)[0]
    assert "flex-wrap: nowrap" in row
    assert "text-overflow: ellipsis" in HTML.split("  .facts .fact {", 1)[1].split("}", 1)[0]


def test_the_page_carries_its_own_icon():
    """Edge's app mode gives the window a tab icon whether we provide one or not.

    The one it invents is a grey globe, which is not what is running in that window.
    """
    icon = re.search(r'<link rel="icon"[^>]*href="([^"]*)"', HTML)

    assert icon, "the window falls back to a generic icon"
    assert icon.group(1).startswith("data:image/svg+xml,"), "the icon is not carried inline"
    assert "%3Ccircle" in icon.group(1), "the icon is not the reactor"


def test_the_desk_rings_a_number_the_way_the_phone_calls_one():
    """One product does not have two words for one verb per window.

    The phone shipped first and says Call, so the desk says Call.
    """
    phone = PHONE.read_text(encoding="utf-8")

    assert '"Call " + d.phone' in phone, "the phone stopped saying Call; this test is stale"
    assert '"Call " + d.phone' in SCRIPT
    assert '"Ring " + d.phone' not in SCRIPT


def test_a_tools_duration_is_rounded_before_anybody_reads_it():
    """The assistant sends this as a float and the phone's reporter sends it whole.

    The chip and the working line are the same width either way; 641.8271000003 ms
    is not.
    """
    for place in ("function fillCard(", "function onTool("):
        body = SCRIPT.split(place, 1)[1].split("\n}", 1)[0]
        assert "Math.round(Number(ev.ms))" in body, f"{place.split()[1]} draws a raw float"
