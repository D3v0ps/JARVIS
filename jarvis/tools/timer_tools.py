"""Timer and reminder tools, layered over :class:`jarvis.core.scheduler.Scheduler`.

The scheduler owns the thread, the persistence and the firing; these tools own the
words. Every confirmation names the moment the job is actually due. Pure standard
library, so this module imports anywhere.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from jarvis.core.logging import get_logger
from jarvis.core.scheduler import Job, parse_when
from jarvis.tools.base import Tier, ToolContext, ToolResult
from jarvis.tools.registry import tool

__all__ = ["set_timer", "set_reminder", "list_timers", "cancel_timer", "MAX_SPOKEN_JOBS"]

_log = get_logger("tools.timer")

#: How many pending jobs ``list_timers`` reads out before it counts the rest.
MAX_SPOKEN_JOBS = 3

#: Longest timer we accept, in minutes (one week). Beyond that, ask for a reminder.
MAX_TIMER_MINUTES = 7 * 24 * 60

#: Spoken when a scheduler is not wired up at all.
NO_SCHEDULER = "I have no way to keep time at the moment, sir."

#: Words that mean "every timer".
_ALL_WORDS = {"all", "all of them", "everything", "every timer", "them all", "alla"}

#: Argument names callers may use for the thing to cancel.
_WHICH_KEYS = ("which", "label_or_id", "label", "id", "timer", "name")

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"]


def _spoken_number(value: int) -> str:
    """Render a small whole number as words so the sentence reads well aloud."""
    number = int(value)
    if number < 0 or number > 999:
        return str(number)
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"
    hundreds, rest = divmod(number, 100)
    head = f"{_UNITS[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_spoken_number(rest)}"


def _plural(count: int, singular: str, plural: str) -> str:
    """"one minute" / "three minutes", with the number spelled out."""
    word = singular if count == 1 else plural
    return f"{_spoken_number(count)} {word}"


def _spoken_duration(minutes: float) -> str:
    """Describe a duration the way a person would say it out loud."""
    seconds = int(round(float(minutes) * 60))
    if seconds < 60:
        return _plural(max(1, seconds), "second", "seconds")
    if seconds < 600 and seconds % 60:
        return _plural(seconds, "second", "seconds")
    whole = int(round(seconds / 60))
    if whole < 60:
        return _plural(whole, "minute", "minutes")
    hours, rest = divmod(whole, 60)
    head = _plural(hours, "hour", "hours")
    return head if rest == 0 else f"{head} and {_plural(rest, 'minute', 'minutes')}"


def _clean(value: Any) -> str:
    """Collapse an argument to a single trimmed line."""
    return " ".join(str(value or "").split())


def _spoken_time(moment: datetime, now: datetime | None = None) -> str:
    """"21:57", "tomorrow at 08:00" or "on Friday at 08:00", as appropriate."""
    reference = now or datetime.now()
    clock = moment.strftime("%H:%M")
    delta_days = (moment.date() - reference.date()).days
    if delta_days <= 0:
        return clock
    if delta_days == 1:
        return f"tomorrow at {clock}"
    if delta_days < 7:
        return f"on {moment.strftime('%A')} at {clock}"
    return f"on {moment.strftime('%d %B')} at {clock}"


def _describe(job: Job, now: datetime | None = None) -> str:
    """One pending job as a spoken fragment, e.g. "tea at 21:57"."""
    what = _clean(job.label) or _clean(job.text) or (
        "a reminder" if job.kind == "reminder" else "a timer"
    )
    return f"{what} at {_spoken_time(job.due_at, now)}"


def _scheduler(ctx: ToolContext) -> Any:
    """The scheduler, or ``None`` when the assistant was built without one."""
    return getattr(ctx, "scheduler", None)


def _pending(ctx: ToolContext) -> list[Job]:
    """Pending jobs, soonest first, never raising."""
    scheduler = _scheduler(ctx)
    if scheduler is None:
        return []
    try:
        return list(scheduler.pending())
    except Exception as exc:  # noqa: BLE001 - a broken store must not kill the turn
        _log.error("Could not read the pending jobs: %s", exc)
        _log.debug("pending() traceback", exc_info=True)
        return []


def _join(items: list[str]) -> str:
    """Join spoken fragments with commas and a final "and"."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


@tool(
    "set_timer",
    description=(
        "Start a countdown and announce it when it runs out. Use it for cooking, "
        "breaks and anything measured in minutes from now."
    ),
    parameters={
        "type": "object",
        "properties": {
            "minutes": {
                "type": "number",
                "description": "How many minutes from now the timer should fire.",
            },
            "label": {
                "type": "string",
                "description": "Optional short name, for example 'pasta' or 'laundry'.",
            },
        },
        "required": ["minutes"],
    },
    tier=Tier.SAFE,
)
def set_timer(ctx: ToolContext, args: dict) -> ToolResult:
    """Start a countdown and confirm it with the wall-clock time it will fire."""
    scheduler = _scheduler(ctx)
    if scheduler is None:
        return ToolResult.fail(NO_SCHEDULER, "ToolContext has no scheduler.")
    raw = args.get("minutes")
    try:
        minutes = float(raw)
    except (TypeError, ValueError):
        minutes = float("nan")
    if not math.isfinite(minutes):  # not a number, NaN or infinity
        return ToolResult.fail(
            "I need a number of minutes for the timer, sir.", f"minutes={raw!r}"
        )
    if minutes <= 0:
        return ToolResult.fail("A timer needs to be longer than nothing, sir.")
    if minutes > MAX_TIMER_MINUTES:
        return ToolResult.fail(
            "That's rather long for a timer, sir, shall I set a reminder instead?",
            f"Requested {minutes} minutes, the cap is {MAX_TIMER_MINUTES}.",
        )
    label = _clean(args.get("label"))
    try:
        job = scheduler.add_timer(minutes, label)
    except Exception as exc:  # noqa: BLE001 - scheduling must never raise at the user
        _log.error("Could not set a timer for %s minutes: %s", minutes, exc)
        _log.debug("add_timer() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't set that timer, sir.", f"{type(exc).__name__}: {exc}"
        )
    named = f" for {label}" if label else ""
    summary = (
        f"Timer{named} set for {_spoken_duration(minutes)}, sir - that's "
        f"{_spoken_time(job.due_at)}."
    )
    return ToolResult(
        ok=True,
        summary=summary,
        detail=f"Job {job.id}: {label or 'timer'} due {job.due_at.isoformat(timespec='seconds')}",
        data={"id": job.id, "label": label, "minutes": minutes, "due": job.due},
    )


@tool(
    "set_reminder",
    description=(
        "Remind the user of something at a given time, for example 'at 19:30', "
        "'in two hours' or 'tomorrow at 08:00'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What to say when the reminder fires.",
            },
            "when": {
                "type": "string",
                "description": (
                    "When to fire it: a clock time like '19:30', a relative time like "
                    "'in two hours', or 'tomorrow at 08:00'."
                ),
            },
        },
        "required": ["text", "when"],
    },
    tier=Tier.SAFE,
)
def set_reminder(ctx: ToolContext, args: dict) -> ToolResult:
    """Schedule a reminder, asking again when the time cannot be understood.

    The expression is parsed here first with :func:`parse_when`, so an unclear phrase
    becomes a short spoken question rather than a guess or a traceback.
    """
    scheduler = _scheduler(ctx)
    if scheduler is None:
        return ToolResult.fail(NO_SCHEDULER, "ToolContext has no scheduler.")
    text = _clean(args.get("text"))
    when = _clean(args.get("when"))
    if not text:
        return ToolResult.fail("What should I remind you about, sir?")
    if not when:
        return ToolResult.fail("When would you like that reminder, sir?")
    moment = parse_when(when)
    if moment is None:
        return ToolResult.fail(
            f"I couldn't make out a time from {when}, sir, could you give me a clock "
            "time or a number of minutes?",
            f"parse_when({when!r}) returned None.",
        )
    try:
        job = scheduler.add_reminder(text, when)
    except Exception as exc:  # noqa: BLE001 - never surface a traceback to the user
        _log.error("Could not set a reminder for %r at %r: %s", text, when, exc)
        _log.debug("add_reminder() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't set that reminder, sir.", f"{type(exc).__name__}: {exc}"
        )
    summary = f"I'll remind you to {text} at {_spoken_time(job.due_at)}, sir."
    return ToolResult(
        ok=True,
        summary=summary,
        detail=f"Job {job.id}: {text} due {job.due_at.isoformat(timespec='seconds')}",
        data={"id": job.id, "text": text, "when": when, "due": job.due},
    )


@tool(
    "list_timers",
    description="Say which timers and reminders are still pending and when they fire.",
    parameters={"type": "object", "properties": {}, "required": []},
    tier=Tier.SAFE,
)
def list_timers(ctx: ToolContext, args: dict) -> ToolResult:
    """Read back the pending timers and reminders, soonest first."""
    scheduler = _scheduler(ctx)
    if scheduler is None:
        return ToolResult.fail(NO_SCHEDULER, "ToolContext has no scheduler.")
    jobs = _pending(ctx)
    if not jobs:
        return ToolResult(
            ok=True,
            summary="Nothing is pending, sir.",
            detail="No scheduled jobs.",
            data={"jobs": []},
        )
    now = datetime.now()
    spoken = [_describe(job, now) for job in jobs[:MAX_SPOKEN_JOBS]]
    extra = len(jobs) - len(spoken)
    summary = f"You have {_join(spoken)}, sir"
    if extra > 0:
        summary += f", and {_spoken_number(extra)} more"
    summary += "."
    return ToolResult(
        ok=True,
        summary=summary,
        detail="\n".join(
            f"{job.id} [{job.kind}] {job.label or job.text} due "
            f"{job.due_at.isoformat(timespec='seconds')}"
            for job in jobs
        ),
        data={"jobs": [{"id": job.id, "kind": job.kind, "label": job.label,
                        "due": job.due} for job in jobs]},
    )


@tool(
    "cancel_timer",
    description=(
        "Cancel a pending timer or reminder, matched by its label or its id. Say "
        "'all' to cancel every pending one."
    ),
    parameters={
        "type": "object",
        "properties": {
            "which": {
                "type": "string",
                "description": (
                    "The timer's label, part of its label, or its id such as 't1'. "
                    "Use 'all' to cancel everything pending."
                ),
            }
        },
        "required": ["which"],
    },
    tier=Tier.SAFE,
)
def cancel_timer(ctx: ToolContext, args: dict) -> ToolResult:
    """Cancel a pending job matched by id, by label, or by part of a label.

    An id wins over a label, so "t1" always means the job with that id. When several
    labels match, the soonest is cancelled and the rest are mentioned.
    """
    scheduler = _scheduler(ctx)
    if scheduler is None:
        return ToolResult.fail(NO_SCHEDULER, "ToolContext has no scheduler.")
    which = ""
    for key in _WHICH_KEYS:
        which = _clean(args.get(key))
        if which:
            break
    if not which:
        return ToolResult.fail("Which timer should I cancel, sir?")

    jobs = _pending(ctx)
    if not jobs:
        return ToolResult(
            ok=True,
            summary="Nothing is pending, sir.",
            detail="No scheduled jobs to cancel.",
            data={"which": which, "cancelled": 0},
        )

    needle = which.lower()
    if needle in _ALL_WORDS:
        cancelled = [job for job in jobs if scheduler.cancel(job.id)]
        if not cancelled:
            return ToolResult.fail("I couldn't cancel those, sir.", f"which={which!r}")
        summary = (
            "Cancelled, sir, nothing is pending now."
            if len(cancelled) == 1
            else f"Cancelled all {_spoken_number(len(cancelled))}, sir."
        )
        return ToolResult(
            ok=True,
            summary=summary,
            detail="Cancelled: " + ", ".join(job.id for job in cancelled),
            data={"which": which, "cancelled": len(cancelled),
                  "ids": [j.id for j in cancelled]},
        )

    matches = [job for job in jobs if job.id.lower() == needle]
    if not matches:
        matches = [
            job
            for job in jobs
            if needle in job.label.lower() or needle in job.text.lower()
        ]
    if not matches:
        return ToolResult(
            ok=True,
            summary=f"I have no timer matching {which}, sir.",
            detail=f"No pending job matched {which!r}.",
            data={"which": which, "cancelled": 0},
        )

    target = matches[0]
    try:
        removed = bool(scheduler.cancel(target.id))
    except Exception as exc:  # noqa: BLE001 - keep storage faults out of the voice path
        _log.error("Could not cancel job %s: %s", target.id, exc)
        _log.debug("cancel() traceback", exc_info=True)
        return ToolResult.fail(
            "I couldn't cancel that one, sir.", f"{type(exc).__name__}: {exc}"
        )
    if not removed:
        return ToolResult.fail(
            f"I couldn't cancel {which}, sir.", f"Scheduler refused to cancel {target.id}."
        )
    what = _clean(target.label) or _clean(target.text) or "that timer"
    summary = f"Cancelled {what}, sir."
    if len(matches) > 1:
        summary = (
            f"Cancelled the next one, {what}, sir - {_spoken_number(len(matches) - 1)} "
            "more still match."
        )
    return ToolResult(
        ok=True,
        summary=summary,
        detail=f"Cancelled job {target.id} ({target.kind}).",
        data={"which": which, "cancelled": 1, "ids": [target.id]},
    )
