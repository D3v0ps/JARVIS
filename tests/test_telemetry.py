"""One reading of the machine, on a machine with no psutil, no GPU and no assistant.

The rule this module exists to enforce is that a missing counter is a missing key.
A status strip that invents a zero is worse than one that shows a dash, and a status
strip that throws takes a face down with it - so every source here is faked absent as
well as present.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from jarvis.core.state import AssistantState, StateBus
from jarvis.core.telemetry import reset_cache, snapshot


@pytest.fixture(autouse=True)
def fresh_cache():
    """There is one machine and therefore one cache; tests must not inherit it."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def no_psutil(monkeypatch):
    """A box where psutil was never installed - the default on this developer's Linux."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    return None


@pytest.fixture
def fake_psutil(monkeypatch):
    """Counters that answer instantly and always say the same thing."""
    gigabyte = 1024 ** 3
    module = SimpleNamespace(
        cpu_percent=lambda interval=None: 37.4,
        virtual_memory=lambda: SimpleNamespace(
            percent=61.2, used=19 * gigabyte, total=32 * gigabyte
        ),
        disk_usage=lambda path: SimpleNamespace(free=250 * gigabyte, total=931 * gigabyte),
        boot_time=lambda: time.time() - 7200.0,
    )
    monkeypatch.setitem(sys.modules, "psutil", module)
    return module


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """No nvidia-smi by default, so no test ever shells out to a real driver."""
    monkeypatch.setattr("jarvis.tools.system_tools._gpu_info", lambda: None)


class FakeAssistant:
    """The two attributes telemetry actually reaches for, and nothing else."""

    def __init__(self, *, scheduler=None, model: str = "qwen3:8b") -> None:
        state = StateBus()
        state.set(AssistantState.THINKING)
        self.parts = SimpleNamespace(
            state=state,
            scheduler=scheduler,
            brain=SimpleNamespace(client=SimpleNamespace(model=model)),
        )


def job(identifier: str, due: float, *, label: str = "", text: str = "", kind: str = "timer"):
    return SimpleNamespace(id=identifier, due=due, label=label, text=text, kind=kind)


# --- the machine ------------------------------------------------------------------------
def test_a_machine_without_psutil_reports_no_counters_rather_than_zeroes(no_psutil):
    """A dash on the strip is honest; a zero would have the operator chasing a ghost."""
    reading = snapshot()

    for key in ("cpu", "ram", "ram_used_gb", "ram_total_gb", "disk_free_gb", "uptime_s"):
        assert key not in reading
    assert reading["state"] == "idle"


def test_the_counters_are_reported_in_the_units_the_faces_draw(fake_psutil):
    reading = snapshot()

    assert reading["cpu"] == 37
    assert reading["ram"] == 61
    assert reading["ram_used_gb"] == 19.0
    assert reading["ram_total_gb"] == 32.0
    assert reading["disk_free_gb"] == 250.0
    assert 7195 <= reading["uptime_s"] <= 7205


def test_one_broken_counter_does_not_take_the_others_with_it(monkeypatch, fake_psutil):
    def explode():
        raise RuntimeError("no memory controller, apparently")

    monkeypatch.setattr(fake_psutil, "virtual_memory", explode)

    reading = snapshot()

    assert reading["cpu"] == 37
    assert "ram" not in reading
    assert "uptime_s" in reading


def test_the_gpu_key_is_absent_on_a_machine_with_no_nvidia_card(fake_psutil):
    assert "gpu" not in snapshot()


def test_the_gpu_is_read_through_the_one_nvidia_smi_reader_we_already_have(monkeypatch):
    """Two readers would mean two subprocesses per poll for the same four numbers."""
    monkeypatch.setattr(
        "jarvis.tools.system_tools._gpu_info",
        lambda: {
            "util_percent": 44.6,
            "temperature_c": 61.2,
            "vram_used_mb": 5120.0,
            "vram_total_mb": 16384.0,
        },
    )

    gpu = snapshot()["gpu"]

    assert gpu == {
        "util_percent": 45,
        "temperature_c": 61,
        "vram_used_mb": 5120,
        "vram_total_mb": 16384,
    }


def test_a_crashing_gpu_reader_costs_the_gpu_key_and_nothing_else(monkeypatch, fake_psutil):
    def explode():
        raise OSError("nvidia-smi went missing mid-poll")

    monkeypatch.setattr("jarvis.tools.system_tools._gpu_info", explode)

    reading = snapshot()

    assert "gpu" not in reading
    assert reading["cpu"] == 37


# --- the cache ---------------------------------------------------------------------------
def test_the_machine_is_read_once_for_all_the_faces_watching_it(monkeypatch, fake_psutil):
    """Two faces polling every two seconds must not mean four readings."""
    calls: list[float] = []
    monkeypatch.setattr(fake_psutil, "cpu_percent", lambda interval=None: calls.append(0) or 12.0)

    snapshot()
    snapshot()
    snapshot()

    assert len(calls) == 1


def test_reset_cache_makes_the_next_reading_a_real_one(monkeypatch, fake_psutil):
    calls: list[float] = []
    monkeypatch.setattr(fake_psutil, "cpu_percent", lambda interval=None: calls.append(0) or 12.0)

    snapshot()
    reset_cache()
    snapshot()

    assert len(calls) == 2


def test_a_caller_that_asks_for_no_cache_always_gets_a_fresh_reading(monkeypatch, fake_psutil):
    calls: list[float] = []
    monkeypatch.setattr(fake_psutil, "cpu_percent", lambda interval=None: calls.append(0) or 12.0)

    snapshot(cache_seconds=0)
    snapshot(cache_seconds=0)

    assert len(calls) == 2


def test_adding_keys_to_one_reading_does_not_poison_the_next(fake_psutil, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.system_tools._gpu_info",
        lambda: {"util_percent": 10.0, "temperature_c": 50.0,
                 "vram_used_mb": 1.0, "vram_total_mb": 2.0},
    )
    first = snapshot()
    first["ok"] = True
    first["gpu"]["temperature_c"] = 999

    second = snapshot()

    assert "ok" not in second
    assert second["gpu"]["temperature_c"] == 50


def test_the_state_is_never_cached_because_a_state_light_must_be_current(fake_psutil):
    assistant = FakeAssistant()

    assert snapshot(assistant)["state"] == "thinking"
    assistant.parts.state.set(AssistantState.SPEAKING)
    assert snapshot(assistant)["state"] == "speaking"


# --- the assistant --------------------------------------------------------------------------
def test_without_an_assistant_the_reading_is_just_the_machine(fake_psutil):
    reading = snapshot()

    assert reading["state"] == "idle"
    assert "timers" not in reading
    assert "model" not in reading


def test_the_soonest_five_timers_are_reported_with_something_to_label_them(fake_psutil):
    jobs = [job(f"t{n}", due=100.0 - n, label=f"job {n}") for n in range(8)]
    assistant = FakeAssistant(scheduler=SimpleNamespace(pending=lambda: jobs))

    timers = snapshot(assistant)["timers"]

    assert [t["id"] for t in timers] == ["t7", "t6", "t5", "t4", "t3"], "soonest first"
    assert timers[0] == {"id": "t7", "label": "job 7", "due": 93.0, "kind": "timer"}


def test_a_timer_with_no_label_falls_back_to_its_text_and_then_its_kind(fake_psutil):
    jobs = [
        job("a", due=1.0, text="take the pasta off"),
        job("b", due=2.0, kind="reminder"),
    ]
    assistant = FakeAssistant(scheduler=SimpleNamespace(pending=lambda: jobs))

    timers = snapshot(assistant)["timers"]

    assert timers[0]["label"] == "take the pasta off"
    assert timers[1]["label"] == "reminder"


def test_a_broken_scheduler_costs_the_timers_and_never_the_reading(fake_psutil):
    """A glance at a face must never throw because a job list did."""
    def explode():
        raise RuntimeError("boom")

    assistant = FakeAssistant(scheduler=SimpleNamespace(pending=explode))

    reading = snapshot(assistant)

    assert reading["timers"] == []
    assert reading["cpu"] == 37


def test_the_model_reported_is_the_one_the_client_is_actually_holding(fake_psutil):
    """Wiring substitutes a pulled model when the configured one is missing."""
    assistant = FakeAssistant(model="llama3.1:8b")

    assert snapshot(assistant)["model"] == "llama3.1:8b"


def test_an_assistant_that_cannot_name_its_model_simply_does_not(fake_psutil):
    assistant = FakeAssistant()
    assistant.parts.brain = None

    assert "model" not in snapshot(assistant)


def test_an_assistant_that_is_barely_built_yet_still_yields_a_reading(fake_psutil):
    """Telemetry is polled during start-up, when half of parts is still None."""
    reading = snapshot(SimpleNamespace())

    assert reading["state"] == "idle"
    assert reading["cpu"] == 37
