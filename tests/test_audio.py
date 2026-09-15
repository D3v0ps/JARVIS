"""Microphone capture and playback, against a fake PortAudio.

``sounddevice`` is not installed on this machine and never will be in CI, so a fake
module is pushed into ``sys.modules`` for the duration of each test. Its streams
record what they were handed and can drive the capture callback from a thread, which
is exactly what the real library does — only deterministically.

What is being pinned down here is what a person notices: audio that keeps flowing
when the consumer stutters (newest frames win, oldest are dropped), a read that
gives up instead of hanging, a barge-in that silences JARVIS mid-sentence, and a
machine with no sound card that still runs the assistant instead of crashing it.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from jarvis.audio.capture import QUEUE_MAXSIZE, MicStream, NullMicStream
from jarvis.audio.player import BLOCK_MS, Player

SR = 24000


# ------------------------------------------------------------- the fake device


class FakeInputStream:
    """Stands in for ``sounddevice.InputStream`` and can fire the callback."""

    def __init__(self, *, samplerate, blocksize, device, channels, dtype, callback):
        self.state = type(self).state
        if self.state.input_error is not None:
            raise self.state.input_error
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.dtype = dtype
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        self.state.input_streams.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.stopped = True

    def close(self):
        self.closed = True

    # -- test helpers -------------------------------------------------------
    def emit(self, block, status=None):
        """Deliver one block through the PortAudio callback, from this thread."""
        array = np.asarray(block, dtype=np.float32)
        self.callback(array, array.shape[0], None, status)

    def emit_tone(self, n, value=0.25):
        block = np.full(n, value, dtype=np.float32)
        self.emit(block)
        return block

    def emit_from_thread(self, blocks, gap=0.0):
        """Push blocks from a background thread, like PortAudio would."""

        def run():
            for block in blocks:
                if gap:
                    time.sleep(gap)
                self.emit(block)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread


class FakeOutputStream:
    """Stands in for ``sounddevice.OutputStream`` and records every written block."""

    def __init__(self, *, samplerate, blocksize, device, channels, dtype):
        self.state = type(self).state
        if self.state.allowed_rates is not None and samplerate not in self.state.allowed_rates:
            raise RuntimeError(f"Invalid sample rate {samplerate}")
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.dtype = dtype
        self.written: list[np.ndarray] = []
        self.started = False
        self.closed = False
        self.state.output_streams.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def write(self, chunk):
        if self.state.write_delay:
            time.sleep(self.state.write_delay)
        if self.state.write_error is not None and len(self.written) >= self.state.write_error_after:
            raise self.state.write_error
        self.written.append(np.array(chunk, dtype=np.float32, copy=True))

    @property
    def samples(self) -> int:
        return int(sum(chunk.size for chunk in self.written))

    def audio(self) -> np.ndarray:
        if not self.written:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.written)


class FakeSoundDeviceState:
    def __init__(self):
        self.input_streams: list[FakeInputStream] = []
        self.output_streams: list[FakeOutputStream] = []
        self.input_error: BaseException | None = None
        self.allowed_rates: set[int] | None = None
        self.write_delay: float = 0.0
        self.write_error: BaseException | None = None
        self.write_error_after: int = 0
        self.devices = [
            {
                "index": 0,
                "name": "Fake Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "index": 1,
                "name": "Fake Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]


@pytest.fixture
def fake_sd(monkeypatch):
    """Insert a fake ``sounddevice`` into sys.modules; monkeypatch removes it after."""
    state = FakeSoundDeviceState()

    module = types.ModuleType("sounddevice")
    input_cls = type("InputStream", (FakeInputStream,), {"state": state})
    output_cls = type("OutputStream", (FakeOutputStream,), {"state": state})
    module.InputStream = input_cls
    module.OutputStream = output_cls
    module.query_devices = lambda: list(state.devices)
    module.default = types.SimpleNamespace(device=(0, 1))
    module.state = state

    monkeypatch.setitem(sys.modules, "sounddevice", module)
    yield state
    assert "sounddevice" in sys.modules  # monkeypatch undo happens after this fixture


@pytest.fixture
def mic(fake_sd):
    """A started MicStream on the fake device, closed afterwards."""
    stream = MicStream(device=None, sample_rate=16000, block_size=160)
    stream.start()
    yield stream
    stream.stop()


def device(fake_sd) -> FakeInputStream:
    return fake_sd.input_streams[-1]


def wait_until(predicate, timeout=1.0, interval=0.005) -> bool:
    """Poll ``predicate`` until true or ``timeout`` expires. Never sleeps long."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


# =========================================================== MicStream capture


def test_frames_pushed_by_the_device_come_back_from_read(mic, fake_sd):
    block = device(fake_sd).emit_tone(160, 0.5)
    got = mic.read(timeout=0.5)
    assert got is not None
    np.testing.assert_allclose(got, block)


def test_stream_is_opened_as_mono_float32_at_the_requested_rate(fake_sd):
    stream = MicStream(device=None, sample_rate=16000, block_size=1280)
    stream.start()
    try:
        opened = device(fake_sd)
        assert opened.samplerate == 16000
        assert opened.blocksize == 1280
        assert opened.channels == 1
        assert opened.dtype == "float32"
        assert opened.started is True
    finally:
        stream.stop()


def test_stereo_input_from_the_driver_is_mixed_down_to_mono(mic, fake_sd):
    stereo = np.stack(
        [np.full(160, 1.0, dtype=np.float32), np.full(160, 0.0, dtype=np.float32)], axis=1
    )
    device(fake_sd).emit(stereo)
    got = mic.read(timeout=0.5)
    assert got.ndim == 1
    assert got.shape == (160,)
    np.testing.assert_allclose(got, 0.5)


def test_single_channel_column_input_is_flattened(mic, fake_sd):
    column = np.full((160, 1), 0.3, dtype=np.float32)
    device(fake_sd).emit(column)
    got = mic.read(timeout=0.5)
    assert got.shape == (160,)
    np.testing.assert_allclose(got, 0.3)


def test_non_float32_driver_input_is_converted(mic, fake_sd):
    device(fake_sd).emit(np.full(160, 0.25, dtype=np.float64))
    got = mic.read(timeout=0.5)
    assert got.dtype == np.float32


def test_overflow_drops_the_oldest_frame_and_counts_it(mic, fake_sd):
    """The assistant must hear the newest audio, not a growing backlog."""
    opened = device(fake_sd)
    extra = 2
    for index in range(QUEUE_MAXSIZE + extra):
        opened.emit(np.full(160, float(index), dtype=np.float32))

    assert mic.dropped == extra
    first = mic.read(timeout=0.5)
    # Frames 0 and 1 were discarded, so the queue now starts at frame 2.
    assert float(first[0]) == pytest.approx(float(extra))


def test_nothing_is_dropped_while_the_queue_has_room(mic, fake_sd):
    opened = device(fake_sd)
    for index in range(QUEUE_MAXSIZE):
        opened.emit(np.full(160, float(index), dtype=np.float32))
    assert mic.dropped == 0


def test_read_returns_none_instead_of_blocking_forever(mic):
    started = time.monotonic()
    assert mic.read(timeout=0.05) is None
    assert time.monotonic() - started < 1.0


def test_read_honours_a_zero_timeout(mic):
    assert mic.read(timeout=0.0) is None


def test_frames_yields_every_queued_block_in_order(mic, fake_sd):
    opened = device(fake_sd)
    for index in range(3):
        opened.emit(np.full(160, float(index), dtype=np.float32))

    collected = []
    for frame in mic.frames():
        collected.append(float(frame[0]))
        if len(collected) == 3:
            break
    assert collected == [0.0, 1.0, 2.0]


def test_stop_ends_frames_within_a_second(mic, fake_sd):
    """A generator that outlives stop() would wedge the whole assistant loop."""
    collected: list[np.ndarray] = []
    finished = threading.Event()

    def consume():
        for frame in mic.frames():
            collected.append(frame)
        finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    opened = device(fake_sd)
    for _ in range(3):
        opened.emit_tone(160)
    assert wait_until(lambda: len(collected) == 3, timeout=1.0)

    started = time.monotonic()
    mic.stop()
    assert finished.wait(timeout=1.0), "frames() did not end after stop()"
    assert time.monotonic() - started < 1.0
    consumer.join(timeout=1.0)
    assert not consumer.is_alive()


def test_flush_empties_the_queue(mic, fake_sd):
    opened = device(fake_sd)
    for _ in range(5):
        opened.emit_tone(160)
    mic.flush()
    assert mic.read(timeout=0.05) is None


def test_flush_on_an_empty_queue_is_harmless(mic):
    mic.flush()
    mic.flush()
    assert mic.running is True


def test_start_twice_opens_only_one_stream(fake_sd):
    stream = MicStream(device=None, sample_rate=16000, block_size=160)
    stream.start()
    stream.start()
    try:
        assert len(fake_sd.input_streams) == 1
        assert stream.running is True
    finally:
        stream.stop()


def test_stop_twice_is_harmless(mic):
    mic.stop()
    mic.stop()
    assert mic.running is False


def test_stop_closes_the_underlying_stream(mic, fake_sd):
    opened = device(fake_sd)
    mic.stop()
    assert opened.stopped is True
    assert opened.closed is True


def test_errors_while_closing_the_stream_are_swallowed(mic, fake_sd):
    """Losing the USB microphone must not take the shutdown path down with it."""
    opened = device(fake_sd)
    opened.stop = lambda: (_ for _ in ()).throw(OSError("device vanished"))
    opened.close = lambda: (_ for _ in ()).throw(OSError("device vanished"))
    mic.stop()
    assert mic.running is False


def test_a_device_that_cannot_be_opened_raises_a_clear_runtime_error(fake_sd):
    fake_sd.input_error = OSError("PortAudio: Device unavailable")
    stream = MicStream(device=None, sample_rate=16000, block_size=160)
    with pytest.raises(RuntimeError) as excinfo:
        stream.start()
    message = str(excinfo.value)
    assert "microphone" in message.lower()
    assert "Device unavailable" in message
    assert stream.running is False


def test_an_unknown_device_name_raises_a_clear_runtime_error(fake_sd):
    stream = MicStream(device="Nonexistent Headset", sample_rate=16000, block_size=160)
    with pytest.raises(RuntimeError) as excinfo:
        stream.start()
    assert "Nonexistent Headset" in str(excinfo.value)
    assert stream.running is False


def test_a_named_device_is_resolved_to_its_index(fake_sd):
    stream = MicStream(device="fake micro", sample_rate=16000, block_size=160)
    stream.start()
    try:
        assert device(fake_sd).device == 0
    finally:
        stream.stop()


def test_missing_sounddevice_raises_a_runtime_error_naming_the_package(monkeypatch):
    """No fake module here: this is exactly the state of a bare Linux box."""
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    stream = MicStream()
    with pytest.raises(RuntimeError) as excinfo:
        stream.start()
    assert "sounddevice" in str(excinfo.value)
    assert stream.running is False


def test_context_manager_starts_and_stops_the_stream(fake_sd):
    with MicStream(device=None, sample_rate=16000, block_size=160) as stream:
        assert stream.running is True
        opened = device(fake_sd)
    assert stream.running is False
    assert opened.closed is True


def test_driver_overrun_status_does_not_lose_the_frame(mic, fake_sd):
    device(fake_sd).emit(np.full(160, 0.4, dtype=np.float32), status="input overflow")
    got = mic.read(timeout=0.5)
    assert got is not None
    np.testing.assert_allclose(got, 0.4)


def test_two_threads_feeding_the_callback_lose_nothing(mic, fake_sd):
    """PortAudio is one thread, but the queue must still be honest under contention."""
    opened = device(fake_sd)
    per_thread = 20
    threads = [
        opened.emit_from_thread([np.full(160, 0.1, dtype=np.float32)] * per_thread)
        for _ in range(2)
    ]
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    received = 0
    while mic.read(timeout=0.05) is not None:
        received += 1
    assert received + mic.dropped == per_thread * 2


# ========================================================== NullMicStream


def test_null_mic_yields_silent_frames_of_the_requested_size():
    stream = NullMicStream(sample_rate=16000, block_size=160)
    stream.start()
    try:
        frame = stream.read(timeout=0.5)
        assert frame is not None
        assert frame.shape == (160,)
        assert frame.dtype == np.float32
        assert not np.any(frame)
    finally:
        stream.stop()


def test_null_mic_read_returns_none_when_it_was_never_started():
    stream = NullMicStream(block_size=160)
    assert stream.read(timeout=0.05) is None


def test_null_mic_frames_end_promptly_after_stop():
    stream = NullMicStream(sample_rate=16000, block_size=160)
    stream.start()
    seen = []
    finished = threading.Event()

    def consume():
        for frame in stream.frames():
            seen.append(frame)
        finished.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert wait_until(lambda: len(seen) >= 1, timeout=1.0)

    started = time.monotonic()
    stream.stop()
    assert finished.wait(timeout=1.0)
    assert time.monotonic() - started < 1.0


def test_null_mic_never_drops_and_needs_no_flush():
    stream = NullMicStream(block_size=160)
    stream.start()
    try:
        assert stream.dropped == 0
        assert stream.flush() is None
        assert stream.running is True
    finally:
        stream.stop()
    assert stream.running is False


def test_null_mic_offers_the_same_interface_as_micstream():
    """Downstream code holds one or the other and must not care which."""
    expected = {"start", "stop", "read", "frames", "flush", "dropped", "running", "stats"}
    assert expected <= set(dir(NullMicStream))
    assert expected <= set(dir(MicStream))
    with NullMicStream(block_size=160) as stream:
        assert set(stream.stats()) == {"frames_read", "dropped", "running"}


# ================================================================= Player


@pytest.fixture
def player(fake_sd):
    instance = Player(device=None)
    yield instance
    instance.close()


def constant_clip(n=2400, value=0.5, dtype=np.float32):
    return np.full(n, value, dtype=dtype)


def test_a_queued_clip_reaches_the_output_stream(player, fake_sd):
    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    assert len(fake_sd.output_streams) == 1
    assert fake_sd.output_streams[0].samples == 480


def test_the_output_stream_is_opened_at_the_clip_rate_as_mono_float32(player, fake_sd):
    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    stream = fake_sd.output_streams[0]
    assert stream.samplerate == SR
    assert stream.channels == 1
    assert stream.dtype == "float32"
    assert stream.blocksize == int(round(SR * BLOCK_MS / 1000.0))


def test_int16_input_is_scaled_into_float_range(player, fake_sd):
    player.play(np.full(480, 16384, dtype=np.int16), SR)
    player.wait(timeout=2.0)
    np.testing.assert_allclose(fake_sd.output_streams[0].audio(), 0.5, atol=1e-4)


def test_stereo_input_is_mixed_down_to_mono(player, fake_sd):
    stereo = np.stack(
        [np.full(480, 0.6, dtype=np.float32), np.full(480, 0.2, dtype=np.float32)], axis=1
    )
    player.play(stereo, SR)
    player.wait(timeout=2.0)
    stream = fake_sd.output_streams[0]
    assert stream.samples == 480
    np.testing.assert_allclose(stream.audio(), 0.4, atol=1e-6)


def test_mono_column_input_is_accepted(player, fake_sd):
    player.play(np.full((480, 1), 0.25, dtype=np.float32), SR)
    player.wait(timeout=2.0)
    assert fake_sd.output_streams[0].samples == 480


def test_resampling_produces_the_expected_length_when_the_device_refuses_the_rate(
    player, fake_sd
):
    """The device only does 24 kHz, so a 48 kHz clip must arrive at half the length."""
    fake_sd.allowed_rates = {SR}
    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)

    player.play(constant_clip(4800, 0.25), 48000)
    player.wait(timeout=2.0)

    assert len(fake_sd.output_streams) == 2
    assert fake_sd.output_streams[1].samplerate == SR
    assert fake_sd.output_streams[1].samples == 2400
    np.testing.assert_allclose(fake_sd.output_streams[1].audio(), 0.25, atol=1e-6)


def test_a_clip_at_the_stream_rate_keeps_its_exact_length(player, fake_sd):
    player.play(constant_clip(1234), SR)
    player.wait(timeout=2.0)
    assert fake_sd.output_streams[0].samples == 1234


def test_stop_clears_everything_still_queued(player, fake_sd):
    fake_sd.write_delay = 0.004
    player.play(constant_clip(24000), SR)  # ~50 blocks
    for _ in range(5):
        player.play(constant_clip(24000), SR)

    assert wait_until(lambda: fake_sd.output_streams and fake_sd.output_streams[0].written)
    player.stop()
    fake_sd.write_delay = 0.0
    player.wait(timeout=1.0)
    # One clip was in flight and got cut; the other five never played at all.
    assert fake_sd.output_streams[0].samples < 6 * 24000


def test_stop_cuts_the_current_clip_well_under_a_second(player, fake_sd):
    """Barge-in: JARVIS has to shut up the moment the user speaks over him."""
    fake_sd.write_delay = 0.004
    long_clip = constant_clip(SR * 5)  # 250 blocks, ~1 s of fake writing
    player.play(long_clip, SR)

    stream_written = lambda: fake_sd.output_streams and len(fake_sd.output_streams[0].written) >= 3
    assert wait_until(stream_written, timeout=1.0)

    started = time.monotonic()
    player.stop()
    assert wait_until(lambda: not player.is_playing, timeout=0.5)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"stop() took {elapsed * 1000:.0f} ms to silence the speakers"
    fake_sd.write_delay = 0.0
    assert fake_sd.output_streams[0].samples < long_clip.size


def test_stop_on_an_idle_player_is_harmless(player):
    player.stop()
    player.stop()
    assert player.is_playing is False


def test_on_first_audio_fires_once_for_a_burst_of_clips(player, fake_sd):
    calls = []
    player.on_first_audio(lambda: calls.append(time.monotonic()))
    fake_sd.write_delay = 0.004

    player.play(constant_clip(4800), SR)  # ~10 blocks, ~40 ms of fake writing
    player.play(constant_clip(4800), SR)
    player.wait(timeout=2.0)
    fake_sd.write_delay = 0.0

    assert len(calls) == 1


def test_on_first_audio_rearms_after_the_player_goes_idle(player, fake_sd):
    calls = []
    player.on_first_audio(lambda: calls.append(1))

    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    assert wait_until(lambda: len(calls) == 1, timeout=1.0)

    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    assert wait_until(lambda: len(calls) == 2, timeout=1.0)


def test_an_on_first_audio_callback_that_raises_does_not_stop_the_audio(player, fake_sd):
    def explode():
        raise RuntimeError("the UI thread is having a moment")

    player.on_first_audio(explode)
    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    assert fake_sd.output_streams[0].samples == 480


def test_a_silent_player_still_fires_on_first_audio(monkeypatch):
    """Latency instrumentation must not depend on there being a sound card."""
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    calls = []
    instance = Player()
    try:
        instance.on_first_audio(lambda: calls.append(1))
        instance.play(constant_clip(480), SR)
        instance.wait(timeout=2.0)
        assert calls == [1]
    finally:
        instance.close()


def test_player_is_a_silent_sink_when_sounddevice_is_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    instance = Player()
    try:
        instance.play(constant_clip(4800), SR)
        instance.wait(timeout=2.0)
        assert wait_until(lambda: not instance.is_playing, timeout=1.0)
        assert instance.is_playing is False
        assert instance.available is False
    finally:
        instance.close()


def test_an_empty_clip_is_ignored_without_opening_a_stream(player, fake_sd):
    player.play(np.zeros(0, dtype=np.float32), SR)
    player.wait(timeout=1.0)
    assert fake_sd.output_streams == []


def test_a_nonsense_sample_rate_is_refused_without_raising(player, fake_sd):
    player.play(constant_clip(480), 0)
    player.play(constant_clip(480), "fast")
    player.wait(timeout=1.0)
    assert fake_sd.output_streams == []


def test_audio_of_the_wrong_shape_is_refused_without_raising(player, fake_sd):
    player.play(np.zeros((4, 3, 2), dtype=np.float32), SR)
    player.wait(timeout=1.0)
    assert fake_sd.output_streams == []


def test_volume_is_applied_to_every_clip(fake_sd):
    instance = Player(device=None, volume=0.5)
    try:
        instance.play(constant_clip(480, 0.8), SR)
        instance.wait(timeout=2.0)
        np.testing.assert_allclose(fake_sd.output_streams[0].audio(), 0.4, atol=1e-6)
    finally:
        instance.close()


def test_a_loud_clip_is_clipped_rather_than_wrapped(fake_sd):
    instance = Player(device=None, volume=4.0)
    try:
        instance.play(constant_clip(480, 0.9), SR)
        instance.wait(timeout=2.0)
        written = fake_sd.output_streams[0].audio()
        assert float(np.max(written)) <= 1.0
        np.testing.assert_allclose(written, 1.0, atol=1e-6)
    finally:
        instance.close()


def test_wait_times_out_instead_of_hanging(player, fake_sd):
    fake_sd.write_delay = 0.01
    player.play(constant_clip(SR * 5), SR)
    started = time.monotonic()
    player.wait(timeout=0.05)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert player.is_playing is True
    player.stop()
    fake_sd.write_delay = 0.0


def test_play_after_close_is_dropped_silently(player, fake_sd):
    player.close()
    player.play(constant_clip(480), SR)
    assert fake_sd.output_streams == [] or fake_sd.output_streams[0].samples == 0


def test_close_is_idempotent(player):
    player.close()
    player.close()
    assert player.is_playing is False


def test_a_failing_write_does_not_kill_the_playback_worker(player, fake_sd):
    fake_sd.write_error = OSError("PortAudio output underflowed")
    fake_sd.write_error_after = 2
    player.play(constant_clip(SR), SR)
    player.wait(timeout=2.0)

    fake_sd.write_error = None
    player.play(constant_clip(480), SR)
    player.wait(timeout=2.0)
    assert fake_sd.output_streams[-1].samples == 480


def test_two_threads_queueing_clips_at_once_all_get_played(player, fake_sd):
    per_thread = 5

    def produce():
        for _ in range(per_thread):
            player.play(constant_clip(480), SR)

    threads = [threading.Thread(target=produce, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    player.wait(timeout=2.0)
    total = sum(stream.samples for stream in fake_sd.output_streams)
    assert total == 480 * per_thread * 2
