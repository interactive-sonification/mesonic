import logging
import math
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from mesonic.events import Event
from mesonic.playback import Clock, Playback
from mesonic.timeline import Timeline

__author__ = "Dennis Reinsch"
__copyright__ = "Dennis Reinsch"
__license__ = "MIT"


@pytest.fixture
def timeline() -> Timeline:
    """Create a Timeline with some Events"""
    tl = Timeline()
    tl.insert(0, [Event(track=100), Event(track=101)])
    tl.insert(1, [Event(track=110), Event(track=111)])
    tl.insert(2, [Event(track=120), Event(track=121), Event(track=122)])
    tl.insert(3, [Event(track=130)])
    return tl


class MockClock(Clock):
    def __init__(self, time) -> None:
        self._time = time
        self.sleep_time = 0.2

    def time(self):
        return self._time


@pytest.fixture
def processor():
    processor = MagicMock()
    processor.latency = 0.2
    return processor


@pytest.fixture
def playback_and_mockclock(
    timeline: Timeline, processor
) -> Tuple[Playback, "MockClock", MagicMock, Timeline]:
    """Create a Playback of a Timeline with a mocked Clock and processor"""
    # mock time using the MockClock and
    # then _advance by hand and check if state is correct
    mock_clock = MockClock(0)

    pb = Playback(timeline=timeline, processor=processor, clock=mock_clock)
    # mock started thread
    pb._init_playback(at=pb.start_time, rate=1)
    pb._thread = MagicMock()
    pb._thread.is_alive = lambda: True
    assert pb.running is True
    return pb, mock_clock, processor, timeline


def test_playback_init(processor):
    """test the the initial state of the Playback"""
    tl = Timeline()
    pb = Playback(timeline=tl, processor=processor)
    assert pb.rate == 1, "Default rate should be 1"
    assert tl.is_empty() and pb._next_bundle is None

    # fill the timeline with two elements : 1 - 2
    tl.insert(1, [Event(track=1)])
    tl.insert(2, [Event(track=2)])

    # (0>) 1 - 2 - init before the first element
    pb = Playback(timeline=tl, processor=processor, start=tl.head.timestamp - 1)
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end

    # 1 - 2 (<3) - init reversed before the last element
    pb = Playback(
        timeline=tl,
        processor=processor,
        rate=-1,
        start=tl.tail.timestamp + 1,
    )
    assert pb._next_bundle is tl.tail
    assert not pb._over_the_end

    # 1 - (2>)2 - init at the last element
    pb = Playback(timeline=tl, processor=processor, start=tl.tail.timestamp)
    assert pb._next_bundle is tl.tail
    assert not pb._over_the_end

    # (1>)1 - 2 - init at the first element
    pb = Playback(timeline=tl, processor=processor, start=tl.head.timestamp)
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end

    # try to init before the first and last element
    # 1 - 2(2.1>)
    pb = Playback(timeline=tl, processor=processor, start=2.1)
    assert pb._next_bundle is tl.tail
    assert pb._over_the_end

    # (<0)1 - 2
    pb = Playback(timeline=tl, processor=processor, rate=-1, start=0)
    assert pb._next_bundle is tl.head
    assert pb._over_the_end

    # init between the events
    # 1 -(1.5>)- 2
    pb = Playback(timeline=tl, processor=processor, start=1.5)
    assert pb._next_bundle is tl.tail
    assert not pb._over_the_end
    # 1 -(<1.5)- 2
    pb = Playback(timeline=tl, processor=processor, rate=-1, start=1.5)
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end


def test_playback_timeouts(processor):
    """test if the controls (time, rate) raise Timeouts"""
    # This should happen when the worker thread fails to adjust the values in time
    tl = Timeline()
    pb = Playback(timeline=tl, processor=processor)
    pb.start()
    # kill worker thread manually to make sure it fails to adjust the values
    pb._stop_event.set()
    pb._thread.join()
    # now mock the worker thread to enforce pb.running == True
    pb._thread = MagicMock()
    pb._stop_event.clear()
    pb._thread.is_alive = lambda: True
    assert pb.running is True
    # change values to make them "dirty" the changing thread should raise TimeoutError
    # as the worker thread does not adapt the time as it is killed and mocked
    # change time to make it "dirty"
    pb.time = 3
    # try to access / change it
    with pytest.raises(TimeoutError):
        pb.time
    with pytest.raises(TimeoutError):
        pb.time = 4
    # change rate to make it "dirty"
    pb.rate = 3
    # try to access / change it
    with pytest.raises(TimeoutError):
        pb.reversed
    with pytest.raises(TimeoutError):
        pb.reverse()


def test_playback_time_control(processor):
    """Test setting the time using the time property"""
    tl = Timeline()
    pb = Playback(timeline=tl, processor=processor)
    with pytest.raises(RuntimeError):
        pb.time
    with pytest.raises(RuntimeError):
        pb.time = 3

    pb.start()
    assert pb.time >= 0

    for y in [3, 4, 0]:
        pb.time = y
        assert pb.time >= y

    pb.reverse()
    assert pb.reversed

    for y in [0, -1, 100]:
        pb.time = y
        pb.time <= y

    pb.stop()


def test_playback_start_and_end_time(timeline, processor):
    """Test getting/setting the start_time and end_time values"""
    pb = Playback(timeline=timeline, processor=processor)

    # test the default values
    assert pb.start_time == 0
    assert pb.end_time == timeline.end_time

    # test setting/getting start_time
    for start in [-3, 0, 6]:
        pb.start_time = start
        assert pb.start_time == start

    # test setting/getting end_time
    for end in [1, 2, 6]:
        pb.end_time = end
        assert pb.end_time == end

    # should be unable to get/set end_time_offset if end_time is not None
    assert pb.end_time is not None
    with pytest.raises(RuntimeError):
        pb.end_time_offset
    with pytest.raises(RuntimeError):
        pb.end_time_offset = 2

    # reset end_time to automatic and test the automatic default value
    pb.end_time = None
    timeline_end_time = timeline.end_time
    assert pb.end_time is timeline_end_time

    # test if end_time adjusts when timeline get longer
    timeline.insert(timeline.end_time + 1, [Event(track=999)])
    assert pb.end_time is timeline_end_time + 1 + timeline.end_time_offset

    # test setting end_time_offset to different values
    for offset in [-1, 0, 2, 6]:
        pb.end_time_offset = offset
        # should replace the timeline offset with the provided offset
        assert pb.end_time_offset == offset
        assert pb.end_time == timeline.end_time - timeline.end_time_offset + offset


def test_playback_rate_control(processor):
    """Test setting/getting the rate property"""
    tl = Timeline()
    pb = Playback(timeline=tl, processor=processor)
    # test rate init value
    assert pb.rate == 1
    # test setting/getting rate value
    for x in range(-10, 10):
        pb.rate = x
        assert pb.rate == x


def test_playback_rate_time_effect(playback_and_mockclock):
    """Test if the time changes like it is expected from the rate"""
    pb, mock_clock, _, _ = playback_and_mockclock
    playback_time = 0
    old_rate = pb.rate
    assert pb.time == playback_time

    def check_time_for_rates(test_values, pb, mock_clock, playback_time, old_rate):
        for delta, new_rate in test_values:
            print(
                f"with rate={old_rate} advance {delta} ({old_rate*delta}), new_rate={new_rate}"
            )
            pb.rate = new_rate
            mock_clock._time += delta
            pb._advance()
            print(pb._run_time)
            print(pb._time_offset)
            print(
                f"- advanced {playback_time} -> {pb.time} = {pb.time - playback_time}",
                end="",
            )
            playback_time += old_rate * delta
            print(f" expecting: {playback_time}")
            old_rate = pb.rate
            assert math.isclose(pb.time, playback_time)
        return playback_time, old_rate

    # playback_time, old_rate = rate_tests(pb, mock_clock, playback_time, old_rate, 1)
    test_values = [
        (10, 3),
        (2, 2),
        (1, 1),
        (2, 0.5),
        (4, 0.25),
        (3, 3),
        (0, 1),
    ]
    playback_time, old_rate = check_time_for_rates(
        test_values, pb, mock_clock, playback_time, old_rate
    )

    time_before_reverse = pb.time
    pb.reverse()
    pb._advance()
    print(f"\nreverse tests: time={pb.time}, reversed={pb.reversed}, rate={pb.rate}\n")
    assert pb.time == time_before_reverse
    assert pb.reversed
    assert pb.rate == old_rate * -1
    old_rate = pb.rate

    test_values = [
        (1, -3),
        (2, -2),
        (1, -1),
        (2, -0.5),
        (4, -0.25),
        (3, -3),
        (0.3, -100),
    ]
    playback_time, old_rate = check_time_for_rates(
        test_values, pb, mock_clock, playback_time, old_rate
    )


def test_playback_startstop_control(processor):
    """Test if start and stop have the desired effect"""
    pb = Playback(timeline=Timeline(), processor=processor)

    assert pb.running is False
    with pytest.raises(RuntimeError):
        pb.stop()

    # start normally
    pb.start()
    assert 0 <= pb.time <= 0.2
    assert pb.reversed is False

    # jump while running
    pb.start(100)
    # logging.debug("check")
    assert pb.time >= 100

    pb.stop()
    assert not pb.running

    # start normally after jump
    pb.start()
    assert pb.reversed is False
    assert 0 <= pb.time <= 0.2

    # reverse and jump while running
    pb.start(20, rate=-1)
    assert pb.reversed is True
    assert pb.time <= 20

    pb.stop()
    assert not pb.running

    # start reversed
    pb.start(rate=-1)
    assert pb.reversed is True
    assert pb.time <= 0

    # jump reversed
    pb.start(at=10, rate=-1)
    assert pb.reversed is True
    assert pb.time <= 10

    # reverse and jump while running
    pb.start(20, rate=1)
    assert pb.reversed is False
    assert pb.time >= 20

    # reverse and jump while running
    pb.start(20, rate=-1)
    assert pb.reversed is True
    assert pb.time <= 20

    pb.stop()
    assert pb.running is False


def test_playback_restart(processor):
    """Test if restarting has the desired effect"""
    pb = Playback(timeline=Timeline(), processor=processor)
    pb.start()
    assert pb.running is True
    pb.restart()
    assert pb.running is True
    pb.stop()
    assert pb.running is False
    pb.restart()
    assert pb.running is True

    pb.stop()
    assert pb.running is False


def test_playback_pause_resume(playback_and_mockclock, processor):
    """Test if pause and resume have the desired effect"""
    pb = Playback(timeline=Timeline(), processor=processor)

    # not running
    with pytest.raises(RuntimeError):
        pb.pause()

    pb.start()
    # already running
    with pytest.raises(RuntimeError):
        pb.resume()

    pb.stop()

    # Test the rest with the mocked playback
    pb, mock_clock, _, _ = playback_and_mockclock

    # set pause time
    pause_time = 5
    mock_clock._time = pause_time
    pb._advance()
    assert pb.time == pause_time

    # pause
    pb.pause()
    pb._advance()
    assert pb.paused
    assert pb.time == pause_time

    # change clock and see if paused playback stays at pause time
    mock_clock._time += 1
    pb._advance()
    assert pb.time == pause_time

    # resume clock and see if the playback continues at the right time
    pb.resume()
    pb._advance()
    assert not pb.paused
    assert pb.time == pause_time

    # advance and see if the playback time is not freezed at the pause time
    mock_clock._time += 1
    pb._advance()
    assert pb.time == pause_time + 1


def test_playback_integrated(
    playback_and_mockclock: Tuple[Playback, MockClock, MagicMock, Timeline]
):
    """Test by mocking the worker thread and checking various playback features"""
    pb, mock_clock, processor, tl = playback_and_mockclock

    expected_processor_calls = 0

    def processor_called(times=1):
        nonlocal expected_processor_calls
        expected_processor_calls += times
        return expected_processor_calls == len(processor.mock_calls)

    # advance execute from head to tail
    current_bundle = tl.head
    while current_bundle is not None:
        assert pb._next_bundle is current_bundle
        mock_clock._time = current_bundle.timestamp
        pb._advance()
        assert processor_called()
        assert processor.mock_calls[-1][1][0] is current_bundle
        current_bundle = current_bundle.next

    # advance once more, should not have moved
    num_calls = len(processor.mock_calls)
    assert pb._next_bundle is tl.tail
    assert pb.time == tl.tail.timestamp
    pb._advance()
    assert pb._next_bundle is tl.tail
    assert pb.time == tl.tail.timestamp
    assert pb._over_the_end
    assert num_calls == len(processor.mock_calls)

    # time goes after the last bundle
    delta = 0.5
    mock_clock._time = tl.tail.timestamp + delta
    pb._advance()
    assert pb._next_bundle is tl.tail
    assert pb._over_the_end
    assert num_calls == len(processor.mock_calls)
    assert pb.time == tl.tail.timestamp + delta

    # we reverse after this time: tail should be next and "unexecuted"
    pb.reverse()
    pb._advance()
    assert pb.reversed is True
    assert pb._over_the_end is False
    assert num_calls == len(processor.mock_calls)
    assert pb._next_bundle is tl.tail
    assert pb.time == tl.tail.timestamp + delta

    # move now in reverse
    mock_clock._time += 2 * delta
    pb._advance()

    assert pb.time == tl.tail.timestamp - delta
    assert processor_called()
    for i in [-1, -2]:
        assert processor.mock_calls[i][1][0] is tl.tail
    assert pb._next_bundle is tl.tail.prev

    # test loop
    # enable loop with 2s duration
    loop_time = 2
    pb.loop_duration = loop_time
    pb.loop = True

    logging.debug("reverse")
    assert pb.reversed is True
    pb.reverse()
    pb._advance()
    assert pb.reversed is False
    assert pb.loop_start == pb.start_time
    assert pb.loop_end == pb.end_time

    # assert we are before tail
    assert pb._next_bundle is tl.tail
    assert pb.time == tl.tail.timestamp - delta

    logging.debug("move to tail")
    mock_clock._time += delta
    pb._advance()

    # assert tail was executed
    assert processor_called()
    for i in [-1, -2, -3]:
        assert processor.mock_calls[i][1][0] is tl.tail
    logging.debug("pb.time -> %s", pb.time)
    logging.debug(processor.mock_calls)
    assert pb.time == tl.tail.timestamp
    assert pb.loop_end == tl.tail.timestamp + 1
    # assert next was set according to looped from tail to head
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end

    logging.debug("move past tail and check if time is right")
    mock_clock._time += delta
    pb._advance()
    assert pb._check_for_loop
    logging.debug("pb.time -> %s", pb.time)
    assert pb.time == tl.tail.timestamp + delta
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end

    logging.debug("jump before tail and check if loop was canceled")
    pb.time = tl.tail.timestamp - delta
    pb._advance()
    assert pb._next_bundle is tl.tail
    assert not pb._check_for_loop

    logging.debug("jump after tail and check if loop was enabled")
    pb.time = tl.tail.timestamp + delta
    pb._advance()
    assert pb._check_for_loop
    assert pb.time == tl.tail.timestamp + delta
    assert pb._next_bundle is tl.head

    logging.debug("move to loop / playback end, check if loop time is correct")
    mock_clock._time += delta
    pb._advance()
    assert not pb._check_for_loop
    assert pb.time == pb.start_time - loop_time
    assert pb._next_bundle is tl.head

    logging.debug("move towards head")
    mock_clock._time += delta
    pb._advance()
    assert pb.time == pb.start_time - loop_time + delta
    assert pb._next_bundle is tl.head
    assert expected_processor_calls == len(processor.mock_calls)

    logging.debug("%s: move to head", pb.time)
    mock_clock._time += 3 * delta
    pb._advance()
    assert pb.time == tl.head.timestamp
    assert processor_called()
    assert pb._next_bundle is tl.head.next

    logging.debug("%s: move (%s) past head", pb.time, delta)
    mock_clock._time += delta
    pb._advance()
    assert pb.time == tl.head.timestamp + delta
    assert expected_processor_calls == len(processor.mock_calls)
    assert pb._next_bundle is tl.head.next

    logging.debug("%s: reverse", pb.time)
    pb.reverse()
    pb._advance()
    assert pb.reversed is True
    assert pb.loop_end == pb.start_time
    assert pb.loop_start == pb.end_time
    assert pb._next_bundle is tl.head
    assert not pb._over_the_end

    logging.debug("%s: move (%s) reversed", pb.time, delta)
    mock_clock._time += delta
    pb._advance()
    assert pb.time == tl.head.timestamp
    assert processor_called()
    assert pb._next_bundle is tl.tail

    logging.debug("%s: move (%s) past head", pb.time, delta)
    mock_clock._time += delta
    pb._advance()
    assert expected_processor_calls == len(processor.mock_calls)
    assert pb._next_bundle is tl.tail
    assert pb.time == pb.loop_start + pb.loop_duration + pb._rate_sign * -1 * delta
    assert pb.end_time == tl.tail.timestamp + 1  # default end_time
    assert pb.time == pb.end_time + pb.loop_duration + pb._rate_sign * -1 * delta

    # disable looping
    pb.loop = False

    logging.debug(
        "test time setting (_search_next_bundle_at) with reversed=%s", pb.reversed
    )
    logging.debug("set 1.5")
    pb.time = 1.5
    pb._advance()
    assert expected_processor_calls == len(processor.mock_calls)
    assert pb.reversed is True
    assert pb._next_bundle is tl.head.next

    logging.debug("set -1.5")
    pb.time = -1.5
    pb._advance()
    assert expected_processor_calls == len(processor.mock_calls)
    assert pb._next_bundle is tl.head
    assert pb._over_the_end

    # reversed == False
    logging.debug("reverse and set 1.5")
    pb.reverse()
    pb.time = 1.5
    pb._advance()
    assert expected_processor_calls == len(processor.mock_calls)
    assert pb.reversed is False
    assert pb._next_bundle is tl.tail.prev
    assert not pb._over_the_end

    # combine time setting with reversing (start)
    # and use a direct bundle time

    logging.debug("set time of tail.prev with reversed = False")
    pb.start(tl.tail.prev.timestamp, rate=1)
    pb._advance()
    assert pb.reversed is False
    assert pb._next_bundle is tl.tail.prev.next
    assert processor_called()
    # assert num_calls + 1 == len(processor.mock_calls)
    assert processor.mock_calls[-1][1][0] is tl.tail.prev

    logging.debug("set time of tail.prev with reversed = False")
    pb.start(tl.tail.prev.timestamp, rate=-1)
    pb._advance()
    assert pb.reversed is True
    assert pb._next_bundle is tl.tail.prev.prev
    assert not pb._over_the_end
    assert processor_called()
    assert processor.mock_calls[-1][1][0] is tl.tail.prev

    logging.debug("jump to the tail of the timeline")
    pb.start(tl.tail.timestamp, rate=1)
    assert not pb._over_the_end
    pb._advance()
    prev_tail = tl.tail
    assert pb._next_bundle is prev_tail
    assert pb._over_the_end
    assert processor_called()

    logging.debug("test if pointer stays")
    pb._advance()
    assert not processor_called()
    assert pb._next_bundle is prev_tail
    assert pb._over_the_end

    logging.debug("test adding new bundles to timeline")
    tl.insert(4, [Event(track=400)])
    pb._advance()
    assert not processor_called()
    assert pb._next_bundle is not prev_tail
    assert pb._next_bundle is prev_tail.next
    assert pb._next_bundle is tl.tail
    assert not pb._over_the_end

    logging.debug("test timeline reset")
    tl.reset()
    pb._advance()
    assert pb._next_bundle is None
    assert pb.running
    pb._advance()
    assert pb._next_bundle is None

    logging.debug("test timeline insert after reset")
    tl.insert(0, [Event(track=100)])
    pb._advance()
    assert pb._next_bundle is tl.head
