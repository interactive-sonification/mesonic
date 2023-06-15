import gc
import weakref

import pytest

from mesonic.events import Event
from mesonic.timeline import Timeline, TimelineEnumerator

__author__ = "Dennis Reinsch"
__copyright__ = "Dennis Reinsch"
__license__ = "MIT"


def test_timeline():
    """API Tests"""
    timeline = Timeline()
    event00 = Event(track=100, info={"val": 0})
    event01 = Event(track=101)
    event10 = Event(track=110, info={"val": 0})
    event11 = Event(track=111)
    event20 = Event(track=120, info={"val": 0})
    event21 = Event(track=121)
    event22 = Event(track=122)
    event30 = Event(track=130, info={"val": 0})
    assert timeline.is_empty()
    with pytest.raises(ValueError, match="Empty Timeline"):
        timeline.last_timestamp

    with pytest.raises(ValueError, match="Empty Timeline"):
        timeline.after(0.0)

    with pytest.raises(ValueError, match="Empty Timeline"):
        timeline.before(0.0)

    timeline.insert(1, [event10])
    time = 2
    with pytest.raises(ValueError, match=f"Timeline ends before time: {time}"):
        timeline.after(time)
    time = 0
    with pytest.raises(ValueError, match=f"Timeline ends before time: {time}"):
        timeline.before(time)
    timeline.insert(0, [event00])
    timeline.insert(0, [event01])
    timeline.insert(2, [event20, event21])
    timeline.insert(2, [event22])
    assert [event00, event01] == timeline.before(0).events
    assert [event10] == timeline.before(1).events
    assert [event10] == timeline.after(1).events
    assert [event20, event21, event22] == timeline.after(1.01).events

    timeline.insert(1, [event11])
    assert [event10, event11] == timeline.after(1).events

    for idx, event in enumerate(timeline.after(1)):
        assert event == [event10, event11][idx]

    timeline.insert(1.5, [event30])
    assert [event30] == timeline.after(1.2).events

    tl_dict = timeline.to_dict()
    assert tl_dict == {
        0: [event00, event01],
        1: [event10, event11],
        1.5: [event30],
        2: [event20, event21, event22],
    }

    def test_enumerator():
        for bundle in TimelineEnumerator(timeline):
            assert bundle.events == tl_dict[bundle.timestamp]

    test_enumerator()

    assert timeline.last_timestamp == 2

    assert timeline.tail.next is None
    assert timeline.head.prev is None

    timeline.filter(lambda event: event.info.get("val", None) == 0)
    tl_dict = timeline.to_dict()
    assert tl_dict == {
        0: [event01],
        1: [event11],
        2: [event21, event22],
    }

    weakrefs = [
        weakref.ref(bundle) for bundle in timeline
    ]  # make weakrefs of the bundles
    timeline.reset()
    assert timeline.is_empty()
    gc.collect()  # collect the bundles after the reset
    for ref in weakrefs:
        assert ref() is None  # weakref should be None after gc.collect()
