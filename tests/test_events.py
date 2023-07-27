from unittest.mock import MagicMock

import pytest

from mesonic.events import SynthEvent, SynthEventType

__author__ = "Dennis Reinsch"
__copyright__ = "Dennis Reinsch"
__license__ = "MIT"


def test_SynthEventType():
    assert SynthEventType.START.reverse() is SynthEventType.STOP
    assert SynthEventType.STOP.reverse() is SynthEventType.START

    assert SynthEventType.PAUSE.reverse() is SynthEventType.RESUME
    assert SynthEventType.RESUME.reverse() is SynthEventType.PAUSE

    assert SynthEventType.SET.reverse() is SynthEventType.SET


class TestSynthEvent:
    @pytest.mark.filterwarnings(
        "ignore:using non-Enums in containment checks will raise TypeError in Python 3.8"
    )
    def test_init(self):
        # test too few arguments
        with pytest.raises(TypeError):
            SynthEvent()
        with pytest.raises(TypeError):
            SynthEvent(synth=MagicMock())
        with pytest.raises(TypeError):
            SynthEvent(etype=SynthEventType.START)

        # test missing keywords
        with pytest.raises(TypeError):
            SynthEvent(3, SynthEventType.START)

        # test wront arguments
        with pytest.raises(ValueError):
            SynthEvent(synth=MagicMock(), etype=5)

        SynthEvent(synth=MagicMock(), etype=SynthEventType.START)

        start_args = {"freq": 300, "amp": 0.3}
        SynthEvent(synth=MagicMock(), etype=SynthEventType.START, data=start_args)

        with pytest.raises(ValueError):
            SynthEvent(synth=MagicMock(), etype=SynthEventType.SET, data=start_args)

        set_args = {"name": "freq", "old_value": 300, "new_value": 400}
        SynthEvent(synth=MagicMock(), etype=SynthEventType.SET, data=set_args)

    def test_reverse(self):
        # set event
        set_args = {"name": "freq", "old_value": 300, "new_value": 400}
        event = SynthEvent(synth=MagicMock(), etype=SynthEventType.SET, data=set_args)
        assert event.etype is SynthEventType.SET
        reversed_event = event.reverse()
        assert event.data["name"] is reversed_event.data["name"]
        assert event.data["old_value"] is reversed_event.data["new_value"]
        assert event.data["new_value"] is reversed_event.data["old_value"]

        for event_type in SynthEventType:
            if event_type is SynthEventType.SET:
                event = SynthEvent(synth=MagicMock(), etype=event_type, data=set_args)
            else:
                event = SynthEvent(synth=MagicMock(), etype=event_type)
            assert event.etype.reverse() is event.reverse().etype
