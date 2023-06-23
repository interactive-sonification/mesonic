from enum import Enum, unique
from typing import TYPE_CHECKING, Callable, Dict

from attrs import define, field, validators

if TYPE_CHECKING:
    from mesonic.record import Record
    from mesonic.synth import Synth


@define(kw_only=True)
class Event:
    """Event base class."""

    track: int = 0
    """The Timeline track of the Event."""
    info: Dict = field(factory=dict, validator=validators.instance_of(dict))
    """Additional information about this Event."""

    def reverse(self) -> "Event":
        """Reverse the Event.

        This is useful when executing the Events backwards or reversing/undoing them.

        Returns
        -------
        Event
            reversed Event
        """
        return self


@unique
class SynthEventType(Enum):
    """SynthEventTypes describes what different SynthEvents are possible."""

    START = 1
    STOP = -1
    PAUSE = 2
    RESUME = -2
    SET = 0

    def reverse(self) -> "SynthEventType":
        """Reverse the SynthEventType

        START <-> STOP
        PAUSE <-> RESUME
        SET <-> SET

        Returns
        -------
        SynthEventType
            the reversed SynthEventType
        """
        return SynthEventType(-self.value)


@define(kw_only=True)
class SynthEvent(Event):
    """Events created by a Synth instance."""

    synth: "Synth" = field(repr=False)
    """The respective Synth instance."""
    etype: SynthEventType = field(validator=validators.in_(SynthEventType))
    """The type of SynthEvent."""
    data: Dict = field(factory=dict)
    """Information about the SynthEvent."""

    def __attrs_post_init__(self):
        # will be executed by attrs after init
        # and is used for validating the whole instance
        # https://www.attrs.org/en/stable/init.html#hooking-yourself-into-initialization
        if self.etype is SynthEventType.SET:
            for must_have in ["name", "old_value", "new_value"]:
                if must_have not in self.data:
                    raise ValueError(f"data is missing key '{must_have}'")

    def reverse(self) -> "SynthEvent":
        """Reverse the SynthEvent by adjusting the etype and data.

        Returns
        -------
        SynthEvent
            the reversed SynthEvent
        """
        data = self.data.copy()
        if self.etype is SynthEventType.SET:
            data.update(
                new_value=self.data["old_value"],
                old_value=self.data["new_value"],
            )
        return SynthEvent(
            synth=self.synth,
            etype=self.etype.reverse(),
            data=data,
        )


@unique
class RecordEventType(Enum):
    """SynthEventTypes describes what different SynthEvents are possible."""

    START = 1
    STOP = -1
    PAUSE = 2
    RESUME = -2

    def reverse(self) -> "RecordEventType":
        """Reverse the RecordEventType

        START <-> STOP
        PAUSE <-> RESUME

        Returns
        -------
        RecordEventType
            the reversed RecordEventType
        """
        return RecordEventType(-self.value)


@define(kw_only=True)
class RecordEvent(Event):
    """Event created by a Record instance."""

    record: "Record"
    """The respective Record instance."""
    etype: RecordEventType
    """The type of RecordEvent."""


@define(kw_only=True)
class GenericEvent(Event):
    """A Generic Event that has two callables."""

    action: Callable
    """The action to be executed when not reversed"""
    reverse_action: Callable
    """The action to be executed when reversed"""
