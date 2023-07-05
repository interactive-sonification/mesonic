import copy
import math
from threading import RLock
from time import time as get_timestamp
from typing import Callable, Dict, List, Optional, Tuple

import attr

from mesonic.events import Event, SynthEvent, SynthEventType


@attr.s(slots=True, repr=True)
class TimeBundle:
    """A Bundle of Events with a timestamp."""

    timestamp: float = attr.ib()
    prev: "TimeBundle" = attr.ib(order=False, repr=False)
    next: "TimeBundle" = attr.ib(order=False, repr=False)
    events: List[Event] = attr.ib(order=False, factory=list)

    def __iter__(self):
        yield from self.events


class Timeline:
    """A sorted double linked list with TimeBundles."""

    def __init__(self) -> None:
        self.head = self.tail = None
        self.lock = RLock()

        self._end_time = None
        self._end_time_offset = 1

        self.last_update_time = 0.0

    def is_empty(self) -> bool:
        """Test if this Timeline is empty.

        Returns
        -------
        bool
            True if empty.

        """
        with self.lock:
            return self.head is None and self.tail is None

    def insert(self, time: float, events: List[Event]):
        """Insert the Events into the Timeline.

        Parameters
        ----------
        time : float
            Time where the Events should be inserted.
        events : List[Event]
            Events to be inserted.

        Raises
        ------
        ValueError
            If the time is not finite.
        """
        self.last_update_time = get_timestamp()
        if not math.isfinite(time):
            raise ValueError(f"time must be finite. Got time={time}")
        with self.lock:
            if self.is_empty():  # empty => insert as head & tail
                self.head = self.tail = TimeBundle(time, None, None, events)
            elif time <= self.head.timestamp:  # insert at the beginning
                if time < self.head.timestamp:  # before head
                    self.head = TimeBundle(time, None, self.head, events)
                    self.head.next.prev = self.head
                else:  # add to head
                    self.head.events.extend(events)
            elif time >= self.tail.timestamp:  # insert at the end
                if time > self.tail.timestamp:  # after tail
                    self.tail = TimeBundle(time, self.tail, None, events)
                    self.tail.prev.next = self.tail
                else:  # add to tail
                    self.tail.events.extend(events)
            else:  # insert in between head.timestamp < time < tail.timestamp
                # search for a TimeBundle where the time is <= insertion time
                cur = self.head
                while cur.timestamp < time:
                    cur = cur.next
                # now we have cur.timestamp >= insertion time
                if cur.timestamp == time:  # add to bundle
                    cur.events.extend(events)
                else:  # cur.timestamp  < time => insert before cur
                    new = TimeBundle(time, cur.prev, cur, events)
                    cur.prev.next = new
                    cur.prev = new

    def reset(self):
        """Reset this Timeline by removing all Events."""
        with self.lock:
            self.head = self.tail = None

    def filter(self, predicate: Callable[[Event], bool]):
        """Filter Events that don't return true if predicate is applied.

        If a TimeBundle has no events after filtering it will be removed.

        Parameters
        ----------
        predicate : Callable[[Event], bool]
            Function used to filter the events.
        """
        with self.lock:
            cur = self.head
            while cur is not None:
                cur.events = [event for event in cur.events if not predicate(event)]
                if not cur.events:  # events are now empty
                    self._remove_timebundle(cur)
                cur = cur.next

    def _remove_timebundle(self, bundle: TimeBundle):
        with self.lock:
            if bundle is self.head:
                self.head = bundle.next
            if bundle is self.tail:
                self.tail = bundle.prev
            if bundle.next is not None:
                bundle.next.prev = bundle.prev
            if bundle.prev is not None:
                bundle.prev.next = bundle.next

    def before(self, time: float) -> TimeBundle:
        """Get the TimeBundle before the provided time.

        Parameters
        ----------
        time : float
            Time that must be after the returned TimeBundle timestamp.

        Returns
        -------
        TimeBundle
            The TimeBundle before the provided time.

        Raises
        ------
        ValueError
            If the Timeline is empty or the Timeline ends before the time.
        """
        with self.lock:
            if self.is_empty():
                raise ValueError("Empty Timeline")
            cur = self.tail
            while cur.timestamp > time:
                if cur.prev is None:
                    raise ValueError(f"Timeline ends before time: {time}")
                cur = cur.prev
            return cur

    def after(self, time: float) -> TimeBundle:
        """Get the TimeBundle after the provided time.

        Parameters
        ----------
        time : float
            Time that must be before the returned TimeBundle timestamp.

        Returns
        -------
        TimeBundle
            The TimeBundle after the provided time.

        Raises
        ------
        ValueError
            If the Timeline is empty or the Timeline ends before the time.
        """
        with self.lock:
            if self.is_empty():
                raise ValueError("Empty Timeline")
            cur = self.head
            while cur.timestamp < time:
                if cur.next is None:
                    raise ValueError(f"Timeline ends before time: {time}")
                cur = cur.next
            # cur.timestamp >= time
            return cur

    def search_at(
        self, at: float, direction_reversed: bool
    ) -> Tuple[Optional[TimeBundle], bool]:
        """Search a next TimeBundle at the provided time.

        If there is no TimeBundle after/before the provided time it will
        fallback to the tail/head of the Timeline if available.

        Parameters
        ----------
        at : float
            At which timepoint the search starts.
        direction_reversed : bool
            True to search forward, False for backwards.

        Returns
        -------
        Tuple[Optional[TimeBundle], bool]
            The TimeBundle or None if the Timeline is empty
            The bool
        """
        # Here it would also be possible to look for skipped messages
        # like a stop for a started continuous synths.
        # However we regard keeping the Timeline correct the task of the user.
        # Future updates could introduce some housekeeping.
        next_bundle = None
        over_the_end = False
        with self.lock:
            if not self.is_empty():
                try:
                    if direction_reversed:
                        next_bundle = self.before(at)
                    else:
                        next_bundle = self.after(at)
                except ValueError:
                    # if timeline ends before at we are over_the_end
                    over_the_end = True
                    if direction_reversed:
                        next_bundle = self.head
                    else:
                        next_bundle = self.tail
            return next_bundle, over_the_end

    def to_dict(self, timeshift: float = 0) -> Dict[float, List[Event]]:
        """Transform this Timeline into a dict.

        Parameters
        ----------
        timeshift : float, optional
            time value that will be added to all times, by default 0

        The timestamps are the keys and the values the lists of events.

        Returns
        -------
        Dict[float, List[Event]]
            This Timeline as dict.
        """
        ret = {}
        with self.lock:
            for bundle in iter(self):
                ret[bundle.timestamp + timeshift] = copy.copy(bundle.events)
        return ret

    def extend(
        self,
        timeline: Dict[float, List[Event]],
        timeshift: float = 0,
        fun: Optional[Callable[[Event], Optional[Event]]] = None,
    ):
        """Extend this Timeline by a dict containing times and Events

        Parameters
        ----------
        timeline : Dict[float, List[Event]]
            dict with events and times.
        timeshift : float, optional
            time value that will be added to all times, by default 0
        fun : Optional[Callable[[Event], Optional[Event]]]
            An optional function that takes an Event and
            returns a transformed Event or None to discard the Event.
            This will be applied to all the Events that are added.

        """
        for time in sorted(timeline):
            events = copy.copy(timeline[time])
            if fun:
                events = list(filter(None, map(fun, events)))
            self.insert(time + timeshift, events)

    @property
    def first_timestamp(self) -> float:
        """float: first timestamp in this Timeline"""
        with self.lock:
            if self.is_empty():
                raise ValueError("Empty Timeline")
            return self.head.timestamp

    @property
    def last_timestamp(self) -> float:
        """float: last timestamp in this Timeline"""
        with self.lock:
            if self.is_empty():
                raise ValueError("Empty Timeline")
            return self.tail.timestamp

    @property
    def end_time(self) -> float:
        """float: end_time of this Timeline

        If not explicitly set this is
        end_time = last_timestamp + end_time_offset
        """
        if self._end_time is not None:
            return self._end_time
        with self.lock:
            if self.is_empty():
                return self._end_time_offset
            return self.last_timestamp + self._end_time_offset

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def end_time_offset(self) -> Optional[float]:
        """Optional[float]: The offset after the last timestamp

        The end time offset effects the end_time if set to a not None value like this:
        end_time = timeline.last_timestamp + end_time_offset

        Raises
        ------
        RuntimeError
            If end_time is set manually
        """
        if self._end_time is not None:
            raise RuntimeError(
                "end_time is set manually, end_time_offset has no effect."
            )
        return self._end_time_offset

    @end_time_offset.setter
    def end_time_offset(self, value):
        if self._end_time is not None:
            raise RuntimeError(
                "end_time is set manually, end_time_offset has no effect."
            )
        if value < 0:
            raise ValueError("end_time_offset must larger or equal 0")
        self._end_time_offset = value

    def plot(self, duration_key="dur", default_duration=0.1):
        """Plot the Timeline.

        Parameters
        ----------
        duration_key : str, optional
            The Parameter name for the Synth duration, by default "dur"
            This is used to extract the duration from the Events.
        default_duration : float, optional
            The fallback value if no duration can be found, by default 0.1

        Raises
        ------
        ImportError
            If matplotlib cannot be imported.
        """
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "plotting the Timeline is only possible when matplotlib is installed."
            ) from err
        # do not plot an empty Timeline
        if self.is_empty():
            return

        fig = plt.figure()
        cmap = plt.get_cmap("Set1")
        axes = fig.add_subplot(1, 1, 1)

        spacing = 1
        y_offset = spacing

        offsets = {}
        colors = {}
        mutable_synths = {}
        immutable_synths = {
            "X_start": [],
            "Y_offset": [],
            "U_duration": [],
            "color": [],
        }
        # store SynthEvents of type SET
        set_events = {"times": [], "offsets": [], "colors": []}
        for time, events in self.to_dict().items():
            for event in events:
                if isinstance(event, SynthEvent):
                    if event.synth not in offsets:
                        offsets[event.synth] = y_offset
                        y_offset += spacing
                    offset = offsets[event.synth]

                    if event.track not in colors:
                        colors[event.track] = cmap(event.track)
                    color = colors[event.track]
                    if not event.synth.mutable:  # immutable synth
                        # event.etype == START if synth is immutable
                        immutable_synths["X_start"].append(time)
                        immutable_synths["Y_offset"].append(offset)
                        immutable_synths["U_duration"].append(
                            event.data.get(duration_key, default_duration)
                        )
                        immutable_synths["color"].append(color)
                    else:  # mutable Synth
                        synth_dict = mutable_synths.get(
                            event.synth, {"start_times": [], "stop_times": []}
                        )
                        if event.etype == SynthEventType.START:
                            dur = event.data.get(duration_key, None)
                            if dur:
                                synth_dict["stop_times"].append(time + dur)
                            synth_dict["start_times"].append(time)
                        elif event.etype == SynthEventType.STOP:
                            synth_dict["stop_times"].append(time)
                        elif event.etype == SynthEventType.SET:
                            set_events["times"].append(time)
                            set_events["offsets"].append(offset)
                            set_events["colors"].append(color)
                        mutable_synths[event.synth] = synth_dict
        max_mutable_time = 0
        for synth, times in mutable_synths.items():
            start_times = times["start_times"]
            stop_times = times["stop_times"]
            durations = [e - s for s, e in zip(start_times, stop_times)]
            if len(durations) < len(start_times):
                assumed_dur = default_duration
                if synth.metadata.get("from_buffer", None):
                    assumed_dur = synth.metadata["from_buffer"].duration
                durations.extend([assumed_dur] * (len(start_times) - len(durations)))
            if start_times[-1] + durations[-1] > max_mutable_time:
                max_mutable_time = start_times[-1] + durations[-1]
            axes.quiver(
                start_times,
                [offsets[synth]] * len(start_times),
                durations,
                0.0,
                scale=1,
                scale_units="x",
                color=colors[synth.track],
            )
        if len(immutable_synths) > 0:
            axes.quiver(
                immutable_synths["X_start"],
                immutable_synths["Y_offset"],
                immutable_synths["U_duration"],
                0.0,
                scale=1,
                scale_units="x",
                alpha=0.5,
                color=immutable_synths["color"],
            )
        if len(set_events) > 0:
            axes.scatter(
                set_events["times"],
                set_events["offsets"],
                marker="o",
                color=set_events["colors"],
                alpha=0.5,
            )

        yticks = list(offsets.values())
        axes.set_yticks(yticks)
        axes.set_yticklabels(
            [
                f"{synth.name}" + (" (mutable)" if synth.mutable else " (immutable)")
                for synth in offsets.keys()
            ]
        )
        axes.set_ylim(yticks[0] - spacing * 0.75, yticks[-1] + spacing * 0.75)
        max_immutable_time = 0
        if len(immutable_synths["X_start"]) > 0:
            max_immutable_time = (
                immutable_synths["X_start"][-1] + immutable_synths["U_duration"][-1]
            )
        axes.set_xlim(
            math.floor(self.first_timestamp) - 0.1,
            max([max_immutable_time, max_mutable_time, self.end_time]) + 0.1,
            auto=True,
        )
        axes.grid(axis="x")
        axes.set_xlabel("time")
        axes.set_ylabel("synths")
        patches = []
        for track in sorted(colors):
            patches.append(mpatches.Patch(color=colors[track], label=f"track {track}"))
        axes.legend(handles=patches, loc="best")
        fig.tight_layout()

    def __repr__(self) -> str:
        if self.is_empty():
            return "Timeline(is_empty=True)"
        with self.lock:
            dict = self.to_dict()
        start = self.first_timestamp
        end = self.end_time
        return f"Timeline({start}-{end} ({end-start}) #entries={len(dict)})"

    def __iter__(self):
        return TimelineEnumerator(self)


class TimelineEnumerator:
    """A Enumerator for the Timeline"""

    def __init__(self, timeline) -> None:
        self._timeline = timeline
        self._current = timeline.head

    def __iter__(self):
        return self

    def __next__(self):
        if self._current is None:
            raise StopIteration
        ret = self._current
        self._current = self._current.next
        return ret
