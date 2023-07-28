import copy
import math
from collections import defaultdict
from threading import RLock
from time import time as get_timestamp
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import attr
import numpy as np
import pyamapping as pam
from attrs import define

from mesonic.events import Event, SynthEvent, SynthEventType

if TYPE_CHECKING:
    from mesonic.synth import Synth


@define(kw_only=True, slots=True, repr=True)
class SynthEventPseudoStop(Event):
    synth: "Synth" = (attr.field(repr=False),)
    etype: SynthEventType = SynthEventType.STOP


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

    def _split_synth_events(self) -> Dict["Synth", Dict["SynthEventType", List[Event]]]:
        splitted_synth_events = {}
        for time, events in self.to_dict().items():
            for event in events:
                if isinstance(event, SynthEvent):
                    if event.synth not in splitted_synth_events:
                        splitted_synth_events[event.synth] = {
                            SynthEventType.START: [],
                            SynthEventType.STOP: [],
                            SynthEventType.SET: [],
                        }
                    splitted_synth_events[event.synth][event.etype].append(
                        (time, event)
                    )
        return splitted_synth_events

    @staticmethod
    def _create_stop_event(
        event_info, duration_key="dur", default_duration=0.1
    ) -> Tuple[float, "SynthEventPseudoStop"]:
        time, start_event = event_info
        duration = start_event.data.get(duration_key, default_duration)
        return time + duration, SynthEventPseudoStop(
            track=start_event.track, info=start_event.info, synth=start_event.synth
        )

    def _check_consistency(self, duration_key="dur"):
        splitted_synth_events = self._split_synth_events()
        for synth, events in splitted_synth_events.items():
            if synth.mutable:
                start_stop_diff = len(events[SynthEventType.START]) - len(
                    events[SynthEventType.STOP]
                )
                if start_stop_diff == 0:
                    # for each start there is a stop
                    continue
                elif start_stop_diff == len(events[SynthEventType.START]):
                    # not stops were made
                    events[SynthEventType.STOP] = list(
                        map(self._create_stop_event, events[SynthEventType.START])
                    )
                elif start_stop_diff > 0:
                    # search missing stops and generate pseudo stops
                    idx = 1
                    nr_stops_generated = 0
                    while (
                        idx < len(events[SynthEventType.START])
                        or nr_stops_generated == start_stop_diff
                    ):
                        start_1 = events[SynthEventType.START][idx - 1]
                        start_2 = events[SynthEventType.START][idx]
                        stop = events[SynthEventType.STOP][idx - 1]
                        assert start_1[0] < start_2[0], "starts overlap"
                        assert start_1[0] < stop[0], "stop without prior start"
                        if start_2[0] < stop[0]:
                            # stop does stop start_2 -> start_1 is not stopped
                            start_time, start_event = start_1
                            events[SynthEventType.STOP].insert(
                                idx - 1, self._create_stop_event(start_event)
                            )
                            nr_stops_generated += 1
                        idx += 1
                    if nr_stops_generated < start_stop_diff:
                        # last start is unmatched.
                        events[SynthEventType.STOP].append(
                            self._create_stop_event(events[SynthEventType.START][-1])
                        )
                else:
                    raise RuntimeError(f"Timeline has more Stop Events for {synth}")
            else:  # synth immutable
                assert len(events[SynthEventType.STOP]) == 0
                assert len(events[SynthEventType.SET]) == 0
                events[SynthEventType.STOP] = list(
                    map(self._create_stop_event, events[SynthEventType.START])
                )
        return splitted_synth_events

    def _create_segments(self, offset_key="freq", width_key="amp"):
        splitted_synth_events = self._check_consistency()
        # for quiver X, U, Y (V=0), arrowwidths
        segments = {
            "begins": [],
            "ends": [],
            "offsets": [],
            "widths": [],
            "connections": [],
        }
        # start & stop tuples (x,y) for scatter, set events: (x,y,x,y2)
        markers = {"starts": [], "stops": [], "sets": []}
        synth_plot_info = {
            synth: {
                "events": events,
                "segments": copy.deepcopy(segments),
                "markers": copy.deepcopy(markers),
            }
            for synth, events in splitted_synth_events.items()
        }
        for synth, events in splitted_synth_events.items():
            begins = synth_plot_info[synth]["segments"]["begins"]
            ends = synth_plot_info[synth]["segments"]["ends"]
            offsets = synth_plot_info[synth]["segments"]["offsets"]
            widths = synth_plot_info[synth]["segments"]["widths"]
            connections = synth_plot_info[synth]["segments"]["connections"]

            for time, event in events[SynthEventType.START]:
                begins.append(time)
                offsets.append((event.data[offset_key]))
                widths.append((event.data[width_key]))
            ends.extend([t[0] for t in events[SynthEventType.STOP]])
            if events[
                SynthEventType.SET
            ]:  # if there are SET events we need to split segments
                # we need to cluster the set events as they are only singular
                combined_set_events = defaultdict(list)
                for set_time, set_event in events[SynthEventType.SET]:
                    combined_set_events[set_time].append(set_event)
                insert_idx = 0
                for set_time, set_events in combined_set_events.items():
                    while insert_idx < len(begins) and begins[insert_idx] < set_time:
                        insert_idx += 1
                    # get properties for the event data
                    found_offset = False
                    found_width = False
                    for set_event in set_events:
                        if set_event.data["name"] == offset_key:
                            offset = set_event.data["new_value"]
                            offsets.insert(insert_idx, offset)
                            found_offset = True
                            connections.append(
                                [
                                    (set_time, set_event.data["old_value"]),
                                    (set_time, offset),
                                ]
                            )
                        if set_event.data["name"] == width_key:
                            width = set_event.data["new_value"]
                            widths.insert(insert_idx, width)
                            found_width = True
                    if found_offset or found_width:
                        # start_idx is now the index of the segment begin that our set interrupts
                        begins.insert(insert_idx, set_time)
                        # insert the new segment end before the old segment end
                        ends.insert(insert_idx - 1, set_time)
                        # if a line property was not found use the value from the prior segment
                        if not found_offset:
                            offset = offsets[insert_idx - 1]
                            offsets.insert(insert_idx, offset)
                        if not found_width:
                            width = widths[insert_idx - 1]
                            widths.insert(insert_idx, width)

            assert len(begins) == len(ends)
            assert len(begins) == len(offsets)
            assert len(begins) == len(widths)
        return synth_plot_info

    def plot_new(self, offset="freq", width="amp", scale=None):
        """Plot the Timeline.

        Parameters
        ----------
        parameter_key : str, optional
            The Parameter name for the y-axis offset, by default "freq"
            This is used to get the bounds of the parameter and set the
            y-axis offset of the synth accordingly.
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
            import matplotlib.ticker as ticker
            from matplotlib.collections import LineCollection

        except ImportError as err:
            raise ImportError(
                "plotting the Timeline is only possible when matplotlib is installed."
            ) from err
        # do not plot an empty Timeline
        if self.is_empty():
            return

        # Events offer
        # track, info
        # SynthEvents offer
        # synth, etype, data
        # dict with synth as key and a inner dict
        # {
        #   for each synth synth
        #   "warp": "lin"|"log"|"exp",
        #   "color":
        #   "y_offset": 0 (depending on synth & track)
        #
        #   events
        #       "starts": [(x,y,s)],
        #       "stops": [(x,y,s)], (virtuell)
        #       "sets": [(x,y,s)],
        #
        #   for each segment
        #   segments
        #       starts
        #       stops
        #       "offsets": [], (depending on 1st value)
        #       "widths": [],  (depending on 2nd value)
        #
        # }
        # so one can use quiver for the segments

        # get segments
        synth_plot_info = self._create_segments(offset_key=offset, width_key=width)

        # synth specifics are added later
        # "warp": None,s
        # "bounds": None,
        # "color": None,
        # "y_offset": 0,

        fig = plt.figure()  # figsize=(8, 2)
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap("Set1")

        x_min, x_max = 0, 0
        y_min, y_max = 0, 0

        patches = []
        synth_colors = {}
        synth_labels = {}
        for synth_idx, synth in enumerate(synth_plot_info.keys()):
            synth_colors[synth] = cmap(synth_idx)
            synth_labels[synth] = f"{synth.name}" + (
                " (mutable)" if synth.mutable else " (immutable)"
            )
            patches.append(
                mpatches.Patch(color=synth_colors[synth], label=synth_labels[synth])
            )

            segments = synth_plot_info[synth]["segments"]
            begins = np.array(segments["begins"])
            ends = np.array(segments["ends"])
            offsets = np.array(segments["offsets"])
            widths = np.array(segments["widths"])

            # TODO implement warps as two or one parameterized fun
            def warp_offset(value, scale=None):
                return value

            def warp_width(value, scale=None, width_min=2, width_max=4):
                if widths.min() == widths.max():
                    return (width_max + width_min) / 2
                ret = pam.linlin(
                    value, widths.min(), widths.max(), width_min, width_max
                )
                return ret

            offsets = warp_offset(offsets)
            widths = warp_width(widths)
            lines = np.array(
                list(
                    zip(
                        list(zip(begins, offsets)),
                        list(zip(ends, offsets)),
                    )
                )
            )
            synth_segments_collection = LineCollection(
                lines, linewidth=widths, color=synth_colors[synth]
            )
            ax.add_collection(synth_segments_collection)
            if synth.mutable:
                connections = synth_plot_info[synth]["segments"]["connections"]
                synth_connections_collection = LineCollection(
                    connections, linewidth=0.5, color=synth_colors[synth]
                )
                ax.add_collection(synth_connections_collection)

            x_min = min(begins.min(), x_min)
            x_max = max(ends.max(), x_max)
            y_min = min(offsets.min(), y_min)
            y_max = max(offsets.max(), y_max)

        # x_lim = (x_max - x_min) * 0.01
        # ax.set_xlim(x_min - x_lim, x_max + x_lim)
        # y_lim = (y_max - y_min) * 0.1
        # axes.set_ylim(y_min - y_lim, y_max + y_lim)

        MAX_LEGEND_NCOLS = 5

        # test
        ax.legend(
            handles=patches,
            loc="lower left",
            mode="expand",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            borderaxespad=0,
            ncol=min(len(patches), MAX_LEGEND_NCOLS),
        )
        ax.grid()
        ax.set_xlabel("time [s]")

        if offset == "freq":
            ax.set_yscale("log")
            ax.set_ylabel("frequency [Hz]")

            # TODO perhaps we also could want mel?
            secax = ax.secondary_yaxis(
                location="right", functions=(pam.cps_to_midi, pam.midi_to_cps)
            )
            secax.set_ylabel("MIDI Note")
            secax.yaxis.set_major_locator(
                ticker.MaxNLocator(nbins="auto", steps=[1, 2, 4, 5, 10], integer=True)
            )
            secax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
        elif offset == "amp":
            ax.set_yscale("log")
            ax.set_ylabel("amplitude")

            def safe_amp_to_db(amp):
                return np.where(amp > 0, pam.amp_to_db(amp), -90)

            secax = ax.secondary_yaxis(
                location="right", functions=(safe_amp_to_db, pam.db_to_amp)
            )
            secax.set_ylabel("level [dB]")
            secax.yaxis.set_major_locator(
                ticker.MaxNLocator(nbins="auto", steps=[1, 2, 4, 5, 10])
            )
            secax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
        else:
            ax.set_ylabel(offset)

        ax.autoscale(enable=True)
        fig.tight_layout()

    # TODO Idea: convention to order the Synth Params by most probalbe usage

    def plot(self, parameter_key="freq", duration_key="dur", default_duration=0.1):
        """Plot the Timeline.

        Parameters
        ----------
        parameter_key : str, optional
            The Parameter name for the y-axis offset, by default "freq"
            This is used to get the bounds of the parameter and set the
            y-axis offset of the synth accordingly.
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
        parameter_offset_min, parameter_offset_max = -0.3, 0.3
        y_offset = spacing

        synth_offsets = {}
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
                    if event.synth not in synth_offsets:
                        synth_offsets[event.synth] = y_offset
                        y_offset += spacing

                    parameter_offset = 0
                    try:
                        x1, x2 = event.synth.params[parameter_key].bounds
                        if x1 is not None and x2 is not None:
                            y1, y2 = parameter_offset_min, parameter_offset_max
                            if event.etype is SynthEventType.SET:
                                if event.data.get("name") == parameter_key:
                                    value = event.data.get("new_value")
                            else:
                                value = event.data.get(
                                    parameter_key,
                                )
                            parameter_offset = (value - x1) / (x2 - x1) * (y2 - y1) + y1
                            print(event.synth, value, parameter_offset)
                    except KeyError:
                        parameter_offset = 0

                    offset = synth_offsets[event.synth] + parameter_offset

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
                            event.synth,
                            {
                                "start_times": [],
                                "stop_times": [],
                                "start_offsets": [],
                                "stop_offsets": [],
                            },
                        )
                        if event.etype == SynthEventType.START:
                            dur = event.data.get(duration_key, None)
                            if dur:
                                synth_dict["stop_times"].append(time + dur)
                                synth_dict["stop_offsets"].append(offset)
                            synth_dict["start_times"].append(time)
                            synth_dict["start_offsets"].append(offset)
                        elif event.etype == SynthEventType.STOP:
                            synth_dict["stop_times"].append(time)
                            synth_dict["stop_offsets"].append(offset)
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
                times["start_offsets"],  # [synth_offsets[synth]] * len(start_times),
                durations,
                [
                    stop - start
                    for (start, stop) in zip(
                        times["start_offsets"], times["stop_offsets"]
                    )
                ],
                angles="xy",
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

        yticks = list(synth_offsets.values())
        axes.set_yticks(yticks)
        axes.set_yticklabels(
            [
                f"{synth.name}" + (" (mutable)" if synth.mutable else " (immutable)")
                for synth in synth_offsets.keys()
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
