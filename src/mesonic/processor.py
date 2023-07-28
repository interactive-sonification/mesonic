import logging
import warnings
from collections import defaultdict
from copy import copy
from typing import Callable, Dict, List, Optional

from mesonic.backend.bases import EventHandler
from mesonic.events import Event
from mesonic.timeline import TimeBundle

_LOGGER = logging.getLogger(__name__)


class BundleProcessor:
    """The BundleProcessor preprocesses the TimeBundles and sends them to multiple event_handlers.

    Parameters
    ----------
    event_handlers : List[EventHandler]
        List of EventHandlers that should handle the TimeBundle contents.
    latency : float, optional
        the latency that will be added to the scheduled time, by default 0.2

    """

    def __init__(self, event_handlers: List[EventHandler], latency=0.2) -> None:
        super().__init__()
        self.latency = latency
        self.event_handlers = {
            handler.get_etype(): handler for handler in event_handlers
        }
        self.selected_tracks: Optional[List[int]] = None
        """Optional list of selected tracks.
        If this is not None all Event with a track not in selected_tracks won't be passed."""
        self.event_filters: List[Callable[[Event], Optional[Event]]] = []
        """Optional event info filter function. Should return None for filtering.
        If this is not None all Event will be filtered using this function."""

    def add_handler(self, handler: EventHandler):
        """Add an EventHandler

        This will overwrite an old EventHandler if they handle the same Type.

        Parameters
        ----------
        handler : EventHandler
            The new EventHandler.
        """
        self.event_handlers[handler.get_etype()] = handler

    def process_bundle(
        self, bundle: TimeBundle, scheduled_time: float, reversed: bool, **kwargs
    ) -> None:
        """Process the Bundle at the scheduled time according to reversed bool.

        Parameters
        ----------
        bundle : TimeBundle
            TimeBundle to process.
        scheduled_time : float
            Timepoint for which this TimeBundle is scheduled.
        reversed : bool
            Wether this TimeBundle should be reversed.
        """

        events = self.prepare_events(bundle.events, self.selected_tracks)

        splitted_events = self.split_events(events)

        # call the event handler for each type
        for etype, events_of_type in splitted_events.items():
            if etype in self.event_handlers:
                self.event_handlers[etype].handle(
                    time=scheduled_time + self.latency,
                    events=events_of_type,
                    reversed=reversed,
                    **kwargs,
                )

    def prepare_events(
        self,
        events: List[Event],
        selected_tracks: Optional[List[int]] = None,
    ) -> List[Event]:
        """Filter and transform Events using the provided filters.

        Note that the filters are applied before selecting the Tracks.
        This means that the function could be used to change the track
        of an Event prior to the selection of tracks.

        Parameters
        ----------
        events : List[Event]
            A list of Events that should be prepared.
        filters : List[Callable[[Event], Optional[Event]]]
            An a list of filter functions that take an Event and
            return a transformed Event or None to discard the Event.
        selected_tracks : Optional[List[int]]
            List of selected tracks, default None
            If this is not None all Event with a track not in
            selected_tracks won't be passed.

        Returns
        -------
        List[Event]
            The list of Events after the transformation.
        """
        events = copy(events)

        def event_selection(event: Event) -> Optional[Event]:
            """Filter events with a combined function."""
            bad_filters = []
            for filter_fun in self.event_filters:
                try:
                    filtered_event = filter_fun(event)
                except Exception as exception:
                    bad_filters.append(filter_fun)
                    warnings.warn(
                        f"event_filter {filter_fun} raised exception of type {type(exception)} ({exception}), removing it"
                    )
                else:
                    if filtered_event is None:
                        return None
                    event = filtered_event
            if bad_filters:
                self.event_filters = [
                    event_filter
                    for event_filter in self.event_filters
                    if event_filter not in bad_filters
                ]
            # select the tracks
            if selected_tracks is not None:
                return event if event and event.track in selected_tracks else None
            return event

        return list(filter(None, map(event_selection, events)))

    @classmethod
    def split_events(cls, events: List[Event]) -> Dict[type, List[Event]]:
        """Split events by type

        Parameters
        ----------
        events : List[Event]
            A list of Events.

        Returns
        -------
        Dict[type, List[Event]]
            Dictionary with types as keys and belonging Events as value.
        """
        splitted_events = defaultdict(list)
        for event in events:
            splitted_events[type(event)].append(event)
        return splitted_events
