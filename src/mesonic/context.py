import logging
import tempfile
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union
from warnings import warn
from weakref import WeakSet

from mesonic.backend import start_backend
from mesonic.backend.bases import B, Backend, EventHandler, Manager
from mesonic.events import Event
from mesonic.playback import Playback
from mesonic.processor import BundleProcessor
from mesonic.timeline import Timeline

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pya import Asig

SYNTHS = "synths"
BUFFERS = "buffers"
RECORDS = "records"

DEFAULT_MANGERS = {
    SYNTHS: lambda backend, context: backend.create_synth_manager(context),
    BUFFERS: lambda backend, context: backend.create_buffer_manager(context),
    RECORDS: lambda backend, context: backend.create_record_manager(context),
}
"""Dict[str, Callable]: Dict containing default managers

Manager names are used as as key and function for thier creation are the values.
This should be adjusted if the Backend is extended with more Managers.
"""


class Context:
    """Context class that offers a central user interface.

    It holds the timeline and all Managers for creating Synths, Records and Buffers.
    The Managers available can be seen using `Context.managers` and can also be accessed
    by the name f.e. `Context.synths` for the SynthManager.

    The Context collects the Events created by the managed Objects and inserts
    them into the timeline.

    Parameters
    ----------
    backend : Backend
        The Backend for this Context

    """

    managers: Dict[str, Manager]
    """Dict with Managers where their names are the key and the instance the value."""
    event_handlers: List[EventHandler]
    """List of the EventHandlers of the Managers"""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._backend.register_context(self)

        self.managers = {}
        self.event_handlers = []

        self._time = None

        self.timeline = Timeline()
        self.processor = BundleProcessor(self.event_handlers)

        self.playbacks = WeakSet()
        self.playback = self.create_playback()
        self._is_realtime = False

        self._in_time_context = False
        self._in_time_context_info = None
        self._context_events: List[Event] = []

        # create managers
        for name, creator in DEFAULT_MANGERS.items():
            try:
                manager = creator(backend, self)
                self.add_manager(name, manager)
            except NotImplementedError as error:
                warn(
                    message=f"Manager for {name} not implemented in {backend}",
                    category=type(error),
                )

    def add_manager(self, name: str, manager: Manager):
        """Add a Manager

        This will also add the respective EventHandler and a property with the name.

        Parameters
        ----------
        name : str
            Name of the Manager
        manager : Manager
            Manager instance

        """
        self.managers[name] = manager
        handler = manager.get_event_handler()
        if handler:
            self.event_handlers.append(handler)
            self.processor.add_handler(handler)
        setattr(Context, name, property(lambda self: self.managers[name]))

    @property
    def time(self) -> float:
        """float: the current time in seconds.

        This respects the is_realtime property."""
        return self.playback.time if self.is_realtime else self._time

    @time.setter
    def time(self, value):
        if self.is_realtime:
            self.playback.time = value
        else:
            self._time = value

    @property
    def backend(self) -> Backend:
        """Backend: the corresponding Backend for this Context"""
        return self._backend

    # Scheduling related methods

    def now(self, delay: float = 0.0, info: Optional[Dict] = None):
        """Context Manager for scheduling Events now (with delay).

        This is useful for grouping Events like demonstrated in the Example.

        Parameters
        ----------
        delay : float, default 0.0
            Amount of latency.
        info : Optional[Dict], optional
            Additional information that will be added to all Events scheduled,
            by default None

        Examples
        --------

        >>> with context.now() as timepoint:
        ...     synth1.start()
        ...     synth2.start()

        Raises
        ------
        RuntimeError
            If used not in realtime mode.
        """
        if not self.is_realtime:
            raise RuntimeError("now can only be used in realtime mode.")
        if self.playback.reversed:
            delay = -delay
        return self.at(self.playback.time + delay, info=info)

    @contextmanager
    def test(self, at=0, rate: Optional[float] = None, reset=True):
        """Context Manager for scheduling Events with prior reset and playback.start after.

        This is useful for repeatedly tweaking and testing your code.

        Parameters
        ----------
        at : float, default 0.0
            time where the playback should start.
        rate : Optional[float], optional
            playback rate, by default None means keep current rate

        Examples
        --------

        You can use
        >>> with context.test(at=0.5):
        ...     with.context.at(1):
        ...         synth.start()
        ...     with.context.at(2):
        ...         synth.stop()

        instead of
        >>> self.context.reset()
        ... with.context.at(1):
        ...     synth.start()
        ... with.context.at(2):
        ...     synth.stop()
        ... context.playback.start(at=0.5)

        Raises
        ------
        RuntimeError
            If used not in realtime mode.
        """
        if reset:
            self.reset()
        try:
            yield self
        except Exception as exception:
            raise RuntimeError(
                "Abort. Exception occured in Context.test"
            ) from exception
        else:
            self.playback.start(at=at, rate=rate)

    @contextmanager
    def at(self, time: float, info: Optional[Dict] = None):
        """Context Manager for scheduling Event at specified time.

        Parameters
        ----------
        time : float
            Timepoint for Timeline at which the Events happen.
        info : Optional[Dict], optional
            Additional information that will be added to all Events scheduled,
            by default None

        Examples
        --------

        >>> with context.at(2.0) as timepoint:
        ...     synth.start()

        Raises
        ------
        RuntimeError
            If scheduling Context Managers are nested.
        """
        if info:
            self._in_time_context_info = info
        if self._in_time_context:
            raise RuntimeError("Already in context manager. Do not nest.")
        self._in_time_context = True
        try:
            yield time
        except Exception as exception:
            raise RuntimeError(
                "Abort. Exception occured in context manager"
            ) from exception
        else:
            assert len(self._context_events), "No events happend"
            if self._in_time_context_info is not None:
                for event in self._context_events:
                    event.info.update(self._in_time_context_info)
            self.timeline.insert(time, self._context_events)
        finally:
            self._in_time_context = False
            self._in_time_context_info = None
            self._context_events = []

    def receive_event(self, event: Event, time: Optional[float] = None):
        """Let the context receive an Event

        This will insert the Event with respect to the current

        Parameters
        ----------
        event : Event
            An Event
        time : Optional[float]
            Optional time of the event.
            If None it will try to use the current Context.time
            Will ignore time when used in scheduling context managers (at/now).
        """
        if self._in_time_context:
            # we are in a scheduling context -> collect the events
            self._context_events.append(event)
            return
        else:
            if time is None:
                time = self.time
            if time is None:
                raise RuntimeError("No time specified")
            self.timeline.insert(time, [event])

    # Playback related methods

    @property
    def is_realtime(self) -> bool:
        """bool: Flag if this is in realtime mode."""
        return self._is_realtime

    def enable_realtime(self, at: float = 0, rate: float = 1) -> Playback:
        """Start the realtime Playback.

        Parameters
        ----------
        at : float, optional
            starting time of the playback, by default 0
        rate : float, optional
            starting rate of the playback, by default 1

        Returns
        -------
        Playback
            realtime Playback instance of this Context.
        """
        if not self.is_realtime:
            self.playback.loop = False
            self.playback.start_time = at
            self.playback.end_time = None
        self.playback.start(at=at, rate=rate)
        self._is_realtime = True
        return self.playback

    def disable_realtime(self) -> Playback:
        """Stop the realtime Playback.

        Returns
        -------
        Playback
            realtime Playback instance of this Contxt.
        """
        self._is_realtime = False
        if self.playback.running:
            self.playback.stop()
        return self.playback

    def create_playback(self, **playback_kwargs) -> Playback:
        """Create a Playback instance.

        This playback the Context.timeline using the Context.processor as processor

        Returns
        -------
        Playback
            created Playback
        """
        pb = Playback(self.timeline, self.processor, **playback_kwargs)
        self.playbacks.add(pb)
        return pb

    # Control related methods

    def reset(self, at: Optional[float] = None, rate: Optional[float] = None):
        """Reset the Context.

        This will clear the Context.timeline and stops the Context.playback
        If Context.is_realtime is True this will restart the Playback instance.

        Parameters
        ----------
        at : float, optional
            Timepoint where the Playback will be restarted, by default None
        rate : float, optional
            Rate for the Playback restart, by default keep current rate

        Returns
        -------
        Playback
            playback instance of Context.

        """
        self.timeline.reset()
        if self.is_realtime:
            return self.playback.restart(at=at, rate=rate)
        elif self.playback.running:
            return self.playback.stop()

    def render(self, output_path=None, **backend_kwargs):
        """Render this context in non-realtime using the Backend.

        See Context.backend.render_nrt for more information.

        Parameters
        ----------
        output_path : path like
            Output path of the rendering.
        """
        self._backend.render_nrt(self, output_path=output_path, **backend_kwargs)

    def render_asig(self, **backend_kwargs) -> "Asig":
        """Render this context in non-realtime as Asig using the Backend.

        See Context.backend.render_nrt for more information.

        """
        from pya import Asig

        asig = None
        with tempfile.TemporaryDirectory(prefix="mesonic_") as tmpdirname:
            filename = f"{tmpdirname}/timeline.wav"
            self._backend.render_nrt(self, output_path=filename, **backend_kwargs)
            asig = Asig(f"{tmpdirname}/timeline.wav")
        return asig

    def stop(self):
        """Stop the current Playbacks and backend audio.

        Note that this will not stop the Playbacks.

        """
        for pb in self.playbacks:
            with suppress(RuntimeError):
                pb.stop()
        self._backend.stop(self)

    def close(self):
        """Close this context.

        This will remove this Context instance from the Backend.
        The context must not be used after calling this method.
        """
        self.stop()
        self._backend.unregister_context(self)


def create_context(
    backend: Union[str, B, Type[B]] = "sc3nb", **backend_kwargs
) -> Context:
    """Create a Context instance using the provided Backend

    Parameters
    ----------
    backend : Union[str, B, Type[B]], optional
        The backend to be used, by default "sc3nb"
        See :py:func:`mesonic.backend.start_backend` for more details.

    Returns
    -------
    Context
        The created Context.
    """
    backend_instance = start_backend(backend=backend, **backend_kwargs)
    return Context(backend_instance)
