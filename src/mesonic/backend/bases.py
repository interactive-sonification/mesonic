"""Defines backend base classes

This module contains the relevant classes that are needed to implement a backend for mesonic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)
from weakref import WeakKeyDictionary

from mesonic.events import Event, RecordEvent, SynthEvent

if TYPE_CHECKING:
    numeric = Union[int, float]

    from pya import Asig

    from mesonic.buffer import Buffer
    from mesonic.context import Context
    from mesonic.record import Record
    from mesonic.synth import Synth


E = TypeVar("E", bound=Event)


class EventHandler(Generic[E]):
    """A generic handler for abitrary Events.

    This describes the EventHandler interface.
    It is used for processing Events at the backend.

    The handling of a Event should be quickly done or run in the background
    using an explicit Thread for example to not stall the processing of multiple
    events.

    """

    @abstractmethod
    def handle(
        self, time: float, events: Iterable[E], reversed: bool, **kwargs
    ) -> None:
        """Handle events with the provided time and state.

        Parameters
        ----------
        time : float
            Timestamp provided for the events.
        events : Iterable[E]
            Events to be handled.
        reversed : bool
            Whether the events must be reversed or not.

        """
        raise NotImplementedError

    @abstractmethod
    def get_etype(self) -> type:
        """Get the type of Event this handler processes.

        Returns
        -------
        type
            type of Event this handler processes.

        """
        raise NotImplementedError


class SynthEventHandler(EventHandler[SynthEvent]):
    """A :py:class:`~mesonic.backend.bases.EventHandler` for :py:class:`~mesonic.events.SynthEvent`."""

    @abstractmethod
    def handle(
        self, time, events: Iterable[SynthEvent], reversed: bool, **kwargs
    ) -> None:
        raise NotImplementedError

    def get_etype(self) -> type:
        return SynthEvent


class RecordEventHandler(EventHandler[RecordEvent]):
    """A :py:class:`~mesonic.backend.bases.EventHandler` for :py:class:`~mesonic.events.RecordEvent`."""

    @abstractmethod
    def handle(
        self, time, events: Iterable[RecordEvent], reversed: bool, **kwargs
    ) -> None:
        raise NotImplementedError

    def get_etype(self) -> type:
        return RecordEvent


H = TypeVar("H", bound=Optional[EventHandler])
B = TypeVar("B", bound="Backend")


class Manager(Generic[B, H]):
    """A generic Manager.

    This describes the Manager interface.
    It is used for preparing the Event generating instances and also prepares the
    :py:class:`~mesonic.backend.bases.EventHandler`.
    It connects the :py:class:`~mesonic.context.Context` with the Backend

    Parameters
    ----------
    backend : B
        backend instance that is used.
    context: Context
        context instance that is used.

    """

    _backend: B
    _context: "Context"
    event_handler: H

    def __init__(self, backend: B, context: "Context") -> None:
        super().__init__()
        self._backend = backend
        self._context = context

    def get_event_handler(self) -> H:
        """Get the responsible EventHandler.

        This should return None if no Events need to be handled.

        Returns
        -------
        H
            EventHandler for the instances managed by this Manager.
        """
        return self.event_handler


class RecordManager(Manager[B, RecordEventHandler]):
    """Manager for Records."""

    def __init__(self, backend: B, context: "Context") -> None:  # pragma: no cover
        super().__init__(backend, context)
        self._records: WeakKeyDictionary["Record", Any] = WeakKeyDictionary()
        self.event_handler = self._create_event_handler(backend, self._records)

    @abstractmethod
    def _create_event_handler(
        self, backend: B, records: WeakKeyDictionary["Record", Any]
    ) -> RecordEventHandler:
        raise NotImplementedError

    @abstractmethod
    def create(self, path, track: int = 0, **kwargs) -> "Record":
        """Create a Record instance.

        Parameters
        ----------
        path : path like
            Path where the record should be stored.
        track : int, optional
            On which track the RecordEvents should be created, by default 0

        Returns
        -------
        Record
            created Record

        """
        raise NotImplementedError

    @property
    def instances(self) -> Set[Record]:
        """Get the current Records.

        Returns
        -------
        Set[Record]
            Set of current Records.
        """
        return set(self._records.keys())

    def __repr__(self) -> str:
        return f"RecordManager(num of records={len(self._records)})"


class SynthManager(Manager[B, SynthEventHandler]):
    """Manager for Synths."""

    def __init__(self, backend: B, context: "Context") -> None:  # pragma: no cover
        super().__init__(backend, context)
        self._synths: WeakKeyDictionary["Synth", Any] = WeakKeyDictionary()
        self.event_handler = self._create_event_handler(backend, self._synths)

    @abstractmethod
    def _create_event_handler(
        self, backend: B, synths: WeakKeyDictionary["Synth", Any]
    ) -> SynthEventHandler:
        raise NotImplementedError

    @abstractmethod
    def create(
        self, name: str, track: int = 0, *, mutable: bool = True, **kwargs
    ) -> "Synth":
        """Create a Synth.

        Parameters
        ----------
        name : str
            Name of Synth
        track : int, optional
            On which track the SynthEvents should be created, by default 0
        mutable : bool, optional
            Wether this Synth is mutable, by default True

        Returns
        -------
        Synth
            The created Synth.

        """
        raise NotImplementedError

    @abstractmethod
    def from_buffer(self, buffer: "Buffer", synth_name: str, **synth_kwargs) -> "Synth":
        """Create a Synth to playback a buffer.

        This will provide the Backend with the information needed to create a Synth
        that can playback the provided Buffer.

        Parameters
        ----------
        buffer : Buffer
            Buffer instance that should be connected to the Synth.
        synth_name : str
            Name of Synth that should playback the Buffer.

        Returns
        -------
        Synth
            The Synth created for the Buffer.

        """
        raise NotImplementedError

    @abstractmethod
    def add_buffer_synth_def(self, name, **kwargs):
        """Create a Synth Definition for SynthManager.from_buffer() to the backend.

        Parameters
        ----------
        name : str
            SynthDef name.

        """
        raise NotImplementedError

    @abstractmethod
    def add_synth_def(self, name, **kwargs):
        """Create a Synth Definition to the backend.

        Parameters
        ----------
        name : str
            SynthDef name.

        """
        raise NotImplementedError

    @property
    def names(self) -> Set[str]:
        """Get the names of the currently used Synths.

        Returns
        -------
        Set[str]
            Set containing the names of the currently used Synths.
        """
        return {synth.name for synth in self._synths.keys()}

    def __repr__(self) -> str:
        synth_overview = Counter(synth.name for synth in self._synths)
        return f"SynthManager({dict(synth_overview)})"


class BufferManager(Manager[B, None]):
    """Manager for Buffers."""

    event_handler: None

    def __init__(self, backend: B, context: "Context") -> None:  # pragma: no cover
        super().__init__(backend, context)
        self._buffers: WeakKeyDictionary["Buffer", Any] = WeakKeyDictionary()
        self.event_handler = None

    @abstractmethod
    def from_data(self, data, sr: "numeric", **kwargs) -> "Buffer":
        """Create a Buffer from the provided data and sampling rate.

        Parameters
        ----------
        data : Any
            Samples for the Buffer. Typically numpy arrays or other collections.
        sr : numeric
            Sampling rate of the Buffer. Type depends on backend.

        Returns
        -------
        Buffer
            The created Buffer.

        """
        raise NotImplementedError

    @abstractmethod
    def from_file(
        self,
        path,
        starting_frame: int = 0,
        num_frames: int = -1,
        channels: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> "Buffer":
        """Create a Buffer from a file.

        Parameters
        ----------
        path : path like
            Path of the file.
        starting_frame : int, optional
            Index of the first frame, by default 0
        num_frames : int, optional
            Number of frames to be loaded, by default -1
        channels : Optional[Union[int, Sequence[int]]], optional
            Specification of the channels to load, by default None
            - None means all channels.
            - If a int is provided only that channel will be loaded.
            - If a Sequence of ints is provided the provided channels will be loaded.

        Returns
        -------
        Buffer
            The Buffer loaded from the file.

        """
        raise NotImplementedError

    def from_asig(self, asig: "Asig", **kwargs) -> "Buffer":  # pragma: no cover
        """Create a Buffer from an :py:class:`pya.Asig`

        Parameters
        ----------
        asig : Asig
            The Asig that should be created as Buffer.

        Returns
        -------
        Buffer
            The Buffer created from the Asig.
        """
        return self.from_data(asig.sig, asig.sr, **kwargs)

    def __repr__(self) -> str:
        return f"BufferManager({list(self._buffers)})"


class Backend(ABC):
    """Base class of the mesonic Backend.

    This describes the Backend interface.
    The Backend should start or connect to the actual backend program.
    It is also creating the single Managers for the Contexts and stores variables.
    """

    _contexts: WeakKeyDictionary["Context", Dict]

    def __init__(self) -> None:
        super().__init__()
        self._contexts = WeakKeyDictionary()

    @property
    @abstractmethod
    def sampling_rate(self):
        """Get the native sampling rate of the Backend"""
        raise NotImplementedError

    @abstractmethod
    def register_context(self, context: "Context") -> None:
        """Register a context to this Backend.

        Parameters
        ----------
        context : Context
            Context that should use this backend.

        """
        raise NotImplementedError

    def get_context_vars(self, context: "Context") -> Dict:
        """Get the stored variables of the provided Context.

        Parameters
        ----------
        context : Context
            A registered Context of which the variables should be returned.

        Returns
        -------
        Dict
            Stored variables of a context.
        """
        return self._contexts[context]

    @abstractmethod
    def unregister_context(self, context: "Context") -> None:
        """Remove the context from the registered contexts.

        Parameters
        ----------
        context : Context
            Context to be removed.

        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, context: "Context") -> None:
        """Stop all sounds of the provided Context.

        Parameters
        ----------
        context : Context
            The Context which creates the sounds to be stopped.

        """
        raise NotImplementedError

    @abstractmethod
    def create_synth_manager(self, context: "Context") -> SynthManager:
        """Create a SynthManager instance for the provided Context

        Parameters
        ----------
        context : Context
            Context instance.

        Returns
        -------
        SynthManager
            SynthManger instance.

        """
        raise NotImplementedError

    @abstractmethod
    def create_buffer_manager(self, context: "Context") -> BufferManager:
        """Create a BufferManager instance for the provided Context

        Parameters
        ----------
        context : Context
            Context instance.

        Returns
        -------
        BufferManager
            BufferManager instance.

        """
        raise NotImplementedError

    @abstractmethod
    def create_record_manager(self, context: "Context") -> RecordManager:
        """Create a RecordManager instance for the provided Context

        Parameters
        ----------
        context : Context
            Context instance.

        Returns
        -------
        RecordManager
            RecordManager instance.

        """
        raise NotImplementedError

    @abstractmethod
    def render_nrt(self, context: "Context", output_path, **backend_kwargs):
        """Render the provided Context in non-realtime.

        Parameters
        ----------
        context : Context
            Context to be rendered.
        output_path : path like
            Path for the output file.

        """
        raise NotImplementedError

    @abstractmethod
    def quit(self):
        """Quit this Backend."""
        raise NotImplementedError
