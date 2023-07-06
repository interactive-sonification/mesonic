from enum import Enum, auto, unique
from typing import TYPE_CHECKING

from .events import RecordEvent, RecordEventType

if TYPE_CHECKING:
    from pya import Asig

    from mesonic.context import Context


@unique
class RecordState(Enum):
    """The different States of the Record"""

    PREPARED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


class Record:
    """A Record allows to save the audio to a file.

    Parameters
    ----------
    context : Context
        The Context in which this Record happens.
    path : path like
        Path where the record should be stored.
    track : int, optional
        On which Timeline track the RecordEvents should be created, by default 0

    """

    def __init__(self, context: "Context", path, track: int = 0) -> None:
        self._context: "Context" = context
        self._path = path
        self._track = track
        self._state = RecordState.PREPARED
        self._finished = False

    def __repr__(self) -> str:
        return f"Record(path={self.path}, finished={self.finished})"

    @property
    def path(self):
        """pathlike: The path where this Record will be stored."""
        return self._path

    @property
    def finished(self) -> bool:
        """bool: Wether the record has been finished."""
        return self._finished

    def to_asig(self) -> "Asig":
        """Transform this record into an pya.Asig.

        Returns
        -------
        Asig
            Asig of the record.

        Raises
        ------
        RuntimeError
            If this is called on an unfinished Record.
        ImportError
            If pya can't be found.
        """
        if not self.finished:
            raise RuntimeError("The record must be finished.")
        # will raise an import error when pya is not available
        from pya import Asig  # pylint: disable=C0415

        return Asig(self._path)

    def _send_event(self, etype: RecordEventType, info):
        """Send a RecordEvent.

        Parameters
        ----------
        etype : RecordEventType
            Type of RecordEvent.
        info : Any
            Additional information about the Event.
        """
        self._context.receive_event(
            RecordEvent(
                info=info or {},
                record=self,
                etype=etype,
            )
        )

    def start(self, info=None):
        """Start the Record.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None

        Raises
        ------
        RuntimeError
            If the RecordState is not PREPARED.
        """
        if self._state is not RecordState.PREPARED:
            raise RuntimeError("A record can only be started once.")
        self._state = RecordState.RUNNING
        self._send_event(RecordEventType.START, info=info)

    def pause(self, info=None):
        """Pause the Record.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None

        Raises
        ------
        RuntimeError
            If the RecordState is not RUNNING.
        """
        if self._state is not RecordState.RUNNING:
            raise RuntimeError("The record must be running.")
        self._state = RecordState.PAUSED
        self._send_event(RecordEventType.PAUSE, info=info)

    def stop(self, info=None):
        """Stop the Record.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None

        Raises
        ------
        RuntimeError
            If the RecordState is not RUNNING.
        """
        if self._state is not RecordState.RUNNING:
            raise RuntimeError("The record must be running.")
        self._state = RecordState.STOPPED
        self._send_event(RecordEventType.STOP, info=info)

    def resume(self, info=None):
        """Resume the Record.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None

        Raises
        ------
        RuntimeError
            If the RecordState is not PAUSED.
        """
        if self._state is not RecordState.PAUSED:
            raise RuntimeError("The record must be paused before resuming.")
        self._state = RecordState.RUNNING
        self._send_event(RecordEventType.RESUME, info=info)
