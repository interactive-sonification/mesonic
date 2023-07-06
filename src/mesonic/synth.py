from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from mesonic.events import SynthEvent, SynthEventType

if TYPE_CHECKING:
    from mesonic.context import Context


@dataclass
class ParameterInfo:
    name: str
    default: Any
    min_value = None
    max_value = None


class Parameter:
    """A Parameter of a Synth.

    Parameters
    ----------
    synth : Synth
        The Synth to which this Parameter belongs.
    name : str
        The name of the Parameter.
    default : Any
        The default Paramter value.
    min_value : Any, optional
        The min value possible for this Parameter, by default None
        None means there is no check.
    max_value : Any, optional
        The max value possible for this Parameter, by default None
        None means there is no check.

    """

    def __init__(
        self,
        synth: "Synth",
        name: str,
        default,
        min_value=None,
        max_value=None,
    ):
        self._synth = synth
        self._name = name
        self._value = None
        self._default = default
        self._bounds = (min_value, max_value)

    # properties

    @property
    def default(self):
        """Default value of the Parameter."""
        return self._default

    @property
    def max(self) -> Any:
        """Maximum allowed value of the Parameter.
        If it is None there is no check when setting."""
        return self._bounds[1]

    @max.setter
    def max(self, value):
        self._bounds = (self._bounds[0], value)

    @property
    def min(self) -> Any:
        """Maximum allowed value of the Parameter.
        If it is None there is no check when setting."""
        return self._bounds[0]

    @min.setter
    def min(self, value):
        self._bounds = (value, self._bounds[1])

    @property
    def bounds(self) -> Tuple[Any, Any]:
        """The bounds (min, max) of the Parameter"""
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        min, max = value
        self.min = min
        self.max = max

    @property
    def name(self) -> str:
        """str: The Paramter name."""
        return self._name

    @property
    def value(self) -> Any:
        """The Parameter value.
        If no explicit value is set return the default Paramter value.
        Setting value is the same as using the set method.
        """
        return self._default if self._value is None else self._value

    @value.setter
    def value(self, value):
        if value is self:
            return
        self.set(value, None)

    def set(self, value, info=None):
        """Set the Parameter value.

        Parameters
        ----------
        value : Any
            The new Parameter value.
        info : Any
            Additional information about the Event.

        """
        # create event data based on old and new values
        data = {"name": self._name, "new_value": value, "old_value": self._value}
        cache_only = not self._synth.mutable
        self._verify_and_adjust(value, cache_only)
        if not cache_only:
            self._synth._send_event(SynthEventType.SET, data=data, info=info)

    def _verify_and_adjust(self, value, cache_only: bool = False):
        """Verify and adjust the value.

        Parameters
        ----------
        value : Any
            The new Parameter value.
        cache_only : bool, optional
            True when we want to ignore mutable, by default False

        Raises
        ------
        ValueError
            If value is out of bounds or Synth is not mutable and cache_only is True
        """
        if not self._synth.mutable and not cache_only:
            raise ValueError("Synth not mutable")
        if self.max is not None and value > self.max:
            raise ValueError(
                f"Value larger then allowed value: {value} > {self.max} (max)"
            )
        if self.min is not None and value < self.min:
            raise ValueError(
                f"Value smaller then allowed value: {value} < {self.min} (min)"
            )
        self._value = value

    def __iadd__(self, other) -> "Parameter":
        self.value += other
        return self

    def __isub__(self, other) -> "Parameter":
        self.value -= other
        return self

    def __imul__(self, other) -> "Parameter":
        self.value *= other
        return self

    def __itruediv__(self, other) -> "Parameter":
        self.value /= other
        return self

    def __ifloordiv__(self, other) -> "Parameter":
        self.value //= other
        return self

    def __repr__(self) -> str:
        return (
            f"Parameter({self.name}={self.value:.2f}, "
            f"default={self.default:.2f}, bounds={self._bounds})"
        )


class Synth:
    """A controllable audio source.

    Parameters
    ----------
    context : Context
        Context for this Synth
    name : str
        name of this Synth
    mutable : bool
        True if this Synth is mutable
    param_info : List[ParameterInfo]
        Parameter information of the Synth
    track : int, optional
        track of the Synth, by default 0
    metadata : Optional[Dict], optional
        additional metadata, by default None
    """

    def __init__(
        self,
        context: "Context",
        name: str,
        mutable: bool,
        param_info: List[ParameterInfo],
        track: int = 0,
        metadata: Optional[Dict] = None,
    ):
        # using object.__settattr__ to avoid the overwritten __setattr__
        # for details see https://github.com/dreinsch/mesonic/issues/6
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mutable", mutable)
        object.__setattr__(
            self,
            "_params",
            {
                pi.name: Parameter(
                    self, pi.name, pi.default, pi.min_value, pi.max_value
                )
                for pi in param_info
            },
        )
        object.__setattr__(self, "_track", track)
        object.__setattr__(self, "metadata", metadata or {})

    @property
    def name(self) -> str:
        """str: name of the Synth."""
        return self._name

    @property
    def track(self) -> int:
        """int: track of the Synth."""
        return self._track

    @property
    def context(self) -> "Context":
        """Context: Context in which the Synth happens."""
        return self._context

    @property
    def mutable(self) -> bool:
        """bool: True if this is a mutable Synth."""
        return self._mutable

    @property
    def params(self) -> Dict[str, "Parameter"]:
        """Dict[str, "Parameter"]: The Parameters of this Synths."""
        return self._params

    # scheduled methods

    def start(self, params: Optional[Dict[str, Any]] = None, info=None, **kwargs):
        """Start the Synth.

        Parameters
        ----------
        params : Optional[Dict[str, Any]], optional
            A dict with (name, value) pairs for the Parameters, by default None
        info : Any, optional
            Additional information about the Event, by default None, by default None
        kwargs: Any, optional
            Additional keyword arguments are added to params.

        """
        if params or kwargs:
            kwargs |= params or {}
            for name, value in kwargs.items():
                assert (
                    name in self._params
                ), f"'{name}' is not in this Synths Parameters"
                self._params[name]._verify_and_adjust(value, cache_only=True)
        data = self._param_values()
        self._send_event(SynthEventType.START, data, info)

    def stop(self, info=None):
        """Stop the Synth.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None, by default None

        """
        self._send_event(SynthEventType.STOP, self._param_values(), info)

    def pause(self, info=None):
        """Pause the Synth.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None, by default None

        """
        self._send_event(SynthEventType.PAUSE, self._param_values(), info)

    def resume(self, info=None):
        """Resume the Synth.

        Parameters
        ----------
        info : Any, optional
            Additional information about the Event, by default None, by default None

        """
        self._send_event(SynthEventType.RESUME, self._param_values(), info)

    def set(self, params: Optional[Dict[str, Any]] = None, info=None, **kwargs):
        """Set the Parameters of the Synth.

        Parameters
        ----------
        params : Optional[Dict[str, Any]], optional
            A dict with (name, value) pairs for the Parameters.
        info : Any, optional
            Additional information about the Event, by default None, by default None
        kwargs: Any, optional
            Additional keyword arguments are added to params.

        """
        if params or kwargs:
            kwargs |= params or {}
            for name, value in kwargs.items():
                self._params[name].set(value, info=info)

    def _param_values(self) -> Dict[str, Any]:
        """Get the current best value for each Parameter.

        Returns
        -------
        Dict[str, Any]
            A dict with the param names and values.

        """
        return {n: p.value for n, p in self._params.items()}

    def _send_event(
        self,
        etype: SynthEventType,
        data: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ):
        """Send a SynthEvent

        Parameters
        ----------
        etype : SynthEventType
            Type of SynthEvent
        data : Dict[str, Any]
            Data about the SynthEvent
        info : Any
            Additional information about the Event.

        Raises
        ------
        RuntimeError
            If a immutable Synth tries to send another SynthEventType as START
        """
        if not self._mutable and etype is not SynthEventType.START:
            raise RuntimeError(
                f"Immutable Synth can only produce {SynthEventType.START} "
                f"but got {etype}"
            )
        einfo: Dict = {}
        einfo |= self.metadata
        if info:
            einfo |= info
        event = SynthEvent(
            track=self._track,
            info=einfo,
            synth=self,
            etype=etype,
            data=data,
        )
        self._context.receive_event(event)

    def __getattr__(self, name: str):
        if name in self._params:
            return self._params[name]
        return object.__getattribute__(self, name)

    def __getitem__(self, name: str):
        return self._params[name]

    def __setattr__(self, name, value):
        # First try regular attribute access.
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            if name not in self._params:
                raise AttributeError("can't set attribute")
            self._params[name].value = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        values = {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in self._param_values().items()
        }
        return f"Synth({self.name}, {values})"
