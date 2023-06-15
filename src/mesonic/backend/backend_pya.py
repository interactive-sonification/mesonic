"""Implementation of the backend using pya."""
from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import reduce, partial
from operator import iconcat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Union, NamedTuple
from weakref import WeakKeyDictionary


from mesonic.backend.bases import (
    Backend,
    BufferManager,
    EventHandler,
    RecordEventHandler,
    RecordManager,
    SynthEventHandler,
    SynthManager,
)
from mesonic.buffer import Buffer, BufferInfo
from mesonic.events import (
    Event,
    RecordEvent,
    RecordEventType,
    SynthEvent,
    SynthEventType,
)
from mesonic.record import Record
from mesonic.synth import ParameterInfo, Synth

if TYPE_CHECKING:
    from typing import List
    from mesonic.context import Context

from pya import *  # TODO: import only needed classes resp. import pya

##################################################################
# synthdef for pya (experimental, later to be moved to pya)
"""Module for creating SynthDefs and Synths in pya """

import re
import sys
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

def get_defaults(fn): 
    """return dict with arguments and default values
    fn can be either a function or a functools.partial.    
    """    
    if type(fn) == partial:
        arguments = list(fn.func.__code__.co_varnames)
        defaults = list(fn.func.__defaults__)
        used_args = list(fn.keywords.keys())
        # print(arguments, defaults, used_args)

        for a in used_args:
            try:
                idx = arguments.index(a)
                # print(a, idx)
                del(arguments[idx])
                del(defaults[idx])
            except:
                pass

    else: # assume function
        arguments = fn.__code__.co_varnames 
        defaults = list(fn.__defaults__)
    defaults += [None] * (len(arguments)-len(defaults))
    return dict(zip(arguments, defaults))

class SynthArgument(NamedTuple):
    """Synth argument, rate and default value"""

    name: str
    rate: str
    default: Any


class PyaSynthDef:
    """Pya SynthDef class"""

    synth_descs = {}
    synth_defs = {}

    @classmethod
    def get_description(
        cls, name: str
    ) -> Optional[Dict[str, "SynthArgument"]]:
        """Get Synth description

        Parameters
        ----------
        name : str
            name of SynthDef

        Returns
        -------
        Dict
            dict with SynthArguments
        """
        if name in cls.synth_descs:
            return cls.synth_descs[name]
        
        synth_desc = None

        try:
            pya_syn_fn = cls.synth_defs[name]
            synth_desc = get_defaults(pya_syn_fn)
            for a, v in synth_desc.items():
                synth_desc[a] = SynthArgument(name=a, rate="ir", default=v)
        except ValueError:
            warnings.warn(
                f"SynthDesc '{name}' is unknown. pya does not know this SynthDef"
            )
        if synth_desc is not None:
            cls.synth_descs[name] = synth_desc
        return synth_desc

    @classmethod
    def load_dir(
        cls,
        synthdef_dir: Optional[str] = None,
    ):
        """Load all SynthDefs from directory.
        TH: these could later be just an import of functions stored in a file

        # Parameters
        # ----------
        # synthdef_dir : str, optional
        #     directory with SynthDefs, by default sc3nb default SynthDefs
        # completion_msg : bytes, optional
        #     Message to be executed by the server when loaded, by default None
        # server : SCServer, optional
        #     Server that gets the SynthDefs,
        #     by default use the SC default server
        """
        if server is None:
            server = Aserver.default

        def _load_synthdefs(path):
            cmd_args: List[Union[str, bytes]] = [path.as_posix()]
            ...
            # server.msg(
            #     SynthDefinitionCommand.LOAD_DIR,
            #     cmd_args,
            #     await_reply=True,
            #     bundle=True,
            # )

        if synthdef_dir is None:
            ...
            # ref = libresources.files(sc3nb.resources) / "synthdefs"
            # with libresources.as_file(ref) as path:
            #     _load_synthdefs(path)
        else:
            path = Path(synthdef_dir)
            if path.exists() and path.is_dir():
                # _load_synthdefs(path)
                ...
            else:
                raise ValueError(f"Provided path {path} does not exist or is not a dir")

    def __init__(self, name: str, definition: function) -> None:
        """Create a dynamic synth definition in pya

        Parameters
        ----------
        name : string
            default name of the synthdef creation.
            The naming convention will be name+int, where int is the amount of
            already created synths of this definition
        definition : function
            Pass the default synthdef definition here. Flexible content
        """
        self.definition = definition
        self.name = name
        self.current_def = definition

    def reset(self) -> "PyaSynthDef":
        """Reset the current synthdef configuration to the self.definition value.

        After this you can restart your configuration with the same root definition

        Returns
        -------
        object of type PyaSynthDef
            the PyaSynthDef object
        """
        self.current_def = self.definition
        return self
    
    def add(
        self,
        name: Optional[str] = None,
    ) -> str:
        """This method will add the current_def to Pya

        If a synth with the same definition was already in sc, this method
        will only return the name.

        Parameters
        ----------
        name : str, optional
            name which this SynthDef will get

        Returns
        -------
        str
            Name of the SynthDef
        """
        if name is not None:
            self.name = name

        # Create new SynthDef and add it to SynthDescLib and get bytes
        PyaSynthDef.synth_defs[self.name] = self.current_def


        return self.name


    def __repr__(self):
        return "repr for PyaSynthDef"
        # try:
        #     pyvars = parse_pyvars(self.current_def)
        # except NameError:
        #     current_def = self.current_def
        # else:
        #     current_def = replace_vars(self.current_def, pyvars)
        # return f"PyaSynthDef('{self.name}', {current_def})"

##################################################################

class EventHandlerPya(EventHandler):
    """An abstract EventHandler as basis pya EventHandlers.

    Breaks down the handling of mutliple Events to just single events by using
    just a loop (instead of using a Bundler in sc3nb)
    """

    backend: "BackendPya"

    @abstractmethod
    def _handle_event(self, event: Event):
        print("_handle_event called for time {time}")
        ...

    def handle(self, time, events: Iterable[Event], reversed: bool, **kwargs) -> None:
        # print(f"EventHandlerPya handle events {events} called with time {time}")
        if time is None:
            time = 0
        # here time is the time which usually is put in the bundle timetag.
        # so I have to make sure that this time is available when the event is written into the acanvas
        # as test let's make it a cls variable timetag of EventHandlerPya
        Aserver._timetag = time
        for event in events:
            if reversed:
                event = event.reverse()
            self._handle_event(event)

class BufferManagerPya(BufferManager):
    """BufferManager for pya
    Buffers are just Asigs, stored in dictionary buffers of the context.
    """

    def _create_buffer(self, buf: Asig) -> Buffer:
        # assert buf.allocated
        buf_info = BufferInfo(buf.samples, buf.channels, buf.sr)  # type: ignore
        buffer = Buffer(self._context, buf_info)
        self._buffers[buffer] = buf
        return buffer

    def from_data(self, data, sr, **kwargs) -> Buffer:
        buf = Asig(data, sr, **kwargs)
        return self._create_buffer(buf)

    def from_file(
        self,
        path,
        starting_frame: int = 0,
        num_frames: int = -1,
        channels: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> Buffer:
        buf = Asig(path, **kwargs)[starting_frame:num_frames, channels]
        return self._create_buffer(buf)

class SynthEventHandlerPya(EventHandlerPya, SynthEventHandler):
    """SynthEventHandler for pya
    Synths are just python functions that create an Asig.
    """

    def __init__(
        self, backend: "BackendPya", synths: WeakKeyDictionary[Synth]
    ) -> None:
        self.backend = backend
        self.synths = synths

    def _handle_event(self, event: Event):
        assert isinstance(event, SynthEvent), "wrong Event type"
        synth = event.synth
        try:
            backend_part = self.synths[synth]
        except KeyError:
            warnings.warn(f"Received event for unkown Synth: {synth}")
            return
        etype = event.etype
        if synth.mutable:
            # not clear howto do mutable synths in pya - omit for now
            # print("no mutable synths in pya")
            node = backend_part
            # # TODO use pattern matching when `python_requires = >= 3.10`
            # if etype is SynthEventType.START:
            #     node.new(controls=event.data)
            # elif etype is SynthEventType.STOP:
            #     node.free()
            # elif etype is SynthEventType.PAUSE:
            #     node.run(False)
            # elif etype is SynthEventType.RESUME:
            #     node.run(True)
            # elif etype is SynthEventType.SET:
            #     data = event.data
            #     node.set(data["name"], data["new_value"])
        else:  # immutable synth
            if etype not in [SynthEventType.START, SynthEventType.STOP]:
                raise RuntimeError(
                    f"Immutable Synth should only produce {SynthEventType.START}"
                    f" and {SynthEventType.STOP} but got {etype}"
                )
            new_immutable_synth = backend_part
            # print(f"SynthEventHandlerPya: _handle_event: immutable start {event}")
            new_immutable_synth(event=event)

def pya_playbuf_synth(out=0, rate=1, pan=0, amp=0.3, asig=None):
    """pya synths are just functions to create an asig
    - a playbuf will just return the buf (which is already an asig)
    - but can modify the buffer as specified by other parameters (rate, amp, etc.)
    """
    # print("pya_playbuf_synth:", asig)
    assert type(asig) == Asig
    return asig.gain(amp).resample(target_sr=44100, rate=rate) # TODO Target rate replace

class SynthManagerPya(SynthManager):

    _backend: "BackendPya"

    buffer_synthdefs: Dict[str, function] = {
        "playbuf": pya_playbuf_synth,
    }

    def _create_event_handler(
        self, backend: "BackendPya", synths: WeakKeyDictionary[Synth]
    ) -> SynthEventHandler:
        return SynthEventHandlerPya(backend, synths)

    def create(
        self, name: str, track: int = 0, *, mutable: bool = False, **kwargs
    ) -> Synth:
        # check if synth name is known to pya
        synth_desc = PyaSynthDef.get_description(name)

        if synth_desc is None:
            raise ValueError(
                f"pya backend does not know a Synth with the name '{name}'"
            )
        # collect the parameters
        # Parameter(self, info.name, info.default)
        params_info: List[ParameterInfo] = []
        for param_name, info in synth_desc.items():
            params_info.append(ParameterInfo(param_name, info.default))

        # context_group: scnode.Group = self._backend.get_context_vars(self._context)[
        #     CONTEXT_GROUP
        # ]

        # create the backend part
        if mutable:  # create sc3nb Synth as backend part.
            # no mutable synths for pya -
            # print("SynthManagerPya::create in 'mutable synth'")
            ...
        else:  # immutable Synth -> create helper method as backend part
            # print("SynthManagerPya::create in 'immutable synth creation'")
            def new_immutable_synth(event: SynthEvent):
                # print("new_immutable_synth function executed for SynthEvent:", event)
                asig = PyaSynthDef.synth_defs[name](**event.data)
                if not self._context.is_realtime:  # FOLLOWING should happen only in nrt-mode:
                    self._backend.acanvas.x[{self._backend._current_timetag:None}] += asig.stereo()
                else:  # IN ENABLE_REALTIME, I'd prefer:
                    asig.play(onset=0.05) # or less latency, let's see...
                # print("backend pya: write code to add asig to acanvas at time t (known only later)")

            backend_part = new_immutable_synth

        # create the frontend / mesonic Synth
        synth = Synth(
            context=self._context,
            name=name,
            mutable=mutable,
            param_info=params_info,
            track=track,
        )

        # save the frontend/backend pair
        self._synths[synth] = backend_part
        return synth

    def _create_buffer_synthdef(self, pyabuffer: Asig, synth_name) -> str:
        try:
            synth_def_code = SynthManagerPya.buffer_synthdefs[synth_name]
        except KeyError as error:
            raise ValueError(
                f"'{synth_name}' is not a known buffer SynthDef pattern"
            ) from error
        
        # gap_values = {
        #     gap: fun(pyabuffer)
        #     for gap, fun in SynthManagerPya.buffer_synthdefs_slots.items()
        # }
        bufnum = pyabuffer.label

        synth_partial_fn = partial(synth_def_code, asig=pyabuffer)
        return (
            PyaSynthDef(
                name=f"pya_{synth_name}_{bufnum}", definition=synth_partial_fn
            ).add()
        )

    def from_buffer(
        self, buffer: Buffer, synth_name: str = "playbuf", **synth_kwargs
    ) -> Synth:
        buffer_manager = self._context.managers.get("buffers", None)
        if buffer_manager is None:
            raise RuntimeError("Context does not have a valid BufferManager.")
        pyabuffer = buffer_manager._buffers[buffer]  # this is the buffer Asig
        synth_name = self._create_buffer_synthdef(pyabuffer, synth_name)
        synth = self.create(synth_name, **synth_kwargs)
        synth.metadata.update({"from_buffer": buffer})
        return synth

### TH: CONTINUE HERE!!!

class RecordEventHandlerPya(EventHandlerPya, RecordEventHandler):
    def __init__(
        self,
        backend: "BackendPya",
        records: WeakKeyDictionary[Record],
    ):
        self.backend = backend
        self.records = records

    def _handle_event(self, event: Event):
        assert isinstance(event, RecordEvent), "wrong Event type"
        record = event.record
        # sc_recorder = self.records[record]
        if record.finished:
            warnings.warn("Record already finished.")
            return

        try:
            ...
            # TODO: use pattern matching when `python_requires = >= 3.10`
            # if event.etype is RecordEventType.START:
            #     if sc_recorder._state != screcorder.RecorderState.RECORDING:
            #         sc_recorder.start()
            # elif event.etype is RecordEventType.STOP:
            #     sc_recorder.stop()
            #     record._finished = True  # pylint: disable=W0212
            # elif event.etype is RecordEventType.PAUSE:
            #     sc_recorder.pause()
            # elif event.etype is RecordEventType.RESUME:
            #     sc_recorder.resume()
        except RuntimeError as error:
            warnings.warn(f"Illegal backend recorder state: {error}")


class RecordManagerPya(RecordManager):
    """RecordManager for pya"""

    def _create_event_handler(
        self, backend: "BackendPya", records: WeakKeyDictionary["Record", Any]
    ) -> RecordEventHandler:
        return RecordEventHandlerPya(backend, records)

    def create(
        self,
        path,
        track=0,
        nr_channels: int = 2,
        rec_header: str = "wav",
        rec_format: str = "int16",
        bufsize: int = 65536,
        **kwargs,
    ) -> "Record":
        print("Record pya - to be solved later")
        record = 0
        # sc_recorder = screcorder.Recorder(
        #     path=path,
        #     nr_channels=nr_channels,
        #     rec_header=rec_header,
        #     rec_format=rec_format,
        #     bufsize=bufsize,
        #     server=self._backend.sc.server,
        #     **kwargs,
        # )
        # record = Record(context=self._context, path=path, track=track)
        # self._records[record] = sc_recorder
        return record


# context data keys

# group for the context
CONTEXT_GROUP = "context_group"
# init hook id of the context group
CONTEXT_GROUP_HOOK = "context_group_hook"


class BackendPya(Backend):
    """Backend for pya"""


    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.aserver = startup(**kwargs)
        # todo: extract label, channels, sr from arguments and implant in next row
        self.acanvas = Asig(0, sr=44100, label="canvas", channels=2)
        self._current_timetag = 0

    @property
    def sampling_rate(self):
        return self.sc.server.nominal_sr

    def register_context(self, context: "Context") -> None:
        pass 
        # # Create a group for this context
        # main_group = scnode.Group(server=self.sc.server)
        # # Make sure that this group will be created at each server init.
        # main_group_hook = self.sc.server.add_init_hook(main_group.new)
        # # Save the objects as context variables.
        # self._contexts[context] = {
        #     CONTEXT_GROUP: main_group,
        #     CONTEXT_GROUP_HOOK: main_group_hook,
        # }

    def unregister_context(self, context: "Context") -> None:
        pass
        # # free context group and the hook
        # self._contexts[context][CONTEXT_GROUP].deep_free()
        # self.sc.server.remove_init_hook(self._contexts[context][CONTEXT_GROUP_HOOK])
        # # remove context from the context lists
        # del self._contexts[context]
        # if len(self._contexts) == 0:
        #     self.sc.exit()

    def stop(self, context: "Context") -> None:
        # clear server schedule (TODO: learn about 'free the context group'.
        self.aserver.stop()
        # self.sc.server.clear_schedule()
        # self._contexts[context][CONTEXT_GROUP].deep_free()

    def clear_acanvas(self):
        self.acanvas = Asig(0, label="acanvas", sr=44100, channels=2)

    def create_synth_manager(self, context: "Context") -> SynthManagerPya:
        return SynthManagerPya(self, context)

    def create_buffer_manager(self, context: "Context") -> BufferManagerPya:
        return BufferManagerPya(self, context)

    def create_record_manager(self, context: "Context") -> RecordManagerPya:
        return RecordManagerPya(self, context)

    def render_nrt(self, context: "Context", output_path, **backend_kwargs) -> None:
        assert (
            context.backend is self
        ), "Provided a Context that not belongs to this Backend"
        # Get the timeline
        timeline = context.timeline
        #print("output file is ignored as rendering is done into an Asig")

        # Determine the time offset so that negative timestamps are positiv
        time_offset = min(timeline.first_timestamp, 0)
        # synth defs do not need to be executed as they are merely functions
        # execute requrired functions and add Asigs to correct location
        for timepoint, events in timeline.to_dict().items():
            synth_events = [
                event for event in events if isinstance(event, SynthEvent)
            ]

            self._current_timetag = timepoint + time_offset

            context.synths.event_handler.handle(
                timepoint + time_offset, synth_events, reversed=False
            )
        time_end = timeline.end_time
        # ToDo: check if it is necessary to truncate acanvas to {0:time_end}

    def quit(self) -> None:
        # super().quit()
        # on SC3NB this is: self.sc.exit()
        Aserver.shutdown_default_server()
        # del(self.aserver)