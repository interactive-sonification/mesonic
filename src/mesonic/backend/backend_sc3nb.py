"""Implementation of the backend using sc3nb."""
from __future__ import annotations

import tempfile
import warnings
from abc import abstractmethod
from functools import reduce
from operator import iconcat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Union
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
    from sc3nb.sc_objects.server import ServerOptions

import sc3nb.osc.osc_communication as scosc
import sc3nb.sc_objects.buffer as scbuf
import sc3nb.sc_objects.node as scnode
import sc3nb.sc_objects.recorder as screcorder
from sc3nb.sc import startup
from sc3nb.sc_objects.score import Score
from sc3nb.sc_objects.synthdef import SynthDef


class EventHandlerSC3NB(EventHandler):
    """A abstract EventHandler as basis sc3nb EventHandlers.

    Breaks down the handling of mutliple Events to just single events by using
    the sc3nb.Bundler.
    """

    backend: "BackendSC3NB"

    @abstractmethod
    def _handle_event(self, event: Event):
        ...

    def handle(self, time, events: Iterable[Event], reversed: bool, **kwargs) -> None:
        if time is None:
            time = 0
        with self.backend.sc.server.bundler(time):
            for event in events:
                if reversed:
                    event = event.reverse()
                self._handle_event(event)


class BufferManagerSC3NB(BufferManager):
    """BufferManager for sc3nb"""

    def _create_buffer(self, buf: scbuf.Buffer) -> Buffer:
        assert buf.allocated
        buf_info = BufferInfo(buf.samples, buf.channels, buf.sr)  # type: ignore
        buffer = Buffer(self._context, buf_info)
        self._buffers[buffer] = buf
        return buffer

    def from_data(self, data, sr, **kwargs) -> Buffer:
        buf = scbuf.Buffer(server=self._backend.sc.server).load_data(data, sr, **kwargs)
        return self._create_buffer(buf)

    def from_file(
        self,
        path,
        starting_frame: int = 0,
        num_frames: int = -1,
        channels: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> Buffer:
        buf = scbuf.Buffer(server=self._backend.sc.server).read(
            path, starting_frame, num_frames, channels, **kwargs
        )
        return self._create_buffer(buf)


class SynthEventHandlerSC3NB(EventHandlerSC3NB, SynthEventHandler):
    """SynthEventHandler for sc3nb"""

    def __init__(
        self, backend: "BackendSC3NB", synths: WeakKeyDictionary[Synth, scnode.Synth]
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
            node = backend_part
            # TODO use pattern matching when `python_requires = >= 3.10`
            if etype is SynthEventType.START:
                node.new(controls=event.data)
            elif etype is SynthEventType.STOP:
                node.free()
            elif etype is SynthEventType.PAUSE:
                node.run(False)
            elif etype is SynthEventType.RESUME:
                node.run(True)
            elif etype is SynthEventType.SET:
                data = event.data
                node.set(data["name"], data["new_value"])
        else:  # immutable synth
            if etype not in [SynthEventType.START, SynthEventType.STOP]:
                raise RuntimeError(
                    f"Immutable Synth should only produce {SynthEventType.START}"
                    f" and {SynthEventType.STOP} but got {etype}"
                )
            new_immutable_synth = backend_part
            new_immutable_synth(event=event)


class SynthManagerSC3NB(SynthManager):
    _backend: "BackendSC3NB"

    buffer_synthdefs: Dict[str, str] = {
        "playbuf": """
{ |out=0, bufnum={{BUFNUM}}, rate=1, loop=0, pan=0, amp=0.3 |
    var sig = PlayBuf.ar({{NUM_CHANNELS}}, bufnum,
        rate*BufRateScale.kr(bufnum),
        loop: loop,
        doneAction: Done.freeSelf);
    Out.ar(out, Pan2.ar(sig, pan, amp))
}""",
    }
    buffer_synthdefs_slots = {
        "NUM_CHANNELS": lambda scbuffer: scbuffer.channels,
        "BUFNUM": lambda scbuffer: scbuffer.bufnum,
    }

    sent_synthdefs = set()

    def _create_event_handler(
        self, backend: "BackendSC3NB", synths: WeakKeyDictionary[Synth, scnode.Synth]
    ) -> SynthEventHandler:
        return SynthEventHandlerSC3NB(backend, synths)

    def create(
        self, name: str, track: int = 0, *, mutable: bool = True, **kwargs
    ) -> Synth:
        # check if synth name is known to sc3nb
        synth_desc = SynthDef.get_description(name, self._backend.sc.lang)
        if synth_desc is None:
            raise ValueError(
                f"sc3nb backend does not know a Synth with the name '{name}'"
            )
        # collect the parameters
        # Parameter(self, info.name, info.default)
        params_info: List[ParameterInfo] = []
        for param_name, info in synth_desc.items():
            params_info.append(ParameterInfo(param_name, info.default))

        context_group: scnode.Group = self._backend.get_context_vars(self._context)[
            CONTEXT_GROUP
        ]

        # create the backend part
        if mutable:  # create sc3nb Synth as backend part.
            node = scnode.Synth(
                name,
                new=False,
                target=context_group.nodeid,
                server=self._backend.sc.server,
                **kwargs,
            )
            backend_part = node
        else:  # immutable Synth -> create helper method as backend part

            def new_immutable_synth(event: SynthEvent):
                msg = scosc.OSCMessage(
                    scnode.SynthCommand.NEW,
                    [
                        name,
                        -1,
                        scnode.AddAction.TO_HEAD.value,
                        context_group.nodeid,
                    ]
                    + reduce(iconcat, event.data.items(), []),
                )
                self._backend.sc.server.send(msg, bundle=True, await_reply=False)

            backend_part = new_immutable_synth

        # create the frontend / mesonic Synth
        synth = Synth(
            context=self._context,
            name=name,
            mutable=mutable,
            param_info=params_info,
            track=track,
        )

        # save the frontend backend pair
        self._synths[synth] = backend_part
        return synth

    def _create_buffer_synthdef(self, scbuffer: scbuf.Buffer, synth_name) -> str:
        try:
            synth_def_code = SynthManagerSC3NB.buffer_synthdefs[synth_name]
        except KeyError as error:
            raise ValueError(
                f"'{synth_name}' is not a known buffer SynthDef pattern"
            ) from error
        gap_values = {
            gap: fun(scbuffer)
            for gap, fun in SynthManagerSC3NB.buffer_synthdefs_slots.items()
        }
        identifier = f"sc3nb_{synth_name}_{scbuffer.bufnum}"
        if identifier not in SynthManagerSC3NB.sent_synthdefs:
            SynthDef(
                name=f"sc3nb_{synth_name}_{scbuffer.bufnum}", definition=synth_def_code
            ).set_contexts(gap_values).add()
            SynthManagerSC3NB.sent_synthdefs.add(identifier)
        return identifier

    def from_buffer(
        self, buffer: Buffer, synth_name: str = "playbuf", **synth_kwargs
    ) -> Synth:
        buffer_manager = self._context.managers.get("buffers", None)
        if buffer_manager is None:
            raise RuntimeError("Context does not have a valid BufferManager.")
        scbuffer = buffer_manager._buffers[buffer]
        synth_name = self._create_buffer_synthdef(scbuffer, synth_name)
        synth = self.create(synth_name, **synth_kwargs)
        synth.metadata.update({"from_buffer": buffer})
        return synth

    def add_buffer_synth_def(self, name, code=None, **kwargs):
        if code is None:
            raise ValueError('Must provide "code"')
        identifier_start = f"sc3nb_{name}"
        if name in SynthManagerSC3NB.buffer_synthdefs:
            SynthManagerSC3NB.sent_synthdefs = {
                synth_def
                for synth_def in SynthManagerSC3NB.sent_synthdefs
                if not synth_def.startswith(identifier_start)
            }
        SynthManagerSC3NB.buffer_synthdefs[name] = code

    def add_synth_def(self, name, code=None, **kwargs):
        if code is None:
            raise ValueError('Must provide "code"')
        SynthDef(name, code).add()


class RecordEventHandlerSC3NB(EventHandlerSC3NB, RecordEventHandler):
    def __init__(
        self,
        backend: "BackendSC3NB",
        records: WeakKeyDictionary[Record, screcorder.Recorder],
    ):
        self.backend = backend
        self.records = records

    def _handle_event(self, event: Event):
        assert isinstance(event, RecordEvent), "wrong Event type"
        record = event.record
        sc_recorder = self.records[record]
        if record.finished:
            warnings.warn("Record already finished.")
            return

        try:
            # TODO: use pattern matching when `python_requires = >= 3.10`
            if event.etype is RecordEventType.START:
                if sc_recorder._state != screcorder.RecorderState.RECORDING:
                    sc_recorder.start()
            elif event.etype is RecordEventType.STOP:
                sc_recorder.stop()
                record._finished = True  # pylint: disable=W0212
            elif event.etype is RecordEventType.PAUSE:
                sc_recorder.pause()
            elif event.etype is RecordEventType.RESUME:
                sc_recorder.resume()
        except RuntimeError as error:
            warnings.warn(f"Illegal backend recorder state: {error}")


class RecordManagerSC3NB(RecordManager):
    """RecordManager for sc3nb"""

    def _create_event_handler(
        self, backend: "BackendSC3NB", records: WeakKeyDictionary["Record", Any]
    ) -> RecordEventHandler:
        return RecordEventHandlerSC3NB(backend, records)

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
        sc_recorder = screcorder.Recorder(
            path=path,
            nr_channels=nr_channels,
            rec_header=rec_header,
            rec_format=rec_format,
            bufsize=bufsize,
            server=self._backend.sc.server,
            **kwargs,
        )
        record = Record(context=self._context, path=path, track=track)
        self._records[record] = sc_recorder
        return record


# context data keys

# group for the context
CONTEXT_GROUP = "context_group"
# init hook id of the context group
CONTEXT_GROUP_HOOK = "context_group_hook"


class BackendSC3NB(Backend):
    """Backend for sc3nb"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.sc = startup(**kwargs)

    @property
    def sampling_rate(self):
        return self.sc.server.nominal_sr

    def register_context(self, context: "Context") -> None:
        # Create a group for this context
        main_group = scnode.Group(server=self.sc.server)
        # Make sure that this group will be created at each server init.
        main_group_hook = self.sc.server.add_init_hook(main_group.new)
        # Save the objects as context variables.
        self._contexts[context] = {
            CONTEXT_GROUP: main_group,
            CONTEXT_GROUP_HOOK: main_group_hook,
        }

    def unregister_context(self, context: "Context") -> None:
        # free context group and the hook
        self._contexts[context][CONTEXT_GROUP].deep_free()
        self.sc.server.remove_init_hook(self._contexts[context][CONTEXT_GROUP_HOOK])
        # remove context from the context lists
        del self._contexts[context]
        if len(self._contexts) == 0:
            self.sc.exit()

    def stop(self, context: "Context") -> None:
        # clear server schedule and free the context group.
        self.sc.server.clear_schedule()
        self._contexts[context][CONTEXT_GROUP].deep_free()

    def create_synth_manager(self, context: "Context") -> SynthManagerSC3NB:
        return SynthManagerSC3NB(self, context)

    def create_buffer_manager(self, context: "Context") -> BufferManagerSC3NB:
        return BufferManagerSC3NB(self, context)

    def create_record_manager(self, context: "Context") -> RecordManagerSC3NB:
        return RecordManagerSC3NB(self, context)

    def render_nrt(
        self,
        context: "Context",
        output_path,
        options: Optional["ServerOptions"] = None,
        **backend_kwargs,
    ) -> None:
        assert (
            context.backend is self
        ), "Provided a Context that not belongs to this Backend"
        # Get the timeline
        timeline = context.timeline

        # Determine the time offset so that negative timestamps are positiv
        time_offset = min(timeline.first_timestamp, 0)
        with self.sc.server.bundler(0, send_on_exit=False) as bundler:
            # Add required SynthDefs to this Bundler
            for synth_def_name in context.synths.names:
                try:
                    synth_def_blob = SynthDef.synth_defs[synth_def_name]
                except KeyError as e:
                    raise RuntimeError(
                        "The SynthDef {synth_def} is not saved in SynthDef.synth_defs"
                        " as blob. Has it been added using SynthDef.add ?"
                    ) from e
                SynthDef.send(synth_def_blob, server=context.backend.sc.server)
            # Add the context group (parent of all the Synths) to this Bundler
            self._contexts[context][CONTEXT_GROUP].new(target=0)
            # Add the SynthEvents from the timeline to this Bundler
            for timepoint, events in timeline.to_dict().items():
                synth_events = [
                    event for event in events if isinstance(event, SynthEvent)
                ]
                context.synths.event_handler.handle(
                    timepoint + time_offset, synth_events, reversed=False
                )
        # Add end message to Bundler with the end_time of the Timeline
        bundler.add(timeline.end_time, "/c_set", [0, 0])

        # use a tempfile for the .osc file
        with tempfile.TemporaryDirectory() as osc_path:
            osc_file = Path(osc_path) / "nrtscore.osc"
            # Use the sc3nb.Score class with the bundler messages to render
            Score.record_nrt(
                bundler.messages(),
                osc_file.as_posix(),
                output_path,
                options=options,
                **backend_kwargs,
            )

    def quit(self) -> None:
        super().quit()
        self.sc.exit()
