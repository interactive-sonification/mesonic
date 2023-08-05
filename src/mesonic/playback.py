import atexit
import logging
import math
import time
import warnings
from threading import Event, Thread
from typing import Optional, SupportsFloat

from mesonic.processor import BundleProcessor
from mesonic.timeline import TimeBundle, Timeline

_LOGGER = logging.getLogger(__name__)


def simple_formatwarning(message, category, filename, lineno, line=None):
    # ignore everything except the category and message
    return "%s: %s\n" % (category.__name__, message)


warnings.formatwarning = simple_formatwarning


def _sign(x: SupportsFloat) -> float:
    """Get a 1 with the sign of x.

    Parameters
    ----------
    x : math._SupportsFloatOrIndex
        A number.

    Returns
    -------
    float
        a 1 with the sign of x.
    """
    return math.copysign(1, x)


class Clock:
    """The Clock class provides the Time for the Playback."""

    sleep_time = 0.01
    """Clock sleeping time."""
    fast_mode = False
    time_source = time.perf_counter

    def __init__(self):
        self._start_time = time.time()
        self._start_ref = time.perf_counter()

    def time(self):
        """Get the current time."""
        return self._start_time + time.perf_counter() - self._start_ref

    def sleep(self):
        """Sleep according to sleep_time"""
        if not self.fast_mode:
            time.sleep(self.sleep_time)


class Playback:
    """A Playback for a Timeline.

    The Playback processes a Timeline by sending the contained Events to the processor
    with respect to the provided Clock instance.

    The Playback offers the user interactive control over the rate and also allows
    to jump back and forth.

    Parameters
    ----------
    timeline : Timeline
        The Timeline instance for this Playback.
    processor : BundleProcessor
        The processor that will receive the due Events.
    rate : float, optional
        The rate of the time, by default 1
    start : float, optional
        The starting time, by default 0
    end : Optional[float], optional
        The ending time, by default the timeline.end_time.
    loop : bool, optional
        Wether the Playback should loop from end to start, by default False
    loop_duration : float, optional
        The duration between the loop end and start, by default 1
    clock : Optional[Clock], optional
        The source of time, by default a TimeClock

    """

    def __init__(
        self,
        timeline: Timeline,
        processor: BundleProcessor,
        rate: float = 1.0,
        start: float = 0.0,
        end: Optional[float] = None,
        loop: bool = False,
        loop_duration: float = 1.0,
        clock: Optional[Clock] = None,
    ) -> None:
        self.clock = Clock() if clock is None else clock

        # _wait_time that the worker has before the getter methods timeout
        self._wait_time = 0.1
        self.grace_time = 0.1

        self.allowed_lateness = 0.5 * processor.latency
        self.sleep_factor = 0.25

        self.processor = processor
        self._timeline = timeline
        self._timeline_insert_time = 0
        self._next_bundle: Optional[TimeBundle] = None

        self.start_time = start
        self._end_time = end
        self._end_time_offset = None

        self.loop = loop
        self.loop_duration = loop_duration
        self._check_for_loop = False

        # event for stopping the worker thread
        self._stop_event = Event()
        atexit.register(self._stop_event.set)  # make sure to stop the thread

        # The following events will be set by the worker
        # and cleared by the user / setter methods

        # event for signaling that time was adjusted by worker
        self._jump_event_processed = Event()
        # event for signaling that time was reversed by worker
        self._rate_event_processed = Event()

        # init playback state
        self._init_playback(at=self.start_time, rate=rate)

    def __repr__(self) -> str:
        if self.running:
            return f"Playback(time={self.time}, rate={self.rate})"
        else:
            return "Playback(running=False)"

    def _init_playback(self, at: float, rate: float):
        """Init the Playback

        Parameters
        ----------
        at : float
            starting time
        rate : float
            initial rate
        """
        # create a new thread and make sure the stop event is cleared
        self._stop_event.clear()
        self._thread = Thread(target=self._worker, daemon=True)

        # mark events as processed
        self._rate_event_processed.set()
        self._jump_event_processed.set()

        # adjust time_offset
        self._time_offset = at

        # set current time references
        self._cur_timestamp = self.clock.time()
        self._time_start_ref = self._cur_timestamp
        self._time_section_ref = self._cur_timestamp
        self._run_time = 0
        self._section_time = 0
        self._time = 0

        self._update_rate(rate)
        self._paused_rate = rate

        # adjust other state info
        self._check_for_loop = False
        self._prev_bundle = None
        self._next_bundle, self._over_the_end = self._timeline.search_at(
            at, self._reversed
        )

    def _update_rate(self, rate: float):
        """Update the Playback rate and associated variables

        Parameters
        ----------
        rate : float
            The new rate.
        """
        self._rate = rate
        self._new_rate = None
        self._rate_sign = _sign(self._rate)
        self._reversed = self._rate_sign == -1.0

    def _wait_for_event(self, event: Event):
        """Wait for the event.

        Parameters
        ----------
        event : Event
            Event for that should be waited.

        Raises
        ------
        TimeoutError
            If waiting timed out.
        """
        if not event.wait(self._wait_time):
            raise TimeoutError(
                "Playback state was changed and has not adjusted yet."
                f" Is the thread running? running={self.running}"
            )

    @property
    def end_time(self) -> float:
        """Get the time where Playback will end processing.

        The Playback will end processing at this time.
        By default this will be the timeline.end_time.

        However if a value is set explicitly it will be used.
        To use the timeline.end_time again this should be set to None.

        See end_time_offset to learn how this is affected by it.

        Returns
        -------
        float
            The Time where the Playback will end processing.
        """
        if self._end_time is not None:
            return self._end_time
        elif self._end_time_offset is not None:
            try:
                # can raise ValueError when timeline is empty
                last_time = self._timeline.last_timestamp
            except ValueError:
                return self._end_time_offset
            else:
                return last_time + self._end_time_offset
        else:
            return self._timeline.end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def end_time_offset(self) -> Optional[float]:
        """Optional[float]: The offset after the last timeline timestamp.

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
    def end_time_offset(self, value: float):
        if self._end_time is not None:
            raise RuntimeError(
                "end_time is set manually, end_time_offset has no effect."
            )
        self._end_time_offset = value

    @property
    def loop_start(self) -> float:
        """float: The loop starting time with respect to reversed state"""
        return self.end_time if self._reversed else self.start_time

    @property
    def loop_end(self) -> float:
        """float: The loop starting time with respect to reversed state"""
        return self.start_time if self._reversed else self.end_time

    @property
    def time(self) -> float:
        """float: current Playback time.
        Only accessible while running."""
        if not self.running:
            raise RuntimeError("Playback must be running to get the time.")
        self._wait_for_event(self._jump_event_processed)
        return self._time

    @time.setter
    def time(self, value):
        if not self.running:
            raise RuntimeError("Playback must be running to set the time.")
        self._wait_for_event(self._jump_event_processed)
        self._new_time = value
        self._jump_event_processed.clear()  # mark as unprocessed

    @property
    def rate(self) -> float:
        """float: current Playback rate."""
        self._wait_for_event(self._rate_event_processed)
        return self._rate

    @rate.setter
    def rate(self, value: float):
        if value != self._rate:
            if self.running:
                if value == 0:
                    self._paused_rate = self._rate
                self._new_rate = value
                self._rate_event_processed.clear()
            else:  # if not running we can set it directly
                self._update_rate(value)

    @property
    def reversed(self) -> bool:
        """bool: True if the Playback is reversed."""
        self._wait_for_event(self._rate_event_processed)
        return self._reversed

    def reverse(self):
        """Reverse the Playback (set rate = -1 * rate)
        Note that this could need 2 sleep cycles as we access the rate property twice.
        """
        self.rate = -1.0 * self.rate

    @property
    def running(self) -> bool:
        """bool: True if the Playback is running."""
        return self._thread.is_alive() and not self._stop_event.is_set()

    def start(self, at: Optional[float] = None, rate: Optional[float] = None):
        """Start the Playback

        Parameters
        ----------
        at : Optional[float], optional
            starting time of the playback, by default self.start_time
        rate : Optional[float], optional
            starting rate of the playback, by default the last used rate

        """
        time_ = self.start_time if at is None else at
        rate_ = self._rate if rate is None else rate
        if not self._thread.is_alive():
            self._init_playback(time_, rate_)
            self._thread.start()
        else:
            self.time = time_
            self.rate = rate_

    def stop(self):
        """Stop the Playback.

        Raises
        ------
        RuntimeError
            If Playback is not running.
        """
        if not self._thread.is_alive():
            raise RuntimeError("Playback is not running and can't be stopped")
        self._stop_event.set()
        self._thread.join()

    def restart(self, at: Optional[float] = None, rate: Optional[float] = None):
        """Restart the Playback by trying to stop and then starting.

        Parameters
        ----------
        at : Optional[float], optional
            starting time of the playback, by default self.start_time
        rate : Optional[float], optional
            starting rate of the playback, by default the last used rate

        """
        try:
            self.stop()
        except RuntimeError:
            pass
        self.start(at=at, rate=rate)

    @property
    def paused(self) -> bool:
        """bool: True if this is paused (rate == 0)"""
        return self.rate == 0

    def pause(self):
        """Pause this Playback (set rate = 0).

        Raises
        ------
        RuntimeError
            If Playback is not running.

        """
        if not self.running:
            raise RuntimeError("Playback is not running and can't be paused")
        self.rate = 0

    def resume(self, rate=None):
        """Resume the Playback.

        Parameters
        ----------
        rate : _type_, optional
            _description_, by default None

        Raises
        ------
        RuntimeError
            If the Playback was not paused

        """
        if not self.paused:
            raise RuntimeError("Playback is not paused")
        self.rate = rate or self._paused_rate

    # Functions used by the Playback worker thread

    def _advance_bundle(self, just_reversed=False):
        """Advance from the current bundle to the next bundle.

        This will determine the next logical bundle with respect to the current
        playback direction.

        If the current bundle is already the last logical bundle then we stay at this
        bundle and set _over_the_end = True

        Parameters
        ----------
        just_reversed : bool, optional
            True if we just reversed, by default False
        """
        current_bundle = self._next_bundle
        _LOGGER.debug(
            "_advance_bundle: current bundle = %s%s",
            current_bundle,
            " (after reversed event)" if just_reversed else "",
        )
        assert current_bundle is not None, "Bundle is unexpectedly None"
        # special case for when reversed event was processed
        if just_reversed and self._over_the_end:
            # _over_the_end should mean that we are at head/tail
            assert (
                current_bundle is self._timeline.head
                or current_bundle is self._timeline.tail
            )
            _LOGGER.debug("_advance_bundle: special case, reverse after end")
            # we just reversed at the after the end of the timeline
            # => current bundle is now the next unexecuted bundle
            self._next_bundle = current_bundle
            self._over_the_end = False
            return
        # normal case
        next_bundle = current_bundle.prev if self._reversed else current_bundle.next
        if next_bundle is None and not self.loop:
            _LOGGER.debug("_advance_bundle: reached end, no loop")
            # we reached head or tail and stay there
            next_bundle = current_bundle
            self._over_the_end = True
        elif self.loop:
            # is the next bundle none or after the loop end?
            if next_bundle is None or (
                (next_bundle.timestamp - self.loop_end) * self._rate_sign >= 0
                and not just_reversed
                # after reversing the time condition is a false positive
            ):
                next_bundle, _ = self._timeline.search_at(
                    self.loop_start, self._reversed
                )
                self._over_the_end = False
                self._check_for_loop = True
                _LOGGER.debug("_advance_bundle: new loop bundle = %s", next_bundle)
        else:  # next_bundle is not current_bundle:
            if (
                self._over_the_end
                and next_bundle.timestamp
                < self._time - self.grace_time * self._rate_sign
            ):
                # if we are over the end and the next_bundle is not the current bundle
                # it is a freshly inserted bundle. We should check if we are already
                # behind the bundle time and keep over_the_end True if so.
                # However Bundles that are only grace_time old won't be skipped
                self._over_the_end = True
            else:
                self._over_the_end = False
            _LOGGER.debug(
                "_advance_bundle: updated bundle to = %s, over_the_end=%s%s",
                next_bundle,
                self._over_the_end,
                " after reversed event" if just_reversed else "",
            )
        self._next_bundle = next_bundle

    def _advance(self):
        """Make one processing step.

        This will calculate the times according to Clock time and
        the user interaction events from f.e. setting the rate.

        After this it will check if a Bundle from the Timeline must be
        send to the processor.
        """
        # process events and calculate the times
        self._process_events_and_adjust_time()

        # next are checks if next bundle is still correct and execute it if needed

        # check if timeline was empty and try to get the next bundle
        if self._next_bundle is None:
            self._next_bundle, self._over_the_end = self._timeline.search_at(
                self._time,
                self._reversed,
            )
            if (  # if the next bundle is in grace time don't skip it.
                self._next_bundle is not None
                and (self._time - self._next_bundle.timestamp) * self._rate_sign
                < self.grace_time
            ):
                self._over_the_end = False
        # check if timeline was reset
        elif self._timeline.is_empty():
            _LOGGER.debug("advance: timeline is empty")
            self._next_bundle = None
            self._over_the_end = False

        # check if timeline has a new bundle before our next bundle inserted
        if (
            self._next_bundle is not None
            and self._timeline.last_update_time > self._timeline_insert_time
        ):
            self._timeline_insert_time = self._timeline.last_update_time
            # get the bundle that should been our prev bundle
            prev = self._next_bundle.next if self._reversed else self._next_bundle.prev
            # check if we are not at the ends of the timeline
            # and that the prev was not executed the last time
            if (
                prev is not None
                and prev is not self._prev_bundle
                and (prev.timestamp - self._time) * self._rate_sign > 0 - 0.05
            ):
                self._next_bundle = prev

        # see if we now have a bundle to execute
        if self._next_bundle is not None:
            # calculate the time difference from now to the next bundle
            next_time_delta = (
                self._next_bundle.timestamp - self._time
            ) * self._rate_sign
            # if the time to the next bundle is equal or smaller zero
            # we should send it to the processor
            if next_time_delta <= 0:
                # execute bundle only if
                # - not already executed (over_the_end)
                # - when the bundle is between the start and end time (out_of_time_bounds)
                # - we are currently awaiting to reach the loop end
                # - we have a non zero rate
                in_time_bounds = (
                    self.start_time <= self._next_bundle.timestamp <= self.end_time
                )
                execute = (
                    not self._over_the_end
                    and in_time_bounds
                    and not self._check_for_loop
                    and self._rate != 0
                )
                if execute:
                    # get the scheduled time for the bundle as the difference of the
                    # timestamp of the bundle and the offset scaled by the rate
                    # and add the last saved section reference to get a timestamp
                    schedule_timestamp = (
                        self._time_section_ref
                        + (self._next_bundle.timestamp - self._time_offset)
                        * 1
                        / self._rate
                    )
                    self.processor.process_bundle(
                        self._next_bundle,
                        schedule_timestamp,
                        self._reversed,
                        pb_time=self._time,
                        pb_time_ref=self._time_start_ref,
                        pb_run_time=self._run_time,
                    )
                    self._prev_bundle = self._next_bundle
                _LOGGER.debug(
                    (
                        "advance: time %s has been reached of Bundle (%s),"
                        " executed=%s, over_the_end=%s, in_time_bounds=%s,"
                        " in_loop=%s"
                    ),
                    self._time,
                    self._next_bundle,
                    execute,
                    self._over_the_end,
                    in_time_bounds,
                    self._check_for_loop,
                )
                # set next bundle only when we are not preparing a loop
                # and have a non zero rate
                if not self._check_for_loop and self._rate != 0:
                    self._advance_bundle()

        if self._next_bundle and not self._over_the_end:
            next_time_delta = (
                self._next_bundle.timestamp - self._time
            ) * self._rate_sign
        else:
            next_time_delta = float("inf")
        return next_time_delta

    def _process_events_and_adjust_time(self):
        """Calculate the new time according to the Clock and the user interactions."""
        # save old timestamp
        old_timestamp = self._cur_timestamp
        # get current timestamp
        self._cur_timestamp = self.clock.time()
        # duration since last _advance = current - oldtimestamp
        scaled_sleep_duration = (self._cur_timestamp - old_timestamp) * self._rate

        # calculate the physical running time with the start time reference
        self._run_time = math.fsum([self._cur_timestamp, -self._time_start_ref])

        # calculate the current_time estimate
        # current time = old time +/- sleep duration
        current_time = self._time + scaled_sleep_duration

        # check if events are not set => they need to be processed
        rate_changed = not self._rate_event_processed.is_set()
        jumped = not self._jump_event_processed.is_set()

        # this would mean that we the time is behind the last executed bundle for
        # this playback is before the loop_end time and the next_bundle is in the
        # logical past.
        # We need to jump to the loop start time when we reach the end of the loop time
        # minus the loop duration

        # check if we need to loop
        if self._check_for_loop:
            # if a jump event occured we cancel the loop time
            time_until_loop_end = (self.loop_end - current_time) * self._rate_sign
            if jumped:
                self._check_for_loop = False
            elif time_until_loop_end <= 0:
                _LOGGER.debug(
                    (
                        "Playback: Adjusting time at %s because of loop"
                        " - loop_start=%s, loop_end=%s, loop_duration=%s"
                    ),
                    current_time,
                    self.loop_start,
                    self.loop_end,
                    self.loop_duration,
                )
                self._check_for_loop = False  # done checking for loop
                # set new_time and set jumped
                # new_time = loop start time - the loop duration
                # + scaled time_until_loop_end for correction
                correction = time_until_loop_end * (1 / self._rate)
                self._new_time = (
                    self.loop_start - self.loop_duration * self._rate_sign + correction
                )
                jumped = True

        if rate_changed or jumped:
            # when a event occurred a new section has begun.
            # and the offset must be updated
            self._time_section_ref = self._cur_timestamp

            direction_change = False
            if rate_changed:
                _LOGGER.debug("Rate change event (%s->%s)", self._rate, self._new_rate)
                # if sign of rate has changed => direction has changed
                direction_change = _sign(self._rate) != _sign(self._new_rate)

                # update rate
                self._update_rate(self._new_rate)

            # get the suitable offsets and update the current bundle
            if jumped:
                # calculate a new offset: simply set it to the new time
                self._time_offset = self._new_time
                # search for the new bundle at the new time
                self._next_bundle, self._over_the_end = self._timeline.search_at(
                    self._new_time, self._reversed
                )
                _LOGGER.debug(
                    "Time change event (->%s)",
                    self._new_time,
                )
            else:  # rate changed
                # the offset is now simpy the current_time
                self._time_offset = current_time

                # need to update the next bundle when direction has changed
                # and we don't jumped
                if direction_change and not self._timeline.is_empty():
                    self._advance_bundle(just_reversed=True)

            _LOGGER.debug(
                "Processed events: current_time=%s, offset=%s,"
                " jumped=%s, rate_changed=%s next_bundle=%s over_the_end=%s",
                current_time,
                self._time_offset,
                jumped,
                rate_changed,
                self._next_bundle,
                self._over_the_end,
            )

        # calculate new times
        # calculate the time of the current section
        self._section_time = math.fsum([self._cur_timestamp, -self._time_section_ref])
        # logical time = (section_time * rate) + time_offset
        self._time = math.fsum([self._section_time * self._rate, self._time_offset])

        # times and state calculated for this iteration
        # set event signals so other threads know
        # => mark events as processed
        if rate_changed:
            self._rate_event_processed.set()
        if jumped:
            self._jump_event_processed.set()

    def _worker(self):
        """Executed by the Playback worker thread.

        Until the stop event is set:
            - Advance one time period and then sleep.
        """
        last_print = 0
        while not self._stop_event.is_set():
            try:
                time_delta = self._advance()
            except Exception as exception:
                _LOGGER.error("Exception occured in Playback", exc_info=exception)
            else:
                # time_delta is the abs time to the next timebundle without latency and it is
                #  > 0 if ahead
                #  = 0 if punctual
                #  < 0 if late
                if time_delta >= self.clock.sleep_time * self.sleep_factor:
                    self.clock.sleep()
                elif time_delta <= -self.allowed_lateness and not self._check_for_loop:
                    now = self.clock.time()
                    if now - last_print > 0.5:
                        last_print = now
                        warnings.warn(f"Playback late {abs(time_delta)}")
