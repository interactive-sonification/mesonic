from typing import TYPE_CHECKING, NamedTuple, Tuple

if TYPE_CHECKING:
    from mesonic.context import Context


class BufferInfo(NamedTuple):
    """Information about a Buffer."""

    frame_count: int
    channel_count: int
    sr: int


class Buffer:
    """A Buffer represents a data section that is accessible by the Backend.

    This can be used to store samples and other audio or other data.

    Parameters
    ----------
    context : Context
        The Context of this Buffer.
    buffer_info : BufferInfo
        Information about the Buffer.

    """

    def __init__(self, context: "Context", buffer_info: BufferInfo):
        self._context = context
        self._buffer_info = buffer_info

    def __repr__(self) -> str:
        return "Buffer({} x {} @ {}Hz = {:.3f}s)".format(
            self.channel_count,
            self.sample_count,
            self.sr,
            self.duration,
        )

    @property
    def context(self) -> "Context":
        """Context: The Context of this Buffer."""
        return self._context

    @property
    def frame_count(self) -> int:
        """int: Number of frames in the Buffer."""
        return self._buffer_info.frame_count

    @property
    def channel_count(self) -> int:
        """int: Number of channels of the Buffer."""
        return self._buffer_info.channel_count

    @property
    def sample_count(self) -> int:
        """int: Number of samples (frames x channels) in the Buffer."""
        return self.channel_count * self.frame_count

    @property
    def shape(self) -> Tuple[int, int]:
        """Tuple[int, int]: Shape (channel_count, frame_count) of the Buffer."""
        return (self.channel_count, self.frame_count)

    @property
    def sr(self) -> int:
        """int: Sampling rate of the Buffer."""
        return self._buffer_info.sr

    @property
    def duration(self) -> float:
        """float: Duration in secounds (frame_count / sr) of the Buffer."""
        return self.frame_count / self.sr
