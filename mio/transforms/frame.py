"""
Nodes that receive and emit frames
"""

from typing import Optional, TypedDict

import numpy as np

from mio.models.buffer import BufferHeader
from mio.models.data import Frame
from mio.models.pipeline import Transform


class MergeBuffersConfig(TypedDict):
    """Configuration for :class:`.MergeBuffers`"""

    width: int
    height: int


class MergeBuffersOutput(TypedDict):
    """Output returned by :meth:`.MergeBuffers.process`"""

    frame: Frame


class MergeBuffers(Transform):
    """
    Merge sequential frame buffers into a single frame
    """

    name = "merge-buffers"
    input_type = tuple[BufferHeader, np.ndarray]
    output_type = MergeBuffersOutput
    config: MergeBuffersConfig

    _headers: Optional[list[BufferHeader]] = None
    _buffers: Optional[list[np.ndarray]] = None
    _last_buffer_n: Optional[int] = None

    def start(self) -> None:
        """Init private containers"""
        self._headers = []
        self._buffers = []
        self._last_buffer_n = 0

    def process(self, header: BufferHeader, buffer: np.ndarray) -> Optional[MergeBuffersOutput]:
        """
        Receive a header/buffer pair. If the frame buffer count has cycled back to zero,
        merge into a completed frame
        """
        if header.frame_buffer_count == 0 and self._last_buffer_n >= 0:
            frame = np.concat(self._buffers).reshape((self.config["width"], self.config["height"]))
            headers = self._headers.copy()
            self._headers = [header]
            self._buffers = [buffer]
            return {"frame": Frame.model_construct(frame=frame, headers=headers)}
        else:
            self._last_buffer_n = header.frame_buffer_count
            self._headers.append(header)
            self._buffers.append(buffer)
            return None
