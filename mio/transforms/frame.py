from typing import ClassVar, TypedDict, Optional
from mio.models.pipeline import Transform, T, U
from mio.models.buffer import BufferHeader
from mio.models.data import Frame
import numpy as np


class MergeBuffersConfig(TypedDict):
    width: int
    height: int


class MergeBuffers(Transform):
    """
    Merge sequential frame buffers into a single frame
    """

    type_ = "merge-buffers"
    input_type = tuple[BufferHeader, np.ndarray]
    output_type = Optional[Frame]
    config: MergeBuffersConfig

    _headers: Optional[list[BufferHeader]] = None
    _buffers: Optional[list[np.ndarray]] = None
    _last_buffer_n: Optional[int] = None

    def start(self) -> None:
        """Init private containers"""
        self._headers = []
        self._buffers = []
        self._last_buffer_n = 0

    def process(self, header: BufferHeader, buffer: np.ndarray) -> Optional[Frame]:
        if header.frame_buffer_count == 0 and self._last_buffer_n >= 0:
            frame = np.concat(self._buffers).reshape((self.config["width"], self.config["height"]))
            headers = self._headers.copy()
            self._headers = []
            self._buffers = []
            return Frame.model_construct(frame=frame, headers=headers)
        else:
            self._last_buffer_n = header.frame_buffer_count
            self._headers.append(header)
            self._buffers.append(buffer)
            return None
