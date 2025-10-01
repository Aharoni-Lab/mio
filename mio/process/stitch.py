"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from mio.io import VideoWriter
from mio.logging import init_logger
from mio.models.stream import StreamDevConfig
from mio.process.metadata_helper import make_combined_list
from mio.stream_daq import StreamDaq

logger = init_logger(name="stitch")


class BufferInfo(BaseModel):
    """
    Container containing information about a single buffer.
    This container is oriented around the buffer receive index.
    """

    buffer_recv_index: int
    buffer_count: int
    frame_buffer_count: int
    timestamp: int
    pixel_count: int
    black_padding_px: int
    buffer_recv_unix_time: float
    pixel_data: Optional[np.ndarray] = None


class FrameInfo(BaseModel):
    """
    Container containing information about a single frame.
    This container is oriented around the reconstructed frame index.
    """

    reconstructed_frame_index: int
    frame_num: int

    buffer_info_list: List[BufferInfo] = []

    def __init__(self, frame_num: int, metadata: pd.DataFrame):
        self.frame_num = frame_num
        # Find all buffer entries for this frame_num
        frame_metadata = metadata[metadata["frame_num"] == self.frame_num]

        if frame_metadata.empty:
            raise ValueError(f"No metadata found for frame_num {self.frame_num}")

        if all(
            frame_metadata["reconstructed_frame_index"].iloc[0] == x
            for x in frame_metadata["reconstructed_frame_index"]
        ):
            self.reconstructed_frame_index = frame_metadata["reconstructed_frame_index"].iloc[0]
        else:
            # Get the majority reconstructed_frame_index
            self.reconstructed_frame_index = frame_metadata["reconstructed_frame_index"].mode()[0]
            logger.warning(
                f"Reconstructed frame index is not the same "
                f"for all buffers in frame {self.frame_num}. "
                f"Using the majority reconstructed_frame_index: {self.reconstructed_frame_index}"
            )

        self.buffer_info_list = []

        # Iterate through all buffer entries for this frame
        # sort based on buffer_recv_index
        frame_metadata = frame_metadata.sort_values(by="buffer_recv_index")
        for i in range(len(frame_metadata)):
            self.buffer_info_list.append(
                BufferInfo(
                    buffer_recv_index=frame_metadata["buffer_recv_index"].iloc[i],
                    buffer_count=frame_metadata["buffer_count"].iloc[i],
                    frame_buffer_count=frame_metadata["frame_buffer_count"].iloc[i],
                    timestamp=frame_metadata["timestamp"].iloc[i],
                    pixel_count=frame_metadata["pixel_count"].iloc[i],
                    black_padding_px=frame_metadata["black_padding_px"].iloc[i],
                    buffer_recv_unix_time=frame_metadata["buffer_recv_unix_time"].iloc[i],
                )
            )


@dataclass
class RecordingData:
    """Container for a single stream's data."""

    video_path: Path
    csv_path: Path
    video_cap: cv2.VideoCapture
    metadata: pd.DataFrame
    _daq: StreamDaq = None
    _buffer_npix: List[int] = None
    _device_config: Optional[StreamDevConfig] = None

    def __post_init__(self):
        self.video_cap = cv2.VideoCapture(str(self.video_path))
        if not self.video_cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.metadata = pd.read_csv(self.csv_path)

    def get_frame_index_from_timestamp(self, timestamp: int) -> int:
        """
        Get the frame index from the timestamp
        """
        if timestamp not in self.metadata["timestamp"].values:
            raise ValueError(f"Timestamp {timestamp} not found in metadata")
        return self.metadata[self.metadata["timestamp"] == timestamp]["frame_index"].iloc[0]

    @property
    def daq(self) -> StreamDaq:
        """
        Get the stream daq.

        .. todo::
            Re-think this, though it is probablynot critical.
            We just need the buffer_npix list to reconstruct the buffers from the frame.
            It could make sense to just make buffer_npix static method on StreamDaq.
        """
        if self._daq is None:
            self._daq = StreamDaq(self._device_config)
        return self._daq

    @property
    def buffer_npix(self) -> List[int]:
        """
        Get the buffer npix.
        """
        if self.daq is None:
            raise ValueError("StreamDaq is not initialized")
        if self._buffer_npix is None:
            self._buffer_npix = self.daq.buffer_npix
        return self._buffer_npix

    def get_frame_from_timestamp(self, timestamp: int) -> np.ndarray:
        """
        Get the frame from the timestamp
        """
        frame_index = self.get_frame_index_from_timestamp(timestamp)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _, frame = self.video_cap.read()
        return frame

    def get_buffer_metadata_from_frame_index(self, frame_index: int) -> List[int]:
        """
        Get the timestamps and buffer frame indices from the frame index.
        """
        timestamp_list = []
        buffer_frame_index_list = []
        for i in range(len(self.buffer_npix)):
            timestamp_list.append(
                self.metadata[self.metadata["frame_index"] == frame_index]["timestamp"].iloc[i]
            )
            buffer_frame_index_list.append(
                self.metadata[self.metadata["frame_index"] == frame_index][
                    "buffer_frame_index"
                ].iloc[i]
            )
        return timestamp_list, buffer_frame_index_list

    def get_frame_as_buffer_time_array(self, timestamp: int) -> FrameInfo:
        """
        Get the frame as a list of buffers and a list of timestamps.

        The buffers are reconstructed from the frame using the buffer_npix list.

        .. todo::
            Handle missing buffers.
        """
        frame_index = self.get_frame_index_from_timestamp(timestamp)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_cap.read()
        buffer_list = []
        timestamp_list = []
        buffer_frame_index_list = []
        timestamp_list, buffer_frame_index_list = self.get_buffer_metadata_from_frame_index(
            frame_index
        )
        for i in range(len(self.buffer_npix)):
            buffer_list.append(frame[i * self.buffer_npix[i] : (i + 1) * self.buffer_npix[i]])
        return FrameInfo(
            buffer_list=buffer_list,
            timestamp_list=timestamp_list,
            buffer_frame_index_list=buffer_frame_index_list,
        )

    def get_frame_info_from_frame_num(self, frame_num: int) -> FrameInfo:
        """
        Get the frame info from the frame num.
        """
        # Get the FrameInfo from the frame num
        frame_info = self.metadata[self.metadata["frame_num"] == frame_num]
        buffer_info_list = []
        for i in range(len(self.buffer_npix)):
            buffer_info_list.append(
                BufferInfo(
                    buffer_recv_index=frame_info["buffer_recv_index"].iloc[i],
                )
            )
        return FrameInfo(
            reconstructed_frame_index=frame_info["reconstructed_frame_index"].iloc[0],
            frame_num=frame_info["frame_num"].iloc[0],
        )


@dataclass
class RecordingDataBundle:
    """Container for a bundle of recording data."""

    recordings: List[RecordingData]
    _combined_buffer_index: List[int] = None
    _combined_frame_num: List[int] = None
    combined_metadata: pd.DataFrame = None
    combined_video_writer: Optional[VideoWriter] = None

    def __init__(self, recordings: List[RecordingData], path: Union[Path, str], fps: int):
        self.combined_video_writer = VideoWriter(path=Path(path), fps=fps)
        self.recordings = recordings

    @property
    def combined_buffer_index(self) -> List[int]:
        """
        Get the combined buffer index.
        This is a list of unique buffer indices across all recordings.
        """
        if self._combined_buffer_index is None:
            self._combined_buffer_index = make_combined_list(
                [recording.metadata["buffer_index"].tolist() for recording in self.recordings]
            )
        return self._combined_buffer_index

    @property
    def combined_frame_num(self) -> List[int]:
        """
        Get the combined frame_num.
        This is a list of unique frame_nums across all recordings.
        """
        if self._combined_frame_num is None:
            self._combined_frame_num = make_combined_list(
                [recording.metadata["frame_num"].tolist() for recording in self.recordings]
            )
        return self._combined_frame_num

    def stitch_recordings(self) -> None:
        """
        Stitch the videos together and store the result in the combined_metadata and combined_video
        """
        for frame_num in self.combined_frame_num:
            frame_info_list = []
            for recording in self.recordings:
                if frame_num in recording.metadata["frame_num"].values:
                    frame_info_list.append(recording.get_frame_info_from_frame_num(frame_num))
            self.combined_metadata = pd.concat(frame_info_list)
