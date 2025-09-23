"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

from mio.process.metadata_helper import linearity_mse, make_combined_list
from mio.stream_daq import StreamDaq


@dataclass
class ReconstructedBufferList:
    """
    Container containing a single frame as lists of buffers, timestamps, and buffer frame indices
    """

    buffer_list: List[np.ndarray]
    timestamp_list: List[int]
    buffer_frame_index_list: List[int]


@dataclass
class RecordingData:
    """Container for a single stream's data."""

    video_path: Path
    csv_path: Path
    video_cap: cv2.VideoCapture
    metadata: pd.DataFrame
    _daq: StreamDaq = None
    _buffer_npix: List[int] = None

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
            It could make sense to just make buffer_npix an independent helper.
        """
        if self._daq is None:
            self._daq = StreamDaq(self.device_config)
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

    def get_frame_as_buffer_time_array(self, timestamp: int) -> ReconstructedBufferList:
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
        return ReconstructedBufferList(
            buffer_list=buffer_list,
            timestamp_list=timestamp_list,
            buffer_frame_index_list=buffer_frame_index_list,
        )


@dataclass
class RecordingDataBundle:
    """Container for a bundle of recording data."""

    recordings: List[RecordingData]
    _combined_buffer_index: List[int] = None
    combined_metadata: pd.DataFrame = None
    combined_video: List[np.ndarray] = None

    def __init__(self, recordings: List[RecordingData]):
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

    def stitch_recordings(self) -> None:
        """
        Stitch the videos together and store the result in the combined_metadata and combined_video
        """
        current_frame_index = -1

        for buffer_index in self.combined_buffer_index:
            if (
                self.recordings[0]
                .metadata[self.recordings[0].metadata["buffer_index"] == buffer_index][
                    "frame_index"
                ]
                .values
                == current_frame_index
            ):
                # skip if the frame index is the same as the previous one
                continue
            timestamp_lists = []
            for recording in self.recordings:
                if buffer_index in recording.metadata["buffer_index"].values:
                    timestamp_lists.append(
                        recording.metadata[recording.metadata["buffer_index"] == buffer_index][
                            "timestamp"
                        ].values
                    )
            # if timestamps don't match, find the best candidate
            if all(np.array_equal(timestamp_lists[0], ts) for ts in timestamp_lists):
                selected_timestamp = timestamp_lists[0][0]
            else:
                mse_list = []
                for recording in self.recordings:
                    if buffer_index in recording.metadata["buffer_index"].values:
                        ts = recording.metadata[recording.metadata["buffer_index"] == buffer_index][
                            "timestamp"
                        ].values
                        frame_indices = [recording.get_frame_index_from_timestamp(t) for t in ts]
                        mse = linearity_mse(frame_indices, 0)
                        mse_list.append((mse, ts[0]))
