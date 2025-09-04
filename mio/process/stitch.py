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


@dataclass
class RecordingData:
    """Container for a single stream's data."""

    video_path: Path
    csv_path: Path
    video_cap: cv2.VideoCapture
    metadata: pd.DataFrame

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

    def get_frame_from_timestamp(self, timestamp: int) -> np.ndarray:
        """
        Get the frame from the timestamp
        """
        frame_index = self.get_frame_index_from_timestamp(timestamp)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_cap.read()
        return frame
    
@dataclass
class RecordingDataBundle:
    """Container for a bundle of recording data."""

    recordings: List[RecordingData]
    timestamp_list: List[int] = None
    combined_metadata: pd.DataFrame = None
    combined_video: cv2.VideoCapture = None

    def __init__(self, recordings: List[RecordingData]):
        self.recordings = recordings
        self.timestamp_list = self._make_combined_buffer_timestamp_list()

    def _make_combined_buffer_timestamp_list(self) -> List[int]:
        """
        Make a list of unique timestamps from all the recordings
        """
        timestamp_list = list(
            set([recording.metadata["timestamp"].iloc[0] for recording in self.recordings])
        )
        timestamp_list.sort()
        return timestamp_list

    def stitch_recordings(self) -> None:
        """
        Stitch the videos together and store the result in the combined_metadata and combined_video
        """
        pass
