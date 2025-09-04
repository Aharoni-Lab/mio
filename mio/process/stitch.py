"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


@dataclass
class StreamData:
    """Container for a single stream's data."""
    video_path: Path
    csv_path: Path
    video_cap: cv2.VideoCapture
    metadata: pd.DataFrame
    
    def __post_init__(self):
        self.video_cap = cv2.VideoCapture(str(self.video_path))
        if not self.video_cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
