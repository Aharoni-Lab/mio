"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
This is still hardcoded around the StreamDevConfig metadata fields.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from mio.io import VideoReader, VideoWriter
from mio.logging import init_logger
from mio.models.stitch import FrameInfo
from mio.models.stream import StreamDevConfig
from mio.process.metadata_helper import make_combined_list
from mio.stream_daq import StreamDaq

logger = init_logger(name="stitch")


class RecordingData:
    """Class for a single stream's data (video + metadata)."""

    def __init__(
        self,
        video_path: Path,
        csv_path: Path,
        device_config: Optional[StreamDevConfig] = None,
    ) -> None:
        self.video_path: Path = video_path
        self.csv_path: Path = csv_path
        self._device_config: Optional[StreamDevConfig] = device_config
        self._daq: Optional[StreamDaq] = None
        self._buffer_npix: Optional[List[int]] = None
        self._video_reader: Optional[VideoReader] = None
        self._metadata: Optional[pd.DataFrame] = None

    @property
    def video_reader(self) -> VideoReader:
        """Get or create the video reader."""
        if self._video_reader is None:
            self._video_reader = VideoReader(str(self.video_path))
        return self._video_reader

    @property
    def metadata(self) -> pd.DataFrame:
        """Get or load metadata CSV as DataFrame."""
        if self._metadata is None:
            self._metadata = pd.read_csv(self.csv_path)
        return self._metadata

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
        # Use the FrameInfo class method which handles the metadata parsing
        return FrameInfo.from_metadata(frame_num=frame_num, metadata=self.metadata)


class RecordingDataBundle:
    """Class for a bundle of recording data."""

    def __init__(
        self,
        recordings: List[RecordingData],
        combined_video_writer: VideoWriter,
    ) -> None:
        self.recordings: List[RecordingData] = recordings
        self.combined_video_writer: VideoWriter = combined_video_writer
        self.combined_metadata: Optional[pd.DataFrame] = None
        self._combined_buffer_index: Optional[List[int]] = None
        self._combined_frame_num: Optional[List[int]] = None

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
            recording_frame_pairs = []

            for recording in self.recordings:
                if frame_num in recording.metadata["frame_num"].values:
                    frame_info = FrameInfo.from_metadata(
                        frame_num=frame_num, metadata=recording.metadata
                    )
                    recording_frame_pairs.append((recording, frame_info))

            # if there are multiple recordings with this frame_num, compare frames
            if len(recording_frame_pairs) > 1:
                try:
                    frames = []
                    for recording, frame_info in recording_frame_pairs:
                        # Use reconstructed_frame_index to get the correct frame
                        frame = recording.video_reader.read_frame(
                            frame_info.reconstructed_frame_index
                        )
                        if frame is not None:
                            frames.append(frame)

                    # check if frames are the same (only if we got multiple valid frames)
                    if len(frames) > 1:
                        if not all(np.array_equal(frames[0], frame) for frame in frames[1:]):
                            # Count differing pixels against the first frame (grayscale)
                            base = frames[0]
                            for idx, frame in enumerate(frames[1:], start=1):
                                if base.shape != frame.shape:
                                    logger.info(
                                        f"Frames differ for frame {frame_num}"
                                        f": shape {base.shape} vs {frame.shape}"
                                    )
                                    continue
                                diff_pixels = int(np.count_nonzero(base != frame))
                                logger.info(
                                    f"Frames are not the same for frame {frame_num} "
                                    f"(Rec {0} vs Rec {idx}): {diff_pixels} px differ"
                                )
                        else:
                            logger.debug(f"Frames are the same for frame {frame_num}")
                except Exception as e:
                    logger.debug(f"Error comparing frames for frame {frame_num}: {e}")


# script run for development
if __name__ == "__main__":
    recordings = [
        RecordingData(
            video_path=Path("user_data/stitch_test/stream1.avi"),
            csv_path=Path("user_data/stitch_test/stream1.csv"),
        ),
        RecordingData(
            video_path=Path("user_data/stitch_test/stream2.avi"),
            csv_path=Path("user_data/stitch_test/stream2.csv"),
        ),
    ]
    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        combined_video_writer=VideoWriter(path=Path("user_data/stitch_test/stitched.avi"), fps=20),
    )
    # list of imported recordings (video filenames)
    logger.info(f"Imported recordings: {[recording.video_path for recording in recordings]}")
    recording_bundle.stitch_recordings()
