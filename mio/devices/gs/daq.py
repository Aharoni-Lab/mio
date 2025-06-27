import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from mio import init_logger
from mio.devices.gs.config import GSDevConfig
from mio.devices.gs.header import GSBufferHeader, GSBufferHeaderFormat
from mio.io import BufferedCSVWriter
from mio.plots.headers import StreamPlotter
from mio.stream_daq import StreamDaq
from mio.types import ConfigSource


def _format_frame(frame_data: list[np.ndarray]) -> np.ndarray:
    return np.concat(frame_data)


class GSStreamDaq(StreamDaq):
    """Mystery scope daq"""

    buffer_header_cls = GSBufferHeader

    def __init__(
        self,
        device_config: Union[GSDevConfig, ConfigSource],
        header_fmt: Union[GSBufferHeaderFormat, ConfigSource] = "gs-buffer-header",
    ) -> None:
        """
        Constructer for the class.
        This parses configuration from the input yaml file.

        Parameters
        ----------
        config : StreamDevConfig | Path
            DAQ configurations imported from the input yaml file.
            Examples and required properties can be found in /mio/config/example.yml

            Passed either as the instantiated config object or a path to on-disk yaml configuration
        header_fmt : MetadataHeaderFormat, optional
            Header format used to parse information from buffer header,
            by default `MetadataHeaderFormat()`.
        """

        self.logger = init_logger("GSStreamDaq")
        self.config = GSDevConfig.from_any(device_config)
        self.header_fmt = GSBufferHeaderFormat.from_any(header_fmt)

        self.preamble = self.config.preamble
        self.terminate = mp.Event()

        self._buffer_npix: Optional[list[int]] = None
        self._nbuffer_per_fm: Optional[int] = None
        self._buffered_writer: Optional[BufferedCSVWriter] = None
        self._header_plotter: Optional[StreamPlotter] = None

    def _format_frame_inner(self, frame_data: list[np.ndarray]) -> np.ndarray:
        # here, process the frame for Naneye camera
        # return super()._format_frame_inner(frame_data) # (super function refers to parent class)

        raw_data = np.concatenate(frame_data)  # concatenates to 1xn

        restructured_data = raw_data.reshape(12, 104960)
        restructured_data_trimmed = restructured_data[
            1:-1, :
        ]  # removes the top and bottom (start/stop bits) 12->10bit

        # go back to original reshape (and keep in separate methods)
        # Now create a mask for columns with the pattern you described
        # Create an array of all column indices
        all_indices = np.arange(104960)

        # Use modulo arithmetic to identify the pattern
        # For each 328-column chunk (8 skip + 320 keep):
        # - Column indices 0-7 in each chunk should be False (skip)
        # - Column indices 8-327 in each chunk should be True (keep)
        mask = (all_indices % 328) >= 8

        # Apply the mask to filter the data
        trimmed_data = restructured_data_trimmed[:, mask]  # now a 320x320x10
        eight_bit_data = (trimmed_data / 4).astype(np.uint8)
        frame = eight_bit_data.reshape(320, 320)

        return frame

    def _handle_frame(
        self,
        image: np.ndarray,
        header_list: list[GSBufferHeaderFormat],
        show_video: bool,
        writer: Optional[cv2.VideoWriter],
        show_metadata: bool,
        metadata: Optional[Path] = None,
    ) -> None:
        """
        Inner handler for :meth:`.capture` to process the frames from the frame queue.

        .. todo::

            Further refactor to break into smaller pieces, not have to pass 100 args every time.

        """
        if show_metadata or metadata:
            for header in header_list:
                if show_metadata:
                    self.logger.debug("Plotting header metadata")
                    try:
                        self._header_plotter.update(header)
                    except Exception as e:
                        self.logger.exception(f"Exception plotting headers: \n{e}")
                if metadata:
                    self.logger.debug("Saving header metadata")
                    try:
                        self._buffered_writer.append(
                            list(header.model_dump(warnings=False).values()) + [time.time()]
                        )
                    except Exception as e:
                        self.logger.exception(f"Exception saving headers: \n{e}")
        if image is None or image.size == 0:
            self.logger.warning("Empty frame received, skipping.")
            return
        if show_video:
            try:
                cv2.imshow("image", image)
                cv2.waitKey(1)
            except cv2.error as e:
                self.logger.exception(f"Error displaying frame: {e}")
        if writer:
            try:
                picture = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # If your image is grayscale
                writer.write(picture)
            except cv2.error as e:
                self.logger.exception(f"Exception writing frame: {e}")
