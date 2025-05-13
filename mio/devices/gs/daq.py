import numpy as np
import multiprocessing as mp
from typing import Union, Optional
import queue
import cv2

from pathlib import Path
from mio import init_logger
from mio.stream_daq import StreamDaq, exact_iter
from mio.devices.gs.config import GSDevConfig
from mio.devices.gs.header import GSBufferHeaderFormat
from mio.plots.headers import StreamPlotter
from mio.types import ConfigSource
from mio.io import BufferedCSVWriter

def _format_frame(frame_data: list[np.ndarray]) -> np.ndarray:
    return np.concat(frame_data)

class GSStreamDaq(StreamDaq):
    """Mystery scope daq"""
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

        self.logger = init_logger("streamDaq")
        self.config = GSDevConfig.from_any(device_config)
        self.header_fmt = GSBufferHeaderFormat.from_any(header_fmt)

        self.preamble = self.config.preamble
        self.terminate = mp.Event()

        self._buffer_npix: Optional[list[int]] = None
        self._nbuffer_per_fm: Optional[int] = None
        self._buffered_writer: Optional[BufferedCSVWriter] = None
        self._header_plotter: Optional[StreamPlotter] = None


    def _format_frame_inner(self, frame_data: list[np.ndarray]) -> np.ndarray:

        return super()._format_frame_inner(frame_data)

    def _format_frame(
        self,
        frame_buffer_queue: mp.Queue,
        imagearray: mp.Queue,
    ) -> None:
        """
        Construct frame from grouped buffers.

        Each frame data is concatenated from a list of buffers in `frame_buffer_queue`
        according to `buffer_npix`.
        If there is any mismatch between the expected length of each buffer
        (defined by `buffer_npix`) and the actual length, then the buffer is either
        truncated or zero-padded at the end to make the length appropriate,
        and a warning is thrown.
        Finally, the concatenated buffer data are converted into a 1d numpy array with
        uint8 dtype and put into `imagearray` queue.

        Parameters
        ----------
        frame_buffer_queue : mp.Queue[list[bytes]]
            Input buffer queue.
        imagearray : mp.Queue[np.ndarray]
            Output image array queue.
        """
        locallogs = init_logger("streamDaq.frame")
        try:
            for frame_data, header_list in exact_iter(frame_buffer_queue.get, None):

                if not frame_data or len(frame_data) == 0:
                    try:
                        imagearray.put(
                            (None, header_list),
                            block=True,
                            timeout=self.config.runtime.queue_put_timeout,
                        )
                    except queue.Full:
                        locallogs.warning("Image array queue full, skipping frame.")
                    continue

                try:
                    frame = self._format_frame_inner(frame_data)
                except ValueError as e:
                    expected_size = self.config.frame_width * self.config.frame_height
                    provided_size = frame_data.size
                    locallogs.exception(
                        "Frame size doesn't match: %s. "
                        " Expected size: %d, got size: %d."
                        "Replacing with zeros.",
                        e,
                        expected_size,
                        provided_size,
                    )
                    frame = np.zeros(
                        (self.config.frame_width, self.config.frame_height), dtype=np.uint8
                    )
                try:
                    imagearray.put(
                        (frame, header_list),
                        block=True,
                        timeout=self.config.runtime.queue_put_timeout,
                    )
                except queue.Full:
                    locallogs.warning("Image array queue full, skipping frame.")
        finally:
            locallogs.debug("Quitting, putting sentinel in queue")
            try:
                imagearray.put(None, block=True, timeout=self.config.runtime.queue_put_timeout)
            except queue.Full:
                locallogs.error("Image array queue full, Could not put sentinel.")
    


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