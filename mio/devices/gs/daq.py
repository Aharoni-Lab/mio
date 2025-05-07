import numpy as np
import multiprocessing as mp
from typing import Union, Optional
import queue

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
        # Set display size to 320x320 as specified
        # Target dimensions
        output_width = 320
        output_height = 320
        
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
                
                # Concatenate all buffer data
                frame_data = np.concatenate(frame_data, axis=0)
                
                try:
                    # Calculate bytes per row (3936 bits = 492 bytes)
                    bytes_per_row = 3936 // 8
                    
                    # Ensure we have enough data for 320 rows
                    total_needed_bytes = bytes_per_row * 320
                    
                    if len(frame_data) < total_needed_bytes:
                        # Pad with zeros if we don't have enough data
                        frame_data = np.pad(frame_data, (0, total_needed_bytes - len(frame_data)))
                    elif len(frame_data) > total_needed_bytes:
                        # Truncate if we have too much data
                        frame_data = frame_data[:total_needed_bytes]
                    
                    # Reshape into 320 rows of 492 bytes each
                    raw_frame = np.reshape(frame_data, (320, bytes_per_row))
                    
                    # Process each row to extract pixel data
                    # Final output will be 320 rows x 320 pixels with 10-bit depth
                    processed_frame = np.zeros((320, 320), dtype=np.uint16)
                    
                    for row_idx in range(320):
                        # Skip first 12 bytes (96 bits)
                        row_data = raw_frame[row_idx, 12:]
                        
                        # Process every 12 bits to extract 10-bit pixels
                        # Each row should have 320 pixels (3840 bits / 12 bits per pixel = 320 pixels)
                        for pixel_idx in range(320):
                            # Calculate byte position and bit offset
                            bit_pos = pixel_idx * 12  # Each pixel takes 12 bits (1 + 10 + 1)
                            byte_pos = bit_pos // 8
                            bit_offset = bit_pos % 8
                            
                            # Make sure we don't go out of bounds
                            if byte_pos + 2 >= len(row_data):
                                break
                                
                            # Extract bytes that contain our 12 bits
                            # We might need up to 3 bytes depending on the alignment
                            if bit_offset <= 4:  # If the 12 bits fit in 2 bytes
                                bytes_chunk = np.array(row_data[byte_pos:byte_pos+2], dtype=np.uint32)
                                value = (bytes_chunk[0] << 8) | bytes_chunk[1]
                                # Shift to align with start of our pattern
                                value = value >> bit_offset
                            else:  # Need 3 bytes
                                bytes_chunk = np.array(row_data[byte_pos:byte_pos+3], dtype=np.uint32)
                                value = (bytes_chunk[0] << 16) | (bytes_chunk[1] << 8) | bytes_chunk[2]
                                # Shift to align with start of our pattern
                                value = value >> bit_offset
                            
                            # Extract just the 10 bits (positions 1-10) from the [1][10 bits][0] pattern
                            pixel_value = (value >> 1) & 0x3FF  # 0x3FF = 1023 (10 bits)
                            
                            # Store in our output frame
                            processed_frame[row_idx, pixel_idx] = pixel_value
                    
                    # The processed_frame is now 320x320 with 10-bit pixel depth
                    
                    # Create 8-bit version for visualization if needed
                    vis_frame = (processed_frame / 1023.0 * 255).astype(np.uint8)
                    
                    # Update header with frame dimensions and depth
                    for header in header_list:
                        if hasattr(header, 'original_width'):
                            header.original_width = 320
                            header.original_height = 320
                            header.display_width = 320
                            header.display_height = 320
                            header.bit_depth = 10
                    
                except Exception as e:
                    locallogs.exception(
                        "Frame processing failed: %s. Replacing with zeros.",
                        e
                    )
                    # Fall back to empty frame
                    processed_frame = np.zeros((output_height, output_width), dtype=np.uint16)
                    vis_frame = np.zeros((output_height, output_width), dtype=np.uint8)
                
                # Store visualization frame for display
                self.current_vis_frame = vis_frame
                    
                try:
                    # Send the 10-bit processed frame to the queue
                    imagearray.put(
                        (processed_frame, header_list),
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
