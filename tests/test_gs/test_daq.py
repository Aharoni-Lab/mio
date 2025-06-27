from mio.devices.gs.daq import format_frame
from mio.devices.gs.testing import patterned_frame, frame_to_naneye_buffers
from mio.devices.gs.header import GSBufferHeaderFormat, GSBufferHeader
from mio.devices.gs.config import GSDevConfig
import numpy as np

def test_format_frames():
    """
    Our format frame method should receive the 1-dimensional pixel buffers
    processed by parsing the headers (tested separately in `test_header`),
    and reassemble it to the original frame.
    """
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")

    frame = patterned_frame(width=config.frame_width, height=config.frame_height, pattern="sequential")
    buffers = frame_to_naneye_buffers(frame)

    processed = [GSBufferHeader.from_buffer(buf, header_fmt=format, config=config) for buf in buffers]
    pixels = [p[1] for p in processed]

    reconstructed = format_frame(pixels, config)
    assert np.array_equal(frame, reconstructed)