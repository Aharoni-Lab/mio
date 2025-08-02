from collections import defaultdict
from mio.devices.gs.daq import format_frame
from mio.devices.gs.testing import patterned_frame, frame_to_naneye_buffers, create_serialized_frame_data
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


def test_format_headers_raw(gs_raw_buffers):
    """
    Use the fixtures and previously recorded .bin files to test the format_headers method.
    """
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")


    frame_buffers = defaultdict(list)
    for i, buffer in enumerate(gs_raw_buffers):
        # this extracts header and pixels
        header, pixels = GSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)
        # add the pixels value to a list of buffers
        frame_buffers[header.frame_num].append(pixels)

    # discard first and last which may be incomplete in the sample
    frames = list(frame_buffers.keys())
    del frame_buffers[frames[0]]
    del frame_buffers[frames[-1]]

    for pixel_arrays in frame_buffers.values():
        reconstructed = format_frame(pixel_arrays, config)
        assert reconstructed.shape == (config.frame_height, config.frame_width)
