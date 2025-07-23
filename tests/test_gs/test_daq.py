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




from mio.devices.gs.testing import create_serialized_frame_data, verify_buffer_structure
def test_12_bit_formatted_frames():
    """
    Our format frame method should receive the 1-dimensional pixel buffers
    processed by parsing the headers (tested separately in `test_header`),
    and reassemble it to the original frame.
    """
    full_buffers, partial_buffer = create_serialized_frame_data(pattern="cross")
    verify_buffer_structure(full_buffers, partial_buffer)

    # Show some sample data
    print(len(full_buffers), len(partial_buffer))
    print(f"First few values of buffer 0: {full_buffers[0][:5]}")
    print(f"First few values of partial buffer: {partial_buffer[:5]}")

from mio.devices.gs.config import GSDevConfig
from mio.devices.gs.header import GSBufferHeader, GSBufferHeaderFormat

def test_image_decoder():
    """
    Test the decoding process of an image using various decoders and verify
    the output compared to expected results.

    Raises:
        AssertionError: If decoded output or process validation fails.
    """
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")

    frame = patterned_frame(width=config.frame_width, height=config.frame_height, pattern="sequential")
    buffers = frame_to_naneye_buffers(frame)

    processed = [GSBufferHeader.from_buffer(buf, header_fmt=format, config=config) for buf in buffers]
    pixels = [p[1] for p in processed]

    reconstructed = format_frame(pixels, config)
    assert np.array_equal(frame, reconstructed)

from mio.devices.gs.header import GSBufferHeaderFormat, GSBufferHeader
from mio.devices.gs.config import GSDevConfig
def test_format_headers_raw(gs_raw_buffers):

    """
    Use the fixutres and previously recorded .bin files to test the format_headers method.
    """
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")
    full_buffer_data_length = 3750
    partial_buffer_data_length = 1860

    size_of_word = 32
    device_px_bitdepth = 12
    list_of_pixels = [];
    for i, buffer in enumerate(gs_raw_buffers):
        # this extracts header and pixels
        header, pixels = GSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)
        # add the pixels value to a list of buffers
        list_of_pixels.append(pixels)
        # print(header, (pixels))
    reconstructed = format_frame(list_of_pixels, config)

        # breakpoint()
