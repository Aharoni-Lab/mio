from mio.devices.gs import testing
from mio.devices.gs.header import GSBufferHeaderFormat, GSBufferHeader
from mio.devices.gs.config import GSDevConfig
import numpy as np


def test_format_headers_synthetic():
    """We can split a buffer into a (header, 1D pixel array) pairs"""
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")

    frame = testing.patterned_frame(pattern="sequence")
    buffers = testing.frame_to_naneye_buffers(frame)

    for i, buffer in enumerate(buffers):
        header, pixels = GSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)

        # the only thing in out headers for now is the frame count, but we should have recovered that correctly
        assert isinstance(header, GSBufferHeader)
        assert header.buffer_count == i

        # all the buffers should just be a sequence of numbers from 0 to 2**10,
        # but they will still have the "training" columns, which are "682" in decimal numbers.
        # so we filter those out first.
        # but also since some pixels will *actually* be 682, we allow diffs to also be 2 for the skips
        pixels = pixels[pixels!=682]

        pix_diff = np.diff(pixels.astype(np.int32))
        assert all([diffed in (1, 2, -((2**10)-1)) for diffed in pix_diff])

def test_format_headers_raw(gs_raw_buffers):
    format = GSBufferHeaderFormat.from_id("gs-buffer-header")
    config = GSDevConfig.from_id("MSUS-test")

    # this is now the header and the RAW pixels
    for i, buffer in enumerate(gs_raw_buffers):
        header, pixels = GSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)

    # todo: confirm the structure of header and pixels (HINT: see test_format_frames)
    # compare to what you might know: pixel values are between [], or are they the same? Is the dropped buffer 0?
    # look at headers (from_header) and to_frame methods!
    # possibly look at the corner pixels (0, 255, 0, 255)
    # shorthand way of accessing code: error is in this part of the code, lets replicate it! Maybe parsing headers is wrong, and we
    # can parse the header to find out why its not working.
    # dont need to display the images in the test pythons, but maybe generate the .avi file from the binary input
