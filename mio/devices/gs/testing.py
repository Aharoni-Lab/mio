import numpy as np

from mio.devices.gs.config import GSDevConfig
from mio.devices.gs.header import GSBufferHeaderFormat


def patterned_frame(width=320, height=320, pattern="sequence") -> np.ndarray:
    """
    Create a frame for the naneye as a uint16 array with a testing pattern
    """

    # Create a base image (10-bit values)
    frame = np.zeros((height, width), dtype=np.uint16)

    # Generate the pattern
    if pattern == "sequence":
        frame = [np.arange(2**10, dtype=np.uint16)] * int(np.ceil((height * width) / (2**10)))
        frame = np.concatenate(frame)
        frame = frame[:(height * width)].reshape((height,width))
    if pattern == "cross":
        # Draw a horizontal line
        frame[height//2-5:height//2+5, :] = 800
        # Draw a vertical line
        frame[:, width//2-5:width//2+5] = 800
    elif pattern == "grid":
        # Draw horizontal lines
        for i in range(0, height, 40):
            frame[i:i+5, :] = 800
        # Draw vertical lines
        for i in range(0, width, 40):
            frame[:, i:i+5] = 800

    return frame


def frame_to_naneye_buffers(frame: np.ndarray | None = None, n_buffers: int = 11) -> list[bytes]:
    if frame is None:
        frame = patterned_frame()

    height, width = frame.shape
    # add the training columns (10-bit `0101010101`, we'll pad all the pixels at once later)
    training = np.array([682] * (height * 8), dtype=np.uint16).reshape((height, 8))

    frame = np.concatenate([training, frame], axis=1)

    # convert uint16 to a (npix * 10) binary array
    # flatten frame, byteswap (so that the top end of the 16-bit number comes first)
    # i.e. so that 256 is 00000001 00000000 rather than 00000000 00000001
    # then reshape to 16-wide
    binarized = np.unpackbits(frame.flatten().byteswap().view(np.uint8), axis=0).reshape((-1, 16))

    # strip 6 leading 0's to get 10-bits
    binarized = binarized[:, -10:]
    # add padding bits
    binarized = np.concatenate([
        np.zeros((binarized.shape[0], 1), dtype=np.uint8),
        binarized,
        np.ones((binarized.shape[0], 1), dtype=np.uint8),
    ], axis=1)

    # split into separate buffers
    split = np.array_split(binarized, n_buffers)

    # convert to bytes
    buffer_bytes = [np.packbits(arr.flatten()).tobytes() for arr in split]

    # create headers
    fmt = GSBufferHeaderFormat.from_id('gs-buffer-header')
    headers = [np.zeros(fmt.header_length, dtype=np.uint32) for _ in range(n_buffers)]
    for i in range(n_buffers):
        headers[i][fmt.buffer_count] = i

    # concat preamble and dummy words and cast to bytes
    config = GSDevConfig.from_id("MSUS-test")
    preamble = config.preamble * config.dummy_words
    header_bytes = [preamble + h.view(np.uint8).tobytes() for h in headers]

    # combine header and pixel buffers
    buffers = [header + buffer for header, buffer in zip(header_bytes, buffer_bytes)]
    return buffers
