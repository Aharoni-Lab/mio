# ruff: noqa: D100

from typing import Optional, Tuple, List

import numpy as np

from mio.devices.gs.config import GSDevConfig
from mio.devices.gs.header import GSBufferHeaderFormat
from mio.devices.gs import testing


def patterned_frame(width: int = 320, height: int = 328, pattern: str = "sequence") -> np.ndarray:
    """
    Create a frame for the naneye as a uint16 array with a testing pattern
    """

    # Create a base image (10-bit values)
    frame = np.zeros((height, width), dtype=np.uint16)

    # Generate the pattern
    if pattern == "sequence":
        frame = [np.arange(2**10, dtype=np.uint16)] * int(np.ceil((height * width) / (2**10)))
        frame = np.concatenate(frame)
        frame = frame[: (height * width)].reshape((height, width))
    if pattern == "cross":
        # Draw a horizontal line
        frame[height // 2 - 5 : height // 2 + 5, :] = 800
        # Draw a vertical line
        frame[:, width // 2 - 5 : width // 2 + 5] = 800
    elif pattern == "grid":
        # Draw horizontal lines
        for i in range(0, height, 40):
            frame[i : i + 5, :] = 800
        # Draw vertical lines
        for i in range(0, width, 40):
            frame[:, i : i + 5] = 800

    return frame


def pack_12bit_to_32bit_buffers(frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Pack 12-bit pixel data into 32-bit buffers.

    Args:
        frame: 2D numpy array of 12-bit values (stored as uint16)

    Returns:
        Tuple of (full_buffers, partial_buffer)
        - full_buffers: List of 10 arrays, each containing 3750 32-bit values
        - partial_buffer: Array containing 1860 32-bit values
    """
    # Flatten the frame to 1D array
    pixels = frame.flatten()

    # Ensure all values are within 12-bit range
    pixels = pixels & 0xFFF

    # Calculate total number of pixels
    total_pixels = len(pixels)
    expected_pixels = 328 * 320  # 104960 pixels

    if total_pixels != expected_pixels:
        raise ValueError(f"Expected {expected_pixels} pixels, got {total_pixels}")

    # Pack 12-bit values into 32-bit buffers
    # Each 32-bit value can hold 2.67 12-bit values, so we need to be careful with packing
    # We'll pack 8 12-bit values into 3 32-bit values (8*12 = 96 bits, 3*32 = 96 bits)

    packed_data = []
    pixel_idx = 0

    while pixel_idx < total_pixels:
        # Process 8 pixels at a time (if available)
        remaining_pixels = total_pixels - pixel_idx
        pixels_to_process = min(8, remaining_pixels)

        if pixels_to_process >= 8:
            # Pack 8 12-bit values into 3 32-bit values
            p = pixels[pixel_idx:pixel_idx + 8]

            # Pack into 3 32-bit values
            val1 = (p[0] & 0xFFF) | ((p[1] & 0xFFF) << 12) | ((p[2] & 0xFF) << 24)
            val2 = ((p[2] & 0xF00) >> 8) | ((p[3] & 0xFFF) << 4) | ((p[4] & 0xFFF) << 16) | ((p[5] & 0xF) << 28)
            val3 = ((p[5] & 0xFF0) >> 4) | ((p[6] & 0xFFF) << 8) | ((p[7] & 0xFFF) << 20)

            packed_data.extend([val1, val2, val3])
            pixel_idx += 8
        else:
            # Handle remaining pixels (pad with zeros if necessary)
            p = np.zeros(8, dtype=np.uint16)
            p[:pixels_to_process] = pixels[pixel_idx:pixel_idx + pixels_to_process]

            val1 = (p[0] & 0xFFF) | ((p[1] & 0xFFF) << 12) | ((p[2] & 0xFF) << 24)
            val2 = ((p[2] & 0xF00) >> 8) | ((p[3] & 0xFFF) << 4) | ((p[4] & 0xFFF) << 16) | ((p[5] & 0xF) << 28)
            val3 = ((p[5] & 0xFF0) >> 4) | ((p[6] & 0xFFF) << 8) | ((p[7] & 0xFFF) << 20)

            packed_data.extend([val1, val2, val3])
            pixel_idx += pixels_to_process

    # Convert to numpy array of 32-bit unsigned integers
    packed_array = np.array(packed_data, dtype=np.uint32)

    # Split into 10 full buffers of 3750 each and 1 partial buffer of 1860
    full_buffers = []
    for i in range(10):
        start_idx = i * 3750
        end_idx = start_idx + 3750
        full_buffers.append(packed_array[start_idx:end_idx])

    # Partial buffer
    partial_start = 10 * 3750
    partial_buffer = packed_array[partial_start:partial_start + 1860]

    return full_buffers, partial_buffer


def create_serialized_frame_data(width: int = 320, height: int = 328, pattern: str = "sequence") -> Tuple[
    List[np.ndarray], np.ndarray]:
    """
    Create serialized frame data with the specified buffer structure.

    Args:
        width: Frame width (default 320)
        height: Frame height (default 328)
        pattern: Pattern type ("sequence", "cross", or "grid")

    Returns:
        Tuple of (full_buffers, partial_buffer)
        - full_buffers: List of 10 arrays, each containing 3750 32-bit values
        - partial_buffer: Array containing 1860 32-bit values
    """
    # Generate the frame pattern
    frame = patterned_frame(width, height, pattern)

    # Pack into 32-bit buffers
    full_buffers, partial_buffer = pack_12bit_to_32bit_buffers(frame)

    return full_buffers, partial_buffer


def verify_buffer_structure(full_buffers: List[np.ndarray], partial_buffer: np.ndarray) -> None:
    """
    Verify that the buffer structure matches the requirements.
    """
    print(f"Number of full buffers: {len(full_buffers)}")
    print(f"Size of each full buffer: {[len(buf) for buf in full_buffers]}")
    print(f"Size of partial buffer: {len(partial_buffer)}")
    print(f"Total 32-bit values: {sum(len(buf) for buf in full_buffers) + len(partial_buffer)}")

    # Calculate expected total based on 12-bit packing
    total_pixels = 328 * 320  # 104960 pixels
    # 8 pixels pack into 3 32-bit values
    expected_32bit_values = (total_pixels * 3) // 8
    print(f"Expected 32-bit values for 12-bit packing: {expected_32bit_values}")


def frame_to_naneye_buffers(
    frame: Optional[np.ndarray] = None, buffer_size: int = 3750
) -> list[bytes]:
    """
    Convert a video frame to a series of naneye-formatted buffers.

    The frame is formatted such that...

    - Each pixel is 10-bit
    - Each pixel is flanked by a 0 and a 1, so `0xxxxxxxxxx1` for a total of 12 bits
    - Each row is preceded by a series of 8 "training" pixels - 0101010101, or 0x555,
      or 682 in decimal.

    The frame is split into buffers, where a :class:`.GSBufferHeader`
    in :class:`.GSBufferHeaderFormat` is prepended.
    """
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
    binarized = np.concatenate(
        [
            np.zeros((binarized.shape[0], 1), dtype=np.uint8),
            binarized,
            np.ones((binarized.shape[0], 1), dtype=np.uint8),
        ],
        axis=1,
    )

    # split into separate buffers
    split = [binarized[i : i + buffer_size, :] for i in range(0, binarized.shape[0], buffer_size)]

    # convert to bytes
    buffer_bytes = [np.packbits(arr.flatten()).tobytes() for arr in split]

    # create headers
    fmt = GSBufferHeaderFormat.from_id("gs-buffer-header")
    headers = [np.zeros(fmt.header_length, dtype=np.uint32) for _ in range(len(buffer_bytes))]
    for i in range(len(buffer_bytes)):
        headers[i][fmt.buffer_count] = i

    # concat preamble and dummy words and cast to bytes
    config = GSDevConfig.from_id("MSUS-test")
    preamble = config.preamble * config.dummy_words
    header_bytes = [preamble + h.view(np.uint8).tobytes() for h in headers]

    # combine header and pixel buffers
    buffers = [header + buffer for header, buffer in zip(header_bytes, buffer_bytes)]
    return buffers
