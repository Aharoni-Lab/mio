import pytest
from mio.devices.gs import daq
import numpy as np
import matplotlib.pyplot as plt

from mio.devices.gs.daq import GSStreamDaq
from mio.devices.gs.header import GSBufferHeaderFormat, GSBufferHeader
from mio.devices.gs.config import GSDevConfig


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


def process_naneye_frame(unprocessed_frame):
    """
    Process a NanEyeC frame by removing training patterns and start/stop bits.
    
    Args:
        unprocessed_frame: Unprocessed frame with training patterns and start/stop bits
        
    Returns:
        Processed frame with 10-bit pixel data
    """
    height = len(unprocessed_frame) - 1  # Subtract 1 for EOF row
    processed_frame = np.zeros((height, 320), dtype=np.uint16)
    
    for row in range(height):
        row_data = unprocessed_frame[row]
        
        # Skip the 8 training pixels at the start of each row
        for col in range(8, 8 + 320):
            # Extract the 10 bits of pixel data (ignoring start and stop bits)
            pixel_bits = row_data[col][1:-1]  # Remove start/stop bits
            
            # Convert binary to integer
            pixel_value = sum(bit << (9 - i) for i, bit in enumerate(pixel_bits))
            processed_frame[row, col - 8] = pixel_value
    
    return processed_frame


def visualize_frames(unprocessed_frame, processed_frame):
    """
    Visualize the unprocessed and processed frames side by side.
    
    Args:
        unprocessed_frame: Unprocessed frame with training patterns and start/stop bits
        processed_frame: Processed frame with 10-bit pixel data
    """
    # Convert unprocessed frame to visualization format
    height = len(unprocessed_frame) - 1  # Subtract 1 for EOF row
    width = 328  # 8 training + 320 data
    unprocessed_vis = np.zeros((height, width), dtype=np.uint16)
    
    for row in range(height):
        for col in range(width):
            if col < len(unprocessed_frame[row]):
                # Convert 12-bit format to a value for visualization
                pixel_bits = unprocessed_frame[row][col]
                # We'll only use the 10 data bits for visualization
                data_bits = pixel_bits[1:-1]
                unprocessed_vis[row, col] = sum(bit << (9 - i) for i, bit in enumerate(data_bits))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display unprocessed frame
    im1 = ax1.imshow(unprocessed_vis, cmap='gray')
    ax1.set_title('Unprocessed Frame (with training patterns)')
    fig.colorbar(im1, ax=ax1)
    
    # Display processed frame
    im2 = ax2.imshow(processed_frame, cmap='gray')
    ax2.set_title('Processed Frame (10-bit pixel data)')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Modified test function to incorporate our new functions
def test_naneye_simulation():
    # Create a simulated NanEyeC frame with a cross pattern
    unprocessed_frame = create_naneye_frame(pattern="cross")
    
    # Process the frame
    processed_frame = process_naneye_frame(unprocessed_frame)
    
    # Visualize the frames
    fig = visualize_frames(unprocessed_frame, processed_frame)
    
    # Basic assertions to ensure the frame dimensions are correct
    assert len(unprocessed_frame) == 321  # 320 rows + 1 EOF row
    assert len(unprocessed_frame[0]) == 328  # 8 training + 320 data pixels
    assert processed_frame.shape == (320, 320)


# Let's also implement the original test function with our simulated data
def test_gs_format_frame():
    # Create sample input for the test
    unprocessed_frame = create_naneye_frame(width=10, height=10, pattern="cross")
    
    # Expected output would be the flattened version of the processed frame
    expected_output = process_naneye_frame(unprocessed_frame).flatten()
    
    # Test data
    sample_pair = {
        'input': unprocessed_frame,
        'output': expected_output
    }
    
    # Run test
    actual = daq._format_frame(sample_pair['input'])
    assert np.array_equal(actual, sample_pair['output'])

# unprocessed_frame = create_naneye_frame(pattern="cross")
#
# # Process the frame
# processed_frame = process_naneye_frame(unprocessed_frame)
#
# # Visualize the frames
# visualize_frames(unprocessed_frame, processed_frame)

sample_pairs = [
    {
        'input': [np.zeros(10) for _ in range(10)],
        'output': np.zeros((100,))
    }
]

@pytest.mark.parametrize('expected', sample_pairs)
def test_gs_format_frame(expected):
    actual = daq._format_frame(expected['input'])
    assert np.array_equal(actual, expected['output'])


def test_trim_camera_data_bit_level():
    raw_data = np.concatenate(create_naneye_frame(pattern="cross"))  # should be 1xn array, NOT a 12x(320*328) array

    cross_pic = GSStreamDaq.trim_camera_data_bit_level(raw_data)

    plt.imsave("cross_pic.png", cross_pic.reshape(320, 320))