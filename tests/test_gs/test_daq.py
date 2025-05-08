import pytest
from mio.devices.gs import daq
import numpy as np


# create a pattern
import matplotlib.pyplot as plt

def create_naneye_frame(width=320, height=320, pattern="cross"):
    """
    Create a simulated NanEyeC camera frame with a specified pattern.
    
    Args:
        width: Width of the image (default: 320)
        height: Height of the image (default: 320)
        pattern: The pattern to generate ('cross', 'grid', etc)
        
    Returns:
        Array containing the unprocessed frame with training patterns and start/stop bits
    """
    # Initialize the full frame with training patterns and EOF patterns
    # Each line starts with 8 training PPs (0x555)
    # Full frame consists of:
    # - 320 rows, each with:
    #    - 8 training pixels (each 12 bits with pattern 0x555)
    #    - 320 data pixels (each 12 bits with 10-bit data wrapped with start/stop bits)
    # - 8 EOF (end of frame) pixels (each 12 bits with pattern 0x000)
    
    # Initialize the frame with zeros
    unprocessed_frame = []
    
    # Training pattern (0x555) in 12-bit format
    # Start bit (0) + 0101 0101 01 + Stop bit (1)
    training_pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # EOF pattern (0x000) in 12-bit format
    # Start bit (0) + 0000 0000 00 + Stop bit (0)
    eof_pattern = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Create a base image (10-bit values)
    base_image = np.zeros((height, width), dtype=np.uint16)
    
    # Generate the pattern
    if pattern == "cross":
        # Draw a horizontal line
        base_image[height//2-5:height//2+5, :] = 800
        # Draw a vertical line
        base_image[:, width//2-5:width//2+5] = 800
    elif pattern == "grid":
        # Draw horizontal lines
        for i in range(0, height, 40):
            base_image[i:i+5, :] = 800
        # Draw vertical lines
        for i in range(0, width, 40):
            base_image[:, i:i+5] = 800
    
    # Create the frame row by row
    for row in range(height):
        row_data = []
        
        # Add 8 training pixels at the start of each row
        for _ in range(8):
            row_data.append(training_pattern.copy())
        
        # Add the actual pixel data
        for col in range(width):
            # Get 10-bit pixel value
            pixel_value = base_image[row, col]
            # Convert to binary (10 bits)
            pixel_bits = [(pixel_value >> bit) & 1 for bit in range(9, -1, -1)]
            # Add start bit (1) and stop bit (0)
            pixel_with_start_stop = [1] + pixel_bits + [0]
            row_data.append(pixel_with_start_stop)
        
        unprocessed_frame.append(row_data)
    
    return unprocessed_frame


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

unprocessed_frame = create_naneye_frame(pattern="cross")

# Process the frame
processed_frame = process_naneye_frame(unprocessed_frame)

# Visualize the frames
visualize_frames(unprocessed_frame, processed_frame)

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


