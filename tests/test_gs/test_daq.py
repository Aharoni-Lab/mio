import pytest
from mio.devices.gs import daq
import numpy as np


def create_naneye_sim_frame(num_lines=320):
    """
    Create a simulated raw NanEyeC camera frame in SEIM mode.
    
    Returns:
        list: A list of numpy arrays, where each array represents a line in the frame
    """
    # Create list to hold lines
    frame_lines = []
    
    # Training pattern (0x555 = 0b010101010101)
    training_pattern = 0x555
    
    # End of frame pattern (0x000 = 0b000000000000)
    eof_pattern = 0x000
    
    # Generate pixel data for each line
    for line in range(num_lines):
        # Each line has 8 training PPs + 320 pixel data PPs
        line_data = np.zeros(328, dtype=np.uint16)
        
        # Add 8 training PPs at the start of each line
        for i in range(8):
            line_data[i] = training_pattern
        
        # Add 320 pixel data PPs (10-bit random data with proper start/stop bits)
        for i in range(320):
            # Generate random 10-bit pixel value (0-1023)
            pixel_value = np.random.randint(0, 1024)
            
            # Format as 12-bit PP: start bit (1) + 10-bit data + stop bit (0)
            pp_value = (1 << 11) | (pixel_value << 1) | 0
            
            line_data[8 + i] = pp_value
        
        frame_lines.append(line_data)
        return frame_lines

def extract_pixel_data(input_lines):
    """
    Extract the 10-bit pixel data from the 12-bit PPs, removing
    start/stop bits, training patterns, and EOF patterns.
    
    Args:
        input_lines (list): List of numpy arrays representing raw frame lines
        
    Returns:
        np.ndarray: Clean 10-bit pixel data as a 320x320 image
    """
    # Create empty image array (320x320)
    image = np.zeros((len(input_lines)-1, 320), dtype=np.uint16)
    
    # Process each line (except the EOF line)
    for line_idx in range(len(input_lines)-1):
        line_data = input_lines[line_idx]
        
        # Extract pixel data (skip training patterns, remove start/stop bits)
        for pixel_idx in range(320):
            pp_value = line_data[8 + pixel_idx]
            
            # Extract 10-bit pixel value (bits 10-1)
            pixel_value = (pp_value >> 1) & 0x3FF
            
            image[line_idx, pixel_idx] = pixel_value
    
    return image.flatten()



sample_pairs = [
    {
        'input': [np.zeros(10) for _ in range(10)],
        'output': np.zeros((100,))
    },

    # Full NanEyeC frame (320 lines + EOF line) with processed output
    {
        'input': create_naneye_sim_frame(320),
        'output': extract_pixel_data(create_naneye_sim_frame(320))
    }
]

@pytest.mark.parametrize('expected', sample_pairs)
def test_gs_format_frame(expected):
    actual = daq._format_frame(expected['input'])
    assert np.array_equal(actual, expected['output'])