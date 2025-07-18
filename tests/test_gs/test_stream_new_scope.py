
from mio.devices.gs.daq import GSStreamDaq
from mio.devices.gs.header import GSBufferHeader

from mio.utils import hash_file
from ..conftest import DATA_DIR

from mio.stream_daq import StreamDaq
from mio.devices.gs.config import GSDevConfig


def test_binary_output(set_okdev_input, tmp_path):
    daqConfig = GSDevConfig.from_id("MSUS-test")

    data_file = DATA_DIR / "test_new_scope.bin"
    set_okdev_input(data_file)

    output_file = tmp_path / "output.bin"

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(source="fpga", binary=output_file, show_video=False)

    assert output_file.exists()

    assert hash_file(data_file) == hash_file(output_file)

from mio.stream_daq import StreamDaq

def test_buffer_npix_calculation(buffer_block_length, block_size, pix_depth, expected_npix):
    """
    Test that the number of pixels per buffer is correctly calculated
    based on buffer structure parameters.
    """
    # Create a minimal config for testing
    config_data = {
        "device": "OK",
        "frame_width": 320,
        "frame_height": 328,
        "preamble": "0x78563412",
        "header_len": 384,
        "buffer_block_length": buffer_block_length,
        "block_size": block_size,
        "pix_depth": pix_depth,
        "num_buffers": 11,
    }
    daqConfig = GSDevConfig.from_id("MSUS-test")
    # Calculate pixels per buffer
    calculated_npix = (daqConfig.buffer_block_length * daqConfig.block_size * 32) // daqConfig.pix_depth

    assert calculated_npix == expected_npix
