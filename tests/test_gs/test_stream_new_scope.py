
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
    # GSStreamDaq
    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(source="fpga", binary=output_file, show_video=False)

    assert output_file.exists()

    assert hash_file(data_file) == hash_file(output_file)

