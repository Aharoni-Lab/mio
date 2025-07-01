import re
from pathlib import Path

import multiprocessing
import os
import pytest
import pandas as pd
import sys
import signal
import time
from contextlib import contextmanager

from mio import BASE_DIR
from mio.devices.gs import testing
from mio.devices.gs.daq import GSStreamDaq
from mio.devices.gs.header import GSBufferHeaderFormat, GSBufferHeader
from mio.devices.gs.config import GSDevConfig
import numpy as np
from mio.utils import hash_video, hash_file
# from .conftest import DATA_DIR, CONFIG_DIR

from mio.stream_daq import StreamDaq
# from test_header import
tests/test_gs/test_new_scope.bin

@classmethod
    @pytest.fixture(params=[pytest.param(5, id="buffer-size-5"), pytest.param(10, id="buffer-size-10")])

    def test_binary_output(config, data, set_okdev_input, tmp_path):
    # test
        daqConfig = GSStreamDaq.from_id(config)

        data_file = DATA_DIR / data
        set_okdev_input(data_file)

        output_file = tmp_path / "output.bin"

        daq_inst = StreamDaq(device_config=daqConfig)
        daq_inst.capture(source="fpga", binary=output_file, show_video=False)

        assert output_file.exists()

        assert hash_file(data_file) == hash_file(output_file)
