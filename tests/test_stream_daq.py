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
from bitstring import BitArray, Bits
from typing import Generator

from mio import BASE_DIR
from mio.stream_daq import StreamDevConfig, StreamDaq, iter_buffers
from mio.utils import hash_video, hash_file
from .conftest import DATA_DIR, CONFIG_DIR


@pytest.fixture(params=[pytest.param(5, id="buffer-size-5"), pytest.param(10, id="buffer-size-10")])
def default_streamdaq(set_okdev_input, request) -> StreamDaq:

    daqConfig = StreamDevConfig.from_id("test-wireless-200px")
    daqConfig.runtime.frame_buffer_queue_size = request.param
    daqConfig.runtime.image_buffer_queue_size = request.param
    daqConfig.runtime.serial_buffer_queue_size = request.param

    data_file = DATA_DIR / "stream_daq_test_fpga_raw_input_200px.bin"
    set_okdev_input(data_file)

    daq_inst = StreamDaq(device_config=daqConfig)
    return daq_inst


@pytest.mark.parametrize("buffer_size", [5, 50])
@pytest.mark.parametrize(
    "config,data,video_hash_list,show_video",
    [
        (
            "test-wireless-200px",
            "stream_daq_test_fpga_raw_input_200px.bin",
            [
                "ee7bdb97c1e98ebeefc65ae651968e3a72d099e57d1fdec5ec05a3598733db93",
            ],
            False,
        )
    ],
)
def test_video_output(
    config, data, video_hash_list, tmp_path, show_video, set_okdev_input, buffer_size
):
    output_video = tmp_path / "output.avi"

    daqConfig = StreamDevConfig.from_id(config)
    daqConfig.runtime.frame_buffer_queue_size = buffer_size
    daqConfig.runtime.image_buffer_queue_size = buffer_size
    daqConfig.runtime.serial_buffer_queue_size = buffer_size

    data_file = DATA_DIR / data
    set_okdev_input(data_file)

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(source="fpga", video=output_video, show_video=show_video)

    assert output_video.exists()

    output_video_hash = hash_video(output_video)

    assert output_video_hash in video_hash_list


@pytest.mark.parametrize(
    "config,data",
    [
        (
            "test-wireless-200px",
            "stream_daq_test_fpga_raw_input_200px.bin",
        )
    ],
)
def test_binary_output(config, data, set_okdev_input, tmp_path):
    daqConfig = StreamDevConfig.from_id(config)

    data_file = DATA_DIR / data
    set_okdev_input(data_file)

    output_file = tmp_path / "output.bin"

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(source="fpga", binary=output_file, show_video=False)

    assert output_file.exists()

    assert hash_file(data_file) == hash_file(output_file)


@pytest.mark.parametrize("write_metadata", [True, False])
def test_csv_output(tmp_path, default_streamdaq, write_metadata, caplog):
    """
    Giving a path to the ``metadata`` capture kwarg should save header metadata to a csv
    """
    output_csv = tmp_path / "output.csv"

    if write_metadata:
        default_streamdaq.capture(source="fpga", metadata=output_csv, show_video=False)

        df = pd.read_csv(output_csv)
        # actually not sure what we should be looking for here, for now we just check for shape
        # this should be the same as long as the test data stays the same,
        # but it's a pretty weak test.
        assert df.shape == (910, 12)

        # ensure there were no errors during capture
        for record in caplog.records:
            assert "Exception saving headers" not in record.msg
    else:
        default_streamdaq.capture(source="fpga", metadata=None, show_video=False)
        assert not output_csv.exists()


# This is a helper function for test_continuous_and_termination() that is currently skipped
"""
def capture_wrapper(default_streamdaq, source, show_video, continuous):
    try:
        default_streamdaq.capture(source=source, show_video=show_video, continuous=continuous)
    except KeyboardInterrupt:
        pass # expected
"""


@pytest.mark.skip(
    "Needs to be implemented. Temporary skipped because tests fail in some OS (See GH actions)."
)
@pytest.mark.timeout(10)
def test_continuous_and_termination(tmp_path, default_streamdaq):
    """
    Make sure continuous mode runs forever until interrupted, and that all processes are
    cleaned up when the capture process is terminated.
    """
    """
    timeout = 1

    capture_process = multiprocessing.Process(target=capture_wrapper, args=(default_streamdaq, "fpga", False, True))

    capture_process.start()
    alive_processes = default_streamdaq.alive_processes()
    initial_alive_processes = len(alive_processes)
    
    time.sleep(timeout)

    alive_processes = default_streamdaq.alive_processes()
    assert len(alive_processes) == initial_alive_processes
    
    os.kill(capture_process.pid, signal.SIGINT)
    capture_process.join()

    alive_processes = default_streamdaq.alive_processes()
    assert len(alive_processes) == 0
    """
    pass


def test_metadata_plotting(tmp_path, default_streamdaq):
    """
    Setting the capture kwarg ``show_metadata == True`` should plot the frame metadata
    during capture.
    """
    default_streamdaq.capture(source="fpga", show_metadata=True, show_video=False)

    # unit tests for the stream plotter should go elsewhere, here we just
    # test that the object was instantiated and that it got the data it should have
    assert default_streamdaq._header_plotter is not None
    assert [
        k for k in default_streamdaq._header_plotter.data.keys()
    ] == default_streamdaq.config.runtime.plot.keys
    assert all(
        [
            len(v) == default_streamdaq.config.runtime.plot.history
            for v in default_streamdaq._header_plotter.data.values()
        ]
    )
    assert (
        len(default_streamdaq._header_plotter.index)
        == default_streamdaq.config.runtime.plot.history
    )


def test_bitfile_names():
    """
    Bitfile names should have no periods or whitespace in the filenames (except for the .bit extension)
    """
    pattern = re.compile(r"\.(?!bit$)|\s")
    for path in Path(BASE_DIR).glob("**/*.bit"):
        assert not pattern.search(str(path.name))


@pytest.mark.parametrize("read_size", [3, 5, 7])
def test_iter_buffers(read_size: int, tmp_path: Path):
    """
    iter_buffers should accept an iterator that yield bytes,
    and split it by the preamble in a way that's insensitive to
    the length of the read size
    """
    preamble_bytes = b"ab"
    n_reps = 3

    preamble = Bits(preamble_bytes)
    buffer = preamble_bytes + b"000"
    buffer_rep = buffer * n_reps

    def _iterator(read_size: int) -> Generator[bytes, None, None]:
        nonlocal buffer_rep
        for i in range(0, len(buffer_rep), read_size):
            yield buffer_rep[i : i + read_size]

    got_buffers = []
    for buf in iter_buffers(_iterator(read_size), preamble=preamble):
        got_buffers.append(buf)

    assert all([buf == buffer for buf in got_buffers])
