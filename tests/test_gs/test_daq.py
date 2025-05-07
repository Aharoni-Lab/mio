import pytest
from mio.devices.gs import daq
import numpy as np

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