import pytest

from mio.process.stitch import frame_timestamp_match_ratio

def test_frame_timestamp_match_ratio():
    timestamp_lists = [[], [1, 2, 3]]
    assert frame_timestamp_match_ratio(timestamp_lists) == pytest.approx(0, abs=1e-5)

    timestamp_lists = [[2, 3], [1, 2, 3]]
    assert frame_timestamp_match_ratio(timestamp_lists) == pytest.approx(0.6666666667, abs=1e-5)

    timestamp_lists = [[4, 6, 30], [1, 2, 3]]
    assert frame_timestamp_match_ratio(timestamp_lists) == pytest.approx(0, abs=1e-5)

    timestamp_lists = [[1, 2, 3], [1, 2, 3], [1, 2, 4]]
    assert frame_timestamp_match_ratio(timestamp_lists) == pytest.approx(0.6666666667, abs=1e-5)

