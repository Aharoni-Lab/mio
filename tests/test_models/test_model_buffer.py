import pytest
from miniscope_io.models.buffer import BufferHeader, BufferHeaderFormat
from pydantic import ValidationError


@pytest.mark.parametrize("construct", [True, False])
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_buffer_from_format(construct):
    """
    Instantiate a BufferHeader from a sequence and a format
    """
    format = BufferHeaderFormat(
        linked_list=0, frame_num=1, buffer_count=2, frame_buffer_count=3, timestamp=4
    )

    bad_vals = ["a", "b", "c", "d", "e"]
    vals = [10, 9, 8, 7, 6]

    # correct vals should work in both cases
    instance = BufferHeader.from_format(vals, format, construct)
    assert list(instance.model_dump().values()) == vals

    # bad vals should only work if we're constructing
    if construct:
        bad_instance = BufferHeader.from_format(bad_vals, format, construct)
        assert list(bad_instance.model_dump().values()) == bad_vals
    else:
        with pytest.raises(ValidationError):
            _ = BufferHeader.from_format(bad_vals, format, construct)