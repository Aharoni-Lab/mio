from typing import Annotated, Literal as L
from annotated_types import Len
from mio.models.stream import StreamDevConfig

class GSDevConfig(StreamDevConfig):
    preamble: Annotated[bytes, Len(min_length=12, max_length=12)]
    """Example docstring"""
    pix_depth: L[10] = 10

    # processing header from buffers, see _parse_header
    # header_len