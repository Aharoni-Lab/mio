from typing import Annotated, Literal
from annotated_types import Len
from mio.models.stream import StreamDevConfig

class GSDevConfig(StreamDevConfig):
    # preamble: Annotated[bytes, Len(min_length=12, max_length=12)]
    """Example docstring"""
    pix_depth: Literal[12]

    # preamble: Annotated[bytes, Len(min_length=12, max_length=12)]
    # """Example docstring"""
    # pix_depth: Literal[10] = 10

    @property
    def frame_width_input (self) -> int:
        """some description about what it is"""
        return self.frame_width + 8
