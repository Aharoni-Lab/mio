# ruff: noqa: D100


from pydantic import ConfigDict

from mio.models.stream import StreamDevConfig


class GSDevConfig(StreamDevConfig):
    # preamble: Annotated[bytes, Len(min_length=12, max_length=12)]
    """Sets the hard-limits of the Miniscope"""
    pix_depth: int = 12

    # preamble: Annotated[bytes, Len(min_length=12, max_length=12)]
    # """Example docstring"""
    # pix_depth: Literal[10] = 10

    model_config = ConfigDict(validate_default=True)

    @property
    def frame_width_input(self) -> int:
        """8 (12 bit) alignment columns removed from 320 rows of imaging data"""
        return self.frame_width + 8

    def pix_depth_input(self) -> int:
        """12 bit raw processed to 10 bit pixel values"""
        return self.pix_depth + 2
