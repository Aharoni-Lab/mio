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
        """some description about what it is"""
        return self.frame_width + 8

    def pix_depth_input(self) -> int:
        """some description about what it is"""
        return self.pix_depth - 4
