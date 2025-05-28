from typing import TYPE_CHECKING, Self

import numpy as np

from mio.models.stream import (
    StreamBufferHeader,
    StreamBufferHeaderFormat
)

if TYPE_CHECKING:
    from mio.devices.gs.config import GSDevConfig

def buffer_to_array(buffer: bytes) -> np.ndarray:
    """
    Given the GS's "12-bit" pixel format,
    where 10-bit pixels are flanked by two pad values
    e.g. (``1xxxxxxxxxx0``)

    Strip the pads, and return a 16-bit ndarray
    """
    # convert to a binary array
    binary = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))
    # reshape to be
    pixel_cols = binary.reshape((12, -1), order="F")
    # remove padding pixels 
    stripped = pixel_cols[1:-1]
    # Cast to 16 bit ndarray
    slice0 = np.packbits(stripped[:-8, :], axis=0, bitorder="little").astype(np.uint16) * 16
    slice1 = np.packbits(stripped[-8:, :], axis=0, bitorder="little").astype(np.uint16)
    out = slice0 + slice1

    return stripped.flatten(order="F")





class GSBufferHeader(StreamBufferHeader):
    @classmethod
    def from_buffer(cls, buffer: bytes, header_fmt: "GSBufferHeaderFormat", config: "GSDevConfig") -> tuple[Self, np.ndarray]:
        header_start = len(config.preamble)
        header_end = header_start + (header_fmt.header_length * 4)
        header_array = np.ndarray(buffer[header_start:header_end], dtype=np.uint32)
        header = cls.from_format(header_array, header_fmt, construct=True)




class GSBufferHeaderFormat(StreamBufferHeaderFormat):
    pass