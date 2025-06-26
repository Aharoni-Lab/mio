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

    TURN INTO 1D PIXEL ARRAY
    """
    # convert to a binary array
    binary = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))

    return binary
    # return packed_16bit.flatten(order="F")





class GSBufferHeader(StreamBufferHeader):
    @classmethod
    def from_buffer(cls, buffer: bytes, header_fmt: "GSBufferHeaderFormat", config: "GSDevConfig") -> tuple[Self, np.ndarray]:
        try:
            header_start = len(config.preamble)*config.dummy_words
            header_end = header_start + (header_fmt.header_length * 4)
            # header_array = np.ndarray(buffer[header_start:header_end], dtype=np.uint32)
            header_array = np.frombuffer(buffer[header_start:header_end], dtype=np.uint32)
            header = cls.from_format(header_array, header_fmt, construct=True)

            payload = buffer_to_array(buffer)
            # payload = buffer_to_array(buffer)

        except:
            print(len(buffer))
            print(header_start)
            print(header_end)
            raise

        return header, payload




class GSBufferHeaderFormat(StreamBufferHeaderFormat):
    pass