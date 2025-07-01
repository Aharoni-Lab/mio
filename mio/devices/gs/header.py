# ruff: noqa: D100

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import logging

from mio.models.stream import StreamBufferHeader, StreamBufferHeaderFormat

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

    # reshape to be n x 12
    pixel_cols = binary.reshape((-1, 12))

    # remove padding pixels (12 bit x n --> 10 bit x n)
    stripped = pixel_cols[:, 1:-1]

    # Cast to 16 bit ndarray
    padded = np.pad(stripped, ((0, 0), (6, 0)), mode="constant", constant_values=0)
    packed_16bit = np.packbits(padded, axis=1).view(np.uint16).byteswap()

    return packed_16bit.flatten()


class GSBufferHeader(StreamBufferHeader):
    """
    Header at the start of GS buffers -

    prefixed by ... with fields...
    formatted by :class:`.GSBufferHeaderFormat`
    (...hemal describe data structure...)
    """

    @classmethod
    def from_buffer(
        cls, buffer: bytes, header_fmt: "GSBufferHeaderFormat", config: "GSDevConfig"
    ) -> tuple[Self, np.ndarray]:
        """Split buffer into a :class:`.GSBufferHeader` and a 1D, 16-bit pixel array."""
        header_start = len(config.preamble) * config.dummy_words
        print(header_start)
        header_end = header_start + (header_fmt.header_length * 4)
        header_array = np.frombuffer(buffer[header_start:header_end], dtype=np.uint32)
        header = cls.from_format(header_array, header_fmt, construct=True)
        payload = buffer_to_array(buffer[header_end+1:]) # added +1 for header end+1

        return header, payload


class GSBufferHeaderFormat(StreamBufferHeaderFormat):
    """Positions of header fields in GS headers"""

    pass
