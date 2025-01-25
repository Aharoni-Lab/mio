"""
File-based data sources
"""

import sys
from pathlib import Path
from typing import BinaryIO, ClassVar, Optional

import numpy as np
from pydantic import Field

from mio.exceptions import EndOfRecordingException, ReadHeaderException
from mio.models.pipeline import Source
from mio.models.sdcard import SDBufferHeader, SDLayout

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class FileSource(Source):
    """
    Generic parent class for file sources
    """

    name = "file-source"


class BinaryFileSource(FileSource):
    """
    A FileSource that yields blocks of binary data
    """

    name = "binary-file-source"
    output_type: ClassVar[bytes]

    path: Path
    offset: int = 0
    """
    The offset position from the start of the file from which to consider the "zero point"
    """
    block_size: int
    """
    Number of bytes to read per processing loop
    """

    _f: Optional[BinaryIO] = None

    def start(self) -> None:
        """Open the file, seek to the offset"""
        self._f = open(self.path, "rb")  # noqa: SIM115
        self._f.seek(self.offset, 0)

    def stop(self) -> None:
        """Close the file, remove the reference"""
        self._f.close()
        self._f = None

    def tell(self) -> int:
        """Return the current position in the file"""
        if self._f is None:
            raise RuntimeError("File has not yet been opened with start")
        return self._f.tell()

    def process(self) -> bytes:
        """Return a block of data"""
        return self._f.read(self.block_size)


class SDFileSourceOutput(TypedDict):
    """Output types returned by :meth:`.SDFileSource.process`"""

    header: SDBufferHeader
    buffer: np.ndarray


class SDFileSource(FileSource):
    """
    Structured binary file that has

    * a global header with metadata values
    * a series of buffers, each containing a

        * buffer header - with metadata for that buffer and
        * buffer data - the data for that buffer

    The source thus has two configurations

    * the ``metadata`` - getter and setter for the actual configuration values of the source
    * the ``layout`` - how the configuration and data are laid out within the file.

    """

    name = "sd-file-source"
    output_type = SDFileSourceOutput

    path: Path
    layout: SDLayout

    _f: Optional[BinaryIO] = None
    _positions: dict[int, int] = Field(default_factory=dict)
    """
    A mapping between frame number and byte position in the video that makes for 
    faster seeking :)

    As we read, we store the locations of each frame before reading it. Later, 
    we can assign to `frame` to seek back to those positions. Assigning to `frame` 
    works without caching position, but has to manually iterate through each frame.
    """
    _last_buffer: int = None
    _frame: int = 0

    @property
    def width(self) -> int:
        """width of the captured video in pixels"""
        return self.config.width

    @property
    def height(self) -> int:
        """height of the captured video in pixels"""
        return self.config.height

    @property
    def offset(self) -> int:
        """Start point of the data sector"""
        return self.layout.sectors.data_pos

    @property
    def header_size(self) -> int:
        """
        Number of bytes in a buffer header

        .. note::

            This isn't guaranteed to be accurate, see:
            https://github.com/Aharoni-Lab/Miniscope-v4-Wire-Free/issues/64
        """
        return (
            max([v for v in self.layout.buffer.model_dump().values() if v is not None]) + 1
        ) * self.layout.word_size

    def start(self) -> None:
        """Open the file, seek to the offset"""
        self._last_buffer = 0

        self._f = open(self.path, "rb")  # noqa: SIM115
        self._f.seek(self.offset, 0)

    def stop(self) -> None:
        """Close the file, remove the reference"""
        self._f.close()
        self._f = None

    def tell(self) -> int:
        """Return the current position in the file"""
        if self._f is None:
            raise RuntimeError("File has not yet been opened with start")
        return self._f.tell()

    def process(self) -> SDFileSourceOutput:
        """
        Read a single data buffer, parsing its header and splitting it from the data
        """
        start_position = self.tell()

        header = self._read_header(self._f)
        self._last_buffer = header.buffer_count
        self._frame = header.frame_num

        if header.frame_num not in self._positions:
            self._positions[header.frame_num] = start_position

        buffer = self._read_buffer(self._f, header)
        buffer = self._trim(buffer, header.data_length)
        return {"header": header, "buffer": buffer}

    def _read_header(self, sd: BinaryIO) -> SDBufferHeader:
        """
        Given an already open file buffer opened in bytes mode,
        seeked to the start of a frame, read the data header
        """
        # Get the length of the header from the first word
        try:
            dataHeader = np.frombuffer(
                sd.read(self.layout.word_size), dtype=np.dtype(self.layout.header_dtype)
            )
        except IndexError as e:
            if "index 0 is out of bounds for axis 0 with size 0" in str(e):
                # end of file if we are reading from a disk image without any
                # additional space on disk
                raise EndOfRecordingException("Reached the end of the video!") from None
            else:
                raise e

        # Get the rest of the values in the header
        try:
            dataHeader = np.append(
                dataHeader,
                np.frombuffer(
                    sd.read(int(dataHeader[0]) * self.layout.word_size),
                    dtype=np.dtype(self.layout.header_dtype),
                ),
            )
        except ValueError as e:
            if "read length must be non-negative" in str(e):
                # end of file! Value error thrown because the dataHeader will be
                # blank,  and thus have a value of 0 for the header size, and we
                # can't read 0 from the card.
                raise EndOfRecordingException("Reached the end of the video!") from None
            else:
                raise e

        # use construct because we're already sure these are ints from the numpy casting
        # https://docs.pydantic.dev/latest/usage/models/#creating-models-without-validation
        try:
            return SDBufferHeader.from_format(dataHeader, self.layout.buffer, construct=True)
        except IndexError as e:
            if (
                self._last_buffer
                >= self.config.n_buffers_recorded + self.config.n_buffers_dropped - 1
            ):
                raise EndOfRecordingException("Reached the end of the video!") from None
            else:
                raise ReadHeaderException(
                    "Could not read header, expected header to have "
                    f"{len(self.layout.buffer.model_dump().keys())} fields, "
                    f"got {len(dataHeader)}. Likely mismatch between specified "
                    "and actual SD Card layout or reached end of data.\n"
                    f"Header Data: {dataHeader}"
                ) from e

    def _read_buffer(self, sd: BinaryIO, header: SDBufferHeader) -> np.ndarray:
        return np.frombuffer(sd.read(self._data_read_size(header)), dtype=np.uint8)

    def _data_read_size(self, header: SDBufferHeader) -> int:
        """
        After the header, how many bytes to read for the data in a buffer
        """
        # blocks are quantized by sector size, so get min number of blocks that cover the data
        n_blocks = np.ceil(
            (header.data_length + (header.length * self.layout.word_size))
            / self.layout.sectors.size
        )
        # expand back to n bytes
        sector_size = n_blocks * self.layout.sectors.size
        # subtract length of header
        return int(sector_size - (header.length * self.layout.word_size))

    def _trim(self, data: np.ndarray, expected_size: int) -> np.ndarray:
        """
        Trim or pad an array to match an expected size.

        This should be the case most of the time -
        number of bytes in a memory sector won't match bytes in a buffer
        """
        if data.shape[0] != expected_size:
            # trim if too long
            if data.shape[0] > expected_size:
                data = data[0:expected_size]
            # pad if too short
            else:
                data = np.pad(data, (0, expected_size - data.shape[0]))

        return data
