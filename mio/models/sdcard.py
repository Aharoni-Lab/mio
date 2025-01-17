"""
Data model for configuring an SD card. Will be instantiated in the constants module with
specific values. This allows for the model to be reused across different miniscopes, and
for consuming code to use a consistent, introspectable API
"""

from typing import Literal, Optional

from mio.models import MiniscopeConfig
from mio.models.buffer import BufferHeader, BufferHeaderFormat
from mio.models.mixins import ConfigYAMLMixin


class SectorConfig(MiniscopeConfig):
    """
    Configuration of sector layout on the SD card.

    For each sector, one can retrieve the position with the attribute ``_pos``,

    Examples:

        >>> sectors = SectorConfig(header=1023, config=1024, data=1025, size=512)
        >>> sectors.header
        1023
        >>> # should be 1023 * 512
        >>> sectors.header_pos
        523776


    """

    header: int = 1023
    """
    Holds user settings to configure Miniscope and recording
    """
    config: int = 1024
    """
    Holds final settings of the actual recording
    """
    data: int = 1025
    """
    Recording data starts here
    """
    size: int = 512
    """
    The size of an individual sector
    """

    @property
    def header_pos(self) -> int:
        """header * sector size"""
        return self.header * self.size

    @property
    def config_pos(self) -> int:
        """config * sector size"""
        return self.config * self.size

    @property
    def data_pos(self) -> int:
        """data * sector size"""
        return self.data * self.size


class ConfigPositions(MiniscopeConfig):
    """
    Image acquisition configuration positions
    """

    width: int = 0
    height: int = 1
    fs: int = 2
    buffer_size: int = 3
    n_buffers_recorded: int = 4
    n_buffers_dropped: int = 5


class SDHeaderPositions(MiniscopeConfig):
    """
    Positions in the header for the whole SD card
    """

    gain: int = 4
    led: int = 5
    ewl: int = 6
    record_length: int = 7
    fs: int = 8
    """Frame rate"""
    delay_start: Optional[int] = None
    battery_cutoff: Optional[int] = None


class SDBufferHeaderFormat(BufferHeaderFormat):
    """
    Positions in the header for each frame
    """

    id: str = "sd-buffer-header"

    length: Literal[0] = 0
    linked_list: int = 1
    frame_num: int = 2
    buffer_count: int = 3
    frame_buffer_count: int = 4
    write_buffer_count: int = 5
    dropped_buffer_count: int = 6
    timestamp: int = 7
    data_length: int = 8
    write_timestamp: Optional[int] = None
    battery_voltage: Optional[int] = None


class SDLayout(MiniscopeConfig, ConfigYAMLMixin):
    """
    Data layout of an SD Card.

    Used by the :class:`.io.WireFreeMiniscope` class to tell it how data on the SD card is laid out.
    """

    word_size: int = 4
    """
    Size of each header word in bytes
    """

    sectors: SectorConfig
    header: SDHeaderPositions = SDHeaderPositions()
    config: ConfigPositions = ConfigPositions()
    buffer: SDBufferHeaderFormat = SDBufferHeaderFormat()
    header_dtype: str = "uint32"
    """
    String form of the numpy dtype that the global and buffer headers are encoded in 
    """
    buffer_dtype: str = "uint8"
    """
    String form of the numpy dtype that each frame buffer is encoded in
    """


class SDConfig(MiniscopeConfig):
    """
    The configuration of a recording taken on this SD card.

    Read from the locations given in :class:`.ConfigPositions`
    for an SD card with a given :class:`.SDLayout`
    """

    width: int
    height: int
    fs: int
    buffer_size: int
    n_buffers_recorded: int
    n_buffers_dropped: int


class SDBufferHeader(BufferHeader):
    """
    Header data at the start of each frame
    """

    length: int
    write_buffer_count: int
    dropped_buffer_count: int
    data_length: int
    write_timestamp: Optional[int] = None
    battery_voltage: Optional[int] = None
