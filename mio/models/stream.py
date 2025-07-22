"""
Models for :mod:`mio.stream_daq`
"""

from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, computed_field, field_validator

from mio import DEVICE_DIR
from mio.models import MiniscopeConfig
from mio.models.buffer import BufferHeader, BufferHeaderFormat
from mio.models.mixins import ConfigYAMLMixin
from mio.models.sinks import CSVWriterConfig, StreamPlotterConfig


class ADCScaling(MiniscopeConfig):
    """
    Configuration for the ADC scaling factors
    """

    ref_voltage: float = Field(
        1.1,
        description="Reference voltage of the ADC",
    )
    bitdepth: int = Field(
        8,
        description="Bit depth of the ADC",
    )
    battery_div_factor: float = Field(
        5.0,
        description="Voltage divider factor for the battery voltage",
    )
    vin_div_factor: float = Field(
        11.3,
        description="Voltage divider factor for the Vin voltage",
    )

    def scale_battery_voltage(self, voltage_raw: float) -> float:
        """
        Scale raw input ADC voltage to Volts

        Args:
            voltage_raw: Voltage as output by the ADC

        Returns:
            float: Scaled voltage
        """
        return voltage_raw / 2**self.bitdepth * self.ref_voltage * self.battery_div_factor

    def scale_input_voltage(self, voltage_raw: float) -> float:
        """
        Scale raw input ADC voltage to Volts

        Args:
            voltage_raw: Voltage as output by the ADC

        Returns:
            float: Scaled voltage
        """
        return voltage_raw / 2**self.bitdepth * self.ref_voltage * self.vin_div_factor


class StreamBufferHeaderFormat(BufferHeaderFormat):
    """
    Refinements of :class:`.BufferHeaderFormat` for
    :class:`~mio.stream_daq.StreamDaq`

    Parameters
    ----------
    pixel_count: int
        Number of pixels in the buffer.
    battery_voltage: int
        Battery voltage. This is currently raw ADC value.
        Mapping to mV will be documented in device documentation.
    vin_voltage: int
        Input voltage. This is currently raw ADC value.
        Mapping to mV will be documented in device documentation.
    """

    pixel_count: int
    battery_voltage_raw: int
    input_voltage_raw: int


class StreamBufferHeader(BufferHeader):
    """
    Refinements of :class:`.BufferHeader` for
    :class:`~mio.stream_daq.StreamDaq`
    """

    pixel_count: int
    battery_voltage_raw: int
    input_voltage_raw: int
    _adc_scaling: ADCScaling = None

    @property
    def adc_scaling(self) -> Optional[ADCScaling]:
        """
        :class:`.ADCScaling` applied to voltage readings
        """
        return self._adc_scaling

    @adc_scaling.setter
    def adc_scaling(self, scaling: ADCScaling) -> None:
        self._adc_scaling = scaling

    @computed_field
    def battery_voltage(self) -> float:
        """
        Scaled battery voltage in Volts.
        """
        if self._adc_scaling is None:
            return self.battery_voltage_raw
        else:
            return self._adc_scaling.scale_battery_voltage(self.battery_voltage_raw)

    @computed_field
    def input_voltage(self) -> float:
        """
        Scaled input voltage in Volts.
        """
        if self._adc_scaling is None:
            return self.input_voltage_raw
        else:
            return self._adc_scaling.scale_input_voltage(self.input_voltage_raw)


class StreamDevRuntime(MiniscopeConfig):
    """
    Runtime configuration for :class:`.StreamDaq`

    Included within :class:`.StreamDevConfig` to separate config that is not
    unique to the device, but how that device is controlled at runtime.
    """

    serial_buffer_queue_size: int = Field(
        10,
        description="Buffer length for serial data reception in streamDaq",
    )
    frame_buffer_queue_size: int = Field(
        5,
        description="Buffer length for storing frames in streamDaq",
    )
    image_buffer_queue_size: int = Field(
        5,
        description="Buffer length for storing images in streamDaq",
    )
    queue_put_timeout: int = Field(
        5,
        description="Timeout for putting data into the queue",
    )
    plot: Optional[StreamPlotterConfig] = Field(
        StreamPlotterConfig(
            keys=["timestamp", "buffer_count", "frame_buffer_count"], update_ms=1000, history=500
        ),
        description="Configuration for plotting header data as it is collected. "
        "If ``None``, use the default params in StreamPlotter. "
        "Note that this does *not* control whether header metadata is plotted during capture, "
        "for enabling/disabling, use the ``show_metadata`` kwarg in the capture method",
    )
    csvwriter: Optional[CSVWriterConfig] = Field(
        CSVWriterConfig(buffer=100),
        description="Default configuration for writing header data to a CSV file. "
        "If ``None``, use the default params in BufferedCSVWriter. "
        "Note that this does *not* control whether header metadata is written during capture, "
        "for enabling/disabling, use the ``metadata`` kwarg in the capture method.",
    )


class StreamDevConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Format model used to parse DAQ configuration yaml file (examples are in ./config)
    The model attributes are key-value pairs needed for reconstructing frames from data streams.

    Parameters
    ----------
    device: str
        Interface hardware used for receiving data.
        Current options are "OK" (Opal Kelly XEM 7310) and "UART" (generic UART-USB converters).
        Only "OK" is supported at the moment.
    bitstream: str, optional
        Required when device is "OK".
        The configuration bitstream file to upload to the Opal Kelly board.
        This uploads a Manchester decoder HDL and different bitstream files are required
        to configure different data rates and bit polarity.
        This is a binary file synthesized using Vivado,
        and details for generating this file will be provided in later updates.
    port: str, optional
        Required when device is "UART".
        COM port connected to the UART-USB converter.
    baudrate: Optional[int]
        Required when device is "UART".
        Baudrate of the connection to the UART-USB converter.
    frame_width: int
        Frame width of transferred image. This is used to reconstruct image.
    frame_height: int
        Frame height of transferred image. This is used to reconstruct image.
    fs: int
        Framerate of acquired stream
    preamble: str
        32-bit preamble used to locate the start of each buffer.
        The header and image data follows this preamble.
        This is used as a hex but imported as a string because yaml doesn't support hex format.
    header_len : int, optional
        Length of header in bits. (For 32-bit words, 32 * number of words)
        This is useful when not all the variable/words in the header are defined in
        :class:`.MetadataHeaderFormat`.
        The user is responsible to ensure that `header_len` is larger than the largest bit
        position defined in :class:`.MetadataHeaderFormat`
        otherwise unexpected behavior might occur.
    pix_depth : int, optional
        Bit-depth of each pixel, by default 8.
    buffer_block_length: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        Number of blocks per each data buffer.
        This is required to calculate the number of pixels contained in one data buffer.
    block_size: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        Number of 32-bit words per data block.
        This is required to calculate the number of pixels contained in one data buffer.
    num_buffers: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        This is the number of buffers that the source microcontroller cycles around.
        This isn't strictly required for data reconstruction but useful for debugging.
    reverse_header_bits : bool, optional
        If True, reverse the bits within each byte of the header.
        Default is False.
    reverse_header_bytes : bool, optional
        If True, reverse the byte order within each 32-bit word of the header.
        This is used for handling endianness in systems where the byte order needs to be swapped.
        Default is False.
    reverse_payload_bits : bool, optional
        If True, reverse the bits within each byte of the payload.
        Default is False.
    reverse_payload_bytes : bool, optional
        If True, reverse the byte order within each 32-bit word of the payload.
        This is used for handling endianness in systems where the byte order needs to be swapped.
        Default is False.
    dummy_words : int, optional
        Number of 32-bit dummy words in the header.
        This is used to stabilize clock recovery in FPGA Manchester decoder.
        This value does not have a meaning for image recovery.

    ..todo::
        Move port (for USART) to a user config area. This should make this pure device config.
    """

    device: Literal["OK", "UART"]
    bitstream: Optional[Path] = None
    port: Optional[str] = None
    baudrate: Optional[int] = None
    frame_width: int
    frame_height: int
    fs: int = 20
    preamble: bytes
    header_len: int
    pix_depth: int = 8
    buffer_block_length: int
    block_size: int
    num_buffers: int
    reverse_header_bits: bool = False
    reverse_header_bytes: bool = False
    reverse_payload_bits: bool = False
    reverse_payload_bytes: bool = False
    dummy_words: int = 0
    adc_scale: Optional[ADCScaling] = ADCScaling()
    runtime: StreamDevRuntime = StreamDevRuntime()

    _px_per_buffer: int = None

    @field_validator("preamble", mode="before")
    def preamble_to_bytes(cls, value: Union[str, bytes, int]) -> bytes:
        """
        Cast ``preamble`` to bytes.

        Args:
            value (str, bytes, int): Recast from `str` (in yaml like ``preamble: "0x12345"`` )
                or `int` (in yaml like `preamble: 0x12345`

        Returns:
            bytes
        """
        if isinstance(value, str):
            return bytes.fromhex(value)
        elif isinstance(value, int):
            return bytes.fromhex(hex(value)[2:])
        else:
            return value

    @field_validator("bitstream", mode="after")
    def resolve_relative(cls, value: Path) -> Path:
        """
        If we are given a relative path to a bitstream, resolve it relative to
        the device path
        """
        if not value.is_absolute():
            value = DEVICE_DIR / value
        return value

    @field_validator("bitstream", mode="after")
    def ensure_exists(cls, value: Optional[Path]) -> Optional[Path]:
        """If a bitstream file has been provided, ensure it exists"""
        if isinstance(value, Path):
            assert (
                value.exists()
            ), f"Configured to use bitstream file {value}, but it does not exist"
        return value

    @property
    def px_per_buffer(self) -> int:
        """
        Number of pixels per buffer
        """

        px_per_word = 32 / self.pix_depth
        if self._px_per_buffer is None:
            self._px_per_buffer = (
                self.buffer_block_length * self.block_size
                - self.header_len / self.pix_depth
                - px_per_word * self.dummy_words
            )
        return self._px_per_buffer
