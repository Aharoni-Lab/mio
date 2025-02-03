import pytest

from mio import DEVICE_DIR
from mio.models.stream import ADCScaling, StreamDevConfig, StreamBufferHeader

from ..conftest import CONFIG_DIR


@pytest.mark.parametrize(
    "config",
    [
        "preamble_hex.yml",
    ],
)
def test_preamble_hex_parsing(config):
    """
    Test that a hexadecimal string is correctly parsed to a byte string
    from a string or a hex integer
    """
    config_file = CONFIG_DIR / config

    instance = StreamDevConfig.from_yaml(config_file)
    assert instance.preamble == b"\x124Vx"


def test_absolute_bitstream():
    """
    Relative paths should be resolved relative to the devices dir
    """
    example = CONFIG_DIR / "wireless_example.yml"

    instance = StreamDevConfig.from_yaml(example)
    assert instance.bitstream.is_absolute()
    assert str(instance.bitstream).startswith(str(DEVICE_DIR))


_default_adc_scale = {
    "ref_voltage": 1.1,
    "bitdepth": 8,
    "battery_div_factor": 5,
    "vin_div_factor": 11.3,
}


@pytest.mark.parametrize("scale", [1, 2, _default_adc_scale["ref_voltage"]])
def test_adc_scaling(scale):
    
    ref_voltage = scale
    bitdepth = 8
    
    battery_div_factor = 5.0
    vin_div_factor = 11.3
    battery_max_voltage = 10.0
    vin_max_voltage = 20.0

    battery_voltage_raw = 100
    input_voltage_raw = 150

    battery_factor = 1 / (2 ** bitdepth) * ref_voltage * battery_div_factor
    vin_factor = 1 / (2 ** bitdepth) * ref_voltage * vin_div_factor

    instance_header = StreamBufferHeader(
        linked_list=0,
        frame_num=0,
        buffer_count=0,
        frame_buffer_count=0,
        write_buffer_count=0,
        dropped_buffer_count=0,
        timestamp=0,
        pixel_count=0,
        write_timestamp=0,
        battery_voltage_raw=battery_voltage_raw,
        input_voltage_raw=input_voltage_raw,
        adc_scale=ADCScaling(
            ref_voltage=ref_voltage,
            bitdepth=bitdepth,
            battery_div_factor=battery_div_factor,
            vin_div_factor=vin_div_factor,
            battery_max_voltage=battery_max_voltage,
            vin_max_voltage=vin_max_voltage,
        ),
    )

    expected_battery_voltage = battery_voltage_raw * battery_factor
    if expected_battery_voltage > battery_max_voltage:
        expected_battery_voltage = -1

    expected_input_voltage = input_voltage_raw * vin_factor
    if expected_input_voltage > vin_max_voltage:
        expected_input_voltage = -1

    assert instance_header.battery_voltage == expected_battery_voltage
    assert instance_header.input_voltage == expected_input_voltage