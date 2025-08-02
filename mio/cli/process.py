"""
Command line interface for offline video pre-processing.
"""

import click

from mio.models.process import DenoiseConfig
from mio.process.video import denoise_run


@click.group()
def process() -> None:
    """
    Command group for video processing.
    """
    pass


@process.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the video file to process.",
)
@click.option(
    "-c",
    "--denoise_config",
    required=True,
    type=str,
    help="Path to the YAML processing configuration file.",
)
def denoise(
    input: str,
    denoise_config: str,
) -> None:
    """
    Denoise a video file.
    """
    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)
    denoise_run(input, denoise_config_parsed)
