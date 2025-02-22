"""
Command line interface for offline video pre-processing.
"""

from pathlib import Path
from typing import Optional

import click

from mio.models.process import DenoiseConfig
from mio.process.video import denoise_run
from mio.utils.config import update_config_from_json


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
@click.option(
    "-u",
    "--update-from-json",
    type=click.Path(exists=True, path_type=Path),
    help="Update config from a processing_log.json file",
)
@click.option(
    "--save-yaml",
    is_flag=True,
    help="Save updated config back to denoise_example.yml",
)
def denoise(
    input: str,
    denoise_config: str,
    update_from_json: Optional[Path] = None,
    save_yaml: bool = False,
) -> None:
    """
    Denoise a video file.
    """
    config = DenoiseConfig.from_any(denoise_config)

    if update_from_json:
        config = update_config_from_json(config, update_from_json)

    denoise_run(input, config)
