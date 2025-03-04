"""Configuration utilities for miniscope-io."""

import json
from pathlib import Path

from mio.models.process import DenoiseConfig


def update_config_from_json(config: DenoiseConfig, json_path: Path) -> DenoiseConfig:
    """Update DenoiseConfig from a processing_log.json file."""
    with open(json_path) as f:
        log_data = json.load(f)

    # Get most recent processing run
    proc_config = log_data["processing_runs"][-1]["processing_config"]

    # Update noise patch config
    config.noise_patch.enable = proc_config["noise_patch"]["enabled"]
    if config.noise_patch.enable:
        config.noise_patch.method = proc_config["noise_patch"]["method"]
        if proc_config["noise_patch"]["gradient_config"]:
            config.noise_patch.gradient_config.threshold = proc_config["noise_patch"][
                "gradient_config"
            ]["threshold"]
        if proc_config["noise_patch"]["black_area_config"]:
            config.noise_patch.black_area_config.consecutive_threshold = proc_config["noise_patch"][
                "black_area_config"
            ]["consecutive_threshold"]
            config.noise_patch.black_area_config.value_threshold = proc_config["noise_patch"][
                "black_area_config"
            ]["value_threshold"]

    # Update frequency masking config
    config.frequency_masking.enable = proc_config["frequency_masking"]["enabled"]
    if config.frequency_masking.enable:
        config.frequency_masking.cast_float32 = proc_config["frequency_masking"]["cast_float32"]
        config.frequency_masking.spatial_LPF_cutoff_radius = proc_config["frequency_masking"][
            "cutoff_radius"
        ]
        config.frequency_masking.vertical_BEF_cutoff = proc_config["frequency_masking"][
            "vertical_BEF"
        ]
        config.frequency_masking.horizontal_BEF_cutoff = proc_config["frequency_masking"][
            "horizontal_BEF"
        ]

    # Update butterworth config
    config.butter_filter.enable = proc_config["butterworth"]["enabled"]
    if config.butter_filter.enable:
        config.butter_filter.order = proc_config["butterworth"]["order"]
        config.butter_filter.cutoff_frequency = proc_config["butterworth"]["cutoff_frequency"]
        config.butter_filter.sampling_rate = proc_config["butterworth"]["sampling_rate"]

    return config
