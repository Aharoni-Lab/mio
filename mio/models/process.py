"""
Module for preprocessing data.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from mio.models.stream import StreamDevConfig


class InteractiveDisplayConfig(BaseModel):
    """
    Configuration for displaying a video.
    """

    enable: bool = Field(
        default=False,
        description="Whether to plot the output .",
    )
    start_frame: Optional[int] = Field(
        default=...,
        description="Frame to start processing at.",
    )
    end_frame: Optional[int] = Field(
        default=...,
        description="Frame to end processing at.",
    )


class MinimumProjectionConfig(BaseModel):
    """
    Configuration for minimum projection.
    """

    enable: bool = Field(
        default=True,
        description="Whether to use minimum projection.",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the video using minimum projection.",
    )
    output_result: bool = Field(
        default=False,
        description="Whether to output the result.",
    )
    output_min_projection: bool = Field(
        default=False,
        description="Whether to output the minimum projection.",
    )


class NoisePatchConfig(BaseModel):
    """
    Configuration for patch based noise handling.
    """

    enable: bool = Field(
        default=True,
        description="Whether to use patch based noise handling.",
    )
    method: Literal["mean_error", "gradient"] = Field(
        default="mean_error",
        description="Method for handling noise.",
    )
    threshold: float = Field(
        default=20,
        description="Threshold for detecting noise.",
    )
    device_config_id: Optional[str] = Field(
        default=None,
        description="ID of the stream device configuration.",
    )
    buffer_split: int = Field(
        default=1,
        description="Number of splits to make in the buffer when detecting noisy areas.",
    )
    diff_multiply: int = Field(
        default=1,
        description="Multiplier for the difference between the mean and the pixel value.",
    )
    output_result: bool = Field(
        default=False,
        description="Whether to output the result.",
    )
    output_noise_patch: bool = Field(
        default=False,
        description="Whether to output the noise patch.",
    )
    output_diff: bool = Field(
        default=False,
        description="Whether to output the difference.",
    )
    output_noisy_frames: bool = Field(
        default=True,
        description="Whether to output the noisy frames as an independent video stream.",
    )

    _device_config: Optional[StreamDevConfig] = None

    @property
    def device_config(self) -> StreamDevConfig:
        """
        Get the stream device configuration.
        """
        if self._device_config is None:
            self._device_config = StreamDevConfig.from_any(self.device_config_id)
        return self._device_config


class FreqencyMaskingConfig(BaseModel):
    """
    Configuration for frequency filtering.
    """

    enable: bool = Field(
        default=True,
        description="Whether to use frequency filtering.",
    )
    spatial_LPF_cutoff_radius: int = Field(
        default=...,
        description="Radius for the spatial cutoff.",
    )
    vertical_BEF_cutoff: int = Field(
        default=5,
        description="Cutoff for the vertical band elimination filter.",
    )
    horizontal_BEF_cutoff: int = Field(
        default=0,
        description="Cutoff for the horizontal band elimination filter.",
    )
    display_mask: bool = Field(
        default=False,
        description="Whether to display the mask.",
    )
    output_result: bool = Field(
        default=False,
        description="Whether to output the result.",
    )
    output_mask: bool = Field(
        default=False,
        description="Whether to output the mask.",
    )
    output_freq_domain: bool = Field(
        default=False,
        description="Whether to output the frequency domain.",
    )


class DenoiseConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Configuration for denoising a video.
    """

    interactive_display: Optional[InteractiveDisplayConfig] = Field(
        default=None,
        description="Configuration for displaying the video.",
    )
    noise_patch: Optional[NoisePatchConfig] = Field(
        default=None,
        description="Configuration for patch based noise handling.",
    )
    frequency_masking: Optional[FreqencyMaskingConfig] = Field(
        default=None,
        description="Configuration for frequency filtering.",
    )
    end_frame: Optional[int] = Field(
        default=None,
        description="Frame to end processing at.",
    )
    minimum_projection: Optional[MinimumProjectionConfig] = Field(
        default=None,
        description="Configuration for minimum projection.",
    )
    output_result: bool = Field(
        default=True,
        description="Whether to output the result.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save the output in.",
    )
