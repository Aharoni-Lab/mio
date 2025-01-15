"""
Module for preprocessing data.
"""

from typing import Optional

from pydantic import BaseModel, Field

from mio.models.mixins import YAMLMixin


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
    method: str = Field(
        default="mean_error",
        description="Method for handling noise.",
    )
    threshold: float = Field(
        default=20,
        description="Threshold for detecting noise.",
    )
    buffer_size: int = Field(
        default=5032,
        description="Size of the buffers composing the image."
        "This premises that the noisy area will appear in units of buffer_size.",
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


class DenoiseConfig(BaseModel, YAMLMixin):
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
