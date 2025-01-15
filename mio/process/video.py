"""
This module contains functions for pre-processing video data.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from mio import init_logger
from mio.io import VideoReader
from mio.models.frames import NamedFrame
from mio.models.process import (
    DenoiseConfig,
    FreqencyMaskingConfig,
    MinimumProjectionConfig,
    NoisePatchConfig,
)
from mio.plots.video import VideoPlotter
from mio.process.frame_helper import FrequencyMaskHelper, NoiseDetectionHelper, ZStackHelper

logger = init_logger("video")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class BaseVideoProcessor:
    """
    Base class for defining an abstract video processor.

    Attributes:
    name (str): The name of the video processor.
    output_frames (list): A list of output frames.
    named_frame (NamedFrame): A NamedFrame object.
    """

    def __init__(self, name: str, output_dir: Path):
        """
        Initialize the BaseVideoProcessor object.

        Parameters:
        name (str): The name of the video processor.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
        output_dir (Path): The output directory.

        Returns:
        BaseVideoProcessor: A BaseVideoProcessor object.
        """
        self.name: str = name
        self.output_dir: Path = output_dir
        self.output_frames: list[np.ndarray] = []
        self.output_enable: bool = True

    @property
    def output_named_frame(self) -> NamedFrame:
        """
        Get the output NamedFrame object.

        Returns:
        NamedFrame: The output NamedFrame object.
        """
        return NamedFrame(name=self.name, video_frame=self.output_frames)

    def append_output_frame(self, input_frame: np.ndarray) -> None:
        """
        Append a frame to the output_frames list.

        Parameters:
        frame (np.ndarray): The frame to append.
        """
        self.output_frames.append(input_frame)

    def export_output_video(self) -> None:
        """
        Export the video to a file.
        """
        if self.output_enable:
            logger.info(f"Exporting {self.name} video to {self.output_dir}")
            self.output_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} output disabled.")

    def process_frame(self) -> None:
        """
        Process a single frame. This method should be implemented in the subclass.

        Parameters:
        frame (np.ndarray): The frame to process.
        """
        raise NotImplementedError("process_frame method must be implemented in the subclass.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. This method should be overridden in the subclass.
        """
        raise NotImplementedError("batch_export_videos method must be implemented in the subclass.")


class NoisePatchProcessor(BaseVideoProcessor):
    """
    A class to apply noise patching to a video.
    """

    def __init__(
        self,
        name: str,
        noise_patch_config: NoisePatchConfig,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        """
        Initialize the NoisePatchProcessor object.

        Parameters:
        name (str): The name of the video processor.
        noise_patch_config (NoisePatchConfig): The noise patch configuration.
        """
        super().__init__(name, output_dir)
        self.noise_detect_helper = NoiseDetectionHelper(height=height, width=width)
        self.noise_patch_config: NoisePatchConfig = noise_patch_config
        self.previous_frame: Optional[np.ndarray] = None
        self.noise_patchs: list[np.ndarray] = []
        self.diff_frames: list[np.ndarray] = []
        self.output_enable: bool = noise_patch_config.output_result

    def process_frame(self, input_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame.

        Parameters:
        raw_frame (np.ndarray): The raw frame to process.
        previous_frame (np.ndarray): The previous frame to compare against.
        For the first frame, this is None.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The processed frame and the noise patch.
        """

        if self.noise_patch_config.enable and self.previous_frame is not None:
            patched_frame, noise_patch = self.noise_detect_helper.patch_noisy_buffer(
                input_frame,
                self.previous_frame,
                self.noise_patch_config,
            )
            self.append_output_frame(patched_frame)
            self.noise_patchs.append(noise_patch * np.iinfo(np.uint8).max)
            self.diff_frames.append(
                cv2.absdiff(input_frame, self.previous_frame)
                * self.noise_patch_config.diff_multiply
            )

            return patched_frame, noise_patch
        if self.noise_patch_config.enable and self.previous_frame is None:
            self.append_output_frame(input_frame)
            self.noise_patchs.append(np.zeros_like(input_frame))
            self.diff_frames.append(np.zeros_like(input_frame))
            self.previous_frame = input_frame
            return input_frame, np.zeros_like(input_frame)
        else:
            return input_frame, np.zeros_like(input_frame)

    @property
    def noise_patch_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the noise patch.
        """
        return NamedFrame(name="patched_area", video_frame=self.noise_patchs)

    @property
    def diff_frames_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the difference frames.
        """
        return NamedFrame(
            name=f"diff_{self.noise_patch_config.diff_multiply}x", video_frame=self.diff_frames
        )

    def export_noise_patch(self) -> None:
        """
        Export the noise patch to a file.
        """
        if self.noise_patch_config.output_noise_patch:
            logger.info(f"Exporting {self.name} noise patch to {self.output_dir}")
            self.noise_patch_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} noise patch output disabled.")

    def export_diff_frames(self) -> None:
        """
        Export the difference frames to a file.
        """
        if self.noise_patch_config.output_diff:
            logger.info(f"Exporting {self.name} difference frames to {self.output_dir}")
            self.diff_frames_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} difference frames output disabled.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_noise_patch()
        self.export_diff_frames()


class FreqencyMaskProcessor(BaseVideoProcessor):
    """
    A class to apply frequency masking to a video.
    """

    def __init__(
        self,
        name: str,
        freq_mask_config: FreqencyMaskingConfig,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        """
        Initialize the FreqencyMaskProcessor object.

        Parameters:
        name (str): The name of the video processor.
        freq_mask_config (FreqencyMaskingConfig): The frequency masking configuration.
        """
        super().__init__(name, output_dir)
        self.freq_mask_helper = FrequencyMaskHelper(height=height, width=width)
        self.freq_mask_config: FreqencyMaskingConfig = freq_mask_config
        self.freq_domain_frames = []
        self._freq_mask: np.ndarray = None
        self.output_enable: bool = freq_mask_config.output_result

    @property
    def freq_mask(self) -> np.ndarray:
        """
        Get the frequency mask.
        """
        if self._freq_mask is None:
            self._freq_mask = self.freq_mask_helper.gen_freq_mask(
                freq_mask_config=self.freq_mask_config,
            )
        return self._freq_mask

    @property
    def freq_mask_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the frequency mask.
        """
        return NamedFrame(name="freq_mask", static_frame=self.freq_mask * np.iinfo(np.uint8).max)

    @property
    def freq_domain_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the frequency domain.
        """
        return NamedFrame(name="freq_domain", video_frame=self.freq_domain_frames)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The filtered frame and the frequency domain.
        """
        if self.freq_mask_config.enable:
            freq_filtered_frame, frame_freq_domain = self.freq_mask_helper.apply_freq_mask(
                img=frame,
                mask=self.freq_mask,
            )
            self.append_output_frame(freq_filtered_frame)
            self.freq_domain_frames.append(frame_freq_domain)

            return freq_filtered_frame, frame_freq_domain
        else:
            return frame, None

    def export_freq_domain_frames(self) -> None:
        """
        Export the frequency domain to a file.
        """
        if self.freq_mask_config.output_freq_domain:
            logger.info(f"Exporting {self.name} frequency domain to {self.output_dir}")
            self.freq_domain_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} frequency domain output disabled.")

    def export_freq_mask(self) -> None:
        """
        Export the frequency mask to a file.
        """
        if self.freq_mask_config.output_mask:
            logger.info(f"Exporting {self.name} frequency mask to {self.output_dir}")
            self.freq_mask_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} frequency mask output disabled.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_freq_mask()
        self.export_freq_domain_frames()


class PassThroughProcessor(BaseVideoProcessor):
    """
    A class to pass through a video.
    """

    def __init__(self, name: str, output_dir: Path):
        """
        Initialize the PassThroughProcessor object.

        Parameters:
        name (str): The name of the video processor.
        output_dir (Path): The output directory.

        Returns:
        PassThroughProcessor: A PassThroughProcessor object.
        """
        super().__init__(name, output_dir)

    @property
    def pass_through_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the pass through.
        """
        return NamedFrame(name=self.name, video_frame=self.output_frames)

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        np.ndarray: The processed frame.
        """
        self.append_output_frame(input_frame)
        return input_frame

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()


class MinProjSubtractProcessor(BaseVideoProcessor):
    """
    A class to apply minimum projection to a video.
    """

    def __init__(
        self,
        name: str,
        minimum_projection_config: MinimumProjectionConfig,
        output_dir: Path,
        video_frames: list[np.ndarray],
    ):
        """
        Initialize the MinimumProjectionProcessor object.

        Parameters:
        name (str): The name of the video processor.
        output_dir (Path): The output directory.

        Returns:
        MinimumProjectionProcessor: A MinimumProjectionProcessor object.
        """
        super().__init__(name, output_dir)
        self.minimum_projection: np.ndarray = ZStackHelper.get_minimum_projection(video_frames)
        self.output_frames: list[np.ndarray] = [
            (frame - self.minimum_projection) for frame in video_frames
        ]
        self.minimum_projection_config: MinimumProjectionConfig = minimum_projection_config
        self.output_enable: bool = minimum_projection_config.output_result

    @property
    def min_proj_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the minimum projection.
        """
        return NamedFrame(name="min_proj", static_frame=self.output_frames[0])

    def normalize_stack(self) -> None:
        """
        Normalize the stack of images.
        """
        self.output_frames = ZStackHelper.normalize_video_stack(self.output_frames)

    def export_minimum_projection(self) -> None:
        """
        Export the minimum projection to a file.
        """

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_minimum_projection()


class VideoProcessor:
    """
    A class to process video files.
    """

    @staticmethod
    def denoise(
        video_path: str,
        config: DenoiseConfig,
    ) -> None:
        """
        Preprocess a video file and display the results.
        """
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is not a required dependency of miniscope-io, to use it, "
                "install it manually or install miniscope-io with `pip install miniscope-io[plot]`"
            )

        reader = VideoReader(video_path)
        pathstem = Path(video_path).stem

        output_dir = Path.cwd() / config.output_dir
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        raw_frame_processor = PassThroughProcessor(
            name=pathstem + "raw",
            output_dir=output_dir,
        )

        output_frame_processor = PassThroughProcessor(
            name=pathstem + "output",
            output_dir=output_dir,
        )

        noise_patch_processor = NoisePatchProcessor(
            output_dir=output_dir,
            name=pathstem + "patch",
            noise_patch_config=config.noise_patch,
            width=reader.width,
            height=reader.height,
        )

        freq_mask_processor = FreqencyMaskProcessor(
            output_dir=output_dir,
            name=pathstem + "freq_mask",
            freq_mask_config=config.frequency_masking,
            width=reader.width,
            height=reader.height,
        )

        try:
            for index, frame in reader.read_frames():
                if config.end_frame and index > config.end_frame and config.end_frame != -1:
                    break

                raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                input_frame = raw_frame_processor.process_frame(raw_frame)
                patched_frame, _ = noise_patch_processor.process_frame(input_frame)
                freq_masked_frame, _ = freq_mask_processor.process_frame(patched_frame)
                _ = output_frame_processor.process_frame(freq_masked_frame)

        finally:
            reader.release()

            output_frames = output_frame_processor.output_frames

            minimum_projection_processor = MinProjSubtractProcessor(
                name=pathstem + "min_proj",
                output_dir=output_dir,
                video_frames=output_frames,
                minimum_projection_config=config.minimum_projection,
            )
            minimum_projection_processor.normalize_stack()

            noise_patch_processor.batch_export_videos()
            freq_mask_processor.batch_export_videos()
            minimum_projection_processor.batch_export_videos()

            if config.interactive_display.enable:
                videos = [
                    raw_frame_processor.pass_through_named_frame,
                    noise_patch_processor.noise_patch_named_frame,
                    noise_patch_processor.output_named_frame,
                    freq_mask_processor.output_named_frame,
                    freq_mask_processor.freq_mask_named_frame,
                    minimum_projection_processor.min_proj_named_frame,
                    freq_mask_processor.freq_domain_named_frame,
                ]
                VideoPlotter.show_video_with_controls(
                    videos,
                    start_frame=config.interactive_display.start_frame,
                    end_frame=config.interactive_display.end_frame,
                )
