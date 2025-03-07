"""
This module contains functions for pre-processing video data.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.signal import butter, filtfilt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from mio import init_logger
from mio.io import VideoReader
from mio.models.frames import NamedFrame, NamedVideo
from mio.models.process import (
    ButterworthFilterConfig,
    DenoiseConfig,
    FreqencyMaskingConfig,
    MinimumProjectionConfig,
    NoisePatchConfig,
)
from mio.plots.video import VideoPlotter
from mio.process.frame_helper import FrequencyMaskHelper, InvalidFrameDetector, mean_intensity
from mio.process.zstack_helper import ZStackHelper

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
        self.output_video: list[np.ndarray] = []
        self.output_enable: bool = True

    @property
    def output_named_video(self) -> NamedVideo:
        """
        Get the output NamedFrame object.

        Returns:
        NamedVideo: The output NamedVideo object.
        """
        return NamedVideo(name=self.name, video=self.output_video)

    def append_output_frame(self, input_frame: np.ndarray) -> None:
        """
        Append a frame to the output_frames list.

        Parameters:
        frame (np.ndarray): The frame to append.
        """
        self.output_video.append(input_frame)

    def export_output_video(self) -> None:
        """
        Export the video to a file.
        """
        if self.output_enable:
            logger.info(f"Exporting {self.name} video to {self.output_dir}")
            self.output_named_video.export(
                output_path=self.output_dir / self.name,
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
        output_dir: Path,
    ) -> None:
        """
        Initialize the NoisePatchProcessor object.

        Parameters:
        name (str): The name of the video processor.
        noise_patch_config (NoisePatchConfig): The noise patch configuration.
        """
        super().__init__(name, output_dir)
        self.noise_patch_config: NoisePatchConfig = noise_patch_config
        self.noise_detect_helper = InvalidFrameDetector(noise_patch_config=noise_patch_config)
        self.noise_patchs: list[np.ndarray] = []
        self.noisy_frames: list[np.ndarray] = []
        self.diff_frames: list[np.ndarray] = []
        self.dropped_frame_indices: list[int] = []

        self.output_enable: bool = noise_patch_config.output_result

        if "mean_error" in noise_patch_config.method:
            logger.warning(
                "The mean_error method is unstable and not fully tested yet." " Use with caution."
            )

    def process_frame(self, input_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Parameters:
        raw_frame (np.ndarray): The raw frame to process.

        Returns:
        Optional[np.ndarray]: The processed frame. If the frame is noisy, a None is returned.
        """
        if input_frame is None:
            return None

        if self.noise_patch_config.enable:
            invalid, noisy_area = self.noise_detect_helper.find_invalid_area(input_frame)

            # Handle noisy frames
            if not invalid:
                self.append_output_frame(input_frame)
                return input_frame
            else:
                index = len(self.output_video) + len(self.noise_patchs)
                logger.info(f"Dropping frame {index} of original video due to noise.")
                logger.debug(f"Adding noise patch for frame {index}.")
                self.noise_patchs.append((noisy_area * np.iinfo(np.uint8).max).astype(np.uint8))
                self.noisy_frames.append(input_frame)
                self.dropped_frame_indices.append(index)
            return None

        self.append_output_frame(input_frame)
        return input_frame

    @property
    def noise_patch_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the noise patch.
        """
        return NamedVideo(name="patched_area", video=self.noise_patchs)

    @property
    def diff_frames_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the difference frames.
        """
        if not hasattr(self.noise_patch_config, "diff_multiply"):
            diff_multiply = 1
        return NamedVideo(name=f"diff_{diff_multiply}x", video=self.diff_frames)

    @property
    def noisy_frames_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the noisy frames.
        """
        return NamedVideo(name="noisy_frames", video=self.noisy_frames)

    def export_noise_patch(self) -> None:
        """
        Export the noise patch to a file.
        """
        if not self.noise_patchs:
            logger.info(f"No noise patches to export for {self.name}.")
            return

        if self.noise_patch_config.output_noise_patch:
            logger.info(f"Exporting {self.name} noise patch to {self.output_dir}")
            self.noise_patch_named_video.export(
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
            self.diff_frames_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} difference frames output disabled.")

    def export_noisy_video(self) -> None:
        """
        Export the noisy frames to a file.
        """
        if self.noise_patch_config.output_noisy_frames:
            logger.info(f"Exporting {self.name} noisy frames to {self.output_dir}")
            self.noisy_frames_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
            # Can be anything. Just for now.
            with open(self.output_dir / f"{self.name}_dropped_frames.txt", "w") as f:
                for index in self.dropped_frame_indices:
                    f.write(f"{index}\n")
        else:
            logger.info(f"{self.name} noisy frames output disabled.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_noise_patch()
        self.export_diff_frames()
        self.export_noisy_video()


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
        self.freq_mask_config: FreqencyMaskingConfig = freq_mask_config
        self.freq_mask_helper = FrequencyMaskHelper(
            height=height, width=width, freq_mask_config=freq_mask_config
        )
        self.freq_domain_frames = []
        self.frame_width: int = width
        self.frame_height: int = height
        self.output_enable: bool = freq_mask_config.output_result

    @property
    def freq_mask(self) -> np.ndarray:
        """
        Get the frequency mask.
        """
        return self.freq_mask_helper.freq_mask

    @property
    def freq_mask_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the frequency mask.
        """
        return NamedFrame(name="freq_mask", frame=self.freq_mask * np.iinfo(np.uint8).max)

    @property
    def freq_domain_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the frequency domain.
        """
        return NamedVideo(name="freq_domain", video=self.freq_domain_frames)

    def process_frame(self, input_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        Optional[np.ndarray]: The processed frame. If the input is none, a None is returned.
        """
        if input_frame is None:
            return None
        if self.freq_mask_config.enable:
            freq_filtered_frame = self.freq_mask_helper.process_frame(img=input_frame)
            frame_freq_domain = self.freq_mask_helper.freq_domain(img=input_frame)
            self.append_output_frame(freq_filtered_frame)
            self.freq_domain_frames.append(frame_freq_domain)

            return freq_filtered_frame
        else:
            return input_frame

    def export_freq_domain_frames(self) -> None:
        """
        Export the frequency domain to a file.
        """
        if self.freq_mask_config.output_freq_domain:
            logger.info(f"Exporting {self.name} frequency domain to {self.output_dir}")
            self.freq_domain_named_video.export(
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
    def pass_through_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the pass through.
        """
        return NamedVideo(name=self.name, video=self.output_video)

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        np.ndarray: The processed frame.
        """
        if input_frame is None:
            return None
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

        if not video_frames:
            logger.warning("No frames provided for minimum projection. Skipping processing.")
            self.minimum_projection = None
            self.output_frames = []
        else:
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
        return NamedFrame(name="min_proj", frame=self.output_frames[0])

    def normalize_stack(self) -> None:
        """
        Normalize the stack of images.
        """
        if not self.output_frames:
            logger.warning(
                "No frames available in output_frames for normalization. Skipping normalization."
            )
            return

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


class ButterworthProcessor(BaseVideoProcessor):
    """
    Processor for applying a Butterworth filter to video frames.

    This class applies a Butterworth filter to each frame in the video,
    using the specified Butterworth filter configuration.
    """

    def __init__(self, name: str, butter_config: ButterworthFilterConfig, output_dir: Path):
        """
        Initialize the ButterworthProcessor.
        """

        super().__init__(name, output_dir)
        self.config = butter_config
        self.output_enable = butter_config.enable and SCIPY_AVAILABLE
        self.intensity_data = []
        self.frames = []
        self.output_video = []  # Add this to store processed frames

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using Butterworth filter.
        """

        if input_frame is None:
            return None

        mean_int = mean_intensity(input_frame)
        self.intensity_data.append(mean_int)
        self.frames.append(input_frame)
        self.output_video.append(input_frame)  # Store original frame for later filtering

        return input_frame

    def apply_filter(self) -> np.ndarray:
        """Apply the Butterworth filter to the collected intensity data."""
        if not self.intensity_data:
            return np.array([])

        data = np.array(self.intensity_data)
        return self.butter_lowpass_filter(
            data=data,
            cutoff=self.config.cutoff_frequency,
            fs=self.config.sampling_rate,
            order=self.config.order,
        )

    def butter_lowpass_filter(
        self, data: np.ndarray, cutoff: float, fs: float, order: int
    ) -> np.ndarray:
        """Apply a Butterworth lowpass filter to the data."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)
        return y

    def apply_filter_to_frames(self, filtered_data: np.ndarray) -> list[np.ndarray]:
        """Apply the filtered intensity values to scale the frames.

        The scaling is done by multiplying each frame by the ratio of filtered to original intensity
        Values are properly scaled and can optionally be clipped to the valid uint8 range.
        """
        if len(filtered_data) != len(self.frames):
            logger.warning("Filtered data length doesn't match frame count")
            return []

        filtered_frames = []
        for i, frame in enumerate(self.frames):
            # Scale the frame based on filtered vs original intensity
            if self.intensity_data[i] != 0:  # Avoid division by zero
                scale_factor = filtered_data[i] / self.intensity_data[i]
                # Apply scaling in float32 precision
                filtered_frame = frame.astype(np.float32) * scale_factor

                # Log warning if values will be clipped
                if np.any(filtered_frame < 0) or np.any(filtered_frame > 255):
                    logger.warning(
                        f"Frame {i}: Values outside valid range "
                        f"(min={filtered_frame.min():.1f}, max={filtered_frame.max():.1f})"
                    )

                # Clip to valid range and convert to uint8
                filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)
            else:
                filtered_frame = frame
            filtered_frames.append(filtered_frame)

        return filtered_frames

    def batch_export_videos(self) -> None:
        """Export the filtered data and plot if enabled."""
        if not self.output_enable:
            logger.info("Butterworth filter disabled, skipping export")
            return

        logger.info(f"Processing {len(self.frames)} frames with Butterworth filter")
        filtered_data = self.apply_filter()
        if len(filtered_data) > 0:
            np.save(self.output_dir / f"{self.name}_filtered_intensity.npy", filtered_data)

            filtered_frames = self.apply_filter_to_frames(filtered_data)
            if filtered_frames:
                logger.info("Saving Butterworth filtered video")
                # Store the filtered frames in output_video for display
                self.output_video = filtered_frames
                output_video = NamedVideo(
                    name=self.name,
                    video=filtered_frames,
                )
                # Save directly in output_dir with high quality GREY codec
                output_video.export(
                    output_path=self.output_dir / self.name,
                    fps=self.config.sampling_rate,
                    suffix=False,
                    codec="GREY",
                    isColor=False,  # Explicitly specify grayscale
                )
            else:
                logger.warning("No frames to save after Butterworth filtering")

            if self.config.plot:
                self.plot_filtered_data(filtered_data)

    def plot_filtered_data(self, filtered_data: np.ndarray) -> None:
        """Plot the original and filtered intensity data."""
        if not self.config.plot or plt is None:
            return

        start_frame = self.config.plot_start_frame or 0
        end_frame = self.config.plot_end_frame or len(self.intensity_data)

        start_frame = max(0, min(start_frame, len(self.intensity_data) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(self.intensity_data)))

        frame_indices = np.arange(start_frame, end_frame)
        original_data = np.array(self.intensity_data)[start_frame:end_frame]
        filtered_data_slice = filtered_data[start_frame:end_frame]

        plt.figure(figsize=(10, 6))
        plt.plot(frame_indices, original_data, "b-", label="Mean Intensity", linewidth=1, alpha=0.7)
        plt.plot(
            frame_indices,
            filtered_data_slice,
            "orange",
            label=f"Filtered (Order={self.config.order})",
            linewidth=2,
        )

        plt.title(f"Mean Intensity per Frame ({start_frame} to {end_frame})")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean Intensity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(
            self.output_dir / f"{self.name}_intensity_plot.png", dpi=300, bbox_inches="tight"
        )

        if plt.get_backend() != "agg":
            plt.savefig("temp_plot.png")  # Save temporarily
            plot_img = cv2.imread("temp_plot.png")
            cv2.imshow("Butterworth Filter Plot", plot_img)
            while True:
                if cv2.waitKey(1) == 27:  # Wait for ESC key
                    break
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Extra waitKey to properly close the window
            Path("temp_plot.png").unlink()  # Remove temporary file

        plt.close()


def plot_video_comparison(
    spatial_video: list[np.ndarray],
    butter_video: list[np.ndarray],
    num_frames: int,
    output_dir: Path,
    name: str,
    start_frame: int = 0,
) -> None:
    """Plot comparison between spatially filtered and Butterworth filtered videos.

    Parameters:
    -----------
    spatial_video : list[np.ndarray]
        Video after spatial filtering
    butter_video : list[np.ndarray]
        Video after Butterworth filtering
    num_frames : int
        Number of frames to plot
    output_dir : Path
        Directory to save the plot
    name : str
        Base name for the plot file
    start_frame : int
        First frame to include in plot
    """
    if plt is None:
        return

    # Calculate mean pixel values for both videos
    spatial_means = [
        np.mean(frame) for frame in spatial_video[start_frame : start_frame + num_frames]
    ]
    butter_means = [
        np.mean(frame) for frame in butter_video[start_frame : start_frame + num_frames]
    ]

    # Calculate pixel-wise difference
    diff_means = [abs(s - b) for s, b in zip(spatial_means, butter_means)]

    frame_indices = np.arange(start_frame, start_frame + num_frames)

    plt.figure(figsize=(12, 8))

    # Plot mean pixel values
    plt.subplot(2, 1, 1)
    plt.plot(frame_indices, spatial_means, "b-", label="Spatial Filter", linewidth=2)
    plt.plot(frame_indices, butter_means, "r-", label="Butterworth Filter", linewidth=2)
    plt.title(f"Mean Pixel Values Comparison (Frames {start_frame} to {start_frame + num_frames})")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Pixel Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot absolute difference
    plt.subplot(2, 1, 2)
    plt.plot(frame_indices, diff_means, "g-", label="Absolute Difference", linewidth=2)
    plt.title("Absolute Difference in Mean Pixel Values")
    plt.xlabel("Frame Index")
    plt.ylabel("Difference")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_comparison_plot.png", dpi=300, bbox_inches="tight")

    if plt.get_backend() != "agg":
        plt.show()

    plt.close()


def denoise_run(
    video_path: str,
    config: DenoiseConfig,
) -> None:
    """
    Preprocess a video file and display the results.

    Parameters:
    video_path (str): The path to the video file.
    config (DenoiseConfig): The denoise configuration.
    """
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is not a required dependency of miniscope-io, to use it, "
            "install it manually or install miniscope-io with `pip install miniscope-io[plot]`"
        )

    # Check for scipy if Butterworth filter is enabled
    if config.butter_filter and config.butter_filter.enable and not SCIPY_AVAILABLE:
        logger.warning(
            "Butterworth filter is enabled but scipy is not installed. "
            "Install with 'pip install miniscope-io[signal]' to use the filter. "
            "Continuing without Butterworth filtering..."
        )

    reader = VideoReader(video_path)
    pathstem = Path(video_path).stem

    output_dir = Path.cwd() / config.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    raw_frame_processor = PassThroughProcessor(
        name=pathstem + "_raw",
        output_dir=output_dir,
    )

    output_frame_processor = PassThroughProcessor(
        name=pathstem + "_output",
        output_dir=output_dir,
    )

    noise_patch_processor = NoisePatchProcessor(
        output_dir=output_dir,
        name=pathstem + "_patch",
        noise_patch_config=config.noise_patch,
    )

    freq_mask_processor = FreqencyMaskProcessor(
        output_dir=output_dir,
        name=pathstem + "_freq_mask",
        freq_mask_config=config.frequency_masking,
        width=reader.width,
        height=reader.height,
    )

    butter_processor = ButterworthProcessor(
        name=pathstem + "_butter_filter",
        butter_config=config.butter_filter,
        output_dir=output_dir,
    )

    if config.interactive_display.display_freq_mask:
        freq_mask_processor.freq_mask_named_frame.display()

    try:
        for index, frame in reader.read_frames():
            if config.end_frame and index > config.end_frame and config.end_frame != -1:
                break

            raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_frame = raw_frame_processor.process_frame(raw_frame)

            # Stage 1: Noise Patch
            if config.noise_patch and config.noise_patch.enable:
                patched_frame = noise_patch_processor.process_frame(input_frame)
            else:
                patched_frame = input_frame

            # Stage 2: Frequency Mask
            if config.frequency_masking and config.frequency_masking.enable:
                freq_masked_frame = freq_mask_processor.process_frame(patched_frame)
            else:
                freq_masked_frame = patched_frame

            # Stage 3: Butterworth
            if config.butter_filter and config.butter_filter.enable:
                logger.debug("Processing frame through Butterworth filter")
                butter_frame = butter_processor.process_frame(freq_masked_frame)
            else:
                butter_frame = freq_masked_frame

            _ = output_frame_processor.process_frame(butter_frame)

    finally:
        reader.release()
        logger.info("Processing complete, exporting results...")

        output_frames = output_frame_processor.output_video

        # First export all the intermediate results
        noise_patch_processor.batch_export_videos()
        freq_mask_processor.batch_export_videos()

        # Then do the Butterworth processing and export
        if config.butter_filter and config.butter_filter.enable:
            logger.info("Exporting Butterworth filter results...")
            butter_processor.batch_export_videos()

        # Finally do minimum projection
        minimum_projection_processor = MinProjSubtractProcessor(
            name=pathstem + "min_proj",
            output_dir=output_dir,
            video_frames=output_frames,
            minimum_projection_config=config.minimum_projection,
        )
        minimum_projection_processor.normalize_stack()
        minimum_projection_processor.batch_export_videos()

        if len(noise_patch_processor.output_named_video.video) == 0:
            logger.warning("No output video available for display.")
        elif (
            len(noise_patch_processor.output_named_video.video)
            < config.interactive_display.end_frame
        ):
            logger.warning(
                f"Output video has {len(noise_patch_processor.output_named_video.video)} frames."
                f" End frame for interactive plot is {config.interactive_display.end_frame}."
                " End frame for interactive plot exceeds the number of frames in the video."
                " Skipping interactive display."
            )
        elif config.interactive_display.show_videos:
            start = config.interactive_display.start_frame
            end = config.interactive_display.end_frame

            videos = [
                noise_patch_processor.output_named_video,
                freq_mask_processor.output_named_video,
                NamedVideo(
                    name=pathstem + "_butter_filter",
                    video=butter_processor.output_video[start:end],
                ),
                freq_mask_processor.freq_domain_named_video,
                minimum_projection_processor.min_proj_named_frame,
            ]
            video_plotter = VideoPlotter(
                videos=videos,
                start_frame=config.interactive_display.start_frame,
                end_frame=config.interactive_display.end_frame,
            )
            video_plotter.show()

            # Add comparison plot using the same frame range as Butterworth filter plot
            plot_video_comparison(
                spatial_video=freq_mask_processor.output_video,
                butter_video=butter_processor.output_video,
                num_frames=config.butter_filter.plot_end_frame
                - config.butter_filter.plot_start_frame,
                output_dir=output_dir,
                name=pathstem,
                start_frame=config.butter_filter.plot_start_frame,
            )
