"""
This module contains a helper class for frame operations.
It should be organized in a different way to make it more readable and maintainable.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from mio import init_logger
from mio.models.process import FreqencyMaskingConfig, NoisePatchConfig

logger = init_logger("frame_helper")


class NoiseDetectionHelper:
    """
    Helper class for noise detection and frame processing.
    """

    def __init__(self, height: int, width: int):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Returns:
            NoiseDetectionHelper: A NoiseDetectionHelper object
        """

    def detect_frame_with_noisy_buffer(
        self,
        current_frame: np.ndarray,
        previous_frame: Optional[np.ndarray],
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Unified noise detection method that supports multiple detection algorithms
        (mean_error, gradient, etc.).

        Parameters:
            current_frame (np.ndarray): The current frame to process.
            previous_frame (Optional[np.ndarray]): The previous frame to compare against.
            config (NoisePatchConfig): Configuration for noise detection.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy,
                and a spatial mask showing noisy regions.
        """
        logger.debug(f"Buffer size: {config.buffer_size}")

        frame_width = current_frame.shape[1]
        frame_height = current_frame.shape[0]

        if config.method == "mean_error":

            if previous_frame is None:
                raise ValueError("mean_error requires a previous frame to compare against")

            serialized_current = current_frame.flatten().astype(np.int16)
            logger.debug(f"Serialized current frame size: {len(serialized_current)}")

            serialized_previous = previous_frame.flatten().astype(np.int16)
            logger.debug(f"Serialized previous frame size: {len(serialized_previous)}")

            split_size = config.buffer_size // config.buffer_split + 1

            split_shape = []

            pixel_index = 0
            while pixel_index < len(serialized_current):
                split_shape.append[split_size]
                pixel_index += split_size

            split_previous = np.split(serialized_previous, split_shape)
            split_current = np.split(serialized_current, split_shape)

            return self._detect_with_mean_error(
                split_current=split_current,
                split_previous=split_previous,
                width=frame_width,
                height=frame_height,
                config=config,
            )

        elif config.method == "gradient":
            return self._detect_with_gradient(current_frame, config)
        else:
            logger.error(f"Unsupported noise detection method: {config.method}")
            raise ValueError(f"Unsupported noise detection method: {config.method}")

    def _detect_with_mean_error(
        self,
        split_current: list[np.ndarray],
        split_previous: list[np.ndarray],
        width: int,
        height: int,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect noise using mean error between current and previous buffers.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        noisy_parts = split_current.copy()
        any_buffer_has_noise = False
        current_buffer_has_noise = False

        logger.debug(f"Config buffer_split: {config.buffer_split}")
        logger.debug(
            f"Actual total splits in current: {len(split_current)}, previous: {len(split_previous)}"
        )

        # Iterate over buffers and split sections
        logger.debug(
            f"Entering mean_error loop: Total splits: {len(split_current)}, "
            f"Buffer split: {config.buffer_split}"
        )
        for buffer_index in range(len(split_current) // config.buffer_split):
            for split_index in range(config.buffer_split):
                i = buffer_index * config.buffer_split + split_index

                # Calculate mean error for each split section
                mean_error = abs(split_current[i] - split_previous[i]).mean()
                logger.debug(
                    f"Mean error for buffer {i}: {mean_error}, Threshold: {config.threshold}"
                )

                if mean_error > config.threshold:
                    logger.debug(f"Buffer {i} exceeds threshold ({config.threshold}): {mean_error}")
                    current_buffer_has_noise = True
                    any_buffer_has_noise = True
                    break
                else:
                    noisy_parts[i] = np.zeros_like(split_current[i], np.uint8)

            # Mark noisy blocks
            if current_buffer_has_noise:
                for split_index in range(config.buffer_split):
                    i = buffer_index * config.buffer_split + split_index
                    noisy_parts[i] = np.ones_like(split_current[i], np.uint8)
                current_buffer_has_noise = False

        # Create a noise mask for visualization
        noise_output = np.concatenate(noisy_parts)[: height * width]
        noise_patch = noise_output.reshape((height, width))

        return any_buffer_has_noise, noise_patch

    def _detect_with_gradient(
        self,
        current_frame: np.ndarray,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect noise using local contrast (second derivative) in the x-dimension
        (along rows, across columns)

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        noisy_mask = np.zeros_like(current_frame, dtype=np.uint8)

        diff_x = np.diff(current_frame.astype(np.int16), n=2, axis=1)
        mean_second_diff = np.abs(diff_x).mean(axis=1)
        noisy_mask[mean_second_diff > config.threshold, :] = 1
        logger.debug("Row-wise means of second derivative: %s", mean_second_diff)

        # Determine if the frame is noisy (if any rows are marked as noisy)
        frame_is_noisy = noisy_mask.any()

        return frame_is_noisy, noisy_mask


class FrequencyMaskHelper:
    """
    Helper class for frame operations.
    """

    def __init__(self):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Returns:
            FrequencyMaskHelper: A FrequencyMaskHelper object.
        """

    def apply_freq_mask(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform FFT/IFFT to remove horizontal stripes from a single frame.

        Parameters:
            img (np.ndarray): The image to process.
            mask (np.ndarray): The frequency mask to apply.

        Returns:
            np.ndarray: The filtered image
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Use log for better visualization

        # Normalize the magnitude spectrum for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
        )

        # Apply mask and inverse FFT
        fshift *= mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back), np.uint8(magnitude_spectrum)

    def gen_freq_mask(
        self,
        width: int,
        height: int,
        freq_mask_config: FreqencyMaskingConfig,
    ) -> np.ndarray:
        """
        Generate a mask to filter out horizontal and vertical frequencies.
        A central circular region can be removed to allow low frequencies to pass.
        """
        crow, ccol = height // 2, width // 2

        # Create an initial mask filled with ones (pass all frequencies)
        mask = np.ones((height, width), np.uint8)

        # Zero out a vertical stripe at the frequency center
        mask[
            :,
            ccol
            - freq_mask_config.vertical_BEF_cutoff : ccol
            + freq_mask_config.vertical_BEF_cutoff,
        ] = 0

        # Zero out a horizontal stripe at the frequency center
        mask[
            crow
            - freq_mask_config.horizontal_BEF_cutoff : crow
            + freq_mask_config.horizontal_BEF_cutoff,
            :,
        ] = 0

        # Define spacial low pass filter
        y, x = np.ogrid[:height, :width]
        center_mask = (x - ccol) ** 2 + (
            y - crow
        ) ** 2 <= freq_mask_config.spatial_LPF_cutoff_radius**2

        # Restore the center circular area to allow low frequencies to pass
        mask[center_mask] = 1

        return mask


class ZStackHelper:
    """
    Helper class for Z-stack operations.
    """

    @staticmethod
    def get_minimum_projection(image_list: list[np.ndarray]) -> np.ndarray:
        """
        Get the minimum projection of a list of images.

        Parameters:
            image_list (list[np.ndarray]): A list of images to project.

        Returns:
            np.ndarray: The minimum projection of the images.
        """
        stacked_images = np.stack(image_list, axis=0)
        min_projection = np.min(stacked_images, axis=0)
        return min_projection

    @staticmethod
    def normalize_video_stack(image_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        Normalize a stack of images to 0-255 using max and minimum values of the entire stack.
        Return a list of images.

        Parameters:
            image_list (list[np.ndarray]): A list of images to normalize.

        Returns:
            list[np.ndarray]: The normalized images as a list.
        """

        # Stack images along a new axis (axis=0)
        stacked_images = np.stack(image_list, axis=0)

        # Find the global min and max across the entire stack
        global_min = stacked_images.min()
        global_max = stacked_images.max()

        range_val = max(global_max - global_min, 1e-5)  # Set an epsilon value for stability

        # Normalize each frame using the global min and max
        normalized_images = []
        for i in range(stacked_images.shape[0]):
            normalized_image = cv2.normalize(
                stacked_images[i],
                None,
                0,
                np.iinfo(np.uint8).max,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            normalized_image = (stacked_images[i] - global_min) / range_val * np.iinfo(np.uint8).max
            normalized_images.append(normalized_image.astype(np.uint8))

        return normalized_images
