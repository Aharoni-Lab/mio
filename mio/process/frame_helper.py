"""
This module contains a helper class for frame operations.
It should be organized in a different way to make it more readable and maintainable.
"""

from typing import Tuple

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

        Parameters:
            height (int): Height of the video frame.
            width (int): Width of the video frame.

        Returns:
            NoiseDetectionHelper: A NoiseDetectionHelper object
        """
        self.height = height
        self.width = width

    def split_by_length(self, array: np.ndarray, segment_length: int) -> list[np.ndarray]:
        """
        Split an array into sub-arrays of a specified length.
        Last sub-array may be shorter if the array length is not a multiple of the segment length.

        Parameters:
            array (np.ndarray): The array to split.
            segment_length (int): The length of each sub-array.

        Returns:
            list[np.ndarray]: A list of sub-arrays.
        """
        num_segments = len(array) // segment_length

        # Split the array into segments of the specified length
        sub_arrays = [
            array[i * segment_length : (i + 1) * segment_length] for i in range(num_segments)
        ]

        # Add the remaining elements as a final shorter segment, if any
        if len(array) % segment_length != 0:
            sub_arrays.append(array[num_segments * segment_length :])

        return sub_arrays

    def detect_frame_with_noisy_buffer(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Unified noise detection method that supports multiple detection algorithms
        (mean_error, gradient, etc.).

        Parameters:
            current_frame (np.ndarray): The current frame to process.
            previous_frame (Optional[np.ndarray]): The previous frame to compare against.
            noise_patch_config (NoisePatchConfig): Configuration for noise detection.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy,
                and a spatial mask showing noisy regions.
        """
        logger.debug(f"Buffer size: {config.buffer_size}")

        serialized_current = current_frame.flatten().astype(np.int16)
        logger.debug(f"Serialized current frame size: {len(serialized_current)}")

        if previous_frame is not None:
            serialized_previous = previous_frame.flatten().astype(np.int16)
            logger.debug(f"Serialized previous frame size: {len(serialized_previous)}")
        else:
            serialized_previous = None
            logger.debug("Previous frame is None.")

        split_current = self.split_by_length(serialized_current, config.buffer_size)

        if previous_frame is not None:
            split_previous = self.split_by_length(serialized_previous, config.buffer_size)
            logger.debug(f"Split previous frame into {len(split_previous)} segments.")
        if config.method == "mean_error" and previous_frame is not None:
            return self._detect_with_mean_error(split_current, split_previous, config)
        elif config.method == "gradient":
            return self._detect_with_gradient(current_frame, config)
        else:
            logger.error(f"Unsupported noise detection method: {config.method}")
            raise ValueError(f"Unsupported noise detection method: {config.method}")

    def _detect_with_mean_error(
        self,
        split_current: list[np.ndarray],
        split_previous: list[np.ndarray],
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

        # buffer_split (splitting each block in smaller segments) cannot be greater than
        # number of buffers in a frame
        if config.buffer_split > len(split_current):
            logger.warning(
                f"buffer_split ({config.buffer_split}) exceeds total splits "
                f"({len(split_current)}). Adjusting to {len(split_current)}."
            )
            config.buffer_split = len(split_current)

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
                    logger.info(f"Buffer {i} exceeds threshold ({config.threshold}): {mean_error}")
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
        noise_output = np.concatenate(noisy_parts)[: self.height * self.width]
        noise_patch = noise_output.reshape((self.height, self.width))

        return any_buffer_has_noise, noise_patch

    def _detect_with_gradient(
        self,
        current_frame: np.ndarray,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect noise using local contrast (second derivative) within blocks.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        height, width = current_frame.shape

        block_height = config.buffer_size // width  # Block spans the entire width
        block_height = max(1, block_height)  # Ensure at least one row per block

        noisy_mask = np.zeros_like(current_frame, dtype=np.uint8)

        block_index = 0  # used for debug logging

        # Slide through the frame vertically in block_height steps
        for y in range(0, height, block_height):
            # select block and cast to int16 to avoid diffs wrapping around 0
            block = current_frame[y : y + block_height, :].astype(np.int16)

            # Skip blocks that exceed the frame boundary
            if block.shape[0] < block_height:
                continue

            # second derivative in x & y for local contrast
            diff_x = np.diff(block, axis=1)
            diff_y = np.diff(block, axis=0)
            # second derivative (change of gradients)
            second_diff_x = np.diff(diff_x, axis=1)
            second_diff_y = np.diff(diff_y, axis=0)

            mean_second_diff = (np.abs(second_diff_x).mean() + np.abs(second_diff_y).mean()) / 2
            logger.debug(f"Mean second derivative for block {block_index}: {mean_second_diff}")

            # Flag block as noisy if contrast exceeds the threshold
            if mean_second_diff > config.threshold:
                noisy_mask[y : y + block_height, :] = 1

            block_index += 1  # used for debug logging

        # Determine if the frame is noisy (if any blocks are marked as noisy)
        frame_is_noisy = noisy_mask.any()

        return frame_is_noisy, noisy_mask


class FrequencyMaskHelper:
    """
    Helper class for frame operations.


    """

    def __init__(self, height: int, width: int):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Parameters:
            height (int): Height of the video frame.
            width (int): Width of the video frame.

        Returns:
            FrequencyMaskHelper: A FrequencyMaskHelper object.
        """
        self.height = height
        self.width = width

    def apply_freq_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
        freq_mask_config: FreqencyMaskingConfig,
    ) -> np.ndarray:
        """
        Generate a mask to filter out horizontal and vertical frequencies.
        A central circular region can be removed to allow low frequencies to pass.
        """
        crow, ccol = self.height // 2, self.width // 2

        # Create an initial mask filled with ones (pass all frequencies)
        mask = np.ones((self.height, self.width), np.uint8)

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
        y, x = np.ogrid[: self.height, : self.width]
        center_mask = (x - ccol) ** 2 + (
            y - crow
        ) ** 2 <= freq_mask_config.spatial_LPF_cutoff_radius**2

        # Restore the center circular area to allow low frequencies to pass
        mask[center_mask] = 1

        # Visualize the mask if needed. Might delete later.
        if freq_mask_config.display_mask:
            cv2.imshow("Mask", mask * np.iinfo(np.uint8).max)
            while True:
                if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit visualization
                    break
            cv2.destroyAllWindows()
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
