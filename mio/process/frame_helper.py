"""
This module contains a helper class for frame operations.
It should be organized in a different way to make it more readable and maintainable.
"""

from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from mio import init_logger
from mio.models.process import FreqencyMaskingConfig, NoisePatchConfig

logger = init_logger("frame_helper")


class SingleFrameHelper:
    """
    Helper class for single frame operations.
    """

    def __init__(self):
        """
        Initialize the SingleFrameHelper object.

        Returns:
            SingleFrameHelper: A SingleFrameHelper object.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            np.ndarray: The processed frame.
        """
        pass


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
        if config.device_config is not None:
            px_per_buffer = config.device_config.px_per_buffer
        else:
            px_per_buffer = 1000
            logger.warning(
                f"Device configuration not found. Using default buffer size: {px_per_buffer}"
            )
        logger.debug(f"Buffer size: {px_per_buffer}")

        methods = [config.method]
        if hasattr(config, "additional_methods") and isinstance(config.additional_methods, list):
            methods.extend(config.additional_methods)

        logger.debug(f"Applying noise detection methods: {methods}")

        noisy_flag = False
        combined_mask = np.zeros_like(current_frame, dtype=np.uint8)

        for method in methods:
            if method == "mean_error":
                if previous_frame is None:
                    raise ValueError("mean_error requires a previous frame to compare against")
                return self._detect_with_mean_error(
                    current_frame=current_frame, previous_frame=previous_frame, config=config
                )

            elif method == "gradient":
                noisy, mask = self._detect_with_gradient(current_frame, config)

            elif method == "black_pixels":
                noisy, mask = self._detect_black_pixels(current_frame, config)

            else:
                logger.error(f"Unsupported noise detection method: {method}")
                continue  # Skip unknown methods

            if noisy:
                logger.info(f"Frame detected as noisy using method: {method}")
            else:
                logger.debug(f"Frame passed as clean using method: {method}")

            # Combine results
            noisy_flag = noisy_flag or noisy
            combined_mask = np.maximum(combined_mask, mask)

        return noisy_flag, combined_mask

    def _get_buffer_shape(
        self, frame_width: int, frame_height: int, px_per_buffer: int
    ) -> list[int]:
        """
        Get the shape of each buffer in a frame.

        Parameters:
            frame_width (int): The width of the frame.
            frame_height (int): The height of the frame.
            px_per_buffer (int): The number of pixels per buffer.

        Returns:
            list[int]: The shape of each buffer in the frame.
        """
        buffer_shape = []

        pixel_index = 0
        while pixel_index < frame_width * frame_height:
            buffer_shape.append(int(pixel_index))
            pixel_index += px_per_buffer
        logger.debug(f"Split shape: {buffer_shape}")
        return buffer_shape

    def _detect_with_mean_error(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect noise using mean error between current and previous buffers. This is deprecated now.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        frame_width = current_frame.shape[1]
        frame_height = current_frame.shape[0]

        serialized_current = current_frame.flatten().astype(np.int16)
        serialized_previous = previous_frame.flatten().astype(np.int16)
        buffer_shape = self._get_buffer_shape(
            frame_width, frame_height, config.device_config.px_per_buffer
        )

        noisy_parts = np.ones_like(serialized_current, np.uint8)
        any_buffer_has_noise = False

        for buffer_index in range(len(buffer_shape)):
            buffer_start = 0 if buffer_index == 0 else buffer_shape[buffer_index]
            buffer_end = (
                frame_width * frame_height
                if buffer_index == len(buffer_shape) - 1
                else buffer_shape[buffer_index + 1]
            )

            comparison_start = buffer_end - config.buffer_split
            while comparison_start > buffer_start:
                mean_error = abs(
                    serialized_current[comparison_start:buffer_end]
                    - serialized_previous[comparison_start:buffer_end]
                ).mean()
                logger.debug(
                    f"Mean error for buffer {buffer_index}:"
                    f" pixels {comparison_start}-{buffer_end}: {mean_error}"
                    f" (threshold: {config.threshold})"
                )

                if mean_error > config.threshold:
                    noisy_parts[comparison_start:buffer_end] = np.zeros_like(
                        serialized_current[comparison_start:buffer_end], np.uint8
                    )
                    any_buffer_has_noise = True
                    break
                comparison_start -= config.buffer_split

        noise_patch = noisy_parts.reshape((frame_height, frame_width))
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

    def _detect_black_pixels(
        self,
        current_frame: np.ndarray,
        config: NoisePatchConfig,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect black-out noise by checking for black pixels (value 0) over rows of pixels.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is corrupted and noise mask.
        """
        height, width = current_frame.shape
        noisy_mask = np.zeros_like(current_frame, dtype=np.uint8)

        # Read values from YAML config
        consecutive_threshold = (
            config.black_pixel_consecutive_threshold
        )  # How many consecutive pixels must be black
        black_pixel_value_threshold = (
            config.black_pixel_value_threshold
        )  # Max pixel value considered "black"

        logger.debug(f"Using black pixel threshold: <= {black_pixel_value_threshold}")
        logger.debug(f"Consecutive black pixel threshold: {consecutive_threshold}")

        frame_is_noisy = False  # Track if frame should be discarded

        for y in range(height):
            row = current_frame[y, :]  # Extract row
            consecutive_count = 0  # Counter for consecutive black pixels

            for x in range(width):
                if row[x] <= black_pixel_value_threshold:  # Check if pixel is "black"
                    consecutive_count += 1
                else:
                    consecutive_count = 0  # Reset if a non-black pixel is found

                # If we exceed the allowed threshold of consecutive black pixels, discard the frame
                if consecutive_count >= consecutive_threshold:
                    logger.debug(
                        f"Frame noisy due to {consecutive_count} consecutive black pixels "
                        f"in row {y}."
                    )
                    noisy_mask[y, :] = 1  # Mark row as noisy
                    frame_is_noisy = True
                    break  # No need to check further in this row

        return frame_is_noisy, noisy_mask


class FrequencyMaskHelper(SingleFrameHelper):
    """
    Helper class for frequency masking operations.
    """

    def __init__(self, height: int, width: int, freq_mask_config: FreqencyMaskingConfig):
        """
        Initialize the FreqMaskHelper object and generate a frequency mask.

        Parameters:
            height (int): The height of the image.
            width (int): The width of the image.
            freq_mask_config (FreqencyMaskingConfig): Configuration for frequency masking

        Returns:
            FreqMaskHelper: A FreqMaskHelper object.
        """
        self._height = height
        self._width = width
        self._freq_mask_config = freq_mask_config
        self._freq_mask = self._gen_freq_mask()

    @property
    def freq_mask(self) -> np.ndarray:
        """
        Get the frequency mask.

        Returns:
            np.ndarray: The frequency mask.
        """
        return self._freq_mask

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Perform FFT/IFFT to remove horizontal stripes from a single frame.

        Parameters:
            img (np.ndarray): The image to process.

        Returns:
            np.ndarray: The filtered image
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # Apply mask and inverse FFT
        fshift *= self.freq_mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def freq_domain(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the frequency spectrum of an image.

        Parameters:
            img (np.ndarray): The image to process.

        Returns:
            np.ndarray: The frequency spectrum of the image.
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)

        # Normalize the magnitude spectrum for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
        )

        return np.uint8(magnitude_spectrum)

    def _gen_freq_mask(
        self,
    ) -> np.ndarray:
        """
        Generate a mask to filter out horizontal and vertical frequencies.
        A central circular region can be removed to allow low frequencies to pass.
        """
        crow, ccol = self._height // 2, self._width // 2

        # Create an initial mask filled with ones (pass all frequencies)
        mask = np.ones((self._height, self._width), np.uint8)

        # Zero out a vertical stripe at the frequency center
        mask[
            :,
            ccol
            - self._freq_mask_config.vertical_BEF_cutoff : ccol
            + self._freq_mask_config.vertical_BEF_cutoff,
        ] = 0

        # Zero out a horizontal stripe at the frequency center
        mask[
            crow
            - self._freq_mask_config.horizontal_BEF_cutoff : crow
            + self._freq_mask_config.horizontal_BEF_cutoff,
            :,
        ] = 0

        # Define spacial low pass filter
        y, x = np.ogrid[: self._height, : self._width]
        center_mask = (x - ccol) ** 2 + (
            y - crow
        ) ** 2 <= self._freq_mask_config.spatial_LPF_cutoff_radius**2

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
