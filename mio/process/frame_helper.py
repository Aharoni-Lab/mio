"""
This module contains a helper class for frame operations.
"""

from abc import abstractmethod
from typing import Tuple

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

    @abstractmethod
    def find_invalid_area(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Find the invalid area in a single frame.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is invalid
            and the processed frame.
        """
        pass


class CombinedNoiseDetector(SingleFrameHelper):
    """
    Helper class for combined invalid frame detection.
    """

    def __init__(self, noise_patch_config: NoisePatchConfig):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Returns:
            NoiseDetectionHelper: A NoiseDetectionHelper object
        """
        self.config = noise_patch_config
        if noise_patch_config.method is None:
            raise ValueError("No noise detection methods provided")
        self.methods = noise_patch_config.method

        if "mean_error" in self.methods:
            self.mse_detector = MSENoiseDetector(noise_patch_config)
        if "gradient" in self.methods:
            self.gradient_detector = GradientNoiseDetector(noise_patch_config)
        if "black_area" in self.methods:
            self.black_detector = BlackAreaDetector(noise_patch_config)

    def find_invalid_area(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        noisy_flag = False
        combined_noisy_area = np.zeros_like(frame, dtype=np.uint8)

        if "mean_error" in self.methods:
            noisy, noisy_area = self.mse_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        if "gradient" in self.methods:
            noisy, noisy_area = self.gradient_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        if "black_area" in self.methods:
            noisy, noisy_area = self.black_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        return noisy_flag, combined_noisy_area


class GradientNoiseDetector(SingleFrameHelper):
    """
    Helper class for gradient noise detection.
    """

    def __init__(self, noise_patch_config: NoisePatchConfig):
        """
        Initialize the GradientNoiseDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            GradientNoiseDetectionHelper: A GradientNoiseDetectionHelper object.
        """
        self.config = noise_patch_config

    def find_invalid_area(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        noisy, mask = self._detect_with_gradient(frame)
        return noisy, mask

    def _detect_with_gradient(
        self,
        current_frame: np.ndarray,
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
        noisy_mask[mean_second_diff > self.config.threshold, :] = 1
        logger.debug("Row-wise means of second derivative: %s", mean_second_diff)

        # Determine if the frame is noisy (if any rows are marked as noisy)
        frame_is_noisy = noisy_mask.any()

        return frame_is_noisy, noisy_mask


class BlackAreaDetector(SingleFrameHelper):
    """
    Helper class for black area detection.
    """

    def __init__(self, noise_patch_config: NoisePatchConfig):
        """
        Initialize the BlackAreaDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            BlackAreaDetectionHelper: A BlackAreaDetectionHelper object.
        """
        self.config = noise_patch_config

    def find_invalid_area(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        noisy, mask = self._detect_black_pixels(frame)
        return noisy, mask

    def _detect_black_pixels(
        self,
        current_frame: np.ndarray,
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
            self.config.black_pixel_consecutive_threshold
        )  # How many consecutive pixels must be black
        black_pixel_value_threshold = (
            self.config.black_pixel_value_threshold
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


class MSENoiseDetector(SingleFrameHelper):
    """
    Helper class for mean squared error noise detection.
    """

    def __init__(self, noise_patch_config: NoisePatchConfig):
        """
        Initialize the MeanErrorNoiseDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            MeanErrorNoiseDetectionHelper: A MeanErrorNoiseDetectionHelper object.
        """
        self.config = noise_patch_config
        self.previous_frame = None

    def register_previous_frame(self, previous_frame: np.ndarray) -> None:
        """
        Register the previous frame for mean error calculation.

        Parameters:
            previous_frame (np.ndarray): The previous frame to compare against.
        """
        self.previous_frame = previous_frame

    def find_invalid_area(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            return False, np.zeros_like(frame, dtype=np.uint8)
        noisy, mask = self._detect_with_mean_error(frame)
        return noisy, mask

    def _detect_with_mean_error(
        self,
        current_frame: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect noise using mean error between current and previous frames.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        if self.previous_frame is None:
            return False, np.zeros_like(current_frame, dtype=np.uint8)

        frame_width = current_frame.shape[1]
        frame_height = current_frame.shape[0]

        serialized_current = current_frame.flatten().astype(np.int16)
        serialized_previous = self.previous_frame.flatten().astype(np.int16)
        buffer_shape = FrameSplitter.get_buffer_shape(
            frame_width, frame_height, self.config.device_config.px_per_buffer
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

            comparison_start = buffer_end - self.config.buffer_split
            while comparison_start > buffer_start:
                mean_error = abs(
                    serialized_current[comparison_start:buffer_end]
                    - serialized_previous[comparison_start:buffer_end]
                ).mean()
                logger.debug(
                    f"Mean error for buffer {buffer_index}:"
                    f" pixels {comparison_start}-{buffer_end}: {mean_error}"
                    f" (threshold: {self.config.threshold})"
                )

                if mean_error > self.config.threshold:
                    noisy_parts[comparison_start:buffer_end] = np.zeros_like(
                        serialized_current[comparison_start:buffer_end], np.uint8
                    )
                    any_buffer_has_noise = True
                    break
                comparison_start -= self.config.buffer_split

        noise_patch = noisy_parts.reshape((frame_height, frame_width))
        return any_buffer_has_noise, noise_patch


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


class FrameSplitter:
    """
    Helper class for splitting frames into buffers.
    Currently only for getting the buffer shape from pixel count.
    """

    def get_buffer_shape(frame_width: int, frame_height: int, px_per_buffer: int) -> list[int]:
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
