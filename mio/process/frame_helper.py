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

    def patch_noisy_buffer(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        noise_patch_config: NoisePatchConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare current frame with the previous frame to find noisy frames.
        Replace noisy blocks with those from the previous frame.
        The comparison is done in blocks of a specified size,
        defined by the buffer_size divided by buffer_split.

        Parameters:
        current_frame (np.ndarray): The current frame to process.
        previous_frame (np.ndarray): The previous frame to compare against.
        noise_threshold (float): The threshold for mean error to consider a block noisy.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The processed frame and the noise patch.
        """
        serialized_current = current_frame.flatten().astype(np.int16)
        serialized_previous = previous_frame.flatten().astype(np.int16)

        buffer_per_frame = len(serialized_current) // noise_patch_config.buffer_size + 1

        split_current = self.split_by_length(
            serialized_current,
            noise_patch_config.buffer_size // noise_patch_config.buffer_split + 1,
        )
        split_previous = self.split_by_length(
            serialized_previous,
            noise_patch_config.buffer_size // noise_patch_config.buffer_split + 1,
        )

        split_output = split_current.copy()
        noisy_parts = split_current.copy()

        buffer_has_noise = False
        for buffer_index in range(buffer_per_frame):
            for split_index in range(noise_patch_config.buffer_split):
                i = buffer_index * noise_patch_config.buffer_split + split_index
                mean_error = abs(split_current[i] - split_previous[i]).mean()
                logger.debug(f"Mean error for buffer {i}: {mean_error}")
                if mean_error > noise_patch_config.threshold:
                    logger.info(f"Replacing buffer {i} with mean error {mean_error}")
                    buffer_has_noise = True
                    break
                else:
                    split_output[i] = split_current[i]
                    noisy_parts[i] = np.zeros_like(split_current[i], np.uint8)
            if buffer_has_noise:
                for split_index in range(noise_patch_config.buffer_split):
                    i = buffer_index * noise_patch_config.buffer_split + split_index
                    split_output[i] = split_previous[i]
                    noisy_parts[i] = np.ones_like(split_current[i], np.uint8)
                buffer_has_noise = False

        serialized_output = np.concatenate(split_output)[: self.height * self.width]
        noise_output = np.concatenate(noisy_parts)[: self.height * self.width]

        # Deserialize processed frame
        processed_frame = serialized_output.reshape(self.width, self.height)
        noise_patch = noise_output.reshape(self.width, self.height)

        return np.uint8(processed_frame), np.uint8(noise_patch)


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
            # Apply global normalization
            normalized_image = (
                (stacked_images[i] - global_min)
                / (global_max - global_min)
                * np.iinfo(np.uint8).max
            )
            normalized_images.append(normalized_image.astype(np.uint8))

        return normalized_images
