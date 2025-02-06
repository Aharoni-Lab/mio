"""
Helper module for Z-stack operations.
"""

import cv2
import numpy as np


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
