import cv2
import yaml
import numpy as np
import pytest

from mio.models.process import DenoiseConfig, NoisePatchConfig
from mio.process.frame_helper import NoiseDetectionHelper

from ..conftest import DATA_DIR


@pytest.mark.parametrize(
    "noise_detection_method",
    [
        "gradient",  # Only testing the "gradient" method
    ],
)
def test_noise_detection_contrast_with_categories(noise_detection_method):
    """
    Contrast method of noise detection should correctly label frames corrupted
    by speckled noise and report how many frames were missed in each category.
    """
    # Load the global configuration for the gradient method
    global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_example")
    config: NoisePatchConfig = global_config.noise_patch
    config.method = noise_detection_method

    # Load expected data with categories from YAML
    with open(DATA_DIR / "wireless_corrupted_extended.yaml") as yfile:
        expected = yaml.safe_load(yfile)

    # Initialize the video and noise detection helper
    video = cv2.VideoCapture(str(DATA_DIR / "wireless_corrupted_extended.avi"))
    detector = NoiseDetectionHelper(
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    detected_frames = []  # Store detected noisy frames
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect noisy frames using the gradient method
        is_noisy, _ = detector.detect_frame_with_noisy_buffer(
            current_frame=frame, previous_frame=None, config=config
        )

        detected_frames.append(is_noisy)

    detected_frame_indices = np.where(detected_frames)[0]

    # Analyze missed frames by category
    missed_counts = {}
    for category, subcategories in expected["frames"].items():
        missed_counts[category] = {}
        for subcategory, expected_frames in subcategories.items():
            expected_set = set(expected_frames)
            detected_set = set(detected_frame_indices)
            missed_frames = expected_set - detected_set  # Frames in expected but not detected
            missed_counts[category][subcategory] = {
                "missed_count": len(missed_frames),
                "total_expected": len(expected_set),
                "missed_frames": sorted(missed_frames),
            }

    # Print or assert results for debugging
    for category, subcategories in missed_counts.items():
        for subcategory, stats in subcategories.items():
            print(
                f"{category} -> {subcategory}: Missed {stats['missed_count']} "
                f"out of {stats['total_expected']} frames."
            )
            print(f"Missed Frames: {stats['missed_frames']}")

    # Example assertion: Ensure no missed frames for critical categories
    assert all(
        stats["missed_count"] == 0
        for category in missed_counts.values()
        for stats in category.values()
    ), "Some frames were missed in one or more categories!"