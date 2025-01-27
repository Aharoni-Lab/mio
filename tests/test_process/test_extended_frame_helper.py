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
        "gradient",
        "mean_error",
    ],
)

def test_noise_detection_contrast(noise_detection_method):
    """
    Test broken buffer noise with detailed categorization of real wirelessly imaging noise.
    Two main categories should be correctly detected: check-pattern and blacked-out labeled in YAML file.
    Subcategories are dependent on the number of pixels (number of pixel rows) that are broken.
    """
    if noise_detection_method == "gradient":
        global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_example")
    elif noise_detection_method == "mean_error":
        global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_example_mean_error")
    else:
        raise ValueError("Invalid noise detection method")

    config: NoisePatchConfig = global_config.noise_patch
    config.method = noise_detection_method

    with open(DATA_DIR / "wireless_corrupted_extended.yaml") as yfile:
        expected = yaml.safe_load(yfile)

    video = cv2.VideoCapture(str(DATA_DIR / "wireless_corrupted_extended.avi"))

    detector = NoiseDetectionHelper(
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    missed_frames = {category: {sub: [] for sub in subs} for category, subs in expected["frames"].items()}
    false_positives = []

    previous_frame = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if noise_detection_method == "gradient":
            is_noisy, _ = detector.detect_frame_with_noisy_buffer(
                current_frame=frame, previous_frame=None, config=config
            )
        elif noise_detection_method == "mean_error":
            if previous_frame is None:
                previous_frame = frame
            is_noisy, _ = detector.detect_frame_with_noisy_buffer(
                current_frame=frame, previous_frame=previous_frame, config=config
            )
            if not is_noisy:
                previous_frame = frame

        # Track missed frames
        detected_in_category = False
        for category, subcategories in expected["frames"].items():
            for subcategory, frame_list in subcategories.items():
                if frame_number in frame_list:
                    detected_in_category = True
                    if not is_noisy:  # Mark as missed if not detected
                        missed_frames[category][subcategory].append(frame_number)

        # Track false positives
        if is_noisy and not detected_in_category:
            false_positives.append(frame_number)

    # Prepare missed message, grouped by method
    missed_message = f"=== Missed Frames ({noise_detection_method}) ==="
    for category, subcategories in missed_frames.items():
        for subcategory, frames in subcategories.items():
            if frames:  # Only include non-empty missed frames
                missed_message += f"\n  Category: {category} -> Subcategory: {subcategory}: {frames}"

    # Prepare false positive message, grouped by method
    false_positive_message = (
        f"\n=== False Positives ({noise_detection_method}) ===\n  Frames: {false_positives}"
        if false_positives
        else f"\n=== False Positives ({noise_detection_method}) ===\n  None"
    )

    print(f"Missed Message: {missed_message}")
    print(f"False Positive Message: {false_positive_message}")

    
    # Combine detailed message
    detailed_message = (
    f"=== Detailed Noise Detection Report ({noise_detection_method}) ===\n"
    f"{missed_message}\n{false_positive_message}"
)

    # Debug: Print the detailed message for verification
    print(detailed_message)

    # Raise assertion error with detailed message
    assert not any(
        missed_frames[cat][sub] for cat in missed_frames for sub in missed_frames[cat]
    ) and not false_positives, detailed_message