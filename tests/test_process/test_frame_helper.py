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
    Contrast method of noise detection should correctly label frames corrupted
    by speckled noise
    """
    global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_example")
    config: NoisePatchConfig = global_config.noise_patch
    config.method = noise_detection_method

    with open(DATA_DIR / "wireless_corrupted.yaml") as yfile:
        expected = yaml.safe_load(yfile)

    video = cv2.VideoCapture(str(DATA_DIR / "wireless_corrupted.avi"))

    detector = NoiseDetectionHelper(
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    frames = []
    masks = []
    previous_frame = None
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if noise_detection_method == "gradient":
            is_noisy, mask = detector.detect_frame_with_noisy_buffer(
                current_frame = frame,
                previous_frame = None,
                config = config)
        if noise_detection_method == "mean_error":
            if previous_frame is None:
                previous_frame = frame
            is_noisy, mask = detector.detect_frame_with_noisy_buffer(
                current_frame = frame,
                previous_frame = previous_frame,
                config = config)
            if not is_noisy:
                previous_frame = frame
        frames.append(is_noisy)
        masks.append(mask)

    assert np.array_equal(np.where(frames)[0], expected["frames"])