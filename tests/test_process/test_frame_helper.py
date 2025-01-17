import cv2
import yaml
import numpy as np

from mio.models.process import DenoiseConfig, NoisePatchConfig
from mio.process.frame_helper import NoiseDetectionHelper

from ..conftest import DATA_DIR


def test_noise_detection_contrast():
    """
    Contrast method of noise detection should correctly label frames corrupted
    by speckled noise
    """
    global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_example")
    config: NoisePatchConfig = global_config.noise_patch
    config.method = "block_contrast"

    with open(DATA_DIR / "wireless_corrupted.yaml") as yfile:
        expected = yaml.safe_load(yfile)

    video = cv2.VideoCapture(str(DATA_DIR / "wireless_corrupted.avi"))

    detector = NoiseDetectionHelper(
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    frames = []
    masks = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_noisy, mask = detector.detect_frame_with_noisy_buffer(frame, None, config)
        frames.append(is_noisy)
        masks.append(mask)

    assert np.array_equal(np.where(frames)[0], expected["frames"])
