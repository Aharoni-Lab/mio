import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from mio.process.video import NoisePatchProcessor, FreqencyMaskProcessor, PassThroughProcessor, MinProjSubtractProcessor, VideoProcessor
from mio.models.process import DenoiseConfig

class TestVideoProcessors(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("/tmp/test_output")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = 100, 100
        self.test_frame = np.ones((self.height, self.width), dtype=np.uint8) * 128
        self.video_frames = [self.test_frame for _ in range(10)]
        

    def test_noise_patch_processor(self):
        denoise_config = DenoiseConfig.from_id("denoise_example")
        denoise_config.noise_patch.enable = True
        denoise_config.noise_patch.output_result = True

        processor = NoisePatchProcessor("denoise_example", denoise_config.noise_patch, self.width, self.height, self.test_dir)
        processed_frame, _ = processor.process_frame(self.test_frame)

        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertEqual(processor.name, "denoise_example")
        self.assertTrue(processor.output_enable)

    def test_freqency_mask_processor(self):
        denoise_config = DenoiseConfig.from_id("denoise_example")
        denoise_config.frequency_masking.enable = True
        denoise_config.frequency_masking.output_result = True

        processor = FreqencyMaskProcessor("test_freq_mask", denoise_config.frequency_masking, self.width, self.height, self.test_dir)
        processed_frame, freq_domain = processor.process_frame(self.test_frame)

        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertIsInstance(freq_domain, np.ndarray)
        self.assertEqual(processor.name, "test_freq_mask")
        self.assertTrue(processor.output_enable)

    def test_pass_through_processor(self):
        processor = PassThroughProcessor("test_pass_through", self.test_dir)
        processed_frame = processor.process_frame(self.test_frame)
        
        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertEqual(processed_frame.all(), self.test_frame.all())
        self.assertEqual(processor.name, "test_pass_through")

    def test_min_proj_subtract_processor(self):
        min_proj_config = MagicMock()
        min_proj_config.output_result = True
        
        processor = MinProjSubtractProcessor("test_min_proj", min_proj_config, self.test_dir, self.video_frames)
        processor.normalize_stack()

        self.assertEqual(processor.name, "test_min_proj")
        self.assertTrue(processor.output_enable)
        self.assertIsInstance(processor.output_frames[0], np.ndarray)

if __name__ == '__main__':
    unittest.main()