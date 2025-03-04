import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

from mio.models.frames import NamedFrame, NamedVideo


class TestNamedFrame(unittest.TestCase):

    def setUp(self):
        # Create a single frame (image) and a list of frames (video)
        self.image_frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.video_frames = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)
        ]
        self.name = "test"

    @patch("cv2.imwrite")
    def test_export_image_frame(self, mock_imwrite):
        # Create instance of NamedFrame with a single image
        named_frame = NamedFrame(name=self.name, frame=self.image_frame)
        # Call the export method
        named_frame.export("output_path", True)
        # Check that cv2.imwrite was called correctly
        mock_imwrite.assert_called_once_with("output_path_test.png", self.image_frame)

    @patch("mio.models.frames.VideoWriter.init_video")
    def test_export_video_frame(self, mock_init_video):
        # Create a mock writer instance
        mock_writer = MagicMock()
        mock_init_video.return_value = mock_writer

        # Create instance of NamedFrame with a video
        named_frame = NamedVideo(name=self.name, video=self.video_frames)
        # Call the export method
        named_frame.export(output_path="output_path", fps=20, suffix=True)

        # Verify init_video was called with correct parameters
        mock_init_video.assert_called_once_with(
            path=Path("output_path_test.avi"),
            width=self.video_frames[0].shape[1],
            height=self.video_frames[0].shape[0],
            fps=20,
            fourcc="XVID",
        )

        # Check that the writer's write method was called for each frame
        self.assertEqual(mock_writer.write.call_count, len(self.video_frames))

    @patch("cv2.imwrite")
    @patch("mio.models.frames.VideoWriter.init_video")
    def test_invalid_frame_type_raises_exception(self, mock_init_video, mock_imwrite):
        # Test with an invalid type (e.g., integer)
        with self.assertRaises(ValueError):
            named_frame = NamedFrame(name=self.name, frame=12345)
            named_frame.export("output_path", 20, True)

        # Test with a list containing non-ndarray elements
        with self.assertRaises(ValueError):
            named_frame = NamedFrame(name=self.name, frame=[123, 456])
            named_frame.export("output_path", 20, True)

        # Ensure that no write methods are called
        mock_imwrite.assert_not_called()
        mock_init_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
