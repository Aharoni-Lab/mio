import unittest
import numpy as np
from pydantic import ValidationError

from mio.models.frames import NamedFrame

class TestNamedFrame(unittest.TestCase):

    def setUp(self):
        self.name = "test_frame"
        self.static_frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.video_frame = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]
        self.video_list_frame = [[np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(5)] for _ in range(3)]

    def test_static_frame_initialization(self):
        frame = NamedFrame(name=self.name, static_frame=self.static_frame)
        self.assertEqual(frame.name, self.name)
        self.assertEqual(frame.data.shape, self.static_frame.shape)
        self.assertEqual(frame.data.dtype, self.static_frame.dtype)

    def test_video_frame_initialization(self):
        frame = NamedFrame(name=self.name, video_frame=self.video_frame)
        self.assertEqual(frame.name, self.name)
        self.assertEqual(len(frame.data), len(self.video_frame))
        self.assertEqual(frame.data[0].shape, self.video_frame[0].shape)

    def test_video_list_frame_initialization(self):
        frame = NamedFrame(name=self.name, video_list_frame=self.video_list_frame)
        self.assertEqual(frame.name, self.name)
        self.assertEqual(len(frame.data), len(self.video_list_frame))
        self.assertEqual(len(frame.data[0]), len(self.video_list_frame[0]))

    def test_invalid_initialization_raises_error(self):
        with self.assertRaises(ValidationError):
            # No frame type provided
            NamedFrame(name=self.name)

        with self.assertRaises(ValueError):
            # Providing more than one frame type should raise a ValueError
            NamedFrame(name=self.name, static_frame=self.static_frame, video_frame=self.video_frame)

    def test_export_static_frame(self):
        # Mocking the actual write operation can be done using unittest.mock, but here I will just focus on invocation
        frame = NamedFrame(name=self.name, static_frame=self.static_frame)

        # Patch the cv2.imwrite to test the export without actually writing to disk
        from unittest.mock import patch

        with patch('cv2.imwrite') as mocked_imwrite:
            frame.export('output.png', 20, True)
            mocked_imwrite.assert_called_once()

    def test_export_video_frame(self):
        frame = NamedFrame(name=self.name, video_frame=self.video_frame)

        from unittest.mock import patch, MagicMock

        with patch('cv2.VideoWriter') as MockVideoWriter:
            writer_instance = MockVideoWriter.return_value
            writer_instance.write = MagicMock()
            writer_instance.release = MagicMock()

            frame.export('output.avi', 20, True)

            writer_instance.write.assert_called()
            writer_instance.release.assert_called_once()

    def test_export_video_list_frame_not_implemented(self):
        frame = NamedFrame(name=self.name, video_list_frame=self.video_list_frame)
        
        with self.assertRaises(NotImplementedError):
            frame.export('output.avi', 20, True)


if __name__ == '__main__':
    unittest.main()