"""
Pydantic models for storing frames and videos.
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mio.io import VideoWriter
from mio.logging import init_logger

logger = init_logger("model.frames")


class NamedFrame(BaseModel):
    """
    Pydantic model to store an an image or a video together with a name.
    """

    name: str = Field(
        ...,
        description="Name of the video.",
    )
    frame: Optional[Union[np.ndarray, List[np.ndarray]]] = Field(
        None,
        description="Frame data, if provided.",
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def export(self, output_path: Union[Path, str], fps: int, suffix: bool) -> None:
        """
        Export the frame data to a file.

        Parameters
        ----------
        output_path : Path, str
            Path to the output file.
        fps : int
            Frames per second for the

        Raises
        ------
        NotImplementedError
            If the frame type is video_list_frame.
        """
        output_path = Path(output_path)
        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        if isinstance(self.frame, np.ndarray):
            # write PNG out
            cv2.imwrite(str(output_path.with_suffix(".png")), self.frame)
        elif isinstance(self.frame, list):
            if all(isinstance(frame, np.ndarray) for frame in self.frame):
                writer = VideoWriter.init_video(
                    path=output_path.with_suffix(".avi"),
                    width=self.frame[0].shape[1],
                    height=self.frame[0].shape[0],
                    fps=20,
                )
                logger.info(
                    f"Writing video to {output_path}.avi:"
                    f"{self.frame[0].shape[1]}x{self.frame[0].shape[0]}"
                )
                try:
                    for frame in self.frame:
                        picture = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        writer.write(picture)
                finally:
                    writer.release()
            else:
                raise ValueError("Not all frames are numpy arrays.")
        else:
            raise ValueError("Unknown frame type or no frame data provided.")
