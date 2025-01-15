"""
Pydantic models for storing frames and videos.
"""

from pathlib import Path
from typing import List, Optional, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from mio.io import VideoWriter
from mio.logging import init_logger

T = TypeVar("T", np.ndarray, List[np.ndarray], List[List[np.ndarray]])

logger = init_logger("model.frames")


class NamedFrame(BaseModel):
    """
    Pydantic model to store an array (frame/video/video list) together with a name.
    """

    name: str = Field(
        ...,
        description="Name of the video.",
    )
    static_frame: Optional[np.ndarray] = Field(
        None,
        description="Frame data, if provided.",
    )
    video_frame: Optional[List[np.ndarray]] = Field(
        None,
        description="Video data, if provided.",
    )
    video_list_frame: Optional[List[List[np.ndarray]]] = Field(
        None,
        description="List of video data, if provided.",
    )
    frame_type: Optional[str] = Field(
        None,
        description="Type of frame data.",
    )

    @model_validator(mode="before")
    def check_frame_type(cls, values: dict) -> dict:
        """
        Ensure that exactly one of static_frame, video_frame, or video_list_frame is provided.
        """
        static = values.get("static_frame")
        video = values.get("video_frame")
        video_list = values.get("video_list_frame")

        # Identify which fields are present
        present_fields = [
            (field_name, field_value)
            for field_name, field_value in zip(
                ("static_frame", "video_frame", "video_list_frame"), (static, video, video_list)
            )
            if field_value is not None
        ]

        if len(present_fields) != 1:
            raise ValueError(
                "Exactly one of static_frame, video_frame, or video_list_frame must be provided."
            )

        # Record which frame type is present
        values["frame_type"] = present_fields[0][0]

        return values

    @property
    def data(self) -> T:
        """Return the content of the populated field."""
        if self.frame_type == "static_frame":
            return self.static_frame
        elif self.frame_type == "video_frame":
            return self.video_frame
        elif self.frame_type == "video_list_frame":
            return self.video_list_frame
        else:
            raise ValueError("Unknown frame type or no frame data provided.")

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
        if self.frame_type == "static_frame":
            # write PNG out
            cv2.imwrite(str(output_path.with_suffix(".png")), self.static_frame)
        elif self.frame_type == "video_frame":
            writer = VideoWriter.init_video(
                path=output_path.with_suffix(".avi"),
                width=self.video_frame[0].shape[1],
                height=self.video_frame[0].shape[0],
                fps=20,
            )
            logger.info(
                f"Writing video to {output_path}.avi:"
                f"{self.video_frame[0].shape[1]}x{self.video_frame[0].shape[0]}"
            )
            try:
                for frame in self.video_frame:
                    picture = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    writer.write(picture)
            finally:
                writer.release()

        elif self.frame_type == "video_list_frame":
            raise NotImplementedError("Exporting video list frames is not yet supported.")
        else:
            raise ValueError("Unknown frame type or no frame data provided.")

    class Config:
        """
        Pydantic config for allowing np.ndarray types.
        Could be an Numpydantic situation so will look into it later.
        """

        arbitrary_types_allowed = True
