"""
Classes for using in-memory data from a miniscope
"""

from typing import List, Literal, Optional, Union, overload

import pandas as pd
from numpydantic import NDArray
from pydantic import BaseModel, field_validator

from mio.models.sdcard import SDBufferHeader


class Frame(BaseModel):
    """
    An individual frame from a miniscope recording

    Typically returned from :meth:`.SDCard.read`
    """

    frame: NDArray
    headers: List[SDBufferHeader]

    @field_validator("headers")
    @classmethod
    def frame_nums_must_be_equal(cls, v: List[SDBufferHeader]) -> Optional[List[SDBufferHeader]]:
        """
        Each frame_number field in each header must be the same
        (they come from the same frame!)
        """

        if v is not None and not all([header.frame_num != v[0].frame_num for header in v]):
            raise ValueError(f"All frame numbers should be equal! Got f{[h.frame_num for h in v]}")
        return v

    @property
    def frame_num(self) -> Optional[int]:
        """
        Frame number for this set of headers, if headers are present
        """
        return self.headers[0].frame_num


class Frames(BaseModel):
    """
    A collection of frames from a miniscope recording
    """

    frames: List[Frame]

    @overload
    def flatten_headers(self, as_dict: Literal[False]) -> List[SDBufferHeader]: ...

    @overload
    def flatten_headers(self, as_dict: Literal[True]) -> List[dict]: ...

    def flatten_headers(self, as_dict: bool = False) -> Union[List[dict], List[SDBufferHeader]]:
        """
        Return flat list of headers, not grouped by frame

        Args:
            as_dict (bool): If `True`, return a list of dictionaries, if `False`
                (default), return a list of :class:`.SDBufferHeader` s.
        """
        h: Union[List[dict], List[SDBufferHeader]] = []
        for frame in self.frames:
            headers: Union[List[dict], List[SDBufferHeader]]
            if as_dict:
                headers = [header.model_dump() for header in frame.headers]
            else:
                headers = frame.headers
            h.extend(headers)
        return h

    def to_df(self, what: Literal["headers"] = "headers") -> pd.DataFrame:
        """
        Convert frames to pandas dataframe

        Arguments:
            what ('headers'): What information from the frame to include in the df,
                currently only 'headers' is possible
        """

        if what == "headers":
            return pd.DataFrame(self.flatten_headers(as_dict=True))
        else:
            raise ValueError("Return mode not implemented!")
