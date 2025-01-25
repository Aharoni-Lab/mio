"""
Wire-Free Miniscope that records data to an SD Card
"""

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import cv2
import numpy as np
from tqdm import tqdm

from mio import init_logger
from mio.devices import DeviceConfig, Miniscope, RecordingCameraMixin
from mio.exceptions import EndOfRecordingException, ReadHeaderException
from mio.models.data import Frame
from mio.models.pipeline import PipelineConfig
from mio.models.sdcard import SDLayout, SDMetadata
from mio.sources import SDFileSource
from mio.types import Resolution


class WireFreePipeline(PipelineConfig):
    """Base skeleton pipeline for the wirefree miniscope"""

    required_nodes = {
        "sdcard": "sd-file-source",
    }


class WireFreeConfig(DeviceConfig):
    """Configuration for wire free miniscope"""

    pipeline: WireFreePipeline = "wirefree-pipeline"
    layout: SDLayout = "wirefree-sd-layout"


@dataclass(kw_only=True)
class WireFreeMiniscope(Miniscope, RecordingCameraMixin):
    """
    I/O for data on an SD Card recorded with a WireFree Miniscope

    an instance of :class:`.sdcard.SDLayout` (typically in :mod:`.formats` )
    configures how the data is laid out on the SD card. This class makes the i/o
    operations abstract over multiple layouts

    Args:
        drive (str, :class:`pathlib.Path`): Path to the SD card drive
        config (:class:`.WireFreeConfig`): Configuration,
            including data layout and pipeline configs

    """

    drive: Path
    """The path to the SD card drive"""
    config: WireFreeConfig = field(default_factory=WireFreeConfig)

    positions: dict[int, int] = field(default_factory=dict)
    """
    A mapping between frame number and byte position in the video that makes for 
    faster seeking :)

    As we read, we store the locations of each frame before reading it. Later, 
    we can assign to `frame` to seek back to those positions. Assigning to `frame` 
    works without caching position, but has to manually iterate through each frame.
    """

    def __post_init__(self) -> None:
        """post-init create private vars"""
        self.layout = SDLayout.from_any(self.config.layout)
        self.logger = init_logger("WireFreeMiniscope")

        # Private attributes used when the file reading context is entered
        self._metadata = None  # type: Optional[SDMetadata]
        self._frame = None  # type: Optional[int]
        self._frame_count = None  # type: Optional[int]
        self._array = None  # type: Optional[np.ndarray]
        """
        n_pix x 1 array used to store pixels while reading buffers
        """

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def metadata(self) -> SDMetadata:
        """
        Read metadata from SD Card
        """
        if self._metadata is None:
            with open(self.drive, "rb") as sd:
                sd.seek(self.layout.sectors.config_pos, 0)
                configSectorData = np.frombuffer(sd.read(self.layout.sectors.size), dtype=np.uint32)

            self._metadata = SDMetadata(
                **{
                    k: configSectorData[v]
                    for k, v in self.layout.metadata.model_dump().items()
                    if v is not None
                }
            )

        return self._metadata

    @classmethod
    def configure(cls, drive: Union[str, Path], config: WireFreeConfig) -> None:
        """
        Configure a WireFreeMiniscope SD card for recording
        """
        raise NotImplementedError()

    @property
    def position(self) -> Optional[int]:
        """
        When entered as context manager, the current position of the internal file
        descriptor
        """
        if self._f is None:
            return None

        return self._f.tell()

    @property
    def frame(self) -> Optional[int]:
        """
        When reading, the number of the frame that would be read if we were to call
        :meth:`.read`
        """
        if self._f is None:
            return None

        return self._frame

    @frame.setter
    def frame(self, frame: int) -> None:
        """
        Seek to a specific frame

        Arguments:
            frame (int): The frame to seek to!
        """
        if self._f is None:
            raise RuntimeError(
                "Havent entered context manager yet! Cant change position without that!"
            )

        if frame == self.frame:
            return

        if frame in self.positions:
            self._f.seek(self.positions[frame], 0)
            self._frame = frame
            return
        else:
            # TODO find the nearest position we do have
            pass

        if frame < self.frame:
            # hard to go back, esp if we haven't already been here
            # (we should have stashed the position)
            # just go to start of data and seek like normally (next case)
            self._f.seek(self.layout.sectors.data_pos, 0)
            self._frame = 0

        if frame > self.frame:
            for _ in range(frame - self.frame):
                self.skip()

    @property
    def buffers_per_frame(self) -> int:
        """
        Number of buffers per frame!

        References:
            https://github.com/Aharoni-Lab/Miniscope-v4-Wire-Free/blob/786663781a4bece89c89e00fc3ac9d95912faba4/Miniscope-v4-Wire-Free-MCU-Firmware/Miniscope-v4-Wire-Free/Miniscope-v4-Wire-Free/main.c#L680
        """
        n_pix = self.metadata.width * self.metadata.height
        return int(np.ceil((n_pix + self._source.header_size) / (self.metadata.buffer_size)))

    @property
    def frame_count(self) -> int:
        """
        Total number of frames in the recording
        """
        return int(
            np.ceil(
                (self.metadata.n_buffers_recorded + self.metadata.n_buffers_dropped)
                / self.buffers_per_frame
            )
        )

    @property
    def _source(self) -> SDFileSource:
        """The SDFileSource node in the pipeline"""
        return self.pipeline.nodes["sdcard"]

    # --------------------------------------------------
    # Context Manager methods
    # --------------------------------------------------

    def __enter__(self) -> "WireFreeMiniscope":
        if self._f is not None:
            raise RuntimeError("Cant enter context, and open the file twice!")

        # init private attrs
        # create an empty frame to hold our data!
        self._array = np.zeros((self.metadata.width * self.metadata.height, 1), dtype=np.uint8)
        self._pixel_count = 0
        self._last_buffer_n = 0
        self._frame = 0

        self._f = open(self.drive, "rb")  # noqa: SIM115 - this is a context handler
        # seek to the start of the data
        self._f.seek(self.layout.sectors.data_pos, 0)
        # store the 0th frame position
        self.positions[0] = self.layout.sectors.data_pos

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self._f.close()
        self._f = None
        self._frame = 0

    @overload
    def read(self, return_header: Literal[True] = True) -> Frame: ...

    @overload
    def read(self, return_header: Literal[False] = False) -> np.ndarray: ...

    def read(self, return_header: bool = False) -> Union[np.ndarray, Frame]:
        """
        Read a single frame

        Arguments:
            return_header (bool): If `True`, return headers from individual buffers
                (default `False`)

        Return:
            :class:`numpy.ndarray` ,
            or a tuple(ndarray, List[:class:`~.SDBufferHeader`]) if `return_header`
            is `True`
        """
        if self._f is None:
            raise RuntimeError(
                "File is not open! Try entering the reader context by using it like "
                "`with sdcard:`"
            )

        self._array[:] = 0
        pixel_count = 0
        last_buffer_n = 0
        headers = []
        while True:
            # stash position before reading header
            last_position = self._f.tell()
            try:
                header = self._read_data_header(self._f)
            except ValueError as e:
                if "read length must be non-negative" in str(e):
                    # end of file! Value error thrown because the dataHeader will be
                    # blank,  and thus have a value of 0 for the header size, and we
                    # can't read 0 from the card.
                    self._f.seek(last_position, 0)
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e
            except IndexError as e:
                if "index 0 is out of bounds for axis 0 with size 0" in str(e):
                    # end of file if we are reading from a disk image without any
                    # additional space on disk
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e
            except ReadHeaderException as e:
                # if we are on the last frame, normal! signal end of iteration
                if self._frame == self.frame_count - 1:
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e

            if header.frame_buffer_count == 0 and last_buffer_n > 0:
                # we are in the next frame!
                # rewind to the beginning of the header, and return
                # the last_position is the start of the header for this frame
                self._f.seek(last_position, 0)
                self._frame += 1
                self.positions[self._frame] = last_position
                frame = np.reshape(self._array, (self.metadata.width, self.metadata.height))
                if return_header:
                    return Frame.model_construct(frame=frame, headers=headers)
                else:
                    return frame

            # grab buffer data and stash
            headers.append(header)
            data = self._read_buffer(self._f, header)
            data = self._trim(data, header.data_length)
            self._array[pixel_count : pixel_count + header.data_length, 0] = data
            pixel_count += header.data_length
            last_buffer_n = header.frame_buffer_count

    # --------------------------------------------------
    # Write methods
    # --------------------------------------------------

    def to_video(
        self,
        path: Union[Path, str],
        fourcc: Literal["GREY", "mp4v", "XVID"] = "GREY",
        isColor: bool = False,
        force: bool = False,
        progress: bool = True,
    ) -> None:
        """
        Save contents of SD card to video with opencv

        Args:
            path (:class:`pathlib.Path`): Output video path, with video extension
                ``.avi`` or ``.mp4``
            fourcc (str): FourCC code used with opencv. Other codecs may be available
                depending on your opencv installation, but by default opencv supports
                one of:

                * ``GREY`` (default)
                * ``mp4v``
                * ``XVID``

            isColor (bool): Indicates whether output video is in color
                (default: `False`)
            force (bool): If `True`, overwrite output video if one already exists
                (default: `False`)
            progress (bool): If `True` (default) show progress bar.
        """
        path = Path(path)
        if path.exists() and not force:
            raise FileExistsError(
                f"{str(path)} already exists, not overwriting. " "Use force=True to overwrite."
            )

        if path.suffix == ".mp4" and fourcc.lower() == "grey":
            self.logger.warning("Cannot use .mp4 with GREY fourcc code. Using .avi instead")
            path = path.with_suffix(".avi")
        elif path.suffix == ".avi" and fourcc.lower() == "mp4v":
            self.logger.warning("Cannot use .avi with mp4v fourcc code, using .mp4 instead")
            path = path.with_suffix(".mp4")

        if progress:
            pbar = tqdm(total=self.frame_count)

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.metadata.fs,
            (self.metadata.width, self.metadata.height),
            isColor=isColor,
        )

        # wrap in try block so we always close video writer class
        try:
            # Open video context manager
            with self:
                # catch the stop iteration signal
                try:
                    while True:
                        # this is sort of an awkward stack, should probably make a
                        # generator version of `read`
                        frame = self.read(return_header=False)
                        writer.write(frame)

                        if progress:
                            pbar.update()

                except StopIteration:
                    # end of the video!
                    pass

        finally:
            writer.release()
            if progress:
                pbar.close()

    def to_img(
        self,
        path: Optional[Union[Path, str]],
        frame: Optional[int] = None,
        force: bool = False,
        chunk_size: int = 1e6,
        progress: bool = True,
    ) -> None:
        """
        Create a new disk image that is truncated to the actual size of the video data

        Typically, making disk images using dd or other tools will create an image
        file that is the full size of the media it's stored on. Rather than
        sending a bunch of 30GB image files around, we can instead create an
        image that is truncated to just the size of the data that has actually been recorded

        Args:
            path (:class:`pathlib.Path`): Path to write .img file to
            frame (int): Optional, if present only write the first n frames. If
                ``None`` , write all frames
            force (bool): If ``True``, overwrite an existing file. If ``False`` , (default)
                don't.
            chunk_size (int): Number of bytes to read/write at once (default ``1e6`` )
            progress (bool): If ``True`` (default), show progress bar
        """
        path = Path(path)
        if path.exists() and not force:
            raise FileExistsError("File exists, use force=True to overwrite")

        path.parent.mkdir(parents=True, exist_ok=True)

        start_frame = self.frame

        if frame is None:
            frame = self.frame_count - 1

        # read it to move the file position to the end of the frame
        with self:
            # go to one before the requested frame to make sure we have its position
            self.frame = frame - 1
            # read advances us to the end of the requested frame -- where we want to truncate
            with contextlib.suppress(EndOfRecordingException):
                _ = self.read(return_header=True)
            # now we can take the diff of the positions to get the expected offset
            # for the end of the last frame
            # these should be populated from the read and frame reposition
            diff = self.positions[frame] - self.positions[frame - 1]
            final_position = self._f.tell() + diff

        # try block to ensure closing pbar
        pos = 0
        pbar = tqdm(total=final_position) if progress else None

        try:
            with open(self.drive, "rb") as readfile, open(path, "wb") as writefile:
                while pos < final_position:
                    readsize = int(np.min([final_position - pos, chunk_size]))
                    data = readfile.read(readsize)
                    writefile.write(data)
                    pos = readfile.tell()
                    if progress:
                        pbar.update(readsize)

        finally:
            if progress:
                pbar.close()

        # return frame to the place it was before we wrote
        if start_frame is not None:
            with self:
                self.frame = start_frame

    # --------------------------------------------------
    # General Methods
    # --------------------------------------------------

    def skip(self) -> None:
        """
        Skip a frame

        Read the buffer headers to determine buffer sizes and just seek ahead
        """
        if self._f is None:
            raise RuntimeError(
                "File is not open! Try entering the reader context by using it like "
                "`with sdcard:`"
            )

        last_position = self._f.tell()
        header = self._read_data_header(self._f)

        if header.frame_buffer_count != 0:
            self._f.seek(last_position, 0)
            raise RuntimeError(
                "Did not start at the first buffer of a frame! Something is wrong with "
                "the way seeking is working. Rewound to where we started"
            )

        while True:
            # jump ahead according to the last header we read
            read_size = self._read_size(header)
            self._f.seek(read_size, 1)

            # stash position before reading next buffer header
            last_position = self._f.tell()
            header = self._read_data_header(self._f)

            # if the frame is over, return to the start of the buffer and break,
            # incrementing the current frame count
            if header.frame_buffer_count == 0:
                self._f.seek(last_position, 0)
                self._frame += 1
                self.positions[self.frame] = last_position
                break

    def check_valid(self) -> bool:
        """
        Checks that the header sector has the appropriate write keys in it

        Returns:
            bool - True if valid, False if not
        """
        with open(self.drive, "rb") as sd:
            sd.seek(self.layout.sectors.header_pos, 0)
            headerSectorData = np.frombuffer(sd.read(self.layout.sectors.size), dtype=np.uint32)

            valid = False
            if (
                headerSectorData[0] == self.layout.write_key0
                and headerSectorData[1] == self.layout.write_key1
                and headerSectorData[2] == self.layout.write_key2
                and headerSectorData[3] == self.layout.write_key3
            ):
                valid = True

        return valid

    # --------------------------------------------------
    # ABC methods
    # --------------------------------------------------

    def init(self) -> None:
        """Do nothing"""
        pass

    def deinit(self) -> None:
        """Do nothing"""
        pass

    def start(self) -> None:
        """start pipeline"""
        raise NotImplementedError()

    def stop(self) -> None:
        """stop pipeline"""
        raise NotImplementedError()

    def join(self) -> None:
        """join pipeline"""
        raise NotImplementedError()

    @property
    def excitation(self) -> float:
        """LED Excitation"""
        raise NotImplementedError()

    @property
    def fps(self) -> int:
        """FPS"""
        return self.metadata.fs

    @property
    def resolution(self) -> Resolution:
        """Resolution of recorded video"""
        return Resolution(self.metadata.width, self.metadata.height)

    def get(self, key: str) -> Any:
        """get a configuration value by its name"""
        return getattr(self.config, key)

    def set(self, key: str, value: Any) -> None:
        """set a configuration value"""
        raise NotImplementedError()
