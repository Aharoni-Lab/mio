"""
Plotting functions for video streams and frames.
"""

from typing import List, Union

from mio import init_logger
from mio.models.frames import NamedFrame, NamedVideo

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.widgets import Button, Slider
except ImportError:
    plt = None
    animation = None
    Button = None
    Slider = None
    KeyEvent = None

logger = init_logger("videoplot")


class VideoPlotter:
    """
    Class to display video streams and static images.
    """

    @staticmethod
    def show_video_with_controls(
        videos: List[Union[NamedFrame, NamedVideo]],
        start_frame: int,
        end_frame: int,
        fps: int = 20,
    ) -> None:
        """
        Plot multiple video streams or static images side-by-side.
        Can play/pause and navigate frames.

        Parameters
        ----------
        videos : NamedFrame
            NamedFrame object containing video data and names.
        start_frame : int
            Starting frame index for the video display.
        end_frame : int
            Ending frame index for the video display.
        fps : int, optional
            Frames per second for the video, by default 20
        """
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is not a required dependency of miniscope-io, to use it, "
                "install it manually or install miniscope-io with `pip install miniscope-io[plot]`"
            )

        # Wrap static images in lists to handle them uniformly
        video_frames = [None] * len(videos)
        for i in range(len(videos)):
            if isinstance(videos[i], NamedFrame):
                video_frames[i] = [videos[i].frame]
            elif isinstance(videos[i], NamedVideo):
                video_frames[i] = videos[i].video

        titles = [video.name for video in videos]

        num_streams = len(video_frames)

        logger.info(f"Displaying {num_streams} video streams.")
        if end_frame > start_frame:
            logger.info(f"Displaying frames {start_frame} to {end_frame}.")
            for stream_index in range(len(video_frames)):
                logger.info(f"Stream length: {len(video_frames[stream_index])}")
                if len(video_frames[stream_index]) > 1:
                    video_frames[stream_index] = video_frames[stream_index][start_frame:end_frame]
                    logger.info(f"Trimmed stream length: {len(video_frames[stream_index])}")

        num_frames = max(len(stream) for stream in video_frames)
        logger.info(f"Max stream length: {num_frames}")

        fig, axes = plt.subplots(1, num_streams, figsize=(20, 5))

        frame_displays = []
        for idx, ax in enumerate(axes):
            initial_frame = video_frames[idx][0]
            frame_display = ax.imshow(initial_frame, cmap="gray", vmin=0, vmax=255)
            frame_displays.append(frame_display)
            if titles:
                ax.set_title(titles[idx])
            ax.axis("off")

        # Slider
        ax_slider = plt.axes([0.1, 0.1, 0.65, 0.05], facecolor="lightgoldenrodyellow")
        slider = Slider(
            ax=ax_slider, label="Frame", valmin=0, valmax=num_frames - 1, valinit=0, valstep=1
        )

        playing = [False]  # Use a mutable object to track play state
        ax_button = plt.axes([0.8, 0.1, 0.1, 0.05])
        button = Button(ax_button, "Play/Pause")

        # Callback to toggle play/pause
        def toggle_play(event: KeyEvent) -> None:
            playing[0] = not playing[0]

        button.on_clicked(toggle_play)

        # Update function for the slider and frame displays
        def update_frame(index: int) -> None:
            for idx, frame_display in enumerate(frame_displays):
                # Repeat last frame for static images or when the index is larger than stream length
                if index < len(video_frames[idx]):
                    frame = video_frames[idx][index]
                else:
                    frame = video_frames[idx][-1]  # Keep showing last frame for shorter streams
                frame_display.set_data(frame)
            fig.canvas.draw_idle()

        # Slider update callback
        def on_slider_change(val: float) -> None:
            index = int(slider.val)
            update_frame(index)

        # Connect the slider update function
        slider.on_changed(on_slider_change)

        # Animation function
        def animate(i: int) -> None:
            if playing[0]:
                current_frame = int(slider.val)
                next_frame = (current_frame + 1) % num_frames
                slider.set_val(next_frame)  # This will also trigger on_slider_change

        # Use FuncAnimation to update the figure at the specified FPS
        # This needs to be stored in a variable to prevent animation getting deleted
        ani = animation.FuncAnimation(  # noqa: F841
            fig, animate, frames=num_frames, interval=1000 // fps, blit=False
        )

        plt.show()
