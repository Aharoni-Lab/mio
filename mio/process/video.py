"""
This module contains functions for pre-processing video data.
"""

from pathlib import Path
from typing import Tuple
from scipy.signal import butter, filtfilt

import cv2
import numpy as np

from mio import init_logger
from mio.io import VideoReader
from mio.models.frames import NamedFrame
from mio.models.process import DenoiseConfig, FreqencyMaskingConfig, NoisePatchConfig
from mio.plots.video import VideoPlotter

logger = init_logger("video")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class FrameProcessor:
    """
    A class to process video frames.
    """

    def __init__(self, height: int, width: int):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Parameters:
        height (int): Height of the video frame.
        width (int): Width of the video frame.

        Returns:
        FrameProcessor: A FrameProcessor object.
        """
        self.height = height
        self.width = width

    def split_by_length(self, array: np.ndarray, segment_length: int) -> list[np.ndarray]:
        """
        Split an array into sub-arrays of a specified length.
        Last sub-array may be shorter if the array length is not a multiple of the segment length.

        Parameters:
        array (np.ndarray): The array to split.
        segment_length (int): The length of each sub-array.

        Returns:
        list[np.ndarray]: A list of sub-arrays.
        """
        num_segments = len(array) // segment_length

        # Split the array into segments of the specified length
        sub_arrays = [
            array[i * segment_length : (i + 1) * segment_length] for i in range(num_segments)
        ]

        # Add the remaining elements as a final shorter segment, if any
        if len(array) % segment_length != 0:
            sub_arrays.append(array[num_segments * segment_length :])

        return sub_arrays

    def patch_noisy_buffer(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        noise_patch_config: NoisePatchConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare current frame with the previous frame to find noisy frames.
        Replace noisy blocks with those from the previous frame.
        The comparison is done in blocks of a specified size,
        defined by the buffer_size divided by buffer_split.

        Parameters:
        current_frame (np.ndarray): The current frame to process.
        previous_frame (np.ndarray): The previous frame to compare against.
        noise_threshold (float): The threshold for mean error to consider a block noisy.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The processed frame and the noise patch.
        """
        serialized_current = current_frame.flatten().astype(np.int16)
        serialized_previous = previous_frame.flatten().astype(np.int16)

        buffer_per_frame = len(serialized_current) // noise_patch_config.buffer_size + 1

        split_current = self.split_by_length(
            serialized_current,
            noise_patch_config.buffer_size // noise_patch_config.buffer_split + 1,
        )
        split_previous = self.split_by_length(
            serialized_previous,
            noise_patch_config.buffer_size // noise_patch_config.buffer_split + 1,
        )

        split_output = split_current.copy()
        noisy_parts = split_current.copy()

        buffer_has_noise = False
        for buffer_index in range(buffer_per_frame):
            for split_index in range(noise_patch_config.buffer_split):
                i = buffer_index * noise_patch_config.buffer_split + split_index
                mean_error = abs(split_current[i] - split_previous[i]).mean()
                logger.debug(f"Mean error for buffer {i}: {mean_error}")
                if mean_error > noise_patch_config.threshold:
                    logger.info(f"Replacing buffer {i} with mean error {mean_error}")
                    buffer_has_noise = True
                    break
                else:
                    split_output[i] = split_current[i]
                    noisy_parts[i] = np.zeros_like(split_current[i], np.uint8)
            if buffer_has_noise:
                for split_index in range(noise_patch_config.buffer_split):
                    i = buffer_index * noise_patch_config.buffer_split + split_index
                    split_output[i] = split_previous[i]
                    noisy_parts[i] = np.ones_like(split_current[i], np.uint8)
                buffer_has_noise = False

        serialized_output = np.concatenate(split_output)[: self.height * self.width]
        noise_output = np.concatenate(noisy_parts)[: self.height * self.width]

        # Deserialize processed frame
        processed_frame = serialized_output.reshape(self.width, self.height)
        noise_patch = noise_output.reshape(self.width, self.height)

        return np.uint8(processed_frame), np.uint8(noise_patch)

    def apply_freq_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform FFT/IFFT to remove horizontal stripes from a single frame.

        Parameters:
        img (np.ndarray): The image to process.
        mask (np.ndarray): The frequency mask to apply.

        Returns:
        np.ndarray: The filtered image
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Use log for better visualization

        # Normalize the magnitude spectrum for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
        )

        # Apply mask and inverse FFT
        fshift *= mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back), np.uint8(magnitude_spectrum)

    def gen_freq_mask(
        self,
        freq_mask_config: FreqencyMaskingConfig,
    ) -> np.ndarray:
        """
        Generate a mask to filter out horizontal and vertical frequencies.
        A central circular region can be removed to allow low frequencies to pass.
        """
        crow, ccol = self.height // 2, self.width // 2

        # Create an initial mask filled with ones (pass all frequencies)
        mask = np.ones((self.height, self.width), np.uint8)

        # Zero out a vertical stripe at the frequency center
        mask[
            :,
            ccol
            - freq_mask_config.vertical_BEF_cutoff : ccol
            + freq_mask_config.vertical_BEF_cutoff,
        ] = 0

        # Zero out a horizontal stripe at the frequency center
        mask[
            crow
            - freq_mask_config.horizontal_BEF_cutoff : crow
            + freq_mask_config.horizontal_BEF_cutoff,
            :,
        ] = 0

        # Define spacial low pass filter
        y, x = np.ogrid[: self.height, : self.width]
        center_mask = (x - ccol) ** 2 + (
            y - crow
        ) ** 2 <= freq_mask_config.spatial_LPF_cutoff_radius**2

        # Restore the center circular area to allow low frequencies to pass
        mask[center_mask] = 1

        # Visualize the mask if needed. Might delete later.
        if freq_mask_config.display_mask:
            cv2.imshow("Mask", mask * np.iinfo(np.uint8).max)
            while True:
                if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit visualization
                    break
            cv2.destroyAllWindows()
        return mask
    
    def mean_intensity(self, frame: np.ndarray) -> float:
        """"
        Calculate the mean pixel intensity for each frame
        
        Paremeters:
        frame (np.ndarray): The frame whose mean intensity is to be calculated on filtered data(broken frames replaced, spatial mask filter).

        Returns:
        float: The mean intensity of frames. 
        """
        return frame.mean()

    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
        """
        Apply a Butterworth lowpass filter to the data.

        Parameters:
        data (np.ndarray): The data to filter.
        cutoff (float): The cutoff frequency of the filter.
        fs (float): The sampling rate of the data.
        order (int): The order of the filter.

        Returns:
        np.ndarray: The filtered data.
        """
        nyquist = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y


class VideoProcessor:
    """
    A class to process video files.
    """

    @staticmethod
    def denoise(
        video_path: str,
        config: DenoiseConfig,
    ) -> None:
        """
        Preprocess a video file and display the results.
        """
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is not a required dependency of miniscope-io, to use it, "
                "install it manually or install miniscope-io with `pip install miniscope-io[plot]`"
            )

        reader = VideoReader(video_path)

        pathstem = Path(video_path).stem
        output_dir = Path.cwd() / config.output_dir
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

         
        # Initiate the frame processor
        processor = FrameProcessor(
            height=reader.height,
            width=reader.width,
        )
        # Initialize lists to store frames
        raw_frames = []
        output_frames = []
        mean_intensities = []
        butter_filtered_frames = []

        if config.noise_patch.enable:
            patched_frames = []
            noise_patchs = []
            diff_frames = []
        if config.frequency_masking.enable:
            freq_domain_frames = []
            freq_filtered_frames = []

       

        if config.noise_patch.enable:
            freq_mask = processor.gen_freq_mask(
                freq_mask_config=config.frequency_masking,
            )

        # index for frame number in original video
        try:
            for index, frame in reader.read_frames():
                if config.end_frame and config.end_frame != -1 and index > config.end_frame:
                    break

                raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                raw_frames.append(raw_frame)

                output_frame = raw_frame.copy()

                if config.noise_patch.enable:
                    if index == 1:
                        previous_frame = raw_frame.copy()
                    logger.debug(f"Processing frame {index}")

                    patched_frame, noise_patch = processor.patch_noisy_buffer(
                        raw_frame,
                        previous_frame,
                        config.noise_patch,
                    )
                    previous_frame = patched_frame
                    patched_frames.append(patched_frame)
                    noise_patchs.append(noise_patch * np.iinfo(np.uint8).max)

                    if config.noise_patch.output_diff:
                        diff_frame = cv2.absdiff(raw_frame, previous_frame)
                        diff_frames.append(diff_frame * config.noise_patch.diff_multiply)

                    output_frame = patched_frame

                if config.frequency_masking.enable:
                    freq_filtered_frame, frame_freq_domain = processor.apply_freq_mask(
                        img=patched_frame,
                        mask=freq_mask,
                    )
                    freq_domain_frames.append(frame_freq_domain)
                    freq_filtered_frames.append(freq_filtered_frame)
                    output_frame = freq_filtered_frame
                output_frames.append(output_frame)

                #mean pixel intensity over each frames 
                meanF = output_frame.mean()
                mean_intensities.append(meanF)

                raw_frames.append(raw_frame)
                output_frames.append(output_frame)


        finally:
            #save mean intensities and generate plot
            mean_intensities_array = np.array(mean_intensities)
            np.save(output_dir / f"{pathstem}_mean_intensities.npy", mean_intensities_array)

            #trim mean intensity graph to start and end frame in config file
            start_frame = config.interactive_display.start_frame or 0
            end_frame = config.interactive_display.end_frame or len(mean_intensities_array)
            mean_intensities_trimmed = mean_intensities_array[start_frame:end_frame]

            
                
            def butter_lowpass_filter(data, cutoff, fs, order):
                """
                apply butterworth filter
                """
                nyquist = 0.5 * fs  # Nyquist frequency
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                y = filtfilt(b, a, data)
                return y

            butterOrder = None
            cutoffFreq = None
            samplingRate = None

            if hasattr(config, 'butter_filter') and getattr(config.butter_filter, 'enable', False):
                try:
                    butterOrder = config.butter_filter.order
                    cutoffFreq = config.butter_filter.cutoff_frequency
                    samplingRate = config.butter_filter.sampling_rate

                    # Ensure the values are valid before applying the filter
                    if butterOrder > 0 and cutoffFreq > 0 and samplingRate > 0:
                    # Apply filter
                        mean_intensities_filtered = butter_lowpass_filter(
                            mean_intensities_trimmed, cutoffFreq, samplingRate, butterOrder
                        )
                    else:
                        logger.warning(
                            "Invalid Butterworth filter parameters (order, cutoff, or sampling rate are <= 0). Skipping filtering."
                        )
                    mean_intensities_filtered = mean_intensities_trimmed
                except AttributeError as e:
                    raise AttributeError(
                        "Missing required parameter in 'butter_filter' section of the config"
                        ) from e
            else:
                logger.info("Butterworth filter is disabled; skipping filtering.")
                butterOrder = None  # Explicitly set butterOrder to None to avoid later errors
                mean_intensities_filtered = mean_intensities_trimmed
                
            if butterOrder is not None:
                if len(mean_intensities_array) > 3 * butterOrder:  # Ensure enough data for filtering
                    #Proceed with filtering logic
                    pass
                else:
                    logger.warning(
                    "Not enough data for Butterworth filtering. Skipping the filter."
                    )
                    mean_intensities_filtered = mean_intensities_trimmed

            mean_intensities_array = np.array(mean_intensities)
            if butterOrder is not None and len(mean_intensities_array) > 3 * butterOrder:  # Ensure enough data for filtering
                mean_intensities_filtered = processor.butter_lowpass_filter(
                mean_intensities_array, cutoffFreq, samplingRate, butterOrder
                )
            else:
                logger.warning("Insufficient data for Butterworth filtering. Skipping.")
                mean_intensities_filtered = mean_intensities_array  # Use raw means if not enough data
            
            mean_intensities_filtered_trim = mean_intensities_filtered[start_frame:end_frame]

            if plt is not None:
                plt.figure(figsize=(10, 6))  # Larger figure for readability
    
                # Plot the unfiltered mean intensities
                plt.plot(
                    range(start_frame, end_frame),
                    mean_intensities_trimmed,
                    label="Mean Intensity",
                    color='blue'
                    )

                # Plot filtered mean if the butter filter enabled
                if butterOrder is not None:
                    plt.plot(
                        range(start_frame, end_frame),
                        mean_intensities_filtered_trim,
                        label=f"Filtered (Order={butterOrder})",
                        color='orange'
                        )
                plt.xlabel("Frame Index", fontsize=12)
                plt.ylabel("Mean Intensity", fontsize=12)
                plt.title(f"Mean Intensity per Frame ({start_frame} to {end_frame})", fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True)

                #save plot
                mean_intensity_plot_path = output_dir / f"{pathstem}_mean_intensity_plot_start_end_frame.png"
                plt.savefig(mean_intensity_plot_path, dpi=300)
                plt.close()
                logger.info(f"Mean intensity plot saved to {mean_intensity_plot_path}")

                #show plot as separate window
                mean_intensity_image = cv2.imread(str(mean_intensity_plot_path))
                cv2.imshow("Mean Intensity Plot", mean_intensity_image)

                while True:
                    if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to continue
                        break

             # Apply scaling to each frame
             #applies an intensity scaling to each frame, based on the Butterworth-filtered mean intensity
            for index, frame in enumerate(freq_filtered_frames):
                if index < len(mean_intensities_filtered):
                    meanF = mean_intensities[index]
                    scaling_factor = 1 + (mean_intensities_filtered[index] - meanF) / meanF
                    scaled_frame = np.clip(frame * scaling_factor, 0, 255).astype(np.uint8)
                    butter_filtered_frames.append(scaled_frame)

            reader.release()
            minimum_projection = VideoProcessor.get_minimum_projection(output_frames)

            subtract_minimum = [(frame - minimum_projection) for frame in output_frames]

            subtract_minimum = VideoProcessor.normalize_video_stack(subtract_minimum)

            raw_video = NamedFrame(name="RAW", video_frame=raw_frames)

            if config.noise_patch.enable:
                patched_video = NamedFrame(name="patched", video_frame=patched_frames)
                if config.noise_patch.output_result:
                    patched_video.export(
                        output_dir / f"{pathstem}",
                        suffix=True,
                        fps=20,
                    )
            if config.noise_patch.output_diff:
                diff_video = NamedFrame(
                    name=f"diff_{config.noise_patch.diff_multiply}x", video_frame=diff_frames
                )
                diff_video.export(
                    output_dir / f"{pathstem}",
                    suffix=True,
                    fps=20,
                )
            if config.noise_patch.output_noise_patch:
                noise_patch = NamedFrame(name="noise_patch", video_frame=noise_patchs)
            if config.frequency_masking.output_mask:
                freq_mask_frame = NamedFrame(
                    name="freq_mask", static_frame=freq_mask * np.iinfo(np.uint8).max
                )
                freq_mask_frame.export(
                    output_dir / f"{pathstem}",
                    suffix=True,
                    fps=20,
                )

            if config.frequency_masking.enable:
                freq_domain_video = NamedFrame(name="freq_domain", video_frame=freq_domain_frames)
                freq_filtered_video = NamedFrame(
                    name="freq_filtered", video_frame=freq_filtered_frames
                )
                if config.frequency_masking.output_freq_domain:
                    freq_domain_video.export(
                        output_dir / f"{pathstem}",
                        suffix=True,
                        fps=20,
                    )
                if config.frequency_masking.output_result:
                    freq_filtered_video.export(
                        (output_dir / f"{pathstem}"),
                        suffix=True,
                        fps=20,
                    )

            min_proj_frame = NamedFrame(name="min_proj", static_frame=minimum_projection)

            if config.interactive_display.enable:
                videos = [
                    raw_video,
                    noise_patch,
                    patched_video,
                    freq_filtered_video,
                    freq_domain_video,
                    min_proj_frame,
                    freq_mask_frame,
                ]

            # Add scaled images if they exist
            if hasattr(config, 'butter_filter') and getattr(config.butter_filter, 'enable', False):
                if butter_filtered_frames:  # Add only if frames exist
                    butter_filtered_video = NamedFrame(name="scaled_images", video_frame=butter_filtered_frames)
                    videos.append(butter_filtered_video)
                
            if config.interactive_display.enable:
                VideoPlotter.show_video_with_controls(
                videos,
                start_frame=config.interactive_display.start_frame,
                end_frame=config.interactive_display.end_frame,
                )
        # Ensure both lists have the same length
        assert len(freq_filtered_frames) == len(butter_filtered_frames), "Frame lists are not the same length!"

        differences = []
        for i, (freq_frame, scaled_frame) in enumerate(zip(freq_filtered_frames, butter_filtered_frames)):
            diff = np.abs(freq_frame.astype(np.float32) - scaled_frame.astype(np.float32))
            differences.append(diff.mean())  # Calculate mean difference for each frame

        # Check summary statistics
        logger.info(f"Average difference per frame freq_filtered vs scaled_images: {np.mean(differences)}")
        logger.info(f"Maximum difference across all frames freq_filtered vs scaled_images: {np.max(differences)}")


    @staticmethod
    def get_minimum_projection(image_list: list[np.ndarray]) -> np.ndarray:
        """
        Get the minimum projection of a list of images.

        Parameters:
        image_list (list[np.ndarray]): A list of images to project.

        Returns:
        np.ndarray: The minimum projection of the images.
        """
        stacked_images = np.stack(image_list, axis=0)
        min_projection = np.min(stacked_images, axis=0)
        return min_projection

    @staticmethod
    def normalize_video_stack(image_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        Normalize a stack of images to 0-255 using max and minimum values of the entire stack.
        Return a list of images.

        Parameters:
        image_list (list[np.ndarray]): A list of images to normalize.

        Returns:
        list[np.ndarray]: The normalized images as a list.
        """

        # Stack images along a new axis (axis=0)
        stacked_images = np.stack(image_list, axis=0)

        # Find the global min and max across the entire stack
        global_min = stacked_images.min()
        global_max = stacked_images.max()

        # Normalize each frame using the global min and max
        normalized_images = []
        for i in range(stacked_images.shape[0]):
            normalized_image = cv2.normalize(
                stacked_images[i],
                None,
                0,
                np.iinfo(np.uint8).max,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            # Apply global normalization
            normalized_image = (
                (stacked_images[i] - global_min)
                / (global_max - global_min)
                * np.iinfo(np.uint8).max
            )
            normalized_images.append(normalized_image.astype(np.uint8))

        return normalized_images
