import io
from typing import Any, Optional

import cv2
import imageio.v3 as iio
import numpy as np
from cv2.typing import MatLike

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter


def transform_frame(frame: MatLike, target_size: tuple[int, int]) -> MatLike:
    frame = cv2.resize(frame, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def transform_keep_ar(frame: MatLike, min_side_size: int) -> MatLike:
    h, w = frame.shape[:2]  # type: ignore
    aspect_ratio = w / h
    if h <= w:
        new_height = min_side_size
        new_width = int(aspect_ratio * new_height)
    else:
        new_width = min_side_size
        new_height = int(new_width / aspect_ratio)

    resized_frame: MatLike = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return resized_frame


class GunnarFarnebackFilter(VideoFilter):
    """
    Gunnar-Farneback filter inference class to get mean optical flow each video.
    The video's current and next frame are used for optical flow calculation between them.
    After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

    Parameters
    ----------
    pass_frames: int = 12
        Number of frames to pass. pass_frames = 1, if need to process all frames.
    num_passes: Optional[int] = None
        Number of flow scores calculations in one video. Set None to calculate flow scores on all video
    min_frame_size: int = 512
        The size of the minimum side of the video frame after resizing
    pyramid_scale: float = 0.5
        Parameter, specifying the image scale (<1) to build pyramids for each image
    levels: int = 3
        Number of pyramid layers including the initial image
    win_size: int = 15
        Averaging window size
    iterations: int = 3
        Number of iterations the algorithm does at each pyramid level
    size_poly_exp: int = 5
        Size of the pixel neighborhood used to find polynomial expansion in each pixel
    poly_sigma: float = 1.2
        Std of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    flags: int = 0
        Operation flags that can be a combination of OPTFLOW_USE_INITIAL_FLOW and/or OPTFLOW_FARNEBACK_GAUSSIAN
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        pass_frames: int = 12,
        num_passes: Optional[int] = None,
        min_frame_size: int = 512,
        pyramid_scale: float = 0.5,
        levels: int = 3,
        win_size: int = 15,
        iterations: int = 3,
        size_poly_exp: int = 5,
        poly_sigma: float = 1.2,
        workers: int = 16,
        flags: int = 0,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        self.num_workers = workers

        self.num_passes = num_passes
        self.min_frame_size = min_frame_size
        self.pass_frames = pass_frames

        self.pyramid_scale = pyramid_scale
        self.levels = levels
        self.win_size = win_size
        self.iterations = iterations
        self.size_poly_exp = size_poly_exp
        self.poly_sigma = poly_sigma
        self.flags = flags

    @property
    def result_columns(self) -> list[str]:
        return ["optical_flow_farneback"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        video_file = modality2data['video']

        frames = iio.imread(io.BytesIO(video_file), plugin="pyav")
        max_frame_to_process = self.num_passes*self.pass_frames if self.num_passes else len(frames)
        frames_transformed = []

        frames_transformed = [
            transform_keep_ar(frames[i], self.min_frame_size)
            for i in range(self.pass_frames, min(max_frame_to_process+1, len(frames)), self.pass_frames)
        ]

        mean_magnitudes: list[float] = []
        for i in range(len(frames_transformed)-1):
            current_frame = frames_transformed[i]
            next_frame = frames_transformed[i+1]
            flow = cv2.calcOpticalFlowFarneback(
                current_frame,
                next_frame,
                None,
                self.pyramid_scale,
                self.levels,
                self.win_size,
                self.iterations,
                self.size_poly_exp,
                self.poly_sigma,
                self.flags
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitudes.append(magnitude)
        mean_optical_flow = np.mean(mean_magnitudes)
        return key, mean_optical_flow

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for data in batch:
            key, mean_optical_flow = data
            df_batch_labels[self.key_column].append(key)
            df_batch_labels[self.result_columns[0]].append(mean_optical_flow)
        return df_batch_labels
