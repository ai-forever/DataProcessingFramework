import io
from typing import Any

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


class GunnarFarnebackFilter(VideoFilter):
    """
    Gunnar-Farneback filter inference class to get mean optical flow each video.
    The video's current and next frame are used for optical flow calculation between them.
    After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

    Parameters
    ----------
    pass_frames: int
        Number of frames to pass. pass_frames = 1, if need to process all frames.
    pyramid_scale: float
        Parameter, specifying the image scale (<1) to build pyramids for each image
    levels: int
        Number of pyramid layers including the initial image
    win_size: int
        Averaging window size
    iterations: int
        Number of iterations the algorithm does at each pyramid level
    size_poly_exp: int
        Size of the pixel neighborhood used to find polynomial expansion in each pixel
    poly_sigma: float
        Std of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    flags: int
        Operation flags that can be a combination of OPTFLOW_USE_INITIAL_FLOW and/or OPTFLOW_FARNEBACK_GAUSSIAN
    """

    def __init__(
        self,
        pass_frames: int = 10,
        pyramid_scale: float = 0.5,
        levels: int = 3,
        win_size: int = 15,
        iterations: int = 3,
        size_poly_exp: int = 5,
        poly_sigma: float = 1.2,
        workers: int = 16,
        flags: int = 0,
        batch_size: int = 1,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        self.num_workers = workers
        self.batch_size = batch_size

        self.pyramid_scale = pyramid_scale
        self.levels = levels
        self.win_size = win_size
        self.iterations = iterations
        self.size_poly_exp = size_poly_exp
        self.poly_sigma = poly_sigma
        self.flags = flags

        self.pass_frames = pass_frames

    @property
    def schema(self) -> list[str]:
        return [
            self.key_column,
            "mean_optical_flow_farneback"
        ]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
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

        if frames.shape[1] > frames.shape[2]:
            frames_resized = [
                transform_frame(frame=frames[i], target_size=(450, 800))
                for i in range(self.pass_frames, len(frames), self.pass_frames)
            ]
        elif frames.shape[2] > frames.shape[1]:
            frames_resized = [
                transform_frame(frame=frames[i], target_size=(800, 450))
                for i in range(self.pass_frames, len(frames), self.pass_frames)
            ]
        else:
            frames_resized = [
                transform_frame(frame=frames[i], target_size=(450, 450))
                for i in range(self.pass_frames, len(frames), self.pass_frames)
            ]
        return key, frames_resized

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        mean_magnitudes = []
        for data in batch:
            key, frames = data
            for i in range(self.pass_frames, len(frames), self.pass_frames):
                current_frame = frames[i - self.pass_frames]
                next_frame = frames[i]
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

            df_batch_labels[self.key_column].append(key)
            df_batch_labels['mean_optical_flow_farneback'].append(round(mean_optical_flow, 3))
        return df_batch_labels
