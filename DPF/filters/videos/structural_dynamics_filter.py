import io
from typing import Any, Optional
import cv2
import imageio.v3 as iio
import numpy as np
import torch
from cv2.typing import MatLike
from torch import Tensor

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter

from pytorch_msssim import MS_SSIM
from time import time


class StructuralDynamicsFilter(VideoFilter):
    """

    Structural dynamics score from https://arxiv.org/pdf/2407.01094
        The video's current and next frame are used for MS-SSIM calculation between them.
        After, the mean value of scores for the entire video is calculated on the array of scores between two frames.

    Parameters
    ----------
    pass_frames: int = 12
        Number of frames to pass. pass_frames = 1, if need to process all frames.
    min_frame_size: int = 512
        The size of the minimum side of the video frame after resizing
    frames_batch_size: int = 16
        Batch size during one video processing
    device: str = "cuda:0"
        Device to use
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        pass_frames: int = 10,
        min_frame_size: int = 512,
        frames_batch_size: int = 16,
        device: str = "cuda:0",
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.device = device

        assert pass_frames >= 1, "Number of pass_frames should be greater or equal to 1."
        self.pass_frames = pass_frames
        self.min_frame_size = min_frame_size
        self.frames_batch_size = frames_batch_size
        self.model = MS_SSIM(data_range=255, size_average=False, channel=3, win_size=11)

    @property
    def result_columns(self) -> list[str]:
        return [f"structural_dynamics", 'structural_dynamics_max', 'structural_dynamics_min']

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
        frames_transformed = []
        frames_transformed = [
            torch.from_numpy(frames[i]).permute(2, 0, 1).float()[None]
            for i in range(self.pass_frames, len(frames), self.pass_frames)
        ]
        return key, frames_transformed

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        values: list[float] = []
        for data in batch:
            key, frames = data
            with torch.no_grad():
                for i in range(0, len(frames)-1, self.frames_batch_size):
                    end = min(i+self.frames_batch_size, len(frames)-1)
                    current_frame = torch.cat(frames[i:end], dim=0)
                    next_frame = torch.cat(frames[i+1:i+self.frames_batch_size+1], dim=0)

                    current_frame_cuda = current_frame.to(self.device)
                    next_frame_cuda = next_frame.to(self.device)

                    t0 = time()
                    ssim = self.model(
                        current_frame_cuda,
                        next_frame_cuda
                    )
                    values.extend(ssim.detach().cpu().numpy())
                    # print('SSIM time=', time() - t0)
                mean_value = np.mean(values)
                mn = np.min(values)
                mx = np.max(values)

                df_batch_labels[self.key_column].append(key)
                df_batch_labels[self.schema[1]].append(round(mean_value, 6))
                df_batch_labels[self.schema[2]].append(round(mx, 6))
                df_batch_labels[self.schema[3]].append(round(mn, 6))
        return df_batch_labels
