import io
from typing import Any, Optional
import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from cv2.typing import MatLike
from torch import Tensor

from DPF.types import ModalityToDataMapping
from .video_filter import VideoFilter

import ptlflow

WEIGHTS_URL = 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'


def transform_frame(frame: MatLike, target_size: tuple[int, int]) -> Tensor:
    frame = cv2.resize(frame, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()[None]

    padder = InputPadder(frame_tensor.shape)  # type: ignore
    frame_tensor = padder.pad(frame_tensor)[0]
    return frame_tensor


def transform_keep_ar(frame: MatLike, min_side_size: int) -> Tensor:
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    if h <= w:
        new_height = min_side_size
        new_width = int(aspect_ratio * new_height)
    else:
        new_width = min_side_size
        new_height = int(new_width / aspect_ratio)

    frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()[None]

    padder = InputPadder(frame_tensor.shape)  # type: ignore
    frame_tensor = padder.pad(frame_tensor)[0]
    return frame_tensor


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims: list[int], mode: str = 'sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                         pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                         0, pad_ht]

    def pad(self, *inputs) -> list[Tensor]:  # type: ignore
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x: Tensor) -> Tensor:
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class RPKnetOpticalFlowFilter(VideoFilter):
    """
    RPKnet model inference class to get mean optical flow each video.
        The video's current and next frame are used for optical flow calculation between them.
        After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://github.com/hmorimitsu/ptlflow

    Parameters
    ----------
    pass_frames: int = 12
        Number of frames to pass. pass_frames = 1, if need to process all frames.
    num_passes: Optional[int] = None
        Number of flow scores calculations in one video. Set None to calculate flow scores on all video
    min_frame_size: int = 512
        The size of the minimum side of the video frame after resizing
    norm: bool = True
        Normalize flow or not
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
        num_passes: Optional[int] = None,
        min_frame_size: int = 512,
        norm: bool = True,
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
        self.num_passes = num_passes
        self.min_frame_size = min_frame_size
        self.frames_batch_size = frames_batch_size
        self.norm = norm

        self.model = ptlflow.get_model('rpknet', pretrained_ckpt='things')
        self.model.to(self.device)
        self.model.eval()

    @property
    def result_columns(self) -> list[str]:
        return [f"optical_flow_rpk_mean", f"optical_flow_rpk_std"]

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
        return key, frames_transformed

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        magnitudes: list[float] = []
        for data in batch:
            key, frames = data
            with torch.no_grad():
                for i in range(0, len(frames)-1, self.frames_batch_size):
                    end = min(i+self.frames_batch_size, len(frames)-1)
                    current_frame = torch.cat(frames[i:end], dim=0)
                    next_frame = torch.cat(frames[i+1:i+self.frames_batch_size+1], dim=0)

                    current_frame_cuda = current_frame.to(self.device)
                    next_frame_cuda = next_frame.to(self.device)

                    inputs = torch.stack([current_frame_cuda, next_frame_cuda], dim=1)
                    
                    flow = self.model({'images': inputs})['flows'][:, 0]
                    if self.norm:
                        h, w = current_frame.shape[-2:]
                        flow[:, 0] = flow[:, 0] / w
                        flow[:, 1] = flow[:, 1] / h
                    magnitude = ((flow[:,0]**2+flow[:,1]**2)**0.5).detach().cpu().numpy()
                    magnitudes.extend(magnitude)
                mean_value = np.mean(magnitudes)
                std_value = np.std(magnitudes)

                df_batch_labels[self.key_column].append(key)
                df_batch_labels[self.schema[1]].append(round(mean_value, 6))
                df_batch_labels[self.schema[2]].append(round(std_value, 6))
        return df_batch_labels
