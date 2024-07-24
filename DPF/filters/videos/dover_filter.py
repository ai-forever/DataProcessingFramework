import os
from typing import Any
from urllib.request import urlretrieve
from io import BytesIO
import numpy as np
import torch

from DPF.types import ModalityToDataMapping
from .video_filter import VideoFilter
import yaml

from dover.datasets import (
    UnifiedFrameSampler,
    get_single_view,
)
import decord
from decord import VideoReader
from dover.models import DOVER

WEIGHTS_URL = {'dover': 'https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth',
               'dover_plus_plus': 'https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth',
               'dover-mobile': 'https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth'}

CONFIGS_URL = {'dover': 'https://raw.githubusercontent.com/teowu/DOVER-Dev/master/dover.yml',
               'dover_plus_plus': 'https://raw.githubusercontent.com/teowu/DOVER-Dev/master/dover.yml',
               'dover-mobile': 'https://raw.githubusercontent.com/teowu/DOVER-Dev/master/dover-mobile.yml'}


def fuse_results(results: list):
    t, a = (results[0] + 0.0758) / 0.0129, (results[1] - 0.1253) / 0.0318
    # t, a = (results[0] - 0.1107) / 0.07355, (results[1] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


def spatial_temporal_view_decomposition(
    video_path, sample_types, samplers, is_train=False, augment=False,
):
    video = {}
    decord.bridge.set_bridge("torch")
    vreader = VideoReader(video_path)
    ### Avoid duplicated video decoding!!! Important!!!!
    all_frame_inds = []
    frame_inds = {}
    for stype in samplers:
        frame_inds[stype] = samplers[stype](len(vreader), is_train)
        all_frame_inds.append(frame_inds[stype])

    ### Each frame is only decoded one time!!!
    all_frame_inds = np.concatenate(all_frame_inds, 0)
    frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

    for stype in samplers:
        imgs = [frame_dict[idx] for idx in frame_inds[stype]]
        video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)
    return sampled_video, frame_inds

class DOVERFilter(VideoFilter):
    """
    DOVER model inference class to get video quality scores.
    More info about the model here: https://github.com/teowu/DOVER/

    Parameters
    ----------
    weights_folder: str
        Path to the folder where the weights are located.
        If there are no weights, they will be downloaded automatically
    model_name: str = "dover"
        "dover_plus_plus", "dover"  or "dover-mobile" version of the model
    device: str = "cuda:0"
        Device to use
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        weights_folder: str,
        model_name: str = 'dover_plus_plus',
        device: str = "cuda:0",
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.device = device

        self.model_name = model_name
        self.weights_folder = weights_folder
        
        # Download checkpoints and configs
        path_to_model = os.path.join(self.weights_folder, self.model_name + '.pth')
        if not os.path.exists(path_to_model):
            os.makedirs(self.weights_folder, exist_ok=True)
            urlretrieve(WEIGHTS_URL[self.model_name], path_to_model)
        path_to_config = os.path.join(self.weights_folder, self.model_name + '.yml')
        if not os.path.exists(path_to_config):
            os.makedirs(self.weights_folder, exist_ok=True)
            urlretrieve(CONFIGS_URL[self.model_name], path_to_config)
        
        # Load model
        with open(path_to_config, "r") as f:
            opt = yaml.safe_load(f)
        self.model = DOVER(**opt["model"]["args"]).to(self.device)
        state_dict = torch.load(path_to_model, map_location=self.device)
        if self.model_name == 'dover_plus_plus':
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict)

        self.dopt = opt["data"]["val-l1080p"]["args"]

    @property
    def result_columns(self) -> list[str]:
        return [f"dover_aesthetic", f"dover_technical", f"dover_overall"]

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
        video_file = BytesIO(modality2data['video'])

        mean, std = (
            torch.FloatTensor([123.675, 116.28, 103.53]),
            torch.FloatTensor([58.395, 57.12, 57.375])
        )
        
        temporal_samplers = {}
        for stype, sopt in self.dopt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        ### View Decomposition
        views, _ = spatial_temporal_view_decomposition(
            video_file, self.dopt["sample_types"], temporal_samplers
        )

        for k, v in views.items():
            num_clips = self.dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
            )

        return key, views

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        key, views = batch[0]
        for k, v in views.items():
            views[k] = v.to(self.device)
                
        with torch.no_grad():
            results = [r.mean().item() for r in self.model(views)]
            rescaled_results = fuse_results(results)
        df_batch_labels[self.key_column].append(key)
        df_batch_labels[self.schema[1]].append(rescaled_results['aesthetic'])
        df_batch_labels[self.schema[2]].append(rescaled_results['technical'])
        df_batch_labels[self.schema[3]].append(rescaled_results['overall'])
        return df_batch_labels
