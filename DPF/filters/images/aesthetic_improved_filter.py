import os
from typing import Any, Dict, List
from urllib.request import urlretrieve

import clip
import numpy as np
import torch
import torch.nn as nn

from DPF.types import ModalityToDataMapping

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter


class MLP(nn.Module):
    def __init__(self, input_size: int, xcol: str = 'emb', ycol: str = 'avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.layers(x)


def normalized(a: np.ndarray[Any, Any], axis: int = -1, order: int = 2) -> Any:
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_improved_aesthetic_model(cache_folder: str) -> MLP:
    """
    Load the aesthetic model
    """
    path_to_model = cache_folder + "/sac+logos+ava1-l14-linearMSE.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/"
            + "sac+logos+ava1-l14-linearMSE.pth"
            + '?raw=true'
        )
        # TODO rework download
        urlretrieve(url_model, path_to_model)

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(path_to_model)

    model.load_state_dict(s)
    model.eval()

    return model


class ImprovedAestheticFilter(ImageFilter):
    """
    ImprovedAestheticFilter class
    """

    def __init__(
        self,
        weights_folder: str,
        device: str = "cuda:0",
        workers: int = 16,
        batch_size: int = 64,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.weights_folder = weights_folder

        self.aesthetic_model = get_improved_aesthetic_model(weights_folder)
        self.aesthetic_model.to(self.device)
        self.clip_model, self.clip_transforms = clip.load("ViT-L/14", device=device)

    @property
    def schema(self) -> List[str]:
        return [self.key_column, "improved_aesthetic_score_ViT-L/14"]

    @property
    def dataloader_kwargs(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: Dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        pil_image = read_image_rgb_from_bytes(modality2data['image'])

        image = self.clip_transforms(pil_image)

        return key, image

    def process_batch(self, batch: List[Any]) -> Dict[str, List[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)  # type: ignore
        with torch.no_grad():
            inputs = self.clip_model.encode_image(batch)
            inputs = normalized(inputs.cpu().detach().numpy())
            outputs = self.aesthetic_model(torch.from_numpy(inputs).to(self.device).type(torch.cuda.FloatTensor))  # type: ignore [attr-defined]

        df_batch_labels["improved_aesthetic_score_ViT-L/14"].extend(outputs.cpu().reshape(-1).tolist())
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
