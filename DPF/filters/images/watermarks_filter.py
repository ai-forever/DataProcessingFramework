import os
from typing import Any

import torch
from torch import nn

from DPF.types import ModalityToDataMapping

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from huggingface_hub import hf_hub_download, hf_hub_url
from torchvision import models, transforms

from DPF.filters.utils import FP16Module
from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter

MODELS = {
    "resnext101_32x8d-large": {
        "resnet": models.resnext101_32x8d,
        "repo_id": "boomb0om/dataset-filters",
        "filename": "watermark_classifier-resnext101_32x8d-input_size320-4epochs_c097_w082.pth",
    },
    "resnext50_32x4d-small": {
        "resnet": models.resnext50_32x4d,
        "repo_id": "boomb0om/dataset-filters",
        "filename": "watermark_classifier-resnext50_32x4d-input_size320-4epochs_c082_w078.pth",
    },
}


def get_watermarks_detection_model(
    name: str,
    device: str = "cuda:0",
    fp16: bool = True,
    cache_dir: str = "/tmp/datasets_utils",
) -> Any:
    assert name in MODELS
    config = MODELS[name]

    model_ft = config["resnet"](pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Download weights directly using hf_hub_download
    weights_path = hf_hub_download(
        repo_id=config["repo_id"],
        filename=config["filename"],
        cache_dir=cache_dir
    )
    
    weights = torch.load(weights_path, device)
    model_ft.load_state_dict(weights)

    if fp16:
        model_ft = FP16Module(model_ft)  # type: ignore

    model_ft.eval()
    model_ft = model_ft.to(device)

    return model_ft


class WatermarksFilter(ImageFilter):
    """
    Filter for detecting watermarks.

    Parameters
    ----------
    watermarks_model: str
        Version of model to use. Available versions: "resnext50_32x4d-small", "resnext101_32x8d-large"
    weights_folder: str
        Path to folder with weights
    device: str = 'cuda:0'
        Torch device to use
    pbar: bool = True
        Flag for displaying progress bar
    workers: int = 16
        Number of processes for use in dataloader
    batch_size: int = 64
        Batch size for model
    """

    def __init__(
        self,
        watermarks_model: str,
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

        self.watermarks_model = watermarks_model
        self.weights_folder = weights_folder
        self.model = get_watermarks_detection_model(
            watermarks_model, device=device, fp16=True, cache_dir=weights_folder
        )
        self.resnet_transforms = transforms.Compose(
            [
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @property
    def result_columns(self) -> list[str]:
        return ["watermark_filter_pass"]

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
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.resnet_transforms(pil_img)
        return key, img_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)  # type: ignore

        with torch.no_grad():
            outputs = self.model(batch)
            # Get predictions (0 or 1) and convert to boolean (True if no watermark, False if watermark)
            predictions = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()
            df_batch_labels["watermark_filter_pass"].extend(
                [pred == 0 for pred in predictions]
            )
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
