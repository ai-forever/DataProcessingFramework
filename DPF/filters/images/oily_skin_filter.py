import os
from typing import Any
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter

class OilySkinFilter(ImageFilter):
    """
    Filter for skin type detection using a ViT model.

    Parameters
    ----------
    model_name: str
        Name or path of the pre-trained model to use
    device: str = "cuda:0"
        Device to use
    workers: int = 16
        Number of processes to use for reading data and calculating scores
    batch_size: int = 64
        Batch size for model
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        model_name: str = "dima806/skin_types_image_detection",
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

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)

        self.id2label = self.model.config.id2label

    @property
    def result_columns(self) -> list[str]:
        return ["skin_type", "confidence_score"]

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
        pil_image = read_image_rgb_from_bytes(modality2data['image'])

        # Apply preprocessing
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()

        return key, pixel_values

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, pixel_values = list(zip(*batch))
        pixel_values = torch.stack(pixel_values).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        for key, pred, prob in zip(keys, predictions, probabilities):
            skin_type = self.id2label[pred.item()]
            confidence = prob[pred].item()
            
            df_batch_labels[self.key_column].append(key)
            df_batch_labels["skin_type"].append(skin_type)
            df_batch_labels["confidence_score"].append(confidence)

        return df_batch_labels