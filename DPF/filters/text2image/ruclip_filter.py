from typing import Any, Optional

import numpy as np

# TODO(review) - зависимость отсутствует в requirements.txt
import ruclip  # type: ignore
import torch
from torch.nn.utils.rnn import pad_sequence

from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes

from .t2i_filter import T2IFilter


class RuCLIPFilter(T2IFilter):
    """
    Filter for calculating similarity score of images and captions with RuCLIP.

    Parameters
    ----------
    clip_model: str
        Version of model to use.
        Check available version here: https://github.com/ai-forever/ru-clip"
    weights_folder: str
        Path to folder with weights
    templates: List[str] = ['{}']
        List of strings to be used as templates for texts.
        Text embedding will be calculated as a mean value of that templates embeddings
    device: str = 'cuda:0'
        Torch device to use
    workers: int = 16
        Number of processes for use in dataloader
    batch_size: int = 64
        Batch size for model
    pbar: bool = True
        Flag for displaying progress bar
    """

    def __init__(
        self,
        ruclip_version: str,
        weights_folder: str,
        templates: Optional[list[str]] = None,
        device: str = "cuda:0",
        workers: int = 16,
        batch_size: int = 64,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        if templates is None:
            templates = ["{}"]
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates

        self.ruclip_version = ruclip_version
        self.weights_folder = weights_folder

        self.ruclip_model, self.ruclip_processor = ruclip.load(
            ruclip_version, device=device, cache_dir=weights_folder
        )
        self.ruclip_predictor = ruclip.Predictor(
            self.ruclip_model,
            self.ruclip_processor,
            device,
            bs=self.batch_size,
            templates=self.templates,
        )

    @property
    def result_columns(self) -> list[str]:
        return [f"{self.ruclip_version}_similarity"]

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
        text = modality2data['text']
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.ruclip_processor.image_transform(pil_img)
        return key, img_tensor, text

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        image_tensors: list[torch.Tensor]
        keys, image_tensors, batch_labels = list(zip(*batch))  # type: ignore

        with torch.no_grad():
            image_tensors = [t.to(self.device) for t in image_tensors]
            inputs = {}
            inputs["pixel_values"] = pad_sequence(image_tensors, batch_first=True)
            text_latents = self.ruclip_predictor.get_text_latents(batch_labels)
            batch_similarity = self.get_similarity(inputs, text_latents).tolist()

        df_batch_labels[f"{self.ruclip_version}_similarity"].extend(batch_similarity)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels

    def get_similarity(self, inputs: dict[str, torch.Tensor], text_latents: torch.Tensor) -> np.ndarray[Any, Any]:
        with torch.no_grad():
            logit_scale = self.ruclip_model.logit_scale.exp()
            image_latents = self.ruclip_model.encode_image(inputs["pixel_values"])
            image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
            logits_per_text = (
                torch.matmul(
                    text_latents.to(self.ruclip_predictor.device), image_latents.t()
                )
                * logit_scale
            )
            logits_per_text = logits_per_text.cpu().numpy()

        return np.diag(logits_per_text)
