from typing import Any, Optional, Union

import numpy as np
import torch
from lavis.models import load_model_and_preprocess  # type: ignore
from torch.nn.utils.rnn import pad_sequence

from DPF.utils import read_image_rgb_from_bytes

from ...types import ModalityToDataMapping
from .t2i_filter import T2IFilter


class BlipFilter(T2IFilter):
    """
    Filter for calculating similarity score of images and captions with RuCLIP.

    Parameters
    ----------
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

        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_image_text_matching", model_type="pretrain_vitL", is_eval=True, device=self.device
        )

    @property
    def schema(self) -> list[str]:
        return [self.key_column, "blip2_ViT-L_similarity"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        text = modality2data['text']
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.vis_processors["eval"](pil_img)
        text_tensor = self.txt_processors["eval"](text)
        return key, img_tensor, text_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        image_tensors: list[torch.Tensor]
        text_tensors: list[torch.Tensor]
        keys, image_tensors, text_tensors = list(zip(*batch))  # type: ignore

        sample: dict[str, Union[torch.Tensor, list[torch.Tensor]]] = {}
        with torch.no_grad():
            image_tensors = [t.to(self.device) for t in image_tensors]
            sample['image'] = pad_sequence(image_tensors, batch_first=True)
            sample['text_input'] = text_tensors
            features_image = self.blip_model.extract_features(sample, mode="image")
            features_text = self.blip_model.extract_features(sample, mode="text")

            batch_similarity = self.get_similarity(features_image, features_text)  # type: ignore

        df_batch_labels["blip2_ViT-L_similarity"].extend(batch_similarity)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels

    def get_similarity(self, features_image, features_text):  # type: ignore
        with torch.no_grad():
            logits_per_image = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
            probs = logits_per_image.cpu().numpy().tolist()

        return np.diag(probs)
