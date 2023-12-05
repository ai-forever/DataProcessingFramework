from typing import List, Dict, Union, Optional
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import clip

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .t2i_filter import T2IFilter


class CLIPFilter(T2IFilter):
    """
    Filter for calculating similarity score of images and captions with RuCLIP.

    Parameters
    ----------
    clip_version: str
        Version of model to use. Check available version here: https://github.com/openai/CLIP"
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

    Attributes
    ----------
    schema: List[str]
        List of columns to be added with this filter.
    dataloader_kwargs: dict:
        Parameters for dataloader (batch_size, num_workers, collate_fn, etc.)
    """

    def __init__(
        self,
        clip_version: str,
        weights_folder: str,
        templates: Optional[List[str]] = None,
        device: str = "cuda:0",
        workers: int = 16,
        batch_size: int = 64,
        pbar: bool = True,
    ):
        super().__init__(pbar)

        if templates is None:
            templates = ["{}"]
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates

        self.clip_version = clip_version
        self.weights_folder = weights_folder

        self.clip_model, self.clip_processor = clip.load(
            clip_version, device=self.device, download_root=self.weights_folder
        )

        self.schema = [self.key_column, f"clip_{self.clip_version}_similarity"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        text = modality2data['text']
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.clip_processor(pil_img)
        return key, img_tensor, text

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors, batch_labels = list(zip(*batch))

        with torch.no_grad():
            image_tensors = [t.to(self.device) for t in image_tensors]
            inputs = {}
            inputs["pixel_values"] = pad_sequence(image_tensors, batch_first=True)
            text_latents = []
            for template in self.templates:
                texts = clip.tokenize(
                    [template.format(class_label) for class_label in batch_labels],
                    truncate=True,
                )
                text_latents.append(self.clip_model.encode_text(texts.to(self.device)))
            text_latents = torch.stack(text_latents).mean(0)
            text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
            batch_similarity = self.get_similarity(inputs, text_latents).tolist()

        df_batch_labels[f"clip_{self.clip_version}_similarity"].extend(batch_similarity)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels

    def get_similarity(self, inputs, text_latents):
        with torch.no_grad():
            image_latents = self.clip_model.encode_image(inputs["pixel_values"])
            image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
            logits_per_image = torch.matmul(image_latents, text_latents.t())
            probs = logits_per_image.cpu().numpy().tolist()

        return np.diag(probs)
