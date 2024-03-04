from typing import Any, Dict, List, Optional, Union

import clip
import torch

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter


class CLIPLabelsFilter(ImageFilter):
    """
    Filter for perfoming zero-shot with CLIP model.

    Parameters
    ----------
    clip_model: str
        Version of model to use. Check available version here: https://github.com/openai/CLIP"
    labels: List[str]
        List of classes for detecting
    weights_folder: str
        Path to folder with weights
    templates: List[str] = ['{}', 'photo of a {}']
        List of strings to be used as templates for texts. Text embedding will be
        calculated as a mean value of that templates embeddings
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
        clip_model: str,
        labels: List[str],
        weights_folder: str,
        device: str = "cuda:0",
        templates: Optional[List[str]] = None,
        workers: int = 16,
        batch_size: int = 64,
        pbar: bool = True,
    ):
        super().__init__(pbar)

        if templates is None:
            templates = ["{}", "photo of a {}"]
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.clip_version = clip_model
        self.templates = templates
        self.labels = labels
        self.weights_folder = weights_folder

        self.clip_model, self.clip_processor = clip.load(
            clip_model, device=self.device, download_root=weights_folder
        )
        #
        self.text_features = self.get_text_features()
        #
        self.label2column = {
            label: f'{self.clip_version} clip score "{label}"' for label in self.labels
        }

    @property
    def schema(self) -> List[str]:
        return [self.key_column] + [self.label2column[label] for label in self.labels]

    @property
    def dataloader_kwargs(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def get_text_features(self):
        text_features = []

        for template in self.templates:
            texts = clip.tokenize(
                [template.format(class_label.strip()) for class_label in self.labels]
            )
            text_features.append(self.clip_model.encode_text(texts.to(self.device)))
        text_features = torch.stack(text_features).mean(0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        pil_img = read_image_rgb_from_bytes(modality2data['image'])

        img_tensor = self.clip_processor(pil_img)
        return key, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors = list(zip(*batch))

        with torch.no_grad():
            batch = default_collate(image_tensors).to(self.device)
            image_features = self.clip_model.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_per_image = torch.matmul(image_features, self.text_features.t())
            probs = logits_per_image.cpu().numpy().tolist()

        for c, label in enumerate(self.labels):
            df_batch_labels[self.label2column[label]] += [i[c] for i in probs]
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
