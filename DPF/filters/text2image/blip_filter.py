from typing import List, Dict, Union, Optional
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from lavis.models import load_model_and_preprocess
from PIL import Image

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .t2i_filter import T2IFilter


class BlipFilter(T2IFilter):
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

        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_feature_extractor", model_type="base", is_eval=True, device=self.device)

        self.schema = [self.key_column, f"blip_similarity"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False
        }

        
    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        text = modality2data['text']
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.vis_processors["eval"](pil_img)
        return key, img_tensor, text

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        keys, image_tensors, batch_labels = list(zip(*batch))
           
        sample  = {}
        with torch.no_grad():
            image_tensors = [t.to(self.device) for t in image_tensors]
            sample['image'] = pad_sequence(image_tensors, batch_first=True)
            sample['text_input'] = [self.txt_processors["eval"](class_label) for class_label in batch_labels]
            features_multimodal = self.blip_model.extract_features(sample)
            features_image = self.blip_model.extract_features(sample, mode="image")
            features_text = self.blip_model.extract_features(sample, mode="text")
            
            batch_similarity = self.get_similarity(features_image, features_text)

        df_batch_labels[f"blip_similarity"].extend(batch_similarity)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels

    def get_similarity(self, features_image, features_text):
        with torch.no_grad():
            logits_per_image = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
            probs = logits_per_image.cpu().numpy().tolist()

        return np.diag(probs)
