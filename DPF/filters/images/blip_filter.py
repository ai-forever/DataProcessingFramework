from typing import Dict, List, Union
import torch
from lavis.models import load_model_and_preprocess

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .img_filter import ImageFilter


class BLIPFilter(ImageFilter):
    """
    BLIPFilter class
    """

    def __init__(self, workers=16, batch_size=64, device="cuda:0", pbar=True):
        super().__init__(pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.blip_model, self.blip_processor, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=self.device,
        )
        self.blip_processor = self.blip_processor["eval"]

        self.schema = [self.key_column, "blip_caption"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.blip_processor(pil_img)
        return key, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors = list(zip(*batch))

        with torch.no_grad():
            batch = default_collate(image_tensors).to(self.device)
            captions = self.blip_model.generate({"image": batch})

        df_batch_labels["blip_caption"].extend(captions)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
