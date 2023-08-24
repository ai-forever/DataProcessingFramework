import os
import torch
from torch import nn
import numpy as np

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
from torchvision import models, transforms
from huggingface_hub import hf_hub_url, cached_download

from DPF.filters.utils import FP16Module, identical_collate_fn
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter

from CRAFT import CRAFTModel, preprocess_image, boxes_area


class CRAFTFilter(ImageFilter):

    def __init__(
        self,
        weights_folder: str,
        use_refiner: bool = False,
        device: str = "cuda:0",
        workers: int = 16,
        pbar: bool = True,
    ):
        super().__init__(pbar)

        self.num_workers = workers
        self.batch_size = 1
        self.device = device

        self.weights_folder = weights_folder
        self.model = CRAFTModel(weights_folder, device, use_refiner=False, fp16=True)

        self.schema = ["image_path", f"text_boxes", "num_text_boxes", "text_area"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "preprocess_f": self.preprocess,
            "collate_fn": identical_collate_fn,
            "drop_last": False,
        }

    def preprocess(self, img_bytes: bytes, data: dict):
        image_path = data["image_path"]
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor, ratio_w, ratio_h = preprocess_image(np.array(pil_img), self.model.canvas_size, self.model.mag_ratio)
        return image_path, img_tensor, ratio_w, ratio_h, pil_img.size

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_path, img_tensor, ratio_w, ratio_h, orig_size = batch[0]

        boxes = self.model._get_boxes_preproc(img_tensor, ratio_w, ratio_h)
        df_batch_labels["text_boxes"].append(boxes)
        df_batch_labels["num_text_boxes"].append(len(boxes))
        df_batch_labels["text_area"].append(boxes_area(boxes)/(orig_size[0]*orig_size[1]))
        df_batch_labels["image_path"].append(image_path)

        return df_batch_labels
