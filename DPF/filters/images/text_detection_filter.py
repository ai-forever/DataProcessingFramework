from typing import Any

import numpy as np
from CRAFT import CRAFTModel, boxes_area, preprocess_image

from DPF.utils import read_image_rgb_from_bytes

from ...types import ModalityToDataMapping
from .img_filter import ImageFilter


class CRAFTFilter(ImageFilter):

    def __init__(
        self,
        weights_folder: str = "/tmp/datasets_utils",
        use_refiner: bool = False,
        device: str = "cuda:0",
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

        self.num_workers = workers
        self.batch_size = 1
        self.device = device

        self.weights_folder = weights_folder
        self.model = CRAFTModel(weights_folder, device, use_refiner=False, fp16=True)

    @property
    def result_columns(self) -> list[str]:
        return ["text_boxes", "num_text_boxes", "text_area", "text_detection_pass"]

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
        img_tensor, ratio_w, ratio_h = preprocess_image(np.array(pil_img), self.model.canvas_size, self.model.mag_ratio)
        return key, img_tensor, ratio_w, ratio_h, pil_img.size

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        key, img_tensor, ratio_w, ratio_h, orig_size = batch[0]

        boxes = self.model._get_boxes_preproc(img_tensor, ratio_w, ratio_h)
        df_batch_labels["text_boxes"].append(boxes)
        df_batch_labels["num_text_boxes"].append(len(boxes))
        df_batch_labels["text_area"].append(boxes_area(boxes)/(orig_size[0]*orig_size[1]))
        df_batch_labels[self.key_column].append(key)
        df_batch_labels["text_detection_pass"].append(len(boxes) == 0)

        return df_batch_labels
