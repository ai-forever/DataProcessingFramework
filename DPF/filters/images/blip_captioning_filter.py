from typing import Any

import torch
from lavis.models import load_model_and_preprocess  # type: ignore

from DPF.types import ModalityToDataMapping

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter


class BLIPCaptioningFilter(ImageFilter):
    """
    BLIPCaptioningFilter class
    """

    def __init__(
        self,
        workers: int = 16,
        batch_size: int = 64,
        device: str = "cuda:0",
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)

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

    @property
    def schema(self) -> list[str]:
        return [self.key_column, "blip_caption"]

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
        img_tensor = self.blip_processor(pil_img)
        return key, img_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, image_tensors = list(zip(*batch))

        with torch.no_grad():
            batch = default_collate(image_tensors).to(self.device)  # type: ignore
            captions = self.blip_model.generate({"image": batch})

        df_batch_labels["blip_caption"].extend(captions)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
