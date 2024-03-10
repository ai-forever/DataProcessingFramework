from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
from PIL import Image

from ...types import ModalityToDataMapping
from .img_filter import ImageFilter


@dataclass
class ImageInfo:
    key: str
    is_correct: bool
    width: Optional[int]
    height: Optional[int]
    channels: Optional[int]
    error: Optional[str]


def get_image_info(img_bytes: bytes, data: dict[str, Any], key_column: str) -> ImageInfo:
    """
    Get image path, read status, width, height, num channels, read error
    """
    key = data[key_column]

    is_correct = True
    width, height, channels = None, None, None
    err_str = None

    try:
        pil_img = Image.open(BytesIO(img_bytes))
        pil_img.load()

        arr = np.array(pil_img)

        width = pil_img.width
        height = pil_img.height
        if len(arr.shape) == 2:
            channels = 1
        else:
            channels = arr.shape[2]
    except Exception as err:
        is_correct = False
        err_str = str(err)

    return ImageInfo(key, is_correct, width, height, channels, err_str)


class ImageInfoFilter(ImageFilter):
    """
    ImageInfoFilter class
    """

    def __init__(self, workers: int = 16, pbar: bool = True, _pbar_position: int = 0):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers

    @property
    def schema(self) -> list[str]:
        return [
            self.key_column,
            "is_correct", "width", "height",
            "channels", "error",
        ]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        return get_image_info(modality2data['image'], metadata, self.key_column)

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for image_info in batch:
            df_batch_labels[self.key_column].append(image_info.key)
            df_batch_labels["is_correct"].append(image_info.is_correct)
            df_batch_labels["width"].append(image_info.width)
            df_batch_labels["height"].append(image_info.height)
            df_batch_labels["channels"].append(image_info.channels)
            df_batch_labels["error"].append(image_info.error)
        return df_batch_labels
