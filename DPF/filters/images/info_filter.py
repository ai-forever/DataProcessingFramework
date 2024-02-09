from typing import Dict, Union
from io import BytesIO
from PIL import Image
import numpy as np

from .img_filter import ImageFilter


def get_image_info(img_bytes, data, key_column):
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

    return key, is_correct, width, height, channels, err_str


class ImageInfoFilter(ImageFilter):
    """
    ImageInfoFilter class
    """

    def __init__(self, workers: int = 16, pbar: bool = True):
        super().__init__(pbar)

        self.num_workers = workers

        self.schema = [
            self.key_column,
            "is_correct",
            "width",
            "height",
            "channels",
            "error",
        ]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        return get_image_info(modality2data['image'], metadata, self.key_column)

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        for data in batch:
            key, is_correct, width, height, channels, error = data
            df_batch_labels[self.key_column].append(key)
            df_batch_labels["is_correct"].append(is_correct)
            df_batch_labels["width"].append(width)
            df_batch_labels["height"].append(height)
            df_batch_labels["channels"].append(channels)
            df_batch_labels["error"].append(error)
        return df_batch_labels
