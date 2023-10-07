from typing import Dict, Any
from io import BytesIO
import numpy as np
from PIL import Image


def default_images_preprocess(column2bytes: Dict[str, bytes], data: Dict[str, str]) -> Any:
    image = Image.open(BytesIO(column2bytes['image_path']))
    image = np.array(image)
    return image, data


def default_preprocess(column2bytes: Dict[str, bytes], data: Dict[str, str]) -> Any:
    return column2bytes, data


def default_collate(batch):
    return batch
