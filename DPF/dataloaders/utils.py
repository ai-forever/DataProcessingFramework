from typing import Dict, Any
from io import BytesIO
import numpy as np
from PIL import Image

# TODO(review) - если методы все относятся к torch - переименовать utils.py в torch_utils.py

# TODO(review) - на вход идет column2bytes, интуитивно непонятно, что это такое. И сама логика препроцессинга непонятна, так как возвращается только np.array картинки
# (я бы назвал convert_image_to_numpy и пускал метод по списку значений)
def default_images_preprocess(column2bytes: Dict[str, bytes], data: Dict[str, str]) -> Any:
    image = Image.open(BytesIO(column2bytes['image_path']))
    image = np.array(image)
    return image, data


# TODO(review) - логика работы непонятна совсем, для чего метод нужен, нужны пояснения + рефактор (выглядит как что-то ненужное)
def default_preprocess(column2bytes: Dict[str, bytes], data: Dict[str, str]) -> Any:
    return column2bytes, data


# TODO(review) - ничего не понял, но очень интересно
# (сказали, что переименовать надо, так как нужно для лоадера в torch)
def default_collate(batch):
    return batch
