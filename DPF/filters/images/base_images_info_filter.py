from io import BytesIO
from PIL import Image
import numpy as np

from DPF.filters.utils import identical_collate_fn
from .img_filter import ImageFilter


def get_image_info(img_bytes, data):
    """
    Get image path, read status, width, height, num channels, read error
    """
    path = data['image_path']

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

    return path, is_correct, width, height, channels, err_str


class ImageInfoGatherer(ImageFilter):
    """
    ImageInfoGatherer class
    """

    def __init__(
            self,
            workers: int = 16,
            pbar: bool = True
    ):
        super().__init__(pbar)

        self.num_workers = workers

        self.schema = ['image_path', 'is_correct', 'width', 'height', 'channels', 'error']
        self.dataloader_kwargs = {
            'num_workers': self.num_workers, 'batch_size': 1,
            'preprocess_f': self.preprocess, 'collate_fn': identical_collate_fn,
            'drop_last': False
        }

    def preprocess(self, img_bytes: bytes, data: dict):
        return get_image_info(img_bytes, data)

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        for data in batch:
            image_path, is_correct, width, height, channels, error = data
            df_batch_labels['image_path'].append(image_path)
            df_batch_labels['is_correct'].append(is_correct)
            df_batch_labels['width'].append(width)
            df_batch_labels['height'].append(height)
            df_batch_labels['channels'].append(channels)
            df_batch_labels['error'].append(error)
        return df_batch_labels
