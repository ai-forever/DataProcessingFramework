from typing import Any

import numpy as np
from PIL import Image
from scipy.fftpack import dct

from DPF.utils import read_image_rgb_from_bytes

from ...types import ModalityToDataMapping
from .img_filter import ImageFilter

Image.MAX_IMAGE_PIXELS = None 

def get_phash(pil_img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    img_size = hash_size * highfreq_factor
    image_array = np.array(pil_img.resize((img_size, img_size), Image.Resampling.LANCZOS))

    dct_coef = dct(dct(image_array, axis=0), axis=1)
    dct_reduced_coef = dct_coef[:hash_size, :hash_size]
    median_coef_val = np.median(dct_reduced_coef)
    hash_mat = dct_reduced_coef >= median_coef_val

    bin_hash_str = "".join(hash_mat.astype(int).astype(str).reshape(-1))
    n = 4
    sub_strings = [
        format(int(bin_hash_str[i : i + n], 2), "x")
        for i in range(0, len(bin_hash_str), n)
    ]
    return "".join(sub_strings)


class PHashFilter(ImageFilter):
    """
    Filter for calculating PHash (perceptual hash) for images

    Parameters
    ----------
    sim_hash_size: int = 8
        Hash size for PHash
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        sim_hash_size: int = 8,
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.sim_hash_size = sim_hash_size

    @property
    def result_columns(self) -> list[str]:
        return [f"image_phash_{self.sim_hash_size}"]

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
        key = metadata[self.key_column]
        img_simhash = get_phash(
            read_image_rgb_from_bytes(modality2data['image']), hash_size=self.sim_hash_size
        )
        return key, img_simhash

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, img_simhashes = list(zip(*batch))
        df_batch_labels[self.key_column].extend(keys)
        df_batch_labels[f"image_phash_{self.sim_hash_size}"].extend(img_simhashes)

        return df_batch_labels
