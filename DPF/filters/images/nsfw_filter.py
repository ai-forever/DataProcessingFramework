import os
import zipfile
from typing import Any, Union
from urllib.request import urlretrieve

import clip
import numpy as np
import torch

from ...types import ModalityToDataMapping

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

import autokeras as ak  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_safety_model(clip_model: str, cache_folder: str, device: Union[str, torch.device]) -> Any:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if clip_model == "ViT-L/14":
        model_dir = os.path.join(cache_folder, "clip_autokeras_binary_nsfw")
        url_model = (
            "https://raw.githubusercontent.com/LAION-AI/"
            "CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        )
    else:
        raise ValueError("Unsupported clip model")

    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)
        path_to_zip_file = os.path.join(cache_folder, "clip_autokeras_binary_nsfw.zip")
        urlretrieve(url_model, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    with tf.device(device):
        print(model_dir)
        loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)

    return loaded_model


def normalized(a: np.ndarray[Any, Any], axis: int = -1, order: int = 2) -> Any:
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class NSFWFilter(ImageFilter):
    """
    NSFWFilter class
    """

    def __init__(
        self,
        weights_folder: str,
        workers: int = 16,
        batch_size: int = 64,
        device: str = "cuda:0",
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        clip_model = "ViT-L/14"
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.weights_folder = weights_folder
        self.clip_model, self.clip_transforms = clip.load(
            clip_model, device=self.device, download_root=weights_folder
        )
        self.safety_model = load_safety_model(
            clip_model,
            weights_folder,
            device=self.device.lower().replace("cuda", "gpu"),
        )

    @property
    def result_columns(self) -> list[str]:
        return ["nsfw_score"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "preprocess_f": self.preprocess_data,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        image_path = metadata["image_path"]
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img_tensor = self.clip_transforms(pil_img)
        return image_path, img_tensor

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        image_paths, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)  # type: ignore

        with torch.no_grad():
            image_features = self.clip_model.encode_image(batch)
            emb = np.asarray(normalized(image_features.detach().cpu()))
        nsfw_values = (
            self.safety_model.predict(emb, batch_size=self.batch_size, verbose=0)
            .reshape(-1)
            .tolist()
        )
        df_batch_labels["nsfw_score"].extend(nsfw_values)
        df_batch_labels["image_path"].extend(image_paths)

        return df_batch_labels
