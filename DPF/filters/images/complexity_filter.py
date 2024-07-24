import os
from typing import Any
from urllib.request import urlretrieve
import numpy as np
import torch

from ...types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


WEIGHTS_URL = {'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
               'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
               'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'}


class ComplexityFilter(ImageFilter):
    """
    Image complexity filter based on SAM: https://github.com/facebookresearch/segment-anything

    Parameters
    ----------
    weights_folder: str
        Folder where the weights will be stored
    model_name: str = 'vit_h'
        Model version to use: vit_h - huge, vit_l - large, vit_b - big
    points_per_side: int = 32
        Parameter that regulates granularity of automatic segmentation
    batch_size: int = 1
        Batch size during mask calculation for one image
    device: str = "cuda:0"
        Device to use
    workers: int = 16
        Number of processes to use for reading data and calculating flow scores
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        weights_folder: str,
        model_name: str = 'vit_h',
        points_per_side: int = 32,
        workers: int = 16,
        batch_size: int = 1,
        device: str = "cuda:0",
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.model_name = model_name
        self.weights_folder = weights_folder
        self.points_per_side = points_per_side
        
        # Download checkpoints
        path_to_model = os.path.join(self.weights_folder, self.model_name + '.pth')
        if not os.path.exists(path_to_model):
            os.makedirs(self.weights_folder, exist_ok=True)
            urlretrieve(WEIGHTS_URL[self.model_name], path_to_model)

        sam = sam_model_registry[self.model_name](checkpoint=path_to_model)
        sam = sam.to(torch.device(self.device))
        self.mask_generator = SamAutomaticMaskGenerator(
                                sam, points_per_batch=batch_size, 
                                points_per_side=points_per_side
                                )

    @property
    def result_columns(self) -> list[str]:
        return ["complexity_num_segments", "complexity_max_segment_area", "complexity_mean_segment_area"]

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
        pil_img = read_image_rgb_from_bytes(modality2data['image'])
        img = np.array(pil_img)
        return key, img

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for data in batch:
            key, img = data
            h, w = img.shape[:2]
            hw = h * w
            with torch.no_grad():
                outputs = self.mask_generator.generate(img)
            num_segments = len(outputs)
            if num_segments > 0:
                areas = [x['area'] for x in outputs]
                bg_area = hw - np.sum(areas)
                areas.append(bg_area)
                max_area = np.max(areas) / hw
                mean_area = np.mean(areas) / hw
            else:
                max_area = mean_area = 0
                    
            df_batch_labels["complexity_num_segments"].extend([num_segments])
            df_batch_labels["complexity_max_segment_area"].extend([max_area])
            df_batch_labels["complexity_mean_segment_area"].extend([mean_area])
            df_batch_labels[self.key_column].extend([key])

        return df_batch_labels
