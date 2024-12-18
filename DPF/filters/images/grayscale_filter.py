import cv2
import numpy as np
import joblib
from typing import Any, List, Dict
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.fftpack import fft2, fftshift
from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter


class GrayscaleFilter(ImageFilter):
    """
    Filter for estimating noise levels in images.

    Parameters
    ----------
    model_path: str
        Path to the trained noise estimation model (joblib file).
    params_path: str
        Path to the feature extraction parameters (joblib file).
    workers: int = 16
        Number of processes to use for reading data and calculating noise levels.
    batch_size: int = 64
        Batch size for processing images.
    pbar: bool = True
        Whether to use a progress bar.
    """

    def __init__(
        self,
        workers: int = 1,
        batch_size: int = 1,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.batch_size = batch_size

    @property
    def result_columns(self) -> list[str]:
        return ["grayscale_pass"]

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
        image = read_image_rgb_from_bytes(modality2data['image'])
        return key, image

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for key, image in batch:
            try:
                image = np.array(image)
                result = self.process_image(image)
                df_batch_labels["grayscale_pass"].append(not result)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                df_batch_labels["grayscale_pass"].append(False)
            df_batch_labels[self.key_column].append(key)

        return df_batch_labels


    def process_image(self, image):
        """
        Detects if an image is grayscale (black and white) or not.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            bool: True if the image is grayscale, False otherwise.
        """
        
        # Check if the image has only one channel
        if len(image.shape) == 2:
            return True
        
        # Convert the image to the RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if the histograms for all channels are identical
        hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
        if np.array_equal(hist_r, hist_g) and np.array_equal(hist_r, hist_b):
            return True
        
        # Check if the histogram is concentrated along the diagonal in the RGB color cube
        hist_3d = cv2.calcHist([image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        diagonal_sum = np.sum([hist_3d[i, i, i] for i in range(256)])
        total_sum = np.sum(hist_3d)
        if diagonal_sum / total_sum > 0.9:
            return True
        
        # Check for low saturation
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = image_hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        std_saturation = np.std(saturation)
        if mean_saturation < 10 and std_saturation < 5:
            return True
        
        return False
