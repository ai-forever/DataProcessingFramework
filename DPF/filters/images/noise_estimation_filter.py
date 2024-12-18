import cv2
import numpy as np
import joblib
from typing import Any, List, Dict
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.fftpack import fft2, fftshift
from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter


class NoiseEstimationFilter(ImageFilter):
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
        model_path: str,
        params_path: str,
        workers: int = 1,
        batch_size: int = 1,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.batch_size = batch_size
        self.model = joblib.load(model_path)
        self.params = joblib.load(params_path)

    @property
    def result_columns(self) -> list[str]:
        return ["estimated_noise_level", "noise_filter_pass"]

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
                noise_level = self.process_image(image)
                df_batch_labels["estimated_noise_level"].append(noise_level)
                df_batch_labels["noise_filter_pass"].append(noise_level<1.0)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                df_batch_labels["estimated_noise_level"].append(None)
                df_batch_labels["noise_filter_pass"].append(False)
            df_batch_labels[self.key_column].append(key)

        return df_batch_labels
    
    def process_image(self, image):
        image = np.array(image)
        cropped_image = self.crop_image(image)
        features = self.extract_features(cropped_image)
        noise_level = self.model.predict([features])[0]
        return noise_level

    def crop_image(self, image):
        target_size = self.params['target_size']
        height, width = image.shape[:2]
        if height > width:
            scale = target_size / height
            new_height = target_size
            new_width = int(width * scale)
        else:
            scale = target_size / width
            new_width = target_size
            new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        start_y = max(0, (new_height - target_size) // 2)
        start_x = max(0, (new_width - target_size) // 2)
        cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
        
        return cropped

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Statistical features
        std_dev = np.std(gray)
        entropy = -np.sum(np.histogram(gray, bins=256, density=True)[0] * 
                          np.log2(np.histogram(gray, bins=256, density=True)[0] + 1e-7))
        
        # Edge detection
        edges = cv2.Canny(gray, self.params['canny_low'], self.params['canny_high'])
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Texture analysis - GLCM
        glcm = graycomatrix(gray, self.params['glcm_distances'], self.params['glcm_angles'], 
                            256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Texture analysis - LBP
        lbp = local_binary_pattern(gray, self.params['lbp_n_points'], self.params['lbp_radius'], method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.params['lbp_n_points'] + 3), 
                                   range=(0, self.params['lbp_n_points'] + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # Frequency domain analysis
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        mean_magnitude = np.mean(magnitude_spectrum)
        
        return np.concatenate(([std_dev, entropy, edge_density, contrast, homogeneity, mean_magnitude], lbp_hist))