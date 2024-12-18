import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
import cv2
import torch
from torch.multiprocessing import Pool, set_start_method
# Set the start method to 'spawn' at the beginning of your script
try:
    set_start_method('spawn')
except RuntimeError:
    pass
from functools import partial
import numpy as np
from scipy.stats import kurtosis
from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter
from PIL import Image, UnidentifiedImageError
from retinaface.pre_trained_models import get_model

class FaceFocusFilter(ImageFilter):
    def __init__(
        self,
        threshold: float = 2000.0,
        detect_face = True,
        workers: int = 1,
        batch_size: int = 1,
        pbar: bool = True,
        device=None,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.threshold = threshold
        self.detect_face = detect_face
        self.num_workers = workers
        self.batch_size = batch_size
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.face_detector = get_model("resnet50_2020-07-20", 
                                       max_size=2048,
                                       device=self.device)
        self.face_detector.eval()


    @property
    def result_columns(self) -> list[str]:
        return ["face_focus_measure", "bg_focus_measure", "bbox", "faces_count", "confidence", "face_focus_pass", 'focus_pass']
    
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
        try:
            pil_image = read_image_rgb_from_bytes(modality2data['image'])
            numpy_image = np.array(pil_image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            return key, opencv_image
        except (OSError, UnidentifiedImageError, ValueError) as e:
            print(f"Error processing image for key {key}: {str(e)}")
            return key, None
    
    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for key, image in batch:
            info = self.process_image(image)

            if info:
                df_batch_labels["face_focus_measure"].append(info["face_focus_measure"])
                df_batch_labels["bg_focus_measure"].append(info["bg_focus_measure"])
                df_batch_labels["bbox"].append(info["bbox"])
                df_batch_labels["faces_count"].append(info["faces_count"])
                df_batch_labels["confidence"].append(info["confidence"])
                df_batch_labels["face_focus_pass"].append(info["face_focus_pass"])
                df_batch_labels["focus_pass"].append(False)
            else:
                df_batch_labels["face_focus_measure"].append(0)
                df_batch_labels["bg_focus_measure"].append(0)
                df_batch_labels["bbox"].append(False)
                df_batch_labels["faces_count"].append(0)
                df_batch_labels["confidence"].append(0.0)
                df_batch_labels["face_focus_pass"].append(False)
                df_batch_labels["focus_pass"].append(False)

            df_batch_labels[self.key_column].append(key)

        return df_batch_labels

    # def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
    #     df_batch_labels = self._get_dict_from_schema()

    #     # Create a partial function with self.process_image
    #     process_image_partial = partial(self.process_image)

    #     # Use multiprocessing to process images in parallel
    #     with Pool() as pool:
    #         results = pool.map(process_image_partial, [image for _, image in batch])

    #     for (key, _), info in zip(batch, results):
    #         for column in self.result_columns:
    #             df_batch_labels[column].append(info.get(column, 0 if column in ['face_focus_measure', 'bg_focus_measure', 'faces_count', 'confidence'] else False))
    #         df_batch_labels[self.key_column].append(key)

    #     return df_batch_labels

    def tenengrad_variance(self, image):
        """
        Calculate the Tenengrad variance focus measure for the given image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gx_squared = np.square(gx)
        gy_squared = np.square(gy)
        tenengrad_variance = np.mean(gx_squared + gy_squared)
        return tenengrad_variance

    def process_image(self, image):
        
        # Calculate the focus measure for the entire image
        bg_focus_measure = self.tenengrad_variance(image)
        
        focus_pass = bg_focus_measure > self.threshold
        if not self.detect_face:
            # Check if the face is in focus
            focus_pass = bg_focus_measure > self.threshold
            return {
                "face_focus_measure": 0,
                "bg_focus_measure": bg_focus_measure,
                "bbox": None,
                "faces_count": 0,
                "confidence":0,
                "face_focus_pass": None,
                "focus_pass": focus_pass
            }

        # Detect faces in the image
        faces = self.face_detector.predict_jsons(image)
        
        # if faces not found
        if faces is None or len(faces) == 0 or faces[0]['score'] == -1 or not faces[0]['bbox']:
            return {
                "face_focus_measure": 0,
                "bg_focus_measure": bg_focus_measure,
                "bbox": None,
                "faces_count": 0,
                "confidence": 0,
                "face_focus_pass": False,
                "focus_pass": focus_pass
            }

        # Get the face with the highest confidence
        face = max(faces, key=lambda x: x['score'])
        
        faces = [x for x in faces if x['score'] > 0.5]

        bbox = face['bbox']
        landmarks = face['landmarks']
    
        # Extract the face region
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            # print(f"Warning: Empty face region detected for image")
            return {
                "face_focus_measure": 0,
                "bg_focus_measure": bg_focus_measure,
                "bbox": None,
                "faces_count": len(faces),
                "confidence": face["score"],
                "face_focus_pass": False,
                "focus_pass": focus_pass
            }
        
        # Calculate the focus measure for the face region
        face_focus_measure = self.tenengrad_variance(face_region)
        
        # Check if the face is in focus
        in_focus = face_focus_measure > self.threshold
        
        return {
            "face_focus_measure": face_focus_measure,
            "bg_focus_measure": bg_focus_measure,
            "bbox": bbox,
            "faces_count": len(faces),
            "confidence": face["score"],
            "focus_pass": focus_pass,
            "face_focus_pass": (len(faces) == 1) and in_focus and face['score'] > 0.5
        }
