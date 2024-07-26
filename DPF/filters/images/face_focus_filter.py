import os
from typing import Any
from deepface import DeepFace
import cv2
import numpy as np
from scipy.stats import kurtosis
from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter

class FaceFocusFilter(ImageFilter):
    """
    Filter for detecting faces and checking if the face is in focus.

    Parameters
    ----------
    threshold: float = 2000.0
        Threshold value for the Tenengrad variance focus measure to determine if the face is in focus.
    workers: int = 16
        Number of processes to use for reading data and calculating focus scores.
    batch_size: int = 64
        Batch size for processing images.
    pbar: bool = True
        Whether to use a progress bar.
    """

    def __init__(
        self,
        threshold: float = 2000.0,
        detect_face = True,
        workers: int = 1,
        batch_size: int = 1,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.threshold = threshold
        self.detect_face = detect_face
        self.num_workers = workers
        self.batch_size = batch_size


    @property
    def result_columns(self) -> list[str]:
        return ["face_detected", "in_focus", "focus_measure"]

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
        pil_image = read_image_rgb_from_bytes(modality2data['image'])
        image = np.array(pil_image)
        return key, image

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for key, image in batch:
            info = process_image(image, 
                                      threshold=self.threshold,
                                      detect_face=self.detect_face)
            if info:
                df_batch_labels["face_detected"].append(info["face_detected"])
                df_batch_labels["in_focus"].append(info["in_focus"])
                df_batch_labels["focus_measure"].append(info["focus_measure"])
            else:
                df_batch_labels["face_detected"].append(False)
                df_batch_labels["in_focus"].append(False)
                df_batch_labels["focus_measure"].append(0.0)
            df_batch_labels[self.key_column].append(key)

        return df_batch_labels

def tenengrad_variance(image):
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

def process_image(image, threshold=2000.0, detect_face=True):
    
    # Calculate the focus measure for the entire image
    focus_measure = tenengrad_variance(image)
    
    if not detect_face:
        # Check if the face is in focus
        in_focus = focus_measure > threshold
        return {
            "face_detected":False,
            "in_focus":in_focus,
            "focus_measure":focus_measure
        }

    # Detect faces in the image
    
    faces = DeepFace.extract_faces(image, 
                    enforce_detection=False, 
                    detector_backend='retinaface')
    
    # Filter faces based on confidence and presence of both eyes
    filtered_faces = [face for face in faces if face['confidence'] > 0.1]
    if not filtered_faces:
        return None
    
    face = max(filtered_faces, key=lambda x: x['confidence'])
    
    # Check if exactly one face is detected after filtering
    if len(filtered_faces) == 1 and face['confidence'] > 0.5:
        face['facial_area']['confidence'] = face['confidence']
        if face['facial_area']['left_eye'] is not None and face['facial_area']['right_eye'] is not None:
            
            # Extract the face region
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
            face_region = image[y:y+h, x:x+w]
            
            # Calculate the focus measure for the face region
            focus_measure = tenengrad_variance(face_region)
            
            # Check if the face is in focus
            in_focus = focus_measure > threshold
        
            # Add the focus information to the face dictionary
            face['facial_area']['face_detected'] = True
            face['facial_area']['in_focus'] = in_focus
            face['facial_area']['focus_measure'] = focus_measure
            
            return face['facial_area']
