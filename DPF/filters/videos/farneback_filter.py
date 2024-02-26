from typing import Dict, Union, List
import imageio.v3 as iio
import io
import numpy as np
import cv2

import torch
from .video_filter import VideoFilter


class GunnarFarnebackFilter(VideoFilter):
    """ 
    RAFT model inference class to get mean optical flow each video.
        The video's current and next frame are used for optical flow calculation between them. 
        After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://github.com/princeton-vl/RAFT
    """ 
    def __init__(self,
             workers: int = 16,
             batch_size: int = 1,
             pbar: bool = True,):
        super().__init__(pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        
        self.schema = [
            self.key_column,
            "mean_optical_flow_farneback"
        ]
            
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }
    
    def load_image(self, image_file):
        image_file = np.array(image_file).astype(np.uint8)
        if image_file.shape[0] > image_file.shape[1]:
            image_file = cv2.resize(image_file, (450, 800))
        elif image_file.shape[1] > image_file.shape[0]:
            image_file = cv2.resize(image_file, (800, 450))
        else:
            image_file = cv2.resize(image_file, (450, 450))
        image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        return image_file

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        video_file = modality2data['video']
        
        frames = iio.imread(io.BytesIO(video_file), plugin="pyav")
        return key, frames
        
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        mags = []
        with torch.no_grad():
            for data in batch:
                key, frames = data
                for current_frame, next_frame in zip(frames[:-1], frames[1:]):
                    current_frame = self.load_image(current_frame)
                    next_frame = self.load_image(next_frame)

                    flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mags.append(mag)
                mean_optical_flow = np.mean(mags)
                
                df_batch_labels[self.key_column].append(key)
                df_batch_labels['mean_optical_flow_farneback'].append(round(mean_optical_flow, 3))
        return df_batch_labels
         
            