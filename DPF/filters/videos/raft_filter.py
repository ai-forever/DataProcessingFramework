import io
from typing import Dict, List, Union, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate

from .raft_core.model import RAFT
from .video_filter import VideoFilter


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 
                         pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                         0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    
def transform_frame(frame: np.ndarray, target_size: Tuple, device: str):
    frame = cv2.resize(frame, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()[None]
    
    padder = InputPadder(frame.shape)
    frame = padder.pad(frame)[0]
    return frame 
    

class RAFTOpticalFlowFilter(VideoFilter):
    """ 
    RAFT model inference class to get mean optical flow each video.
        The video's current and next frame are used for optical flow calculation between them. 
        After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://github.com/princeton-vl/RAFT
    
    Parameters
    ----------
    weights_path: str
        Path to the local modal weights
    small: bool
        Use small model
    """ 
    
    def __init__(self,
                 pass_frames: int = 10,
                 weights_path: str = "raft-things.pth",
                 small: bool = False,
                 device: str = "cuda:0",
                 workers: int = 16,
                 batch_size: int = 1,
                 pbar: bool = True):
        super().__init__(pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.pass_frames = pass_frames
        
        self.model = torch.nn.DataParallel(RAFT(small=small))
        self.model.load_state_dict(torch.load(weights_path))
        
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()
        
        self.schema = [
            self.key_column,
            "mean_optical_flow_raft"
        ]
            
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        video_file = modality2data['video']
        
        frames = iio.imread(io.BytesIO(video_file), plugin="pyav")
        if frames.shape[1] > frames.shape[2]:
            frames = [transform_frame(frame=i, target_size=(450, 800), device=self.device) for i in frames]
        elif frames.shape[2] > frames.shape[1]:
            frames = [transform_frame(frame=i, target_size=(800, 450), device=self.device) for i in frames]
        else:
            frames = [transform_frame(frame=i, targe_size=(450, 450), device=self.device) for i in frames]
        return key, frames
        
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        mean_magnitudes = []
        for data in batch:
            key, frames = data
            with torch.no_grad():
                for i in range(self.pass_frames, len(frames), self.pass_frames):
                    current_frame = frames[i - self.pass_frames]
                    next_frame = frames[i]
                    
                    _, flow = self.model(current_frame.to(self.device),
                                         next_frame.to(self.device),
                                         iters=20, test_mode=True)
                    
                    flow = flow.detach().cpu().numpy()
                    magnitude, angle = cv2.cartToPolar(flow[0][..., 0], flow[0][..., 1])
                    mean_magnitudes.append(magnitude)
                mean_value = np.mean(mean_magnitudes)
                
                df_batch_labels[self.key_column].append(key)
                df_batch_labels['mean_optical_flow_raft'].append(round(mean_value, 3))
        return df_batch_labels
                 