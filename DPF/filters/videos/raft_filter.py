from typing import Dict, Union, List
import imageio.v3 as iio
import io
import numpy as np
import cv2

import torch
from .video_filter import VideoFilter
from .raft_core.model import RAFT


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


class RAFTOpticalFlowFilter(VideoFilter):
    """ 
    RAFT model inference class to get mean optical flow each video.
        The video's current and next frame are used for optical flow calculation between them. 
        After, the mean value of optical flow for the entire video is calculated on the array of optical flow between two frames.
    More info about the model here: https://github.com/princeton-vl/RAFT
    """ 
    def __init__(self,
             weights_path: str = 'raft-things.pth'
             small: bool = False,
             device: str = "cuda:0",
             workers: int = 16,
             batch_size: int = 64,
             pbar: bool = True,):
        super().__init__(pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.model = torch.nn.DataParallel(RAFT(small=small))
        self.model.load_state_dict(torch.load(weights_path))
        
        self.schema = [
            self.key_column,
            "mean_optical_flow_raft"
        ]
            
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }
    
    def load_image(self, image_file):
        image_file = np.array(image_file).astype(np.uint8)
        image_file = cv2.resize(image_file, (800, 450))
        image_file = torch.from_numpy(image_file).permute(2, 0, 1).float()
        return image_file[None].to(self.device)

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        video_file = modality2data['video']
        
        frames = iio.imread(io.BytesIO(video_file), plugin="pyav")
        return key, frames
        
    def process(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        optical_flows = []
        with torch.no_grad():
            for data in batch:
                key, frames = data
                for current_frame, next_frame in zip(frames[:-1], frames[1:]):
                    current_frame = self.load_image(current_frame)
                    next_frame = self.load_image(next_frame)

                    padder = InputPadder(current_frame.shape)
                    current_frame, next_frame = padder.pad(current_frame, next_frame)

                    flow_low, flow_up = self.model(current_frame, next_frame, iters=20, test_mode=True)
                    optical_flows.append(flow_up)
                mean_optical_flow = np.mean(optical_flows)
                
                df_batch_labels[self.key_column].append(key)
                df_batch_labels['mean_optical_flow_raft'].append(mean_optical_flow)
        return df_batch_labels
         
            