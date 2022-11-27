from typing import List, Optional
import os
import pandas as pd
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
from torchvision import datasets, models, transforms
from huggingface_hub import hf_hub_url, cached_download

from DPF.filters.utils import FP16Module, identical_collate_fn
from DPF.utils import read_image_rgb_from_bytes
from .img_filter import ImageFilter


MODELS = {
    'resnext101_32x8d-large': dict(
        resnet=models.resnext101_32x8d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext101_32x8d-input_size320-4epochs_c097_w082.pth',
    ),
    'resnext50_32x4d-small': dict(
        resnet=models.resnext50_32x4d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext50_32x4d-input_size320-4epochs_c082_w078.pth',
    )
}

def get_watermarks_detection_model(
        name: str, 
        device: str = 'cuda:0', 
        fp16: bool = True, 
        cache_dir: str = '/tmp/datasets_utils'
    ):
    assert name in MODELS
    config = MODELS[name]
    model_ft = config['resnet'](pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    weights = torch.load(os.path.join(cache_dir, config['filename']), device)
    model_ft.load_state_dict(weights)
    
    if fp16:
        model_ft = FP16Module(model_ft)
        
    model_ft.eval()
    model_ft = model_ft.to(device)
    
    return model_ft


class WatermarksFilter(ImageFilter):
    """
    Filter for detecting watermarks.
    
    Parameters
    ----------
    watermarks_model: str
        Version of model to use. Available versions: "resnext50_32x4d-small", "resnext101_32x8d-large"
    weights_folder: str
        Path to folder with weights
    device: str = 'cuda:0'
        Torch device to use
    pbar: bool = True
        Flag for displaying progress bar
    workers: int = 16
        Number of processes for use in dataloader
    batch_size: int = 64
        Batch size for model
    save_parquets_dir: Optional[str] = None
        TODO
    save_parquets: bool = False
        TODO
        
    Attributes
    ----------
    schema: List[str]
        List of columns to be added with this filter.
    dataloader_kwargs: dict:
        Parameters for dataloader (batch_size, num_workers, collate_fn, etc.)
    """
    
    def __init__(
            self, 
            watermarks_model: str, 
            weights_folder: str, 
            device: str = 'cuda:0',
            task_name: Optional[str] = None, save_parquets_dir: Optional[str] = None, 
            save_parquets: bool = False, pbar: bool = True, workers: int = 16, batch_size: int = 64
        ):
        super(WatermarksFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.watermarks_model = watermarks_model
        self.weights_folder = weights_folder
        self.model = get_watermarks_detection_model(watermarks_model, device=device, fp16=True, cache_dir=weights_folder)
        self.resnet_transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
        self.schema = ['image_path', f'watermark_{self.watermarks_model}']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=identical_collate_fn,
            drop_last=False
        )
        
    def preprocess(self, img_bytes: bytes, data: dict):
        image_path = data['image_path']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.resnet_transforms(pil_img)
        return image_path, img_tensor
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_paths, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            df_batch_labels[f'watermark_{self.watermarks_model}'].extend(torch.max(outputs, 1)[1].cpu().reshape(-1).tolist())
        df_batch_labels['image_path'].extend(image_paths)
                
        return df_batch_labels