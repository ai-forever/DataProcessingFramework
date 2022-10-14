import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import random
import string
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import BatchSampler, DataLoader

import ruclip

from .t2ifilter import T2IFilter
from DPF.dataloaders.text2image import UniversalT2IDataloader
from DPF.utils import init_logger, read_image_rgb_from_bytes


def get_similarity(predictor, inputs, text_latents):
    with torch.no_grad():
        image_latents = predictor.clip_model.encode_image(inputs['pixel_values'])
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        logits_per_text = torch.matmul(text_latents.to(predictor.device), image_latents.t())
        logits_per_text = logits_per_text.cpu().numpy()
    
    return np.diag(logits_per_text)


class RuCLIPFilter(T2IFilter):
    
    def __init__(self, ruclip_version, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, device='cuda:0', workers=16, batch_size=64,
                 templates=['{}', 'изображение с {}', 'фото с {}'],
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates
        
        self.task_name = task_name if task_name else ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        logfile = f'log_ruclip_similarity.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
            
        self.ruclip_version = ruclip_version
        self.weights_folder = weights_folder
        self.ruclip_model, self.ruclip_processor = ruclip.load(ruclip_version, device=device, cache_dir=weights_folder)
        self.ruclip_predictor = ruclip.Predictor(self.ruclip_model, self.ruclip_processor, 
                                                 device, bs=self.batch_size, templates=self.templates)
        
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=lambda x: x,
            drop_last=False, cols_to_return=['caption']
        )
        
    def preprocess(self, img_bytes, data):
        image_path = data[0]
        text = data[1]
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.ruclip_processor.image_transform(pil_img)
        return image_path, img_tensor, text
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = {
            'image_path': [],
            'ruclip_similarity': [],
        }
        
        image_paths, image_tensors, batch_labels = list(zip(*batch))
        image_tensors = [t.to(self.device) for t in image_tensors]
        
        with torch.no_grad():
            inputs = {}
            inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True)
            text_latents = self.ruclip_predictor.get_text_latents(batch_labels)
            batch_similarity = get_similarity(self.ruclip_predictor, inputs, text_latents).tolist()
            df_batch_labels['ruclip_similarity'].extend(batch_similarity)
            df_batch_labels['image_path'].extend(image_paths)
                
        return df_batch_labels
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f'Starting task {self.task_name}')
        self.logger.info(f'Files to process: {len(df)}')
        
        dataloader = UniversalT2IDataloader(df, **self.dataloader_kwargs)
        
        df_labels = {
            'image_path': [],
            'ruclip_similarity': [],
        }
        
        for batch in tqdm(dataloader):
            df_batch_labels = self.process_batch(batch)
            self.add_values_from_batch(df_labels, df_batch_labels)
        
        df_result = pd.DataFrame(df_labels)
        self.logger.info(f'Processing task {self.task_name} finished')
        
        df = pd.merge(df, df_result, on='image_path')
        
        if self.save_parquets:
            parquet_path = f'{self.save_parquets_dir}/{self.task_name}.parquet'
            df.to_parquet(
                parquet_path,
                index=False
            )
            self.logger.info(f'Parquet saved to {parquet_path}')
        
        return df