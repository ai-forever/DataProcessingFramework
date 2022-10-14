import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io
import random
import string
import hashlib
import numpy as np
from scipy.fftpack import dct
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import BatchSampler, DataLoader

from .t2ifilter import T2IFilter
from DPF.dataloaders.text2image import UniversalT2IDataloader
from DPF.utils import init_logger, read_image_rgb_from_bytes


def get_md5_hash(img_byte_arr):
    return hashlib.md5(img_byte_arr).hexdigest()

def get_sim_hash(pil_img, size=10):
    image_array = np.array(pil_img)
    dct_coef = dct(dct(image_array, axis=0), axis=1)
    dct_reduced_coef = dct_coef[:size, :size]
    median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])
    hash_mat = dct_reduced_coef >= median_coef_val
    bin_hash_str = ''.join(hash_mat.astype(int).astype(str).reshape(-1))
    n = 4
    sub_strings = [format(int(bin_hash_str[i:i+n], 2), 'x') for i in range(0, len(bin_hash_str), n)]
    return ''.join(sub_strings)


class HashFilter(T2IFilter):
    
    def __init__(self, sim_hash_size=10, task_name=None, save_parquets_dir=None,
                 save_parquets=False, workers=16,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        self.num_workers = workers
        self.sim_hash_size = sim_hash_size
        
        self.task_name = task_name if task_name else ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        logfile = f'log_ruclip_similarity.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
            
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=1,
            preprocess_f=self.preprocess, collate_fn=lambda x: x,
            drop_last=False
        )
        
    def preprocess(self, img_bytes, data):
        image_path = data[0]
        img_md5 = get_md5_hash(img_bytes)
        img_simhash = get_sim_hash(Image.open(io.BytesIO(img_bytes)), size=self.sim_hash_size)
        return image_path, img_md5, img_simhash
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = {
            'image_path': [],
            'image_md5': [],
            'image_simhash': [],
        }
        
        image_paths, img_md5s, img_simhashes = list(zip(*batch))
        df_batch_labels['image_path'].extend(image_paths)
        df_batch_labels['image_md5'].extend(img_md5s)
        df_batch_labels['image_simhash'].extend(img_simhashes)
                
        return df_batch_labels
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f'Starting task {self.task_name}')
        self.logger.info(f'Files to process: {len(df)}')
        
        dataloader = UniversalT2IDataloader(df, **self.dataloader_kwargs)
        
        df_labels = {
            'image_path': [],
            'image_md5': [],
            'image_simhash': [],
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