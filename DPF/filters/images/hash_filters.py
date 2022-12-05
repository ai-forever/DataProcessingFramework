from typing import List, Optional
import os
import pandas as pd
from PIL import Image
import io
import hashlib
import numpy as np
from scipy.fftpack import dct

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .img_filter import ImageFilter


def get_md5_hash(img_byte_arr):
    return hashlib.md5(img_byte_arr).hexdigest()

def get_phash(pil_img, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    image_array = np.array(pil_img.resize((img_size, img_size), Image.ANTIALIAS))
    
    dct_coef = dct(dct(image_array, axis=0), axis=1)
    dct_reduced_coef = dct_coef[:hash_size, :hash_size]
    median_coef_val = np.median(dct_reduced_coef)
    hash_mat = dct_reduced_coef >= median_coef_val
    
    bin_hash_str = ''.join(hash_mat.astype(int).astype(str).reshape(-1))
    n = 4
    sub_strings = [format(int(bin_hash_str[i:i+n], 2), 'x') for i in range(0, len(bin_hash_str), n)]
    return ''.join(sub_strings)



class PHashFilter(ImageFilter):
    
    def __init__(
            self, 
            sim_hash_size: int = 8,
            task_name: Optional[str] = None, save_parquets_dir: Optional[str] = None, 
            save_parquets: bool = False, pbar: bool = True, workers: int = 16
        ):
        super(PHashFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
        self.sim_hash_size = sim_hash_size
            
        self.schema = ['image_path', f'image_simhash_{self.sim_hash_size}']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=1,
            preprocess_f=self.preprocess, collate_fn=identical_collate_fn,
            drop_last=False
        )
        
    def preprocess(self, img_bytes: bytes, data: dict):
        image_path = data['image_path']
        img_simhash = get_phash(read_image_rgb_from_bytes(img_bytes), hash_size=self.sim_hash_size)
        return image_path, img_simhash
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_paths, img_simhashes = list(zip(*batch))
        df_batch_labels['image_path'].extend(image_paths)
        df_batch_labels[f'image_simhash_{self.sim_hash_size}'].extend(img_simhashes)
                
        return df_batch_labels


class MD5Filter(ImageFilter):
    
    def __init__(
            self, 
            task_name: Optional[str] = None, save_parquets_dir: Optional[str] = None, 
            save_parquets: bool = False, pbar: bool = True, workers: int = 16
        ):
        super(MD5Filter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
            
        self.schema = ['image_path', 'image_md5']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=1,
            preprocess_f=self.preprocess, collate_fn=identical_collate_fn,
            drop_last=False
        )
        
    def preprocess(self, img_bytes: bytes, data: dict):
        image_path = data['image_path']
        img_md5 = get_md5_hash(img_bytes)
        return image_path, img_md5
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_paths, img_md5s = list(zip(*batch))
        df_batch_labels['image_path'].extend(image_paths)
        df_batch_labels['image_md5'].extend(img_md5s)
                
        return df_batch_labels